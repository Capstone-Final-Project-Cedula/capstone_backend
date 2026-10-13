"""
app/services/speech_service.py
OpenAI Whisper ASR integration — the speech-to-text step of the live
interview (Whisper -> DistilBERT + Regex -> voice read-back).

Calibrated for short, one-question answers recorded in the browser:
  * per-question hints (initial_prompt), e.g. the barangay names of
    Baungon on the address question, so local words are heard right;
  * greedy decoding (WHISPER_BEAM_SIZE=1) and a short temperature fallback,
    since a 2-second answer gains little from beam search;
  * the model is loaded once at startup (WHISPER_PRELOAD) and shared;
  * phrases Whisper invents on near-silence ("Thank you.") are dropped.
"""
import whisper
import numpy as np
import io
import re
import threading
import time
import tempfile
import os
from math import gcd
from typing import Optional, Union
from scipy.io import wavfile
from scipy.signal import resample_poly
from app.core.config import settings
from app.services.barangay_gazetteer import get_barangay_list_for_prompt
import logging

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


def _decode_wav(audio_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode WAV bytes to 16 kHz mono float32 in [-1, 1], the input Whisper expects.
    Returns None for anything that is not a readable WAV file.
    """
    if not audio_bytes.startswith(b"RIFF"):
        return None
    try:
        rate, data = wavfile.read(io.BytesIO(audio_bytes))
    except Exception:  # noqa: BLE001 — unreadable WAV: let ffmpeg try
        return None

    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        samples = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        samples = (data.astype(np.float32) - 128.0) / 128.0
    else:
        samples = data.astype(np.float32)

    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != WHISPER_SAMPLE_RATE:
        factor = gcd(rate, WHISPER_SAMPLE_RATE)
        samples = resample_poly(samples, WHISPER_SAMPLE_RATE // factor, rate // factor)
    return samples.astype(np.float32)


# Context hints per interview field — dramatically improves accuracy on
# short, ambiguous answers by biasing Whisper's decoding toward expected
# vocabulary. Keep these short; long prompts can backfire.
BUKIDNON_PLACES = (
    "Baungon, Malaybalay, Valencia, Talakag, Libona, Manolo Fortich, Sumilao, Impasugong, "
    "Lantapan, Cagayan de Oro, Bukidnon, Misamis Oriental"
)

FIELD_PROMPTS = {
    "surname": "Filipino surname. Examples: Santos, Reyes, Dela Cruz, Bautista, Garcia, Lumansoc, Dagondon.",
    "given_name": "Filipino given name. Examples: Maria, Jose, Juan, Ana, Ramon, Jun-jun.",
    "middle_name": "Filipino middle name, or the word none.",
    "address": (
        f"Purok, sitio and barangay in Baungon, Bukidnon. Barangays: {get_barangay_list_for_prompt()}."
    ),
    "date": "A birthday, for example May 10, 1990 or the 10th of May 1990.",
    "sex": "Male or female. Lalaki or babae.",
    "civil_status": "Single, married, widowed or separated.",
    "citizenship": "A citizenship, usually Filipino.",
    "icr_no": "An ICR number with letters and digits, or the word none.",
    "place_of_birth": f"A town or city and province, for example {BUKIDNON_PLACES}.",
    "height": "A height, for example 160 centimeters or 5 feet 4 inches.",
    "weight": "A weight, for example 60 kilos or 130 pounds.",
    "occupation": "An occupation. Examples: farmer, teacher, driver, vendor, housewife, student, laborer.",
    "gross_annual_income": "An amount in pesos, for example 20,000 a month or 240,000 a year, or none.",
    "spelling": "A name spelled letter by letter, for example C, R, U, Z, or the word yes.",
    "confirm": "Yes or no. Opo, oo, hindi.",
    "review": "Yes, or a correction such as: change surname to Cruz.",
}

# Text Whisper tends to invent on silence or noise; never treat it as an answer.
HALLUCINATIONS = re.compile(
    r"^(?:thank you\.?|thanks\.?|thank you for watching\.?|thanks for watching\.?|you\.?|bye\.?|"
    r"please subscribe\.?|\.+|subtitles by .*|amara\.org.*)$",
    re.IGNORECASE,
)

_model_lock = threading.Lock()   # one transcription at a time on the shared model


def _decode_options(initial_prompt: Optional[str]) -> dict:
    """Short-answer settings: greedy by default, small fallback ladder for unclear audio."""
    beam = max(1, int(settings.WHISPER_BEAM_SIZE))
    options = dict(
        language="en",
        task="transcribe",
        verbose=False,
        fp16=False,                        # CPU servers have no FP16
        condition_on_previous_text=False,  # short single answers, not long-form speech
        no_speech_threshold=settings.WHISPER_NO_SPEECH_THRESHOLD,
        logprob_threshold=settings.WHISPER_LOGPROB_THRESHOLD,
        initial_prompt=initial_prompt,
        temperature=(0.0, 0.4, 0.8),       # retried only when the first pass looks unreliable
    )
    if beam > 1:
        options.update(beam_size=beam, best_of=beam)
    return options


def _clean_transcript(text: str) -> str:
    text = (text or "").strip()
    return "" if HALLUCINATIONS.match(text) else text


class SpeechService:
    """
    Singleton service wrapping OpenAI Whisper ASR.
    Model is loaded once on first use to avoid repeated cold starts.
    """

    _model = None  # Cached Whisper model (shared across requests)

    @classmethod
    def get_model(cls):
        """Load model once and cache it (lazy initialization)."""
        if cls._model is None:
            with _model_lock:
                if cls._model is None:
                    logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL}")
                    cls._model = whisper.load_model(settings.WHISPER_MODEL)
                    logger.info("Whisper model loaded successfully.")
        return cls._model

    @classmethod
    def warm_up(cls) -> None:
        """Load the model (and run one tiny pass) at startup so the first answer is fast."""
        try:
            model = cls.get_model()
            with _model_lock:
                model.transcribe(np.zeros(WHISPER_SAMPLE_RATE // 2, dtype=np.float32), **_decode_options(None))
            logger.info("Whisper warmed up.")
        except Exception as exc:  # never block startup on this
            logger.warning(f"Whisper warm-up failed: {exc}")

    @classmethod
    def _build_prompt(cls, field_key: Optional[str]) -> Optional[str]:
        if not field_key:
            return None
        return FIELD_PROMPTS.get(field_key)

    @classmethod
    def transcribe_file(cls, audio_path: Union[str, np.ndarray], field_key: Optional[str] = None) -> dict:
        """
        Transcribe audio using Whisper.

        Args:
            audio_path: Path to an audio file (WAV, MP3, M4A, etc. — decoded by
                        ffmpeg), or 16 kHz mono float32 samples (no ffmpeg needed).
            field_key: Optional interview field key, used to bias decoding
                       toward expected vocabulary via initial_prompt.

        Returns:
            dict with keys: transcript, language, confidence, segments,
                            processing_time_seconds
        """
        model = cls.get_model()
        start_time = time.time()
        initial_prompt = cls._build_prompt(field_key)

        source = audio_path if isinstance(audio_path, str) else f"{len(audio_path) / WHISPER_SAMPLE_RATE:.1f}s of audio"
        logger.info(f"Transcribing {source} (field={field_key})")

        with _model_lock:
            result = model.transcribe(audio_path, **_decode_options(initial_prompt))

        processing_time = time.time() - start_time

        segments = result.get("segments", [])
        if segments:
            avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segments])
            confidence = float(np.clip(np.exp(avg_logprob), 0.0, 1.0))
        else:
            confidence = 0.0

        transcript = _clean_transcript(result.get("text", ""))

        # Only the length is logged: transcripts are applicants' personal data (Data Privacy Act).
        logger.info(f"Transcription complete in {processing_time:.2f}s | Confidence: {confidence:.2f} | "
                    f"{len(transcript)} characters")

        return {
            "transcript": transcript,
            "language": result.get("language", "en"),
            "confidence": confidence,
            "segments": [
                {
                    "id": s["id"],
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"].strip(),
                    "avg_logprob": s.get("avg_logprob", 0),
                }
                for s in segments
            ],
            "processing_time_seconds": round(processing_time, 3),
        }

    @classmethod
    def transcribe_bytes(cls, audio_bytes: bytes, suffix: str = ".wav", field_key: Optional[str] = None) -> dict:
        """
        Transcribe audio from raw bytes (e.g., uploaded file from API).
        WAV (what the web app uploads) is decoded here, so it works without
        ffmpeg; other formats go through Whisper's ffmpeg-based loader.
        """
        samples = _decode_wav(audio_bytes)
        if samples is not None:
            return cls.transcribe_file(samples, field_key=field_key)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = cls.transcribe_file(tmp_path, field_key=field_key)
        finally:
            os.unlink(tmp_path)  # Clean up temp file

        return result

    @classmethod
    def transcribe_numpy(cls, audio_array: np.ndarray, sample_rate: int = 16000, field_key: Optional[str] = None) -> dict:
        """
        Transcribe a numpy float32 audio array (from sounddevice capture).
        """
        model = cls.get_model()
        start_time = time.time()
        initial_prompt = cls._build_prompt(field_key)

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        audio_array = audio_array.astype(np.float32)

        with _model_lock:
            result = model.transcribe(audio_array, **_decode_options(initial_prompt))

        processing_time = time.time() - start_time
        segments = result.get("segments", [])
        confidence = (
            float(np.clip(np.exp(np.mean([s.get("avg_logprob", -1.0) for s in segments])), 0, 1))
            if segments else 0.0
        )

        return {
            "transcript": _clean_transcript(result.get("text", "")),
            "language": result.get("language", "en"),
            "confidence": confidence,
            "segments": segments,
            "processing_time_seconds": round(processing_time, 3),
        }