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

Two engines run the same Whisper models (WHISPER_ENGINE, plan 002): "openai" (openai-whisper)
and "faster" (faster-whisper / CTranslate2, int8 on CPU, with a voice-activity filter).
faster-whisper API: https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py
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
from app.services import place_gazetteer
from app.services.barangay_gazetteer import get_barangay_list_for_prompt, region_town
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
    # Name hints are written like real answers, not as a list: Whisper copies the format
    # (local spellings, "po" as its own word) instead of copying a list of example names
    # into the transcript ("Dela Cruz, Bautista, Garcia."). Benchmark with Philippine
    # voices: 30 -> 43 of 144 names exactly right with Whisper base.
    "surname": "Lagbas po. Ybañez po. My surname is Sumagaysay. Manlangit po. Tagaylo.",
    "given_name": "Jessa po. My first name is Rodel. Mylene po. Gerlie Mae.",
    "middle_name": "Lagbas po. My middle name is Ybañez. None po. Sumagaysay.",
    "address": (
        f"Purok, sitio and barangay in Baungon, Bukidnon. Barangays: {get_barangay_list_for_prompt()}."
    ),
    # The address in 3 short questions (plan 001.6). The barangay hint is built per town: barangay_prompt().
    "address_town": f"The town or city, for example {BUKIDNON_PLACES}.",
    "address_purok": "Purok 2. Zone 1. Sitio, street or house number, or none.",
    "date": "A birthday, for example May 10, 1990 or the 10th of May 1990.",
    "sex": "Male or female. Lalaki or babae.",
    "civil_status": "Single, married, widowed or separated.",
    "citizenship": "A citizenship, usually Filipino.",
    "icr_no": "An ICR number with letters and digits, or the word none.",
    "place_of_birth": f"A town or city and province, for example {BUKIDNON_PLACES}.",
    # The place of birth in 2 steps (plan 001.8). The town hint is built per province: birth_town_prompt().
    "place_of_birth_province": (
        "The province, for example Bukidnon, Misamis Oriental, Lanao del Norte, Misamis Occidental, Cebu, "
        "or a city: Cagayan de Oro, Iligan, Davao."
    ),
    "height": "A height, for example 160 centimeters or 5 feet 4 inches.",
    "weight": "A weight, for example 60 kilos or 130 pounds.",
    "occupation": "An occupation. Examples: farmer, teacher, driver, vendor, housewife, student, laborer.",
    "gross_annual_income": "An amount in pesos, for example 20,000 a month or 240,000 a year, or none.",
    "spelling": "Letters spelled one by one, for example: D, E, L, A, space, C, R, U, Z, done.",
    # No "Go back." here (plan 001.6): the tiny model copied it into replies, so a yes went back a question.
    "confirm": "Yes or no. Opo, oo, hindi.",
    "review": "Yes, or a correction such as: change surname to Cruz.",
    "missing_choice": "Type it, or say it again. Type. Say it again.",
}

# Text Whisper tends to invent on silence or noise; never treat it as an answer.
HALLUCINATIONS = re.compile(
    r"^(?:thank you\.?|thanks\.?|thank you for watching\.?|thanks for watching\.?|you\.?|bye\.?|"
    r"please subscribe\.?|\.+|subtitles by .*|amara\.org.*)$",
    re.IGNORECASE,
)

_model_lock = threading.Lock()   # one transcription at a time on the shared model


def barangay_prompt(town_psgc: Optional[str]) -> str:
    """Whisper hint for "What is your barangay?": the barangays of the town said (office town if none)."""
    if not town_psgc:
        return f"The barangay. Barangays: {get_barangay_list_for_prompt()}."
    town = region_town(town_psgc)
    if not town:
        return "The name of the barangay."          # outside Region X: no list, never the office's
    where = f"{town['town']}, {town['province']}".strip(", ")
    return f"Barangay in {where}. Barangays: {get_barangay_list_for_prompt(town_psgc)}."


BIRTH_TOWN_HINT_MAX = 40   # a long prompt can backfire, and Whisper keeps only its last 223 tokens


def birth_town_prompt(province_psgc: Optional[str]) -> str:
    """Whisper hint for "Which town or city in <province>?": that province's cities and towns."""
    province = place_gazetteer.province_by_code(province_psgc or "")
    if not province:
        return "The town or city where you were born."
    names = ", ".join(p.name for p in place_gazetteer.towns_in_province(province.psgc)[:BIRTH_TOWN_HINT_MAX])
    return f"A town or city in {province.display}: {names}."


FALLBACK_TEMPERATURES = (0.0, 0.4, 0.8)   # greedy only: retried when the first pass looks unreliable


def _beam() -> int:
    return max(1, int(settings.WHISPER_BEAM_SIZE))


def _temperatures(beam: int):
    # Beam search gets no fallback ladder: each retry re-runs the whole beam, which is how one
    # small/beam-5 answer took 56 s (plan 002). One beam pass is already the careful decode.
    return FALLBACK_TEMPERATURES if beam == 1 else 0.0


def _decode_options(initial_prompt: Optional[str]) -> dict:
    """openai-whisper short-answer settings: greedy by default, small fallback ladder for unclear audio."""
    beam = _beam()
    options = dict(
        language="en",
        task="transcribe",
        verbose=False,
        fp16=False,                        # CPU servers have no FP16
        condition_on_previous_text=False,  # short single answers, not long-form speech
        no_speech_threshold=settings.WHISPER_NO_SPEECH_THRESHOLD,
        logprob_threshold=settings.WHISPER_LOGPROB_THRESHOLD,
        initial_prompt=initial_prompt,
        temperature=_temperatures(beam),
    )
    if beam > 1:
        options.update(beam_size=beam, best_of=beam)
    return options


def _faster_options(initial_prompt: Optional[str]) -> dict:
    """The same settings for faster-whisper. Every option is given: its defaults differ (beam 5,
    condition_on_previous_text on, a 6-step temperature ladder, no VAD)."""
    beam = _beam()
    return dict(
        language="en",
        task="transcribe",
        beam_size=beam,
        best_of=beam,                      # openai-whisper samples once per fallback step when greedy
        condition_on_previous_text=False,
        no_speech_threshold=settings.WHISPER_NO_SPEECH_THRESHOLD,
        log_prob_threshold=settings.WHISPER_LOGPROB_THRESHOLD,
        initial_prompt=initial_prompt,
        temperature=_temperatures(beam),
        vad_filter=settings.WHISPER_VAD,   # Silero VAD (onnxruntime): silence is cut before decoding
    )


_engine_warned = False


def _engine() -> str:
    """The engine in use: "faster" only when chosen and installed; otherwise "openai"."""
    global _engine_warned
    choice = (settings.WHISPER_ENGINE or "openai").strip().lower()
    if choice == "openai":
        return "openai"
    if choice == "faster":
        try:
            import faster_whisper  # noqa: F401
            return "faster"
        except ImportError:
            reason = "faster-whisper is not installed (pip install -r requirements.txt)"
    else:
        reason = f"unknown engine {settings.WHISPER_ENGINE!r} (use openai or faster)"
    if not _engine_warned:
        logger.error(f"WHISPER_ENGINE={settings.WHISPER_ENGINE}: {reason}. Using openai-whisper instead.")
        _engine_warned = True
    return "openai"


def _load_model(engine: str, name: str):
    if engine == "faster":
        from faster_whisper import WhisperModel
        return WhisperModel(name, device="cpu", compute_type=settings.WHISPER_COMPUTE_TYPE)
    return whisper.load_model(name)


def _run(engine: str, model, audio: Union[str, np.ndarray], initial_prompt: Optional[str]) -> dict:
    """One decode on either engine, returned in openai-whisper's shape: text, language, segments."""
    if engine == "faster":
        segments, info = model.transcribe(audio, **_faster_options(initial_prompt))
        segments = list(segments)          # a lazy generator: decoding happens here
        return {
            "text": "".join(s.text for s in segments),
            "language": info.language,
            "segments": [{"id": s.id, "start": s.start, "end": s.end, "text": s.text,
                          "avg_logprob": s.avg_logprob} for s in segments],
        }
    return model.transcribe(audio, **_decode_options(initial_prompt))


def _clean_transcript(text: str) -> str:
    text = (text or "").strip()
    return "" if HALLUCINATIONS.match(text) else text


class SpeechService:
    """
    Singleton service wrapping OpenAI Whisper ASR.
    Model is loaded once on first use to avoid repeated cold starts.
    """

    _models: dict = {}  # Cached Whisper models by (engine, name) (shared across requests)

    @classmethod
    def model_name_for(cls, field_key: Optional[str]) -> str:
        """Yes/no replies use the small, fast model; everything else the main one."""
        if field_key == "confirm" and settings.WHISPER_CONFIRM_MODEL:
            return settings.WHISPER_CONFIRM_MODEL
        return settings.WHISPER_MODEL

    @classmethod
    def get_model(cls, name: Optional[str] = None):
        """Load a model once and cache it (lazy initialization)."""
        name = name or settings.WHISPER_MODEL
        key = (_engine(), name)
        if key not in cls._models:
            with _model_lock:
                if key not in cls._models:
                    logger.info(f"Loading Whisper model: {name} ({key[0]} engine)")
                    cls._models[key] = _load_model(*key)
                    logger.info("Whisper model loaded successfully.")
        return cls._models[key]

    @classmethod
    def warm_up(cls) -> None:
        """Load the models (and run one short pass) at startup so the first answer is fast."""
        try:
            for name in {settings.WHISPER_MODEL, cls.model_name_for("confirm")}:
                model = cls.get_model(name)
                with _model_lock:
                    _run(_engine(), model, np.zeros(WHISPER_SAMPLE_RATE // 2, dtype=np.float32), None)
            logger.info("Whisper warmed up.")
        except Exception as exc:  # never block startup on this
            logger.warning(f"Whisper warm-up failed: {exc}")

    @classmethod
    def _build_prompt(cls, field_key: Optional[str], town: Optional[str] = None,
                      province: Optional[str] = None) -> Optional[str]:
        if not field_key:
            return None
        if field_key == "address_barangay":
            return barangay_prompt(town)
        if field_key == "place_of_birth_town":
            return birth_town_prompt(province)
        return FIELD_PROMPTS.get(field_key)

    @classmethod
    def transcribe_file(cls, audio_path: Union[str, np.ndarray], field_key: Optional[str] = None,
                        town: Optional[str] = None, province: Optional[str] = None) -> dict:
        """
        Transcribe audio using Whisper.

        Args:
            audio_path: Path to an audio file (WAV, MP3, M4A, etc. — decoded by
                        ffmpeg), or 16 kHz mono float32 samples (no ffmpeg needed).
            field_key: Optional interview field key, used to bias decoding
                       toward expected vocabulary via initial_prompt.
            town: PSGC code of the town already answered ("address_barangay" hint).
            province: PSGC code of the province of birth already answered ("place_of_birth_town" hint).

        Returns:
            dict with keys: transcript, language, confidence, segments,
                            processing_time_seconds
        """
        model = cls.get_model(cls.model_name_for(field_key))
        start_time = time.time()
        initial_prompt = cls._build_prompt(field_key, town, province)

        source = audio_path if isinstance(audio_path, str) else f"{len(audio_path) / WHISPER_SAMPLE_RATE:.1f}s of audio"
        logger.info(f"Transcribing {source} (field={field_key})")

        with _model_lock:
            result = _run(_engine(), model, audio_path, initial_prompt)

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
    def transcribe_bytes(cls, audio_bytes: bytes, suffix: str = ".wav", field_key: Optional[str] = None,
                         town: Optional[str] = None, province: Optional[str] = None) -> dict:
        """
        Transcribe audio from raw bytes (e.g., uploaded file from API).
        WAV (what the web app uploads) is decoded here, so it works without
        ffmpeg; other formats go through Whisper's ffmpeg-based loader.
        """
        samples = _decode_wav(audio_bytes)
        if samples is not None:
            return cls.transcribe_file(samples, field_key=field_key, town=town, province=province)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = cls.transcribe_file(tmp_path, field_key=field_key, town=town, province=province)
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
            result = _run(_engine(), model, audio_array, initial_prompt)

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