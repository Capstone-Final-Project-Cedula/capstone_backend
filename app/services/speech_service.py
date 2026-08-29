"""
app/services/speech_service.py
OpenAI Whisper ASR integration.
Handles audio file transcription and microphone capture.

CHANGED: FIELD_PROMPTS["address"] now includes the real list of barangay
names from the gazetteer, so Whisper is biased toward recognizing them
correctly in the first place (not just corrected after the fact).
"""
import whisper
import numpy as np
import time
import tempfile
import os
from typing import Optional
from app.core.config import settings
from app.services.barangay_gazetteer import get_barangay_list_for_prompt
import logging

logger = logging.getLogger(__name__)


# Context hints per interview field — dramatically improves accuracy on
# short, ambiguous answers by biasing Whisper's decoding toward expected
# vocabulary. Keep these short; long prompts can backfire.
FIELD_PROMPTS = {
    "surname": "Filipino surname. Examples: Santos, Reyes, Dela Cruz, Bautista, Garcia, Mendoza.",
    "given_name": "Filipino given name. Examples: Maria, Jose, Juan, Ana, Ramon.",
    "middle_name": "Filipino middle name, or the word none.",
    "address": (
        "A Philippine address with house number and street, in one of "
        f"these barangays of Mambajao, Camiguin: {get_barangay_list_for_prompt()}."
    ),
    "date": "A date of birth in year month day format, for example 1990 dash 01 dash 15.",
    "sex": "The answer is either Male or Female.",
    "civil_status": "The answer is one of: Single, Married, Widowed, Separated.",
    "citizenship": "A citizenship, usually Filipino.",
    "icr_no": "An ICR number, or blank if not applicable.",
    "place_of_birth": "A Philippine city or municipality and province.",
    "height": "A height in centimeters, a number like 160 or 172.",
    "weight": "A weight in kilograms, a number like 55 or 68.",
    "occupation": "An occupation. Examples: Teacher, Farmer, Driver, Vendor, Fisherman.",
    "gross_annual_income": "A gross annual income amount in Philippine pesos, a number.",
    "spelling": "A name being spelled out letter by letter, or the word yes.",
}


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
            logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL}")
            cls._model = whisper.load_model(settings.WHISPER_MODEL)
            logger.info("Whisper model loaded successfully.")
        return cls._model

    @classmethod
    def _build_prompt(cls, field_key: Optional[str]) -> Optional[str]:
        if not field_key:
            return None
        return FIELD_PROMPTS.get(field_key)

    @classmethod
    def transcribe_file(cls, audio_path: str, field_key: Optional[str] = None) -> dict:
        """
        Transcribe an audio file using Whisper.

        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, etc.)
            field_key: Optional interview field key, used to bias decoding
                       toward expected vocabulary via initial_prompt.

        Returns:
            dict with keys: transcript, language, confidence, segments,
                            processing_time_seconds
        """
        model = cls.get_model()
        start_time = time.time()
        initial_prompt = cls._build_prompt(field_key)

        logger.info(f"Transcribing audio file: {audio_path} (field={field_key})")

        result = model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            verbose=False,
            fp16=False,                        # Disable FP16 for CPU compatibility
            condition_on_previous_text=False,  # short single answers, not long-form speech
            no_speech_threshold=settings.WHISPER_NO_SPEECH_THRESHOLD,
            logprob_threshold=settings.WHISPER_LOGPROB_THRESHOLD,
            initial_prompt=initial_prompt,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # fallback ladder, cuts hallucination
        )

        processing_time = time.time() - start_time

        segments = result.get("segments", [])
        if segments:
            avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segments])
            confidence = float(np.clip(np.exp(avg_logprob), 0.0, 1.0))
        else:
            confidence = 0.0

        transcript = result.get("text", "").strip()

        logger.info(f"Transcription complete in {processing_time:.2f}s | Confidence: {confidence:.2f}")
        logger.info(f"Transcript: {transcript}")

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
        """
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

        result = model.transcribe(
            audio_array,
            language="en",
            task="transcribe",
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,
            no_speech_threshold=settings.WHISPER_NO_SPEECH_THRESHOLD,
            logprob_threshold=settings.WHISPER_LOGPROB_THRESHOLD,
            initial_prompt=initial_prompt,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )

        processing_time = time.time() - start_time
        segments = result.get("segments", [])
        confidence = (
            float(np.clip(np.exp(np.mean([s.get("avg_logprob", -1.0) for s in segments])), 0, 1))
            if segments else 0.0
        )

        return {
            "transcript": result.get("text", "").strip(),
            "language": result.get("language", "en"),
            "confidence": confidence,
            "segments": segments,
            "processing_time_seconds": round(processing_time, 3),
        }