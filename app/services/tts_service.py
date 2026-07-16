"""
app/services/tts_service.py
Text-to-speech voice feedback using pyttsx3.
Reads extracted CTC fields back to the applicant for verification.
"""
import pyttsx3
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Thread lock — pyttsx3 engine is not thread-safe
_tts_lock = threading.Lock()


def _build_verification_script(fields: dict) -> str:
    """
    Build a natural-language script that reads the extracted fields
    back to the applicant for confirmation.
    """
    lines = ["Please confirm the following information captured for your Cedula."]

    name = f"{fields.get('given_name', '')} {fields.get('surname', '')}".strip()
    if name:
        lines.append(f"Name: {name}.")

    if fields.get("date_of_birth"):
        lines.append(f"Date of birth: {fields['date_of_birth']}.")

    if fields.get("address"):
        lines.append(f"Address: {fields['address']}.")

    if fields.get("sex"):
        lines.append(f"Sex: {fields['sex']}.")

    if fields.get("civil_status"):
        lines.append(f"Civil status: {fields['civil_status']}.")

    if fields.get("citizenship"):
        lines.append(f"Citizenship: {fields['citizenship']}.")

    if fields.get("place_of_birth"):
        lines.append(f"Place of birth: {fields['place_of_birth']}.")

    if fields.get("height_cm"):
        lines.append(f"Height: {fields['height_cm']} centimeters.")

    if fields.get("weight_kg"):
        lines.append(f"Weight: {fields['weight_kg']} kilograms.")

    if fields.get("occupation"):
        lines.append(f"Occupation: {fields['occupation']}.")

    if fields.get("gross_annual_income"):
        income = fields["gross_annual_income"]
        lines.append(f"Gross annual income: {income:,.2f} pesos.")

    lines.append("If all information is correct, please proceed. Otherwise, notify the staff for corrections.")

    return " ".join(lines)


def speak_verification(fields: dict, rate: int = 150, volume: float = 1.0) -> bool:
    """
    Read extracted form fields back to the applicant using TTS.

    Args:
        fields: Dict of extracted BIR Form 0016 field values
        rate: Speech rate in words per minute (default 150)
        volume: Volume from 0.0 to 1.0 (default 1.0)

    Returns:
        True if speech completed, False on error
    """
    script = _build_verification_script(fields)
    return speak_text(script, rate=rate, volume=volume)


def speak_text(text: str, rate: int = 150, volume: float = 1.0) -> bool:
    """
    Speak arbitrary text using pyttsx3.

    Args:
        text: Text to speak
        rate: Words per minute
        volume: Volume (0.0 - 1.0)

    Returns:
        True on success, False on error
    """
    with _tts_lock:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)

            # Try to select a clearer voice if available
            voices = engine.getProperty("voices")
            for voice in voices:
                if "english" in voice.name.lower() or "en" in voice.id.lower():
                    engine.setProperty("voice", voice.id)
                    break

            engine.say(text)
            engine.runAndWait()
            engine.stop()
            return True

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False


def speak_async(text: str) -> None:
    """
    Speak text in a background thread (non-blocking).
    Use when the API response should not wait for speech to finish.
    """
    thread = threading.Thread(target=speak_text, args=(text,), daemon=True)
    thread.start()


def get_verification_script(fields: dict) -> str:
    """Return the verification script text without speaking it (for display in UI)."""
    return _build_verification_script(fields)
