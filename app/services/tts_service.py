"""
app/services/tts_service.py
Text-to-speech voice feedback using Microsoft Edge neural TTS (edge-tts).
Reads extracted CTC fields back to the applicant for verification.

Requires internet access (calls Microsoft's free Edge Read Aloud service).
Falls back to pyttsx3 (offline, lower quality) automatically if the
network call fails, so the kiosk never goes silent.

    pip install edge-tts pygame
"""
import asyncio
import os
import tempfile
import threading
import logging
from collections import OrderedDict
from typing import Optional

import edge_tts

logger = logging.getLogger(__name__)

# Thread lock — only one voice line plays at a time on the kiosk
_tts_lock = threading.Lock()

# Calm, deep, natural-sounding voice — closest "AI assistant" tone available
# in Edge's neural voice set without attempting to clone any copyrighted
# character's actual voice performance.
# Alternatives to try: "en-US-GuyNeural" (US male), "en-US-DavisNeural"
# (US male, slightly warmer), "en-GB-ThomasNeural" (UK male, younger).
DEFAULT_VOICE = "en-GB-RyanNeural"


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


def _rate_to_edge_percent(rate: int) -> str:
    """Convert a words-per-minute-ish rate into edge-tts's percent-offset format."""
    percent = int((rate - 175) / 175 * 100)
    percent = max(-50, min(50, percent))
    sign = "+" if percent >= 0 else ""
    return f"{sign}{percent}%"


def _volume_to_edge_percent(volume: float) -> str:
    """Convert a 0.0-1.0 volume into edge-tts's percent-offset format."""
    percent = int((volume - 1.0) * 100)
    percent = max(-100, min(0, percent))
    sign = "+" if percent >= 0 else ""
    return f"{sign}{percent}%"


async def _synthesize_to_file(text: str, out_path: str, rate: int, volume: float, voice: str) -> None:
    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        rate=_rate_to_edge_percent(rate),
        volume=_volume_to_edge_percent(volume),
    )
    await communicate.save(out_path)


def _play_audio_file(path: str) -> None:
    import pygame

    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    pygame.mixer.quit()


def _speak_text_offline_fallback(text: str, rate: int, volume: float) -> bool:
    """Offline fallback using pyttsx3, used only if edge-tts/network fails."""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        voices = engine.getProperty("voices")
        for v in voices:
            if "english" in v.name.lower() or "en" in v.id.lower():
                engine.setProperty("voice", v.id)
                break
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return True
    except Exception as e:
        logger.error(f"Offline TTS fallback also failed: {e}")
        return False


def speak_verification(fields: dict, rate: int = 185, volume: float = 1.0, voice: str = DEFAULT_VOICE) -> bool:
    """
    Read extracted form fields back to the applicant using TTS.

    Args:
        fields: Dict of extracted BIR Form 0016 field values
        rate: Speech rate, ~185 = normal conversational pace (default 185)
        volume: Volume from 0.0 to 1.0 (default 1.0)
        voice: edge-tts neural voice name (default: calm deep UK male)

    Returns:
        True if speech completed, False on error
    """
    script = _build_verification_script(fields)
    return speak_text(script, rate=rate, volume=volume, voice=voice)


def speak_text(text: str, rate: int = 185, volume: float = 1.0, voice: str = DEFAULT_VOICE) -> bool:
    """
    Speak arbitrary text using Microsoft Edge neural TTS, with an
    automatic offline fallback to pyttsx3 if the network call fails.

    Args:
        text: Text to speak
        rate: Words-per-minute-ish rate (default 185)
        volume: Volume (0.0 - 1.0)
        voice: edge-tts neural voice name

    Returns:
        True on success, False on error
    """
    with _tts_lock:
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
            os.close(tmp_fd)

            asyncio.run(_synthesize_to_file(text, tmp_path, rate, volume, voice))
            _play_audio_file(tmp_path)
            return True

        except Exception as e:
            logger.warning(f"edge-tts failed ({e}), falling back to offline TTS.")
            return _speak_text_offline_fallback(text, rate, volume)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# The interview asks the same questions of every applicant, so generated audio is
# kept in memory: a repeated line plays at once instead of waiting ~1.2 s for the
# online voice service. Only good online (edge-tts) audio is cached.
_AUDIO_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_AUDIO_CACHE_MAX = 400
_audio_cache_lock = threading.Lock()


def synthesize_bytes(text: str, rate: int = 185, volume: float = 1.0, voice: str = DEFAULT_VOICE) -> tuple:
    """
    Synthesize speech and return the raw audio bytes instead of playing
    them on the server machine. Use this when the browser (not the
    server) should be the one producing sound — e.g. the web frontend.

    Falls back to pyttsx3's offline file export if edge-tts/network fails.

    Returns:
        (audio_bytes, mime_type) tuple
    """
    key = (text, rate, volume, voice)
    with _audio_cache_lock:
        if key in _AUDIO_CACHE:
            _AUDIO_CACHE.move_to_end(key)
            return _AUDIO_CACHE[key]
    result = _synthesize_bytes_uncached(text, rate, volume, voice)
    if result[0] and result[1] == "audio/mpeg":
        with _audio_cache_lock:
            _AUDIO_CACHE[key] = result
            while len(_AUDIO_CACHE) > _AUDIO_CACHE_MAX:
                _AUDIO_CACHE.popitem(last=False)
    return result


def _synthesize_bytes_uncached(text: str, rate: int, volume: float, voice: str) -> tuple:
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(tmp_fd)

        asyncio.run(_synthesize_to_file(text, tmp_path, rate, volume, voice))
        with open(tmp_path, "rb") as f:
            return f.read(), "audio/mpeg"

    except Exception as e:
        logger.warning(f"edge-tts failed ({e}), falling back to offline TTS export.")
        return _synthesize_bytes_offline_fallback(text, rate, volume)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _synthesize_bytes_offline_fallback(text: str, rate: int, volume: float) -> tuple:
    """Offline fallback: render speech to a WAV file via pyttsx3 and return its bytes."""
    tmp_path = None
    try:
        import pyttsx3

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)

        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        voices = engine.getProperty("voices")
        for v in voices:
            if "english" in v.name.lower() or "en" in v.id.lower():
                engine.setProperty("voice", v.id)
                break
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        with open(tmp_path, "rb") as f:
            return f.read(), "audio/wav"

    except Exception as e:
        logger.error(f"Offline TTS export also failed: {e}")
        return b"", "audio/mpeg"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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
