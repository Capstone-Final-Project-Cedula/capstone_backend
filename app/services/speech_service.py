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
from difflib import SequenceMatcher
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

# Hints whose sentences are each a real answer or command ("Lagbas po.", "Purok 2.", "Type."), or whose first
# sentence looks like one ("Barangay in Baungon, Bukidnon."): only a run of two or more of their sentences is an
# echo of the hint, never one sentence alone (plan 002.2).
ANSWER_LIKE_HINTS = {"surname", "given_name", "middle_name", "address_purok", "address_barangay",
                     "missing_choice", "spelling"}
ECHO_SIMILARITY = 0.85

_model_lock = threading.Lock()   # one transcription at a time on the shared model

# One Whisper model and beam per interview question (plan 006, Ralph's "PLAN 6" table).
# All answer fields and spelling: Medium b5. The type-or-spell choice keeps Base b1.
# Yes/no: Tiny b5. A key that is not here (or none) uses WHISPER_MODEL + WHISPER_BEAM_SIZE.
FIELD_MODELS = {
    **{key: ("medium", 5) for key in (
        "surname", "given_name", "middle_name",
        "address", "address_town", "address_barangay", "address_purok",
        "place_of_birth", "place_of_birth_province", "place_of_birth_town",
        "occupation", "spelling",
        "date", "sex", "civil_status", "citizenship", "height", "weight",
        "gross_annual_income", "icr_no")},
    "missing_choice": ("base", 1),
    "confirm": ("tiny", 5),
    "review": ("tiny", 5),      # first pass; a reply that is not a plain yes is decoded again with REVIEW_SECOND
}
REVIEW_SECOND = ("medium", 5)

_PLAIN_YES = re.compile(r"^(?:yes|yeah|yep|opo|oo|okay|ok|correct|right|sure)(?: (?:po|please))?$")
_FIELD_WORDS = re.compile(
    r"\b(?:change|but|no|not|name|surname|birth|address|purok|barangay|sex|civil|status|height|weight|"
    r"occupation|citizenship|income|male|female|single|married)\b")


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
# One interview answer (even a spelled name) is far below this; the cap keeps a decode that loops on
# the same words short (plan 002: small/beam-5 turned one address into 157 words, 7 of them distinct).
MAX_ANSWER_TOKENS = 128
REPETITIVE_RATIO = 2.4   # Whisper's own compression-ratio limit: above it the text is a repetition loop


def _beam(beam: Optional[int] = None) -> int:
    return max(1, int(settings.WHISPER_BEAM_SIZE if beam is None else beam))


def _temperatures(beam: int):
    # Beam search gets no fallback ladder: each retry re-runs the whole beam, which is how one
    # small/beam-5 answer took 56 s (plan 002). One beam pass is already the careful decode.
    return FALLBACK_TEMPERATURES if beam == 1 else 0.0


def _decode_options(initial_prompt: Optional[str], beam: Optional[int] = None) -> dict:
    """openai-whisper short-answer settings: greedy by default, small fallback ladder for unclear audio."""
    beam = _beam(beam)
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
        sample_len=MAX_ANSWER_TOKENS,
    )
    if beam > 1:
        options.update(beam_size=beam, best_of=beam)
    return options


def _faster_options(initial_prompt: Optional[str], beam: Optional[int] = None) -> dict:
    """The same settings for faster-whisper. Every option is given: its defaults differ (beam 5,
    condition_on_previous_text on, a 6-step temperature ladder, no VAD)."""
    beam = _beam(beam)
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
        max_new_tokens=MAX_ANSWER_TOKENS,
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


def _run(engine: str, model, audio: Union[str, np.ndarray], initial_prompt: Optional[str],
         warm_up: bool = False, beam: Optional[int] = None) -> dict:
    """One decode on either engine, returned in openai-whisper's shape: text, language, segments."""
    if engine == "faster":
        options = _faster_options(initial_prompt, beam)
        if warm_up:
            options["vad_filter"] = False  # the VAD cuts the silent warm-up clip away: the model would never run
        segments, info = model.transcribe(audio, **options)
        segments = list(segments)          # a lazy generator: decoding happens here
        return {
            "text": "".join(s.text for s in segments),
            "language": info.language,
            "segments": [{"id": s.id, "start": s.start, "end": s.end, "text": s.text,
                          "avg_logprob": s.avg_logprob, "compression_ratio": s.compression_ratio} for s in segments],
        }
    return model.transcribe(audio, **_decode_options(initial_prompt, beam))


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[0-9a-zñ]+", (text or "").lower()))


def is_plain_yes(text: str) -> bool:
    """The whole reply is a yes ("Yes." / "Opo" / "yes po"): at most 3 words, no field name, no "change"/"but".
    "Yes, my surname is Cruz" is a correction, never an approval (plan 006)."""
    words = _normalize(text)
    return bool(words) and len(words.split()) <= 3 and not _FIELD_WORDS.search(words) and bool(_PLAIN_YES.match(words))


def _sentences(text: str) -> list:
    return [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if _normalize(s)]


def _sentence_words(text: str) -> str:
    """A sentence compared with the hint: normalized, without the polite "po" ("Tagaylo po." = "Tagaylo.")."""
    return " ".join(w for w in _normalize(text).split() if w != "po")


def _same_sentence(said: str, hint: str, cutoff: float = ECHO_SIMILARITY) -> bool:
    """One sentence read back from the hint: equal, or close with the same numbers ("Purok 5." is not "Purok 2.")."""
    if not said or not hint:
        return False
    if said == hint:
        return True
    return (re.findall(r"\d+", said) == re.findall(r"\d+", hint)
            and SequenceMatcher(None, said, hint).ratio() >= cutoff)


_WORD = re.compile(r"[0-9a-zñ]+")


def _hint_run(words: list, hint: list) -> tuple:
    """(hint sentences, words) of the longest run of the hint's sentences, in the hint's order, at the start
    of the words. Matched on words, not on the transcript's punctuation: Whisper may merge or split sentences."""
    best = (0, 0)
    for first in range(len(hint)):
        at = count = 0
        for sentence in hint[first:]:
            size = len(sentence.split())
            for n in (size, size - 1, size + 1):          # "Gerlie Mae" may come back as "Gerliemae"
                if n > 0 and at + n <= len(words) and _same_sentence(" ".join(words[at:at + n]), sentence):
                    at += n
                    count += 1
                    break
            else:
                break
        best = max(best, (count, at))
    return best


def _clean_answer(text: str, segments: Optional[list] = None, prompt: Optional[str] = None,
                  field_key: Optional[str] = None) -> tuple:
    """(transcript, why it was dropped or None). Drops a decoding loop, a silence phrase and the hint read
    back; a reply that starts with the hint and then answers keeps only the answer (plan 002.2).
    For the answer-like hints (names, purok, the end-step commands) an echo is two or more hint sentences in
    a row, so one hint example said alone ("Lagbas po.") stays an answer."""
    text = (text or "").strip()
    if any(s.get("compression_ratio", 0) > REPETITIVE_RATIO for s in segments or []):
        logger.info("Repetitive transcript dropped (a decoding loop, not an answer).")
        return "", "repetition"
    if HALLUCINATIONS.match(text):
        return "", "silence-phrase"
    hint = [_sentence_words(s) for s in _sentences(prompt or "")]
    kept = [(m.group(), m.start()) for m in _WORD.finditer(text.lower()) if m.group() != "po"]
    if not hint or not kept:
        return text, None
    shortest = min(2 if field_key in ANSWER_LIKE_HINTS else 1, len(hint))
    count, used = _hint_run([w for w, _ in kept], hint)
    if count < shortest:
        return text, None
    if used == len(kept):
        logger.info("Transcript dropped: it repeats the question's hint (field=%s).", field_key)
        return "", "echo"
    return text[kept[used][1]:].strip(), None


def _clean_transcript(text: str, segments: Optional[list] = None) -> str:
    """Drop what Whisper invents: silence phrases, and a decode stuck repeating the same words."""
    return _clean_answer(text, segments)[0]


class SpeechService:
    """
    Singleton service wrapping OpenAI Whisper ASR.
    Model is loaded once on first use to avoid repeated cold starts.
    """

    _models: dict = {}  # Cached Whisper models by (engine, name) (shared across requests)

    @classmethod
    def model_and_beam_for(cls, field_key: Optional[str], second_pass: bool = False) -> tuple:
        """(model name, beam) for a question. With WHISPER_FIELD_MODELS off, every question uses
        WHISPER_MODEL + WHISPER_BEAM_SIZE, and only that model is loaded (plan 006, Codex verify)."""
        if not settings.WHISPER_FIELD_MODELS:
            return settings.WHISPER_MODEL, _beam()
        if second_pass:
            return REVIEW_SECOND
        return FIELD_MODELS.get(field_key or "", (settings.WHISPER_MODEL, _beam()))

    @classmethod
    def model_name_for(cls, field_key: Optional[str]) -> str:
        return cls.model_and_beam_for(field_key)[0]

    @classmethod
    def _decode(cls, audio: Union[str, np.ndarray], field_key: Optional[str], prompt: Optional[str]) -> dict:
        """Decode with the question's model and beam. `review` is two-pass: Tiny first; unless that is a plain
        yes (or silence), the same audio is decoded again with Medium, whose text is used."""
        passes = [False]
        result: dict = {}
        ran: list = []   # what each pass really used, for the evaluation log (plan 003); decoding is unchanged
        for second in passes:
            name, beam = cls.model_and_beam_for(field_key, second)
            model = cls.get_model(name)
            logger.info(f"Whisper model for {field_key or 'no field'}: {name}, beam {beam}"
                        f"{' (second pass)' if second else ''}")
            pass_start = time.time()
            with _model_lock:
                result = _run(_engine(), model, audio, prompt, beam=beam)
            ran.append({"model": name, "beam": beam, "second_pass": second,
                        "seconds": round(time.time() - pass_start, 3), "text": result.get("text", "")})
            if field_key == "review" and not second and settings.WHISPER_FIELD_MODELS:
                heard = _clean_transcript(result.get("text", ""), result.get("segments", []))
                if heard and not is_plain_yes(heard):
                    passes.append(True)
        result["eval_passes"] = ran
        return result

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
            if settings.WHISPER_FIELD_MODELS:
                names = {name for name, _ in FIELD_MODELS.values()} | {REVIEW_SECOND[0]}
            else:
                names = {settings.WHISPER_MODEL}
            for name in sorted(names):
                model = cls.get_model(name)
                with _model_lock:
                    _run(_engine(), model, np.zeros(WHISPER_SAMPLE_RATE // 2, dtype=np.float32), None, warm_up=True)
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
        start_time = time.time()
        initial_prompt = cls._build_prompt(field_key, town, province)

        source = audio_path if isinstance(audio_path, str) else f"{len(audio_path) / WHISPER_SAMPLE_RATE:.1f}s of audio"
        logger.info(f"Transcribing {source} (field={field_key})")

        result = cls._decode(audio_path, field_key, initial_prompt)

        processing_time = time.time() - start_time

        segments = result.get("segments", [])
        if segments:
            avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segments])
            confidence = float(np.clip(np.exp(avg_logprob), 0.0, 1.0))
        else:
            confidence = 0.0

        transcript, dropped = _clean_answer(result.get("text", ""), segments, initial_prompt, field_key)

        # Only the length is logged: transcripts are applicants' personal data (Data Privacy Act).
        logger.info(f"Transcription complete in {processing_time:.2f}s | Confidence: {confidence:.2f} | "
                    f"{len(transcript)} characters")

        return {
            "transcript": transcript,
            "dropped": dropped,
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
            "passes": result.get("eval_passes", []),
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
        start_time = time.time()
        initial_prompt = cls._build_prompt(field_key)

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        audio_array = audio_array.astype(np.float32)

        result = cls._decode(audio_array, field_key, initial_prompt)

        processing_time = time.time() - start_time
        segments = result.get("segments", [])
        confidence = (
            float(np.clip(np.exp(np.mean([s.get("avg_logprob", -1.0) for s in segments])), 0, 1))
            if segments else 0.0
        )

        return {
            "transcript": _clean_transcript(result.get("text", ""), segments),
            "language": result.get("language", "en"),
            "confidence": confidence,
            "segments": segments,
            "processing_time_seconds": round(processing_time, 3),
        }
