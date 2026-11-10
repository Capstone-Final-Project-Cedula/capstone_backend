"""
app/services/eval_service.py
Evaluation mode (plan 003): the settings, the session log, scoring (three accuracy numbers, WER),
the aggregated statistics and the "delete evaluation audio now" batches.
Plan 009: generated made-up cards (the next session claims the oldest unclaimed one), WER pooled over every
counted session and split into S / D / I, number-lenient text rules, hand re-scoring from the saved audio.

Rules kept here:
  - logging never blocks an interview or a CTC: every writer swallows and logs its own errors;
  - the server derives tries, models and times from /speech/transcribe and /extraction/answer;
    the browser's one batch only adds what the server cannot see (typed vs staff, skips, seconds);
  - nothing is ever deleted automatically: only delete_batch(), called by the admin button.
"""
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import and_, case, distinct, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings as env_settings
from app.core.database import SessionLocal
from app.models.app_settings import AppSettings
from app.models.eval import (
    ATTEMPT_STEPS, EvalAnswer, EvalCard, EvalSession, SecurityEvent,
)
from app.services import eval_cards, storage_service

logger = logging.getLogger(__name__)

CARDS_PATH = Path(__file__).resolve().parents[1] / "eval" / "reading_cards.json"

# The 14 questions of the paper, in the order the interview asks them (plan 009: one overall WER, no subsets).
FIELDS14 = ("surname", "given_name", "middle_name", "address", "date", "sex", "civil_status", "citizenship",
            "icr_no", "place_of_birth", "height", "weight", "occupation", "gross_annual_income")
PARTS = {
    "address": ("address_town", "address_barangay", "address_purok"),
    "place_of_birth": ("place_of_birth_province", "place_of_birth_town"),
}
PART_KEYS = tuple(key for parts in PARTS.values() for key in parts)
PART_PARENT = {key: parent for parent, parts in PARTS.items() for key in parts}
LOGGED_FIELDS = frozenset(FIELDS14) | frozenset(PART_KEYS)
VOICE_STEPS = ("spoken", "spelled", "end_said")
NONE_FIELDS = frozenset({"middle_name", "icr_no", "address_purok"})     # "none" is a real answer here
NUMBER_FIELDS = frozenset({"height", "weight", "gross_annual_income"})
STOP_WORDS_FULL = frozenset({"barangay", "brgy", "bukidnon"})            # only on the joined address / birthplace
FILLER = re.compile(r"^(?:u+h*m+|u+h+|h+m+|m+|e+h+|a+h+|er+m*|mm+)$")
COUNTED_TARGET = 30          # participants (plan 009)
WER_TARGET_PCT = 15.0        # paper 3.5.12
EXTRACTION_TARGET_PCT = 90.0


# ─── Settings ─────────────────────────────────────────────────────────────────

def _row(db: Session) -> Optional[AppSettings]:
    return db.get(AppSettings, 1)


def get_eval_settings(db: Session) -> dict:
    """The evaluation settings (admin view). No row yet = the defaults (on, practice, quiet)."""
    row = _row(db)
    return {
        "evaluation_mode": True if row is None else bool(row.evaluation_mode),
        "round_type": "practice" if row is None else row.eval_round_type,
        "round_condition": "quiet" if row is None else row.eval_round_condition,
        "card_set": None if row is None else row.eval_round_card_set,
        "round_no": 1 if row is None else row.eval_round_no,
        "counts": True if row is None else bool(row.eval_counts),
    }


def recording_on(db: Session) -> bool:
    """Evaluation mode on (no settings row yet = on)."""
    row = _row(db)
    return True if row is None else bool(row.evaluation_mode)


def update_eval_settings(db: Session, values: dict, admin_id: int, new_round: bool = False) -> dict:
    """Admin only (the route enforces it). A changed round tag, or new_round, starts round_no + 1."""
    from app.services import settings_service
    row = _row(db)
    if row is None:
        row = AppSettings(id=1, **vars(settings_service._defaults()))
        row.evaluation_mode, row.eval_round_type = True, "practice"
        row.eval_round_condition, row.eval_round_no = "quiet", 1
        row.eval_counts = True
        db.add(row)
        db.flush()
    changed = False
    mapping = {
        "round_type": "eval_round_type", "round_condition": "eval_round_condition",
        "card_set": "eval_round_card_set",
    }
    if "evaluation_mode" in values and values["evaluation_mode"] is not None:
        row.evaluation_mode = bool(values["evaluation_mode"])
    if "counts" in values and values["counts"] is not None:
        row.eval_counts = bool(values["counts"])     # read by the next session; never starts a new round
    for key, column in mapping.items():
        if key in values:
            value = values[key]
            if getattr(row, column) != value:
                changed = True
            setattr(row, column, value)
    if changed or new_round:
        row.eval_round_no = (row.eval_round_no or 1) + 1
    row.updated_by = admin_id
    db.commit()
    return get_eval_settings(db)


# ─── Reading cards ────────────────────────────────────────────────────────────

@lru_cache()
def load_cards() -> dict:
    with open(CARDS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def card_set_names() -> List[str]:
    return list(load_cards()["sets"].keys())


def card_answers(db: Session, session: EvalSession) -> Optional[dict]:
    """
    The card a session read, as {field: {expected, read, parts?}}, or None (a "no card" session: no WER, and
    accuracy is compared with the saved value). Plan 009 sessions carry a generated card (`card_id`); plan 003
    scripted sessions keep their fixed card number and are scored as before.
    """
    if session.card_id:
        card = db.get(EvalCard, session.card_id)
        return json.loads(card.card)["answers"] if card is not None else None
    if session.round_type == "scripted" and session.card_no:
        card = next((c for c in load_cards()["cards"] if c["no"] == session.card_no), None)
        return card["answers"] if card is not None else None
    return None


def expected_in(answers: Optional[dict], field: str) -> Optional[dict]:
    """The card's {expected, read} for a field or a part, else None."""
    if not answers:
        return None
    if field in answers:
        return answers[field]
    parent = PART_PARENT.get(field)
    return answers.get(parent, {}).get("parts", {}).get(field) if parent else None


# ─── Generated cards (plan 009) ──────────────────────────────────────────────

def create_card(db: Session, admin_id: Optional[int]) -> EvalCard:
    """"Generate card": a new made-up person, waiting to be claimed. Never a copy of the newest card."""
    count = db.query(func.count(EvalCard.id)).scalar() or 0
    newest = db.query(EvalCard).order_by(EvalCard.id.desc()).first()
    newest_answers = json.loads(newest.card)["answers"] if newest is not None else None
    for _ in range(5):
        data = eval_cards.generate_card(eval_cards.style_for(count))
        if data["answers"] != newest_answers:
            break
    card = EvalCard(created_by=admin_id, round_no=get_eval_settings(db)["round_no"], style=data["style"],
                    card=json.dumps(data, ensure_ascii=False))
    db.add(card)
    db.commit()
    return card


def next_card(db: Session) -> Optional[EvalCard]:
    """The card the next session will claim: the oldest one nobody has claimed."""
    return (db.query(EvalCard).filter(EvalCard.claimed_by_session_id.is_(None))
            .order_by(EvalCard.id).first())


def waiting_cards(db: Session) -> int:
    return db.query(func.count(EvalCard.id)).filter(EvalCard.claimed_by_session_id.is_(None)).scalar() or 0


def _claim_card(db: Session, session: EvalSession) -> None:
    """Give the session the oldest unclaimed card; two stations starting at once never get the same one."""
    for _ in range(3):
        card = next_card(db)
        if card is None:
            return
        taken = (db.query(EvalCard)
                 .filter(EvalCard.id == card.id, EvalCard.claimed_by_session_id.is_(None))
                 .update({EvalCard.claimed_by_session_id: session.id,
                          EvalCard.claimed_at: datetime.now(timezone.utc)}, synchronize_session=False))
        if taken:
            session.card_id = card.id
            return


def _release_unused_card(db: Session, session: EvalSession) -> None:
    """A session that ended without one Whisper try (QR, typed, or left at once) gives its card back."""
    if not session.card_id:
        return
    heard = (db.query(func.count(EvalAnswer.id))
             .filter(EvalAnswer.session_id == session.id, EvalAnswer.engine == "whisper",
                     EvalAnswer.step.in_(VOICE_STEPS)).scalar() or 0)
    if heard:
        return
    db.query(EvalCard).filter(EvalCard.id == session.card_id).update(
        {EvalCard.claimed_by_session_id: None, EvalCard.claimed_at: None}, synchronize_session=False)
    session.card_id = None


# ─── Comparing values ─────────────────────────────────────────────────────────

_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]


def norm(field: str, value) -> str:
    """One comparable form for a value (the app's own forms: ISO birthday, plain numbers, no case)."""
    text = "" if value is None else str(getattr(value, "value", value)).strip()
    if field == "date":
        match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
        if match:
            return text
        match = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})", text)
        if match and match.group(1).lower() in _MONTHS:
            return f"{int(match.group(3)):04d}-{_MONTHS.index(match.group(1).lower()) + 1:02d}-{int(match.group(2)):02d}"
        return text.lower()
    if field in NUMBER_FIELDS:
        found = re.search(r"\d[\d,]*(?:\.\d+)?", text)
        if not found:
            return "0" if field == "gross_annual_income" and (text == "" or text.lower() in ("none", "wala")) else ""
        number = float(found.group().replace(",", ""))
        if number == 0 and field != "gross_annual_income":
            return ""
        return str(int(number)) if number.is_integer() else repr(number)
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    if field in NONE_FIELDS and words in (["none"], ["wala"], ["walay"]):
        return ""
    if field in ("address", "place_of_birth"):
        words = [w for w in words if w not in STOP_WORDS_FULL]
    return " ".join(words)


def _words(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", (text or "").lower()).split()


_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
_TEENS = {w: i + 10 for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: (i + 2) * 10 for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 1000, "million": 1000000}
_ORDINALS = {
    **{w: i + 1 for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth".split())},
    **{w: i + 10 for i, w in enumerate("tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth "
                                       "seventeenth eighteenth nineteenth".split())},
    "twentieth": 20, "thirtieth": 30,
}
_MONTH_WORDS = frozenset(_MONTHS)


def _kind(word: str, ordinal_ok: bool) -> Optional[Tuple[str, int]]:
    if word in _UNITS:
        return "unit", _UNITS[word]
    if word in _TEENS:
        return "teen", _TEENS[word]
    if word in _TENS:
        return "tens", _TENS[word]
    if word == "hundred":
        return "hundred", 100
    if word in _SCALES:
        return "scale", _SCALES[word]
    if ordinal_ok and word in _ORDINALS:            # "march fourteenth", "twenty first"
        value = _ORDINALS[word]
        return ("unit" if value < 10 else "teen" if value < 20 else "tens"), value
    return None


def _number_groups(run: List[Tuple[str, int]]) -> List[str]:
    """Number words -> digits: "one hundred twenty thousand" -> 120000; "nineteen ninety" -> 1990."""
    groups: List[int] = []
    total = current = 0
    last = None
    for kind, value in run:
        if kind in ("unit", "teen", "tens", "digit") and (
                last in ("unit", "teen", "digit") or (last == "tens" and kind != "unit")):
            groups.append(total + current)        # "two five" or "nineteen ninety": a new number begins
            total = current = 0
        if kind == "hundred":
            current = (current or 1) * 100
        elif kind == "scale":
            total += (current or 1) * value
            current = 0
        else:
            current += value
        last = kind
    groups.append(total + current)
    if len(groups) >= 2 and all(10 <= g <= 99 for g in groups[-2:]) and (len(groups) == 2 or groups[0] <= 31):
        # A year said in two halves, after a day too ("fourteen nineteen ninety" -> 14 1990).
        return [str(g) for g in groups[:-2]] + [f"{groups[-2]}{groups[-1]:02d}"]
    if len(groups) > 1 and all(g < 10 for g in groups):
        return ["".join(str(g) for g in groups)]          # digit by digit: "one five eight" -> 158
    return [str(g) for g in groups]


def wer_words(text: Optional[str]) -> List[str]:
    """
    The one text rule for WER (plan 009), used for the card, the transcript and a hand re-score alike:
    lower case, no punctuation, spaces collapsed, thousands commas and ordinal endings dropped from numbers,
    and number words written as digits ("fifty five" = "55"). Names and Bisaya / Tagalog words are kept
    exactly as they are: a misheard name is a real error.
    """
    text = (text or "").lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)          # 180,000 -> 180000
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", text)  # 14th -> 14
    words = re.sub(r"[^\w\s,]", " ", text.replace(",", " , ")).split()    # a comma still ends a number
    out: List[str] = []
    run: List[Tuple[str, int]] = []

    def flush():
        if run:
            out.extend(_number_groups(run))
            run.clear()

    for i, word in enumerate(words):
        if word == ",":
            flush()                                         # "march fourteen, nineteen ninety"
            continue
        nxt = words[i + 1] if i + 1 < len(words) else ""
        kind = _kind(word, ordinal_ok=bool(run) or (out[-1] in _MONTH_WORDS if out else False))
        if kind is None and word.isdigit() and (nxt == "hundred" or nxt in _SCALES):
            kind = ("digit", int(word))                     # "180 thousand"
        if kind is None and word == "and" and run and run[-1][0] in ("hundred", "scale") and _kind(nxt, True):
            continue                                        # "one hundred and twenty"
        if kind is None:
            flush()
            out.append(word)
        else:
            run.append(kind)
            if word in _ORDINALS:
                flush()                                     # nothing follows an ordinal: "fourteenth, 1990"
    flush()
    return out


def wer_split(reference: Optional[str], heard: Optional[str]) -> Tuple[int, int, int, int]:
    """(substitutions, deletions, insertions, reference words) of the best word alignment."""
    ref, hyp = wer_words(reference), wer_words(heard)
    n, m = len(ref), len(hyp)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist[i][j] = min(dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), dist[i - 1][j] + 1, dist[i][j - 1] + 1)
    subs = dels = ins = 0
    i, j = n, m
    while i or j:
        if i and j and dist[i][j] == dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            subs += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and dist[i][j] == dist[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return subs, dels, ins, n


def wer_counts(reference: str, heard: str) -> Tuple[int, int]:
    """(word errors S + D + I, reference words N)."""
    subs, dels, ins, n = wer_split(reference, heard)
    return subs + dels + ins, n


def is_filler(transcript: Optional[str]) -> bool:
    """Empty, or nothing but "um/uh": not a try (plan 006: fillers are never an answer or a try)."""
    words = _words(transcript or "")
    return not words or all(FILLER.match(w) for w in words)


# ─── Sessions ─────────────────────────────────────────────────────────────────

def start_session(db: Session, user, input_method: str, device: Optional[str], build_label: str = "") -> Optional[EvalSession]:
    """"I agree, start": a logged session tagged with the current round. None when logging is off."""
    if not recording_on(db):
        return None
    cfg = get_eval_settings(db)
    for _ in range(2):          # a second volunteer starting at the same moment gets the next number
        number = (db.query(func.max(EvalSession.participant_no))
                  .filter(EvalSession.round_no == cfg["round_no"]).scalar() or 0) + 1
        session = EvalSession(
            round_no=cfg["round_no"], round_type=cfg["round_type"], round_condition=cfg["round_condition"],
            card_set=cfg["card_set"] if cfg["round_type"] == "scripted" else None,
            participant_no=number, participant_code=f"P{number:03d}",
            input_method=input_method, device=(device or "")[:200] or None,
            build_label=build_label[:60] or None, staff_user_id=user.id, outcome="open",
            counts=cfg["counts"],
        )
        db.add(session)
        try:
            db.flush()
            _claim_card(db, session)     # plan 009: the oldest unclaimed generated card (none left: "no card")
            db.commit()
            return session
        except IntegrityError:
            db.rollback()
    return None


def own_open_session(db: Session, session_id: Optional[int], user_id: Optional[int]) -> Optional[EvalSession]:
    if not session_id:
        return None
    session = db.get(EvalSession, session_id)
    if session is None or (user_id is not None and session.staff_user_id != user_id):
        return None
    return session


# ─── Logging a try ────────────────────────────────────────────────────────────

def _try_row(db: Session, session_id: int, try_id: Optional[str], field: str, step: str) -> EvalAnswer:
    """The row of this try (joined on try_id), or a new one. Transcribe and extraction may arrive in either order."""
    row = None
    if try_id:
        row = (db.query(EvalAnswer)
               .filter(EvalAnswer.session_id == session_id, EvalAnswer.try_id == try_id).first())
    if row is None:
        row = EvalAnswer(session_id=session_id, field=field, step=step, try_id=try_id or None, attempt_no=0)
        db.add(row)
    return row


def _save_row(db: Session, row_factory) -> None:
    """Run row_factory(db) and commit; if two writers created the same try at once, run it once more."""
    for _ in range(2):
        try:
            row_factory(db)
            db.commit()
            return
        except IntegrityError:
            db.rollback()


def log_transcribe(session_id: int, user_id: int, *, field: str, step: str, try_id: Optional[str],
                   engine: str, model: str, beam: Optional[int], transcript: str, seconds: Optional[float],
                   clip: Optional[bytes] = None, clip_type: str = "audio/webm") -> None:
    """
    Background task after /speech/transcribe: one try row with what Whisper heard, the model, beam and
    time; then the compressed clip (only when evaluation mode is on, the answer was heard by Whisper and
    the session is a voice session). Never raises.
    """
    db = SessionLocal()
    try:
        session = own_open_session(db, session_id, user_id)
        if session is None or step not in ATTEMPT_STEPS or field not in LOGGED_FIELDS | {"review", "confirm"}:
            return

        def write(d: Session):
            row = _try_row(d, session.id, try_id, field, step)
            row.field, row.step = field, step
            row.engine, row.model, row.beam = engine, model, beam
            row.transcript, row.whisper_seconds = transcript, seconds
            if step in ("confirm", "review"):
                row.yes_no = "yes" if re.fullmatch(r"(?:yes|yeah|yep|opo|oo|okay|ok|correct|right|sure)(?: po)?",
                                                   " ".join(_words(transcript))) else ("no" if transcript else "other")

        _save_row(db, write)
        # The session starts at the consent notice, before voice or typing is chosen: a Whisper call makes it voice.
        if engine == "whisper" and session.input_method == "manual":
            session.input_method = "voice"
            db.commit()
        if clip and engine == "whisper" and session.input_method == "voice" and recording_on(db):
            _save_clip(db, session.id, try_id, field, step, clip, clip_type)
    except Exception:
        db.rollback()
        logger.exception("Could not log an evaluation try (session %s)", session_id)
    finally:
        db.close()


def _save_clip(db: Session, session_id: int, try_id: Optional[str], field: str, step: str,
               clip: bytes, clip_type: str) -> None:
    if not storage_service.is_enabled():
        return
    extension = "wav" if "wav" in clip_type else ("ogg" if "ogg" in clip_type else "webm")
    path = f"s{session_id}/{field}-{step}-{try_id or uuid.uuid4().hex[:12]}.{extension}"
    try:
        ref = storage_service.upload_bytes(env_settings.SUPABASE_EVAL_BUCKET, path, clip, clip_type)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Evaluation clip not saved: %s", exc)
        return
    row = (db.query(EvalAnswer)
           .filter(EvalAnswer.session_id == session_id, EvalAnswer.try_id == try_id).first()) if try_id else None
    if row is None:
        logger.warning("Evaluation clip has no row to belong to; deleting it.")
        storage_service.delete_object(ref)
        return
    row.audio_ref = ref
    db.commit()


def log_extraction(session_id: int, user_id: int, *, field: str, step: str, try_id: Optional[str], spoken: bool,
                   value: Optional[str], ok: bool, transcript: Optional[str] = None) -> None:
    """Background task after /extraction/answer: what the system made of the try. Never raises."""
    db = SessionLocal()
    try:
        session = own_open_session(db, session_id, user_id)
        if session is None or step not in ATTEMPT_STEPS or field not in LOGGED_FIELDS:
            return

        def write(d: Session):
            existed = bool(try_id) and d.query(EvalAnswer).filter(
                EvalAnswer.session_id == session.id, EvalAnswer.try_id == try_id).first() is not None
            row = _try_row(d, session.id, try_id, field, step)
            if not existed:      # no Whisper call before it: heard by the browser, or typed
                row.field, row.step = field, step
                row.engine = "browser" if spoken else "typed"
                if spoken:
                    row.transcript = (transcript or "")[:500]
            elif row.step in ("confirm", "review"):
                row.step, row.yes_no = step, None     # a reply at "is that correct?" that was really a new answer
            row.extracted_value, row.extract_ok = value, bool(ok)

        _save_row(db, write)
    except Exception:
        db.rollback()
        logger.exception("Could not log an evaluation extraction (session %s)", session_id)
    finally:
        db.close()


def record_security_event(route: str, kind: str) -> None:
    """A failed login or a 401/403: time, route and kind only. Never raises."""
    db = SessionLocal()
    try:
        db.add(SecurityEvent(route=route[:120], kind=kind))
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("Could not record a security event", exc_info=True)
    finally:
        db.close()


# ─── Scoring ──────────────────────────────────────────────────────────────────

def _finished_in(row: Optional[EvalAnswer], numbers: Dict[int, int]) -> Optional[str]:
    if row is None:
        return None
    if row.step == "end_said":
        return "end_said"
    return f"{row.step}_{min(numbers[row.id], 3 if row.step == 'spoken' else 2)}"


def score_unit(session: EvalSession, field: str, rows: List[EvalAnswer], note: Optional[dict],
               final_value: Optional[str], expected: Optional[dict]) -> dict:
    """
    One field (or part) of one session: how it finished, the three correct flags and its WER counts.
    `final_value` is None for a part (parts are not stored apart): the part's own accepted value is its final.
    """
    note = note or {}
    voice_rows = [r for r in rows if r.step in VOICE_STEPS]
    counted = [r for r in voice_rows if r.engine in ("typed", "browser") or not is_filler(r.transcript)]
    numbers: Dict[int, int] = {}
    seen: Dict[str, int] = {}
    for r in counted:
        seen[r.step] = seen.get(r.step, 0) + 1
        numbers[r.id] = seen[r.step]
        r.attempt_no = numbers[r.id]
    accepted = next((r for r in reversed(counted) if r.extract_ok), None)
    if note.get("accepted_engine") == "skipped":
        accepted = None          # the read-back was refused and the question skipped: nothing was accepted
    if note.get("accepted_engine") == "typed":
        accepted = None          # finished by typing: an earlier voice answer that was refused earns nothing
    if final_value is None:
        final_value = accepted.extracted_value if accepted else ""
    final_norm = norm(field, final_value)
    accepted_norm = norm(field, accepted.extracted_value) if accepted else None
    note_engine = note.get("accepted_engine")

    if accepted is not None and accepted_norm == final_norm and (
            accepted.engine != "typed" or note_engine in (None, "typed")):
        finished = "end_typed" if accepted.engine == "typed" else _finished_in(accepted, numbers)
    elif note_engine == "typed" and final_norm == norm(field, note.get("accepted_value")):
        finished = "end_typed"                     # typed at the end step (or a picker tap)
    elif note_engine == "typed" and final_norm == "" and note.get("left_blank"):
        finished = "end_typed"                     # "leave it blank" at the end step: an answer of none
    elif final_norm == "" or (field == "gross_annual_income" and final_norm == "0"):
        finished = "blank"                         # nothing was said or typed: unanswered, a miss
    else:
        finished = "staff_typed"                   # the saved value is not what was heard: typed on review

    target = norm(field, expected["expected"]) if expected else final_norm
    answered = finished != "blank"
    correct_finished = bool(answered and final_norm == target)

    def voice_hit(row: EvalAnswer) -> bool:
        if target == "" and finished == "end_typed":
            return False         # a none that had to be typed was not accepted by voice
        return bool(row.extract_ok and row.engine != "typed" and norm(field, row.extracted_value) == target)

    first_spoken = next((r for r in counted if r.step == "spoken" and r.engine != "typed"), None)
    correct_first = bool(first_spoken and voice_hit(first_spoken))
    correct_ladder = any(voice_hit(r) for r in counted if r.step in ("spoken", "spelled"))
    if not answered:
        correct_first = correct_ladder = False         # left unanswered (even after a rejected "none"): a miss
    # A wrong or blank final cannot be a voice success in the non-scripted rounds, where the target is the final.
    if not expected and not correct_finished:
        correct_first = correct_ladder = False
    skipped = note_engine == "skipped" or bool(note.get("skipped"))
    if skipped:
        # Plan 009: a skip is always a miss for field extraction accuracy, even when the question was finished
        # later at the end step, and even when an earlier voice try matched but was refused at the read-back.
        correct_first = correct_ladder = False

    # WER: the first spoken try only, and only when Whisper heard it and the session read a card.
    wer_errors = wer_words = s_errors = d_errors = i_errors = None
    if expected and first_spoken is not None and first_spoken.engine == "whisper":
        s_errors, d_errors, i_errors, wer_words = wer_split(expected["read"], first_spoken.transcript or "")
        wer_errors = s_errors + d_errors + i_errors

    first_row = first_spoken or (counted[0] if counted else None)
    return {
        "finished_in": finished,
        "spoken_tries": sum(1 for r in counted if r.step == "spoken"),
        "spelled_tries": sum(1 for r in counted if r.step == "spelled"),
        "skipped": skipped,
        "correct_first": correct_first, "correct_ladder": correct_ladder, "correct_finished": correct_finished,
        "wer_errors": wer_errors, "wer_words": wer_words,
        "s_errors": s_errors, "d_errors": d_errors, "i_errors": i_errors,
        "model": first_row.model if first_row else None,
        "final_value": final_value, "expected_value": expected["expected"] if expected else None,
        "seconds": note.get("seconds"),
    }


def score_session(db: Session, session: EvalSession, record_values: Dict[str, str], notes: Dict[str, dict]) -> None:
    """
    Write the 'final' rows of a session (idempotent: earlier final rows are replaced).
    `record_values` = the saved CTC value of each of the 14 fields; `notes` = the browser's batch per field.
    """
    old_finals = db.query(EvalAnswer).filter(EvalAnswer.session_id == session.id, EvalAnswer.step == "final").all()
    hand_scores = {r.field: {c: getattr(r, c) for c in HAND_COLUMNS} for r in old_finals if r.hand_words is not None}
    db.query(EvalAnswer).filter(EvalAnswer.session_id == session.id, EvalAnswer.step == "final").delete()
    answers = card_answers(db, session)
    tries = (db.query(EvalAnswer).filter(EvalAnswer.session_id == session.id, EvalAnswer.step != "final")
             .order_by(EvalAnswer.id).all())
    by_field: Dict[str, List[EvalAnswer]] = {}
    for row in tries:
        by_field.setdefault(row.field, []).append(row)

    for field in FIELDS14:
        parts = PARTS.get(field)
        expected = expected_in(answers, field)
        if not parts:
            unit = score_unit(session, field, by_field.get(field, []), notes.get(field),
                              record_values.get(field, ""), expected)
        else:
            part_units = []
            for key in parts:
                part_expected = expected_in(answers, key)
                if answers is not None and part_expected is None:
                    continue                     # the card has no such step (a city has no town step)
                if answers is None and not by_field.get(key):
                    continue                     # a step that was never asked (a city needs no town step)
                unit = score_unit(session, key, by_field.get(key, []), notes.get(key), None, part_expected)
                part_units.append((key, unit))
                db.add(_final_row(session, key, unit))
            # The joined field: the saved value against the card; voice hits only if every part was a voice hit.
            whole = score_unit(session, field, [], notes.get(field), record_values.get(field, ""), expected)
            if part_units:
                whole["correct_first"] = whole["correct_finished"] and all(u["correct_first"] for _, u in part_units)
                whole["correct_ladder"] = whole["correct_finished"] and all(u["correct_ladder"] for _, u in part_units)
                order = ["spoken_1", "spoken_2", "spoken_3", "spelled_1", "spelled_2", "end_said", "end_typed",
                         "staff_typed", "blank"]
                whole["finished_in"] = max((u["finished_in"] for _, u in part_units), key=order.index)
                whole["spoken_tries"] = sum(u["spoken_tries"] for _, u in part_units)
                whole["spelled_tries"] = sum(u["spelled_tries"] for _, u in part_units)
                # The browser may note a skip on the joined field itself (the place of birth): keep it, and a skip
                # anywhere is a miss (Codex verify round 1).
                whole["skipped"] = whole["skipped"] or any(u["skipped"] for _, u in part_units)
                if whole["skipped"]:
                    whole["correct_first"] = whole["correct_ladder"] = False
                wer = [u for _, u in part_units if u["wer_words"] is not None]
                whole["wer_errors"] = sum(u["wer_errors"] for u in wer) if wer else None
                whole["wer_words"] = sum(u["wer_words"] for u in wer) if wer else None
                for key in ("s_errors", "d_errors", "i_errors"):
                    split = [u[key] for u in wer if u[key] is not None]
                    whole[key] = sum(split) if split else None
                whole["model"] = part_units[0][1]["model"]
            if part_units and not expected:
                # Voice credit only if what was heard still is in the saved value: staff who retyped the whole
                # address on the review page get no voice hit (Codex verify round 1).
                saved = set(norm(field, record_values.get(field, "")).split())
                heard = [set(norm(field, u["final_value"]).split()) for _, u in part_units]
                # Every heard part is still in the saved value, and the saved value holds nothing that was not heard
                # (an accepted blank purok changed to "Purok 7" is a staff edit).
                kept = all(words <= saved for words in heard) and saved <= set().union(*heard)
                if not kept:
                    whole["correct_first"] = whole["correct_ladder"] = False
                    if whole["finished_in"] != "blank":
                        whole["finished_in"] = "staff_typed"
            unit = whole
            if not expected:        # no card: nothing more to say about the joined text
                unit["expected_value"] = None
        db.add(_final_row(session, field, unit))
    db.flush()
    # A hand re-score survives scoring again (the late batch of an issued session re-scores it).
    finals = {r.field: r for r in db.query(EvalAnswer).filter(EvalAnswer.session_id == session.id,
                                                              EvalAnswer.step == "final")}
    for field, values in hand_scores.items():
        if field in finals and field not in PARTS:
            for column, value in values.items():
                setattr(finals[field], column, value)
    for parent in PARTS:
        if parent in finals:
            _roll_up_hand(finals, parent)
    session.scored = True
    db.flush()


def _final_row(session: EvalSession, field: str, unit: dict) -> EvalAnswer:
    return EvalAnswer(session_id=session.id, field=field, step="final", attempt_no=0, **unit)


# ─── Hand re-scoring from the saved audio (plan 009) ─────────────────────────

HAND_COLUMNS = ("hand_reference", "hand_hypothesis", "hand_s", "hand_d", "hand_i", "hand_words",
                "hand_scored_at", "hand_scored_by")


def _roll_up_hand(finals: Dict[str, EvalAnswer], parent: str) -> None:
    """The joined address / place of birth: its parts' hand values, or their Whisper values where not re-scored."""
    row = finals[parent]
    parts = [finals[k] for k in PARTS[parent] if k in finals]
    hand = [p for p in parts if p.hand_words is not None]
    for column in HAND_COLUMNS:
        setattr(row, column, None)
    if not hand:
        return
    totals = [0, 0, 0, 0]
    for p in parts:
        if p.hand_words is not None:
            values = (p.hand_s, p.hand_d, p.hand_i, p.hand_words)
        elif p.wer_words is not None:
            # A part without the split (plan 003 rows) counts all its errors as substitutions.
            values = (p.s_errors if p.s_errors is not None else p.wer_errors, p.d_errors, p.i_errors, p.wer_words)
        else:
            continue
        totals = [t + (v or 0) for t, v in zip(totals, values)]
    row.hand_s, row.hand_d, row.hand_i, row.hand_words = totals
    row.hand_scored_at = hand[-1].hand_scored_at
    row.hand_scored_by = hand[-1].hand_scored_by


def hand_score(db: Session, answer_id: int, reference: Optional[str], hypothesis: Optional[str],
               admin_id: Optional[int]) -> Optional[EvalAnswer]:
    """
    The admin listened to the clip: what the participant really said (reference) and what Whisper heard
    (hypothesis) are counted with the same text rule as the card. Both None clears the hand score.
    Only the result row of one question or one part that has a WER count takes a hand score; the joined
    address and place of birth add up their parts. Returns the row, or None when it cannot take one.
    """
    row = db.get(EvalAnswer, answer_id)
    if row is None or row.step != "final":
        return None
    finals = {r.field: r for r in db.query(EvalAnswer).filter(EvalAnswer.session_id == row.session_id,
                                                              EvalAnswer.step == "final")}
    if row.field in PARTS and any(k in finals for k in PARTS[row.field]):
        return None                               # re-score its parts instead
    if reference is None and hypothesis is None:
        for column in HAND_COLUMNS:
            setattr(row, column, None)
    elif row.wer_words is None:
        # Only a question that has a WER count (a card, and a first spoken try heard by Whisper) can be
        # re-scored: a typed answer or a "no card" session adds nothing to WER (Codex verify round 1).
        return None
    else:
        subs, dels, ins, words = wer_split(reference or "", hypothesis or "")
        row.hand_reference, row.hand_hypothesis = (reference or "")[:500], (hypothesis or "")[:500]
        row.hand_s, row.hand_d, row.hand_i, row.hand_words = subs, dels, ins, words
        row.hand_scored_at, row.hand_scored_by = datetime.now(timezone.utc), admin_id
    parent = PART_PARENT.get(row.field)
    if parent and parent in finals:
        _roll_up_hand(finals, parent)
    db.commit()
    return row


def finish_session(db: Session, session: EvalSession, applicant_values: Optional[Dict[str, str]],
                   batch: Optional[dict], issuance_id: Optional[int] = None) -> None:
    """
    Close a session. With the saved CTC values it is scored ("issued"); without, it is "abandoned".
    Called by /forms/issue and by the staff /finish route; safe to call again with the late batch.
    """
    batch = batch or {}
    _release_unused_card(db, session)       # before scoring: a session with no Whisper try reads no card
    notes = {n["field"]: n for n in batch.get("notes", []) if isinstance(n, dict) and isinstance(n.get("field"), str)}
    if applicant_values is not None:
        session.outcome = "issued"
        session.issuance_id = issuance_id or session.issuance_id
        if session.ended_at is None:
            session.ended_at = datetime.now(timezone.utc)
        started = session.started_at
        if started is not None:
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            session.app_seconds = round((session.ended_at - started).total_seconds(), 1) \
                if session.ended_at.tzinfo else None
        score_session(db, session, applicant_values, notes)
    elif session.outcome == "open":
        session.outcome = "abandoned"
        session.ended_at = datetime.now(timezone.utc)
    if "errors" in batch:
        session.error_count = int(batch["errors"] or 0)
    if batch.get("qr_seconds") is not None:
        session.qr_seconds = float(batch["qr_seconds"])
    db.commit()


# ─── Statistics (aggregated: a constant number of queries) ───────────────────

def counted_clause():
    """
    Plan 009: a session counts toward the evaluation when it was issued (finished), the admin's "counts toward
    evaluation" box was ticked, it read a generated card, and it was not a Practice round. Every session was
    started with evaluation mode on (an off switch logs nothing). One session = one participant code.
    """
    return and_(EvalSession.outcome == "issued", EvalSession.counts.is_(True), EvalSession.card_id.isnot(None),
                EvalSession.round_type != "practice")


def _filtered_sessions(db: Session, filters: dict):
    query = db.query(EvalSession.id)
    if (filters.get("view") or "counted") == "counted":
        query = query.filter(counted_clause())
    if filters.get("round_type"):
        query = query.filter(EvalSession.round_type == filters["round_type"])
    if filters.get("condition"):
        query = query.filter(EvalSession.round_condition == filters["condition"])
    if filters.get("input_method"):
        query = query.filter(EvalSession.input_method == filters["input_method"])
    if filters.get("date_from"):
        query = query.filter(EvalSession.started_at >= filters["date_from"])
    if filters.get("date_to"):
        query = query.filter(EvalSession.started_at <= filters["date_to"])
    return query


def _rate(hits, n) -> Optional[float]:
    return round(100.0 * (hits or 0) / n, 1) if n else None


GROUP_KEYS = ("n", "first", "ladder", "finished", "wer_errors", "wer_words", "s", "d", "i", "hand_errors",
              "hand_words", "hand_rows", "skipped", "typed", "spelled", "tries", "seconds", "timed")


def _group(rows: Iterable[dict]) -> dict:
    total = dict.fromkeys(GROUP_KEYS, 0)
    for r in rows:
        for key in total:
            total[key] += r[key] or 0
    return {
        "answers": total["n"],
        "field_extraction_pct": _rate(total["ladder"], total["n"]),     # the headline: voice, retries counted
        "first_try_pct": _rate(total["first"], total["n"]),
        "after_ladder_pct": _rate(total["ladder"], total["n"]),
        "finished_pct": _rate(total["finished"], total["n"]),
        "wer_pct": _rate(total["wer_errors"], total["wer_words"]),
        "wer_errors": total["wer_errors"],
        "wer_words": total["wer_words"],
        "substitutions": total["s"], "deletions": total["d"], "insertions": total["i"],
        "hand_wer_pct": _rate(total["hand_errors"], total["hand_words"]),
        "hand_errors": total["hand_errors"], "hand_words": total["hand_words"], "hand_scored": total["hand_rows"],
        "skipped": total["skipped"],
        "skip_pct": _rate(total["skipped"], total["n"]),
        "typed_pct": _rate(total["typed"], total["n"]),
        "spelled_pct": _rate(total["spelled"], total["n"]),
        "avg_spoken_tries": round(total["tries"] / total["n"], 2) if total["n"] else None,
        "avg_seconds": round(total["seconds"] / total["timed"], 1) if total["timed"] else None,
    }


def stats(db: Session, filters: dict) -> dict:
    """
    Plan 009: WER = (S + D + I) / N pooled over every scored first spoken Whisper try of every session in view
    (total errors / total reference words, never an average of percentages). Field extraction accuracy = voice
    hits counting the ladder's retries over all answers, a skip or a blank being a miss.
    """
    ids = _filtered_sessions(db, filters).subquery()
    in_view = EvalSession.id.in_(db.query(ids.c.id))
    session_rows = (db.query(EvalSession.outcome, func.count(EvalSession.id), func.sum(EvalSession.error_count),
                             func.avg(EvalSession.app_seconds), func.avg(EvalSession.qr_seconds))
                    .filter(in_view).group_by(EvalSession.outcome).all())
    sessions = {outcome: n for outcome, n, _e, _a, _q in session_rows}
    errors = sum(int(e or 0) for _o, _n, e, _a, _q in session_rows)
    issued = next(((a, q) for o, _n, _e, a, q in session_rows if o == "issued"), (None, None))
    counted = db.query(func.count(EvalSession.id)).filter(counted_clause()).scalar() or 0

    def b(column):
        return func.sum(case((column.is_(True), 1), else_=0))

    final = EvalAnswer
    hand_on = and_(final.hand_words.isnot(None), final.wer_words.isnot(None))   # only where WER is counted
    not_skipped = final.skipped.isnot(True)
    field_rows = (db.query(
        final.field, func.count(final.id), b(final.correct_first), b(final.correct_ladder),
        b(final.correct_finished), func.sum(final.wer_errors), func.sum(final.wer_words),
        func.sum(final.s_errors), func.sum(final.d_errors), func.sum(final.i_errors),
        # The hand value where the admin re-scored the clip, else Whisper's own count.
        func.sum(case((hand_on, final.hand_s + final.hand_d + final.hand_i), else_=final.wer_errors)),
        func.sum(case((hand_on, final.hand_words), else_=final.wer_words)),
        func.sum(case((hand_on, 1), else_=0)),
        b(final.skipped),
        func.sum(case((final.finished_in.in_(("end_typed", "staff_typed")), 1), else_=0)),
        func.sum(case((final.spelled_tries > 0, 1), else_=0)), func.sum(final.spoken_tries),
        # Seconds per question leave skipped questions out: their clock runs on to the end step.
        func.sum(case((not_skipped, final.seconds), else_=None)),
        func.sum(case((and_(not_skipped, final.seconds.isnot(None)), 1), else_=0)))
        .join(EvalSession, EvalSession.id == final.session_id)
        .filter(final.step == "final", EvalSession.outcome == "issued", in_view)
        .group_by(final.field).all())
    by_field = {row[0]: dict(zip(GROUP_KEYS, row[1:])) for row in field_rows}

    def pick(keys):
        return _group(by_field[k] for k in keys if k in by_field)

    wer_people = (db.query(func.count(distinct(final.session_id)))
                  .join(EvalSession, EvalSession.id == final.session_id)
                  .filter(final.step == "final", final.field.in_(FIELDS14), EvalSession.outcome == "issued",
                          in_view, (final.wer_words > 0) | (final.hand_words > 0)).scalar() or 0)
    model_rows = (db.query(final.model, func.count(final.id), b(final.correct_first))
                  .join(EvalSession, EvalSession.id == final.session_id)
                  .filter(final.step == "final", final.field.in_(FIELDS14), EvalSession.outcome == "issued",
                          in_view, final.model.isnot(None))
                  .group_by(final.model).all())
    whisper_avg = (db.query(func.avg(EvalAnswer.whisper_seconds), func.count(EvalAnswer.id))
                   .filter(EvalAnswer.step == "spoken", EvalAnswer.session_id.in_(db.query(ids.c.id))).one())
    skipped_time = (db.query(func.count(final.id), func.avg(final.seconds))
                    .join(EvalSession, EvalSession.id == final.session_id)
                    .filter(final.step == "final", final.field.in_(FIELDS14), final.skipped.is_(True),
                            EvalSession.outcome == "issued", in_view).one())
    security = dict(db.query(SecurityEvent.kind, func.count(SecurityEvent.id)).group_by(SecurityEvent.kind).all())
    overall = pick(FIELDS14)
    return {
        "view": filters.get("view") or "counted",
        "progress": {"counted": counted, "target": COUNTED_TARGET},
        "targets": {"wer_pct": WER_TARGET_PCT, "field_extraction_pct": EXTRACTION_TARGET_PCT},
        "sessions": {"total": sum(sessions.values()), **{k: sessions.get(k, 0) for k in ("issued", "abandoned", "open")}},
        "wer": {
            "pct": overall["wer_pct"], "errors": overall["wer_errors"], "words": overall["wer_words"],
            "substitutions": overall["substitutions"], "deletions": overall["deletions"],
            "insertions": overall["insertions"], "participants": wer_people,
            "hand_pct": overall["hand_wer_pct"], "hand_errors": overall["hand_errors"],
            "hand_words": overall["hand_words"], "hand_scored": overall["hand_scored"],
        },
        "overall": overall,
        "fields": {f: pick((f,)) for f in FIELDS14 if f in by_field},
        "place_of_birth_steps": {k: pick((k,)) for k in PARTS["place_of_birth"] if k in by_field},
        "address_parts": {k: pick((k,)) for k in PARTS["address"] if k in by_field},
        "models": [{"model": m, "answers": n, "first_try_pct": _rate(h, n)} for m, n, h in model_rows],
        "timing": {
            "avg_seconds_per_question": overall["avg_seconds"],
            "skipped_questions": skipped_time[0] or 0,
            "avg_seconds_skipped": round(skipped_time[1], 1) if skipped_time[1] is not None else None,
            "avg_seconds_per_application": round(issued[0], 1) if issued[0] is not None else None,
            "avg_seconds_qr_to_issued": round(issued[1], 1) if issued[1] is not None else None,
            "avg_whisper_seconds": round(whisper_avg[0], 2) if whisper_avg[0] is not None else None,
        },
        "errors": errors,
        "security": {"login_failed": security.get("login_failed", 0), "unauthorized": security.get("unauthorized", 0),
                     "forbidden": security.get("forbidden", 0)},
    }


# ─── Delete evaluation audio now (admin button; never automatic) ─────────────

def _holds_text():
    """A row that still holds a clip or real speech as text (plan 009 adds the hand re-score's two texts)."""
    return ((EvalAnswer.audio_ref.isnot(None)) | (EvalAnswer.transcript.isnot(None)) |
            (EvalAnswer.extracted_value.isnot(None)) | (EvalAnswer.final_value.isnot(None)) |
            (EvalAnswer.expected_value.isnot(None)) | (EvalAnswer.hand_reference.isnot(None)) |
            (EvalAnswer.hand_hypothesis.isnot(None)))


def pending_delete_count(db: Session) -> int:
    return (db.query(func.count(EvalAnswer.id))
            .filter(_holds_text()).scalar() or 0)


def delete_batch(db: Session, size: int = 100) -> dict:
    """
    One batch of the "Delete evaluation audio now" button: removes up to `size` rows' clips and text
    (transcripts, extracted, final and expected values). The numbers (flags, WER counts, times) stay.
    """
    size = max(1, min(int(size), 500))
    rows = (db.query(EvalAnswer)
            .filter(_holds_text())
            .order_by(EvalAnswer.id).limit(size).all())
    clips = failed = 0
    for row in rows:
        if row.audio_ref:
            try:
                storage_service.delete_object_strict(row.audio_ref)
                clips += 1
            except Exception as exc:  # noqa: BLE001
                # Storage did not confirm: keep the reference (and the row's text) so the next click retries it.
                failed += 1
                logger.warning("Evaluation clip not deleted: %s", exc)
                continue
        row.audio_ref = row.transcript = row.extracted_value = row.final_value = row.expected_value = None
        row.hand_reference = row.hand_hypothesis = None      # the hand counts stay
    db.commit()
    return {"rows": len(rows) - failed, "clips": clips, "failed": failed, "remaining": pending_delete_count(db)}


# ─── Called by /forms/issue ──────────────────────────────────────────────────

# Interview question key (what the web calls it) -> the ApplicantRecord column saved at issue.
FIELD_TO_RECORD = {
    "surname": "surname",
    "given_name": "given_name",
    "middle_name": "middle_name",
    "address": "address",
    "date": "date_of_birth",
    "sex": "sex",
    "civil_status": "civil_status",
    "citizenship": "citizenship",
    "icr_no": "icr_number",
    "place_of_birth": "place_of_birth",
    "height": "height_cm",
    "weight": "weight_kg",
    "occupation": "occupation",
    "gross_annual_income": "gross_annual_income",
}


def _canon(field: str, value) -> str:
    """One text form for a value, so a voice value and the saved value can be compared."""
    value = getattr(value, "value", value)         # enums (sex, civil status)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        text = str(int(number)) if number.is_integer() else repr(number)   # 158.0 and 158 are the same value
    else:
        text = str(value).strip()
    # A blank height / weight is saved as nothing, though the form sends 0.
    if field in ("height", "weight") and text == "0":
        return ""
    return text


def _final_value(field: str, applicant) -> str:
    attr = FIELD_TO_RECORD[field]
    value = getattr(applicant, attr, None)
    if attr == "date_of_birth" and value is not None:
        value = value.isoformat()
    return _canon(field, value)

def record_values_for(db: Session, session: EvalSession) -> Dict[str, str]:
    """The saved CTC value of each of the 14 fields."""
    from app.models.issuance import CTCIssuance
    issuance = db.get(CTCIssuance, session.issuance_id) if session.issuance_id else None
    applicant = issuance.applicant if issuance else None
    if applicant is None:
        return {}
    return {field: _final_value(field, applicant) for field in FIELD_TO_RECORD}


def link_issuance(db: Session, session_id: Optional[int], user, issuance, batch: Optional[dict]) -> None:
    """After the CTC is saved: close and score the session. Never raises (the CTC must reach the applicant)."""
    if not session_id:
        return
    try:
        session = own_open_session(db, session_id, user.id)
        if session is None or session.outcome == "issued":
            return
        session.issuance_id = issuance.id
        values = record_values_for(db, session)
        finish_session(db, session, values, batch, issuance_id=issuance.id)
    except Exception:
        db.rollback()
        logger.exception("Could not score evaluation session %s", session_id)
