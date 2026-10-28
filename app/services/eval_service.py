"""
app/services/eval_service.py
Evaluation mode (plan 003): the settings, the session log, scoring (three accuracy numbers, WER),
the aggregated statistics and the "delete evaluation audio now" batches.

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

from rapidfuzz.distance import Levenshtein
from sqlalchemy import case, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings as env_settings
from app.core.database import SessionLocal
from app.models.app_settings import AppSettings
from app.models.eval import (
    ATTEMPT_STEPS, EvalAnswer, EvalSession, SecurityEvent,
)
from app.services import storage_service

logger = logging.getLogger(__name__)

CARDS_PATH = Path(__file__).resolve().parents[1] / "eval" / "reading_cards.json"

# The 14 questions of the paper; "hard 5" = the fields of the earlier 10-recording test (15.9% WER).
HARD5 = ("surname", "given_name", "middle_name", "address", "place_of_birth")
EASY9 = ("date", "sex", "civil_status", "citizenship", "icr_no", "height", "weight", "occupation",
         "gross_annual_income")
FIELDS14 = HARD5 + EASY9
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
        "practice_log_conflict": bool(env_settings.PRACTICE_LOG_ENABLED),
    }


def recording_on(db: Session) -> bool:
    """Evaluation mode on, and the temporary practice log off (the two logs never run together)."""
    if env_settings.PRACTICE_LOG_ENABLED:
        return False
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
        db.add(row)
        db.flush()
    changed = False
    mapping = {
        "round_type": "eval_round_type", "round_condition": "eval_round_condition",
        "card_set": "eval_round_card_set",
    }
    if "evaluation_mode" in values and values["evaluation_mode"] is not None:
        row.evaluation_mode = bool(values["evaluation_mode"])
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


def card_for(card_set: Optional[str], participant_no: int) -> Tuple[int, dict]:
    """Cards go out in order, wrapping after the last one in the set (P001 -> first card, ...)."""
    data = load_cards()
    numbers = data["sets"].get(card_set or "") or data["sets"]["all"]
    card_no = numbers[(participant_no - 1) % len(numbers)]
    card = next(c for c in data["cards"] if c["no"] == card_no)
    return card_no, card


def _expected_for(session: EvalSession, field: str) -> Optional[dict]:
    """The card's {expected, read} for a field or part in a scripted session, else None."""
    if session.round_type != "scripted" or not session.card_no:
        return None
    card = next((c for c in load_cards()["cards"] if c["no"] == session.card_no), None)
    if card is None:
        return None
    if field in card["answers"]:
        return card["answers"][field]
    parent = PART_PARENT.get(field)
    return card["answers"][parent]["parts"].get(field) if parent else None


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


def wer_counts(reference: str, heard: str) -> Tuple[int, int]:
    """(word errors, reference words): word edit distance over lower-cased, punctuation-free words."""
    ref, hyp = _words(reference), _words(heard)
    return Levenshtein.distance(ref, hyp), len(ref)


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
        )
        if cfg["round_type"] == "scripted":
            session.card_no, _card = card_for(cfg["card_set"], number)
        db.add(session)
        try:
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
        return bool(row.extract_ok and row.engine != "typed" and norm(field, row.extracted_value) == target)

    first_spoken = next((r for r in counted if r.step == "spoken" and r.engine != "typed"), None)
    correct_first = bool(first_spoken and voice_hit(first_spoken))
    correct_ladder = any(voice_hit(r) for r in counted if r.step in ("spoken", "spelled"))
    if not answered:
        correct_first = correct_ladder = False         # left unanswered (even after a rejected "none"): a miss
    # A wrong or blank final cannot be a voice success in the non-scripted rounds, where the target is the final.
    if not expected and not correct_finished:
        correct_first = correct_ladder = False

    wer_errors = wer_words = None
    if expected and first_spoken is not None and first_spoken.engine == "whisper":
        wer_errors, wer_words = wer_counts(expected["read"], first_spoken.transcript or "")

    first_row = first_spoken or (counted[0] if counted else None)
    return {
        "finished_in": finished,
        "spoken_tries": sum(1 for r in counted if r.step == "spoken"),
        "spelled_tries": sum(1 for r in counted if r.step == "spelled"),
        "skipped": note_engine == "skipped" or bool(note.get("skipped")),
        "correct_first": correct_first, "correct_ladder": correct_ladder, "correct_finished": correct_finished,
        "wer_errors": wer_errors, "wer_words": wer_words,
        "model": first_row.model if first_row else None,
        "final_value": final_value, "expected_value": expected["expected"] if expected else None,
        "seconds": note.get("seconds"),
    }


def score_session(db: Session, session: EvalSession, record_values: Dict[str, str], notes: Dict[str, dict]) -> None:
    """
    Write the 'final' rows of a session (idempotent: earlier final rows are replaced).
    `record_values` = the saved CTC value of each of the 14 fields; `notes` = the browser's batch per field.
    """
    db.query(EvalAnswer).filter(EvalAnswer.session_id == session.id, EvalAnswer.step == "final").delete()
    tries = (db.query(EvalAnswer).filter(EvalAnswer.session_id == session.id, EvalAnswer.step != "final")
             .order_by(EvalAnswer.id).all())
    by_field: Dict[str, List[EvalAnswer]] = {}
    for row in tries:
        by_field.setdefault(row.field, []).append(row)

    for field in FIELDS14:
        parts = PARTS.get(field)
        expected = _expected_for(session, field)
        if not parts:
            unit = score_unit(session, field, by_field.get(field, []), notes.get(field),
                              record_values.get(field, ""), expected)
        else:
            part_units = []
            for key in parts:
                part_expected = _expected_for(session, key)
                if session.round_type == "scripted" and part_expected is None:
                    continue                     # the card has no such step (a city has no town step)
                if session.round_type != "scripted" and not by_field.get(key):
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
                whole["skipped"] = any(u["skipped"] for _, u in part_units)
                wer = [u for _, u in part_units if u["wer_words"] is not None]
                whole["wer_errors"] = sum(u["wer_errors"] for u in wer) if wer else None
                whole["wer_words"] = sum(u["wer_words"] for u in wer) if wer else None
                whole["model"] = part_units[0][1]["model"]
            if part_units and not expected:
                # Voice credit only if what was heard still is in the saved value: staff who retyped the whole
                # address on the review page get no voice hit (Codex verify round 1).
                saved = set(norm(field, record_values.get(field, "")).split())
                kept = all(set(norm(field, u["final_value"]).split()) <= saved for _, u in part_units)
                if not kept:
                    whole["correct_first"] = whole["correct_ladder"] = False
                    if whole["finished_in"] != "blank":
                        whole["finished_in"] = "staff_typed"
            unit = whole
            if not expected:        # not a scripted round: nothing more to say about the joined text
                unit["expected_value"] = None
        db.add(_final_row(session, field, unit))
    session.scored = True
    db.flush()


def _final_row(session: EvalSession, field: str, unit: dict) -> EvalAnswer:
    return EvalAnswer(session_id=session.id, field=field, step="final", attempt_no=0, **unit)


def finish_session(db: Session, session: EvalSession, applicant_values: Optional[Dict[str, str]],
                   batch: Optional[dict], issuance_id: Optional[int] = None) -> None:
    """
    Close a session. With the saved CTC values it is scored ("issued"); without, it is "abandoned".
    Called by /forms/issue and by the staff /finish route; safe to call again with the late batch.
    """
    batch = batch or {}
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

def _filtered_sessions(db: Session, filters: dict):
    query = db.query(EvalSession.id)
    if filters.get("round_type"):
        query = query.filter(EvalSession.round_type == filters["round_type"])
    else:   # practice and live are kept out of the study numbers unless asked for
        query = query.filter(EvalSession.round_type.in_(("scripted", "uat")))
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


def _group(rows: Iterable[dict]) -> dict:
    total = {"n": 0, "first": 0, "ladder": 0, "finished": 0, "wer_errors": 0, "wer_words": 0, "skipped": 0,
             "typed": 0, "spelled": 0, "tries": 0}
    for r in rows:
        for key in total:
            total[key] += r[key] or 0
    return {
        "answers": total["n"],
        "first_try_pct": _rate(total["first"], total["n"]),
        "after_ladder_pct": _rate(total["ladder"], total["n"]),
        "finished_pct": _rate(total["finished"], total["n"]),
        "wer_pct": _rate(total["wer_errors"], total["wer_words"]),
        "wer_words": total["wer_words"],
        "skip_pct": _rate(total["skipped"], total["n"]),
        "typed_pct": _rate(total["typed"], total["n"]),
        "spelled_pct": _rate(total["spelled"], total["n"]),
        "avg_spoken_tries": round(total["tries"] / total["n"], 2) if total["n"] else None,
    }


def stats(db: Session, filters: dict) -> dict:
    ids = _filtered_sessions(db, filters).subquery()
    session_rows = (db.query(EvalSession.outcome, func.count(EvalSession.id), func.sum(EvalSession.error_count),
                             func.avg(EvalSession.app_seconds), func.avg(EvalSession.qr_seconds))
                    .filter(EvalSession.id.in_(db.query(ids.c.id)))
                    .group_by(EvalSession.outcome).all())
    sessions = {outcome: n for outcome, n, _e, _a, _q in session_rows}
    errors = sum(int(e or 0) for _o, _n, e, _a, _q in session_rows)
    issued = next(((a, q) for o, _n, _e, a, q in session_rows if o == "issued"), (None, None))

    def b(column):
        return func.sum(case((column.is_(True), 1), else_=0))

    final = EvalAnswer
    field_rows = (db.query(
        final.field, func.count(final.id), b(final.correct_first), b(final.correct_ladder),
        b(final.correct_finished), func.sum(final.wer_errors), func.sum(final.wer_words), b(final.skipped),
        func.sum(case((final.finished_in.in_(("end_typed", "staff_typed")), 1), else_=0)),
        func.sum(case((final.spelled_tries > 0, 1), else_=0)), func.sum(final.spoken_tries))
        .join(EvalSession, EvalSession.id == final.session_id)
        .filter(final.step == "final", EvalSession.outcome == "issued",
                EvalSession.id.in_(db.query(ids.c.id)))
        .group_by(final.field).all())
    by_field = {}
    for f, n, first, ladder, fin, we, ww, skipped, typed, spelled, tries in field_rows:
        by_field[f] = {"n": n, "first": first, "ladder": ladder, "finished": fin, "wer_errors": we,
                       "wer_words": ww, "skipped": skipped, "typed": typed, "spelled": spelled, "tries": tries}

    def pick(keys):
        return _group(by_field[k] for k in keys if k in by_field)

    model_rows = (db.query(final.model, func.count(final.id), b(final.correct_first))
                  .join(EvalSession, EvalSession.id == final.session_id)
                  .filter(final.step == "final", final.field.in_(FIELDS14), EvalSession.outcome == "issued",
                          EvalSession.id.in_(db.query(ids.c.id)), final.model.isnot(None))
                  .group_by(final.model).all())
    whisper_avg = (db.query(func.avg(EvalAnswer.whisper_seconds), func.count(EvalAnswer.id))
                   .filter(EvalAnswer.step == "spoken", EvalAnswer.session_id.in_(db.query(ids.c.id))).one())
    seconds_avg = (db.query(func.avg(final.seconds))
                   .filter(final.step == "final", final.field.in_(FIELDS14),
                           final.session_id.in_(db.query(ids.c.id))).scalar())
    security = dict(db.query(SecurityEvent.kind, func.count(SecurityEvent.id)).group_by(SecurityEvent.kind).all())
    return {
        "sessions": {"total": sum(sessions.values()), **{k: sessions.get(k, 0) for k in ("issued", "abandoned", "open")}},
        "overall": pick(FIELDS14),
        "hard5": pick(HARD5),
        "easy9": pick(EASY9),
        "fields": {f: pick((f,)) for f in FIELDS14 if f in by_field},
        "place_of_birth_steps": {k: pick((k,)) for k in PARTS["place_of_birth"] if k in by_field},
        "address_parts": {k: pick((k,)) for k in PARTS["address"] if k in by_field},
        "models": [{"model": m, "answers": n, "first_try_pct": _rate(h, n)} for m, n, h in model_rows],
        "timing": {
            "avg_seconds_per_question": round(seconds_avg, 1) if seconds_avg is not None else None,
            "avg_seconds_per_application": round(issued[0], 1) if issued[0] is not None else None,
            "avg_seconds_qr_to_issued": round(issued[1], 1) if issued[1] is not None else None,
            "avg_whisper_seconds": round(whisper_avg[0], 2) if whisper_avg[0] is not None else None,
        },
        "errors": errors,
        "security": {"login_failed": security.get("login_failed", 0), "unauthorized": security.get("unauthorized", 0),
                     "forbidden": security.get("forbidden", 0)},
    }


# ─── Delete evaluation audio now (admin button; never automatic) ─────────────

def pending_delete_count(db: Session) -> int:
    return (db.query(func.count(EvalAnswer.id))
            .filter((EvalAnswer.audio_ref.isnot(None)) | (EvalAnswer.transcript.isnot(None)) |
                    (EvalAnswer.extracted_value.isnot(None)) | (EvalAnswer.final_value.isnot(None)) |
                    (EvalAnswer.expected_value.isnot(None))).scalar() or 0)


def delete_batch(db: Session, size: int = 100) -> dict:
    """
    One batch of the "Delete evaluation audio now" button: removes up to `size` rows' clips and text
    (transcripts, extracted, final and expected values). The numbers (flags, WER counts, times) stay.
    """
    size = max(1, min(int(size), 500))
    rows = (db.query(EvalAnswer)
            .filter((EvalAnswer.audio_ref.isnot(None)) | (EvalAnswer.transcript.isnot(None)) |
                    (EvalAnswer.extracted_value.isnot(None)) | (EvalAnswer.final_value.isnot(None)) |
                    (EvalAnswer.expected_value.isnot(None)))
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
    db.commit()
    return {"rows": len(rows) - failed, "clips": clips, "failed": failed, "remaining": pending_delete_count(db)}


# ─── Called by /forms/issue ──────────────────────────────────────────────────

def record_values_for(db: Session, session: EvalSession) -> Dict[str, str]:
    """The saved CTC value of each of the 14 fields (same forms the practice log compares)."""
    from app.api.routes.practice_log import FIELD_TO_RECORD, _final_value
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
