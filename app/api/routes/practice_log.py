"""
app/api/routes/practice_log.py
TEMPORARY practice session log (plan 001.7; plan 003 replaces it).

POST /api/practice-log/sessions                  staff: start a session ("I agree, start")
POST /api/practice-log/sessions/{id}/answers     staff: one note per finished question
POST /api/practice-log/sessions/{id}/finish      staff: the page was left ("abandoned")
GET  /api/practice-log/status                    admin: on/off, mode, build
PUT  /api/practice-log/mode                      admin: new sessions count as test or real
GET  /api/practice-log/sessions                  admin: list sessions
GET  /api/practice-log/sessions/{id}             admin: one session with its answers

Staff routes are write-only: they return ids and ok, never anything that was logged.
Off (PRACTICE_LOG_ENABLED=False, the default) writes nothing and answers {enabled: false}.
A logging failure never blocks an interview or issuing (same rule as log_record_access).
"""
import logging
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import require_admin, require_staff
from app.models.applicant import ApplicantRecord
from app.models.issuance import CTCIssuance
from app.models.practice_log import (
    PracticeLogAnswer, PracticeLogSession, PracticeLogSettings,
)
from app.models.staff_user import StaffUser

logger = logging.getLogger(__name__)
router = APIRouter()

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
# The address is asked as 3 parts, then joined into the one `address` row.
ADDRESS_PART_FIELDS = ("address_town", "address_barangay", "address_purok")
LOGGED_FIELDS = frozenset(FIELD_TO_RECORD) | frozenset(ADDRESS_PART_FIELDS)
NOT_VOICE = ("typed", "spelled")


# ─── Helpers ──────────────────────────────────────────────────────────────────

@lru_cache()
def backend_commit() -> str:
    """Short git commit this backend runs from, read once; "unknown" outside a git checkout."""
    try:
        repo = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo,
            capture_output=True, text=True, timeout=3,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


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


def _final_value(field: str, applicant: ApplicantRecord) -> str:
    attr = FIELD_TO_RECORD[field]
    value = getattr(applicant, attr, None)
    if attr == "date_of_birth" and value is not None:
        value = value.isoformat()
    return _canon(field, value)


def _apply_final(row: PracticeLogAnswer, applicant: Optional[ApplicantRecord]) -> None:
    """
    Fill final_value and changed_after_voice. Compared in the SAVED form (the birthday as
    the ISO date, the address as the joined string), never the spoken one.
    - an interview question: final = what the record holds;
    - an address part: the three parts are not stored apart, so final = the accepted part;
    - the joined address row: changed only when the saved address differs from the joined one
      (a typed correction of a part shows on that part's own row).
    """
    first = _canon(row.field, row.first_value)
    if row.field in FIELD_TO_RECORD:
        if applicant is None:
            return
        final = _final_value(row.field, applicant)
    else:
        final = _canon(row.field, row.accepted_value)
    row.final_value = final
    if row.field == "address":
        row.changed_after_voice = final != first
    else:
        row.changed_after_voice = row.accepted_engine in NOT_VOICE or final != first


def _current_mode(db: Session) -> str:
    row = db.get(PracticeLogSettings, 1)
    return row.mode if row else "test"


def _issued_applicant(db: Session, session: PracticeLogSession) -> Optional[ApplicantRecord]:
    if session.outcome != "issued" or session.issuance_id is None:
        return None
    issuance = db.get(CTCIssuance, session.issuance_id)
    return issuance.applicant if issuance else None


def _locked_session_query(db: Session, session_id: int):
    """The session row, locked until commit (Postgres; SQLite has no row locks and ignores it)."""
    return db.query(PracticeLogSession).filter(PracticeLogSession.id == session_id).with_for_update()


def _own_session(db: Session, session_id: int, user: StaffUser) -> PracticeLogSession:
    """
    A staff account can write only to its own sessions. The row is locked, so a note, a finish and
    /forms/issue on the same session run one after another: notes are sent without blocking and can
    overlap the issue call.
    """
    session = _locked_session_query(db, session_id).populate_existing().first()
    if session is None or session.staff_user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Practice session not found.")
    return session


def link_issuance(db: Session, session_id: Optional[int], user: StaffUser, issuance: CTCIssuance) -> None:
    """
    Called by POST /api/forms/issue after the CTC is saved: ends the session as "issued",
    links the record and fills every logged answer's final value and changed-after-voice.
    Never raises: the CTC is already saved and must reach the applicant.
    """
    if not settings.PRACTICE_LOG_ENABLED or not session_id:
        return
    try:
        session = _locked_session_query(db, session_id).populate_existing().first()
        if session is None or session.staff_user_id != user.id or session.outcome == "issued":
            db.rollback()                                    # releases the lock
            return
        session.outcome = "issued"
        session.ended_at = datetime.now(timezone.utc)
        session.issuance_id = issuance.id
        applicant = issuance.applicant
        for row in db.query(PracticeLogAnswer).filter(PracticeLogAnswer.session_id == session.id):
            _apply_final(row, applicant)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Could not link practice session %s to the issued CTC", session_id)


# ─── Staff: write-only ────────────────────────────────────────────────────────

class StartBody(BaseModel):
    input_method: Literal["voice", "qr", "manual"] = "voice"


class StartResponse(BaseModel):
    enabled: bool
    session_id: Optional[int] = None


class AttemptNote(BaseModel):
    heard: str = Field("", max_length=500)                       # what the engine heard, or the typed text
    engine: Literal["whisper", "browser", "typed", "spelled"]
    seconds: Optional[float] = Field(None, ge=0, le=600)         # Whisper round trip
    parsed: Optional[str] = Field(None, max_length=500)          # what /answer made of it
    ok: Optional[bool] = None
    reason: Optional[str] = Field(None, max_length=60)
    # Plan 002.2: "end_step" = asked again at the end (it had been skipped). The web sends a question's tries
    # cumulatively, so the note that finishes a skipped question still holds its interview tries.
    stage: Optional[Literal["interview", "end_step"]] = None


class AnswerNote(BaseModel):
    field: str = Field(..., min_length=1, max_length=40)
    attempts: int = Field(1, ge=0, le=50)
    attempt_details: List[AttemptNote] = Field(default_factory=list, max_length=30)
    first_value: Union[str, float, int, None] = None             # first voice-parsed value, SAVED form
    accepted_value: Optional[str] = Field(None, max_length=500)
    accepted_engine: Literal["whisper", "browser", "typed", "spelled", "skipped"]
    seconds: Optional[float] = Field(None, ge=0, le=3600)        # question asked -> answer accepted


class OkResponse(BaseModel):
    ok: bool
    enabled: bool = True


@router.post("/sessions", response_model=StartResponse)
def start_session(
    body: StartBody,
    db: Session = Depends(get_db),
    current_user: StaffUser = Depends(require_staff),
):
    if not settings.PRACTICE_LOG_ENABLED:
        return StartResponse(enabled=False)
    try:
        session = PracticeLogSession(
            build_label=settings.PRACTICE_LOG_BUILD[:60],
            backend_commit=backend_commit(),
            mode=_current_mode(db),
            staff_user_id=current_user.id,
            input_method=body.input_method,
            outcome="open",
        )
        db.add(session)
        db.commit()
        return StartResponse(enabled=True, session_id=session.id)
    except Exception:
        db.rollback()
        logger.exception("Could not start a practice session")
        return StartResponse(enabled=False)


@router.post("/sessions/{session_id}/answers", response_model=OkResponse)
def add_answer_note(
    session_id: int,
    note: AnswerNote,
    db: Session = Depends(get_db),
    current_user: StaffUser = Depends(require_staff),
):
    if not settings.PRACTICE_LOG_ENABLED:
        return OkResponse(ok=True, enabled=False)
    if note.field not in LOGGED_FIELDS:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown field.")
    session = _own_session(db, session_id, current_user)
    try:
        row = (
            db.query(PracticeLogAnswer)
            .filter(PracticeLogAnswer.session_id == session.id, PracticeLogAnswer.field == note.field)
            .first()
        )
        if row is None:
            row = PracticeLogAnswer(session_id=session.id, field=note.field)
            db.add(row)
        # The session starts at the consent screen, before voice or typing is chosen: the first
        # note heard by an engine makes it a voice session.
        heard_by_engine = note.accepted_engine in ("whisper", "browser") or any(
            a.engine in ("whisper", "browser") for a in note.attempt_details
        )
        if heard_by_engine and session.input_method != "voice":
            session.input_method = "voice"
        # A question asked again (go back, a fix on the review page) replaces the earlier note.
        row.attempts = note.attempts
        row.attempt_details = [a.model_dump() for a in note.attempt_details]
        row.first_value = _canon(note.field, note.first_value) or None
        row.accepted_value = note.accepted_value
        row.accepted_engine = note.accepted_engine
        row.seconds = note.seconds
        row.final_value = None
        row.changed_after_voice = None
        if note.field in ADDRESS_PART_FIELDS:
            _apply_final(row, None)                       # parts are never saved apart
        else:
            # The note may arrive after /forms/issue closed the session (notes do not block).
            _apply_final(row, _issued_applicant(db, session))
        db.commit()
        return OkResponse(ok=True)
    except Exception:
        db.rollback()
        logger.exception("Could not write a practice log note (session %s)", session_id)
        return OkResponse(ok=False)


@router.post("/sessions/{session_id}/finish", response_model=OkResponse)
def finish_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: StaffUser = Depends(require_staff),
):
    if not settings.PRACTICE_LOG_ENABLED:
        return OkResponse(ok=True, enabled=False)
    session = _own_session(db, session_id, current_user)
    if session.outcome != "open":
        return OkResponse(ok=True)                        # an issued session stays issued
    try:
        session.outcome = "abandoned"
        session.ended_at = datetime.now(timezone.utc)
        db.commit()
        return OkResponse(ok=True)
    except Exception:
        db.rollback()
        logger.exception("Could not finish practice session %s", session_id)
        return OkResponse(ok=False)


# ─── Admin: read and mode ─────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    enabled: bool
    mode: str
    build_label: str
    backend_commit: str


class ModeBody(BaseModel):
    mode: Literal["test", "real"]


class AnswerResponse(BaseModel):
    field: str
    attempts: int
    attempt_details: list
    first_value: Optional[str] = None
    accepted_value: Optional[str] = None
    accepted_engine: str
    seconds: Optional[float] = None
    final_value: Optional[str] = None
    changed_after_voice: Optional[bool] = None


class SessionResponse(BaseModel):
    id: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    build_label: Optional[str] = None
    backend_commit: Optional[str] = None
    mode: str
    staff_username: Optional[str] = None
    input_method: str
    outcome: str
    answers_count: int = 0


class SessionDetail(SessionResponse):
    answers: List[AnswerResponse] = []


def _session_response(session: PracticeLogSession, username: Optional[str], count: int) -> dict:
    return {
        "id": session.id,
        "started_at": session.started_at,
        "ended_at": session.ended_at,
        "build_label": session.build_label,
        "backend_commit": session.backend_commit,
        "mode": session.mode,
        "staff_username": username,
        "input_method": session.input_method,
        "outcome": session.outcome,
        "answers_count": count,
    }


@router.get("/status", response_model=StatusResponse)
def practice_log_status(db: Session = Depends(get_db), _admin: StaffUser = Depends(require_admin)):
    enabled = settings.PRACTICE_LOG_ENABLED
    return StatusResponse(
        enabled=enabled,
        mode=_current_mode(db) if enabled else "test",
        build_label=settings.PRACTICE_LOG_BUILD,
        backend_commit=backend_commit(),
    )


@router.put("/mode", response_model=StatusResponse)
def set_mode(body: ModeBody, db: Session = Depends(get_db), _admin: StaffUser = Depends(require_admin)):
    if not settings.PRACTICE_LOG_ENABLED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The practice log is off.")
    row = db.get(PracticeLogSettings, 1)
    if row is None:
        row = PracticeLogSettings(id=1, mode=body.mode)
        db.add(row)
    else:
        row.mode = body.mode
    db.commit()
    return practice_log_status(db, _admin)


@router.get("/sessions", response_model=List[SessionResponse])
def list_sessions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _admin: StaffUser = Depends(require_admin),
):
    if not settings.PRACTICE_LOG_ENABLED:
        return []
    counts = (
        db.query(PracticeLogAnswer.session_id, func.count(PracticeLogAnswer.id).label("n"))
        .group_by(PracticeLogAnswer.session_id)
        .subquery()
    )
    rows = (
        db.query(PracticeLogSession, StaffUser.username, func.coalesce(counts.c.n, 0))
        .outerjoin(StaffUser, StaffUser.id == PracticeLogSession.staff_user_id)
        .outerjoin(counts, counts.c.session_id == PracticeLogSession.id)
        .order_by(PracticeLogSession.started_at.desc(), PracticeLogSession.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_session_response(s, username, count) for s, username, count in rows]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    _admin: StaffUser = Depends(require_admin),
):
    if not settings.PRACTICE_LOG_ENABLED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The practice log is off.")
    session = db.get(PracticeLogSession, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Practice session not found.")
    answers = (
        db.query(PracticeLogAnswer)
        .filter(PracticeLogAnswer.session_id == session.id)
        .order_by(PracticeLogAnswer.created_at, PracticeLogAnswer.id)
        .all()
    )
    user = db.get(StaffUser, session.staff_user_id) if session.staff_user_id else None
    return {
        **_session_response(session, user.username if user else None, len(answers)),
        "answers": [
            {
                "field": a.field, "attempts": a.attempts, "attempt_details": a.attempt_details or [],
                "first_value": a.first_value, "accepted_value": a.accepted_value,
                "accepted_engine": a.accepted_engine, "seconds": a.seconds,
                "final_value": a.final_value, "changed_after_voice": a.changed_after_voice,
            }
            for a in answers
        ],
    }
