"""
app/api/routes/eval.py
Evaluation mode (plan 003).

Staff (write-only; they return a session id or ok, never anything that was logged):
  GET  /api/eval/consent                    only {recording: bool}, to word the consent popup
  POST /api/eval/sessions                   "I agree, start": returns {session_id} (null when logging is off)
  POST /api/eval/sessions/{id}/finish       the page was left, or the late batch of an issued session

Admin only (require_admin on every one; staff get 403, a citizen token 401):
  GET/PUT /api/eval/settings                evaluation on/off and the current round
  GET  /api/eval/stats                      the numbers (aggregated queries)
  GET  /api/eval/sessions, /sessions/{id}   sessions and their tries
  GET  /api/eval/answers/{id}/audio         play one saved clip (through the backend, never a public link)
  GET  /api/eval/export/{sessions|answers|summary}.csv    streamed CSV
  GET  /api/eval/delete-audio, POST /api/eval/delete-audio   count / one batch of "Delete evaluation audio now"
"""
import csv
import io
import logging
from datetime import datetime
from typing import Iterator, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal, get_db
from app.core.security import require_admin, require_staff
from app.models.eval import EvalAnswer, EvalSession
from app.models.staff_user import StaffUser
from app.services import eval_service, storage_service

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Staff: write-only ────────────────────────────────────────────────────────

class ConsentResponse(BaseModel):
    recording: bool


class StartBody(BaseModel):
    input_method: Literal["voice", "qr", "manual"] = "voice"
    device: Optional[str] = Field(None, max_length=200)


class StartResponse(BaseModel):
    session_id: Optional[int] = None


class FieldNote(BaseModel):
    """What only the browser knows about one question (the practice recorder's note, plan 001.7)."""
    field: str = Field(..., min_length=1, max_length=40)
    accepted_engine: Literal["whisper", "browser", "typed", "spelled", "skipped"] = "whisper"
    accepted_value: Optional[str] = Field(None, max_length=500)
    left_blank: bool = False
    skipped: bool = False
    seconds: Optional[float] = Field(None, ge=0, le=3600)


class EvalBatch(BaseModel):
    notes: List[FieldNote] = Field(default_factory=list, max_length=40)
    errors: int = Field(0, ge=0, le=1000)
    qr_seconds: Optional[float] = Field(None, ge=0, le=36000)


class OkResponse(BaseModel):
    ok: bool


@router.get("/consent", response_model=ConsentResponse)
def consent(db: Session = Depends(get_db), _user: StaffUser = Depends(require_staff)):
    return ConsentResponse(recording=eval_service.recording_on(db))


@router.post("/sessions", response_model=StartResponse)
def start_session(body: StartBody, db: Session = Depends(get_db), user: StaffUser = Depends(require_staff)):
    try:
        session = eval_service.start_session(db, user, body.input_method, body.device,
                                             settings.PRACTICE_LOG_BUILD)
        return StartResponse(session_id=session.id if session else None)
    except Exception:
        db.rollback()
        logger.exception("Could not start an evaluation session")
        return StartResponse()


@router.post("/sessions/{session_id}/finish", response_model=OkResponse)
def finish_session(session_id: int, body: Optional[EvalBatch] = None, db: Session = Depends(get_db),
                   user: StaffUser = Depends(require_staff)):
    session = eval_service.own_open_session(db, session_id, user.id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    try:
        batch = body.model_dump() if body else None
        if session.outcome == "issued":
            # The late batch of an issued session: score again with the same saved values.
            values = eval_service.record_values_for(db, session)
            eval_service.finish_session(db, session, values, batch)
        else:
            eval_service.finish_session(db, session, None, batch)
        return OkResponse(ok=True)
    except Exception:
        db.rollback()
        logger.exception("Could not finish evaluation session %s", session_id)
        return OkResponse(ok=False)


# ─── Admin ────────────────────────────────────────────────────────────────────

class EvalSettings(BaseModel):
    evaluation_mode: bool
    round_type: Literal["practice", "scripted", "uat", "live"]
    round_condition: Literal["quiet", "noisy"]
    card_set: Optional[str] = None
    round_no: int
    practice_log_conflict: bool = False
    card_sets: List[str] = []


class EvalSettingsUpdate(BaseModel):
    evaluation_mode: Optional[bool] = None
    round_type: Optional[Literal["practice", "scripted", "uat", "live"]] = None
    round_condition: Optional[Literal["quiet", "noisy"]] = None
    card_set: Optional[str] = Field(None, max_length=40)
    new_round: bool = False


def _settings_response(db: Session) -> EvalSettings:
    return EvalSettings(**eval_service.get_eval_settings(db), card_sets=eval_service.card_set_names())


@router.get("/settings", response_model=EvalSettings)
def read_settings(db: Session = Depends(get_db), _admin: StaffUser = Depends(require_admin)):
    return _settings_response(db)


@router.put("/settings", response_model=EvalSettings)
def save_settings(body: EvalSettingsUpdate, db: Session = Depends(get_db), admin: StaffUser = Depends(require_admin)):
    values = body.model_dump(exclude={"new_round"}, exclude_unset=True)
    if values.get("card_set") is not None and values["card_set"] not in eval_service.card_set_names():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown card set.")
    if values.get("round_type", "") != "scripted" and "round_type" in values:
        values["card_set"] = None                       # only a scripted round has a card set
    eval_service.update_eval_settings(db, values, admin.id, new_round=body.new_round)
    return _settings_response(db)


def _filters(round_type, condition, input_method, date_from, date_to) -> dict:
    return {"round_type": round_type, "condition": condition, "input_method": input_method,
            "date_from": date_from, "date_to": date_to}


@router.get("/stats")
def stats(
    round_type: Optional[Literal["practice", "scripted", "uat", "live"]] = None,
    condition: Optional[Literal["quiet", "noisy"]] = None,
    input_method: Optional[Literal["voice", "qr", "manual"]] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: Session = Depends(get_db),
    _admin: StaffUser = Depends(require_admin),
):
    return eval_service.stats(db, _filters(round_type, condition, input_method, date_from, date_to))


def _session_dict(s: EvalSession) -> dict:
    return {
        "id": s.id, "started_at": s.started_at, "ended_at": s.ended_at, "round_no": s.round_no,
        "round_type": s.round_type, "round_condition": s.round_condition, "card_set": s.card_set,
        "participant_code": s.participant_code, "card_no": s.card_no, "input_method": s.input_method,
        "device": s.device, "outcome": s.outcome, "issuance_id": s.issuance_id, "error_count": s.error_count,
        "app_seconds": s.app_seconds, "qr_seconds": s.qr_seconds, "scored": s.scored,
    }


def _answer_dict(a: EvalAnswer) -> dict:
    return {
        "id": a.id, "field": a.field, "step": a.step, "attempt_no": a.attempt_no, "engine": a.engine,
        "model": a.model, "beam": a.beam, "has_audio": bool(a.audio_ref), "transcript": a.transcript,
        "whisper_seconds": a.whisper_seconds, "extracted_value": a.extracted_value, "extract_ok": a.extract_ok,
        "yes_no": a.yes_no, "expected_value": a.expected_value, "final_value": a.final_value,
        "finished_in": a.finished_in, "spoken_tries": a.spoken_tries, "spelled_tries": a.spelled_tries,
        "skipped": a.skipped, "correct_first": a.correct_first, "correct_ladder": a.correct_ladder,
        "correct_finished": a.correct_finished, "wer_errors": a.wer_errors, "wer_words": a.wer_words,
        "seconds": a.seconds,
    }


@router.get("/sessions")
def list_sessions(
    round_type: Optional[Literal["practice", "scripted", "uat", "live"]] = None,
    condition: Optional[Literal["quiet", "noisy"]] = None,
    input_method: Optional[Literal["voice", "qr", "manual"]] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _admin: StaffUser = Depends(require_admin),
):
    query = db.query(EvalSession)
    if round_type:
        query = query.filter(EvalSession.round_type == round_type)
    if condition:
        query = query.filter(EvalSession.round_condition == condition)
    if input_method:
        query = query.filter(EvalSession.input_method == input_method)
    rows = query.order_by(EvalSession.started_at.desc(), EvalSession.id.desc()).offset(offset).limit(limit).all()
    return [_session_dict(s) for s in rows]


@router.get("/sessions/{session_id}")
def get_session(session_id: int, db: Session = Depends(get_db), _admin: StaffUser = Depends(require_admin)):
    session = db.get(EvalSession, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    rows = db.query(EvalAnswer).filter(EvalAnswer.session_id == session.id).order_by(EvalAnswer.id).all()
    return {**_session_dict(session), "answers": [_answer_dict(a) for a in rows]}


@router.get("/answers/{answer_id}/audio")
def play_audio(answer_id: int, db: Session = Depends(get_db), _admin: StaffUser = Depends(require_admin)):
    row = db.get(EvalAnswer, answer_id)
    if row is None or not row.audio_ref:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No clip saved for this answer.")
    try:
        data, content_type = storage_service.download_bytes(row.audio_ref)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The clip is gone.")
    return Response(content=data, media_type=content_type, headers={"Cache-Control": "no-store"})


# ─── CSV (streamed row by row) ────────────────────────────────────────────────

CSV_PAGE = 500
SESSION_COLUMNS = ["id", "participant_code", "round_no", "round_type", "round_condition", "card_set", "card_no",
                   "input_method", "device", "outcome", "started_at", "ended_at", "app_seconds", "qr_seconds",
                   "error_count"]
ANSWER_COLUMNS = ["session_id", "participant_code", "round_type", "field", "step", "attempt_no", "engine", "model",
                  "beam", "transcript", "whisper_seconds", "extracted_value", "extract_ok", "yes_no",
                  "expected_value", "final_value", "finished_in", "spoken_tries", "spelled_tries", "skipped",
                  "correct_first", "correct_ladder", "correct_finished", "wer_errors", "wer_words", "seconds"]


def _csv_line(values) -> str:
    buffer = io.StringIO()
    csv.writer(buffer).writerow(["" if v is None else v for v in values])
    return buffer.getvalue()


def _stream(lines: Iterator[str], name: str) -> StreamingResponse:
    return StreamingResponse(lines, media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


def _session_lines() -> Iterator[str]:
    """Pages of 500 rows on the generator's own database session (the request's closes before streaming ends)."""
    db = SessionLocal()
    try:
        yield _csv_line(SESSION_COLUMNS)
        last = 0
        while True:
            rows = db.query(EvalSession).filter(EvalSession.id > last).order_by(EvalSession.id).limit(CSV_PAGE).all()
            if not rows:
                return
            for s in rows:
                yield _csv_line(_session_dict(s).get(c) for c in SESSION_COLUMNS)
            last = rows[-1].id
    finally:
        db.close()


def _answer_lines() -> Iterator[str]:
    db = SessionLocal()
    try:
        yield _csv_line(ANSWER_COLUMNS)
        last = 0
        while True:
            rows = (db.query(EvalAnswer, EvalSession.participant_code, EvalSession.round_type)
                    .join(EvalSession, EvalSession.id == EvalAnswer.session_id)
                    .filter(EvalAnswer.id > last).order_by(EvalAnswer.id).limit(CSV_PAGE).all())
            if not rows:
                return
            for a, code, round_type in rows:
                data = {**_answer_dict(a), "session_id": a.session_id, "participant_code": code,
                        "round_type": round_type}
                yield _csv_line(data.get(c) for c in ANSWER_COLUMNS)
            last = rows[-1][0].id
    finally:
        db.close()


def _summary_lines(db: Session, filters: dict) -> Iterator[str]:
    """Shaped like the Chapter 4 tables: headline, hard 5 / easy 9, then each field."""
    data = eval_service.stats(db, filters)
    columns = ["group", "name", "answers", "first_try_pct", "after_ladder_pct", "finished_pct", "wer_pct",
               "skip_pct", "typed_pct", "spelled_pct", "avg_spoken_tries"]
    yield _csv_line(columns)
    blocks = [("overall", "all 14 fields", data["overall"]), ("subset", "hard 5", data["hard5"]),
              ("subset", "easy 9", data["easy9"])]
    blocks += [("field", name, values) for name, values in data["fields"].items()]
    blocks += [("place_of_birth_step", name, values) for name, values in data["place_of_birth_steps"].items()]
    blocks += [("address_part", name, values) for name, values in data["address_parts"].items()]
    for group, name, v in blocks:
        yield _csv_line([group, name, v["answers"], v["first_try_pct"], v["after_ladder_pct"], v["finished_pct"],
                         v["wer_pct"], v["skip_pct"], v["typed_pct"], v["spelled_pct"], v["avg_spoken_tries"]])
    for key, value in data["timing"].items():
        yield _csv_line(["timing", key, value])
    for key, value in {**data["sessions"], "errors": data["errors"], **data["security"]}.items():
        yield _csv_line(["count", key, value])


@router.get("/export/{kind}.csv")
def export_csv(
    kind: Literal["sessions", "answers", "summary"],
    round_type: Optional[Literal["practice", "scripted", "uat", "live"]] = None,
    condition: Optional[Literal["quiet", "noisy"]] = None,
    input_method: Optional[Literal["voice", "qr", "manual"]] = None,
    db: Session = Depends(get_db),
    _admin: StaffUser = Depends(require_admin),
):
    if kind == "sessions":
        return _stream(_session_lines(), "evaluation-sessions.csv")
    if kind == "answers":
        return _stream(_answer_lines(), "evaluation-answers.csv")
    return _stream(_summary_lines(db, _filters(round_type, condition, input_method, None, None)),
                   "evaluation-summary.csv")


# ─── Delete evaluation audio now (manual only) ───────────────────────────────

class DeleteBody(BaseModel):
    confirm: bool = False
    size: int = Field(100, ge=1, le=500)


@router.get("/delete-audio")
def delete_status(db: Session = Depends(get_db), _admin: StaffUser = Depends(require_admin)):
    return {"remaining": eval_service.pending_delete_count(db)}


@router.post("/delete-audio")
def delete_audio(body: DeleteBody, db: Session = Depends(get_db), _admin: StaffUser = Depends(require_admin)):
    """One batch per call; the page repeats the call until `remaining` is 0. Needs confirm=true."""
    if not body.confirm:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Confirm the deletion first.")
    return eval_service.delete_batch(db, body.size)
