"""
app/api/routes/eval.py
Evaluation mode (plan 003).

Staff (write-only; they return a session id or ok, never anything that was logged):
  GET  /api/eval/consent                    only {recording: bool}, to word the consent popup
  POST /api/eval/sessions                   "I agree, start": returns {session_id} (null when logging is off)
  POST /api/eval/sessions/{id}/finish       the page was left, or the late batch of an issued session

Admin only (require_admin on every one; staff get 403, a citizen token 401):
  GET/PUT /api/eval/settings                evaluation on/off, the current round, "counts toward evaluation"
  GET  /api/eval/stats                      the numbers (aggregated queries); view=counted (default) or all
  GET  /api/eval/sessions, /sessions/{id}   sessions and their tries (with the card's words)
  GET/POST /api/eval/cards                  the next card to be read / "Generate card" (plan 009)
  GET  /api/eval/cards/{id}, /cards/{id}.pdf   one card, and its printable one-page PDF
  GET  /api/eval/answers/{id}/audio         play one saved clip (through the backend, never a public link)
  PUT/DELETE /api/eval/answers/{id}/hand-score   a hand re-score from the clip (plan 009)
  GET  /api/eval/export/{sessions|answers|summary}.csv    streamed CSV
  GET  /api/eval/delete-audio, POST /api/eval/delete-audio   count / one batch of "Delete evaluation audio now"
"""
import csv
import io
import json
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
from app.models.eval import EvalAnswer, EvalCard, EvalSession
from app.models.staff_user import StaffUser
from app.services import eval_card_pdf, eval_service, storage_service

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
    """What only the browser knows about one question (the browser's recorder note)."""
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
                                             settings.APP_BUILD)
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
    counts: bool = True
    card_sets: List[str] = []


class EvalSettingsUpdate(BaseModel):
    evaluation_mode: Optional[bool] = None
    round_type: Optional[Literal["practice", "scripted", "uat", "live"]] = None
    round_condition: Optional[Literal["quiet", "noisy"]] = None
    card_set: Optional[str] = Field(None, max_length=40)
    counts: Optional[bool] = None
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


View = Literal["counted", "all"]


def _filters(round_type, condition, input_method, date_from, date_to, view="counted") -> dict:
    return {"round_type": round_type, "condition": condition, "input_method": input_method,
            "date_from": date_from, "date_to": date_to, "view": view}


@router.get("/stats")
def stats(
    view: View = "counted",
    round_type: Optional[Literal["practice", "scripted", "uat", "live"]] = None,
    condition: Optional[Literal["quiet", "noisy"]] = None,
    input_method: Optional[Literal["voice", "qr", "manual"]] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: Session = Depends(get_db),
    _admin: StaffUser = Depends(require_admin),
):
    return eval_service.stats(db, _filters(round_type, condition, input_method, date_from, date_to, view))


def _session_dict(s: EvalSession) -> dict:
    return {
        "id": s.id, "started_at": s.started_at, "ended_at": s.ended_at, "round_no": s.round_no,
        "round_type": s.round_type, "round_condition": s.round_condition, "card_set": s.card_set,
        "participant_code": s.participant_code, "card_no": s.card_no, "input_method": s.input_method,
        "device": s.device, "outcome": s.outcome, "issuance_id": s.issuance_id, "error_count": s.error_count,
        "app_seconds": s.app_seconds, "qr_seconds": s.qr_seconds, "scored": s.scored,
        "card_id": s.card_id, "counts": s.counts,
        # The same rule as eval_service.counted_clause().
        "counted": bool(s.outcome == "issued" and s.counts and s.card_id and s.round_type != "practice"),
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
        "s_errors": a.s_errors, "d_errors": a.d_errors, "i_errors": a.i_errors,
        "hand_reference": a.hand_reference, "hand_hypothesis": a.hand_hypothesis, "hand_s": a.hand_s,
        "hand_d": a.hand_d, "hand_i": a.hand_i, "hand_words": a.hand_words,
        "hand_errors": None if a.hand_words is None else (a.hand_s or 0) + (a.hand_d or 0) + (a.hand_i or 0),
        "seconds": a.seconds,
    }


@router.get("/sessions")
def list_sessions(
    view: View = "counted",
    round_type: Optional[Literal["practice", "scripted", "uat", "live"]] = None,
    condition: Optional[Literal["quiet", "noisy"]] = None,
    input_method: Optional[Literal["voice", "qr", "manual"]] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _admin: StaffUser = Depends(require_admin),
):
    query = db.query(EvalSession)
    if view == "counted":
        query = query.filter(eval_service.counted_clause())
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
    card = eval_service.card_answers(db, session)

    def with_card(a: EvalAnswer) -> dict:
        expected = eval_service.expected_in(card, a.field)
        # The card's words next to what Whisper heard, so the admin can re-score from the clip.
        return {**_answer_dict(a), "card_read": expected["read"] if expected else None}

    return {**_session_dict(session), "has_card": card is not None, "answers": [with_card(a) for a in rows]}


# ─── Generated cards (plan 009) ───────────────────────────────────────────────

def _card_dict(db: Session, card: Optional[EvalCard]) -> Optional[dict]:
    if card is None:
        return None
    data = json.loads(card.card)
    claimed = db.get(EvalSession, card.claimed_by_session_id) if card.claimed_by_session_id else None
    return {
        "id": card.id, "created_at": card.created_at, "style": card.style, "round_no": card.round_no,
        "claimed_by": claimed.participant_code if claimed else None, "claimed_at": card.claimed_at,
        "lines": eval_card_pdf.card_lines(data["answers"]),
    }


def _cards_response(db: Session) -> dict:
    recent = db.query(EvalCard).order_by(EvalCard.id.desc()).limit(10).all()
    return {
        "next": _card_dict(db, eval_service.next_card(db)),
        "waiting": eval_service.waiting_cards(db),
        "recent": [{"id": c.id, "created_at": c.created_at, "style": c.style,
                    "claimed_by": (_card_dict(db, c) or {}).get("claimed_by")} for c in recent],
    }


@router.get("/cards")
def list_cards(db: Session = Depends(get_db), _admin: StaffUser = Depends(require_admin)):
    """The next card to be read (the oldest unclaimed one: what the next session is scored against)."""
    return _cards_response(db)


@router.post("/cards")
def generate_card(db: Session = Depends(get_db), admin: StaffUser = Depends(require_admin)):
    """"Generate card": a new made-up person, queued behind any card still waiting."""
    eval_service.create_card(db, admin.id)
    return _cards_response(db)


@router.get("/cards/{card_id}.pdf")
def card_pdf(card_id: int, db: Session = Depends(get_db), _admin: StaffUser = Depends(require_admin)):
    card = db.get(EvalCard, card_id)
    if card is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Card not found.")
    pdf = eval_card_pdf.render(card.id, json.loads(card.card)["answers"])
    return Response(content=pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="evaluation-card-{card.id}.pdf"', "Cache-Control": "no-store"})


@router.get("/cards/{card_id}")
def get_card(card_id: int, db: Session = Depends(get_db), _admin: StaffUser = Depends(require_admin)):
    card = db.get(EvalCard, card_id)
    if card is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Card not found.")
    return _card_dict(db, card)


# ─── Hand re-score (plan 009) ─────────────────────────────────────────────────

class HandScoreBody(BaseModel):
    reference: str = Field(..., max_length=500)        # what the participant really said, from the clip
    hypothesis: str = Field(..., max_length=500)       # what Whisper heard


@router.put("/answers/{answer_id}/hand-score")
def save_hand_score(answer_id: int, body: HandScoreBody, db: Session = Depends(get_db),
                    admin: StaffUser = Depends(require_admin)):
    row = eval_service.hand_score(db, answer_id, body.reference, body.hypothesis, admin.id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Only one question's (or one address / birthplace step's) result with a WER count "
                                   "(a card and a first spoken Whisper try) takes a hand score.")
    return _answer_dict(row)


@router.delete("/answers/{answer_id}/hand-score")
def clear_hand_score(answer_id: int, db: Session = Depends(get_db), admin: StaffUser = Depends(require_admin)):
    row = eval_service.hand_score(db, answer_id, None, None, admin.id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such result row.")
    return _answer_dict(row)


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
                   "card_id", "counts", "counted", "input_method", "device", "outcome", "started_at", "ended_at",
                   "app_seconds", "qr_seconds", "error_count"]
ANSWER_COLUMNS = ["session_id", "participant_code", "round_type", "field", "step", "attempt_no", "engine", "model",
                  "beam", "transcript", "whisper_seconds", "extracted_value", "extract_ok", "yes_no",
                  "expected_value", "final_value", "finished_in", "spoken_tries", "spelled_tries", "skipped",
                  "correct_first", "correct_ladder", "correct_finished", "wer_errors", "wer_words", "s_errors",
                  "d_errors", "i_errors", "hand_reference", "hand_hypothesis", "hand_s", "hand_d", "hand_i",
                  "hand_words", "seconds"]


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


SUMMARY_COLUMNS = ["group", "name", "answers", "field_extraction_accuracy_pct", "first_try_pct", "finished_pct",
                   "wer_pct", "wer_errors", "reference_words", "substitutions", "deletions", "insertions",
                   "hand_wer_pct", "hand_scored", "skip_pct", "typed_pct", "spelled_pct", "avg_spoken_tries",
                   "avg_seconds"]


def _summary_lines(db: Session, filters: dict) -> Iterator[str]:
    """Shaped like the Chapter 4 tables: all 14 questions (Table 7), then each field (Table 8)."""
    data = eval_service.stats(db, filters)
    yield _csv_line(SUMMARY_COLUMNS)
    blocks = [("overall", "all 14 questions", data["overall"])]
    blocks += [("field", name, values) for name, values in data["fields"].items()]
    blocks += [("place_of_birth_step", name, values) for name, values in data["place_of_birth_steps"].items()]
    blocks += [("address_part", name, values) for name, values in data["address_parts"].items()]
    for group, name, v in blocks:
        yield _csv_line([group, name, v["answers"], v["field_extraction_pct"], v["first_try_pct"], v["finished_pct"],
                         v["wer_pct"], v["wer_errors"], v["wer_words"], v["substitutions"], v["deletions"],
                         v["insertions"], v["hand_wer_pct"], v["hand_scored"], v["skip_pct"], v["typed_pct"],
                         v["spelled_pct"], v["avg_spoken_tries"], v["avg_seconds"]])
    yield _csv_line(["count", "wer_participants", data["wer"]["participants"]])
    yield _csv_line(["count", "counted_sessions", data["progress"]["counted"]])
    yield _csv_line(["target", "wer_pct_at_most", data["targets"]["wer_pct"]])
    yield _csv_line(["target", "field_extraction_accuracy_pct_at_least", data["targets"]["field_extraction_pct"]])
    for key, value in data["timing"].items():
        yield _csv_line(["timing", key, value])
    for key, value in {**data["sessions"], "errors": data["errors"], **data["security"]}.items():
        yield _csv_line(["count", key, value])


@router.get("/export/{kind}.csv")
def export_csv(
    kind: Literal["sessions", "answers", "summary"],
    view: View = "counted",
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
    return _stream(_summary_lines(db, _filters(round_type, condition, input_method, None, None, view)),
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
