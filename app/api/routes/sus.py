"""
app/api/routes/sus.py
System Usability Scale (SUS) survey (capstone research).
POST /api/sus/responses — web or mobile app submits a survey (no login needed,
                          so the standalone /sus page works from a QR code)
GET  /api/sus/responses — admin/staff: list responses
GET  /api/sus/summary   — admin/staff: counts for the Settings page
POST /api/sus/resync    — admin: send unsynced responses to Google Sheets now

New rows reach the Google Sheet on their own: a database trigger calls the
sus-sheets-sync Edge Function (see supabase/migrations). /resync only asks it
to run again, e.g. after the internet was down.
"""
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.core.database import IS_POSTGRES, get_db
from app.core.security import require_admin, require_staff
from app.models.issuance import CTCIssuance
from app.models.sus_response import SusResponse

router = APIRouter()

Role = Literal["barangay_secretary", "cedula_personnel", "citizen_applicant"]
Experience = Literal["none", "basic", "comfortable", "advanced"]
Feature = Literal["voice_input", "qr_prefill", "form_generation", "receipt", "voice_feedback"]
Platform = Literal["web", "mobile"]

# A kiosk laptop sends every citizen's survey from one address, so the limit
# only stops scripted flooding, not a busy office.
RATE_LIMIT, RATE_WINDOW_SECONDS = 30, 600
_recent_submissions: dict[str, deque] = defaultdict(deque)
_rate_lock = threading.Lock()


def _check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    with _rate_lock:
        recent = _recent_submissions[client_ip]
        while recent and now - recent[0] > RATE_WINDOW_SECONDS:
            recent.popleft()
        if len(recent) >= RATE_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many surveys from this device. Please try again later.",
            )
        recent.append(now)


class SusResponseCreate(BaseModel):
    respondent_name: Optional[str] = Field(None, max_length=120)
    role: Role
    computer_experience: Experience
    features_used: List[Feature] = Field(default_factory=list, max_length=5)
    q1: int = Field(..., ge=1, le=5)
    q2: int = Field(..., ge=1, le=5)
    q3: int = Field(..., ge=1, le=5)
    q4: int = Field(..., ge=1, le=5)
    q5: int = Field(..., ge=1, le=5)
    q6: int = Field(..., ge=1, le=5)
    q7: int = Field(..., ge=1, le=5)
    q8: int = Field(..., ge=1, le=5)
    q9: int = Field(..., ge=1, le=5)
    q10: int = Field(..., ge=1, le=5)
    comments: Optional[str] = Field(None, max_length=2000)
    consent_given: bool
    platform: Platform
    transaction_id: Optional[int] = Field(None, ge=1)

    @field_validator("respondent_name", "comments")
    @classmethod
    def blank_to_none(cls, value: Optional[str]) -> Optional[str]:
        value = (value or "").strip()
        return value or None

    @field_validator("features_used")
    @classmethod
    def unique_features(cls, value: List[str]) -> List[str]:
        return list(dict.fromkeys(value))

    @field_validator("consent_given")
    @classmethod
    def consent_required(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("Consent is required to submit the survey.")
        return value


class SusResponseOut(BaseModel):
    id: UUID
    created_at: datetime
    respondent_name: Optional[str]
    role: str
    computer_experience: str
    features_used: List[str]
    q1: int
    q2: int
    q3: int
    q4: int
    q5: int
    q6: int
    q7: int
    q8: int
    q9: int
    q10: int
    sus_score: float
    comments: Optional[str]
    platform: str
    transaction_id: Optional[int]
    synced_to_sheets: bool

    model_config = {"from_attributes": True}


@router.post("/responses", status_code=201)
def submit_response(payload: SusResponseCreate, request: Request, db: Session = Depends(get_db)):
    """Save one completed survey. The score is not returned to the respondent."""
    _check_rate_limit(request.client.host if request.client else "unknown")

    if payload.transaction_id is not None and db.get(CTCIssuance, payload.transaction_id) is None:
        raise HTTPException(status_code=422, detail="Unknown transaction.")

    response = SusResponse(**payload.model_dump())
    db.add(response)
    db.commit()
    return {"id": str(response.id), "message": "Thank you for answering the survey."}


@router.get("/responses", response_model=List[SusResponseOut])
def list_responses(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user=Depends(require_staff),
):
    return (
        db.query(SusResponse)
        .order_by(SusResponse.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/summary")
def response_summary(db: Session = Depends(get_db), current_user=Depends(require_staff)):
    total, pending, mean = db.query(
        func.count(SusResponse.id),
        func.count(SusResponse.id).filter(SusResponse.synced_to_sheets.is_(False)),
        func.avg(SusResponse.sus_score),
    ).one()
    return {
        "total": total,
        "pending_sync": pending,
        "mean_sus_score": round(float(mean), 1) if mean is not None else None,
    }


@router.post("/resync")
def resync_to_sheets(db: Session = Depends(get_db), admin=Depends(require_admin)):
    """Ask the sus-sheets-sync Edge Function to send every unsynced response."""
    if not IS_POSTGRES:
        raise HTTPException(status_code=503, detail="Google Sheets sync needs the Supabase database.")
    pending = db.execute(text("select public.sus_request_sync()")).scalar()
    db.commit()  # pg_net sends the queued request only after commit
    return {
        "pending_sync": pending,
        "message": (
            f"Sending {pending} response(s) to Google Sheets. Refresh in a minute to check."
            if pending else "Every response is already in Google Sheets."
        ),
    }
