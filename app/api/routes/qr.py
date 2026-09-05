"""
app/api/routes/qr.py
QR code endpoints for mobile-to-desktop pre-fill workflow.
POST /api/qr/generate  — mobile app: generate QR from form data
POST /api/qr/scan      — desktop: decode and claim QR submission
POST /api/qr/decode    — desktop: decode QR from uploaded image
"""
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_access_token, get_current_user
from app.models.citizen_user import CitizenUser
from app.models.mobile_submission import MobileSubmission
from app.services.qr_service import generate_qr_code, claim_qr_submission, decode_qr_from_image
from app.schemas.schemas import (
    QRGenerateRequest, QRGenerateResponse,
    QRScanRequest, QRScanResponse, ApplicantCreate
)

router = APIRouter()
citizen_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/citizen/login", auto_error=False)


def get_optional_current_citizen(
    token: Optional[str] = Depends(citizen_oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[CitizenUser]:
    """
    Return the authenticated citizen when a mobile bearer token is present.
    QR generation remains public, so missing or invalid citizen tokens are ignored.
    """
    if not token:
        return None

    payload = decode_access_token(token)
    if not payload or payload.get("type") != "citizen":
        return None

    username = payload.get("sub")
    if not username:
        return None

    user = db.query(CitizenUser).filter(CitizenUser.username == username).first()
    if not user or not user.is_active:
        return None

    return user


def get_current_citizen(
    token: str = Depends(citizen_oauth2_scheme),
    db: Session = Depends(get_db),
) -> CitizenUser:
    """Require a valid mobile citizen bearer token."""
    user = get_optional_current_citizen(token=token, db=db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate citizen credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@router.post("/generate", response_model=QRGenerateResponse, status_code=201)
def generate_qr(
    payload: QRGenerateRequest,
    db: Session = Depends(get_db),
    current_citizen: Optional[CitizenUser] = Depends(get_optional_current_citizen),
):
    """
    Generate a QR code for a mobile pre-fill submission.
    This endpoint is called by the Android app (no auth required — public endpoint).

    The QR code encodes a secure token. The full form data is stored server-side
    and only retrieved when the desktop scans the code.
    """
    form_data = payload.form_data.model_dump()

    # Convert date objects to ISO strings for JSON storage
    for k, v in form_data.items():
        if hasattr(v, "isoformat"):
            form_data[k] = v.isoformat()

    try:
        token, img_b64, expires_at, submission_id = generate_qr_code(
            form_data=form_data,
            db=db,
            mobile_user_id=str(current_citizen.id) if current_citizen else None,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QR generation failed: {str(e)}",
        )

    return QRGenerateResponse(
        qr_token=token,
        qr_image_base64=img_b64,
        expires_at=expires_at,
        submission_id=submission_id,
    )


@router.get("/history")
def get_qr_history(
    db: Session = Depends(get_db),
    current_citizen: CitizenUser = Depends(get_current_citizen),
):
    """
    Return QR submissions created by the signed-in mobile citizen only.
    Staff users and other citizens cannot see this list.
    """
    submissions = (
        db.query(MobileSubmission)
        .filter(MobileSubmission.mobile_user_id == str(current_citizen.id))
        .order_by(MobileSubmission.submitted_at.desc())
        .limit(20)
        .all()
    )

    return [
        {
            "id": submission.id,
            "submission_id": submission.id,
            "qr_token": submission.qr_token,
            "qr_image_base64": "",
            "expires_at": submission.expires_at,
            "submitted_at": submission.submitted_at,
            "form_data": submission.form_data,
            "is_claimed": submission.is_claimed,
        }
        for submission in submissions
    ]


@router.post("/scan", response_model=QRScanResponse)
def scan_qr_token(
    payload: QRScanRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Claim a QR submission by token string.
    Called by the desktop after parsing the QR code value.

    Returns the form data pre-filled from the citizen's mobile submission.
    Marks the token as claimed to prevent re-use.
    """
    result = claim_qr_submission(payload.qr_token, db)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="QR code is invalid, expired, or already used. "
                   "Ask the citizen to generate a new QR code.",
        )

    # Parse date strings back to date objects
    form_data = result["form_data"]
    if form_data.get("date_of_birth") and isinstance(form_data["date_of_birth"], str):
        from datetime import date
        try:
            form_data["date_of_birth"] = date.fromisoformat(form_data["date_of_birth"])
        except ValueError:
            form_data["date_of_birth"] = None

    try:
        applicant = ApplicantCreate(**form_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"QR form data is malformed: {str(e)}",
        )

    return QRScanResponse(
        form_data=applicant.model_dump(),
        submission_id=result["submission_id"],
        submitted_at=result["submitted_at"],
    )


@router.post("/decode-image")
async def decode_qr_image(
    image_file: UploadFile = File(..., description="Image containing the QR code"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Upload an image containing a QR code and auto-claim the submission.
    Used by the desktop when the camera captures the QR as an image.
    """
    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file uploaded.")

    raw_data = decode_qr_from_image(image_bytes)
    if not raw_data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No QR code detected in the image. Ensure good lighting and focus.",
        )

    # Extract token from JSON payload
    try:
        payload_dict = json.loads(raw_data)
        token = payload_dict.get("token", raw_data)
    except json.JSONDecodeError:
        token = raw_data

    result = claim_qr_submission(token, db)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="QR code is invalid, expired, or already used.",
        )

    form_data = result["form_data"]
    if form_data.get("date_of_birth") and isinstance(form_data["date_of_birth"], str):
        from datetime import date
        try:
            form_data["date_of_birth"] = date.fromisoformat(form_data["date_of_birth"])
        except ValueError:
            form_data["date_of_birth"] = None

    applicant = ApplicantCreate(**form_data)
    return QRScanResponse(
        form_data=applicant.model_dump(),
        submission_id=result["submission_id"],
        submitted_at=result["submitted_at"],
    )
