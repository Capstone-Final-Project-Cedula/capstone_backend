"""
app/services/qr_service.py
QR code generation (for mobile submissions) and decoding (for desktop scanning).
Uses qrcode for generation and pyzbar for decoding.
"""
import qrcode
import json
import uuid
import base64
import io
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.mobile_submission import MobileSubmission

logger = logging.getLogger(__name__)


# ─── QR Generation ─────────────────────────────────────────────────────────────

def generate_qr_code(
    form_data: dict,
    db: Session,
    mobile_user_id: Optional[str] = None,
) -> Tuple[str, str, datetime, int]:
    """
    Store form data in the database and generate a QR code image.

    Args:
        form_data: Dict of all BIR Form 0016 fields
        db: SQLAlchemy session
        mobile_user_id: Optional citizen app user ID

    Returns:
        Tuple of (qr_token, qr_image_base64_png, expires_at, submission_id)
    """
    # Generate unique token
    token = str(uuid.uuid4()).replace("-", "").upper()[:32]
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.QR_EXPIRY_MINUTES)

    # Serialize dates to strings for JSON
    serializable_data = {}
    for k, v in form_data.items():
        if hasattr(v, "isoformat"):
            serializable_data[k] = v.isoformat()
        elif v is None:
            serializable_data[k] = None
        else:
            serializable_data[k] = v

    # Store in database
    submission = MobileSubmission(
        qr_token=token,
        form_data=serializable_data,
        expires_at=expires_at,
        mobile_user_id=mobile_user_id,
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    # Generate QR code image
    # The QR encodes ONLY the token (not the full data) for security
    qr_payload = json.dumps({"token": token, "system": "cedula"})

    qr_img = qrcode.QRCode(
        version=None,           # Auto-select version
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
        box_size=10,
        border=4,
    )
    qr_img.add_data(qr_payload)
    qr_img.make(fit=True)

    img = qr_img.make_image(fill_color="black", back_color="white")

    # Convert to base64 PNG
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    logger.info(f"QR code generated: token={token[:8]}... expires={expires_at}")
    return token, img_base64, expires_at, submission.id


# ─── QR Decoding (Desktop Scan) ────────────────────────────────────────────────

def decode_qr_from_image(image_bytes: bytes) -> Optional[str]:
    """
    Decode a QR code from raw image bytes using pyzbar.

    Args:
        image_bytes: Raw bytes of a PNG/JPEG image containing the QR code

    Returns:
        The decoded string (QR payload), or None if no QR found
    """
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))
        decoded = pyzbar_decode(img)
        if decoded:
            return decoded[0].data.decode("utf-8")
    except ImportError:
        logger.warning("pyzbar not installed. QR scanning via image unavailable.")
    except Exception as e:
        logger.error(f"QR decode error: {e}")
    return None


def claim_qr_submission(
    qr_token: str,
    db: Session,
) -> Optional[dict]:
    """
    Retrieve and claim a mobile QR submission by token.
    Returns the form_data dict if valid, None otherwise.

    Args:
        qr_token: Token extracted from the scanned QR code
        db: SQLAlchemy session

    Returns:
        form_data dict or None
    """
    # Try to parse token from JSON payload if needed
    try:
        payload = json.loads(qr_token)
        token = payload.get("token", qr_token)
    except (json.JSONDecodeError, TypeError):
        token = qr_token

    submission = db.query(MobileSubmission).filter(
        MobileSubmission.qr_token == token
    ).first()

    if not submission:
        logger.warning(f"QR token not found: {token[:8]}...")
        return None

    if submission.is_claimed:
        logger.warning(f"QR token already claimed: {token[:8]}...")
        return None

    if submission.is_expired():
        logger.warning(f"QR token expired: {token[:8]}...")
        return None

    # Mark as claimed
    submission.is_claimed = True
    submission.claimed_at = datetime.now(timezone.utc)
    db.commit()

    logger.info(f"QR submission claimed: token={token[:8]}...")
    return {
        "form_data": submission.form_data,
        "submission_id": submission.id,
        "submitted_at": submission.submitted_at,
    }
