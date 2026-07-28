"""
app/models/mobile_submission.py
MobileSubmission model — stores QR pre-fill submissions from the Android app.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime, timezone


class MobileSubmission(Base):
    __tablename__ = "mobile_submissions"

    id = Column(Integer, primary_key=True, index=True)

    # ── QR Token ─────────────────────────────────────────────────────────────
    qr_token = Column(String(64), unique=True, nullable=False, index=True)
    # UUID-based token embedded in the QR code

    # ── Submitted Data (raw JSON payload from Android app) ───────────────────
    form_data = Column(JSON, nullable=False)
    # Full BIR Form 0016 fields as JSON

    # ── Status ───────────────────────────────────────────────────────────────
    is_claimed = Column(Boolean, default=False)
    # True once desktop scans and processes this submission

    claimed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)

    # ── Submission metadata ───────────────────────────────────────────────────
    submitted_at = Column(DateTime(timezone=True), server_default=func.now())
    mobile_user_id = Column(String(100), nullable=True)  # Optional citizen ID

    def is_expired(self) -> bool:
        expires_at = self.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > expires_at

    def is_valid(self) -> bool:
        return not self.is_claimed and not self.is_expired()

    def __repr__(self):
        return f"<MobileSubmission token={self.qr_token[:8]}... claimed={self.is_claimed}>"
