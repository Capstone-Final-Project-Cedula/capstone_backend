"""
app/models/issuance.py
CTCIssuance model — represents each Community Tax Certificate transaction.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum


class InputMethodEnum(str, enum.Enum):
    VOICE = "voice"
    QR = "qr"
    MANUAL = "manual"


class CTCIssuance(Base):
    __tablename__ = "ctc_issuance"

    id = Column(Integer, primary_key=True, index=True)

    # ── CTC Identification ───────────────────────────────────────────────────
    ctc_number = Column(String(30), unique=True, nullable=False, index=True)
    # Format: CTC-YYYYMMDD-XXXX (e.g., CTC-20240115-0001)

    # ── Tax Computation (BIR Rule) ───────────────────────────────────────────
    basic_community_tax = Column(Float, default=5.00)
    # ₱5.00 flat for all individuals
    additional_community_tax = Column(Float, default=0.00)
    # ₱1.00 per ₱1,000 of income/real property (max ₱5,000)
    total_tax = Column(Float, default=5.00)
    interest = Column(Float, default=0.00)
    total_amount_paid = Column(Float, default=5.00)

    # ── Processing Info ──────────────────────────────────────────────────────
    input_method = Column(Enum(InputMethodEnum), default=InputMethodEnum.VOICE)
    raw_transcript = Column(Text, nullable=True)  # Whisper output, if voice
    issued_at = Column(DateTime(timezone=True), server_default=func.now())
    pdf_path = Column(String(255), nullable=True)  # Path to generated PDF

    # ── Foreign Keys ─────────────────────────────────────────────────────────
    applicant_id = Column(Integer, ForeignKey("applicant_records.id"), nullable=False)
    issued_by = Column(Integer, ForeignKey("staff_users.id"), nullable=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    applicant = relationship("ApplicantRecord", back_populates="issuances")
    staff = relationship("StaffUser", back_populates="issuances")

    def __repr__(self):
        return f"<CTCIssuance ctc_number={self.ctc_number} total={self.total_tax}>"
