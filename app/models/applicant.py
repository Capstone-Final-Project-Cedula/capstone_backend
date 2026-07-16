"""
app/models/applicant.py
ApplicantRecord model — stores all BIR Form 0016 fields for each CTC applicant.
"""
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum


class SexEnum(str, enum.Enum):
    MALE = "Male"
    FEMALE = "Female"


class CivilStatusEnum(str, enum.Enum):
    SINGLE = "Single"
    MARRIED = "Married"
    WIDOWED = "Widowed"
    SEPARATED = "Separated"


class TaxClassificationEnum(str, enum.Enum):
    A = "A"  # Individual with income
    B = "B"  # Corporation
    C = "C"  # Individual without income


class ApplicantRecord(Base):
    __tablename__ = "applicant_records"

    id = Column(Integer, primary_key=True, index=True)

    # ── BIR Form 0016 Fields ────────────────────────────────────────────────
    surname = Column(String(100), nullable=False)
    given_name = Column(String(100), nullable=False)
    middle_name = Column(String(100), nullable=True)
    date_of_birth = Column(Date, nullable=True)
    address = Column(String(255), nullable=False)
    sex = Column(Enum(SexEnum), nullable=True)
    civil_status = Column(Enum(CivilStatusEnum), nullable=True)
    citizenship = Column(String(50), default="Filipino")
    icr_number = Column(String(50), nullable=True)       # For aliens only
    place_of_birth = Column(String(100), nullable=True)
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    occupation = Column(String(100), nullable=True)
    gross_annual_income = Column(Float, nullable=True, default=0.0)
    tax_classification = Column(
        Enum(TaxClassificationEnum), default=TaxClassificationEnum.A
    )

    # ── System Metadata ─────────────────────────────────────────────────────
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # ── Relationships ────────────────────────────────────────────────────────
    issuances = relationship("CTCIssuance", back_populates="applicant")

    def full_name(self) -> str:
        parts = [self.given_name]
        if self.middle_name:
            parts.append(self.middle_name[0] + ".")  # Middle initial
        parts.append(self.surname)
        return " ".join(parts)

    def __repr__(self):
        return f"<ApplicantRecord id={self.id} name={self.full_name()}>"
