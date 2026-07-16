"""
app/schemas/schemas.py
Pydantic v2 schemas for request validation and response serialization.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Any
from datetime import date, datetime
from app.models.applicant import SexEnum, CivilStatusEnum, TaxClassificationEnum
from app.models.issuance import InputMethodEnum
from app.models.staff_user import RoleEnum


# ─── Auth Schemas ──────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


# ─── Staff User Schemas ────────────────────────────────────────────────────────

class StaffUserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: str
    position: Optional[str] = None
    email: Optional[str] = None
    role: RoleEnum = RoleEnum.STAFF


class StaffUserResponse(BaseModel):
    id: int
    username: str
    full_name: str
    position: Optional[str]
    email: Optional[str]
    role: RoleEnum
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Applicant Schemas ─────────────────────────────────────────────────────────

class ApplicantCreate(BaseModel):
    surname: str = Field(..., min_length=1, max_length=100)
    given_name: str = Field(..., min_length=1, max_length=100)
    middle_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    address: str
    sex: Optional[SexEnum] = None
    civil_status: Optional[CivilStatusEnum] = None
    citizenship: str = "Filipino"
    icr_number: Optional[str] = None
    place_of_birth: Optional[str] = None
    height_cm: Optional[float] = Field(None, gt=0, lt=300)
    weight_kg: Optional[float] = Field(None, gt=0, lt=500)
    occupation: Optional[str] = None
    gross_annual_income: Optional[float] = Field(None, ge=0)
    tax_classification: TaxClassificationEnum = TaxClassificationEnum.A


class ApplicantResponse(ApplicantCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Issuance Schemas ──────────────────────────────────────────────────────────

class IssuanceCreate(BaseModel):
    applicant_data: ApplicantCreate
    input_method: InputMethodEnum = InputMethodEnum.VOICE
    raw_transcript: Optional[str] = None


class IssuanceResponse(BaseModel):
    id: int
    ctc_number: str
    basic_community_tax: float
    additional_community_tax: float
    total_tax: float
    input_method: InputMethodEnum
    issued_at: datetime
    pdf_path: Optional[str]
    applicant: ApplicantResponse

    model_config = {"from_attributes": True}


class DashboardStats(BaseModel):
    daily_count: int
    monthly_count: int
    total_count: int
    total_tax_collected_today: float
    total_tax_collected_month: float
    recent_issuances: List[IssuanceResponse]


# ─── Speech Schemas ───────────────────────────────────────────────────────────

class TranscriptionResponse(BaseModel):
    transcript: str
    language: str
    confidence: float
    segments: List[dict]
    processing_time_seconds: float


# ─── Extraction Schemas ───────────────────────────────────────────────────────

class ExtractionRequest(BaseModel):
    transcript: str


class ExtractionResponse(BaseModel):
    extracted_fields: ApplicantCreate
    confidence_scores: dict
    unextracted_fields: List[str]  # Fields that couldn't be extracted
    warnings: List[str]


# ─── QR Schemas ──────────────────────────────────────────────────────────────

class QRGenerateRequest(BaseModel):
    form_data: ApplicantCreate


class QRGenerateResponse(BaseModel):
    qr_token: str
    qr_image_base64: str  # Base64-encoded PNG of the QR code
    expires_at: datetime
    submission_id: int


class QRScanRequest(BaseModel):
    qr_token: str


class QRScanResponse(BaseModel):
    form_data: ApplicantCreate
    submission_id: int
    submitted_at: datetime


# ─── PDF Schemas ──────────────────────────────────────────────────────────────

class PDFGenerateRequest(BaseModel):
    issuance_id: int


class PDFGenerateResponse(BaseModel):
    pdf_path: str
    ctc_number: str
    download_url: str


# ─── Tax Computation ──────────────────────────────────────────────────────────

class TaxComputationResult(BaseModel):
    basic_community_tax: float = 5.00
    additional_community_tax: float
    total_tax: float
    tax_classification: TaxClassificationEnum
    breakdown: str  # Human-readable explanation
