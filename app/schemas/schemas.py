"""
app/schemas/schemas.py
Pydantic v2 schemas for request validation and response serialization.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Any, Union
from datetime import date, datetime
from enum import Enum

from app.models.staff_user import RoleEnum


# ─── Enums for Schemas ─────────────────────────────────────────────────────────

class SexEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class CivilStatusEnum(str, Enum):
    SINGLE = "Single"
    MARRIED = "Married"
    WIDOWED = "Widowed"
    SEPARATED = "Separated"


class TaxClassificationEnum(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class InputMethodEnum(str, Enum):
    VOICE = "voice"
    QR = "qr"
    MANUAL = "manual"


# NOTE: RoleEnum is imported from app.models.staff_user above — do not redefine
# it here. It previously existed as a separate class with different values
# ("ADMIN"/"STAFF", uppercase) than the SQLAlchemy model's RoleEnum
# ("admin"/"staff"/"viewer", lowercase), which caused ResponseValidationError
# on any endpoint returning a staff role (e.g. GET /auth/me).


# ─── Auth Schemas ──────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


# ─── Staff User Schemas ────────────────────────────────────────────────────────

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=8)


class CitizenPasswordResetRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)


class StaffUserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
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
    date_of_birth: Optional[Union[str, date]] = None  # Accept both string and date
    address: str
    sex: Optional[str] = None  # Accept string, will be validated
    civil_status: Optional[str] = None  # Accept string, will be validated
    citizenship: str = "Filipino"
    icr_number: Optional[str] = None
    place_of_birth: Optional[str] = None
    height_cm: Optional[float] = Field(None, gt=0, lt=300)
    weight_kg: Optional[float] = Field(None, gt=0, lt=500)
    occupation: Optional[str] = None
    gross_annual_income: Optional[float] = Field(None, ge=0)
    tax_classification: str = "A"

    @validator('sex', pre=True)
    def validate_sex(cls, v):
        if v is None:
            return None
        v = str(v).strip().lower()
        if v in ['male', 'm', '1']:
            return 'Male'
        if v in ['female', 'f', '2']:
            return 'Female'
        return v

    @validator('civil_status', pre=True)
    def validate_civil_status(cls, v):
        if v is None:
            return None
        v = str(v).strip().lower()
        if v in ['single', 's', '1']:
            return 'Single'
        if v in ['married', 'm', '2']:
            return 'Married'
        if v in ['widowed', 'widow', 'w', '3']:
            return 'Widowed'
        if v in ['separated', 'divorced', 'd', '4']:
            return 'Separated'
        return v

    @validator('date_of_birth', pre=True)
    def validate_date_of_birth(cls, v):
        if v is None:
            return None
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            # Try YYYY-MM-DD
            try:
                return datetime.strptime(v, "%Y-%m-%d").date()
            except ValueError:
                pass
            # Try MM/DD/YYYY
            try:
                return datetime.strptime(v, "%m/%d/%Y").date()
            except ValueError:
                pass
            # Try MM-DD-YYYY
            try:
                return datetime.strptime(v, "%m-%d-%Y").date()
            except ValueError:
                pass
            # Try natural language
            try:
                from dateutil import parser
                return parser.parse(v).date()
            except:
                return None
        return v


class ApplicantResponse(BaseModel):
    id: int
    surname: str
    given_name: str
    middle_name: Optional[str]
    date_of_birth: Optional[date]
    address: str
    sex: Optional[str]
    civil_status: Optional[str]
    citizenship: str
    icr_number: Optional[str]
    place_of_birth: Optional[str]
    height_cm: Optional[float]
    weight_kg: Optional[float]
    occupation: Optional[str]
    gross_annual_income: Optional[float]
    tax_classification: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Issuance Schemas ──────────────────────────────────────────────────────────

class IssuanceCreate(BaseModel):
    applicant_data: Union[ApplicantCreate, dict]  # Accept both Pydantic model and dict
    input_method: str = "voice"
    raw_transcript: Optional[str] = None

    @validator('input_method', pre=True)
    def validate_input_method(cls, v):
        if v in ['voice', 'qr', 'manual']:
            return v
        return 'voice'


class IssuanceResponse(BaseModel):
    id: int
    ctc_number: str
    applicant_id: int
    basic_community_tax: float
    additional_community_tax: float
    total_tax: float
    interest: float
    total_amount_paid: float
    input_method: str
    issued_at: datetime
    pdf_path: Optional[str]
    applicant: Optional[ApplicantResponse] = None

    model_config = {"from_attributes": True}


class DashboardStats(BaseModel):
    total_issued: int
    today_issued: int
    voice_processed: int
    qr_processed: int
    daily_data: List[dict]
    manual_processed: int = 0
    total_revenue: float = 0.0
    today_revenue: float = 0.0
    average_paid: float = 0.0
    monthly_data: List[dict] = []
    input_method_breakdown: List[dict] = []
    tax_classification_breakdown: List[dict] = []
    top_barangays: List[dict] = []
    recent_activity: List[dict] = []
    daily_count: int = 0
    monthly_count: int = 0
    total_count: int = 0


# ─── Speech Schemas ───────────────────────────────────────────────────────────

class TranscriptionResponse(BaseModel):
    transcript: str
    language: str
    confidence: float
    segments: List[dict] = []
    processing_time_seconds: float


# ─── Extraction Schemas ───────────────────────────────────────────────────────

class ExtractionRequest(BaseModel):
    transcript: str


class ExtractionResponse(BaseModel):
    extracted_fields: ApplicantCreate
    confidence_scores: dict = {}
    unextracted_fields: List[str] = []
    warnings: List[str] = []


# ─── QR Schemas ──────────────────────────────────────────────────────────────

class QRGenerateRequest(BaseModel):
    form_data: ApplicantCreate


class QRGenerateResponse(BaseModel):
    qr_token: str
    qr_image_base64: str
    expires_at: datetime
    submission_id: int


class QRScanRequest(BaseModel):
    qr_token: str


class QRScanResponse(BaseModel):
    form_data: dict
    submission_id: int
    submitted_at: datetime


# ─── Tax Computation ──────────────────────────────────────────────────────────

class TaxComputationResult(BaseModel):
    basic_community_tax: float = 5.00
    additional_community_tax: float
    total_tax: float
    tax_classification: str
    breakdown: str


# ─── Record Schemas ──────────────────────────────────────────────────────────

class RecordListResponse(BaseModel):
    id: int
    ctc_number: str
    applicant: ApplicantResponse
    total_tax: float
    total_amount_paid: float
    input_method: str
    issued_at: datetime

    model_config = {"from_attributes": True}
