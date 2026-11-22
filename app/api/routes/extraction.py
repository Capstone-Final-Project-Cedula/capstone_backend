"""
app/api/routes/extraction.py
NLP field extraction endpoints.
POST /api/extraction/fields   — extract BIR Form 0016 fields from transcript
POST /api/extraction/answer   — understand one interview answer (live interview)
POST /api/extraction/answers  — check every interview answer before issuing
POST /api/extraction/tax      — compute community tax from income
"""
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.security import get_current_user
from app.services.answer_service import understand_answer, understand_answers
from app.services.extraction_service import extract_fields
from app.services.tax_service import compute_tax
from app.schemas.schemas import (
    ExtractionRequest, ExtractionResponse,
    TaxComputationResult, ApplicantCreate
)
from app.models.applicant import TaxClassificationEnum

router = APIRouter()

InterviewField = Literal[
    "surname", "given_name", "middle_name", "address", "date", "sex", "civil_status",
    "citizenship", "icr_no", "place_of_birth", "height", "weight", "occupation", "gross_annual_income",
    "address_town", "address_barangay", "address_purok",   # the address in 3 questions (plan 001.6)
    "place_of_birth_province", "place_of_birth_town",      # the place of birth in 2 steps (plan 001.8)
]


class AnswerRequest(BaseModel):
    field_key: InterviewField
    transcript: str = Field("", max_length=500)
    spoken: bool = False   # True: Whisper's transcript in the voice interview (names may be respelled)
    town: Optional[str] = Field(None, pattern=r"^\d{0,10}$")   # PSGC code of the town, for "address_barangay"
    province: Optional[str] = Field(None, pattern=r"^\d{0,10}$")   # PSGC code of the province, "place_of_birth_town"


class AnswerResponse(BaseModel):
    field_key: str
    ok: bool
    value: Optional[str] = None
    display: Optional[str] = None
    normalized: Optional[str] = None
    message: Optional[str] = None
    reason: Optional[str] = None
    needs_confirmation: bool = False
    methods: List[str] = []
    confidence: float = 0.0
    warnings: List[str] = []
    suggestions: List[Dict[str, str]] = []   # place of birth: the closest list entries to tap (plan 001.8)
    candidates: List[Dict[str, str]] = []    # reason "ambiguous_town": every province with that town (plan 006)


class AnswersRequest(BaseModel):
    answers: Dict[str, Optional[str]]


class AnswersResponse(BaseModel):
    ok: bool
    results: Dict[str, AnswerResponse]


@router.post("/answer", response_model=AnswerResponse)
def understand_interview_answer(payload: AnswerRequest, current_user=Depends(get_current_user)):
    """
    Understand ONE answer of the live interview (DistilBERT NER for names and
    places, regex for dates, numbers, units and choices). Returns the
    normalized value, or ok=false with a short message to say before asking
    again — e.g. "nails" for sex is rejected instead of saved.
    """
    result = understand_answer(payload.field_key, payload.transcript, spoken=payload.spoken, town=payload.town or None,
                               province=payload.province or None)
    return AnswerResponse(**result.as_dict())


@router.post("/answers", response_model=AnswersResponse)
def check_interview_answers(payload: AnswersRequest, current_user=Depends(get_current_user)):
    """Re-check every answer (including manual edits on the review page) before issuing."""
    results = {k: AnswerResponse(**r.as_dict()) for k, r in understand_answers(payload.answers).items()}
    return AnswersResponse(ok=all(r.ok for r in results.values()), results=results)


@router.post("/fields", response_model=ExtractionResponse)
def extract_form_fields(
    payload: ExtractionRequest,
    current_user=Depends(get_current_user),
):
    """
    Extract all BIR Form 0016 fields from a Whisper transcript.

    Uses DistilBERT NER + Regex dual-approach extraction.
    Returns extracted fields, per-field confidence scores, and
    a list of fields that could not be automatically extracted.
    """
    if not payload.transcript or not payload.transcript.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript is empty. Please provide text to extract from.",
        )

    try:
        result = extract_fields(payload.transcript)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}",
        )

    # Build the ApplicantCreate from extracted fields
    # Fill in defaults for required fields if missing
    fields = result["extracted_fields"]
    if not fields.get("surname"):
        fields["surname"] = "Unknown"
    if not fields.get("given_name"):
        fields["given_name"] = "Unknown"
    if not fields.get("address"):
        fields["address"] = "To be filled"

    # Auto-compute tax classification based on income
    income = fields.get("gross_annual_income") or 0.0
    if income > 0:
        fields["tax_classification"] = TaxClassificationEnum.A
    else:
        fields["tax_classification"] = TaxClassificationEnum.C

    try:
        applicant = ApplicantCreate(**fields)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Extracted data validation failed: {str(e)}",
        )

    return ExtractionResponse(
        extracted_fields=applicant,
        confidence_scores=result["confidence_scores"],
        unextracted_fields=result["unextracted_fields"],
        warnings=result["warnings"],
    )


@router.post("/tax", response_model=TaxComputationResult)
def compute_community_tax(
    gross_annual_income: float = 0.0,
    real_property_value: float = 0.0,
    current_user=Depends(get_current_user),
):
    """
    Compute community tax based on income and/or property value.

    Rules (RA 7160):
    - Basic: ₱5.00 flat
    - Additional: ₱1.00 per ₱1,000 of income or assessed property value
    - Maximum additional: ₱5,000.00
    """
    if gross_annual_income < 0 or real_property_value < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Income and property values must be non-negative.",
        )

    result = compute_tax(
        gross_annual_income=gross_annual_income,
        real_property_value=real_property_value,
    )

    return TaxComputationResult(**result)
