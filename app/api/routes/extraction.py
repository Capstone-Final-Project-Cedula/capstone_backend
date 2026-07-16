"""
app/api/routes/extraction.py
NLP field extraction endpoints.
POST /api/extraction/fields   — extract BIR Form 0016 fields from transcript
POST /api/extraction/tax      — compute community tax from income
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import get_current_user
from app.services.extraction_service import extract_fields
from app.services.tax_service import compute_tax
from app.schemas.schemas import (
    ExtractionRequest, ExtractionResponse,
    TaxComputationResult, ApplicantCreate
)
from app.models.applicant import TaxClassificationEnum

router = APIRouter()


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
