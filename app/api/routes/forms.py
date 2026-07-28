"""
app/api/routes/forms.py
CTC issuance and PDF generation endpoints.
"""
import os
from datetime import datetime, date, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.applicant import ApplicantRecord
from app.models.issuance import CTCIssuance, InputMethodEnum
from app.models.staff_user import StaffUser
from app.services.pdf_service import generate_cedula_pdf, generate_receipt_pdf
from app.services.tax_service import compute_tax
from app.schemas.schemas import (
    IssuanceCreate,
    IssuanceResponse,
    DashboardStats,
    ApplicantResponse,
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def parse_date(date_str):
    """Parse date string to date object."""
    if not date_str:
        return None
    if isinstance(date_str, date):
        return date_str
    try:
        return datetime.strptime(str(date_str), "%Y-%m-%d").date()
    except ValueError:
        try:
            return datetime.strptime(str(date_str), "%m/%d/%Y").date()
        except ValueError:
            try:
                from dateutil import parser
                return parser.parse(str(date_str)).date()
            except:
                return None


def get_sex_value(value):
    """Convert string to proper Sex value."""
    if not value:
        return None
    v = str(value).strip().lower()
    if v in ['male', 'm', '1']:
        return 'Male'
    if v in ['female', 'f', '2']:
        return 'Female'
    return value


def get_civil_status_value(value):
    """Convert string to proper Civil Status value."""
    if not value:
        return None
    v = str(value).strip().lower()
    if v in ['single', 's', '1']:
        return 'Single'
    if v in ['married', 'm', '2']:
        return 'Married'
    if v in ['widowed', 'widow', 'w', '3']:
        return 'Widowed'
    if v in ['separated', 'divorced', 'd', '4']:
        return 'Separated'
    return value


def generate_ctc_number(db: Session) -> str:
    """Generate a unique CTC number."""
    now = datetime.now()
    date_prefix = now.strftime("%Y%m%d")
    
    count = db.query(CTCIssuance).filter(
        CTCIssuance.ctc_number.like(f"CTC-{date_prefix}-%")
    ).count()
    serial = str(count + 1).zfill(4)
    return f"CTC-{date_prefix}-{serial}"


@router.post("/issue", response_model=IssuanceResponse, status_code=201)
def issue_ctc(
    payload: IssuanceCreate,
    db: Session = Depends(get_db),
    current_user: StaffUser = Depends(get_current_user),
):
    """
    Full CTC issuance workflow.
    """
    try:
        # ── 1. Extract Applicant Data ──────────────────────────────────────
        if isinstance(payload.applicant_data, dict):
            applicant_data = payload.applicant_data
        else:
            applicant_data = payload.applicant_data.model_dump()
        
        logger.info(f"Processing CTC for: {applicant_data.get('surname')}, {applicant_data.get('given_name')}")

        # ── 2. Create Applicant ──────────────────────────────────────────────
        sex_value = get_sex_value(applicant_data.get("sex"))
        civil_status_value = get_civil_status_value(applicant_data.get("civil_status"))
        date_of_birth = parse_date(applicant_data.get("date_of_birth"))

        applicant = ApplicantRecord(
            surname=applicant_data.get("surname", "").strip(),
            given_name=applicant_data.get("given_name", "").strip(),
            middle_name=applicant_data.get("middle_name") or None,
            date_of_birth=date_of_birth,
            address=applicant_data.get("address", "").strip(),
            sex=sex_value,
            civil_status=civil_status_value,
            citizenship=applicant_data.get("citizenship", "Filipino").strip(),
            icr_number=applicant_data.get("icr_number") or None,
            place_of_birth=applicant_data.get("place_of_birth") or None,
            height_cm=applicant_data.get("height_cm") or None,
            weight_kg=applicant_data.get("weight_kg") or None,
            occupation=applicant_data.get("occupation") or None,
            gross_annual_income=float(applicant_data.get("gross_annual_income", 0.0) or 0.0),
            tax_classification=applicant_data.get("tax_classification", "A"),
        )

        db.add(applicant)
        db.flush()

        # ── 3. Compute Tax ──────────────────────────────────────────────────
        income = float(applicant.gross_annual_income or 0.0)
        tax_result = compute_tax(income)

        basic_tax = 5.00
        additional_tax = tax_result.get("additional_tax", 0.0)
        total_tax = basic_tax + additional_tax
        interest = 0.00
        total_amount_paid = total_tax + interest

        # ── 4. Generate CTC Number ──────────────────────────────────────────
        ctc_number = generate_ctc_number(db)

        # ── 5. Generate PDF ──────────────────────────────────────────────────
        applicant_dict = {
            "surname": applicant.surname,
            "given_name": applicant.given_name,
            "middle_name": applicant.middle_name,
            "address": applicant.address,
            "sex": applicant.sex or "",
            "civil_status": applicant.civil_status or "",
            "citizenship": applicant.citizenship,
            "icr_number": applicant.icr_number,
            "place_of_birth": applicant.place_of_birth,
            "height_cm": applicant.height_cm,
            "weight_kg": applicant.weight_kg,
            "occupation": applicant.occupation,
            "gross_annual_income": applicant.gross_annual_income,
            "tax_classification": applicant.tax_classification,
            "date_of_birth": applicant.date_of_birth.isoformat() if applicant.date_of_birth else None,
        }

        tax_dict = {
            "basic_community_tax": basic_tax,
            "additional_community_tax": additional_tax,
            "total_tax": total_tax,
            "interest": interest,
            "total_amount_paid": total_amount_paid,
        }

        try:
            pdf_path = generate_cedula_pdf(ctc_number, applicant_dict, tax_dict)
            receipt_path = generate_receipt_pdf(ctc_number, applicant_dict, tax_dict)
        except Exception as e:
            db.rollback()
            logger.error(f"PDF generation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"PDF generation failed: {str(e)}",
            )

        # ── 6. Save Issuance ──────────────────────────────────────────────────
        input_method = InputMethodEnum.VOICE
        if payload.input_method == "qr":
            input_method = InputMethodEnum.QR
        elif payload.input_method == "manual":
            input_method = InputMethodEnum.MANUAL

        issuance = CTCIssuance(
            ctc_number=ctc_number,
            applicant_id=applicant.id,
            basic_community_tax=basic_tax,
            additional_community_tax=additional_tax,
            total_tax=total_tax,
            interest=interest,
            total_amount_paid=total_amount_paid,
            input_method=input_method,
            raw_transcript=payload.raw_transcript,
            pdf_path=pdf_path,
            issued_by=current_user.id,
        )
        
        db.add(issuance)
        db.commit()
        db.refresh(issuance)

        logger.info(f"CTC {ctc_number} issued successfully")

        # ── 7. Build Response ──────────────────────────────────────────────────
        applicant_response = ApplicantResponse(
            id=applicant.id,
            surname=applicant.surname,
            given_name=applicant.given_name,
            middle_name=applicant.middle_name,
            date_of_birth=applicant.date_of_birth,
            address=applicant.address,
            sex=applicant.sex,
            civil_status=applicant.civil_status,
            citizenship=applicant.citizenship,
            icr_number=applicant.icr_number,
            place_of_birth=applicant.place_of_birth,
            height_cm=applicant.height_cm,
            weight_kg=applicant.weight_kg,
            occupation=applicant.occupation,
            gross_annual_income=applicant.gross_annual_income,
            tax_classification=applicant.tax_classification,
            created_at=applicant.created_at,
        )

        return IssuanceResponse(
            id=issuance.id,
            ctc_number=issuance.ctc_number,
            applicant_id=issuance.applicant_id,
            basic_community_tax=issuance.basic_community_tax,
            additional_community_tax=issuance.additional_community_tax,
            total_tax=issuance.total_tax,
            interest=issuance.interest,
            total_amount_paid=issuance.total_amount_paid,
            input_method=issuance.input_method.value,
            issued_at=issuance.issued_at,
            pdf_path=issuance.pdf_path,
            applicant=applicant_response,
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error issuing CTC: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/download/{ctc_number}")
def download_cedula(
    ctc_number: str,
    db: Session = Depends(get_db),
    current_user: StaffUser = Depends(get_current_user),
):
    """Download the generated PDF for a specific CTC number."""
    issuance = db.query(CTCIssuance).filter(
        CTCIssuance.ctc_number == ctc_number
    ).first()

    if not issuance:
        raise HTTPException(status_code=404, detail=f"CTC '{ctc_number}' not found.")

    if not issuance.pdf_path or not os.path.exists(issuance.pdf_path):
        raise HTTPException(
            status_code=404,
            detail="PDF file not found.",
        )

    return FileResponse(
        issuance.pdf_path,
        media_type="application/pdf",
        filename=f"{ctc_number}.pdf",
    )


@router.get("/records", response_model=List[IssuanceResponse])
def list_records(
    skip: int = 0,
    limit: int = 50,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: StaffUser = Depends(get_current_user),
):
    """List all CTC issuance records."""
    query = db.query(CTCIssuance).join(ApplicantRecord)

    if search:
        search_term = f"%{search}%"
        query = query.filter(
            ApplicantRecord.surname.ilike(search_term) |
            ApplicantRecord.given_name.ilike(search_term) |
            CTCIssuance.ctc_number.ilike(search_term)
        )

    records = (
        query
        .order_by(CTCIssuance.issued_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    result = []
    for issuance in records:
        applicant = issuance.applicant
        applicant_response = ApplicantResponse(
            id=applicant.id,
            surname=applicant.surname,
            given_name=applicant.given_name,
            middle_name=applicant.middle_name,
            date_of_birth=applicant.date_of_birth,
            address=applicant.address,
            sex=applicant.sex,
            civil_status=applicant.civil_status,
            citizenship=applicant.citizenship,
            icr_number=applicant.icr_number,
            place_of_birth=applicant.place_of_birth,
            height_cm=applicant.height_cm,
            weight_kg=applicant.weight_kg,
            occupation=applicant.occupation,
            gross_annual_income=applicant.gross_annual_income,
            tax_classification=applicant.tax_classification,
            created_at=applicant.created_at,
        )
        
        result.append(IssuanceResponse(
            id=issuance.id,
            ctc_number=issuance.ctc_number,
            applicant_id=issuance.applicant_id,
            basic_community_tax=issuance.basic_community_tax,
            additional_community_tax=issuance.additional_community_tax,
            total_tax=issuance.total_tax,
            interest=issuance.interest,
            total_amount_paid=issuance.total_amount_paid,
            input_method=issuance.input_method.value,
            issued_at=issuance.issued_at,
            pdf_path=issuance.pdf_path,
            applicant=applicant_response,
        ))
    
    return result


@router.get("/records/{issuance_id}", response_model=IssuanceResponse)
def get_record(
    issuance_id: int,
    db: Session = Depends(get_db),
    current_user: StaffUser = Depends(get_current_user),
):
    """Get a single CTC issuance by ID."""
    issuance = db.query(CTCIssuance).filter(CTCIssuance.id == issuance_id).first()
    if not issuance:
        raise HTTPException(status_code=404, detail="Issuance record not found.")
    
    applicant = issuance.applicant
    applicant_response = ApplicantResponse(
        id=applicant.id,
        surname=applicant.surname,
        given_name=applicant.given_name,
        middle_name=applicant.middle_name,
        date_of_birth=applicant.date_of_birth,
        address=applicant.address,
        sex=applicant.sex,
        civil_status=applicant.civil_status,
        citizenship=applicant.citizenship,
        icr_number=applicant.icr_number,
        place_of_birth=applicant.place_of_birth,
        height_cm=applicant.height_cm,
        weight_kg=applicant.weight_kg,
        occupation=applicant.occupation,
        gross_annual_income=applicant.gross_annual_income,
        tax_classification=applicant.tax_classification,
        created_at=applicant.created_at,
    )
    
    return IssuanceResponse(
        id=issuance.id,
        ctc_number=issuance.ctc_number,
        applicant_id=issuance.applicant_id,
        basic_community_tax=issuance.basic_community_tax,
        additional_community_tax=issuance.additional_community_tax,
        total_tax=issuance.total_tax,
        interest=issuance.interest,
        total_amount_paid=issuance.total_amount_paid,
        input_method=issuance.input_method.value,
        issued_at=issuance.issued_at,
        pdf_path=issuance.pdf_path,
        applicant=applicant_response,
    )


@router.get("/dashboard/stats", response_model=DashboardStats)
def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: StaffUser = Depends(get_current_user),
):
    """Return dashboard statistics."""
    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(today, datetime.max.time())

    total_issued = db.query(func.count(CTCIssuance.id)).scalar() or 0
    today_issued = db.query(func.count(CTCIssuance.id)).filter(
        CTCIssuance.issued_at.between(today_start, today_end)
    ).scalar() or 0
    
    voice_processed = db.query(func.count(CTCIssuance.id)).filter(
        CTCIssuance.input_method == InputMethodEnum.VOICE
    ).scalar() or 0
    
    qr_processed = db.query(func.count(CTCIssuance.id)).filter(
        CTCIssuance.input_method == InputMethodEnum.QR
    ).scalar() or 0

    # Last 7 days
    daily_data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        day_start = datetime.combine(day, datetime.min.time())
        day_end = datetime.combine(day, datetime.max.time())
        count = db.query(func.count(CTCIssuance.id)).filter(
            CTCIssuance.issued_at.between(day_start, day_end)
        ).scalar() or 0
        daily_data.append({
            "date": day.strftime("%Y-%m-%d"),
            "count": count
        })

    return DashboardStats(
        total_issued=total_issued,
        today_issued=today_issued,
        voice_processed=voice_processed,
        qr_processed=qr_processed,
        daily_data=daily_data,
    )