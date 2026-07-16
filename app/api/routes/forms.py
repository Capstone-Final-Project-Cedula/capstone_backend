"""
app/api/routes/forms.py
CTC issuance and PDF generation endpoints.
POST /api/forms/issue       — create a full CTC issuance (applicant + tax + PDF)
GET  /api/forms/download    — download a generated Cedula PDF
GET  /api/records           — list all issuance records
GET  /api/records/{id}      — get single issuance
"""
import os
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.applicant import ApplicantRecord
from app.models.issuance import CTCIssuance, InputMethodEnum
from app.services.pdf_service import generate_cedula_pdf, generate_receipt_pdf
from app.services.tax_service import compute_tax, generate_ctc_number
from app.schemas.schemas import IssuanceCreate, IssuanceResponse, DashboardStats

router = APIRouter()


@router.post("/issue", response_model=IssuanceResponse, status_code=201)
def issue_ctc(
    payload: IssuanceCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Full CTC issuance workflow:
    1. Save applicant record
    2. Compute tax
    3. Generate CTC number
    4. Generate PDF
    5. Save issuance record
    6. Return full issuance response
    """
    # Step 1: Save applicant
    applicant_data = payload.applicant_data.model_dump()
    applicant = ApplicantRecord(**applicant_data)
    db.add(applicant)
    db.flush()  # Get the ID without committing yet

    # Step 2: Compute tax
    tax_result = compute_tax(
        gross_annual_income=applicant.gross_annual_income or 0.0,
    )

    # Step 3: Generate CTC number
    ctc_number = generate_ctc_number(db)

    # Step 4: Generate PDF
    try:
        pdf_path = generate_cedula_pdf(
            ctc_number=ctc_number,
            applicant_data=applicant_data,
            tax_data=tax_result,
        )
        generate_receipt_pdf(
            ctc_number=ctc_number,
            applicant_data=applicant_data,
            tax_data=tax_result,
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF generation failed: {str(e)}",
        )

    # Step 5: Save issuance record
    issuance = CTCIssuance(
        ctc_number=ctc_number,
        basic_community_tax=tax_result["basic_community_tax"],
        additional_community_tax=tax_result["additional_community_tax"],
        total_tax=tax_result["total_tax"],
        input_method=payload.input_method,
        raw_transcript=payload.raw_transcript,
        pdf_path=pdf_path,
        applicant_id=applicant.id,
        issued_by=current_user.id,
    )
    db.add(issuance)
    db.commit()
    db.refresh(issuance)

    return issuance


@router.get("/download/{ctc_number}")
def download_cedula(
    ctc_number: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
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
            detail="PDF file not found. It may have been deleted or not yet generated.",
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
    current_user=Depends(get_current_user),
):
    """
    List all CTC issuance records.
    Optional search by applicant surname or CTC number.
    """
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

    return records


@router.get("/records/{issuance_id}", response_model=IssuanceResponse)
def get_record(
    issuance_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get a single CTC issuance by ID."""
    issuance = db.query(CTCIssuance).filter(CTCIssuance.id == issuance_id).first()
    if not issuance:
        raise HTTPException(status_code=404, detail="Issuance record not found.")
    return issuance


@router.get("/dashboard/stats", response_model=DashboardStats)
def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Return daily / monthly / total CTC issuance counts and tax totals.
    Used by the desktop dashboard home screen.
    """
    from datetime import date
    from sqlalchemy import func

    today = date.today()
    month_start = today.replace(day=1)

    # Daily count
    daily_count = (
        db.query(func.count(CTCIssuance.id))
        .filter(func.date(CTCIssuance.issued_at) == today)
        .scalar() or 0
    )

    # Monthly count
    monthly_count = (
        db.query(func.count(CTCIssuance.id))
        .filter(func.date(CTCIssuance.issued_at) >= month_start)
        .scalar() or 0
    )

    # Total count
    total_count = db.query(func.count(CTCIssuance.id)).scalar() or 0

    # Tax collected today
    tax_today = (
        db.query(func.sum(CTCIssuance.total_tax))
        .filter(func.date(CTCIssuance.issued_at) == today)
        .scalar() or 0.0
    )

    # Tax collected this month
    tax_month = (
        db.query(func.sum(CTCIssuance.total_tax))
        .filter(func.date(CTCIssuance.issued_at) >= month_start)
        .scalar() or 0.0
    )

    # Recent 5 issuances
    recent = (
        db.query(CTCIssuance)
        .order_by(CTCIssuance.issued_at.desc())
        .limit(5)
        .all()
    )

    return DashboardStats(
        daily_count=daily_count,
        monthly_count=monthly_count,
        total_count=total_count,
        total_tax_collected_today=float(tax_today),
        total_tax_collected_month=float(tax_month),
        recent_issuances=recent,
    )
