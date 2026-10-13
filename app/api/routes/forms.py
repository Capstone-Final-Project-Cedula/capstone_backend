"""
app/api/routes/forms.py
CTC issuance and PDF generation endpoints.
"""
import os
from datetime import datetime, date, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from typing import Optional, List
from collections import Counter

from app.core.config import settings
from app.core.database import get_db
from app.core.security import get_current_user, require_admin, require_staff
from app.models.applicant import ApplicantRecord
from app.models.issuance import CTCIssuance, InputMethodEnum
from app.models.staff_user import StaffUser
from app.services import name_lexicon, storage_service
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

LOCAL_TZ = timezone(timedelta(hours=settings.LOCAL_UTC_OFFSET_HOURS))


def local_today() -> date:
    """Today's date in barangay local time, regardless of the device's clock zone."""
    return datetime.now(LOCAL_TZ).date()


def local_day_start_utc(day: date) -> datetime:
    """
    Midnight of a local calendar day, as a naive UTC datetime — the same form
    issued_at is compared in (the Postgres session time zone is pinned to UTC).
    """
    local_midnight = datetime.combine(day, datetime.min.time(), tzinfo=LOCAL_TZ)
    return local_midnight.astimezone(timezone.utc).replace(tzinfo=None)


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
    """
    Generate a unique CTC number.
    Format: CTC-YYYYMMDD-XXXX — the date shows exactly when it was issued,
    but the serial (XXXX) counts issuances for the current YEAR and resets
    to 0001 on January 1st, rather than resetting every single day.

    The serial is the year's highest existing serial + 1 (not a row count, which
    re-used numbers after a record was deleted). On Postgres an advisory lock,
    held until the issuing transaction ends, keeps two devices from taking the
    same number at the same time.
    """
    now = datetime.now(LOCAL_TZ)
    year_prefix = now.strftime("%Y")
    date_prefix = now.strftime("%Y%m%d")

    if db.get_bind().dialect.name == "postgresql":
        db.execute(text("SELECT pg_advisory_xact_lock(hashtext('ctc_issuance.ctc_number'))"))

    existing = db.query(CTCIssuance.ctc_number).filter(
        CTCIssuance.ctc_number.like(f"CTC-{year_prefix}%")
    ).all()
    serials = [
        int(number.rsplit("-", 1)[-1])
        for (number,) in existing
        if number.rsplit("-", 1)[-1].isdigit()
    ]
    serial = str(max(serials, default=0) + 1).zfill(4)
    return f"CTC-{date_prefix}-{serial}"


def extract_barangay(address: Optional[str]) -> str:
    """Infer the barangay/location label from a free-form address."""
    if not address:
        return "Unspecified"

    parts = [part.strip() for part in address.split(",") if part.strip()]
    for part in parts:
        lower_part = part.lower()
        if "barangay" in lower_part or lower_part.startswith("brgy"):
            cleaned = part.replace("Brgy.", "", 1).replace("Brgy", "", 1).strip()
            if cleaned.lower().startswith("barangay"):
                return cleaned
            return f"Barangay {cleaned}"

    return parts[0] if parts else "Unspecified"


@router.post("/issue", response_model=IssuanceResponse, status_code=201)
def issue_ctc(
    payload: IssuanceCreate,
    db: Session = Depends(get_db),
    current_user: StaffUser = Depends(require_staff),
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

        basic_tax = tax_result["basic_community_tax"]
        additional_tax = tax_result["additional_community_tax"]
        total_tax = tax_result["total_tax"]
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

        # Share the PDF with every device through Supabase Storage. If the
        # upload fails the issuance still goes through with the local path,
        # downloadable from this computer only.
        try:
            pdf_path = storage_service.store_pdf(pdf_path)
        except Exception as e:
            logger.warning(f"PDF upload for {ctc_number} failed, keeping local copy: {e}")

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
        name_lexicon.refresh_records()   # the next interview knows this applicant's names

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

    if storage_service.is_storage_ref(issuance.pdf_path):
        try:
            content, _ = storage_service.download_bytes(issuance.pdf_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="PDF file not found.")
        except Exception as e:
            logger.error(f"PDF download for {ctc_number} failed: {e}")
            raise HTTPException(status_code=502, detail="Could not fetch the PDF from storage.")
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{ctc_number}.pdf"'},
        )

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
    limit: int = 10,
    search: Optional[str] = None,
    input_method: Optional[InputMethodEnum] = None,
    response: Response = None,
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

    if input_method:
        query = query.filter(CTCIssuance.input_method == input_method)

    # Keep the response body backward-compatible while exposing the count
    # required by the numbered pagination UI.
    total = query.count()
    response.headers["X-Total-Count"] = str(total)

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


@router.delete("/records/{issuance_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_record(
    issuance_id: int,
    db: Session = Depends(get_db),
    admin: StaffUser = Depends(require_admin),
):
    """Permanently delete an issuance and its applicant record (admin only)."""
    issuance = db.query(CTCIssuance).filter(CTCIssuance.id == issuance_id).first()
    if not issuance:
        raise HTTPException(status_code=404, detail="Issuance record not found.")

    pdf_path = issuance.pdf_path
    applicant = issuance.applicant
    db.delete(issuance)
    if applicant:
        db.delete(applicant)
    db.commit()

    if storage_service.is_storage_ref(pdf_path):
        storage_service.delete_object(pdf_path)
    elif pdf_path and os.path.isfile(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError:
            logger.warning("Record %s was deleted but its PDF could not be removed.", issuance_id)


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
    current_user: StaffUser = Depends(require_admin),
):
    """Return dashboard statistics."""
    # Day/month boundaries are barangay-local (UTC+8) midnights converted to UTC,
    # so "today" matches the clerk's calendar instead of the UTC date.
    today = local_today()
    today_start = local_day_start_utc(today)
    today_end = local_day_start_utc(today + timedelta(days=1))
    month_start = today.replace(day=1)
    month_start_dt = local_day_start_utc(month_start)

    total_issued = db.query(func.count(CTCIssuance.id)).scalar() or 0
    today_issued = db.query(func.count(CTCIssuance.id)).filter(
        CTCIssuance.issued_at >= today_start,
        CTCIssuance.issued_at < today_end,
    ).scalar() or 0

    monthly_count = db.query(func.count(CTCIssuance.id)).filter(
        CTCIssuance.issued_at >= month_start_dt
    ).scalar() or 0

    total_revenue = db.query(func.coalesce(func.sum(CTCIssuance.total_amount_paid), 0.0)).scalar() or 0.0
    today_revenue = db.query(func.coalesce(func.sum(CTCIssuance.total_amount_paid), 0.0)).filter(
        CTCIssuance.issued_at >= today_start,
        CTCIssuance.issued_at < today_end,
    ).scalar() or 0.0
    average_paid = (float(total_revenue) / total_issued) if total_issued else 0.0
    
    voice_processed = db.query(func.count(CTCIssuance.id)).filter(
        CTCIssuance.input_method == InputMethodEnum.VOICE
    ).scalar() or 0
    
    qr_processed = db.query(func.count(CTCIssuance.id)).filter(
        CTCIssuance.input_method == InputMethodEnum.QR
    ).scalar() or 0

    manual_processed = db.query(func.count(CTCIssuance.id)).filter(
        CTCIssuance.input_method == InputMethodEnum.MANUAL
    ).scalar() or 0

    # Last 7 days
    daily_data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        day_start = local_day_start_utc(day)
        day_end = local_day_start_utc(day + timedelta(days=1))
        count = db.query(func.count(CTCIssuance.id)).filter(
            CTCIssuance.issued_at >= day_start,
            CTCIssuance.issued_at < day_end,
        ).scalar() or 0
        revenue = db.query(func.coalesce(func.sum(CTCIssuance.total_amount_paid), 0.0)).filter(
            CTCIssuance.issued_at >= day_start,
            CTCIssuance.issued_at < day_end,
        ).scalar() or 0.0
        daily_data.append({
            "date": day.strftime("%Y-%m-%d"),
            "count": count,
            "revenue": round(float(revenue), 2),
        })

    monthly_data = []
    for i in range(5, -1, -1):
        month = today.month - i
        year = today.year
        while month <= 0:
            month += 12
            year -= 1

        period_start = date(year, month, 1)
        if month == 12:
            period_end = date(year + 1, 1, 1)
        else:
            period_end = date(year, month + 1, 1)

        period_start_dt = local_day_start_utc(period_start)
        period_end_dt = local_day_start_utc(period_end)
        count = db.query(func.count(CTCIssuance.id)).filter(
            CTCIssuance.issued_at >= period_start_dt,
            CTCIssuance.issued_at < period_end_dt,
        ).scalar() or 0
        revenue = db.query(func.coalesce(func.sum(CTCIssuance.total_amount_paid), 0.0)).filter(
            CTCIssuance.issued_at >= period_start_dt,
            CTCIssuance.issued_at < period_end_dt,
        ).scalar() or 0.0
        monthly_data.append({
            "month": period_start.strftime("%b %Y"),
            "count": count,
            "revenue": round(float(revenue), 2),
        })

    input_method_breakdown = []
    method_labels = {
        InputMethodEnum.VOICE: "Voice",
        InputMethodEnum.QR: "QR",
        InputMethodEnum.MANUAL: "Manual",
    }
    for method, label in method_labels.items():
        count = db.query(func.count(CTCIssuance.id)).filter(
            CTCIssuance.input_method == method
        ).scalar() or 0
        input_method_breakdown.append({"method": label, "count": count})

    tax_rows = (
        db.query(ApplicantRecord.tax_classification, func.count(ApplicantRecord.id))
        .join(CTCIssuance, CTCIssuance.applicant_id == ApplicantRecord.id)
        .group_by(ApplicantRecord.tax_classification)
        .all()
    )
    tax_classification_breakdown = [
        {
            "classification": getattr(classification, "value", classification) or "Unspecified",
            "count": count,
        }
        for classification, count in tax_rows
    ]

    address_rows = (
        db.query(ApplicantRecord.address)
        .join(CTCIssuance, CTCIssuance.applicant_id == ApplicantRecord.id)
        .all()
    )
    barangay_counts = Counter(extract_barangay(row[0]) for row in address_rows)
    top_barangays = [
        {"barangay": barangay, "count": count}
        for barangay, count in barangay_counts.most_common(5)
    ]

    recent_records = (
        db.query(CTCIssuance)
        .join(ApplicantRecord)
        .order_by(CTCIssuance.issued_at.desc())
        .limit(5)
        .all()
    )
    recent_activity = []
    for issuance in recent_records:
        applicant = issuance.applicant
        recent_activity.append({
            "id": issuance.id,
            "ctc_number": issuance.ctc_number,
            "applicant_name": applicant.full_name() if applicant else "Unknown applicant",
            "input_method": issuance.input_method.value,
            "amount_paid": round(float(issuance.total_amount_paid or 0.0), 2),
            "issued_at": issuance.issued_at.isoformat() if issuance.issued_at else None,
        })

    return DashboardStats(
        total_issued=total_issued,
        today_issued=today_issued,
        voice_processed=voice_processed,
        qr_processed=qr_processed,
        daily_data=daily_data,
        manual_processed=manual_processed,
        total_revenue=round(float(total_revenue), 2),
        today_revenue=round(float(today_revenue), 2),
        average_paid=round(float(average_paid), 2),
        monthly_data=monthly_data,
        input_method_breakdown=input_method_breakdown,
        tax_classification_breakdown=tax_classification_breakdown,
        top_barangays=top_barangays,
        recent_activity=recent_activity,
        daily_count=today_issued,
        monthly_count=monthly_count,
        total_count=total_issued,
    )
