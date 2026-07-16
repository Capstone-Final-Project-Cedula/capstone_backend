"""
app/services/tax_service.py
BIR Community Tax computation logic based on Republic Act 7160 (Local Government Code).

Rules:
- Basic Community Tax: ₱5.00 (flat for all individuals)
- Additional Community Tax: ₱1.00 per ₱1,000 of:
    a. Gross annual income (from business, profession, or employment)
    b. Assessed value of real property owned
- Maximum Additional Tax: ₱5,000.00
- Combined Maximum: ₱5,005.00
"""
from typing import Optional
from app.models.applicant import TaxClassificationEnum


# ─── Tax Brackets ──────────────────────────────────────────────────────────────

BASIC_TAX = 5.00
ADDITIONAL_RATE = 0.001       # ₱1 per ₱1,000 = 0.001 multiplier
MAX_ADDITIONAL_TAX = 5_000.00


def compute_tax(
    gross_annual_income: Optional[float] = 0.0,
    real_property_value: Optional[float] = 0.0,
) -> dict:
    """
    Compute community tax for an individual.

    Args:
        gross_annual_income: Total gross income from all sources (PHP)
        real_property_value: Assessed value of real property owned (PHP)

    Returns:
        dict with basic, additional, total tax, classification, and breakdown text
    """
    income = gross_annual_income or 0.0
    property_val = real_property_value or 0.0

    # Additional tax is based on the higher of income or property value
    taxable_base = max(income, property_val)

    additional_tax = round(taxable_base * ADDITIONAL_RATE, 2)
    additional_tax = min(additional_tax, MAX_ADDITIONAL_TAX)  # Cap at ₱5,000

    total_tax = round(BASIC_TAX + additional_tax, 2)

    # Determine tax classification
    if income > 0 or property_val > 0:
        classification = TaxClassificationEnum.A   # With income or property
    else:
        classification = TaxClassificationEnum.C   # No income, no property

    # Human-readable breakdown
    breakdown_lines = [
        f"Basic Community Tax:        ₱{BASIC_TAX:.2f}",
        f"Gross Annual Income:        ₱{income:,.2f}",
    ]
    if property_val > 0:
        breakdown_lines.append(f"Real Property Value:        ₱{property_val:,.2f}")
    breakdown_lines += [
        f"Taxable Base (higher):      ₱{taxable_base:,.2f}",
        f"Additional Tax (×₱1/₱1K):  ₱{additional_tax:,.2f}",
        f"─────────────────────────────────────",
        f"TOTAL COMMUNITY TAX:        ₱{total_tax:,.2f}",
        f"Tax Classification:         {classification.value}",
    ]
    breakdown = "\n".join(breakdown_lines)

    return {
        "basic_community_tax": BASIC_TAX,
        "additional_community_tax": additional_tax,
        "total_tax": total_tax,
        "tax_classification": classification,
        "breakdown": breakdown,
    }


def generate_ctc_number(db_session, issuance_date=None) -> str:
    """
    Generate a unique CTC number.
    Format: CTC-YYYYMMDD-NNNN (NNNN = daily sequence number)

    Args:
        db_session: SQLAlchemy session for querying today's count
        issuance_date: Date object (defaults to today)

    Returns:
        CTC number string e.g. "CTC-20240115-0001"
    """
    from datetime import date
    from app.models.issuance import CTCIssuance
    from sqlalchemy import func, cast, Date

    today = issuance_date or date.today()
    date_str = today.strftime("%Y%m%d")

    # Count today's issuances to get sequence number
    today_count = (
        db_session.query(func.count(CTCIssuance.id))
        .filter(
            func.date(CTCIssuance.issued_at) == today
        )
        .scalar()
        or 0
    )

    sequence = today_count + 1
    return f"CTC-{date_str}-{sequence:04d}"
