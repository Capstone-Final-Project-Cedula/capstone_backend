"""
app/services/pdf_service.py
BIR Form 0016 (Community Tax Certificate / Cedula) PDF generation using ReportLab.
Replicates the official CTC card layout (YEAR / PLACE OF ISSUE / CCI No. header,
form-field grid, A/B tax computation block, thumbprint + signature footer) plus a
separate transaction receipt.
"""
import os
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

from app.core.config import settings

import logging
logger = logging.getLogger(__name__)

# ─── Colors ────────────────────────────────────────────────────────────────
BLACK = colors.black
GRAY = colors.HexColor("#666666")
LIGHT_GRAY = colors.HexColor("#EDEDED")
WHITE = colors.white

# ─── Card page size (matches the printed CTC card format) ──────────────────
CARD_WIDTH = 8.5 * inch
CARD_HEIGHT = 4.3 * inch
MARGIN = 0.18 * inch


def _ensure_output_dir():
    os.makedirs(settings.PDF_OUTPUT_DIR, exist_ok=True)


def _checkbox(c: pdfcanvas.Canvas, x, y, checked: bool, size=6.5):
    """Draw a small square checkbox, filled if checked."""
    c.setLineWidth(0.6)
    c.rect(x, y, size, size, stroke=1, fill=1 if checked else 0)
    if checked:
        c.setFillColor(BLACK)
        c.rect(x, y, size, size, stroke=1, fill=1)


def _field(c, x, y, w, label, value, label_size=5.3, value_size=8.2, pad=2.5):
    """Draw a bordered field cell with a small caption label and value text below it."""
    c.setFont("Helvetica", label_size)
    c.setFillColor(GRAY)
    c.drawString(x + pad, y - label_size - 1, label)
    c.setFillColor(BLACK)
    c.setFont("Helvetica-Bold", value_size)
    c.drawString(x + pad, y - label_size - value_size - 2, value if value else "")


def generate_cedula_pdf(
    ctc_number: str,
    applicant_data: dict,
    tax_data: dict,
    output_filename: Optional[str] = None,
) -> str:
    """
    Generate a BIR Form 0016 Community Tax Certificate (Cedula) PDF,
    laid out to match the official printed CTC card.

    Args:
        ctc_number: Unique CTC number, formatted as "CCI{year}  {serial}"
                    or any string; it is split/printed as-is.
        applicant_data: Dict with BIR Form 0016 applicant fields.
        tax_data: Dict with basic_community_tax, additional_community_tax,
                  total_tax, and optionally gross_receipts_business,
                  salaries_professional_income, income_real_property, interest.
        output_filename: Optional filename; auto-generated if None.

    Returns:
        Full path to the generated PDF file.
    """
    _ensure_output_dir()

    if not output_filename:
        output_filename = f"{ctc_number.replace('-', '_').replace(' ', '_')}.pdf"

    output_path = os.path.join(settings.PDF_OUTPUT_DIR, output_filename)

    a = applicant_data
    t = tax_data
    now = datetime.now()
    year_issued = str(a.get("year_issued") or now.year)
    place_of_issue = a.get("place_of_issue") or f"{settings.MUNICIPALITY}, {settings.PROVINCE}"
    date_issued = a.get("date_issued") or now.strftime("%m/%d/%Y")

    surname = a.get("surname", "")
    given_name = a.get("given_name", "")
    middle_name = a.get("middle_name", "") or ""
    address = a.get("address", "")
    tin = a.get("tin", "")
    sex = (a.get("sex") or "").strip().lower()
    citizenship = a.get("citizenship", "Filipino")
    icr_number = a.get("icr_number", "")
    place_of_birth = a.get("place_of_birth", "")
    height_cm = f"{a.get('height_cm')} cm" if a.get("height_cm") else ""
    weight_kg = f"{a.get('weight_kg')} kg" if a.get("weight_kg") else ""
    civil_status = (a.get("civil_status") or "").strip().lower()
    date_of_birth = str(a.get("date_of_birth", "")) if a.get("date_of_birth") else ""
    occupation = a.get("occupation", "")
    income = float(a.get("gross_annual_income") or 0.0)

    basic = t.get("basic_community_tax", 5.00)
    additional = t.get("additional_community_tax", 0.00)
    total = t.get("total_tax", 5.00)
    gross_business = t.get("gross_receipts_business", 0.00)
    salaries_income = t.get("salaries_professional_income", 0.00)
    real_property_income = t.get("income_real_property", 0.00)
    interest = t.get("interest", 0.00)
    total_paid = t.get("total_amount_paid", total)

    c = pdfcanvas.Canvas(output_path, pagesize=(CARD_WIDTH, CARD_HEIGHT))
    c.setTitle(f"Community Tax Certificate - {ctc_number}")
    c.setAuthor(settings.BARANGAY_NAME)

    x0, y0 = MARGIN, MARGIN
    x1, y1 = CARD_WIDTH - MARGIN, CARD_HEIGHT - MARGIN
    full_w = x1 - x0

    # ── Outer border ─────────────────────────────────────────────────────
    c.setLineWidth(1)
    c.rect(x0, y0, full_w, y1 - y0)

    # ── Top strip: BIR form ref | INDIVIDUAL | CCI No. | Taxpayer's copy ──
    top_h = 42
    top_y = y1
    c.setLineWidth(0.6)
    c.line(x0, top_y - top_h, x1, top_y - top_h)

    c.setFont("Helvetica", 6.5)
    c.setFillColor(BLACK)
    c.drawString(x0 + 4, top_y - 10, "BIR FORM 0015 [MARCH, 2000]")

    c.setFont("Helvetica-Bold", 13)
    c.drawString(x0 + 4, top_y - 30, "COMMUNITY TAX CERTIFICATE")

    # INDIVIDUAL box
    ind_w, ind_h = 62, 14
    ind_x = x0 + 205
    ind_y = top_y - 34
    c.setFillColor(BLACK)
    c.rect(ind_x, ind_y, ind_w, ind_h, stroke=1, fill=1)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawCentredString(ind_x + ind_w / 2, ind_y + 4, "INDIVIDUAL")
    c.setFillColor(BLACK)

    # CCI No. box (top right)
    cci_x = x0 + 280
    cci_w = x1 - cci_x
    c.rect(cci_x, top_y - top_h, cci_w, top_h)
    c.line(cci_x, top_y - 13, x1, top_y - 13)
    c.setFont("Helvetica", 6.5)
    c.drawCentredString(cci_x + cci_w / 2, top_y - 10, "TAXPAYER'S COPY")
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(cci_x + cci_w / 2, top_y - top_h + 14, ctc_number)

    # ── Grid: left field block + right field block ──────────────────────
    grid_top = top_y - top_h
    foot_h = 22
    tax_h = 78
    sig_h = 34
    grid_bottom = y0 + foot_h + tax_h + sig_h
    left_w = full_w * 0.62
    right_x = x0 + left_w
    c.line(right_x, grid_bottom, right_x, grid_top)

    # Row heights for the left block (7 stacked rows)
    n_left_rows = 6
    row_h = (grid_top - grid_bottom) / n_left_rows
    ys = [grid_top - i * row_h for i in range(n_left_rows + 1)]
    for yy in ys[1:-1]:
        c.line(x0, yy, right_x, yy)

    # Row 1: YEAR | PLACE OF ISSUE | DATE ISSUED
    seg_w = left_w / 3
    c.line(x0 + seg_w, ys[0], x0 + seg_w, ys[1])
    c.line(x0 + 2 * seg_w, ys[0], x0 + 2 * seg_w, ys[1])
    _field(c, x0, ys[0], seg_w, "YEAR", year_issued)
    _field(c, x0 + seg_w, ys[0], seg_w, "PLACE OF ISSUE (City/Mun/Prov)", place_of_issue, label_size=5, value_size=7)
    _field(c, x0 + 2 * seg_w, ys[0], seg_w, "DATE ISSUED", date_issued)

    # Row 2: SURNAME | FIRST | MIDDLE
    c.line(x0 + seg_w, ys[1], x0 + seg_w, ys[2])
    c.line(x0 + 2 * seg_w, ys[1], x0 + 2 * seg_w, ys[2])
    _field(c, x0, ys[1], seg_w, "SURNAME", surname)
    _field(c, x0 + seg_w, ys[1], seg_w, "FIRST", given_name)
    _field(c, x0 + 2 * seg_w, ys[1], seg_w, "MIDDLE", middle_name)

    # Row 3: ADDRESS (full width of left block)
    _field(c, x0, ys[2], left_w, "ADDRESS", address)

    # Row 4: CITIZENSHIP | ICR NO. (if an Alien) | PLACE OF BIRTH
    c.line(x0 + seg_w, ys[3], x0 + seg_w, ys[4])
    c.line(x0 + 2 * seg_w, ys[3], x0 + 2 * seg_w, ys[4])
    _field(c, x0, ys[3], seg_w, "CITIZENSHIP", citizenship)
    _field(c, x0 + seg_w, ys[3], seg_w, "ICR NO. (if an Alien)", icr_number)
    _field(c, x0 + 2 * seg_w, ys[3], seg_w, "PLACE OF BIRTH", place_of_birth)

    # Row 5: CIVIL STATUS checkboxes
    c.setFont("Helvetica", 5.3)
    c.setFillColor(GRAY)
    c.drawString(x0 + 3, ys[4] - 6, "CIVIL STATUS")
    c.setFillColor(BLACK)
    statuses = [("Single", "single"), ("Married", "married"),
                ("Widow/Widower", "widow"), ("Legally Separated", "legally separated")]
    cx = x0 + 3
    cy = ys[4] - 17
    c.setFont("Helvetica", 6.3)
    for label, key in statuses:
        checked = key in civil_status
        _checkbox(c, cx, cy, checked, size=6)
        c.drawString(cx + 9, cy + 1, label)
        cx += 8 * len(label) * 0.5 + 22

    # Row 6: PROFESSION / OCCUPATION / BUSINESS
    _field(c, x0, ys[5], left_w, "PROFESSION / OCCUPATION / BUSINESS", occupation)

    # ── Right block: TIN | SEX, HEIGHT, WEIGHT, DATE OF BIRTH, TAXABLE AMOUNT, TAX DUE
    r_rows = 6
    r_row_h = (grid_top - grid_bottom) / r_rows
    rys = [grid_top - i * r_row_h for i in range(r_rows + 1)]
    for yy in rys[1:-1]:
        c.line(right_x, yy, x1, yy)

    # Row 1: TIN + SEX checkboxes
    tin_w = (x1 - right_x) * 0.62
    c.line(right_x + tin_w, rys[0], right_x + tin_w, rys[1])
    _field(c, right_x, rys[0], tin_w, "TIN (If any)", tin, label_size=5, value_size=7)
    c.setFont("Helvetica", 5.3)
    c.setFillColor(GRAY)
    c.drawString(right_x + tin_w + 3, rys[0] - 6, "SEX")
    c.setFillColor(BLACK)
    c.setFont("Helvetica", 6)
    _checkbox(c, right_x + tin_w + 3, rys[0] - 13, sex.startswith("m"), size=5.5)
    c.drawString(right_x + tin_w + 11, rys[0] - 12, "MALE")
    _checkbox(c, right_x + tin_w + 3, rys[0] - 21, sex.startswith("f"), size=5.5)
    c.drawString(right_x + tin_w + 11, rys[0] - 20, "FEMALE")

    _field(c, right_x, rys[1], x1 - right_x, "HEIGHT", height_cm)
    _field(c, right_x, rys[2], x1 - right_x, "WEIGHT", weight_kg)
    _field(c, right_x, rys[3], x1 - right_x, "DATE OF BIRTH", date_of_birth)
    _field(c, right_x, rys[4], x1 - right_x, "TAXABLE AMOUNT", f"P {income:,.2f}")
    _field(c, right_x, rys[5], x1 - right_x, "COMMUNITY TAX DUE", f"P {total:,.2f}")

    # ── Right Thumb Print / Taxpayer's Signature / Treasurer strip ───────
    sig_top = grid_bottom
    sig_bottom = sig_top - sig_h
    c.setLineWidth(0.6)
    c.rect(x0, sig_bottom, full_w, sig_h)
    thumb_w = 70
    sig_line_x = x0 + thumb_w
    sig_line_w = full_w * 0.38
    c.line(sig_line_x, sig_bottom, sig_line_x, sig_top)
    treas_x = sig_line_x + sig_line_w
    c.line(treas_x, sig_bottom, treas_x, sig_top)

    c.setFont("Helvetica", 6)
    c.setFillColor(GRAY)
    c.drawCentredString(x0 + thumb_w / 2, sig_bottom + sig_h - 10, "RIGHT THUMB")
    c.drawCentredString(x0 + thumb_w / 2, sig_bottom + sig_h - 18, "PRINT")
    c.setFillColor(BLACK)

    c.setFont("Helvetica", 7)
    c.drawCentredString(sig_line_x + sig_line_w / 2, sig_bottom + sig_h - 12, "___________________________")
    c.setFont("Helvetica", 5.5)
    c.setFillColor(GRAY)
    c.drawCentredString(sig_line_x + sig_line_w / 2, sig_bottom + 4, "TAXPAYER'S SIGNATURE")
    c.setFillColor(BLACK)

    treasurer_name = a.get("treasurer_name") or getattr(settings, "TREASURER_NAME", "")
    c.setFont("Helvetica-Bold", 7.5)
    c.drawCentredString(treas_x + (x1 - treas_x) / 2, sig_bottom + sig_h - 12, treasurer_name or "")
    c.setFont("Helvetica", 5.5)
    c.setFillColor(GRAY)
    c.drawCentredString(treas_x + (x1 - treas_x) / 2, sig_bottom + 4, "MUNICIPAL / CITY TREASURER")
    c.setFillColor(BLACK)

    # ── Tax computation block (A / B, items 1-3) ─────────────────────────
    tax_top = sig_bottom
    tax_bottom = tax_top - tax_h
    tax_left_w = full_w * 0.78
    amt_x = x0 + tax_left_w
    c.setLineWidth(0.6)
    c.rect(x0, tax_bottom, full_w, tax_top - tax_bottom)
    c.line(amt_x, tax_bottom, amt_x, tax_top)

    lines = [
        ("A. BASIC COMMUNITY TAX (P5.00)  Voluntary or Exempted (P1.00)", f"P {basic:,.2f}", True),
        ("B. ADDITIONAL COMMUNITY TAX (tax not to exceed P5,000.00)", "", True),
        ("   1. GROSS RECEIPTS OR EARNINGS DERIVED FROM BUSINESS DURING THE", "", False),
        ("      PRECEDING YEAR (P1.00 for every P1,000)", f"P {gross_business:,.2f}", False),
        ("   2. SALARIES OR GROSS RECEIPT OR EARNINGS DERIVED FROM EXERCISE OF", "", False),
        ("      PROFESSION OR PURSUIT OF ANY OCCUPATION (P1.00 for every P1,000)", f"P {salaries_income:,.2f}", False),
        ("   3. INCOME FROM REAL PROPERTY (P1.00 for every P1,000)", f"P {real_property_income:,.2f}", False),
    ]
    n = len(lines)
    lh = (tax_top - tax_bottom) / n
    yy = tax_top
    for text, amt, bold in lines:
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 6 if bold else 5.6)
        c.drawString(x0 + 4, yy - lh + 2.5, text)
        if amt:
            c.setFont("Helvetica", 6.3)
            c.drawRightString(x1 - 4, yy - lh + 2.5, amt)
        yy -= lh
    for i in range(1, n):
        yy_line = tax_top - i * lh
        c.setLineWidth(0.3)
        c.line(x0, yy_line, amt_x, yy_line)

    # ── TOTAL / INTEREST / TOTAL AMOUNT PAID footer strip ────────────────
    foot_w3 = full_w / 3
    c.setLineWidth(0.6)
    c.rect(x0, y0, full_w, foot_h)
    c.line(x0 + foot_w3, y0, x0 + foot_w3, tax_bottom)
    c.line(x0 + 2 * foot_w3, y0, x0 + 2 * foot_w3, tax_bottom)

    c.setFont("Helvetica-Bold", 6.5)
    c.drawString(x0 + 4, y0 + foot_h - 9, "TOTAL")
    c.drawRightString(x0 + foot_w3 - 4, y0 + foot_h - 9, f"P {total:,.2f}")

    c.drawString(x0 + foot_w3 + 4, y0 + foot_h - 9, "INTEREST")
    c.drawRightString(x0 + 2 * foot_w3 - 4, y0 + foot_h - 9, f"P {interest:,.2f}")

    c.drawString(x0 + 2 * foot_w3 + 4, y0 + foot_h - 9, "TOTAL AMOUNT PAID")
    c.drawRightString(x1 - 4, y0 + foot_h - 9, f"P {total_paid:,.2f}")

    c.save()
    logger.info(f"Cedula PDF generated: {output_path}")
    return output_path


def generate_receipt_pdf(ctc_number: str, applicant_data: dict, tax_data: dict) -> str:
    """
    Generate a short transaction receipt PDF (half-page, plain letter layout —
    kept separate from the card-formatted Cedula itself).
    Returns path to the receipt PDF.
    """
    _ensure_output_dir()
    output_path = os.path.join(settings.PDF_OUTPUT_DIR, f"receipt_{ctc_number.replace('-', '_')}.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=(LETTER[0], LETTER[1] / 2),
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    h_style = ParagraphStyle("H", parent=styles["Normal"], fontSize=12, textColor=colors.HexColor("#003366"), alignment=TA_CENTER)
    n_style = ParagraphStyle("N", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER)

    a = applicant_data
    full_name = f"{a.get('given_name', '')} {a.get('surname', '')}".strip()

    story.append(Paragraph(f"<b>{settings.BARANGAY_NAME}</b>", h_style))
    story.append(Paragraph("OFFICIAL RECEIPT — Community Tax Certificate", n_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#003366")))
    story.append(Spacer(1, 8))

    receipt_data = [
        ["CTC Number:", ctc_number],
        ["Name:", full_name],
        ["Amount Paid:", f"P{tax_data.get('total_tax', 5.00):,.2f}"],
        ["Date Issued:", datetime.now().strftime("%B %d, %Y")],
    ]
    r_table = Table(receipt_data, colWidths=[2 * inch, 4 * inch])
    r_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(r_table)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
    story.append(Paragraph("Thank you! Keep this receipt as proof of payment.", n_style))

    doc.build(story)
    logger.info(f"Receipt PDF generated: {output_path}")
    return output_path