"""
app/services/pdf_service.py
BIR Form 0016 (DECEMBER, 2014) — Community Tax Certificate PDF generation.
Layout matches the official printed CTC card as closely as possible:
  - Header: BIR seal | BIR FORM 0016 | COMMUNITY TAX CERTIFICATE | INDIVIDUAL | CCI{year}  {serial}
  - Row 1:  YEAR | PLACE OF ISSUE | DATE ISSUED | TAXPAYER'S COPY (right) | SEX + TIN (right)
  - Row 2:  NAME (SURNAME) | (FIRST) | (MIDDLE) | HEIGHT (right)
  - Row 3:  ADDRESS | WEIGHT (right)
  - Row 4:  CITIZENSHIP | ICR NO. | PLACE OF BIRTH | DATE OF BIRTH (right)
  - Row 5:  CIVIL STATUS (4 checkboxes) | TAXABLE AMOUNT (right)
  - Row 6:  PROFESSION / OCCUPATION / BUSINESS (spans rows 6-7) | COMMUNITY TAX DUE (right)
  - Tax:    A. BASIC | B. ADDITIONAL -> 1. Business | 2. Salaries | 3. Real Property
  - Sig:    Right Thumb Print | Taxpayer's Signature | Municipal/City Treasurer
  - Footer: TOTAL | INTEREST | TOTAL AMOUNT PAID | (In words)
"""
import os
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Colors
BLACK      = colors.black
WHITE      = colors.white
DARK_GRAY  = colors.HexColor("#333333")
MID_GRAY   = colors.HexColor("#666666")
LIGHT_GRAY = colors.HexColor("#CCCCCC")
HEADER_BG  = colors.HexColor("#2B4D7C")
SEAL_BLUE  = colors.HexColor("#1E3A6E")
SEAL_GOLD  = colors.HexColor("#B8860B")

# Card dimensions
CARD_W = 8.27 * inch
CARD_H = 4.5  * inch
MARGIN = 0.20 * inch


def _ensure_output_dir():
    os.makedirs(settings.PDF_OUTPUT_DIR, exist_ok=True)


def _amount_in_words(amount: float) -> str:
    """Convert peso amount to words without num2words dependency."""
    try:
        ones = ["", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty",
                "sixty", "seventy", "eighty", "ninety"]

        def say(n):
            if n == 0:
                return "zero"
            if n < 20:
                return ones[n]
            if n < 100:
                return tens[n // 10] + ("-" + ones[n % 10] if n % 10 else "")
            if n < 1000:
                return ones[n // 100] + " hundred" + (" " + say(n % 100) if n % 100 else "")
            if n < 1_000_000:
                return say(n // 1000) + " thousand" + (" " + say(n % 1000) if n % 1000 else "")
            return say(n // 1_000_000) + " million" + (" " + say(n % 1_000_000) if n % 1_000_000 else "")

        pesos = int(amount)
        centavos = round((amount - pesos) * 100)
        result = say(pesos).upper() + " PESOS"
        if centavos > 0:
            result += " AND " + say(centavos).upper() + " CENTAVOS"
        return result
    except Exception:
        return ""


# Drawing helpers

def _border(c, x, y, w, h, lw=0.6):
    c.setLineWidth(lw)
    c.rect(x, y, w, h, stroke=1, fill=0)


def _hline(c, x1, y, x2, lw=0.5):
    c.setLineWidth(lw)
    c.line(x1, y, x2, y)


def _vline(c, x, y1, y2, lw=0.5):
    c.setLineWidth(lw)
    c.line(x, y1, x, y2)


def _label(c, x, y, text, size=5.0):
    c.setFont("Helvetica", size)
    c.setFillColor(MID_GRAY)
    c.drawString(x, y, text)
    c.setFillColor(BLACK)


def _value(c, x, y, text, size=8.5, bold=False):
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.setFillColor(BLACK)
    c.drawString(x, y, str(text) if text else "")


def _cell(c, x, y, w, h, caption, value, cap_size=5.0, val_size=8.2,
          val_bold=False, pad_x=3, pad_y=3):
    """Caption pinned to the top of the cell, value pinned to the bottom.
    Works for any cell height, including merged (multi-row) cells."""
    cap_y = y + h - cap_size - pad_y
    val_y = y + pad_y + 1
    # never let the value collide with the caption on very short cells
    val_y = min(val_y, cap_y - val_size - 2)
    _label(c, x + pad_x, cap_y, caption, size=cap_size)
    _value(c, x + pad_x, val_y, value, size=val_size, bold=val_bold)


def _checkbox(c, x, y, size=6.5, checked=False, number=None):
    c.setLineWidth(0.6)
    c.setFillColor(BLACK if checked else WHITE)
    c.rect(x, y, size, size, stroke=1, fill=1 if checked else 0)
    if number:
        c.setFillColor(WHITE if checked else BLACK)
        c.setFont("Helvetica", size - 1.5)
        c.drawCentredString(x + size / 2, y + size / 2 - 2, str(number))
    c.setFillColor(BLACK)


def _draw_bir_seal(c, cx, cy, r=15):
    """Draw a simple stylised BIR-style circular seal (vector, not a
    reproduction of the official artwork) so the card reads as an official
    government form even without an embedded logo asset."""
    c.saveState()
    # outer ring
    c.setLineWidth(1.1)
    c.setStrokeColor(SEAL_BLUE)
    c.setFillColor(WHITE)
    c.circle(cx, cy, r, stroke=1, fill=1)
    c.setLineWidth(0.5)
    c.circle(cx, cy, r - 2.2, stroke=1, fill=0)

    # simple 8-point sunburst in the center (evokes the Philippine sun)
    c.setFillColor(SEAL_GOLD)
    import math
    rays = 8
    outer_r = r - 4.5
    inner_r = outer_r * 0.35
    pts = []
    for i in range(rays * 2):
        ang = math.pi * i / rays
        rad = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + rad * math.sin(ang), cy + rad * math.cos(ang)))
    p = c.beginPath()
    p.moveTo(pts[0][0], pts[0][1])
    for (px, py) in pts[1:]:
        p.lineTo(px, py)
    p.close()
    c.setStrokeColor(SEAL_GOLD)
    c.drawPath(p, stroke=1, fill=1)

    # small center dot
    c.setFillColor(SEAL_BLUE)
    c.circle(cx, cy, 1.6, stroke=0, fill=1)

    # curved text around the ring (top: REPUBLIKA NG PILIPINAS)
    c.setFillColor(SEAL_BLUE)
    c.setFont("Helvetica-Bold", 3.6)
    top_text = "REPUBLIKA NG PILIPINAS"
    _draw_text_on_arc(c, top_text, cx, cy, r - 1.2, start_deg=200, end_deg=-20, size=3.4)
    bottom_text = "BUREAU OF INTERNAL REVENUE"
    _draw_text_on_arc(c, bottom_text, cx, cy, r - 1.2, start_deg=160, end_deg=380, size=3.0, flip=True)
    c.restoreState()


def _draw_text_on_arc(c, text, cx, cy, radius, start_deg, end_deg, size=3.5, flip=False):
    """Draw text following a circular arc (used by the seal)."""
    import math
    c.setFont("Helvetica-Bold", size)
    n = len(text)
    if n == 0:
        return
    span = end_deg - start_deg
    for i, ch in enumerate(text):
        frac = (i + 0.5) / n
        deg = start_deg + span * frac
        rad = math.radians(deg)
        x = cx + radius * math.cos(rad)
        y = cy + radius * math.sin(rad)
        angle = deg + 90 if not flip else deg - 90
        c.saveState()
        c.translate(x, y)
        c.rotate(angle)
        c.drawCentredString(0, 0, ch)
        c.restoreState()


def generate_cedula_pdf(
    ctc_number: str,
    applicant_data: dict,
    tax_data: dict,
    output_filename: Optional[str] = None,
) -> str:
    _ensure_output_dir()

    if not output_filename:
        safe = ctc_number.replace("-", "_").replace(" ", "_")
        output_filename = f"{safe}.pdf"
    output_path = os.path.join(settings.PDF_OUTPUT_DIR, output_filename)

    a = applicant_data
    t = tax_data
    now = datetime.now()

    # Parse CTC number -> CCI header parts
    try:
        parts = ctc_number.split("-")
        cci_year   = parts[1][:4] if len(parts) > 1 else str(now.year)
        cci_serial = parts[2].zfill(8) if len(parts) > 2 else "00000001"
    except Exception:
        cci_year   = str(now.year)
        cci_serial = "00000001"

    year_issued  = cci_year
    place_issued = a.get("place_of_issue") or f"{settings.MUNICIPALITY}, {settings.PROVINCE}"

    # Windows-safe date (no %-m or %-d)
    date_issued  = f"{now.month}/{now.day}/{str(now.year)[2:]}"
    dop_date     = f"{str(now.month).zfill(2)}.{str(now.day).zfill(2)}.{now.year}"

    surname      = (a.get("surname")      or "").upper()
    given_name   = (a.get("given_name")   or "").upper()
    middle_name  = (a.get("middle_name")  or "").upper()
    address      = (a.get("address")      or "").upper()
    tin          =  a.get("tin")          or ""
    sex          = (a.get("sex")          or "").lower()
    citizenship  = (a.get("citizenship")  or "FIL.").upper()
    icr_no       =  a.get("icr_number")   or ""
    place_birth  = (a.get("place_of_birth") or "").upper()
    height       =  str(a.get("height_cm")  or "")
    weight_val   =  str(a.get("weight_kg")  or "")
    weight_str   =  f"{weight_val} kg." if weight_val else ""
    civil_status = (a.get("civil_status") or "").lower()
    dob          =  str(a.get("date_of_birth") or "")
    occupation   = (a.get("occupation")   or "").upper()
    income       =  float(a.get("gross_annual_income") or 0.0)
    treasurer    =  getattr(settings, "TREASURER_NAME", "")

    basic       = float(t.get("basic_community_tax",       5.00))
    gross_biz   = float(t.get("gross_receipts_business",   0.00))
    salaries    = float(t.get("salaries_professional_income", income))
    real_prop   = float(t.get("income_real_property",      0.00))
    additional  = float(t.get("additional_community_tax",  0.00))
    total_tax   = float(t.get("total_tax", basic + additional))
    interest    = float(t.get("interest",                  0.00))
    total_paid  = float(t.get("total_amount_paid", total_tax + interest))
    taxable_amt = gross_biz + salaries + real_prop
    in_words    = _amount_in_words(total_paid)

    # Canvas
    c = pdfcanvas.Canvas(output_path, pagesize=(CARD_W, CARD_H))
    c.setTitle(f"Community Tax Certificate - {ctc_number}")
    c.setAuthor(settings.BARANGAY_NAME)

    L  = MARGIN
    R  = CARD_W - MARGIN
    T  = CARD_H - MARGIN
    B  = MARGIN
    FW = R - L

    _border(c, L, B, FW, T - B, lw=1.0)

    # ── HEADER ────────────────────────────────────────────────────────────
    HDR_H   = 36
    hdr_top = T
    hdr_bot = hdr_top - HDR_H
    _hline(c, L, hdr_bot, R, lw=1.0)

    # BIR seal (top-left)
    SEAL_R = 15
    seal_cx = L + SEAL_R + 4
    seal_cy = hdr_bot + HDR_H / 2
    _draw_bir_seal(c, seal_cx, seal_cy, r=SEAL_R)

    text_x = seal_cx + SEAL_R + 6
    c.setFont("Helvetica", 5.5)
    c.setFillColor(BLACK)
    c.drawString(text_x, hdr_top - 8, "BIR FORM 0016 (DECEMBER, 2014)")

    c.setFont("Helvetica-Bold", 11.5)
    c.drawString(text_x, hdr_bot + 7, "COMMUNITY TAX CERTIFICATE")

    IND_W, IND_H = 62, HDR_H
    IND_X = L + 230
    c.setFillColor(HEADER_BG)
    c.rect(IND_X, hdr_bot, IND_W, IND_H, stroke=1, fill=1)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawCentredString(IND_X + IND_W / 2, hdr_bot + IND_H / 2 - 4, "INDIVIDUAL")
    c.setFillColor(BLACK)

    CCI_X = IND_X + IND_W + 4
    CCI_W = R - CCI_X
    _border(c, CCI_X, hdr_bot, CCI_W, HDR_H, lw=1.0)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(CCI_X + 5, hdr_bot + 9, f"CCI{cci_year}")
    c.drawRightString(R - 5, hdr_bot + 9, cci_serial)

    # ── VERTICAL ZONES ───────────────────────────────────────────────────
    WORK_TOP = hdr_bot
    FOOT_H   = 27
    TAX_H    = 73
    SIG_H    = 36
    WORK_BOT = B + FOOT_H + TAX_H + SIG_H
    WORK_H   = WORK_TOP - WORK_BOT
    LEFT_W   = FW * 0.635
    RIGHT_X  = L + LEFT_W
    RIGHT_W  = R - RIGHT_X

    N_ROWS = 7
    # Row 0 (TAXPAYER'S COPY / SEX / TIN) needs more vertical room than the
    # others, so rows are weighted rather than split evenly.
    ROW_WEIGHTS = [1.35, 1.0, 1.0, 1.0, 1.0, 0.85, 0.80]
    W_TOTAL = sum(ROW_WEIGHTS)
    ys = [WORK_TOP]
    for w in ROW_WEIGHTS:
        ys.append(ys[-1] - WORK_H * (w / W_TOTAL))
    ROW_H = WORK_H / N_ROWS  # kept for cells that assume the average height

    def row_top(i): return ys[i]
    def row_bot(i): return ys[i + 1]

    _vline(c, RIGHT_X, WORK_BOT, WORK_TOP)
    # Row separators on the LEFT side only. Skip the boundary between row 5
    # and row 6 (index 6) because the occupation cell spans both rows.
    for i in range(1, N_ROWS):
        if i == 6:
            continue
        _hline(c, L, ys[i], RIGHT_X)

    # ROW 0 — YEAR | PLACE OF ISSUE | DATE ISSUED
    r0t, r0b = row_top(0), row_bot(0)
    r0h = r0t - r0b
    YEAR_W  = LEFT_W * 0.18
    PLACE_W = LEFT_W * 0.55
    DATE_W  = LEFT_W - YEAR_W - PLACE_W
    _vline(c, L + YEAR_W,           r0b, r0t)
    _vline(c, L + YEAR_W + PLACE_W, r0b, r0t)

    _cell(c, L,                      r0b, YEAR_W,  r0h, "YEAR",
          year_issued, val_size=9, val_bold=True)
    _cell(c, L + YEAR_W,             r0b, PLACE_W, r0h,
          "PLACE OF ISSUE (City/Mun./Prov.)", place_issued, cap_size=4.5, val_size=7.2)
    _cell(c, L + YEAR_W + PLACE_W,   r0b, DATE_W,  r0h, "DATE ISSUED",
          date_issued, val_size=8.5, val_bold=True)

    # Right side of row 0 has three stacked pieces: TAXPAYER'S COPY (top),
    # SEX checkboxes (middle), TIN boxes (bottom). Row 0 is weighted taller
    # than the other rows specifically to give this stack enough room.
    copy_band_h = r0h * 0.30
    sex_band_h  = r0h * 0.36
    tin_band_h  = r0h - copy_band_h - sex_band_h

    copy_y = r0t - copy_band_h
    sex_y  = copy_y - sex_band_h
    tin_y0 = r0b

    _hline(c, RIGHT_X, copy_y, R)
    _hline(c, RIGHT_X, sex_y, R)

    c.setFont("Helvetica-Bold", 6.3)
    c.setFillColor(BLACK)
    c.drawCentredString(RIGHT_X + RIGHT_W / 2, copy_y + copy_band_h / 2 - 3, "TAXPAYER'S COPY")

    _label(c, RIGHT_X + 3, sex_y + sex_band_h - 6, "SEX:", size=4.8)
    sx = RIGHT_X + 20
    sy = sex_y + (sex_band_h - 6) / 2
    _checkbox(c, sx, sy, size=6, checked=sex.startswith("m"), number=1)
    c.setFont("Helvetica", 5.2); c.setFillColor(BLACK)
    c.drawString(sx + 8, sy + 1, "MALE")
    _checkbox(c, sx + 32, sy, size=6, checked=sex.startswith("f"), number=2)
    c.drawString(sx + 40, sy + 1, "FEMALE")

    tin_lbl = "TIN:"
    c.setFont("Helvetica", 4.4)
    c.setFillColor(MID_GRAY)
    tin_lbl_w = c.stringWidth(tin_lbl, "Helvetica", 4.4)
    c.drawString(RIGHT_X + 3, tin_y0 + 2.5, tin_lbl)
    c.setFillColor(BLACK)
    TIN_BOX_W  = min(11.5, (RIGHT_W - 8 - tin_lbl_w) / 9 - 1.2)
    TIN_BOX_H  = max(7.5, tin_band_h - 4)
    tin_start_x = RIGHT_X + 6 + tin_lbl_w
    tin_yb = tin_y0 + 1.5
    for i in range(9):
        bx = tin_start_x + i * (TIN_BOX_W + 1.5)
        digit = tin[i] if i < len(tin) and tin[i].isdigit() else ""
        c.rect(bx, tin_yb, TIN_BOX_W, TIN_BOX_H, stroke=1, fill=0)
        if digit:
            c.setFont("Helvetica", 6.2)
            c.drawCentredString(bx + TIN_BOX_W / 2, tin_yb + 2, digit)

    # ROW 1 — NAME | HEIGHT
    r1t, r1b = row_top(1), row_bot(1)
    NAME_SUR_W = LEFT_W * 0.40
    NAME_FST_W = LEFT_W * 0.35
    NAME_MID_W = LEFT_W - NAME_SUR_W - NAME_FST_W
    _vline(c, L + NAME_SUR_W,              r1b, r1t)
    _vline(c, L + NAME_SUR_W + NAME_FST_W, r1b, r1t)

    _cell(c, L,                             r1b, NAME_SUR_W, ROW_H,
          "NAME (SURNAME)", surname,    val_size=9, val_bold=True)
    _cell(c, L + NAME_SUR_W,                r1b, NAME_FST_W, ROW_H,
          "(FIRST)",        given_name, val_size=9, val_bold=True)
    _cell(c, L + NAME_SUR_W + NAME_FST_W,   r1b, NAME_MID_W, ROW_H,
          "(MIDDLE)",       middle_name, val_size=9, val_bold=True)
    _cell(c, RIGHT_X, r1b, RIGHT_W, ROW_H, "HEIGHT",
          f"{height} cm." if height else "", val_size=9, val_bold=True)

    # ROW 2 — ADDRESS | WEIGHT
    r2t, r2b = row_top(2), row_bot(2)
    _cell(c, L,       r2b, LEFT_W, ROW_H, "ADDRESS", address,   val_size=8.5, val_bold=True)
    _cell(c, RIGHT_X, r2b, RIGHT_W, ROW_H, "WEIGHT",  weight_str, val_size=9,   val_bold=True)

    # ROW 3 — CITIZENSHIP | ICR | PLACE OF BIRTH | DATE OF BIRTH
    r3t, r3b = row_top(3), row_bot(3)
    CIT_W = LEFT_W * 0.25
    ICR_W = LEFT_W * 0.28
    POB_W = LEFT_W - CIT_W - ICR_W
    _vline(c, L + CIT_W,         r3b, r3t)
    _vline(c, L + CIT_W + ICR_W, r3b, r3t)

    _cell(c, L,               r3b, CIT_W, ROW_H, "CITIZENSHIP",           citizenship, val_size=8.5, val_bold=True)
    _cell(c, L + CIT_W,       r3b, ICR_W, ROW_H, "ICR NO. (If an Alien)", icr_no,      val_size=8)
    _cell(c, L + CIT_W+ICR_W, r3b, POB_W, ROW_H, "PLACE OF BIRTH",        place_birth, val_size=7.6, val_bold=True)
    _cell(c, RIGHT_X,         r3b, RIGHT_W, ROW_H, "DATE OF BIRTH",       dob,         val_size=9,   val_bold=True)

    # ROW 4 — CIVIL STATUS | TAXABLE AMOUNT
    r4t, r4b = row_top(4), row_bot(4)
    _label(c, L + 3, r4t - 6.5, "CIVIL STATUS", size=5.2)

    cs_items = [
        ("1", ["Single"],                          "single"),
        ("2", ["Married"],                         "married"),
        ("3", ["Widow/Widower/", "Legally Separated"], "widow"),
        ("4", ["Divorced"],                        "divorced"),
    ]
    CS_COL = LEFT_W / 4
    c.setFont("Helvetica", 5.4)
    for i, (num, lines, key) in enumerate(cs_items):
        col_x = L + i * CS_COL
        checked = (
            key in civil_status
            or (key == "widow"    and "widowed"   in civil_status)
            or (key == "divorced" and "separated" in civil_status)
        )
        box_x = col_x + 4
        box_y = r4b + ROW_H * 0.16
        _checkbox(c, box_x, box_y, size=6.2, checked=checked, number=num)
        c.setFont("Helvetica", 5.2)
        c.setFillColor(BLACK)
        c.drawString(box_x + 9, box_y + 1, lines[0])
        if len(lines) > 1:
            c.drawString(box_x + 9, box_y - 5.5, lines[1])

    _cell(c, RIGHT_X, r4b, RIGHT_W, ROW_H, "TAXABLE AMOUNT",
          f"P {taxable_amt:,.2f}", val_size=9, val_bold=True)

    # ROW 5+6 — PROFESSION / OCCUPATION / BUSINESS (merged, spans both rows)
    r5t = row_top(5)
    r5b = row_bot(5)
    r6b = row_bot(6)
    OCC_H = r5t - r6b
    _cell(c, L, r6b, LEFT_W, OCC_H,
          "PROFESSION / OCCUPATION / BUSINESS", occupation,
          cap_size=5.2, val_size=8.5, val_bold=True, pad_y=4)

    # Right side keeps two stacked cells: COMMUNITY TAX DUE label (row 5)
    # with the peso amount emphasised in the row 6 band beneath it.
    r5h = r5t - r5b
    r6h = r5b - r6b
    _hline(c, RIGHT_X, r5b, R)
    _label(c, RIGHT_X + 3, r5t - 6.5, "COMMUNITY TAX DUE", size=5.2)
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(BLACK)
    c.drawCentredString(RIGHT_X + RIGHT_W / 2, r6b + (r6h - 8) / 2, f"P {total_tax:,.2f}")

    # ── TAX COMPUTATION BLOCK ────────────────────────────────────────────
    tax_top = WORK_BOT
    tax_bot = tax_top - TAX_H
    _border(c, L, tax_bot, FW, TAX_H)

    AMT_COL_W = FW * 0.22
    amt_x = R - AMT_COL_W
    _vline(c, amt_x, tax_bot, tax_top)

    TAX_LINES = [
        ("A. BASIC COMMUNITY TAX (P5.00)  Voluntary or Exempted (P 1.00)",
         f"P  {basic:,.2f}", True, 0),
        ("B. ADDITIONAL COMMUNITY TAX (Tax not to exceed P5,000.00)",
         None, True, 0),
        ("1.  GROSS RECEIPTS OR EARNINGS DERIVED FROM BUSINESS DURING THE\n"
         "     PRECEDING YEAR ( P1.00 for every P 1,000.00 )",
         f"P  {gross_biz:,.2f}", False, 1),
        ("2.  SALARIES OR GROSS RECEIPT OR EARNINGS DERIVED FROM EXERCISE OF\n"
         "     PROFESSION OR PURSUIT OF ANY OCCUPATION ( P1.00 for every P 1,000 )",
         f"P  {salaries:,.2f}", False, 1),
        ("3.  INCOME FROM REAL PROPERTY ( P1.00 for every P 1,000 )",
         f"P  {real_prop:,.2f}", False, 1),
    ]
    n_tax   = len(TAX_LINES)
    t_row_h = TAX_H / n_tax
    for i, (text, amt, bold, indent) in enumerate(TAX_LINES):
        ty = tax_top - (i + 1) * t_row_h
        if i > 0:
            _hline(c, L, ty + t_row_h, amt_x, lw=0.3)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 5.8 if bold else 5.4)
        c.setFillColor(BLACK)
        lines = text.split("\n")
        c.drawString(L + 4 + indent * 6, ty + t_row_h - 8, lines[0])
        if len(lines) > 1:
            c.drawString(L + 4 + indent * 6, ty + t_row_h - 14, lines[1])
        if amt:
            c.setFont("Helvetica", 6.5)
            c.drawRightString(R - 4, ty + t_row_h - 10, amt)

    # ── SIGNATURE BLOCK ──────────────────────────────────────────────────
    sig_top  = tax_bot
    sig_bot  = sig_top - SIG_H
    _border(c, L, sig_bot, FW, SIG_H)

    THUMB_W = FW * 0.13
    SIG_W   = FW * 0.42
    TREAS_W = FW - THUMB_W - SIG_W
    sig_div1 = L + THUMB_W
    sig_div2 = sig_div1 + SIG_W
    _vline(c, sig_div1, sig_bot, sig_top)
    _vline(c, sig_div2, sig_bot, sig_top)

    c.setFont("Helvetica-Bold", 6.5)
    c.setFillColor(BLACK)
    c.drawCentredString(L + THUMB_W / 2, sig_bot + SIG_H * 0.65, "Right Thumb")
    c.drawCentredString(L + THUMB_W / 2, sig_bot + SIG_H * 0.45, "Print")

    _label(c, sig_div1 + 3, sig_top - 7, "TAXPAYER'S SIGNATURE", size=5.5)
    c.setLineWidth(0.5)
    c.line(sig_div1 + 10, sig_bot + 14, sig_div2 - 10, sig_bot + 14)

    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(BLACK)
    c.drawCentredString(sig_div2 + TREAS_W / 2, sig_bot + SIG_H * 0.65,
                        treasurer.upper() if treasurer else "")
    _label(c, sig_div2 + 3, sig_bot + 5, "MUNICIPAL / CITY TREASURER", size=5.5)

    # ── FOOTER ───────────────────────────────────────────────────────────
    foot_top = sig_bot
    foot_bot = B
    _border(c, L, foot_bot, FW, foot_top - foot_bot)

    FOOT_COL_W = FW * 0.26
    f2x = L + FOOT_COL_W
    f3x = f2x + FOOT_COL_W
    _vline(c, f2x, foot_bot, foot_top)
    _vline(c, f3x, foot_bot, foot_top)

    mid_y = foot_bot + (foot_top - foot_bot) / 2

    c.setFont("Helvetica-Bold", 6.5); c.setFillColor(BLACK)
    c.drawString(L + 4, mid_y + 3, "TOTAL")
    c.setFont("Helvetica-Bold", 8)
    c.drawRightString(f2x - 4, mid_y - 6, f"P  {total_tax:,.2f}")

    c.setFont("Helvetica-Bold", 6.5)
    c.drawString(f2x + 4, mid_y + 3, "INTEREST")
    c.setFont("Helvetica-Bold", 8)
    c.drawRightString(f3x - 4, mid_y - 6, f"P  {interest:,.2f}")

    c.setFont("Helvetica-Bold", 6.5)
    c.drawString(f3x + 4, foot_top - 7, "TOTAL AMOUNT PAID")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(f3x + 4, foot_top - 17, f"P  {total_paid:,.2f}")
    c.setFont("Helvetica", 4.6)
    c.setFillColor(MID_GRAY)
    words_line = f"(In words): {in_words}"
    if len(words_line) > 46:
        words_line = words_line[:46] + "…"
    c.drawString(f3x + 4, foot_bot + 4, words_line)
    c.setFillColor(BLACK)

    # DOP stamp
    c.setFont("Helvetica", 5.5)
    c.setFillColor(MID_GRAY)
    c.drawString(L, B - 10, f"DOP: {dop_date}")
    c.setFillColor(BLACK)

    c.save()
    logger.info(f"Cedula PDF generated: {output_path}")
    return output_path


def generate_receipt_pdf(ctc_number: str, applicant_data: dict, tax_data: dict) -> str:
    """Generate a short transaction receipt (half-page)."""
    from reportlab.lib.units import inch as _inch
    _ensure_output_dir()
    output_path = os.path.join(
        settings.PDF_OUTPUT_DIR,
        f"receipt_{ctc_number.replace('-', '_')}.pdf"
    )
    doc = SimpleDocTemplate(
        output_path,
        pagesize=(LETTER[0], LETTER[1] / 2),
        rightMargin=0.5 * _inch, leftMargin=0.5 * _inch,
        topMargin=0.4 * _inch, bottomMargin=0.4 * _inch,
    )
    styles = getSampleStyleSheet()
    h_style = ParagraphStyle("H", parent=styles["Normal"], fontSize=12,
                              textColor=colors.HexColor("#003366"), alignment=TA_CENTER)
    n_style = ParagraphStyle("N", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER)
    a = applicant_data
    full_name  = f"{a.get('given_name', '')} {a.get('surname', '')}".strip()
    total      = float(tax_data.get("total_tax", 5.00))
    total_paid = float(tax_data.get("total_amount_paid", total))
    now        = datetime.now()
    date_str   = f"{now.strftime('%B')} {now.day}, {now.year}"

    story = [
        Paragraph(f"<b>{settings.BARANGAY_NAME}</b>", h_style),
        Paragraph("OFFICIAL RECEIPT — Community Tax Certificate", n_style),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#003366")),
        Spacer(1, 8),
    ]
    data = [
        ["CTC Number:", ctc_number],
        ["Name:",       full_name],
        ["Amount Paid:", f"P{total_paid:,.2f}"],
        ["Date Issued:", date_str],
    ]
    tbl = Table(data, colWidths=[2 * _inch, 4 * _inch])
    tbl.setStyle(TableStyle([
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("FONTNAME",      (0, 0), (0,  -1), "Helvetica-Bold"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [
        tbl,
        Spacer(1, 10),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")),
        Paragraph("Thank you! Keep this receipt as proof of payment.", n_style),
    ]
    doc.build(story)
    logger.info(f"Receipt PDF generated: {output_path}")
    return output_path