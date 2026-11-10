"""
app/services/eval_card_pdf.py
Plan 009: a generated evaluation card as the participant reads it — one line per question or step, in the
order the interview asks them — and the same card as a printable one-page PDF (ReportLab, like
pdf_service.py). The card is a made-up person; the PDF says so at the top.
"""
import io
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# (key, label): the interview's order, with the address and the place of birth in their steps.
LINES = (
    ("surname", "Surname"), ("given_name", "Given name"), ("middle_name", "Middle name"),
    ("address_town", "Address: town or city"), ("address_barangay", "Address: barangay"),
    ("address_purok", "Address: purok or street"),
    ("date", "Date of birth"), ("sex", "Sex"), ("civil_status", "Civil status"), ("citizenship", "Citizenship"),
    ("icr_no", "ICR number"),
    ("place_of_birth_province", "Place of birth: province or city"), ("place_of_birth_town", "Place of birth: town"),
    ("height", "Height"), ("weight", "Weight"), ("occupation", "Occupation"),
    ("gross_annual_income", "Gross annual income"),
)
PARENT = {"address_town": "address", "address_barangay": "address", "address_purok": "address",
          "place_of_birth_province": "place_of_birth", "place_of_birth_town": "place_of_birth"}


def card_lines(answers: dict) -> List[dict]:
    """What the participant says, question by question ({key, label, read}); a step the card lacks is left out."""
    lines = []
    for key, label in LINES:
        parent = PARENT.get(key)
        answer = answers.get(parent, {}).get("parts", {}).get(key) if parent else answers.get(key)
        if answer:
            lines.append({"key": key, "label": label, "read": answer["read"]})
    return lines


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(card_id: int, answers: dict) -> bytes:
    """The one-page card: large text, the question on the left, the words to say on the right."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title=f"CeduSync evaluation card {card_id}", author="CeduSync")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=2)
    note = ParagraphStyle("n", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#555555"))
    label = ParagraphStyle("l", parent=styles["Normal"], fontSize=10, leading=13, textColor=colors.HexColor("#333333"))
    say = ParagraphStyle("s", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=13, leading=16)
    rows = [[Paragraph(_escape(line["label"]), label), Paragraph(_escape(line["read"]), say)]
            for line in card_lines(answers)]
    table = Table(rows, colWidths=[2.0 * inch, 5.1 * inch])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE1DE")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    doc.build([
        Paragraph(f"Evaluation card #{card_id}", title),
        Paragraph("A made-up person, not a real resident. When the system asks a question, read the words on the "
                  "right exactly as written.", note),
        Spacer(1, 10),
        table,
    ])
    return buffer.getvalue()
