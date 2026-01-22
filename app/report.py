from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from datetime import datetime


def _wrap_text(c, text, x, y, max_width, font_name="Helvetica", font_size=10, line_gap=0.55 * cm):
    c.setFont(font_name, font_size)
    words = (text or "").split()
    line = ""

    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, font_name, font_size) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= line_gap
            line = w

    if line:
        c.drawString(x, y, line)
        y -= line_gap

    return y


def generate_pdf_report(out_path: str, patient: dict, inputs: dict, prediction: dict):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4

    # Header
    c.setFillColorRGB(0.06, 0.09, 0.16)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, height - 2.2 * cm, "Women Health Insight Report")

    c.setFillColorRGB(0.25, 0.28, 0.34)
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, height - 2.9 * cm, f"Generated on: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

    c.setStrokeColorRGB(0.85, 0.88, 0.92)
    c.line(2 * cm, height - 3.2 * cm, width - 2 * cm, height - 3.2 * cm)

    y = height - 4.2 * cm

    def section(title):
        nonlocal y
        c.setFillColorRGB(0.06, 0.09, 0.16)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, y, title)
        y -= 0.7 * cm
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("Helvetica", 11)

    # Patient
    section("Patient Details")
    c.drawString(2 * cm, y, f"Patient Name: {patient.get('name', '-')}")
    y -= 0.55 * cm
    c.drawString(2 * cm, y, f"Patient ID: {patient.get('id', '-')}")
    y -= 0.55 * cm
    c.drawString(2 * cm, y, f"Age: {patient.get('age', '-')}")
    y -= 0.9 * cm

    # Inputs
    section("Inputs")
    for k, v in inputs.items():
        c.drawString(2 * cm, y, f"{k.replace('_', ' ').title()}: {v}")
        y -= 0.55 * cm
    y -= 0.35 * cm

    # Prediction
    section("Prediction Summary")
    c.drawString(2 * cm, y, f"Predicted Delay: {prediction.get('predicted_delay', 0):.1f} days")
    y -= 0.55 * cm
    c.drawString(2 * cm, y, f"Risk Level: {prediction.get('risk_level', '-')}")
    y -= 0.55 * cm
    c.drawString(2 * cm, y, f"Interpretation: {prediction.get('interpretation', '-')}")
    y -= 0.9 * cm

    # Notes
    section("Clinical Notes")
    notes = prediction.get("notes", "").strip() or "No additional notes."
    max_width = width - 4 * cm
    y = _wrap_text(c, notes, 2 * cm, y, max_width, font_name="Helvetica", font_size=10)

    # Disclaimer
    c.setFillColorRGB(0.35, 0.38, 0.42)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(2 * cm, 1.8 * cm, "Disclaimer: AI-assisted report. Not a replacement for clinical diagnosis.")

    c.showPage()
    c.save()
