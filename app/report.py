from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Paragraph, SimpleDocTemplate, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.utils import ImageReader
from datetime import datetime
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
import numpy as np


def _wrap_text(c, text, x, y, max_width, font_name="Helvetica", font_size=10, line_gap=0.5 * cm):
    """Wrap text to fit within max_width"""
    if not text:
        return y
    
    c.setFont(font_name, font_size)
    words = text.split()
    line = ""

    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, font_name, font_size) <= max_width:
            line = test
        else:
            if line:
                c.drawString(x, y, line)
                y -= line_gap
            line = w

    if line:
        c.drawString(x, y, line)
        y -= line_gap

    return y


def _check_new_page(c, y, min_space, height, width):
    """Check if new page is needed and return updated y position"""
    if y < min_space:
        c.showPage()
        # Recreate header on new page
        c.setFillColorRGB(0.23, 0.51, 0.96)
        c.rect(0, height - 2.5 * cm, width, 2.5 * cm, fill=True, stroke=False)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, height - 1.6 * cm, "Women Health Insight Report (continued)")
        y = height - 3.5 * cm
    return y


def _create_wellness_gauge(wellness_score):
    """Create a wellness score gauge chart"""
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='white')
    
    # Determine color based on score
    if wellness_score >= 75:
        color = '#4ade80'  # Green
        status = 'Excellent'
    elif wellness_score >= 50:
        color = '#fb923c'  # Orange
        status = 'Fair'
    else:
        color = '#ef4444'  # Red
        status = 'Needs Attention'
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    # Background arc
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=15)
    
    # Score arc
    score_theta = np.linspace(0, np.pi * (wellness_score / 100), 100)
    ax.plot(r * np.cos(score_theta), r * np.sin(score_theta), color, linewidth=15)
    
    # Center text
    ax.text(0, -0.1, f'{wellness_score:.0f}', ha='center', va='center', fontsize=40, fontweight='bold')
    ax.text(0, -0.35, status, ha='center', va='center', fontsize=14, color=color, fontweight='bold')
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.3)
    ax.axis('off')
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    return buf


def _create_lifestyle_chart(inputs):
    """Create a radar chart for lifestyle factors"""
    categories = ['Sleep', 'Exercise', 'Nutrition', 'Hydration', 'Stress\nManagement']
    
    # Calculate scores (0-100)
    sleep_hours = inputs.get('sleep_hours', 7)
    sleep_score = min(100, max(0, (sleep_hours - 5) * 33.33)) if sleep_hours <= 8 else max(0, 100 - (sleep_hours - 8) * 20)
    
    exercise = inputs.get('exercise_frequency', 'moderate').lower()
    exercise_score = {'sedentary': 20, 'light': 50, 'moderate': 85, 'active': 95, 'very active': 90}.get(exercise, 60)
    
    diet = inputs.get('diet_quality', 'good').lower()
    diet_score = {'poor': 30, 'fair': 50, 'good': 75, 'excellent': 95}.get(diet, 70)
    
    water = inputs.get('water_intake', 6)
    water_score = min(100, (water / 8) * 100)
    
    stress = inputs.get('stress_level', 'medium').lower()
    stress_score = {'high': 30, 'medium': 60, 'low': 90}.get(stress, 60)
    
    values = [sleep_score, exercise_score, diet_score, water_score, stress_score]
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(projection='polar'), facecolor='white')
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#3b82f6')
    ax.fill(angles, values, alpha=0.25, color='#3b82f6')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], size=8)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Lifestyle Factors Assessment', size=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    return buf


def _create_health_metrics_bar(inputs, prediction):
    """Create a bar chart for key health metrics"""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    
    metrics = ['Cycle\nLength', 'Period\nDuration', 'Sleep\nHours', 'Water\nIntake', 'Predicted\nDelay']
    values = [
        inputs.get('cycle_length', 28),
        inputs.get('period_duration', 5),
        inputs.get('sleep_hours', 7),
        inputs.get('water_intake', 6),
        prediction.get('predicted_delay', 0)
    ]
    optimal = [28, 5, 8, 8, 0]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, values, width, label='Your Values', color='#3b82f6', alpha=0.8)
    bars2 = ax.bar(x + width/2, optimal, width, label='Optimal Range', color='#4ade80', alpha=0.6)
    
    ax.set_ylabel('Days / Hours / Glasses', fontsize=10, fontweight='bold')
    ax.set_title('Health Metrics Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    return buf


def generate_pdf_report(out_path: str, patient: dict, inputs: dict, prediction: dict):
    """Generate comprehensive PDF report with all analysis and recommendations"""
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    
    # ═══════════════════════════════════════════════════════════════
    # PAGE 1: HEADER & SUMMARY
    # ═══════════════════════════════════════════════════════════════
    
    # Header with colored background - Enhanced
    c.setFillColorRGB(0.23, 0.51, 0.96)  # Modern blue
    c.rect(0, height - 4 * cm, width, 4 * cm, fill=True, stroke=False)
    
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(2 * cm, height - 1.8 * cm, "🩺 Women Health Insight Report")
    
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, height - 2.5 * cm, f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(2 * cm, height - 3 * cm, "AI-Powered Menstrual Health Analysis System")
    c.drawString(2 * cm, height - 3.5 * cm, "Confidential Medical Document - For Patient and Healthcare Provider Use Only")
    
    y = height - 5 * cm
    
    # Patient Information Box - Enhanced Design
    c.setFillColorRGB(0.97, 0.98, 1)
    c.setStrokeColorRGB(0.23, 0.51, 0.96)
    c.setLineWidth(1.5)
    c.roundRect(1.5 * cm, y - 3.8 * cm, width - 3 * cm, 3.8 * cm, 0.4 * cm, fill=True, stroke=True)
    
    c.setFillColorRGB(0.23, 0.51, 0.96)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y - 0.7 * cm, "👤 PATIENT INFORMATION")
    
    c.setFont("Helvetica-Bold", 10)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(2.5 * cm, y - 1.5 * cm, "Name:")
    c.drawString(2.5 * cm, y - 2.2 * cm, "Patient ID:")
    c.drawString(2.5 * cm, y - 2.9 * cm, "Age:")
    
    c.setFont("Helvetica", 10)
    c.drawString(5 * cm, y - 1.5 * cm, f"{patient.get('name', 'Not Provided')}")
    c.drawString(5 * cm, y - 2.2 * cm, f"{patient.get('id', 'N/A')}")
    c.drawString(5 * cm, y - 2.9 * cm, f"{patient.get('age', 'Not Specified')} years")
    
    y -= 4.8 * cm
    
    # Critical Summary Box
    pred_delay = prediction.get('predicted_delay', 0)
    risk_level = prediction.get('risk_level', 'Low')
    wellness_score = prediction.get('wellness_score', 70)
    bmi = inputs.get('bmi', 22)
    
    # Determine urgency color
    if pred_delay >= 60:
        box_color = (1, 0.9, 0.9)  # Light red
        border_color = (0.8, 0.2, 0.2)
        urgency = "🚨 URGENT"
    elif pred_delay >= 45:
        box_color = (1, 0.95, 0.85)  # Light orange
        border_color = (0.9, 0.5, 0.1)
        urgency = "⚠️ HIGH PRIORITY"
    elif pred_delay >= 15:
        box_color = (1, 1, 0.9)  # Light yellow
        border_color = (0.8, 0.7, 0.2)
        urgency = "🟡 ATTENTION NEEDED"
    else:
        box_color = (0.9, 1, 0.9)  # Light green
        border_color = (0.2, 0.7, 0.3)
        urgency = "🟢 NORMAL RANGE"
    
    c.setFillColorRGB(*box_color)
    c.setStrokeColorRGB(*border_color)
    c.setLineWidth(2)
    c.roundRect(1.5 * cm, y - 5.2 * cm, width - 3 * cm, 5.2 * cm, 0.4 * cm, fill=True, stroke=True)
    
    c.setFillColorRGB(0.06, 0.09, 0.16)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(2 * cm, y - 0.8 * cm, "📊 PREDICTION SUMMARY")
    
    c.setFont("Helvetica-Bold", 13)
    c.setFillColorRGB(*border_color)
    c.drawString(2 * cm, y - 1.6 * cm, urgency)
    
    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(2.5 * cm, y - 2.5 * cm, "Predicted Cycle Delay:")
    c.setFont("Helvetica", 11)
    c.drawString(7.5 * cm, y - 2.5 * cm, f"{pred_delay:.1f} days")
    
    if pred_delay >= 30:
        months = pred_delay / 30
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        c.drawString(2.5 * cm, y - 3.1 * cm, f"(Approximately {months:.1f} month{'s' if months > 1 else ''} delay)")
        y_offset = 3.7
    else:
        y_offset = 3.2
    
    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(2.5 * cm, y - y_offset * cm, "Risk Level:")
    c.setFont("Helvetica", 11)
    c.drawString(7.5 * cm, y - y_offset * cm, risk_level)
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2.5 * cm, y - (y_offset + 0.6) * cm, "Wellness Score:")
    c.setFont("Helvetica", 11)
    wellness_color = (0.2, 0.7, 0.2) if wellness_score >= 75 else (0.8, 0.5, 0.1) if wellness_score >= 50 else (0.8, 0.1, 0.1)
    c.setFillColorRGB(*wellness_color)
    c.drawString(7.5 * cm, y - (y_offset + 0.6) * cm, f"{wellness_score:.0f}/100")
    
    y -= 6.2 * cm
    
    # Key Health Metrics - Enhanced
    c.setFillColorRGB(0.23, 0.51, 0.96)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "📋 KEY HEALTH METRICS")
    c.setStrokeColorRGB(0.23, 0.51, 0.96)
    c.setLineWidth(1)
    c.line(2 * cm, y - 0.2 * cm, width - 2 * cm, y - 0.2 * cm)
    y -= 1 * cm
    
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    
    metrics = [
        ("Cycle Length:", f"{inputs.get('cycle_length', '-')} days"),
        ("Period Duration:", f"{inputs.get('period_duration', '-')} days"),
        ("Sleep Hours:", f"{inputs.get('sleep_hours', '-')} hrs/night"),
        ("Stress Level:", f"{inputs.get('stress_level', '-')}".title()),
        ("Exercise:", f"{inputs.get('exercise_frequency', '-')}"),
        ("BMI:", f"{bmi:.1f} ({prediction.get('bmi_category', 'Normal')})"),
        ("Water Intake:", f"{inputs.get('water_intake', '-')} glasses/day"),
        ("Diet Quality:", f"{inputs.get('diet_quality', '-')}".title()),
    ]
    
    col1_x = 2.5 * cm
    col2_x = 11 * cm
    
    for i, (label, value) in enumerate(metrics):
        row = i % 4
        col_x = col1_x if i < 4 else col2_x
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(col_x, y - (row * 0.6 * cm), label)
        c.setFont("Helvetica", 10)
        c.drawString(col_x + 3 * cm, y - (row * 0.6 * cm), str(value))
    
    y -= 3 * cm
    
    # Medical Conditions
    has_pcos = inputs.get('has_pcos', False)
    has_endo = inputs.get('has_endometriosis', False)
    has_thyroid = inputs.get('has_thyroid', False)
    
    if has_pcos or has_endo or has_thyroid:
        c.setFont("Helvetica-Bold", 12)
        c.setFillColorRGB(0.7, 0.1, 0.1)
        c.drawString(2 * cm, y, "⚕️ DIAGNOSED CONDITIONS:")
        y -= 0.6 * cm
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        
        if has_pcos:
            c.drawString(2.5 * cm, y, "• Polycystic Ovary Syndrome (PCOS)")
            y -= 0.5 * cm
        if has_endo:
            c.drawString(2.5 * cm, y, "• Endometriosis")
            y -= 0.5 * cm
        if has_thyroid:
            c.drawString(2.5 * cm, y, "• Thyroid Disorder")
            y -= 0.5 * cm
    
    y -= 0.5 * cm
    
    # Clinical Interpretation
    c.setFillColorRGB(0.23, 0.51, 0.96)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "CLINICAL INTERPRETATION")
    c.setLineWidth(1)
    c.line(2 * cm, y - 0.2 * cm, width - 2 * cm, y - 0.2 * cm)
    y -= 0.9 * cm
    
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    
    interpretation = prediction.get('interpretation', '')
    if pred_delay >= 60:
        interpretation = f"SEVERE IRREGULARITY: {pred_delay:.0f}-day delay indicates SKIPPED PERIOD(S) - likely hormonal imbalance requiring immediate medical evaluation. Possible amenorrhea, PCOS, thyroid dysfunction, or significant metabolic disruption."
    elif pred_delay >= 45:
        interpretation = f"SIGNIFICANT DELAY: {pred_delay:.0f}-day delay suggests oligomenorrhea (infrequent periods). Medical investigation recommended to identify underlying hormonal or metabolic causes."
    elif pred_delay >= 15:
        interpretation = f"MODERATE IRREGULARITY: {pred_delay:.0f}-day delay indicates cycle irregularity. May be related to stress, lifestyle factors, or early hormonal imbalance. Monitor closely."
    elif pred_delay >= 7:
        interpretation = f"SLIGHT VARIATION: {pred_delay:.0f}-day delay is within acceptable range but should be monitored. Often related to stress or lifestyle factors."
    else:
        interpretation = f"NORMAL RANGE: {pred_delay:.0f}-day variation is typical. Cycle appears regular and healthy."
    
    max_width = width - 4 * cm
    y = _wrap_text(c, interpretation, 2 * cm, y, max_width, font_size=10)
    
    # Footer for page 1
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(2 * cm, 1.5 * cm, "Page 1 of 3+ | Women Health Insight System | Confidential Medical Document")
    
    # ═══════════════════════════════════════════════════════════════
    # PAGE 2: DETAILED ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    c.showPage()
    
    # Page 2 Header - Enhanced
    c.setFillColorRGB(0.23, 0.51, 0.96)
    c.rect(0, height - 2.5 * cm, width, 2.5 * cm, fill=True, stroke=False)
    
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, height - 1.6 * cm, "🔬 DETAILED HEALTH ANALYSIS")
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, height - 2.1 * cm, "Comprehensive Lifestyle & Physical Health Assessment")
    
    y = height - 3.5 * cm
    
    # Lifestyle Factors Analysis
    c.setFillColorRGB(0.23, 0.51, 0.96)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "📊 LIFESTYLE FACTORS ASSESSMENT")
    c.setStrokeColorRGB(0.23, 0.51, 0.96)
    c.setLineWidth(1)
    c.line(2 * cm, y - 0.2 * cm, width - 2 * cm, y - 0.2 * cm)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    y -= 1 * cm
    
    c.setFont("Helvetica", 10)
    
    # Sleep Analysis
    sleep_hours = inputs.get('sleep_hours', 7)
    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0.1, 0.1, 0.1)  # Black
    c.drawString(2.5 * cm, y, "😴 Sleep Quality:")
    c.setFont("Helvetica", 10)
    y -= 0.6 * cm
    
    if sleep_hours < 6:
        sleep_assessment = f"CRITICAL: Only {sleep_hours:.1f} hours - severe sleep deprivation disrupting hormonal balance."
        c.setFillColorRGB(0.8, 0.1, 0.1)
    elif sleep_hours < 7:
        sleep_assessment = f"INSUFFICIENT: {sleep_hours:.1f} hours - below optimal range, may affect cycle regularity."
        c.setFillColorRGB(0.8, 0.5, 0.1)
    elif sleep_hours <= 9:
        sleep_assessment = f"OPTIMAL: {sleep_hours:.1f} hours - excellent for hormonal health."
        c.setFillColorRGB(0.2, 0.6, 0.2)
    else:
        sleep_assessment = f"EXCESSIVE: {sleep_hours:.1f} hours - may indicate underlying fatigue or health issues."
        c.setFillColorRGB(0.8, 0.5, 0.1)
    
    y = _wrap_text(c, sleep_assessment, 2.5 * cm, y, width - 4 * cm, font_size=10)
    c.setFillColorRGB(0.1, 0.1, 0.1)  # Reset to black
    y -= 0.5 * cm
    
    # Stress Analysis
    stress_level = inputs.get('stress_level', 'medium')
    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(2.5 * cm, y, "🧘 Stress Level:")
    c.setFont("Helvetica", 10)
    y -= 0.6 * cm
    
    if stress_level == 'high':
        stress_assessment = "HIGH STRESS: Significantly impacts menstrual health through cortisol elevation, suppressing reproductive hormones."
        c.setFillColorRGB(0.8, 0.1, 0.1)
    elif stress_level == 'medium':
        stress_assessment = "MODERATE STRESS: May affect cycle regularity. Stress management recommended."
        c.setFillColorRGB(0.8, 0.5, 0.1)
    else:
        stress_assessment = "LOW STRESS: Excellent for hormonal balance and cycle regularity."
        c.setFillColorRGB(0.2, 0.6, 0.2)
    
    y = _wrap_text(c, stress_assessment, 2.5 * cm, y, width - 4 * cm, font_size=10)
    c.setFillColorRGB(0.1, 0.1, 0.1)  # Reset to black
    y -= 0.5 * cm
    
    # Exercise Analysis
    exercise = inputs.get('exercise_frequency', 'moderate')
    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(2.5 * cm, y, "💪 Physical Activity:")
    c.setFont("Helvetica", 10)
    y -= 0.6 * cm
    
    if 'sedentary' in exercise.lower():
        exercise_assessment = "SEDENTARY: Lack of exercise increases insulin resistance and hormonal imbalances."
        c.setFillColorRGB(0.8, 0.1, 0.1)
    elif 'light' in exercise.lower():
        exercise_assessment = "LIGHT ACTIVITY: Good start, but increasing intensity would improve hormonal health."
        c.setFillColorRGB(0.8, 0.7, 0.2)
    elif 'moderate' in exercise.lower():
        exercise_assessment = "MODERATE ACTIVITY: Excellent balance for hormonal health and cycle regularity."
        c.setFillColorRGB(0.2, 0.6, 0.2)
    else:
        if bmi < 18.5:
            exercise_assessment = "VERY ACTIVE + LOW BMI: Risk of athletic amenorrhea. May need to reduce intensity."
            c.setFillColorRGB(0.8, 0.5, 0.1)
        else:
            exercise_assessment = "VERY ACTIVE: Excellent for metabolic and hormonal health."
            c.setFillColorRGB(0.2, 0.6, 0.2)
    
    y = _wrap_text(c, exercise_assessment, 2.5 * cm, y, width - 4 * cm, font_size=10)
    c.setFillColorRGB(0.1, 0.1, 0.1)  # Reset to black
    y -= 0.5 * cm
    
    # Nutrition & Hydration
    diet = inputs.get('diet_quality', 'good')
    water = inputs.get('water_intake', 6)
    
    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(2.5 * cm, y, "🥗 Nutrition & Hydration:")
    c.setFont("Helvetica", 10)
    y -= 0.6 * cm
    
    nutrition_assessment = f"Diet Quality: {diet.title()} | Water Intake: {water} glasses/day. "
    if diet in ['poor', 'fair'] or water < 6:
        nutrition_assessment += "NEEDS IMPROVEMENT: Poor nutrition and/or hydration directly affect hormonal balance."
        c.setFillColorRGB(0.8, 0.1, 0.1)
    elif diet == 'good' and water >= 6:
        nutrition_assessment += "GOOD: Adequate nutrition and hydration support hormonal health."
    y = _wrap_text(c, nutrition_assessment, 2.5 * cm, y, width - 4 * cm, font_size=10)
    c.setFillColorRGB(0.1, 0.1, 0.1)  # Reset to black
    y -= 0.8 * cm
    
    y = _check_new_page(c, y, 15 * cm, height, width)
    
    # Add Lifestyle Radar Chart
    lifestyle_chart = _create_lifestyle_chart(inputs)
    c.drawImage(ImageReader(lifestyle_chart), 2.5 * cm, y - 11 * cm, width=12 * cm, height=10 * cm, preserveAspectRatio=True)
    y -= 12 * cm
    
    y = _check_new_page(c, y, 8 * cm, height, width)
    
    # Physical Health Metrics1, 0.1)  # Reset to black
    y -= 0.8 * cm
    
    y = _check_new_page(c, y, 8 * cm, height, width)
    
    # Physical Health Metrics
    c.setFont("Helvetica-Bold", 13)
    c.setFillColorRGB(0.23, 0.51, 0.96)
    c.drawString(2 * cm, y, "⚖️ PHYSICAL HEALTH METRICS")
    c.setStrokeColorRGB(0.23, 0.51, 0.96)
    c.line(2 * cm, y - 0.2 * cm, width - 2 * cm, y - 0.2 * cm)
    y -= 1 * cm
    
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    
    weight = inputs.get('weight', 60)
    height_cm = inputs.get('height', 165)
    bmi_category = prediction.get('bmi_category', 'Normal')
    
    bmi_text = f"BMI: {bmi:.1f} - {bmi_category}"
    c.drawString(2.5 * cm, y, bmi_text)
    y -= 0.5 * cm
    
    c.drawString(2.5 * cm, y, f"Weight: {weight:.1f} kg | Height: {height_cm:.1f} cm")
    y -= 0.6 * cm
    
    if bmi < 18.5:
        bmi_impact = "UNDERWEIGHT: Low BMI can cause amenorrhea and anovulation due to insufficient body fat for hormone production."
        c.setFillColorRGB(0.8, 0.1, 0.1)
    elif bmi < 25:
        bmi_impact = "HEALTHY WEIGHT: Optimal BMI for reproductive health and regular cycles."
        c.setFillColorRGB(0.2, 0.6, 0.2)
    elif bmi < 30:
        bmi_impact = "OVERWEIGHT: Elevated BMI increases insulin resistance and androgen production, affecting cycle regularity."
        c.setFillColorRGB(0.8, 0.5, 0.1)
    else:
        bmi_impact = "OBESE: Significantly increases risk of anovulation, PCOS, and metabolic syndrome. Weight loss can restore cycles."
        c.setFillColorRGB(0.8, 0.1, 0.1)
    
    y = _wrap_text(c, bmi_impact, 2.5 * cm, y, width - 4 * cm, font_size=10)
    c.setFillColorRGB(0.1, 0.1, 0.1)  # Reset to black
    y -= 1 * cm
    
    y = _check_new_page(c, y, 13 * cm, height, width)
    
    # Add Health Metrics Bar Chart
    metrics_chart = _create_health_metrics_bar(inputs, prediction)
    c.drawImage(ImageReader(metrics_chart), 1.5 * cm, y - 10 * cm, width=14 * cm, height=9 * cm, preserveAspectRatio=True)
    y -= 11 * cm
    
    # Symptom Assessment
    y = _check_new_page(c, y, 6 * cm, height, width)
    
    symptoms = inputs.get('symptoms', [])
    cramp_severity = inputs.get('cramp_severity', 0)
    mood = inputs.get('mood_state', 'neutral')
    
    c.setFont("Helvetica-Bold", 13)
    c.setFillColorRGB(0.23, 0.51, 0.96)
    c.drawString(2 * cm, y, "💫 SYMPTOMS & QUALITY OF LIFE")
    c.setStrokeColorRGB(0.23, 0.51, 0.96)
    c.line(2 * cm, y - 0.2 * cm, width - 2 * cm, y - 0.2 * cm)
    y -= 1 * cm
    
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    
    c.drawString(2.5 * cm, y, f"Menstrual Cramp Severity: {cramp_severity}/10")
    y -= 0.6 * cm
    
    if cramp_severity >= 7:
        cramp_text = "SEVERE PAIN: Not normal - suggests possible endometriosis, fibroids, or adenomyosis. Medical evaluation urgently needed."
        c.setFillColorRGB(0.8, 0.1, 0.1)
    elif cramp_severity >= 4:
        cramp_text = "MODERATE PAIN: Can be managed with NSAIDs, magnesium, and lifestyle modifications."
        c.setFillColorRGB(0.8, 0.5, 0.1)
    elif cramp_severity > 0:
        cramp_text = "MILD PAIN: Normal prostaglandin-mediated cramping, manageable naturally."
        c.setFillColorRGB(0.2, 0.6, 0.2)
    else:
        cramp_text = "NO PAIN: Minimal menstrual discomfort."
        c.setFillColorRGB(0.2, 0.6, 0.2)
    
    y = _wrap_text(c, cramp_text, 2.5 * cm, y, width - 4 * cm, font_size=10)
    c.setFillColorRGB(0.1, 0.1, 0.1)  # Reset to black
    y -= 0.5 * cm
    
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(2.5 * cm, y, f"Mood State: {mood.title()}")
    y -= 0.5 * cm
    
    if symptoms and 'none' not in str(symptoms).lower():
        c.drawString(2.5 * cm, y, "Current Symptoms:")
        y -= 0.5 * cm
        
        if isinstance(symptoms, list):
            for symptom in symptoms[:8]:  # Limit to 8 symptoms
                c.drawString(3 * cm, y, f"• {symptom}")
                y -= 0.4 * cm
        else:
            c.drawString(3 * cm, y, f"• {symptoms}")
            y -= 0.4 * cm
    else:
        c.drawString(2.5 * cm, y, "Current Symptoms: None reported")
        y -= 0.5 * cm
    
    # Footer for page 2
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(2 * cm, 1.5 * cm, "Page 2 of 3+ | Detailed Analysis | Confidential")
    
    # ═══════════════════════════════════════════════════════════════
    # PAGE 3: RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════
    c.showPage()
    
    # Page 3 Header - Enhanced
    c.setFillColorRGB(0.23, 0.51, 0.96)
    c.rect(0, height - 2.5 * cm, width, 2.5 * cm, fill=True, stroke=False)
    
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, height - 1.6 * cm, "💡 PERSONALIZED RECOMMENDATIONS")
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, height - 2.1 * cm, "Evidence-Based Action Steps for Optimal Health")
    
    y = height - 3.5 * cm
    
    recommendations = prediction.get('recommendations', [])
    
    if recommendations:
        page_num = 3
        for idx, rec in enumerate(recommendations, 1):
            # Check if we need a new page
            y = _check_new_page(c, y, 5 * cm, height, width)
            
            # Recommendation number and category
            priority = rec.get('priority', 'LOW')
            
            # Priority indicator
            if priority == 'CRITICAL':
                priority_symbol = "🚨"
            elif priority == 'HIGH':
                priority_symbol = "⚠️"
            elif priority == 'MODERATE':
                priority_symbol = "🟡"
            else:
                priority_symbol = "🟢"
            
            c.setFont("Helvetica-Bold", 11)
            c.setFillColorRGB(0.1, 0.1, 0.1)  # Black text
            c.drawString(2 * cm, y, f"{priority_symbol} #{idx}. {rec.get('category', 'General')}")
            y -= 0.7 * cm
            
            # Title
            c.setFillColorRGB(0.06, 0.09, 0.16)
            c.setFont("Helvetica-Bold", 10)
            title = rec.get('title', '')
            y = _wrap_text(c, title, 2.5 * cm, y, width - 4 * cm, font_name="Helvetica-Bold", font_size=10, line_gap=0.5*cm)
            y -= 0.3 * cm
            
            # Advice
            c.setFillColorRGB(0.1, 0.1, 0.1)
            c.setFont("Helvetica", 9)
            advice = rec.get('advice', '')
            y = _wrap_text(c, f"Why: {advice}", 2.5 * cm, y, width - 4 * cm, font_size=9, line_gap=0.45*cm)
            y -= 0.4 * cm
            
            # Action
            c.setFont("Helvetica-Bold", 9)
            c.drawString(2.5 * cm, y, "Action Steps:")
            y -= 0.4 * cm
            c.setFont("Helvetica", 9)
            action = rec.get('action', '')
            y = _wrap_text(c, action, 2.5 * cm, y, width - 4 * cm, font_size=9, line_gap=0.45*cm)
            y -= 0.7 * cm
            
            # Separator line
            c.setStrokeColorRGB(0.85, 0.88, 0.92)
            c.line(2 * cm, y, width - 2 * cm, y)
            y -= 0.5 * cm
            
            # Update page number in footer if needed
            if y < 3 * cm:
                c.setFont("Helvetica-Oblique", 8)
                c.setFillColorRGB(0.4, 0.4, 0.4)
                c.drawString(2 * cm, 1.5 * cm, f"Page {page_num} | Recommendations | Confidential")
                page_num += 1
    else:
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.drawString(2 * cm, y, "No specific recommendations generated.")
        y -= 1 * cm
    
    # Final Notes
    y = _check_new_page(c, y, 6 * cm, height, width)
    
    c.setFont("Helvetica-Bold", 13)
    c.setFillColorRGB(0.23, 0.51, 0.96)
    c.drawString(2 * cm, y, "📝 CLINICAL NOTES")
    c.setStrokeColorRGB(0.23, 0.51, 0.96)
    c.line(2 * cm, y - 0.2 * cm, width - 2 * cm, y - 0.2 * cm)
    y -= 0.9 * cm
    
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    notes = prediction.get("notes", "").strip() or "No additional clinical notes recorded for this assessment."
    y = _wrap_text(c, notes, 2 * cm, y, width - 4 * cm, font_size=10)
    
    # Disclaimer Section
    y -= 1.5 * cm
    y = _check_new_page(c, y, 4 * cm, height, width)
    
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.roundRect(1.5 * cm, y - 3 * cm, width - 3 * cm, 3 * cm, 0.2 * cm, fill=True, stroke=True)
    
    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0.5, 0.1, 0.1)
    c.drawString(2 * cm, y - 0.6 * cm, "⚠️ IMPORTANT DISCLAIMER")
    
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    disclaimer = ("This report is generated by an AI-powered system for educational and informational purposes only. "
                  "It is NOT a medical diagnosis and should NOT replace professional medical advice, diagnosis, or treatment. "
                  "Always consult with a qualified healthcare provider (gynecologist, endocrinologist, or primary care physician) "
                  "for medical concerns, especially for irregular cycles, severe symptoms, or diagnosed conditions. "
                  "If you experience severe pain, heavy bleeding, or symptoms of emergency, seek immediate medical attention.")
    
    y = _wrap_text(c, disclaimer, 2 * cm, y - 1 * cm, width - 4 * cm, font_size=9, line_gap=0.4*cm)
    
    # Final Footer
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(2 * cm, 1.5 * cm, f"End of Report | Generated by Women Health Insight System | {datetime.now().strftime('%Y')}")
    
    c.showPage()
    c.save()
    
    return out_path
