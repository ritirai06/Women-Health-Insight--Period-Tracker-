import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from db import save_to_csv, save_to_sqlite, load_history, make_report_path
from report import generate_pdf_report
from recommendations import generate_personalized_recommendations, get_bmi_category


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Women Health Insight System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "delay_predictor_model.pkl")


# ═══════════════════════════════════════════════════════════════════
# PROFESSIONAL MEDICAL DASHBOARD CSS - DARK THEME
# Modern, sleek design inspired by advanced analytics dashboards
# ═══════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─────────────────────────────────────────────────────────────── */
/* Global Styles - Modern Dark Theme */
/* ─────────────────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Hide hamburger menu and default streamlit elements */
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden;}
header {visibility: hidden;}
button[kind="header"] {display: none !important;}

section[data-testid="stSidebar"] > div > div:first-child > button {
    display: none !important;
}

/* Main content container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1600px;
}

/* ─────────────────────────────────────────────────────────────── */
/* Sidebar - Modern Dark Sidebar */
/* ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f3a 0%, #0f1829 100%);
    border-right: 1px solid rgba(59, 130, 246, 0.3);
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

/* Sidebar section headers */
.sidebar-section-header {
    font-size: 13px;
    font-weight: 700;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(59, 130, 246, 0.3);
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, transparent 100%);
    padding-left: 0.5rem;
    border-radius: 4px;
}

/* ─────────────────────────────────────────────────────────────── */
/* Cards - Glassmorphism Effect */
/* ─────────────────────────────────────────────────────────────── */
.card {
    background: rgba(26, 31, 58, 0.6);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.3);
    transform: translateY(-2px);
    border-color: rgba(59, 130, 246, 0.4);
}

/* Card with accent border */
.card-accent {
    background: rgba(26, 31, 58, 0.7);
    border: 1px solid rgba(59, 130, 246, 0.4);
    border-left: 4px solid #3b82f6;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}

.card-accent:hover {
    box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.4);
    transform: translateY(-2px);
}

/* Analytics card with glow */
.analytics-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 0 30px rgba(139, 92, 246, 0.2);
    backdrop-filter: blur(10px);
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.analytics-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #8b5cf6, transparent);
}

.analytics-card:hover {
    box-shadow: 0 0 40px rgba(139, 92, 246, 0.4);
    transform: translateY(-4px);
    border-color: rgba(139, 92, 246, 0.5);
}

/* ─────────────────────────────────────────────────────────────── */
/* Typography - Modern & Clean */
/* ─────────────────────────────────────────────────────────────── */
.app-title {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
    letter-spacing: -0.5px;
}

.app-subtitle {
    font-size: 15px;
    color: #94a3b8;
    margin-top: 0.5rem;
    font-weight: 400;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    text-shadow: 0 2px 10px rgba(59, 130, 246, 0.5);
    padding-bottom: 0.75rem;
    border-bottom: 2px solid rgba(59, 130, 246, 0.3);
}

.subsection-title {
    font-size: 16px;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1rem 0 0.75rem 0;
    text-shadow: 0 1px 5px rgba(0, 0, 0, 0.5);
}

/* ─────────────────────────────────────────────────────────────── */
/* KPI Metrics - Glowing Cards */
/* ─────────────────────────────────────────────────────────────── */
.kpi-container {
    text-align: center;
    padding: 1.5rem;
    background: rgba(15, 23, 42, 0.8);
    border-radius: 16px;
    border: 1px solid rgba(59, 130, 246, 0.3);
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.kpi-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at top, rgba(59, 130, 246, 0.1), transparent);
    z-index: 0;
}

.kpi-container > * {
    position: relative;
    z-index: 1;
}

.kpi-container:hover {
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.4);
    transform: translateY(-4px) scale(1.02);
    border-color: rgba(59, 130, 246, 0.5);
}

.kpi-value {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}

.kpi-label {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ─────────────────────────────────────────────────────────────── */
/* Risk Level Badges - Modern Design */
/* ─────────────────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.2rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.badge-low {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
    border: 1.5px solid #10b981;
    color: #6ee7b7;
}

.badge-moderate {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.2) 100%);
    border: 1.5px solid #f59e0b;
    color: #fcd34d;
}

.badge-high {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
    border: 1.5px solid #ef4444;
    color: #fca5a5;
}

/* ─────────────────────────────────────────────────────────────── */
/* Buttons - Futuristic Style */
/* ─────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 14px;
    border: 1px solid rgba(59, 130, 246, 0.3);
    background: rgba(30, 41, 59, 0.5);
    color: #e2e8f0;
    transition: all 0.3s ease;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(59, 130, 246, 0.3);
    border-color: rgba(59, 130, 246, 0.6);
    background: rgba(59, 130, 246, 0.1);
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    border: none;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    box-shadow: 0 6px 30px rgba(59, 130, 246, 0.6);
}

/* ─────────────────────────────────────────────────────────────── */
/* CRITICAL FIX: Dropdown Text - BLACK TEXT ON WHITE BACKGROUND */
/* ─────────────────────────────────────────────────────────────── */

/* Hide instruction text */
div[data-testid="InputInstructions"] {
    display: none !important;
}

/* Dropdown container - WHITE BACKGROUND */
div[data-baseweb="select"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    border-radius: 12px !important;
}

/* Fix dropdown button alignment - CENTER TEXT VERTICALLY */
div[data-baseweb="select"] > div {
    display: flex !important;
    align-items: center !important;
    min-height: 44px !important;
    padding: 0.5rem 0.75rem !important;
    background: #FFFFFF !important;
}

/* Selected value container - ENSURE CENTERED */
div[data-baseweb="select"] [role="button"] {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: 100% !important;
    min-height: 44px !important;
    background: #FFFFFF !important;
}

/* Selected value text - FORCE BLACK AND CENTERED */
div[data-baseweb="select"] [role="button"] > div {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    height: auto !important;
}

/* Dropdown selected text - BLACK AND VERY VISIBLE */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div {
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    display: inline-flex !important;
    align-items: center !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Make select value super visible - BLACK TEXT */
div[data-baseweb="select"] [data-baseweb="select-value"],
div[data-baseweb="select"] [data-baseweb="select-value"] > * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* Dropdown arrow icon - DARK */
div[data-baseweb="select"] svg {
    fill: #1e293b !important;
}

/* Dropdown menu items - WHITE BACKGROUND WITH BLACK TEXT */
div[data-baseweb="popover"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(59, 130, 246, 0.5) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(15px) !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.7) !important;
    z-index: 9999 !important;
}

div[data-baseweb="popover"] ul[role="listbox"] {
    background: #FFFFFF !important;
    padding: 0.5rem !important;
}

/* Dropdown list items - BLACK TEXT ON WHITE */
div[data-baseweb="popover"] ul[role="listbox"] li,
div[data-baseweb="popover"] ul[role="listbox"] li *,
div[data-baseweb="popover"] ul[role="listbox"] li span,
div[data-baseweb="popover"] ul[role="listbox"] li div {
    color: #000000 !important;
    background: #FFFFFF !important;
    padding: 0.75rem 1rem !important;
    border-radius: 8px !important;
    margin: 0.25rem 0 !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
    display: flex !important;
    align-items: center !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 500 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Hover state - LIGHT BLUE BACKGROUND */
div[data-baseweb="popover"] ul[role="listbox"] li:hover,
div[data-baseweb="popover"] ul[role="listbox"] li:hover * {
    background: rgba(59, 130, 246, 0.15) !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Number inputs - WHITE VISIBLE TEXT */
.stNumberInput input,
input[type="number"],
div[data-testid="stNumberInput"] input {
    color: #FFFFFF !important;
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    border-radius: 12px !important;
    padding: 0.75rem !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Text inputs - WHITE VISIBLE TEXT */
div[data-baseweb="input"] input,
input[type="text"],
textarea {
    color: #FFFFFF !important;
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    border-radius: 12px !important;
    padding: 0.75rem !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Input wrappers */
div[data-baseweb="input"] > div {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    border-radius: 12px !important;
}

/* Focus states */
div[data-baseweb="select"]:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3) !important;
    background: #FFFFFF !important;
    outline: none !important;
}

div[data-baseweb="input"] > div:focus-within,
.stNumberInput input:focus,
input:focus,
textarea:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3) !important;
    background: rgba(30, 41, 59, 0.95) !important;
    outline: none !important;
}

/* Labels - WHITE AND VISIBLE */
label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stTextArea label {
    font-weight: 600 !important;
    color: #FFFFFF !important;
    font-size: 13px !important;
    margin-bottom: 0.5rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Slider styling */
div[data-baseweb="slider"] {
    padding-top: 1rem !important;
}

div[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.3) 0%, #3b82f6 100%) !important;
}

div[data-baseweb="slider"] [role="slider"] {
    background: #3b82f6 !important;
    border: 3px solid #1e293b !important;
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.5) !important;
}

/* Slider value text - WHITE AND VISIBLE */
div[data-baseweb="slider"] div,
div[data-baseweb="slider"] span,
.stSlider div,
.stSlider span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Text area */
textarea {
    min-height: 100px !important;
    font-family: inherit !important;
    resize: vertical !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Specific textarea styling */
.stTextArea textarea,
div[data-baseweb="textarea"] textarea {
    color: #FFFFFF !important;
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    border-radius: 12px !important;
    padding: 0.75rem !important;
    font-weight: 600 !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Radio buttons & Checkboxes - WHITE TEXT */
div[role="radiogroup"] label,
div[role="radiogroup"] span,
.stRadio label,
.stRadio span,
.stRadio div,
div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] span,
div[data-testid="stCheckbox"] p,
.stCheckbox label,
.stCheckbox span,
.stCheckbox p {
    color: #FFFFFF !important;
    font-weight: 500 !important;
    text-transform: none !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Radio & Checkbox input markers */
div[role="radio"][aria-checked="true"],
div[data-testid="stCheckbox"] [aria-checked="true"] {
    background: #3b82f6 !important;
    border-color: #3b82f6 !important;
}

/* ─────────────────────────────────────────────────────────────── */
/* Tabs - Modern Design */
/* ─────────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 14px;
    color: #94a3b8;
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    background: transparent;
    border: 1px solid transparent;
    transition: all 0.3s ease;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #60a5fa;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.4);
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
}

div[data-baseweb="tab-border"] {
    background: rgba(59, 130, 246, 0.2);
}

/* ─────────────────────────────────────────────────────────────── */
/* COMPREHENSIVE TEXT VISIBILITY FIX */
/* ─────────────────────────────────────────────────────────────── */

/* All input placeholders */
::placeholder,
input::placeholder,
textarea::placeholder {
    color: #64748b !important;
    opacity: 0.7 !important;
}

/* Ensure all text in inputs is visible */
input,
select,
textarea,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Sidebar specific selectbox - BLACK TEXT ON WHITE */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    background: #FFFFFF !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] div {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Ensure helper text and captions are visible */
.stTextInput small,
.stNumberInput small,
.stSelectbox small,
.stSlider small,
small,
caption {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* ─────────────────────────────────────────────────────────────── */
/* Data Tables - Dark Theme */
/* ─────────────────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid rgba(59, 130, 246, 0.2);
    overflow: hidden;
    background: rgba(15, 23, 42, 0.5);
}

/* ─────────────────────────────────────────────────────────────── */
/* Alerts & Messages - Dark Theme */
/* ─────────────────────────────────────────────────────────────── */
.stAlert {
    border-radius: 12px;
    border-left-width: 4px;
    padding: 1rem 1.25rem;
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(10px);
}

div[data-baseweb="notification"][kind="success"] {
    background: rgba(16, 185, 129, 0.1);
    border-left-color: #10b981;
    color: #6ee7b7;
}

div[data-baseweb="notification"][kind="info"] {
    background: rgba(59, 130, 246, 0.1);
    border-left-color: #3b82f6;
    color: #93c5fd;
}

div[data-baseweb="notification"][kind="error"] {
    background: rgba(239, 68, 68, 0.1);
    border-left-color: #ef4444;
    color: #fca5a5;
}

/* ─────────────────────────────────────────────────────────────── */
/* Download Button */
/* ─────────────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    box-shadow: 0 6px 30px rgba(16, 185, 129, 0.5);
    transform: translateY(-2px);
}

/* ─────────────────────────────────────────────────────────────── */
/* Info Box */
/* ─────────────────────────────────────────────────────────────── */
.info-box {
    background: rgba(26, 31, 58, 0.7);
    border: 1px solid rgba(59, 130, 246, 0.4);
    border-left: 4px solid #3b82f6;
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(10px);
}

.info-box-title {
    font-size: 20px;
    font-weight: 700;
    color: #60a5fa;
    margin: 0 0 1rem 0;
}

.info-box-content {
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.8;
}

/* ─────────────────────────────────────────────────────────────── */
/* Loading State */
/* ─────────────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: #3b82f6 !important;
}

/* ─────────────────────────────────────────────────────────────── */
/* Expander Styling - Make Headers Visible */
/* ─────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(26, 31, 58, 0.8) !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    border-radius: 12px !important;
    padding: 1rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    color: #e2e8f0 !important;
    transition: all 0.3s ease !important;
}

.streamlit-expanderHeader:hover {
    background: rgba(59, 130, 246, 0.15) !important;
    border-color: rgba(59, 130, 246, 0.5) !important;
    transform: translateX(4px) !important;
}

/* Expander header text - BRIGHT WHITE */
.streamlit-expanderHeader p,
.streamlit-expanderHeader span,
.streamlit-expanderHeader div,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span,
div[data-testid="stExpander"] summary div,
div[data-testid="stExpander"] summary strong {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

/* Expander content area */
.streamlit-expanderContent {
    background: transparent !important;
    border: none !important;
    padding: 1rem 0 !important;
}

/* Expander icon/arrow - WHITE */
.streamlit-expanderHeader svg {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
}

/* ─────────────────────────────────────────────────────────────── */
/* All Paragraph and Text Elements - Ensure Visibility */
/* ─────────────────────────────────────────────────────────────── */
p, span, div, h1, h2, h3, h4, h5, h6, strong, b, i, em {
    color: inherit;
    opacity: 1 !important;
}

/* Main content text - DEFAULT WHITE */
.main p,
.main span:not([data-baseweb]),
.main div:not([data-baseweb]):not([class*="plot"]):not([class*="chart"]) {
    color: #e2e8f0;
}

/* Card text elements - BRIGHT */
.card p,
.card span,
.card div,
.card-accent p,
.card-accent span,
.card-accent div,
.analytics-card p,
.analytics-card span,
.analytics-card div {
    color: #e2e8f0 !important;
}

/* Headings in cards - VERY BRIGHT */
.card h1, .card h2, .card h3, .card h4, .card h5, .card h6,
.card-accent h1, .card-accent h2, .card-accent h3, .card-accent h4, .card-accent h5, .card-accent h6 {
    color: #FFFFFF !important;
}

/* Strong/Bold text - BRIGHT */
strong, b {
    color: #FFFFFF;
    font-weight: 700;
}

/* Info/Success/Warning/Error boxes - Keep text visible */
.stAlert p,
.stAlert span,
.stAlert div {
    opacity: 1 !important;
}

/* ─────────────────────────────────────────────────────────────── */
/* Scrollbar Styling */
/* ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.5);
}

::-webkit-scrollbar-thumb {
    background: rgba(59, 130, 246, 0.5);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(59, 130, 246, 0.7);
}

/* ─────────────────────────────────────────────────────────────── */
/* Responsive Design */
/* ─────────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .app-title {
        font-size: 28px;
    }
    
    .kpi-value {
        font-size: 32px;
    }
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════
def load_model_fn():
    """Load the trained ML model with error handling"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ **Model not found** at: `{MODEL_PATH}`")
        st.info("Please ensure the model file exists before running predictions.")
        st.stop()
    return joblib.load(MODEL_PATH)

# Load model without caching decorator to avoid tokenize issues
try:
    model = load_model_fn()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def risk_level(days: float) -> str:
    """Determine risk level based on predicted delay"""
    if days <= 1.5:
        return "Low"
    elif days <= 4:
        return "Moderate"
    return "High"


# Note: get_bmi_category function moved to recommendations.py
# Import it at the top: from recommendations import get_bmi_category


def get_wellness_score(cycle_length, period_duration, sleep_hours, stress_level, 
                       exercise_frequency, water_intake, diet_quality, bmi, 
                       cramp_severity, has_pcos, has_endometriosis, has_thyroid):
    """Calculate comprehensive wellness score"""
    score = 100
    
    # Cycle health (20 points)
    if cycle_length < 24 or cycle_length > 35:
        score -= 10
    if period_duration > 7:
        score -= 5
    if period_duration < 2:
        score -= 5
    
    # Sleep quality (15 points)
    if sleep_hours < 6:
        score -= 15
    elif sleep_hours < 7:
        score -= 8
    elif sleep_hours > 9:
        score -= 5
    
    # Stress management (15 points)
    stress_impact = {"low": 0, "medium": 8, "high": 15}
    score -= stress_impact.get(stress_level, 8)
    
    # Exercise (10 points)
    exercise_scores = {"sedentary": -10, "light (1-2 days/week)": -5, 
                      "moderate (3-4 days/week)": 0, "active (5+ days/week)": 5}
    score += exercise_scores.get(exercise_frequency, 0)
    
    # Hydration (10 points)
    if water_intake < 4:
        score -= 10
    elif water_intake < 6:
        score -= 5
    elif water_intake >= 8:
        score += 5
    
    # Diet quality (10 points)
    diet_scores = {"poor": -10, "fair": -5, "good": 0, "excellent": 5}
    score += diet_scores.get(diet_quality, 0)
    
    # BMI (10 points)
    if bmi < 18.5 or bmi >= 30:
        score -= 10
    elif bmi >= 25:
        score -= 5
    
    # Cramp severity (5 points)
    if cramp_severity >= 7:
        score -= 5
    elif cramp_severity >= 4:
        score -= 3
    
    # Medical conditions (15 points)
    if has_pcos:
        score -= 8
    if has_endometriosis:
        score -= 8
    if has_thyroid:
        score -= 5
    
    return max(0, min(110, score))  # Allow bonus points up to 110





def risk_badge(risk: str) -> str:
    """Generate HTML badge for risk level"""
    badges = {
        "Low": "<span class='badge badge-low'>🟢 Low Risk</span>",
        "Moderate": "<span class='badge badge-moderate'>🟡 Moderate Risk</span>",
        "High": "<span class='badge badge-high'>🔴 High Risk</span>"
    }
    return badges.get(risk, "")


def interpretation(days: float) -> str:
    """Provide clinical interpretation of the prediction"""
    if days <= 1.5:
        return "Normal variation"
    elif days <= 4:
        return "Slight delay likely"
    return "Irregularity risk"


def build_gauge(delay_days: float, risk: str):
    """Create professional gauge chart for cycle delay visualization"""
    color_map = {
        "Low": "#10B981",
        "Moderate": "#F59E0B",
        "High": "#EF4444"
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=delay_days,
        number={
            "suffix": " days",
            "font": {"size": 48, "color": "#e2e8f0", "family": "Inter"}
        },
        gauge={
            "axis": {
                "range": [0, 10],
                "tickcolor": "#64748b",
                "tickfont": {"size": 12, "color": "#94a3b8"}
            },
            "bar": {"color": color_map.get(risk, "#3b82f6"), "thickness": 0.8},
            "steps": [
                {"range": [0, 1.5], "color": "rgba(16, 185, 129, 0.1)"},
                {"range": [1.5, 4], "color": "rgba(245, 158, 11, 0.1)"},
                {"range": [4, 10], "color": "rgba(239, 68, 68, 0.1)"},
            ],
            "threshold": {
                "line": {"color": "#e2e8f0", "width": 3},
                "thickness": 0.85,
                "value": delay_days
            },
            "bgcolor": "rgba(0,0,0,0)"
        },
        title={
            "text": f"<b style='color:#FFFFFF'>Cycle Delay Prediction</b><br><span style='font-size:14px;color:#cbd5e1'>Risk Level: {risk}</span>",
            "font": {"size": 18}
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0")
    )
    
    return fig


def build_health_score_gauge(cycle_length, period_duration, sleep_hours, stress_level):
    """Calculate and visualize health score"""
    # Simple health score calculation
    score = 100
    
    # Deduct for irregular cycle
    if cycle_length < 24 or cycle_length > 35:
        score -= 15
    
    # Deduct for poor sleep
    if sleep_hours < 6:
        score -= 20
    elif sleep_hours < 7:
        score -= 10
    
    # Deduct for high stress
    if stress_level == "high":
        score -= 20
    elif stress_level == "medium":
        score -= 10
    
    # Deduct for long periods
    if period_duration > 7:
        score -= 10
    
    score = max(0, min(100, score))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "suffix": "%",
            "font": {"size": 52, "color": "#60a5fa", "family": "Inter", "weight": "bold"}
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#64748b",
                "tickfont": {"size": 11, "color": "#94a3b8"}
            },
            "bar": {
                "color": "#60a5fa",
                "thickness": 0.9,
                "line": {"color": "#1e293b", "width": 2}
            },
            "steps": [
                {"range": [0, 40], "color": "rgba(239, 68, 68, 0.15)"},
                {"range": [40, 70], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [70, 100], "color": "rgba(16, 185, 129, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#a78bfa", "width": 4},
                "thickness": 0.9,
                "value": score
            },
            "bgcolor": "rgba(0,0,0,0)"
        },
        title={
            "text": "<b style='color:#FFFFFF'>Cycle Health Score</b><br><span style='font-size:12px;color:#cbd5e1'>Overall wellness indicator</span>",
            "font": {"size": 16}
        }
    ))
    
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0")
    )
    
    return fig, score


def build_trend_chart(hist_df):
    """Build trend analysis chart"""
    if hist_df.empty:
        return None
    
    fig = go.Figure()
    
    # Predicted delay trend
    fig.add_trace(go.Scatter(
        x=list(range(len(hist_df))),
        y=hist_df['predicted_delay'],
        mode='lines+markers',
        name='Predicted Delay',
        line=dict(color='#60a5fa', width=3),
        marker=dict(size=8, color='#3b82f6', line=dict(width=2, color='#1e293b')),
        fill='tozeroy',
        fillcolor='rgba(96, 165, 250, 0.1)'
    ))
    
    # Add risk threshold lines
    fig.add_hline(y=1.5, line_dash="dash", line_color="#10b981", 
                  annotation_text="Low Risk", annotation_position="right")
    fig.add_hline(y=4, line_dash="dash", line_color="#f59e0b", 
                  annotation_text="Moderate Risk", annotation_position="right")
    
    fig.update_layout(
        title="<b style='color:#FFFFFF;'>Cycle Delay Trend Analysis</b>",
        xaxis_title="Assessment Number",
        yaxis_title="Predicted Delay (days)",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        font=dict(family="Inter", color="#e2e8f0", size=12),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor="rgba(59, 130, 246, 0.3)",
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(59, 130, 246, 0.1)',
        zeroline=False
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(59, 130, 246, 0.1)',
        zeroline=False
    )
    
    return fig


def build_correlation_heatmap(cycle_length, period_duration, sleep_hours, flow_level, stress_level):
    """Build parameter correlation heatmap"""
    # Encode categorical variables
    flow_encoding = {"light": 1, "medium": 2, "heavy": 3}
    stress_encoding = {"low": 1, "medium": 2, "high": 3}
    
    # Create correlation data
    data = {
        'Cycle Length': [cycle_length],
        'Period Duration': [period_duration],
        'Sleep Hours': [sleep_hours],
        'Flow Level': [flow_encoding.get(flow_level, 2)],
        'Stress Level': [stress_encoding.get(stress_level, 2)]
    }
    
    df = pd.DataFrame(data)
    
    # Create simple correlation visualization
    fig = go.Figure(data=go.Heatmap(
        z=[[cycle_length/60, period_duration/10, sleep_hours/12, 
            flow_encoding.get(flow_level, 2)/3, stress_encoding.get(stress_level, 2)/3]],
        x=['Cycle Length', 'Period Duration', 'Sleep Hours', 'Flow Level', 'Stress Level'],
        y=['Current Values'],
        colorscale=[
            [0, '#1e293b'],
            [0.5, '#3b82f6'],
            [1, '#a78bfa']
        ],
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Normalized<br>Value",
                side="right"
            ),
            tickmode="linear",
            tick0=0,
            dtick=0.25,
            tickfont=dict(color="#cbd5e1")
        ),
        hovertemplate='<b>%{x}</b><br>Normalized Value: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b style='color:#FFFFFF;'>Health Parameter Distribution</b>",
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0", size=12),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            showticklabels=False
        ),
        margin=dict(l=20, r=100, t=60, b=80)
    )
    
    return fig


def build_radar_chart(cycle_length, period_duration, sleep_hours, flow_level, stress_level):
    """Build radar chart for health parameters"""
    # Normalize values
    flow_encoding = {"light": 70, "medium": 50, "heavy": 30}
    stress_encoding = {"low": 80, "medium": 50, "high": 20}
    
    categories = ['Cycle<br>Regularity', 'Period<br>Duration', 'Sleep<br>Quality', 
                  'Flow<br>Level', 'Stress<br>Management']
    
    # Calculate normalized scores (higher is better)
    values = [
        100 - abs(cycle_length - 28) * 3,  # Cycle regularity
        100 - abs(period_duration - 5) * 10,  # Period duration
        (sleep_hours / 8) * 100,  # Sleep quality
        flow_encoding.get(flow_level, 50),  # Flow level
        stress_encoding.get(stress_level, 50)  # Stress management
    ]
    
    # Clamp values
    values = [max(0, min(100, v)) for v in values]
    values.append(values[0])  # Close the radar chart
    categories_closed = categories + [categories[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(96, 165, 250, 0.2)',
        line=dict(color='#60a5fa', width=3),
        marker=dict(size=8, color='#3b82f6'),
        name='Health Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(15, 23, 42, 0.3)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color='#94a3b8'),
                gridcolor='rgba(59, 130, 246, 0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#cbd5e1'),
                gridcolor='rgba(59, 130, 246, 0.2)'
            )
        ),
        showlegend=False,
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        title="<b style='color:#FFFFFF;'>Wellness Profile</b>",
        margin=dict(l=80, r=80, t=60, b=40)
    )
    
    return fig


def build_risk_distribution_chart(hist_df):
    """Build risk level distribution chart"""
    if hist_df.empty:
        return None
    
    risk_counts = hist_df['risk_level'].value_counts()
    
    colors = {
        'Low': '#10b981',
        'Moderate': '#f59e0b',
        'High': '#ef4444'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.6,
        marker=dict(
            colors=[colors.get(level, '#3b82f6') for level in risk_counts.index],
            line=dict(color='#1e293b', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=14, color='#e2e8f0', family='Inter'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="<b style='color:#FFFFFF;'>Risk Level Distribution</b>",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor="rgba(59, 130, 246, 0.3)",
            borderwidth=1,
            font=dict(size=12)
        ),
        annotations=[dict(
            text='Risk<br>Levels',
            x=0.5, y=0.5,
            font=dict(size=16, color='#94a3b8'),
            showarrow=False
        )],
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def build_parameter_timeline(hist_df):
    """NEW: Build timeline chart showing parameter changes over time"""
    if hist_df.empty or len(hist_df) < 2:
        return None
    
    fig = go.Figure()
    
    # Add trace for each parameter
    fig.add_trace(go.Scatter(
        x=list(range(len(hist_df))),
        y=hist_df['cycle_length'],
        mode='lines+markers',
        name='Cycle Length',
        line=dict(color='#60a5fa', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(hist_df))),
        y=hist_df['sleep_hours'] * 3,  # Scale for visibility
        mode='lines+markers',
        name='Sleep Hours (×3)',
        line=dict(color='#a78bfa', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(hist_df))),
        y=hist_df['period_duration'] * 5,  # Scale for visibility
        mode='lines+markers',
        name='Period Duration (×5)',
        line=dict(color='#10b981', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="<b style='color:#FFFFFF;'>Parameter Evolution Timeline</b>",
        xaxis_title="Assessment Number",
        yaxis_title="Parameter Values (Scaled)",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        font=dict(family="Inter", color="#e2e8f0", size=12),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor="rgba(59, 130, 246, 0.3)",
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(59, 130, 246, 0.1)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(59, 130, 246, 0.1)'
    )
    
    return fig


def build_sleep_stress_analysis(hist_df):
    """NEW: Analyze correlation between sleep and stress levels"""
    if hist_df.empty or len(hist_df) < 3:
        return None
    
    # Encode stress levels
    stress_map = {'low': 1, 'medium': 2, 'high': 3}
    hist_df['stress_encoded'] = hist_df['stress_level'].map(stress_map)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_df['sleep_hours'],
        y=hist_df['stress_encoded'],
        mode='markers',
        marker=dict(
            size=hist_df['predicted_delay'] * 5,  # Size based on delay
            color=hist_df['predicted_delay'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Delay<br>(days)",
                    side="right"
                ),
                tickfont=dict(color="#cbd5e1")
            ),
            line=dict(width=1, color='#1e293b')
        ),
        text=hist_df.index,
        hovertemplate='<b>Assessment %{text}</b><br>Sleep: %{x} hrs<br>Stress: %{y}<br>Delay: %{marker.color:.1f} days<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b style='color:#FFFFFF;'>Sleep vs Stress Analysis</b>",
        xaxis_title="Sleep Hours",
        yaxis_title="Stress Level",
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['Low', 'Medium', 'High']
        ),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        font=dict(family="Inter", color="#e2e8f0", size=12),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(59, 130, 246, 0.1)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(59, 130, 246, 0.1)'
    )
    
    return fig


def build_cycle_regularity_chart(hist_df):
    """NEW: Visualize cycle regularity over time"""
    if hist_df.empty or len(hist_df) < 2:
        return None
    
    # Calculate deviation from ideal 28-day cycle
    hist_df['cycle_deviation'] = abs(hist_df['cycle_length'] - 28)
    
    fig = go.Figure()
    
    # Bar chart showing cycle deviation
    colors = ['#10b981' if x <= 3 else '#f59e0b' if x <= 7 else '#ef4444' 
              for x in hist_df['cycle_deviation']]
    
    fig.add_trace(go.Bar(
        x=list(range(len(hist_df))),
        y=hist_df['cycle_deviation'],
        marker=dict(
            color=colors,
            line=dict(color='#1e293b', width=1)
        ),
        hovertemplate='<b>Assessment %{x}</b><br>Deviation: %{y} days<br>Cycle Length: %{customdata}<extra></extra>',
        customdata=hist_df['cycle_length']
    ))
    
    # Add reference line
    fig.add_hline(y=3, line_dash="dash", line_color="#10b981", 
                  annotation_text="Normal Range", annotation_position="right")
    
    fig.update_layout(
        title="<b style='color:#FFFFFF;'>Cycle Regularity Analysis</b>",
        xaxis_title="Assessment Number",
        yaxis_title="Deviation from 28-day Cycle",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        font=dict(family="Inter", color="#e2e8f0", size=12),
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(59, 130, 246, 0.1)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(59, 130, 246, 0.1)'
    )
    
    return fig


def build_health_metrics_comparison(cycle_length, period_duration, sleep_hours, flow_level, stress_level):
    """NEW: Compare current metrics against ideal ranges"""
    
    # Define ideal ranges
    metrics = {
        'Cycle Length': {
            'current': cycle_length,
            'ideal_min': 24,
            'ideal_max': 35,
            'ideal': 28
        },
        'Period Duration': {
            'current': period_duration,
            'ideal_min': 3,
            'ideal_max': 7,
            'ideal': 5
        },
        'Sleep Hours': {
            'current': sleep_hours,
            'ideal_min': 7,
            'ideal_max': 9,
            'ideal': 8
        }
    }
    
    fig = go.Figure()
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        # Current value
        fig.add_trace(go.Bar(
            name='Current',
            x=[metric_name],
            y=[values['current']],
            marker=dict(color='#60a5fa'),
            showlegend=(i == 0)
        ))
        
        # Ideal value
        fig.add_trace(go.Bar(
            name='Ideal',
            x=[metric_name],
            y=[values['ideal']],
            marker=dict(color='#10b981'),
            showlegend=(i == 0)
        ))
    
    fig.update_layout(
        title="<b style='color:#FFFFFF;'>Current vs Ideal Health Metrics</b>",
        xaxis_title="",
        yaxis_title="Value",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        font=dict(family="Inter", color="#e2e8f0", size=12),
        barmode='group',
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor="rgba(59, 130, 246, 0.3)",
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(59, 130, 246, 0.1)'
    )
    
    return fig


def build_lifestyle_score_breakdown(exercise_frequency, water_intake, diet_quality, sleep_hours, stress_level):
    """NEW: Visualize lifestyle factors as a score breakdown"""
    
    # Calculate individual scores
    exercise_scores = {"sedentary": 20, "light (1-2 days/week)": 50, 
                      "moderate (3-4 days/week)": 75, "active (5+ days/week)": 100}
    exercise_score = exercise_scores.get(exercise_frequency, 50)
    
    hydration_score = min(100, (water_intake / 8) * 100)
    
    diet_scores = {"poor": 25, "fair": 50, "good": 75, "excellent": 100}
    diet_score = diet_scores.get(diet_quality, 75)
    
    sleep_score = 100 if 7 <= sleep_hours <= 9 else (75 if 6 <= sleep_hours <= 10 else 50)
    
    stress_scores = {"low": 100, "medium": 60, "high": 30}
    stress_score = stress_scores.get(stress_level, 60)
    
    categories = ['Exercise', 'Hydration', 'Diet', 'Sleep', 'Stress<br>Management']
    values = [exercise_score, hydration_score, diet_score, sleep_score, stress_score]
    
    colors = ['#10b981' if v >= 75 else '#f59e0b' if v >= 50 else '#ef4444' for v in values]
    
    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=colors,
            line=dict(color='#1e293b', width=1)
        ),
        text=[f"{v:.0f}%" for v in values],
        textposition='outside',
        textfont=dict(size=14, color='#e2e8f0', family='Inter', weight='bold'),
        hovertemplate='<b>%{x}</b><br>Score: %{y:.0f}%<extra></extra>'
    ))
    
    # Add target line at 75%
    fig.add_hline(y=75, line_dash="dash", line_color="#60a5fa", 
                  annotation_text="Target: 75%", annotation_position="right")
    
    fig.update_layout(
        title="<b style='color:#FFFFFF;'>Lifestyle Factor Scores</b>",
        xaxis_title="",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 110]),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        font=dict(family="Inter", color="#e2e8f0", size=12),
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    fig.update_xaxes(
        showgrid=False
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(59, 130, 246, 0.1)'
    )
    
    return fig


def build_symptom_frequency_chart(symptoms):
    """NEW: Visualize current symptoms"""
    if not symptoms or symptoms == ["none"]:
        return None
    
    symptom_list = [s for s in symptoms if s != "none"]
    if not symptom_list:
        return None
    
    # Severity mapping (simplified)
    severity_colors = {
        'bloating': '#f59e0b',
        'headache': '#ef4444',
        'fatigue': '#ef4444',
        'breast tenderness': '#f59e0b',
        'mood swings': '#f59e0b',
        'acne': '#3b82f6',
        'back pain': '#ef4444',
        'nausea': '#ef4444'
    }
    
    colors = [severity_colors.get(s, '#60a5fa') for s in symptom_list]
    
    fig = go.Figure(go.Bar(
        y=symptom_list,
        x=[1] * len(symptom_list),
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='#1e293b', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Present<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title="<b style='color:#FFFFFF;'>Current Symptoms Tracker</b>",
        xaxis_title="",
        yaxis_title="",
        height=max(250, len(symptom_list) * 40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        font=dict(family="Inter", color="#e2e8f0", size=12),
        margin=dict(l=150, r=50, t=60, b=50),
        xaxis=dict(showticklabels=False, showgrid=False)
    )
    
    fig.update_yaxes(
        showgrid=False
    )
    
    return fig


def build_wellness_gauge_comprehensive(wellness_score):
    """NEW: Comprehensive wellness score gauge"""
    
    # Determine color based on score
    if wellness_score >= 80:
        color = "#10b981"
        status = "Excellent"
    elif wellness_score >= 60:
        color = "#60a5fa"
        status = "Good"
    elif wellness_score >= 40:
        color = "#f59e0b"
        status = "Fair"
    else:
        color = "#ef4444"
        status = "Needs Attention"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=wellness_score,
        number={
            "suffix": "/110",
            "font": {"size": 48, "color": color, "family": "Inter", "weight": "bold"}
        },
        delta={
            'reference': 80,
            'increasing': {'color': "#10b981"},
            'decreasing': {'color': "#ef4444"}
        },
        gauge={
            "axis": {
                "range": [0, 110],
                "tickcolor": "#64748b",
                "tickfont": {"size": 11, "color": "#94a3b8"}
            },
            "bar": {
                "color": color,
                "thickness": 0.85,
                "line": {"color": "#1e293b", "width": 2}
            },
            "steps": [
                {"range": [0, 40], "color": "rgba(239, 68, 68, 0.12)"},
                {"range": [40, 60], "color": "rgba(245, 158, 11, 0.12)"},
                {"range": [60, 80], "color": "rgba(96, 165, 250, 0.12)"},
                {"range": [80, 110], "color": "rgba(16, 185, 129, 0.12)"},
            ],
            "threshold": {
                "line": {"color": "#a78bfa", "width": 3},
                "thickness": 0.85,
                "value": 80
            },
            "bgcolor": "rgba(0,0,0,0)"
        },
        title={
            "text": f"<b style='color:#FFFFFF'>Comprehensive Wellness Score</b><br><span style='font-size:14px;color:#cbd5e1'>Status: {status}</span>",
            "font": {"size": 17}
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0")
    )
    
    return fig


def build_bmi_visualization(weight, height, bmi):
    """NEW: BMI visualization with healthy range indicators"""
    
    bmi_category = get_bmi_category(bmi)
    
    # Determine color based on BMI
    if bmi < 18.5:
        color = "#f59e0b"
    elif bmi < 25:
        color = "#10b981"
    elif bmi < 30:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        number={
            "font": {"size": 52, "color": color, "family": "Inter", "weight": "bold"}
        },
        gauge={
            "axis": {
                "range": [15, 40],
                "tickcolor": "#64748b",
                "tickfont": {"size": 10, "color": "#94a3b8"}
            },
            "bar": {
                "color": color,
                "thickness": 0.8,
                "line": {"color": "#1e293b", "width": 2}
            },
            "steps": [
                {"range": [15, 18.5], "color": "rgba(245, 158, 11, 0.15)", "name": "Underweight"},
                {"range": [18.5, 25], "color": "rgba(16, 185, 129, 0.15)", "name": "Normal"},
                {"range": [25, 30], "color": "rgba(245, 158, 11, 0.15)", "name": "Overweight"},
                {"range": [30, 40], "color": "rgba(239, 68, 68, 0.15)", "name": "Obese"},
            ],
            "threshold": {
                "line": {"color": "#e2e8f0", "width": 3},
                "thickness": 0.85,
                "value": bmi
            },
            "bgcolor": "rgba(0,0,0,0)"
        },
        title={
            "text": f"<b style='color:#FFFFFF'>Body Mass Index (BMI)</b><br><span style='font-size:13px;color:#cbd5e1'>{weight}kg / {height}cm = {bmi_category}</span>",
            "font": {"size": 16}
        }
    ))
    
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=75, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0")
    )
    
    return fig


def build_pain_mood_correlation(cramp_severity, mood_state):
    """NEW: Show relationship between pain and mood"""
    
    mood_scores = {
        "excellent": 5,
        "good": 4,
        "neutral": 3,
        "low": 2,
        "anxious": 2,
        "depressed": 1
    }
    
    mood_score = mood_scores.get(mood_state, 3)
    
    fig = go.Figure()
    
    # Pain level
    fig.add_trace(go.Bar(
        name='Cramp Severity',
        x=['Pain & Mood Assessment'],
        y=[cramp_severity],
        marker=dict(color='#ef4444'),
        text=[f"Pain: {cramp_severity}/10"],
        textposition='outside',
        yaxis='y',
        offsetgroup=1
    ))
    
    # Mood level
    fig.add_trace(go.Bar(
        name='Mood Score',
        x=['Pain & Mood Assessment'],
        y=[mood_score],
        marker=dict(color='#60a5fa'),
        text=[f"Mood: {mood_state.title()}"],
        textposition='outside',
        yaxis='y2',
        offsetgroup=2
    ))
    
    fig.update_layout(
        title="<b style='color:#FFFFFF;'>Pain & Mood Correlation</b>",
        yaxis=dict(
            title=dict(text="Cramp Severity (0-10)", font=dict(color="#ef4444")),
            tickfont=dict(color="#ef4444"),
            range=[0, 10],
            showgrid=True,
            gridcolor='rgba(239, 68, 68, 0.1)'
        ),
        yaxis2=dict(
            title=dict(text="Mood Score (1-5)", font=dict(color="#60a5fa")),
            tickfont=dict(color="#60a5fa"),
            overlaying='y',
            side='right',
            range=[0, 5],
            showgrid=False
        ),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        font=dict(family="Inter", color="#e2e8f0", size=12),
        barmode='group',
        showlegend=True,
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor="rgba(59, 130, 246, 0.3)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis=dict(showticklabels=False)
    )
    
    return fig


def encode_input(cycle_length, period_duration, sleep_hours, flow_level, stress_level):
    """Encode user inputs into model-compatible format"""
    df = pd.DataFrame({
        "cycle_length": [cycle_length],
        "period_duration": [period_duration],
        "sleep_hours": [sleep_hours],
        "flow_level_light": [1 if flow_level == "light" else 0],
        "flow_level_medium": [1 if flow_level == "medium" else 0],
        "stress_level_low": [1 if stress_level == "low" else 0],
        "stress_level_medium": [1 if stress_level == "medium" else 0],
    })

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        df = df[expected]

    return df


# ═══════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ═══════════════════════════════════════════════════════════════════
def login_screen():
    """Render professional login interface"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<h2 class='app-title' style='text-align:center;'>🩺 Women Health Insight System</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p class='app-subtitle' style='text-align:center;'>AI-powered cycle prediction and health monitoring platform</p>",
            unsafe_allow_html=True
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("<div class='section-title' style='justify-content:center;'>🔐 Secure Login</div>", unsafe_allow_html=True)
        
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            do_login = st.button("➡️  Login", type="primary", use_container_width=True)
        with col_btn2:
            st.markdown(
                "<div style='padding:0.75rem 0; text-align:center; color:#94a3b8; font-size:13px;'>"
                "Demo: <b style='color:#60a5fa;'>admin / admin123</b></div>",
                unsafe_allow_html=True
            )
        
        if do_login:
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.success("✅ Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_screen()
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# MAIN APPLICATION HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 class='app-title'>🩺 Women Health Insight System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p class='app-subtitle'>Advanced AI-powered predictions • Comprehensive analytics • Professional reporting</p>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR - STRUCTURED INPUT FORM
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:1rem 0; border-bottom:1px solid rgba(59, 130, 246, 0.3); margin-bottom:1.5rem;'>"
        "<h3 style='margin:0; color:#e2e8f0; font-weight:700;'>📋 Patient Assessment</h3>"
        "</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("<div class='sidebar-section-header'>👤 Patient Information</div>", unsafe_allow_html=True)
    patient_name = st.text_input("Patient Name", value="Patient", placeholder="Enter patient name")
    patient_id = st.text_input("Patient ID", value="P-0001", placeholder="e.g., P-0001")
    age = st.number_input("Age (years)", min_value=10, max_value=80, value=22, help="Patient's age in years")
    
    st.markdown("<div class='sidebar-section-header'>🩺 Cycle Parameters</div>", unsafe_allow_html=True)
    
    cycle_length = st.slider(
        "Cycle Length (days)",
        min_value=20,
        max_value=60,
        value=28,
        help="Average menstrual cycle length"
    )
    
    period_duration = st.slider(
        "Period Duration (days)",
        min_value=1,
        max_value=10,
        value=5,
        help="Average duration of menstrual period"
    )
    
    flow_level = st.selectbox(
        "Flow Level",
        options=["light", "medium", "heavy"],
        index=1,
        help="Typical menstrual flow intensity"
    )
    
    st.markdown("<div class='sidebar-section-header'>💪 Lifestyle & Wellness</div>", unsafe_allow_html=True)
    
    sleep_hours = st.slider(
        "Sleep Hours (per night)",
        min_value=0.0,
        max_value=12.0,
        value=7.0,
        step=0.5,
        help="Average hours of sleep per night"
    )
    
    stress_level = st.selectbox(
        "Stress Level",
        options=["low", "medium", "high"],
        index=1,
        help="Current stress level assessment"
    )
    
    exercise_frequency = st.selectbox(
        "Exercise Frequency",
        options=["sedentary", "light (1-2 days/week)", "moderate (3-4 days/week)", "active (5+ days/week)"],
        index=1,
        help="How often patient exercises per week"
    )
    
    water_intake = st.slider(
        "Water Intake (glasses/day)",
        min_value=0,
        max_value=15,
        value=6,
        help="Average daily water consumption"
    )
    
    diet_quality = st.selectbox(
        "Diet Quality",
        options=["poor", "fair", "good", "excellent"],
        index=2,
        help="Overall nutritional quality of diet"
    )
    
    st.markdown("<div class='sidebar-section-header'>📊 Physical Metrics</div>", unsafe_allow_html=True)
    
    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=60.0,
        step=0.1,
        help="Current body weight in kilograms"
    )
    
    height = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=220.0,
        value=165.0,
        step=0.1,
        help="Height in centimeters"
    )
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    st.info(f"📏 **BMI:** {bmi:.1f} ({get_bmi_category(bmi)})")
    
    st.markdown("<div class='sidebar-section-header'>🩺 Medical History</div>", unsafe_allow_html=True)
    
    contraceptive_use = st.selectbox(
        "Contraceptive Use",
        options=["none", "oral contraceptive", "IUD", "implant", "other"],
        index=0,
        help="Current contraceptive method"
    )
    
    has_pcos = st.checkbox("PCOS Diagnosis", value=False, help="Polycystic Ovary Syndrome")
    has_endometriosis = st.checkbox("Endometriosis", value=False, help="Endometriosis diagnosis")
    has_thyroid = st.checkbox("Thyroid Issues", value=False, help="Thyroid disorder")
    
    st.markdown("<div class='sidebar-section-header'>😊 Symptoms & Mood</div>", unsafe_allow_html=True)
    
    mood_state = st.selectbox(
        "Current Mood",
        options=["excellent", "good", "neutral", "low", "anxious", "depressed"],
        index=1,
        help="Overall emotional state"
    )
    
    cramp_severity = st.slider(
        "Cramp Severity (0-10)",
        min_value=0,
        max_value=10,
        value=3,
        help="Pain level during menstruation"
    )
    
    symptoms = st.multiselect(
        "Current Symptoms",
        options=["bloating", "headache", "fatigue", "breast tenderness", "mood swings", 
                 "acne", "back pain", "nausea", "none"],
        default=["none"],
        help="Select all applicable symptoms"
    )
    
    st.markdown("<div class='sidebar-section-header'>📝 Clinical Notes</div>", unsafe_allow_html=True)
    notes = st.text_area(
        "Doctor's Observations",
        height=100,
        placeholder="Enter any relevant clinical observations or notes...",
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='sidebar-section-header'>⚙️ Settings</div>", unsafe_allow_html=True)
    save_mode = st.radio(
        "Save History To",
        options=["SQLite (recommended)", "CSV"],
        help="Choose how to store patient history"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_predict, col_logout = st.columns([3, 1])
    with col_predict:
        run = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
    with col_logout:
        if st.button("🚪", use_container_width=True, help="Logout"):
            st.session_state.logged_in = False
            st.rerun()


# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ═══════════════════════════════════════════════════════════════════

if run:
    with st.spinner("🔄 Running AI prediction model..."):
        input_df = encode_input(cycle_length, period_duration, sleep_hours, flow_level, stress_level)
        pred = model.predict(input_df)
        pred_days = max(0.0, float(pred[0]))
        
        risk = risk_level(pred_days)
        interp = interpretation(pred_days)
        
        # Calculate comprehensive wellness score
        wellness_score = get_wellness_score(
            cycle_length, period_duration, sleep_hours, stress_level,
            exercise_frequency, water_intake, diet_quality, bmi,
            cramp_severity, has_pcos, has_endometriosis, has_thyroid
        )
        
        # Calculate health score (legacy)
        health_score_fig, health_score = build_health_score_gauge(
            cycle_length, period_duration, sleep_hours, stress_level
        )
        
        # Generate personalized recommendations
        recommendations = generate_personalized_recommendations(
            wellness_score, cycle_length, period_duration, sleep_hours,
            stress_level, exercise_frequency, water_intake, diet_quality,
            bmi, cramp_severity, has_pcos, has_endometriosis, has_thyroid,
            mood_state, symptoms, pred_days, age, contraceptive_use
        )
    
    st.success("✅ Prediction completed successfully!")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # KPI SUMMARY ROW
    # ═══════════════════════════════════════════════════════════════
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(
            f"<div class='kpi-container'>"
            f"<div class='kpi-value'>{pred_days:.1f}</div>"
            f"<div class='kpi-label'>Predicted Delay (days)</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with kpi2:
        st.markdown(
            f"<div class='kpi-container'>"
            f"<div class='kpi-value'>{wellness_score:.0f}</div>"
            f"<div class='kpi-label'>Wellness Score</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with kpi3:
        st.markdown(
            f"<div class='kpi-container'>"
            f"<div class='kpi-value' style='font-size:28px;'>{risk}</div>"
            f"<div class='kpi-label'>Risk Level</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with kpi4:
        bmi_cat = get_bmi_category(bmi)
        st.markdown(
            f"<div class='kpi-container'>"
            f"<div class='kpi-value' style='font-size:22px;'>{bmi:.1f}</div>"
            f"<div class='kpi-label'>BMI ({bmi_cat})</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Risk badge
    st.markdown(f"<div style='text-align:center;'>{risk_badge(risk)}</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # ANALYTICS DASHBOARD - 2x2 GRID
    # ═══════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>📊 Advanced Analytics Dashboard</div>", unsafe_allow_html=True)
    
    row1_col1, row1_col2 = st.columns(2, gap="large")
    
    with row1_col1:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.plotly_chart(build_gauge(pred_days, risk), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with row1_col2:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.plotly_chart(build_wellness_gauge_comprehensive(wellness_score), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    row2_col1, row2_col2 = st.columns(2, gap="large")
    
    with row2_col1:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.plotly_chart(build_bmi_visualization(weight, height, bmi), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with row2_col2:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.plotly_chart(build_lifestyle_score_breakdown(exercise_frequency, water_intake, 
                                                        diet_quality, sleep_hours, stress_level), 
                       use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # NEW: ADDITIONAL ANALYTICS ROW
    # ═══════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>🔬 Detailed Health Analysis</div>", unsafe_allow_html=True)
    
    analysis_col1, analysis_col2 = st.columns(2, gap="large")
    
    with analysis_col1:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.plotly_chart(build_radar_chart(cycle_length, period_duration, sleep_hours,
                                          flow_level, stress_level), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with analysis_col2:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.plotly_chart(build_pain_mood_correlation(cramp_severity, mood_state), 
                       use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Symptom tracker row
    if symptoms and symptoms != ["none"]:
        st.markdown("<br>", unsafe_allow_html=True)
        symptom_col1, symptom_col2 = st.columns(2, gap="large")
        
        with symptom_col1:
            st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
            symptom_fig = build_symptom_frequency_chart(symptoms)
            if symptom_fig:
                st.plotly_chart(symptom_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with symptom_col2:
            st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
            st.plotly_chart(build_health_metrics_comparison(cycle_length, period_duration, 
                                                            sleep_hours, flow_level, stress_level), 
                           use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        single_col = st.columns(1)[0]
        with single_col:
            st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
            st.plotly_chart(build_health_metrics_comparison(cycle_length, period_duration, 
                                                            sleep_hours, flow_level, stress_level), 
                           use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # PERSONALIZED RECOMMENDATIONS SECTION
    # ═══════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>💡 Personalized Health Recommendations</div>", unsafe_allow_html=True)
    
    if recommendations:
        # Group recommendations by category
        categories = {}
        for rec in recommendations:
            cat = rec['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(rec)
        
        # Display recommendations in organized sections
        for category, recs in categories.items():
            with st.expander(f"**{category}** ({len(recs)} recommendation{'s' if len(recs) > 1 else ''})", 
                           expanded=('High Priority' in category or 'Overall Wellness' in category)):
                for rec in recs:
                    st.markdown(
                        f"<div class='card-accent' style='margin:1rem 0;'>"
                        f"<h4 style='color:#FFFFFF; margin:0 0 0.75rem 0; font-size:18px; font-weight:700;'>📌 {rec['title']}</h4>"
                        f"<p style='color:#e2e8f0; margin:0 0 0.75rem 0; line-height:1.8; font-size:15px;'>{rec['advice']}</p>"
                        f"<div style='background:rgba(59, 130, 246, 0.15); padding:1rem; border-radius:10px; "
                        f"border-left:4px solid #60a5fa;'>"
                        f"<strong style='color:#93c5fd; font-size:14px; font-weight:700;'>💡 ACTION STEP:</strong> "
                        f"<span style='color:#FFFFFF; font-size:14px; font-weight:500;'>{rec['action']}</span>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
    else:
        st.info("✨ No specific recommendations at this time. Keep up the good work!")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # HISTORICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    hist_df = load_history(limit=50)
    
    if not hist_df.empty:
        st.markdown("<div class='section-title'>📈 Historical Trends & Insights</div>", unsafe_allow_html=True)
        
        hist_col1, hist_col2 = st.columns(2, gap="large")
        
        with hist_col1:
            st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
            trend_fig = build_trend_chart(hist_df)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with hist_col2:
            st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
            risk_dist_fig = build_risk_distribution_chart(hist_df)
            if risk_dist_fig:
                st.plotly_chart(risk_dist_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional historical analysis row
        if len(hist_df) >= 3:
            st.markdown("<br>", unsafe_allow_html=True)
            
            hist_col3, hist_col4 = st.columns(2, gap="large")
            
            with hist_col3:
                st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
                sleep_stress_fig = build_sleep_stress_analysis(hist_df)
                if sleep_stress_fig:
                    st.plotly_chart(sleep_stress_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with hist_col4:
                st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
                regularity_fig = build_cycle_regularity_chart(hist_df)
                if regularity_fig:
                    st.plotly_chart(regularity_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # DETAILED REPORTS & DATA
    # ═══════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>📋 Detailed Reports</div>", unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 Full Report", "📥 Input Summary", "📈 Patient History", "💾 Save & Export"])
    
    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        report_df = pd.DataFrame({
            "Field": [
                "Patient Name", "Patient ID", "Age",
                "Cycle Length (days)", "Period Duration (days)", "Sleep Hours",
                "Flow Level", "Stress Level",
                "Predicted Delay (days)", "Risk Level", "Health Score", "Clinical Interpretation",
                "Clinical Notes"
            ],
            "Value": [
                patient_name, patient_id, age,
                cycle_length, period_duration, sleep_hours,
                flow_level.capitalize(), stress_level.capitalize(),
                f"{pred_days:.1f}", risk, f"{health_score:.0f}%", interp,
                notes.strip() if notes.strip() else "No additional notes"
            ]
        })
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        inp_show = pd.DataFrame({
            "Parameter": [
                "Cycle Length",
                "Period Duration",
                "Sleep Hours",
                "Flow Level",
                "Stress Level"
            ],
            "Value": [
                f"{cycle_length} days",
                f"{period_duration} days",
                f"{sleep_hours} hours",
                flow_level.capitalize(),
                stress_level.capitalize()
            ],
            "Status": [
                "Normal" if 24 <= cycle_length <= 35 else "Attention needed",
                "Normal" if period_duration <= 7 else "Attention needed",
                "Good" if sleep_hours >= 7 else "Insufficient",
                "Monitored",
                "Monitored"
            ]
        })
        st.dataframe(inp_show, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if hist_df.empty:
            st.info("ℹ️ No patient history available yet. This will populate after saving records.")
        else:
            st.dataframe(hist_df.head(15), use_container_width=True, hide_index=True)
            if len(hist_df) > 15:
                st.caption(f"Showing 15 of {len(hist_df)} total records")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_id": patient_id,
            "patient_name": patient_name,
            "age": int(age),
            "cycle_length": float(cycle_length),
            "period_duration": float(period_duration),
            "sleep_hours": float(sleep_hours),
            "flow_level": flow_level,
            "stress_level": stress_level,
            "predicted_delay": float(pred_days),
            "risk_level": risk,
            "interpretation": interp,
            "notes": notes.strip()
        }
        
        col_save1, col_save2 = st.columns(2)
        
        with col_save1:
            if st.button("💾 Save to Database", use_container_width=True):
                with st.spinner("💾 Saving patient record..."):
                    if save_mode.startswith("SQLite"):
                        save_to_sqlite(record)
                        st.success("✅ Record saved to SQLite database")
                    else:
                        save_to_csv(record)
                        st.success("✅ Record saved to CSV file")
        
        with col_save2:
            with st.spinner("📄 Generating PDF report..."):
                pdf_path = make_report_path(patient_name, patient_id)
                generate_pdf_report(
                    out_path=pdf_path,
                    patient={"name": patient_name, "id": patient_id, "age": age},
                    inputs={
                        "cycle_length": cycle_length,
                        "period_duration": period_duration,
                        "sleep_hours": sleep_hours,
                        "flow_level": flow_level,
                        "stress_level": stress_level,
                        "exercise_frequency": exercise_frequency,
                        "water_intake": water_intake,
                        "diet_quality": diet_quality,
                        "weight": weight,
                        "height": height,
                        "bmi": bmi,
                        "contraceptive_use": contraceptive_use,
                        "has_pcos": has_pcos,
                        "has_endometriosis": has_endometriosis,
                        "has_thyroid": has_thyroid,
                        "mood_state": mood_state,
                        "cramp_severity": cramp_severity,
                        "symptoms": symptoms
                    },
                    prediction={
                        "predicted_delay": pred_days,
                        "risk_level": risk,
                        "interpretation": interp,
                        "wellness_score": wellness_score,
                        "bmi_category": get_bmi_category(bmi),
                        "recommendations": recommendations,
                        "notes": notes
                    }
                )
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    use_container_width=True
                )
        
        st.markdown(
            f"<div style='font-size:11px; color:#94a3b8; margin-top:1rem; text-align:center;'>"
            f"📁 Reports saved to: <code style='color:#60a5fa;'>data/db_data/reports/</code></div>",
            unsafe_allow_html=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # ═══════════════════════════════════════════════════════════════
    # GETTING STARTED SCREEN
    # ═══════════════════════════════════════════════════════════════
    # st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("<div class='info-box-title'>🚀 Getting Started</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box-content'>"
        "<p><strong>Welcome to the Women Health Insight System!</strong></p>"
        "<p>To generate a comprehensive health assessment:</p>"
        "<ol style='margin:0.75rem 0; padding-left:1.5rem; line-height:2;'>"
        "<li>Enter patient information in the sidebar (name, ID, age)</li>"
        "<li>Adjust health parameters using the sliders and dropdowns</li>"
        "<li>Add any clinical notes or observations in the text area</li>"
        "<li>Click the <strong style='color:#60a5fa;'>🚀 Run Prediction</strong> button</li>"
        "</ol>"
        "<p style='margin-top:1.5rem;'>The AI model will analyze the inputs and provide:</p>"
        "<ul style='margin:0.5rem 0; padding-left:1.5rem; line-height:2;'>"
        "<li>Cycle delay predictions with risk assessment</li>"
        "<li>Comprehensive health score analysis</li>"
        "<li>Visual analytics and trend reports</li>"
        "<li>Downloadable PDF reports for patient records</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("<div class='section-title'>✨ Key Features</div>", unsafe_allow_html=True)
    
    feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
    
    with feat_col1:
        st.markdown(
            "<div class='analytics-card' style='text-align:center;'>"
            "<div style='font-size:48px; margin-bottom:1rem;'>🎯</div>"
            "<h3 style='color:#60a5fa; font-size:18px; margin:0.5rem 0;'>AI Predictions</h3>"
            "<p style='color:#94a3b8; font-size:13px; line-height:1.6;'>Advanced machine learning for accurate cycle delay forecasting</p>"
            "</div>",
            unsafe_allow_html=True
        )
    
    with feat_col2:
        st.markdown(
            "<div class='analytics-card' style='text-align:center;'>"
            "<div style='font-size:48px; margin-bottom:1rem;'>📊</div>"
            "<h3 style='color:#60a5fa; font-size:18px; margin:0.5rem 0;'>Visual Analytics</h3>"
            "<p style='color:#94a3b8; font-size:13px; line-height:1.6;'>Interactive charts and comprehensive data visualization</p>"
            "</div>",
            unsafe_allow_html=True
        )
    
    with feat_col3:
        st.markdown(
            "<div class='analytics-card' style='text-align:center;'>"
            "<div style='font-size:48px; margin-bottom:1rem;'>🏥</div>"
            "<h3 style='color:#60a5fa; font-size:18px; margin:0.5rem 0;'>Health Scoring</h3>"
            "<p style='color:#94a3b8; font-size:13px; line-height:1.6;'>Holistic wellness evaluation and risk assessment</p>"
            "</div>",
            unsafe_allow_html=True
        )
    
    with feat_col4:
        st.markdown(
            "<div class='analytics-card' style='text-align:center;'>"
            "<div style='font-size:48px; margin-bottom:1rem;'>📄</div>"
            "<h3 style='color:#60a5fa; font-size:18px; margin:0.5rem 0;'>PDF Reports</h3>"
            "<p style='color:#94a3b8; font-size:13px; line-height:1.6;'>Professional documentation for patient records</p>"
            "</div>",
            unsafe_allow_html=True
        )