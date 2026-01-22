import os
from datetime import datetime

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import save_to_csv, save_to_sqlite, load_history, make_report_path
from report import generate_pdf_report


# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Women Health Insight System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "delay_predictor_model.pkl")

# ---------------- Light Theme CSS (BORDERS + visibility) ----------------
CUSTOM_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at 20% 20%, #FFFFFF 0%, #F3F6FB 45%, #EEF3FA 100%);
    color: #101828;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.2rem;
    max-width: 1250px;
}

section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 2px solid #E6EAF2;
    box-shadow: 6px 0 22px rgba(16, 24, 40, 0.06);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.0rem;
}

.card {
    background: #FFFFFF;
    border: 1px solid #E4EAF4;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 22px rgba(16, 24, 40, 0.08);
}

.section-title {
    font-size: 18px;
    font-weight: 900;
    margin: 0 0 12px 0;
    color: #0F172A;
}

.small {
    font-size: 12px;
    color: #64748B;
}

.kpi {
    font-size: 34px;
    font-weight: 900;
    margin: 0;
    line-height: 1.0;
    color: #0F172A;
}
.kpi-label {
    font-size: 13px;
    color: #475569;
    margin-top: 6px;
    font-weight: 800;
}

.badge-low, .badge-mod, .badge-high {
    display:inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 900;
    font-size: 12px;
}
.badge-low { background: #ECFDF3; border: 1px solid #A6F4C5; color: #067647; }
.badge-mod { background: #FFFAEB; border: 1px solid #FEDF89; color: #B54708; }
.badge-high { background: #FEF3F2; border: 1px solid #FECDCA; color: #B42318; }

.stButton>button {
    border-radius: 14px;
    padding: 10px 18px;
    font-weight: 900;
    border: 1px solid #CBD5E1;
    box-shadow: 0 8px 18px rgba(16, 24, 40, 0.08);
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
textarea {
    border-radius: 14px !important;
    border: 1px solid #E2E8F0 !important;
    background: #FFFFFF !important;
}

button[data-baseweb="tab"] {
    font-weight: 900;
    border-radius: 14px;
}
div[data-baseweb="tab-border"] {
    background: #E2E8F0;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    overflow: hidden;
}

hr {
    border: none;
    height: 1px;
    background: #E2E8F0;
}

.plotly-graph-div > div {
    background: transparent !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()


# ---------------- Helpers ----------------
def risk_level(days: float) -> str:
    if days <= 1.5:
        return "Low"
    if days <= 4:
        return "Moderate"
    return "High"


def risk_badge(risk: str) -> str:
    if risk == "Low":
        return "<span class='badge-low'>🟢 Low Risk</span>"
    if risk == "Moderate":
        return "<span class='badge-mod'>🟡 Moderate Risk</span>"
    return "<span class='badge-high'>🔴 High Risk</span>"


def interpretation(days: float) -> str:
    if days <= 1.5:
        return "Normal variation ✅"
    if days <= 4:
        return "Slight delay likely ⚠️"
    return "Irregularity risk — monitor closely 🚨"


def build_gauge(delay_days: float, risk: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=delay_days,
        number={"suffix": " days", "font": {"size": 40, "color": "#0F172A"}},
        gauge={
            "axis": {"range": [0, 10], "tickcolor": "#94A3B8"},
            "bar": {"color": "#0F172A"},
            "steps": [
                {"range": [0, 2], "color": "#ECFDF3"},
                {"range": [2, 5], "color": "#FFFAEB"},
                {"range": [5, 10], "color": "#FEF3F2"},
            ],
            "threshold": {
                "line": {"color": "#0F172A", "width": 4},
                "thickness": 0.75,
                "value": delay_days
            }
        },
        title={"text": f"<b>🧭 Cycle Delay Gauge</b><br><span style='font-size:13px;color:#475569'>Risk: {risk}</span>"}
    ))
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A")
    )
    return fig


def encode_input(cycle_length, period_duration, sleep_hours, flow_level, stress_level):
    # MUST match training columns
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


def login_screen():
    # st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown(
        "<h2 style='margin:0;color:#0F172A'>🩺 Women Health Insight System</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='small'>AI-assisted dashboard for cycle delay prediction & reporting.</div>",
        unsafe_allow_html=True
    )
    st.write("")

    st.markdown("<div class='section-title'>🔐 Login</div>", unsafe_allow_html=True)
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")

    c1, c2 = st.columns([1, 2])
    with c1:
        do_login = st.button("➡️ Login", type="primary")
    with c2:
        st.markdown("<div class='small'>Demo: <b>admin / admin123</b></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if do_login:
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("✅ Login successful.")
            st.rerun()
        else:
            st.error("❌ Invalid credentials.")


# ---------------- Session State ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_screen()
    st.stop()


# ---------------- Header ----------------
# st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown(
    "<h1 style='margin:0;color:#0F172A'>🩺 Women Health Insight System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='small'>🏥 Clinic-style dashboard • 📊 Prediction • 🧾 Reporting • 🗂️ Patient History</div>",
    unsafe_allow_html=True
)
# st.markdown("</div>", unsafe_allow_html=True)
st.write("")


# ---------------- Sidebar ----------------
st.sidebar.markdown("## 🧑‍⚕️ Patient Info")
patient_name = st.sidebar.text_input("🧾 Patient Name", value="Patient")
patient_id = st.sidebar.text_input("🆔 Patient ID", value="P-0001")
age = st.sidebar.number_input("🎂 Age", min_value=10, max_value=80, value=22)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🩺 Health Inputs")

cycle_length = st.sidebar.slider("📅 Cycle Length (days)", 20, 60, 28)
period_duration = st.sidebar.slider("🩸 Period Duration (days)", 1, 10, 5)
sleep_hours = st.sidebar.slider("😴 Sleep Hours", 0.0, 12.0, 7.0, 0.5)
flow_level = st.sidebar.selectbox("💧 Flow Level", ["light", "medium", "heavy"])
stress_level = st.sidebar.selectbox("😣 Stress Level", ["low", "medium", "high"])

st.sidebar.markdown("---")
notes = st.sidebar.text_area("📝 Doctor Notes / Observations", height=120)

save_mode = st.sidebar.radio("💾 Save History To", ["SQLite (recommended)", "CSV"], horizontal=False)

col_run, col_logout = st.sidebar.columns(2)
with col_run:
    run = st.button("🚀 Predict", type="primary")
with col_logout:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()


# ---------------- Main Layout ----------------
left, right = st.columns([1.05, 1.25], gap="large")

if run:
    input_df = encode_input(cycle_length, period_duration, sleep_hours, flow_level, stress_level)
    pred = model.predict(input_df)
    pred_days = max(0.0, float(pred[0]))

    risk = risk_level(pred_days)
    interp = interpretation(pred_days)

    # -------- LEFT --------
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📌 Prediction Summary</div>", unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"<div class='kpi'>{pred_days:.1f}</div><div class='kpi-label'>⏳ Predicted Delay</div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi'>{risk}</div><div class='kpi-label'>🚦 Risk Level</div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi'>{interp}</div><div class='kpi-label'>🧾 Interpretation</div>", unsafe_allow_html=True)

        st.markdown(f"<div style='margin-top:10px'>{risk_badge(risk)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        tabs = st.tabs(["🧾 Doctor Report", "📥 Inputs", "🗂️ History"])

        with tabs[0]:
            st.subheader("🧾 Doctor-friendly Report")
            report_df = pd.DataFrame({
                "Field": [
                    "Patient Name", "Patient ID", "Age",
                    "Cycle Length", "Period Duration", "Sleep Hours",
                    "Flow Level", "Stress Level",
                    "Predicted Delay (days)", "Risk Level", "Interpretation",
                    "Notes"
                ],
                "Value": [
                    patient_name, patient_id, age,
                    cycle_length, period_duration, sleep_hours,
                    flow_level, stress_level,
                    f"{pred_days:.1f}", risk, interp,
                    notes.strip() if notes.strip() else "-"
                ]
            })
            st.dataframe(report_df, width="stretch", hide_index=True)

        with tabs[1]:
            st.subheader("📥 Input Summary")
            inp_show = pd.DataFrame({
                "Parameter": ["Cycle Length", "Period Duration", "Sleep Hours", "Flow Level", "Stress Level"],
                "Value": [cycle_length, period_duration, sleep_hours, flow_level, stress_level]
            })
            st.dataframe(inp_show, width="stretch", hide_index=True)

        with tabs[2]:
            st.subheader("🗂️ Patient History")
            hist = load_history(limit=200)
            if hist.empty:
                st.info("ℹ️ No patient history found yet.")
            else:
                st.dataframe(hist, width="stretch", hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- RIGHT --------
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(build_gauge(pred_days, risk), use_container_width=True)

        st.markdown("<div class='section-title'>📊 Parameter Snapshot</div>", unsafe_allow_html=True)
        chart_df = pd.DataFrame(
            {"value": [cycle_length, period_duration, sleep_hours]},
            index=["cycle_length", "period_duration", "sleep_hours"]
        )
        st.bar_chart(chart_df, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📄 Report & Storage</div>", unsafe_allow_html=True)

        # Save history
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

        if save_mode.startswith("SQLite"):
            save_to_sqlite(record)
            st.success("✅ Saved to SQLite patient history.")
        else:
            save_to_csv(record)
            st.success("✅ Saved to CSV patient history.")

        # Save report PDF in db_data/reports
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
            },
            prediction={
                "predicted_delay": pred_days,
                "risk_level": risk,
                "interpretation": interp,
                "notes": notes
            }
        )

        st.markdown("<div class='small'>📁 Saved at: <b>data/db_data/reports/</b></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=f,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )

else:
    
    st.markdown("<div class='section-title'>🚀 Getting Started</div>", unsafe_allow_html=True)
    st.info("👉 Enter patient details & health inputs in sidebar, then click **Predict**.")
    
