import os
import re
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_DATA_DIR = os.path.join(BASE_DIR, "data", "db_data")
REPORTS_DIR = os.path.join(DB_DATA_DIR, "reports")

CSV_PATH = os.path.join(DB_DATA_DIR, "patient_history.csv")
SQLITE_PATH = os.path.join(DB_DATA_DIR, "patient_history.db")


def ensure_dirs():
    os.makedirs(DB_DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def sanitize_filename(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    return text or "patient"


def init_sqlite():
    """
    Creates table if not exists.
    Also auto-migrates missing columns (so old DB won't crash).
    """
    ensure_dirs()
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()

    # base schema
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patient_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            patient_id TEXT,
            patient_name TEXT,
            age INTEGER,
            cycle_length REAL,
            period_duration REAL,
            sleep_hours REAL,
            flow_level TEXT,
            stress_level TEXT,
            predicted_delay REAL,
            risk_level TEXT,
            interpretation TEXT,
            notes TEXT
        )
        """
    )

    # --- auto migration for old DB ---
    cur.execute("PRAGMA table_info(patient_history)")
    existing_cols = {row[1] for row in cur.fetchall()}

    required_cols = {
        "timestamp": "TEXT",
        "patient_id": "TEXT",
        "patient_name": "TEXT",
        "age": "INTEGER",
        "cycle_length": "REAL",
        "period_duration": "REAL",
        "sleep_hours": "REAL",
        "flow_level": "TEXT",
        "stress_level": "TEXT",
        "predicted_delay": "REAL",
        "risk_level": "TEXT",
        "interpretation": "TEXT",
        "notes": "TEXT",
    }

    for col, col_type in required_cols.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE patient_history ADD COLUMN {col} {col_type}")

    con.commit()
    con.close()


def save_to_sqlite(record: dict):
    init_sqlite()
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()

    cur.execute(
        """
        INSERT INTO patient_history (
            timestamp, patient_id, patient_name, age,
            cycle_length, period_duration, sleep_hours,
            flow_level, stress_level,
            predicted_delay, risk_level, interpretation, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.get("timestamp"),
            record.get("patient_id"),
            record.get("patient_name"),
            record.get("age"),
            record.get("cycle_length"),
            record.get("period_duration"),
            record.get("sleep_hours"),
            record.get("flow_level"),
            record.get("stress_level"),
            record.get("predicted_delay"),
            record.get("risk_level"),
            record.get("interpretation"),
            record.get("notes"),
        ),
    )

    con.commit()
    con.close()


def save_to_csv(record: dict):
    ensure_dirs()
    df = pd.DataFrame([record])

    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(CSV_PATH, index=False)


def load_history(limit=200) -> pd.DataFrame:
    ensure_dirs()

    if os.path.exists(SQLITE_PATH):
        con = sqlite3.connect(SQLITE_PATH)
        df = pd.read_sql_query(
            f"SELECT * FROM patient_history ORDER BY id DESC LIMIT {int(limit)}", con
        )
        con.close()
        return df

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        return df.tail(limit).iloc[::-1].reset_index(drop=True)

    return pd.DataFrame()


def make_report_path(patient_name: str, patient_id: str) -> str:
    """
    Save reports in: data/db_data/reports/
    Format:
      report_<patientname>.pdf
      if duplicate name exists -> report_<patientname>_<patientid>.pdf
      if still duplicate -> add suffix _2, _3 ...
    """
    ensure_dirs()

    safe_name = sanitize_filename(patient_name)
    safe_id = sanitize_filename(patient_id)

    candidate1 = os.path.join(REPORTS_DIR, f"report_{safe_name}.pdf")
    if not os.path.exists(candidate1):
        return candidate1

    candidate2 = os.path.join(REPORTS_DIR, f"report_{safe_name}_{safe_id}.pdf")
    if not os.path.exists(candidate2):
        return candidate2

    i = 2
    while True:
        candidate3 = os.path.join(REPORTS_DIR, f"report_{safe_name}_{safe_id}_{i}.pdf")
        if not os.path.exists(candidate3):
            return candidate3
        i += 1
