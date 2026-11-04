
import os
import hmac
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
)


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "user_activity.csv"
UPLOAD_PATH = DATA_DIR / "uploaded_activity.csv"

ISOLATION_MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest.joblib"

FEATURES = [
    "hour",
    "login_failures",
    "files_accessed",
    "data_transfer_mb",
    "usb_transfer",
    "off_hours",
]

# Create data directory if it does not exist.
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# ------------------------------------------------------------
# FLASK SECRET KEY
# ------------------------------------------------------------

SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")

if not SECRET_KEY:
    if os.environ.get("RENDER"):
        raise RuntimeError(
            "FLASK_SECRET_KEY must be configured on Render."
        )

    SECRET_KEY = "local-development-only-change-this-secret"

app.secret_key = SECRET_KEY

# ------------------------------------------------------------
# ADMIN LOGIN CREDENTIALS
# ------------------------------------------------------------

ADMIN_EMAIL = os.environ.get(
    "ADMIN_EMAIL",
    "admin@securesight.demo"
)

ADMIN_PASSWORD = os.environ.get(
    "ADMIN_PASSWORD",
    "SecureSight@123"
)

# ------------------------------------------------------------
# SECURITY CONFIGURATION
# ------------------------------------------------------------

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# HTTPS cookie security on Render.
app.config["SESSION_COOKIE_SECURE"] = bool(
    os.environ.get("RENDER")
)


# ============================================================
# 2. LOAD TRAINED MACHINE-LEARNING MODELS
# ============================================================

if not ISOLATION_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Isolation Forest model not found: {ISOLATION_MODEL_PATH}"
    )

if not RANDOM_FOREST_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Random Forest model not found: {RANDOM_FOREST_MODEL_PATH}"
    )

isolation_model = joblib.load(ISOLATION_MODEL_PATH)
rf_model = joblib.load(RANDOM_FOREST_MODEL_PATH)


# ============================================================
# 3. ACTIVITY ANALYSIS AND RISK SCORING
# ============================================================

def analyze_activity(df):
    """Analyze activity records and assign risk scores."""

    df = df.copy()

    # Check required columns.
    missing = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    # Convert required features to numeric values.
    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

    # Validate finite values.
    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Feature values must be finite numbers."
        )

    X = df[FEATURES]

    # --------------------------------------------------------
    # ISOLATION FOREST ANOMALY DETECTION
    # --------------------------------------------------------

    isolation_predictions = isolation_model.predict(X)

    df["anomaly"] = (
        isolation_predictions == -1
    ).astype(int)

    # --------------------------------------------------------
    # RANDOM FOREST SUSPICIOUS ACTIVITY PROBABILITY
    # --------------------------------------------------------

    probabilities = rf_model.predict_proba(X)

    classes = list(rf_model.classes_)

    if 1 in classes:

        suspicious_index = classes.index(1)

        df["suspicious_probability"] = probabilities[
            :, suspicious_index
        ]

    else:

        df["suspicious_probability"] = 0.0

    # --------------------------------------------------------
    # RISK SCORING
    # --------------------------------------------------------

    anomaly_score = df["anomaly"] * 20

    ml_probability_score = (
        df["suspicious_probability"] * 30
    )

    login_score = (
        df["login_failures"].clip(0, 10) / 10
    ) * 15

    file_access_score = (
        df["files_accessed"].clip(0, 100) / 100
    ) * 10

    data_transfer_score = (
        df["data_transfer_mb"].clip(0, 2000) / 2000
    ) * 10

    off_hours_score = df["off_hours"] * 8

    usb_score = df["usb_transfer"] * 7

    df["risk_score"] = (
        anomaly_score
        + ml_probability_score
        + login_score
        + file_access_score
        + data_transfer_score
        + off_hours_score
        + usb_score
    ).clip(0, 100).round().astype(int)

    # --------------------------------------------------------
    # RISK LEVEL CLASSIFICATION
    # --------------------------------------------------------

    df["risk_level"] = np.select(
        [
            df["risk_score"] >= 70,
            df["risk_score"] >= 35,
        ],
        [
            "High",
            "Medium",
        ],
        default="Low",
    )

    # --------------------------------------------------------
    # HUMAN-READABLE RISK INDICATORS
    # --------------------------------------------------------

    def get_indicators(row):

        indicators = []

        if row["anomaly"] == 1:
            indicators.append(
                "Unusual activity pattern"
            )

        if row["login_failures"] >= 5:
            indicators.append(
                "Multiple failed logins"
            )

        if row["files_accessed"] >= 40:
            indicators.append(
                "High file access"
            )

        if row["data_transfer_mb"] >= 500:
            indicators.append(
                "Large data transfer"
            )

        if row["usb_transfer"] == 1:
            indicators.append(
                "USB transfer recorded"
            )

        if row["off_hours"] == 1:
            indicators.append(
                "Activity outside normal hours"
            )

        return (
            ", ".join(indicators)
            if indicators
            else "No major indicators"
        )

    df["indicators"] = df.apply(
        get_indicators,
        axis=1
    )

    # --------------------------------------------------------
    # DEFAULT EVENT AND USER IDS
    # --------------------------------------------------------

    if "event_id" not in df.columns:

        df.insert(
            0,
            "event_id",
            range(1, len(df) + 1)
        )

    if "user_id" not in df.columns:

        df["user_id"] = "demo_user"

    return df


# ============================================================
# 4. LOGIN PAGE
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if session.get("logged_in"):
        return redirect(url_for("dashboard"))

    error = None

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        # Compare credentials securely.
        email_valid = hmac.compare_digest(
            email.encode("utf-8"),
            ADMIN_EMAIL.encode("utf-8")
        )

        password_valid = hmac.compare_digest(
            password.encode("utf-8"),
            ADMIN_PASSWORD.encode("utf-8")
        )

        if email_valid and password_valid:

            session.clear()
            session["logged_in"] = True

            return redirect(
                url_for("dashboard")
            )

        error = (
            "Invalid email or password. Please try again."
        )

    return render_template(
        "login.html",
        error=error
    )


# ============================================================
# 5. ROOT PAGE - SECURESIGHT LANDING PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("landing.html")


# ============================================================
# 6. MAIN SECURITY DASHBOARD
# ============================================================

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    # Protect dashboard from unauthenticated visitors.
    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    message = ""
    error = ""

    # --------------------------------------------------------
    # CSV UPLOAD AND VALIDATION
    # --------------------------------------------------------

    if request.method == "POST":

        uploaded = request.files.get("file")

        if not uploaded or not uploaded.filename:

            error = "Please select a CSV file."

        elif not uploaded.filename.lower().endswith(".csv"):

            error = "Only CSV files are supported."

        else:

            try:

                incoming = pd.read_csv(uploaded)

                missing = [
                    col for col in FEATURES
                    if col not in incoming.columns
                ]

                if missing:

                    error = (
                        "Missing required columns: "
                        + ", ".join(missing)
                    )

                elif incoming.empty:

                    error = (
                        "The uploaded CSV has no records."
                    )

                elif len(incoming) > 10000:

                    error = (
                        "Please upload at most 10,000 records at a time."
                    )

                else:

                    # Convert features to numeric values.
                    for col in FEATURES:

                        incoming[col] = pd.to_numeric(
                            incoming[col],
                            errors="raise"
                        )

                    # Check missing values.
                    if not incoming[FEATURES].notna().all().all():

                        raise ValueError(
                            "Required columns contain empty values."
                        )

                    # Check finite values.
                    if not np.isfinite(
                        incoming[FEATURES].to_numpy(dtype=float)
                    ).all():

                        raise ValueError(
                            "Values must be finite numbers."
                        )

                    # Validate hour.
                    if not incoming["hour"].between(0, 23).all():

                        raise ValueError(
                            "hour must be between 0 and 23."
                        )

                    # Validate non-negative values.
                    for col in [
                        "login_failures",
                        "files_accessed",
                        "usb_transfer",
                        "off_hours",
                    ]:

                        if (incoming[col] < 0).any():

                            raise ValueError(
                                f"{col} cannot be negative."
                            )

                    # Validate binary features.
                    for col in [
                        "usb_transfer",
                        "off_hours",
                    ]:

                        if not incoming[col].isin([0, 1]).all():

                            raise ValueError(
                                f"{col} must contain only 0 or 1."
                            )

                    # Validate data transfer.
                    if (
                        incoming["data_transfer_mb"] < 0
                    ).any():

                        raise ValueError(
                            "data_transfer_mb cannot be negative."
                        )

                    # Add identifiers when absent.
                    if "event_id" not in incoming.columns:

                        incoming.insert(
                            0,
                            "event_id",
                            range(1, len(incoming) + 1)
                        )

                    if "user_id" not in incoming.columns:

                        incoming["user_id"] = "uploaded_user"

                    # Store uploaded dataset.
                    incoming.to_csv(
                        UPLOAD_PATH,
                        index=False
                    )

                    message = (
                        f"Uploaded {len(incoming)} records successfully. "
                        "The dashboard now shows your uploaded dataset."
                    )

            except Exception as exc:

                error = f"Could not process CSV: {exc}"

    # --------------------------------------------------------
    # LOAD CURRENT DATASET
    # --------------------------------------------------------

    source = (
        UPLOAD_PATH
        if UPLOAD_PATH.exists()
        else DATA_PATH
    )

    if not source.exists():

        raise FileNotFoundError(
            f"Activity dataset not found: {source}"
        )

    df = pd.read_csv(source)

    df = analyze_activity(df)

    # --------------------------------------------------------
    # SEARCH AND RISK FILTERS
    # --------------------------------------------------------

    search = request.args.get(
        "search",
        ""
    ).strip()

    selected_risk = request.args.get(
        "risk",
        "All"
    )

    filtered = df.copy()

    # Search by user ID or event ID.
    if search:

        matches_user = filtered[
            "user_id"
        ].astype(str).str.contains(
            search,
            case=False,
            na=False,
            regex=False
        )

        matches_event = filtered[
            "event_id"
        ].astype(str).str.contains(
            search,
            case=False,
            na=False,
            regex=False
        )

        filtered = filtered[
            matches_user | matches_event
        ]

    # Filter by risk level.
    if selected_risk in [
        "Low",
        "Medium",
        "High",
    ]:

        filtered = filtered[
            filtered["risk_level"] == selected_risk
        ]

    # Sort and display top 100 records.
    records = (
        filtered.sort_values(
            "risk_score",
            ascending=False
        )
        .head(100)
        .to_dict(orient="records")
    )

    # --------------------------------------------------------
    # DASHBOARD METRICS AND CHART
    # --------------------------------------------------------

    chart_data = {

        "labels": [
            "Low",
            "Medium",
            "High"
        ],

        "values": [

            int(
                (df["risk_level"] == "Low").sum()
            ),

            int(
                (df["risk_level"] == "Medium").sum()
            ),

            int(
                (df["risk_level"] == "High").sum()
            ),
        ],
    }

    return render_template(

        "index.html",

        total_records=len(df),

        anomalies=int(
            df["anomaly"].sum()
        ),

        high_risk=int(
            (df["risk_level"] == "High").sum()
        ),

        medium_risk=int(
            (df["risk_level"] == "Medium").sum()
        ),

        records=records,

        search=search,

        selected_risk=selected_risk,

        chart_data=chart_data,

        message=message,

        error=error,

        data_source=source.name,
    )


# ============================================================
# 7. LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# 8. HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "message": (
            "SecureSight Insider Threat Detection is running"
        ),
    }


# ============================================================
# 9. RESET TO DEMO DATA
# ============================================================

@app.route("/reset")
def reset_data():

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    if UPLOAD_PATH.exists():

        UPLOAD_PATH.unlink()

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# 10. START FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
