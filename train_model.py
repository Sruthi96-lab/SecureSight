
import os
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create model directory
os.makedirs("models", exist_ok=True)

# Load our synthetic activity dataset
df = pd.read_csv("data/user_activity.csv")

FEATURES = [
    "hour",
    "login_failures",
    "files_accessed",
    "data_transfer_mb",
    "usb_transfer",
    "off_hours",
]

X = df[FEATURES]

# ------------------------------------------------
# MODEL 1: Isolation Forest anomaly detection
# ------------------------------------------------

isolation_model = IsolationForest(
    n_estimators=150,
    contamination=0.05,
    random_state=42
)

isolation_model.fit(X)

# Isolation Forest: -1 = anomaly, 1 = normal
df["anomaly_prediction"] = isolation_model.predict(X)
df["anomaly"] = (df["anomaly_prediction"] == -1).astype(int)

# ------------------------------------------------
# Create demonstration labels for Random Forest
# ------------------------------------------------
# These are rule-generated synthetic labels, not
# verified real-world insider threat labels.

df["synthetic_label"] = (
    (df["login_failures"] >= 5)
    | (df["files_accessed"] >= 40)
    | (df["data_transfer_mb"] >= 500)
    | (
        (df["off_hours"] == 1)
        & (df["usb_transfer"] == 1)
    )
).astype(int)

# ------------------------------------------------
# MODEL 2: Random Forest classification
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    df["synthetic_label"],
    test_size=0.2,
    random_state=42,
    stratify=df["synthetic_label"]
)

rf_model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

predictions = rf_model.predict(X_test)

print("\n===== RANDOM FOREST DEMO EVALUATION =====")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# ------------------------------------------------
# Save trained models and labelled demo records
# ------------------------------------------------

joblib.dump(isolation_model, "models/isolation_forest.joblib")
joblib.dump(rf_model, "models/random_forest.joblib")

df.to_csv("data/analyzed_activity.csv", index=False)

print("===== TRAINING COMPLETED =====")
print("Isolation Forest model saved.")
print("Random Forest model saved.")
print("Analyzed records saved to data/analyzed_activity.csv")
print(f"Records analyzed: {len(df)}")
print(f"Anomalies detected: {df['anomaly'].sum()}")
print(f"Synthetic suspicious labels: {df['synthetic_label'].sum()}")
