"""
config.py
=========
Single source of truth for file paths, feature schema, and business
constants used across the AI Customer Retention Analytics Platform.

Every path here is relative to the project root (computed dynamically),
so the project runs identically on any machine or on Streamlit
Community Cloud without edits.

IMPORTANT — feature schema:
The NUMERIC_FEATURES / CATEGORICAL_FEATURES lists below are not
arbitrary. They were reverse-engineered directly from the trained
pipeline shipped in ``models/customer_churn_model.pkl`` (its
ColumnTransformer's ``transformers_`` attribute) so that any code
calling ``model.predict()`` sends exactly the columns the model was
fitted on, in a schema it recognizes. If you retrain the model with a
different feature set, update this file to match.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "Telco_customer_churn.xlsx"
DASHBOARD_DATA_PATH = PROCESSED_DATA_DIR / "dashboard_data.csv"
MODEL_PATH = MODELS_DIR / "customer_churn_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Raw dataset columns that carry no predictive signal or that leak the
# target (identifiers, geography, and post-outcome fields such as the
# churn score/reason IBM computed themselves).
# ---------------------------------------------------------------------------
LEAKAGE_AND_ID_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Zip Code",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
]

# Dropped separately during model-data prep (high-cardinality / redundant
# with fields already present, mirrors the notebook's EDA decisions).
DROP_FOR_MODELING = ["State", "City", "CLTV"]

TARGET_COLUMN = "Churn Label"
TARGET_MAP = {"No": 0, "Yes": 1}

# ---------------------------------------------------------------------------
# Exact feature schema expected by the trained model pipeline.
# Sourced from customer_churn_model.pkl -> preprocessor.transformers_
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Business rules — ported verbatim from the notebook (cells: risk_level,
# recommend_action) so dashboard and training code never diverge.
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "Low": 0.30,
    "Medium": 0.60,
    "High": 0.80,
    # >= 0.80 -> Critical
}

RISK_LEVEL_ORDER = ["Low", "Medium", "High", "Critical"]

RISK_COLORS = {
    "Low": "#2ECC71",
    "Medium": "#F5B041",
    "High": "#E67E22",
    "Critical": "#E74C3C",
}

# ---------------------------------------------------------------------------
# Model training configuration — matches the GridSearchCV winner that was
# actually saved to customer_churn_model.pkl (verified against the
# pipeline's fitted classifier params: n_estimators=200, max_depth=10,
# min_samples_split=5, random_state=42).
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

RF_PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [5, 10, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
}
GRID_SEARCH_CV_FOLDS = 5
GRID_SEARCH_SCORING = "f1"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL = "INFO"
