"""
feature_engineering.py
=======================
Business-facing engineered features for dashboard analytics and
segmentation (e.g. "show me long-term customers with high bills").

IMPORTANT — these features are NOT sent to the model:
The notebook explores a second round of feature engineering
(``Avg Monthly Spend``, ``Long Term Customer``, ``High Monthly Bill``,
``Total Services``, ``Has Internet``, ``Support Count``,
``Family Size``) after the model had already been trained and saved.
Inspecting ``customer_churn_model.pkl`` directly confirms its
ColumnTransformer only accepts the original 19 raw features
(``config.MODEL_FEATURES``) — these engineered columns were never
part of its training schema.

Rather than silently discarding that analysis or quietly feeding the
model columns it wasn't trained on (which would raise at prediction
time, or worse, be dropped by ``handle_unknown="ignore"`` without
warning), this module keeps them as a clearly separate, opt-in
enrichment step for dashboard segmentation and business insights only.
If you want the model itself to use these signals, retrain via
``src/train.py`` with an expanded feature list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

SERVICE_COLUMNS = [
    "Phone Service",
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]

LONG_TERM_TENURE_MONTHS = 24


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add analytics-only engineered features for dashboard segmentation.
    Safe to call on any dataframe containing the standard Telco churn
    columns; missing source columns are skipped gracefully.

    Added columns
    -------------
    Avg Monthly Spend : float
        Total Charges / Tenure Months (0 where tenure is 0, to avoid
        division-by-zero).
    Long Term Customer : int (0/1)
        1 if Tenure Months >= 24.
    High Monthly Bill : int (0/1)
        1 if Monthly Charges is above the dataset median.
    Total Services : int
        Count of subscribed services among SERVICE_COLUMNS.
    Has Internet : int (0/1)
        1 if Internet Service != "No".
    Family Size : int
        Partner (0/1) + Dependents (0/1).
    """
    out = df.copy()

    if {"Total Charges", "Tenure Months"}.issubset(out.columns):
        safe_tenure = out["Tenure Months"].replace(0, 1)
        out["Avg Monthly Spend"] = (out["Total Charges"] / safe_tenure).replace(
            [np.inf, -np.inf], 0
        )
        out["Long Term Customer"] = (out["Tenure Months"] >= LONG_TERM_TENURE_MONTHS).astype(int)

    if "Monthly Charges" in out.columns:
        median_bill = out["Monthly Charges"].median()
        out["High Monthly Bill"] = (out["Monthly Charges"] > median_bill).astype(int)

    present_service_cols = [c for c in SERVICE_COLUMNS if c in out.columns]
    if present_service_cols:
        out["Total Services"] = (out[present_service_cols] == "Yes").sum(axis=1)

    if "Internet Service" in out.columns:
        out["Has Internet"] = (out["Internet Service"] != "No").astype(int)

    if {"Partner", "Dependents"}.issubset(out.columns):
        out["Family Size"] = (out["Partner"] == "Yes").astype(int) + (
            out["Dependents"] == "Yes"
        ).astype(int)

    added = set(out.columns) - set(df.columns)
    logger.info("Added %s business features: %s", len(added), sorted(added))
    return out
