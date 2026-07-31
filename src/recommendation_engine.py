"""
recommendation_engine.py
=========================
Rule-based retention actions keyed off risk level, contract type, and
monthly charges. Ported verbatim from the notebook's
``recommend_action`` function so business logic never diverges
between the notebook, the dashboard, and any batch job.

This is intentionally a simple, auditable rule table rather than a
second model — for a retention program, "why did the model say to
call this customer" needs to be answerable in one sentence to a
non-technical stakeholder.
"""

from __future__ import annotations

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

MONTHLY_CHARGE_HIGH_THRESHOLD = 80


def recommend_action(row: pd.Series) -> str:
    """
    Return a retention action for a single customer row.

    Expects the row to contain 'Risk Level', 'Contract', and
    'Monthly Charges'.
    """
    risk = row["Risk Level"]
    contract = row["Contract"]
    monthly = row["Monthly Charges"]

    if risk == "Critical":
        if contract == "Month-to-month":
            return "Immediate retention call + 20% discount + Upgrade to yearly contract"
        else:
            return "Priority retention call + Loyalty reward"

    elif risk == "High":
        if monthly > MONTHLY_CHARGE_HIGH_THRESHOLD:
            return "Offer personalized discount"
        else:
            return "Offer premium support"

    elif risk == "Medium":
        return "Send promotional email"

    else:
        return "No action required"


def apply_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Recommendation' column to df by applying recommend_action row-wise."""
    required = {"Risk Level", "Contract", "Monthly Charges"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {missing}")

    out = df.copy()
    out["Recommendation"] = out.apply(recommend_action, axis=1)

    logger.info("Generated recommendations: %s", out["Recommendation"].value_counts().to_dict())
    return out
