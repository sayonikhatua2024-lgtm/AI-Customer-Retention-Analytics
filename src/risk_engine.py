"""
risk_engine.py
==============
Converts a raw churn probability into a business-friendly risk
category. Thresholds are ported verbatim from the notebook's
``risk_level`` function (Low < 0.30, Medium < 0.60, High < 0.80,
Critical >= 0.80) and centralized in ``src.config`` so the dashboard
and any batch scoring job stay in sync.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def classify_risk(probability: float) -> str:
    """
    Map a single churn probability to a risk level.

    Parameters
    ----------
    probability : float
        Predicted probability of churn, in [0, 1].
    """
    if probability < config.RISK_THRESHOLDS["Low"]:
        return "Low"
    elif probability < config.RISK_THRESHOLDS["Medium"]:
        return "Medium"
    elif probability < config.RISK_THRESHOLDS["High"]:
        return "High"
    else:
        return "Critical"


def apply_risk_levels(df: pd.DataFrame, prob_col: str = "Churn Probability") -> pd.DataFrame:
    """
    Add a 'Risk Level' column to df based on `prob_col`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `prob_col`.
    prob_col : str
        Name of the column holding churn probabilities.
    """
    if prob_col not in df.columns:
        raise ValueError(f"'{prob_col}' column not found in dataframe.")

    out = df.copy()
    out["Risk Level"] = out[prob_col].apply(classify_risk)

    counts = out["Risk Level"].value_counts().to_dict()
    logger.info("Risk level distribution: %s", counts)
    return out
