import pandas as pd
import pytest

from src.risk_engine import apply_risk_levels, classify_risk


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "Low"),
        (0.29, "Low"),
        (0.30, "Medium"),
        (0.59, "Medium"),
        (0.60, "High"),
        (0.79, "High"),
        (0.80, "Critical"),
        (1.0, "Critical"),
    ],
)
def test_classify_risk_boundaries(probability, expected):
    assert classify_risk(probability) == expected


def test_apply_risk_levels_adds_column():
    df = pd.DataFrame({"Churn Probability": [0.1, 0.45, 0.65, 0.9]})
    out = apply_risk_levels(df)
    assert list(out["Risk Level"]) == ["Low", "Medium", "High", "Critical"]


def test_apply_risk_levels_missing_column_raises():
    df = pd.DataFrame({"Not Probability": [0.1]})
    with pytest.raises(ValueError):
        apply_risk_levels(df)
