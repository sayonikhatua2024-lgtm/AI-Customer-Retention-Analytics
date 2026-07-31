import pandas as pd

from src.recommendation_engine import apply_recommendations, recommend_action


def test_critical_month_to_month():
    row = pd.Series({"Risk Level": "Critical", "Contract": "Month-to-month", "Monthly Charges": 50})
    assert recommend_action(row) == "Immediate retention call + 20% discount + Upgrade to yearly contract"


def test_critical_long_term_contract():
    row = pd.Series({"Risk Level": "Critical", "Contract": "Two year", "Monthly Charges": 50})
    assert recommend_action(row) == "Priority retention call + Loyalty reward"


def test_high_risk_high_bill():
    row = pd.Series({"Risk Level": "High", "Contract": "One year", "Monthly Charges": 90})
    assert recommend_action(row) == "Offer personalized discount"


def test_high_risk_low_bill():
    row = pd.Series({"Risk Level": "High", "Contract": "One year", "Monthly Charges": 50})
    assert recommend_action(row) == "Offer premium support"


def test_medium_risk():
    row = pd.Series({"Risk Level": "Medium", "Contract": "One year", "Monthly Charges": 50})
    assert recommend_action(row) == "Send promotional email"


def test_low_risk():
    row = pd.Series({"Risk Level": "Low", "Contract": "Two year", "Monthly Charges": 50})
    assert recommend_action(row) == "No action required"


def test_apply_recommendations_on_dataframe(sample_customers):
    out = apply_recommendations(sample_customers)
    assert "Recommendation" in out.columns
    assert len(out) == len(sample_customers)
    assert out["Recommendation"].notna().all()
