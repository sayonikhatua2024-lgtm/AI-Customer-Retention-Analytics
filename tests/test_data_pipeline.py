import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_pipeline import clean_data, prepare_model_data


@pytest.fixture
def raw_like_df() -> pd.DataFrame:
    """A minimal dataframe with the columns clean_data/prepare_model_data expect."""
    base = {col: ["No"] for col in config.CATEGORICAL_FEATURES}
    base.update(
        {
            "CustomerID": ["0001-AAA"],
            "Count": [1],
            "Country": ["United States"],
            "Lat Long": ["0, 0"],
            "Latitude": [0.0],
            "Longitude": [0.0],
            "Zip Code": [90001],
            "Churn Value": [0],
            "Churn Score": [50],
            "Churn Reason": [np.nan],
            "State": ["California"],
            "City": ["Los Angeles"],
            "CLTV": [4000],
            "Tenure Months": [12],
            "Monthly Charges": [70.0],
            "Total Charges": [" "],  # blank string, as in the real dataset
            "Churn Label": ["No"],
        }
    )
    return pd.DataFrame(base)


def test_clean_data_drops_leakage_columns(raw_like_df):
    cleaned = clean_data(raw_like_df)
    for col in config.LEAKAGE_AND_ID_COLUMNS:
        assert col not in cleaned.columns


def test_clean_data_coerces_total_charges(raw_like_df):
    cleaned = clean_data(raw_like_df)
    assert cleaned["Total Charges"].dtype.kind in "if"
    assert cleaned.loc[0, "Total Charges"] == 0  # blank string -> NaN -> 0


def test_prepare_model_data_matches_model_schema(raw_like_df):
    cleaned = clean_data(raw_like_df)
    X, y = prepare_model_data(cleaned)
    assert list(X.columns) == config.MODEL_FEATURES
    assert y.iloc[0] == 0  # "No" -> 0
