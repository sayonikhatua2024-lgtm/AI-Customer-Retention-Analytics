"""Shared pytest fixtures and path setup."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import pytest


@pytest.fixture
def sample_customers() -> pd.DataFrame:
    """A small, hand-built dataframe covering each risk tier's boundary."""
    return pd.DataFrame(
        {
            "Risk Level": ["Low", "Medium", "High", "High", "Critical", "Critical"],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "Two year",
            ],
            "Monthly Charges": [45.0, 60.0, 95.0, 60.0, 100.0, 30.0],
        }
    )
