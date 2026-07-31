"""
data_pipeline.py
=================
Loading and cleaning logic for the Telco customer churn dataset.

This module reproduces — as functions, not notebook cells — the
cleaning steps from ``notebooks/01_Data_Understanding.ipynb``:
dropping identifier/leakage columns, coercing ``Total Charges`` to
numeric, and preparing the modeling-ready feature/target split.

It also owns ``generate_dashboard_dataset``, which orchestrates the
raw data, the trained model, and the risk/recommendation engines to
(re)produce ``data/processed/dashboard_data.csv`` when it's missing —
this is what lets the app run immediately after a fresh clone, even
before anyone has opened the notebook.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.utils import get_logger, require_file, MissingAssetError

logger = get_logger(__name__)


def load_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco churn Excel file.

    Raises
    ------
    MissingAssetError
        If the raw dataset is not present.
    """
    require_file(
        path,
        friendly_name="raw Telco churn dataset",
        how_to_fix=(
            "Place 'Telco_customer_churn.xlsx' in the 'data/raw/' folder."
        ),
    )
    logger.info("Loading raw dataset from %s", path)
    df = pd.read_excel(path)
    logger.info("Loaded raw dataset: %s rows, %s columns", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the notebook's cleaning steps:
    - Drop identifier / geography / leakage columns.
    - Coerce 'Total Charges' to numeric (it ships as a string with
      blanks for brand-new customers) and fill resulting NaNs with 0,
      matching the notebook's treatment.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as loaded from ``load_raw_data``.
    """
    df_clean = df.copy()

    cols_to_drop = [c for c in config.LEAKAGE_AND_ID_COLUMNS if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    if "Total Charges" in df_clean.columns:
        df_clean["Total Charges"] = pd.to_numeric(
            df_clean["Total Charges"], errors="coerce"
        )
        n_missing = df_clean["Total Charges"].isna().sum()
        if n_missing:
            logger.info(
                "Filling %s missing 'Total Charges' values with 0 "
                "(new customers with no billing history yet)",
                n_missing,
            )
        df_clean["Total Charges"] = df_clean["Total Charges"].fillna(0)

    logger.info("Cleaned dataset: %s rows, %s columns", *df_clean.shape)
    return df_clean


def prepare_model_data(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Produce the modeling-ready feature matrix X and target vector y,
    matching exactly the schema the trained model expects
    (``config.MODEL_FEATURES``).

    Returns
    -------
    (X, y) : tuple[pd.DataFrame, pd.Series]
    """
    df_model = df_clean.copy()

    extra_drop = [c for c in config.DROP_FOR_MODELING if c in df_model.columns]
    df_model = df_model.drop(columns=extra_drop)

    if config.TARGET_COLUMN in df_model.columns:
        df_model[config.TARGET_COLUMN] = df_model[config.TARGET_COLUMN].map(
            config.TARGET_MAP
        )
        y = df_model[config.TARGET_COLUMN]
        X = df_model.drop(columns=[config.TARGET_COLUMN])
    else:
        y = None
        X = df_model

    missing_cols = [c for c in config.MODEL_FEATURES if c not in X.columns]
    if missing_cols:
        raise ValueError(
            f"Prepared data is missing columns required by the model: {missing_cols}"
        )

    X = X[config.MODEL_FEATURES]
    logger.info("Prepared model data: X=%s, y=%s", X.shape, None if y is None else y.shape)
    return X, y


def generate_dashboard_dataset(model, save: bool = True) -> pd.DataFrame:
    """
    Rebuild the dashboard dataset from scratch: raw data -> clean ->
    score every customer with the trained model -> assign risk levels
    -> assign retention recommendations.

    Unlike the original notebook (which scored only the held-out test
    split, ~20% of customers), this scores the **entire customer
    base**, which is what a real retention dashboard needs. Existing
    columns and logic are otherwise unchanged from the notebook.

    Parameters
    ----------
    model :
        A fitted sklearn Pipeline (preprocessor + classifier) capable
        of ``predict_proba`` on ``config.MODEL_FEATURES``.
    save : bool
        If True, writes the result to ``config.DASHBOARD_DATA_PATH``.

    Returns
    -------
    pd.DataFrame
        The full scored dataset used by the dashboard.
    """
    # Local imports to avoid a circular import at module load time.
    from src.risk_engine import apply_risk_levels
    from src.recommendation_engine import apply_recommendations

    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    X, y = prepare_model_data(clean_df)

    logger.info("Scoring %s customers with the trained model", len(X))
    churn_probability = model.predict_proba(X)[:, 1]

    dashboard_df = X.copy()
    if y is not None:
        dashboard_df["Actual Churn"] = y.values
    dashboard_df["Churn Probability"] = churn_probability

    dashboard_df = apply_risk_levels(dashboard_df, prob_col="Churn Probability")
    dashboard_df = apply_recommendations(dashboard_df)

    if save:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        dashboard_df.to_csv(config.DASHBOARD_DATA_PATH, index=False)
        logger.info("Saved dashboard dataset to %s", config.DASHBOARD_DATA_PATH)

    return dashboard_df


def load_dashboard_data(model=None) -> pd.DataFrame:
    """
    Load the processed dashboard dataset, transparently regenerating
    it from the raw data + trained model if it's missing. This is the
    entry point the Streamlit app should call.

    Parameters
    ----------
    model :
        A fitted model, required only if the CSV needs to be rebuilt.
        Pass ``None`` if you're confident the CSV already exists.
    """
    if config.DASHBOARD_DATA_PATH.exists():
        logger.info("Loading existing dashboard dataset from %s", config.DASHBOARD_DATA_PATH)
        return pd.read_csv(config.DASHBOARD_DATA_PATH)

    logger.warning(
        "Dashboard dataset not found at %s — attempting to regenerate it.",
        config.DASHBOARD_DATA_PATH,
    )
    if model is None:
        raise MissingAssetError(
            f"'{config.DASHBOARD_DATA_PATH.name}' was not found and no model "
            "was supplied to regenerate it. Provide the trained model, or "
            "place a precomputed dashboard_data.csv in 'data/processed/'."
        )
    return generate_dashboard_dataset(model, save=True)
