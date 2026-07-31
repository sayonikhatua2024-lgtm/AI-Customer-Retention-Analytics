"""
explainability.py
==================
SHAP-based explainability for the trained Random Forest churn model.
Wraps the notebook's TreeExplainer usage (global summary, feature
importance bar chart, and per-customer waterfall) into reusable
functions the dashboard can call without duplicating SHAP boilerplate.

Note: SHAP is computed on the *post-preprocessing* feature space
(after scaling/one-hot encoding), since that's what the tree model
actually sees — feature names are pulled from the fitted
ColumnTransformer so results stay human-readable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


def _ensure_shap_available() -> None:
    if shap is None:
        raise ImportError(
            "The 'shap' package is required for explainability features. "
            "Install it with: pip install shap"
        )


def build_explainer(model: Pipeline):
    """Build a SHAP TreeExplainer around the model's classifier step."""
    _ensure_shap_available()
    classifier = model.named_steps["classifier"]
    return shap.TreeExplainer(classifier)


def transform_features(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Run X through the model's fitted preprocessor only (no prediction)."""
    preprocessor = model.named_steps["preprocessor"]
    return preprocessor.transform(X[config.MODEL_FEATURES])


def compute_shap_values(model: Pipeline, X: pd.DataFrame):
    """
    Compute SHAP values for the positive (churn) class over X.

    Returns
    -------
    shap_values : np.ndarray
        Array of shape (n_samples, n_features) for the churn class.
    X_transformed : np.ndarray
        The preprocessed feature matrix SHAP values correspond to.
    feature_names : list[str]
        Human-readable post-preprocessing feature names.
    """
    _ensure_shap_available()
    explainer = build_explainer(model)
    X_transformed = transform_features(model, X)
    feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())

    raw_shap_values = explainer.shap_values(X_transformed)

    # TreeExplainer on a binary-classification RandomForest can return
    # either a (n_samples, n_features, n_classes) array or a list of
    # per-class arrays depending on the SHAP version — normalize both
    # to a single (n_samples, n_features) array for the churn class (1).
    if isinstance(raw_shap_values, list):
        churn_shap_values = raw_shap_values[1]
    elif raw_shap_values.ndim == 3:
        churn_shap_values = raw_shap_values[:, :, 1]
    else:
        churn_shap_values = raw_shap_values

    logger.info("Computed SHAP values for %s samples, %s features", *churn_shap_values.shape)
    return churn_shap_values, X_transformed, feature_names


def get_top_shap_features(shap_values: np.ndarray, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    """
    Rank features by mean absolute SHAP value (global importance).

    Returns
    -------
    pd.DataFrame with columns ['Feature', 'Mean |SHAP value|']
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP value|": mean_abs_shap})
    return df.sort_values(by="Mean |SHAP value|", ascending=False).head(top_n).reset_index(drop=True)


def explain_customer(
    model: Pipeline,
    X_row: pd.DataFrame,
):
    """
    Build a SHAP Explanation object for a single customer row, suitable
    for ``shap.plots.waterfall``.

    Parameters
    ----------
    X_row : pd.DataFrame
        A single-row dataframe (e.g. ``X.iloc[[i]]``).
    """
    _ensure_shap_available()
    explainer = build_explainer(model)
    X_transformed = transform_features(model, X_row)
    feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())

    raw_shap_values = explainer.shap_values(X_transformed)
    if isinstance(raw_shap_values, list):
        values = raw_shap_values[1][0]
        base_value = explainer.expected_value[1]
    elif raw_shap_values.ndim == 3:
        values = raw_shap_values[0, :, 1]
        base_value = explainer.expected_value[1]
    else:
        values = raw_shap_values[0]
        base_value = explainer.expected_value

    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=X_transformed[0],
        feature_names=feature_names,
    )
