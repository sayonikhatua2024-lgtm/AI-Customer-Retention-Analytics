"""
model.py
========
Loading and inference wrappers around the trained sklearn Pipeline
(``preprocessor`` + ``classifier``) saved at ``models/customer_churn_model.pkl``.

Keeping this as a thin, dedicated layer means the dashboard and the
training script both call the same code path for prediction, so
predictions can never silently drift between the two.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.utils import get_logger, require_file

logger = get_logger(__name__)


def load_model(path=config.MODEL_PATH) -> Pipeline:
    """
    Load the trained churn-prediction pipeline from disk.

    Raises
    ------
    MissingAssetError
        If the model file is not present.
    """
    require_file(
        path,
        friendly_name="trained churn model",
        how_to_fix=(
            "Place 'customer_churn_model.pkl' in the 'models/' folder, "
            "or run 'python -m src.train' to train and save a new one."
        ),
    )
    logger.info("Loading model from %s", path)
    model = joblib.load(path)
    logger.info("Loaded model: %s", type(model.named_steps.get("classifier", model)).__name__)
    return model


def predict(model: Pipeline, X: pd.DataFrame):
    """Return hard class predictions (0 = retained, 1 = churn)."""
    return model.predict(X[config.MODEL_FEATURES])


def predict_proba(model: Pipeline, X: pd.DataFrame):
    """Return churn probability (positive class) for each row of X."""
    return model.predict_proba(X[config.MODEL_FEATURES])[:, 1]


def get_feature_names(model: Pipeline) -> list[str]:
    """Return the post-preprocessing feature names (after one-hot encoding)."""
    preprocessor = model.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def get_feature_importances(model: Pipeline) -> pd.DataFrame:
    """
    Return a DataFrame of (Feature, Importance), sorted descending,
    using the classifier's native feature_importances_ where available
    (Random Forest / tree-based models).
    """
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        raise AttributeError(
            f"Classifier {type(classifier).__name__} does not expose "
            "feature_importances_. Use explainability.py (SHAP) instead."
        )

    feature_names = get_feature_names(model)
    importances = classifier.feature_importances_

    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    return df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
