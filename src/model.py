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

import json
import sys
from importlib import metadata

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.utils import get_logger, require_file

logger = get_logger(__name__)


def load_model(path=config.MODEL_PATH, metadata_path=config.MODEL_METADATA_PATH) -> Pipeline:
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
    validate_model_metadata(model, metadata_path=metadata_path)
    logger.info("Loaded model: %s", type(model.named_steps.get("classifier", model)).__name__)
    return model


def load_model_metadata(path=config.MODEL_METADATA_PATH) -> dict:
    """Load the model governance metadata JSON artifact."""
    require_file(
        path,
        friendly_name="model metadata",
        how_to_fix=(
            "Place 'model_metadata.json' in the 'models/' folder. "
            "This file documents the trained model artifact and schema."
        ),
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_model_metadata(model: Pipeline, metadata_path=config.MODEL_METADATA_PATH) -> None:
    """
    Validate that the model artifact agrees with its governance metadata.

    This protects the app from silently serving a model whose feature
    schema, estimator configuration, or transformed feature count has
    drifted away from the checked-in metadata.
    """
    metadata_json = load_model_metadata(metadata_path)
    schema = metadata_json.get("feature_schema", {})
    training_config = metadata_json.get("training_configuration", {})

    expected_numeric = schema.get("numeric_features")
    expected_categorical = schema.get("categorical_features")
    expected_features = (expected_numeric or []) + (expected_categorical or [])
    if expected_features != config.MODEL_FEATURES:
        raise ValueError("Model metadata feature schema does not match src.config.MODEL_FEATURES.")

    transformed_count = schema.get("transformed_feature_count")
    actual_transformed_count = len(model.named_steps["preprocessor"].get_feature_names_out())
    if transformed_count != actual_transformed_count:
        raise ValueError(
            "Model metadata transformed feature count does not match the loaded model "
            f"({transformed_count} != {actual_transformed_count})."
        )

    transformer_columns = {
        name: list(columns)
        for name, _, columns in model.named_steps["preprocessor"].transformers_
        if name in {"num", "cat"}
    }
    if transformer_columns.get("num") != config.NUMERIC_FEATURES:
        raise ValueError("Loaded model numeric feature schema does not match src.config.NUMERIC_FEATURES.")
    if transformer_columns.get("cat") != config.CATEGORICAL_FEATURES:
        raise ValueError("Loaded model categorical feature schema does not match src.config.CATEGORICAL_FEATURES.")

    classifier = model.named_steps["classifier"]
    expected_params = training_config.get("best_params", {})
    for param_name, expected_value in expected_params.items():
        classifier_param = param_name.removeprefix("classifier__")
        actual_value = classifier.get_params().get(classifier_param)
        if actual_value != expected_value:
            raise ValueError(
                f"Model metadata parameter mismatch for {param_name}: "
                f"expected {expected_value!r}, got {actual_value!r}."
            )

    runtime = metadata_json.get("runtime", {})
    sklearn_version = metadata.version("scikit-learn")
    if runtime.get("scikit_learn_version") != sklearn_version:
        logger.warning(
            "Model metadata was recorded with scikit-learn %s, but runtime is %s.",
            runtime.get("scikit_learn_version"),
            sklearn_version,
        )
    if runtime.get("python_version") != ".".join(map(str, sys.version_info[:3])):
        logger.warning(
            "Model metadata was recorded with Python %s, but runtime is %s.",
            runtime.get("python_version"),
            ".".join(map(str, sys.version_info[:3])),
        )


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
