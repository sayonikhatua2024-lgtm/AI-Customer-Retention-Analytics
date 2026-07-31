"""
train.py
========
End-to-end training script: raw data -> clean -> split -> preprocess
-> GridSearchCV over a Random Forest -> evaluate -> save model ->
regenerate the dashboard dataset.

This reproduces the model selection process from
``notebooks/01_Data_Understanding.ipynb`` (Logistic Regression,
Decision Tree, Random Forest, and XGBoost were compared; Random
Forest tuned via GridSearchCV was the version actually saved to
``customer_churn_model.pkl``). Run this directly to retrain from
scratch:

    python -m src.train

The resulting model will overwrite ``models/customer_churn_model.pkl``
and ``data/processed/dashboard_data.csv`` will be regenerated to match.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from importlib import metadata

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data_pipeline import clean_data, generate_dashboard_dataset, load_raw_data, prepare_model_data
from src.utils import get_logger

logger = get_logger(__name__)


def build_pipeline() -> Pipeline:
    """Build the untrained preprocessing + Random Forest pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
            ),
        ]
    )
    return pipeline


def train(tune: bool = True, save: bool = True) -> Pipeline:
    """
    Run the full training pipeline.

    Parameters
    ----------
    tune : bool
        If True (default), run GridSearchCV over ``config.RF_PARAM_GRID``.
        If False, fit a single Random Forest with the grid-search-winning
        params baked in, for a much faster iteration loop.
    save : bool
        If True, persist the model and regenerate the dashboard dataset.
    """
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    X, y = prepare_model_data(clean_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    logger.info("Train/test split: %s / %s", X_train.shape, X_test.shape)

    if tune:
        pipeline = build_pipeline()
        logger.info("Running GridSearchCV over %s ...", config.RF_PARAM_GRID)
        grid = GridSearchCV(
            pipeline,
            config.RF_PARAM_GRID,
            cv=config.GRID_SEARCH_CV_FOLDS,
            scoring=config.GRID_SEARCH_SCORING,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        logger.info("Best params: %s", grid.best_params_)
        logger.info("Best CV F1: %.4f", grid.best_score_)
        model = grid.best_estimator_
    else:
        # Known-good params from the original GridSearchCV run.
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), config.NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
            ]
        )
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_split=5,
                        random_state=config.RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)
        logger.info("Saved trained model to %s", config.MODEL_PATH)
        save_model_metadata(model, metrics, raw_df.shape[0], raw_df.shape[1], len(X_test))

        generate_dashboard_dataset(model, save=True)

    return model


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    """Log evaluation metrics and return a metadata-ready metrics dictionary."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        "test_size": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 3),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 3),
        "precision_churn": round(float(report["1"]["precision"]), 3),
        "recall_churn": round(float(report["1"]["recall"]), 3),
        "f1_churn": round(float(report["1"]["f1-score"]), 3),
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
    }

    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("ROC-AUC: %.4f", metrics["roc_auc"])
    logger.info("PR-AUC: %.4f", metrics["pr_auc"])
    logger.info("Confusion matrix:\n%s", matrix)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))
    return metrics


def save_model_metadata(model: Pipeline, metrics: dict, raw_rows: int, raw_columns: int, test_rows: int) -> None:
    """Persist governance metadata next to the trained model artifact."""
    classifier = model.named_steps["classifier"]
    metadata_payload = {
        "model_version": "1.0.0",
        "training_timestamp": datetime.now(UTC).isoformat(),
        "artifact_path": str(config.MODEL_PATH.relative_to(config.BASE_DIR)),
        "model_type": "sklearn.pipeline.Pipeline",
        "estimator": type(classifier).__name__,
        "metrics": {**metrics, "test_size": test_rows},
        "feature_schema": {
            "numeric_features": config.NUMERIC_FEATURES,
            "categorical_features": config.CATEGORICAL_FEATURES,
            "target_column": config.TARGET_COLUMN,
            "target_map": config.TARGET_MAP,
            "transformed_feature_count": len(model.named_steps["preprocessor"].get_feature_names_out()),
        },
        "runtime": {
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "scikit_learn_version": metadata.version("scikit-learn"),
        },
        "training_configuration": {
            "random_state": config.RANDOM_STATE,
            "test_size": config.TEST_SIZE,
            "grid_search_cv_folds": config.GRID_SEARCH_CV_FOLDS,
            "grid_search_scoring": config.GRID_SEARCH_SCORING,
            "best_params": {
                "classifier__n_estimators": classifier.get_params()["n_estimators"],
                "classifier__max_depth": classifier.get_params()["max_depth"],
                "classifier__min_samples_split": classifier.get_params()["min_samples_split"],
            },
            "param_grid": config.RF_PARAM_GRID,
        },
        "data": {
            "source": "IBM Telco Customer Churn dataset",
            "raw_path": str(config.RAW_DATA_PATH.relative_to(config.BASE_DIR)),
            "raw_rows": raw_rows,
            "raw_columns": raw_columns,
            "processed_dashboard_rows": raw_rows,
        },
    }
    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
        f.write("\n")
    logger.info("Saved model metadata to %s", config.MODEL_METADATA_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the customer churn model.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip GridSearchCV and fit a single RF with known-good params (faster).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Train and evaluate without overwriting the saved model/dashboard data.",
    )
    args = parser.parse_args()

    train(tune=not args.fast, save=not args.no_save)


if __name__ == "__main__":
    main()
