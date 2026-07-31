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

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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

    evaluate(model, X_test, y_test)

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)
        logger.info("Saved trained model to %s", config.MODEL_PATH)

        generate_dashboard_dataset(model, save=True)

    return model


def evaluate(model: Pipeline, X_test, y_test) -> None:
    """Print accuracy, confusion matrix, classification report, and ROC-AUC."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    logger.info("Accuracy: %.4f", accuracy_score(y_test, y_pred))
    logger.info("ROC-AUC: %.4f", roc_auc_score(y_test, y_prob))
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))


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
