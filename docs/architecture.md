# Architecture

## Overview

The project is an end-to-end customer retention analytics application:

```text
Raw Excel data
  -> data cleaning
  -> sklearn model training/scoring
  -> churn probability
  -> risk tier
  -> retention recommendation
  -> processed dashboard dataset
  -> Streamlit dashboard
```

## Key Modules

- `src.config` centralizes paths, model schema, risk thresholds, and training constants.
- `src.data_pipeline` loads raw data, cleans it, prepares model features, and generates the scored dashboard dataset.
- `src.model` loads the model artifact, validates governance metadata, runs inference, and exposes feature importances.
- `src.risk_engine` maps churn probabilities to business-facing risk tiers.
- `src.recommendation_engine` maps risk/customer attributes to auditable retention actions.
- `src.feature_engineering` adds dashboard-only segmentation features that are not sent to the model.
- `src.explainability` wraps SHAP feature transformation and customer-level explanation logic.
- `dashboard.app` renders the Streamlit dashboard.

## Artifact Governance

The model artifact is stored at `models/customer_churn_model.pkl`. Its companion governance file, `models/model_metadata.json`, records model version, training timestamp, metrics, schema, runtime versions, and training configuration.

`src.model.load_model()` validates this metadata before serving predictions.

## Dashboard Data Flow

The dashboard reads `data/processed/dashboard_data.csv` when available. If missing, it can regenerate the scored dataset from the raw data and trained model. Customer identifiers are preserved from `CustomerID` when available and used for display/export only. For older processed CSVs without identifiers, the data layer restores `CustomerID` only after verifying that the dashboard rows align exactly with the raw model-feature rows.

## Deployment Notes

The app is configured for Streamlit Community Cloud with relative paths and no required secrets. For production, prefer a read-only dashboard backed by a scheduled scoring job, signed/trusted model artifacts, authentication, and monitored data/model drift.
