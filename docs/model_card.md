# Model Card: Customer Churn Classifier

## Intended Use

This model predicts the probability that a telecom customer will churn. It is intended for retention analytics, risk segmentation, and prioritizing customer outreach in the Streamlit dashboard.

It should not be used as a fully automated decision system without human review, business-cost calibration, and operational monitoring.

## Model Details

- **Artifact:** `models/customer_churn_model.pkl`
- **Metadata:** `models/model_metadata.json`
- **Model version:** `1.0.0`
- **Algorithm:** scikit-learn `Pipeline` with preprocessing plus `RandomForestClassifier`
- **Preprocessing:** numeric scaling and categorical one-hot encoding
- **Training source:** IBM Telco Customer Churn dataset
- **Training script:** `python -m src.train`

## Features

The model uses 19 raw account/service features:

- Numeric: `Tenure Months`, `Monthly Charges`, `Total Charges`
- Categorical: gender, senior citizen status, partner/dependents, phone/internet services, contract, paperless billing, and payment method fields

Identifier, geography, and known leakage/post-outcome fields are excluded from modeling.

## Reported Holdout Metrics

The checked-in metadata records the following holdout metrics from the shipped model:

| Metric | Value |
|---|---:|
| Accuracy | 0.805 |
| ROC-AUC | 0.851 |
| PR-AUC | 0.670 |
| Precision, churn class | 0.659 |
| Recall, churn class | 0.548 |
| F1, churn class | 0.599 |

Confusion matrix:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 929 | 106 |
| Actual Yes | 169 | 205 |

## Limitations

- Churn recall is moderate; the model misses a meaningful number of true churners.
- Risk thresholds are fixed business rules and should be calibrated against intervention costs before production use.
- The model is trained on a static public dataset and does not include monitoring for drift.
- The artifact is a trusted pickle/joblib file and must not be loaded from untrusted sources.

## Governance Notes

`src.model.load_model()` validates the metadata artifact before returning the model. Validation checks the configured feature schema, transformed feature count, and key classifier hyperparameters. Runtime Python and scikit-learn versions are logged as warnings if they differ from the metadata.
