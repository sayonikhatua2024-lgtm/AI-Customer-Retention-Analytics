# AI Customer Retention Analytics Platform

An end-to-end machine learning system that predicts customer churn,
segments customers by risk, explains *why* the model thinks a
customer will churn, and recommends a retention action — wrapped in
an interactive Streamlit dashboard.

Built on the IBM Telco Customer Churn dataset (7,043 customers).

---

## Overview

Telecom companies lose a meaningful share of revenue every year to
avoidable customer churn. This project builds a full retention
pipeline on top of that problem:

**Raw data → cleaning → model → risk scoring → explainability → retention recommendation → dashboard**

It's designed to be read end-to-end by a technical reviewer in a few
minutes: a Jupyter notebook for the exploratory / model-selection
work, and a clean `src/` package that turns that notebook into
production code the dashboard actually runs on (no logic is
duplicated between the two).

## Business Problem

Given a customer's account and service attributes, predict the
probability they will churn, and translate that into three things a
retention team can act on immediately:

1. **Risk tier** (Low / Medium / High / Critical)
2. **Why** — the specific features driving that customer's score (SHAP)
3. **What to do about it** — a concrete, auditable recommended action

## Features

- **Churn prediction** for the full customer base via a tuned Random Forest
- **Risk segmentation** into four tiers with configurable thresholds
- **Rule-based retention recommendations**, keyed on risk tier, contract type, and monthly spend
- **SHAP explainability** — global feature importance and per-customer waterfall plots
- **Interactive Streamlit dashboard** — KPI cards, Plotly charts, customer lookup, high-risk customer export
- **Self-healing data layer** — if the processed dashboard dataset is missing, the app regenerates it from the raw data and trained model automatically
- **Retrainable pipeline** — `src/train.py` reproduces the exact model-selection process from the notebook end-to-end

## Architecture

```
Raw Excel (Telco_customer_churn.xlsx)
        │
        ▼
 data_pipeline.py   — load, clean, coerce types
        │
        ▼
   train.py          — preprocessing + GridSearchCV Random Forest ──► models/customer_churn_model.pkl
        │
        ▼
 data_pipeline.py    — score full customer base
        │
        ▼
 risk_engine.py       — probability → risk tier
        │
        ▼
 recommendation_engine.py — risk tier → retention action
        │
        ▼
 data/processed/dashboard_data.csv
        │
        ▼
 dashboard/app.py  ◄── explainability.py (SHAP)
                    ◄── feature_engineering.py (segmentation features)
                    ◄── model.py (feature importances)
```

Each pipeline stage is its own module with a single responsibility —
no logic is duplicated between the notebook, the training script, and
the dashboard.

## Folder Structure

```
AI-Customer-Retention-Analytics/
├── dashboard/
│   └── app.py                    # Streamlit dashboard (presentation layer only)
├── src/
│   ├── config.py                 # Paths, feature schema, thresholds — single source of truth
│   ├── data_pipeline.py          # Load / clean / dashboard dataset generation
│   ├── feature_engineering.py    # Business-facing analytics features (dashboard only)
│   ├── model.py                  # Model load / predict / feature importance
│   ├── risk_engine.py            # Probability → risk tier
│   ├── recommendation_engine.py  # Risk tier → retention action
│   ├── explainability.py         # SHAP wrappers
│   ├── train.py                  # End-to-end, CLI-runnable training script
│   └── utils.py                  # Logging + friendly missing-file errors
├── data/
│   ├── raw/                      # Telco_customer_churn.xlsx
│   └── processed/                # dashboard_data.csv (auto-regenerated if missing)
├── models/
│   └── customer_churn_model.pkl  # Trained sklearn Pipeline
├── notebooks/
│   └── 01_Data_Understanding.ipynb  # EDA, model comparison, SHAP, notebook-native workflow
├── screenshots/                  # Dashboard screenshots (see screenshots/README.md)
├── tests/                        # pytest unit tests for the ML modules
├── requirements.txt
└── .gitignore
```

## Technologies

- **ML:** scikit-learn (Pipeline, ColumnTransformer, RandomForestClassifier, GridSearchCV)
- **Explainability:** SHAP (TreeExplainer)
- **Dashboard:** Streamlit, Plotly, Matplotlib
- **Data:** pandas, NumPy, openpyxl
- **Testing:** pytest

## Installation

```bash
git clone <your-repo-url>
cd AI-Customer-Retention-Analytics

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

**Run the dashboard:**

```bash
streamlit run dashboard/app.py
```

On first run, if `data/processed/dashboard_data.csv` isn't present,
the app scores the full customer base from
`data/raw/Telco_customer_churn.xlsx` using the trained model in
`models/customer_churn_model.pkl` and generates it automatically. You
can also force a rescore anytime from the sidebar
("Regenerate dashboard data").

**Retrain the model from scratch:**

```bash
python -m src.train              # full GridSearchCV (slower, matches notebook exactly)
python -m src.train --fast       # single fit with known-good params (fast iteration)
```

**Run the test suite:**

```bash
pytest tests/ -v
```

**Explore the original analysis:**

```bash
jupyter notebook notebooks/01_Data_Understanding.ipynb
```

## Results

Held-out test set (1,409 customers, 20% split, `random_state=42`),
evaluated directly against the shipped `customer_churn_model.pkl`
(Random Forest, GridSearchCV-tuned: `n_estimators=200`,
`max_depth=10`, `min_samples_split=5`):

| Metric | Value |
|---|---|
| Accuracy | 0.805 |
| ROC-AUC | 0.851 |
| PR-AUC | 0.670 |
| Precision (churn class) | 0.659 |
| Recall (churn class) | 0.548 |
| F1 (churn class) | 0.599 |

```
Confusion Matrix
                Predicted: No   Predicted: Yes
Actual: No           929              106
Actual: Yes          169              205
```

**Reading these numbers honestly:** recall on the churn class (0.548)
means the model misses roughly 45% of customers who actually churn —
common for imbalanced churn datasets (~27% positive class here) and a
known trade-off of optimizing for F1 during tuning. The
`Explainability` tab and `src/risk_engine.py`'s adjustable thresholds
exist specifically so a retention team can trade precision for recall
(e.g., lower the "High" threshold to catch more at-risk customers, at
the cost of more false positives to act on).

## Explainable AI (SHAP)

The dashboard's Explainability tab exposes two views:

- **Global feature importance** — which features matter most to the
  model overall (Random Forest's native `feature_importances_`)
- **Per-customer SHAP waterfall** — for any selected customer, exactly
  which features pushed their churn probability up or down from the
  model's average prediction, and by how much

This matters for a retention program in practice: a call-center agent
or account manager needs a *reason*, not just a score, before they can
have a productive retention conversation.

## Dashboard Screenshots

See [`screenshots/README.md`](screenshots/README.md) for what to
capture and where it's referenced. Screenshots aren't included yet —
run the dashboard locally and add real ones.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (the `.gitignore` already excludes
   local/venv artifacts — `data/`, `models/`, and `notebooks/` are
   intentionally **not** ignored, since the dashboard needs them at
   runtime).
2. On [share.streamlit.io](https://share.streamlit.io), create a new
   app pointing at this repo.
3. Set the main file path to `dashboard/app.py`.
4. Deploy. No secrets or environment variables are required — all
   paths are relative (`src/config.py`), so the app runs identically
   locally and in the cloud.

## Known Limitations & Design Decisions

- **Class imbalance:** ~27% of customers in the training data
  actually churned. This caps achievable recall without either
  resampling or adjusting the classification threshold — both are
  natural next steps (see below).
- **Rule-based recommendations, not a second model:** the
  risk → action mapping in `recommendation_engine.py` is an explicit,
  auditable rule table rather than a learned policy, by design — a
  retention program needs to be able to explain *why* a given action
  was suggested to a non-technical stakeholder in one sentence.
- **Engineered features are dashboard-only:** the notebook explores
  additional features (`Avg Monthly Spend`, `Long Term Customer`,
  etc., see `feature_engineering.py`) that were never fed back into
  model training — inspecting the shipped pickle confirms its
  `ColumnTransformer` only accepts the original 19 raw features. They
  remain available for dashboard segmentation, and as a candidate
  feature set for a future retraining pass (see below).
- **Synthetic Customer ID:** the original `CustomerID` field is
  dropped upstream as a non-predictive identifier (consistent with
  the notebook's EDA). The dashboard's `Customer ID` column is a
  display-only identifier generated at dashboard-build time, not the
  original account ID.

## Future Improvements

- Retrain including the currently dashboard-only engineered features, and compare against the current 19-feature baseline
- Address class imbalance directly (class weighting, SMOTE, or threshold tuning tied to a business cost matrix)
- Add model monitoring / drift detection for production use
- Persist a model registry (e.g. MLflow) instead of a single `.pkl` file, to track experiments across retrains
- Add authentication if this were to move from a portfolio piece to an internal tool
- A/B test the rule-based recommendation engine against actual retention outcomes, and replace rules that don't move the needle

---

*Built as a portfolio project demonstrating end-to-end ML engineering: data cleaning, model selection, hyperparameter tuning, explainability, business rule design, and production dashboard development.*
