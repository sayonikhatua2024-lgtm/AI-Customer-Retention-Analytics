# Data Card: IBM Telco Customer Churn Dataset

## Dataset Summary

- **Source file:** `data/raw/Telco_customer_churn.xlsx`
- **Rows:** 7,043 customers
- **Raw columns:** 33
- **Processed dashboard file:** `data/processed/dashboard_data.csv`

The dataset contains telecom customer demographics, account details, subscribed services, billing details, and churn labels.

## Target

- **Target column:** `Churn Label`
- **Target mapping:** `No -> 0`, `Yes -> 1`

## Excluded Columns

The training pipeline drops identifiers, geography fields, and leakage/post-outcome fields before modeling. Examples include `CustomerID`, `Churn Value`, `Churn Score`, and `Churn Reason`.

`CustomerID` is preserved for dashboard display/export when available, but it is not used as a model feature.

## Cleaning Rules

- `Total Charges` is coerced to numeric.
- Blank or non-numeric `Total Charges` values are filled with `0`, matching the original notebook treatment for brand-new customers.
- Modeling uses the exact schema defined in `src.config.MODEL_FEATURES`.

## Known Limitations

- This is a static public dataset and may not reflect current telecom customer behavior.
- The data does not include intervention history, offer acceptance, agent capacity, or true retention outcome uplift.
- Geography fields are dropped for the current model; production use may require fairness and regional performance analysis.

## Recommended Production Additions

- Data freshness checks
- Schema drift checks
- Missing-value monitoring
- Segment-level model performance reporting
- Privacy review for customer-level exports
