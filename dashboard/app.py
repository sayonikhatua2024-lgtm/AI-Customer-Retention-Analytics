"""
app.py
======
AI Customer Retention Analytics Platform — Streamlit dashboard.

Run with:
    streamlit run dashboard/app.py

Design notes
------------
- All business logic (risk thresholds, recommendation rules, feature
  schema) lives in `src/`, not here — this file is presentation only.
- Every data/model load goes through a friendly error path: if an
  asset is missing, the user sees an actionable message instead of a
  stack trace, and (where possible) the app offers to regenerate it.
- Cached with st.cache_data / st.cache_resource so filtering and
  navigating tabs stays snappy.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable regardless of the CWD Streamlit was
# launched from (needed for `streamlit run dashboard/app.py` to find `src`).
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.data_pipeline import generate_dashboard_dataset, load_dashboard_data
from src.feature_engineering import add_business_features
from src.model import get_feature_importances, load_model
from src.utils import MissingAssetError, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Customer Retention Analytics",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_THEME_CSS = """
<style>
    .stApp { background-color: #0E1117; }
    section[data-testid="stSidebar"] { background-color: #131722; }

    .kpi-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #161A26 100%);
        border: 1px solid #262B3D;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: left;
    }
    .kpi-label {
        color: #8B93A7;
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #F5F6FA;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-sub {
        color: #6FCF97;
        font-size: 0.82rem;
        margin-top: 6px;
    }
    .kpi-sub.warn { color: #F5B041; }
    .kpi-sub.danger { color: #E74C3C; }

    .section-title {
        color: #F5F6FA;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 6px 0 12px 0;
        border-left: 4px solid #4C6FFF;
        padding-left: 10px;
    }

    div[data-testid="stMetricValue"] { color: #F5F6FA; }
</style>
"""
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def get_model():
    return load_model()


@st.cache_data(show_spinner="Loading customer data...")
def get_dashboard_df(_model) -> pd.DataFrame:
    df = load_dashboard_data(model=_model)
    df = add_business_features(df)
    df = df.reset_index(drop=True)
    if "Customer ID" not in df.columns:
        if "CustomerID" in df.columns:
            df.insert(0, "Customer ID", df["CustomerID"])
        else:
            df.insert(0, "Customer ID", [f"CUST-{i:06d}" for i in range(1, len(df) + 1)])
    return df


@st.cache_data(show_spinner=False)
def get_feature_importance_df(_model) -> pd.DataFrame:
    return get_feature_importances(_model)


# ---------------------------------------------------------------------------
# Boot sequence with friendly error handling
# ---------------------------------------------------------------------------
def render_missing_asset_screen(error: MissingAssetError) -> None:
    st.error("**Missing required asset**")
    st.markdown(str(error).replace("\n", "\n\n"))
    st.info(
        "This dashboard expects the standard project layout:\n\n"
        "- `models/customer_churn_model.pkl`\n"
        "- `data/raw/Telco_customer_churn.xlsx`\n"
        "- `data/processed/dashboard_data.csv` (auto-generated if missing, "
        "as long as the two files above exist)\n\n"
        "See the README for setup instructions."
    )
    st.stop()


try:
    model = get_model()
except MissingAssetError as e:
    render_missing_asset_screen(e)

try:
    df = get_dashboard_df(model)
except MissingAssetError as e:
    render_missing_asset_screen(e)
except Exception as e:  # noqa: BLE001 - surface any other load error clearly
    st.error(f"Failed to load or generate the dashboard dataset: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
st.sidebar.title("\U0001F4CA Retention Analytics")
st.sidebar.caption("AI-powered customer churn intelligence")
st.sidebar.divider()

st.sidebar.subheader("Filters")

risk_filter = st.sidebar.multiselect(
    "Risk Level",
    options=config.RISK_LEVEL_ORDER,
    default=config.RISK_LEVEL_ORDER,
)

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=sorted(df["Contract"].unique()),
    default=sorted(df["Contract"].unique()),
)

internet_filter = st.sidebar.multiselect(
    "Internet Service",
    options=sorted(df["Internet Service"].unique()),
    default=sorted(df["Internet Service"].unique()),
)

tenure_min, tenure_max = int(df["Tenure Months"].min()), int(df["Tenure Months"].max())
tenure_range = st.sidebar.slider(
    "Tenure (months)", min_value=tenure_min, max_value=tenure_max, value=(tenure_min, tenure_max)
)

charge_min, charge_max = float(df["Monthly Charges"].min()), float(df["Monthly Charges"].max())
charge_range = st.sidebar.slider(
    "Monthly Charges ($)",
    min_value=charge_min,
    max_value=charge_max,
    value=(charge_min, charge_max),
)

st.sidebar.divider()
if st.sidebar.button("\U0001F504 Regenerate dashboard data", use_container_width=True):
    with st.spinner("Rescoring all customers with the trained model..."):
        generate_dashboard_dataset(model, save=True)
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    f"Scored {len(df):,} customers · Model: "
    f"{type(model.named_steps['classifier']).__name__}"
)

filtered_df = df[
    df["Risk Level"].isin(risk_filter)
    & df["Contract"].isin(contract_filter)
    & df["Internet Service"].isin(internet_filter)
    & df["Tenure Months"].between(*tenure_range)
    & df["Monthly Charges"].between(*charge_range)
]
has_filtered_rows = not filtered_df.empty

# ---------------------------------------------------------------------------
# Header + KPI cards
# ---------------------------------------------------------------------------
st.title("AI Customer Retention Analytics Platform")
st.caption(
    "Predictive churn intelligence, risk segmentation, and explainable "
    "retention recommendations for the telecom customer base."
)

total_customers = len(filtered_df)
avg_churn_prob = filtered_df["Churn Probability"].mean() if total_customers else 0
high_risk_count = filtered_df["Risk Level"].isin(["High", "Critical"]).sum()
revenue_at_risk = filtered_df.loc[
    filtered_df["Risk Level"].isin(["High", "Critical"]), "Monthly Charges"
].sum()

if "Actual Churn" in filtered_df.columns and filtered_df["Actual Churn"].notna().any():
    actual_churn_rate = filtered_df["Actual Churn"].mean() * 100
    churn_kpi_label = "Historical Churn Rate"
    churn_kpi_value = f"{actual_churn_rate:.1f}%"
else:
    churn_kpi_label = "Avg. Predicted Churn Risk"
    churn_kpi_value = f"{avg_churn_prob * 100:.1f}%"

kpi_cols = st.columns(4)
kpi_data = [
    ("Total Customers", f"{total_customers:,}", "In current filter", ""),
    (churn_kpi_label, churn_kpi_value, "Across filtered segment", "warn"),
    ("High / Critical Risk", f"{high_risk_count:,}", f"{(high_risk_count / total_customers * 100 if total_customers else 0):.1f}% of segment", "danger"),
    ("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}", "From High/Critical customers", "danger"),
]
for col, (label, value, sub, tone) in zip(kpi_cols, kpi_data):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub {tone}">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_customers, tab_explain, tab_insights, tab_data = st.tabs(
    [
        "\U0001F4C8 Overview",
        "\U0001F465 Customers & Risk",
        "\U0001F9E0 Explainability",
        "\U0001F4A1 Business Insights",
        "\U0001F4C1 Data & Export",
    ]
)

PLOTLY_TEMPLATE = "plotly_dark"

if not has_filtered_rows:
    st.warning("No customers match the current filters. Adjust the sidebar filters to view dashboard content.")
    st.stop()

# ---- Overview tab ----------------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.4])

    with c1:
        risk_counts = (
            filtered_df["Risk Level"]
            .value_counts()
            .reindex(config.RISK_LEVEL_ORDER)
            .fillna(0)
            .reset_index()
        )
        risk_counts.columns = ["Risk Level", "Count"]
        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            hole=0.55,
            color="Risk Level",
            color_discrete_map=config.RISK_COLORS,
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            filtered_df,
            x="Churn Probability",
            nbins=30,
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#4C6FFF"],
        )
        fig.update_layout(
            title="Churn Probability Distribution",
            margin=dict(t=40, b=10, l=10, r=10),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Churn Drivers by Segment</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        contract_risk = (
            filtered_df.groupby(["Contract", "Risk Level"]).size().reset_index(name="Count")
        )
        fig = px.bar(
            contract_risk,
            x="Contract",
            y="Count",
            color="Risk Level",
            color_discrete_map=config.RISK_COLORS,
            category_orders={"Risk Level": config.RISK_LEVEL_ORDER},
            template=PLOTLY_TEMPLATE,
            barmode="stack",
        )
        fig.update_layout(title="Risk by Contract Type", height=360, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        internet_risk = (
            filtered_df.groupby(["Internet Service", "Risk Level"]).size().reset_index(name="Count")
        )
        fig = px.bar(
            internet_risk,
            x="Internet Service",
            y="Count",
            color="Risk Level",
            color_discrete_map=config.RISK_COLORS,
            category_orders={"Risk Level": config.RISK_LEVEL_ORDER},
            template=PLOTLY_TEMPLATE,
            barmode="stack",
        )
        fig.update_layout(title="Risk by Internet Service", height=360, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fig = px.box(
            filtered_df,
            x="Risk Level",
            y="Monthly Charges",
            color="Risk Level",
            color_discrete_map=config.RISK_COLORS,
            category_orders={"Risk Level": config.RISK_LEVEL_ORDER},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(title="Monthly Charges vs. Risk Level", height=340, margin=dict(t=40), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        fig = px.box(
            filtered_df,
            x="Risk Level",
            y="Tenure Months",
            color="Risk Level",
            color_discrete_map=config.RISK_COLORS,
            category_orders={"Risk Level": config.RISK_LEVEL_ORDER},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(title="Tenure vs. Risk Level", height=340, margin=dict(t=40), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ---- Customers & Risk tab --------------------------------------------------
with tab_customers:
    st.markdown('<div class="section-title">High-Risk Customers</div>', unsafe_allow_html=True)

    high_risk_df = filtered_df[filtered_df["Risk Level"].isin(["High", "Critical"])].sort_values(
        "Churn Probability", ascending=False
    )
    st.dataframe(
        high_risk_df[
            [
                "Customer ID",
                "Risk Level",
                "Churn Probability",
                "Contract",
                "Monthly Charges",
                "Tenure Months",
                "Recommendation",
            ]
        ],
        use_container_width=True,
        height=320,
        column_config={
            "Churn Probability": st.column_config.ProgressColumn(
                "Churn Probability", min_value=0, max_value=1, format="%.2f"
            ),
            "Monthly Charges": st.column_config.NumberColumn("Monthly Charges", format="$%.2f"),
        },
    )

    st.download_button(
        "\U0001F4E5 Download High-Risk Customers (CSV)",
        data=high_risk_df.to_csv(index=False).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )

    st.markdown('<div class="section-title">Customer Lookup</div>', unsafe_allow_html=True)
    selected_id = st.selectbox("Select a Customer ID", options=filtered_df["Customer ID"])
    customer_row = filtered_df[filtered_df["Customer ID"] == selected_id].iloc[0]

    p1, p2, p3 = st.columns(3)
    p1.metric("Churn Probability", f"{customer_row['Churn Probability']:.1%}")
    p2.metric("Risk Level", customer_row["Risk Level"])
    p3.metric("Monthly Charges", f"${customer_row['Monthly Charges']:.2f}")

    st.info(f"**Recommended Action:** {customer_row['Recommendation']}")

    with st.expander("View full customer profile"):
        profile_cols = [c for c in config.MODEL_FEATURES if c in customer_row.index]
        st.table(customer_row[profile_cols].rename("Value").to_frame())

# ---- Explainability tab ----------------------------------------------------
with tab_explain:
    st.markdown('<div class="section-title">Global Feature Importance</div>', unsafe_allow_html=True)
    st.caption(
        "Native Random Forest feature importances — how much each feature "
        "reduces prediction error across the whole model."
    )

    try:
        importance_df = get_feature_importance_df(model).head(15)
        fig = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#4C6FFF"],
        )
        fig.update_layout(height=460, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    except AttributeError as e:
        st.warning(str(e))

    st.markdown('<div class="section-title">Why did the model flag this customer?</div>', unsafe_allow_html=True)
    st.caption(
        "SHAP waterfall plot: how each feature pushed this customer's "
        "predicted churn probability up (red) or down (blue) from the "
        "model's average prediction."
    )

    explain_id = st.selectbox(
        "Select a Customer ID to explain", options=filtered_df["Customer ID"], key="explain_select"
    )

    if st.button("Generate SHAP Explanation", type="primary"):
        try:
            from src.explainability import explain_customer

            row = filtered_df[filtered_df["Customer ID"] == explain_id]
            X_row = row[config.MODEL_FEATURES]

            with st.spinner("Computing SHAP values..."):
                explanation = explain_customer(model, X_row)

            import shap

            fig, ax = plt.subplots(figsize=(9, 6))
            plt.sca(ax)
            shap.plots.waterfall(explanation, max_display=15, show=False)
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except ImportError:
            st.error("SHAP is not installed. Run: pip install shap")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not compute SHAP explanation for this customer: {e}")

# ---- Business Insights tab -------------------------------------------------
with tab_insights:
    st.markdown('<div class="section-title">Key Observations</div>', unsafe_allow_html=True)
    st.caption("Derived from the exploratory data analysis in `notebooks/01_Data_Understanding.ipynb`.")

    insights = [
        (
            "Fiber Optic customers churn the most",
            "Customers on Fiber Optic internet show a markedly higher churn rate than DSL "
            "users, while customers with no internet service churn the least. "
            "**Action:** review Fiber Optic pricing, reliability, and support quality.",
        ),
        (
            "Electronic Check payers are highest-risk",
            "Customers paying via Electronic Check churn far more than those on automatic "
            "payment methods. **Action:** incentivize migration to autopay with a small "
            "discount or cashback.",
        ),
        (
            "Month-to-month contracts drive churn",
            "Short-term, flexible contracts correlate strongly with churn versus one- and "
            "two-year contracts. **Action:** promote annual plans with a loyalty discount.",
        ),
        (
            "Tenure and lifetime value are closely linked",
            "Tenure shows the strongest positive relationship with customer lifetime value. "
            "**Action:** early-tenure retention has outsized long-term revenue impact.",
        ),
        (
            "Customers with dependents churn less",
            "Family-oriented customers (with partners and/or dependents) show lower churn. "
            "**Action:** bundle family plans to increase stickiness.",
        ),
    ]
    for title, body in insights:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(body)

    st.markdown('<div class="section-title">Recommendation Mix</div>', unsafe_allow_html=True)
    rec_counts = filtered_df["Recommendation"].value_counts().reset_index()
    rec_counts.columns = ["Recommendation", "Count"]
    fig = px.bar(
        rec_counts,
        x="Count",
        y="Recommendation",
        orientation="h",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=["#6FCF97"],
    )
    fig.update_layout(height=380, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

# ---- Data & Export tab ------------------------------------------------------
with tab_data:
    st.markdown('<div class="section-title">Filtered Dataset</div>', unsafe_allow_html=True)
    st.caption(f"{len(filtered_df):,} customers match the current filters.")
    st.dataframe(filtered_df, use_container_width=True, height=460)

    st.download_button(
        "\U0001F4E5 Download Filtered Data (CSV)",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customer_data.csv",
        mime="text/csv",
    )

    st.markdown('<div class="section-title">About this dataset</div>', unsafe_allow_html=True)
    st.markdown(
        "- Source: IBM Telco Customer Churn dataset (`data/raw/Telco_customer_churn.xlsx`)\n"
        "- Scored by: `models/customer_churn_model.pkl` (Random Forest, tuned via GridSearchCV)\n"
        "- Risk thresholds and recommendation rules: see `src/risk_engine.py` and "
        "`src/recommendation_engine.py`\n"
        "- `Customer ID` is a synthetic display identifier generated by the dashboard — "
        "the original `CustomerID` field was dropped upstream as a non-predictive identifier."
    )
