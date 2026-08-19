"""
Owner: UI/Dashboard

SupplyPrescript operational dashboard. Reads logged decisions directly
from the FastAPI write-back API (GET /decisions), so what's shown here
is always the real, persisted state of the system — not a recomputed
guess. Includes a "Run Pipeline Now" button that triggers the full
predict -> recommend -> log -> simulate -> evaluate loop live.

Run with: streamlit run ui/app.py
Requires the API server running separately:
    python -m uvicorn api.main:app --port 8000
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
import streamlit as st

from model.close_the_loop import run_full_loop, api_is_reachable, API_BASE

st.set_page_config(page_title="SupplyPrescript", page_icon="📦", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; max-width: 1150px; }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📦 SupplyPrescript")
st.caption("Closed-loop prescriptive analytics — predict a delay, recommend an action, track whether it worked.")


@st.cache_data(ttl=5)
def fetch_decisions() -> pd.DataFrame:
    response = requests.get(f"{API_BASE}/decisions", timeout=10)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def compute_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    closed = df.dropna(subset=["actual_cost"])
    return {
        "total_decisions": len(df),
        "predicted_cost": df["predicted_cost"].sum(),
        "actual_cost": closed["actual_cost"].sum() if not closed.empty else 0,
        "mean_error_pct": (
            ((closed["actual_cost"] - closed["predicted_cost"]).abs() / closed["predicted_cost"] * 100).mean()
            if not closed.empty else 0
        ),
        "closed_count": len(closed),
    }


if not api_is_reachable():
    st.error(
        f"⚠️ Can't reach the write-back API at {API_BASE}. "
        "Start it in a separate terminal with:\n\n`python -m uvicorn api.main:app --port 8000`"
    )
    st.stop()

with st.sidebar:
    st.header("📦 SupplyPrescript")
    st.caption("XGBoost delay prediction + PuLP optimization + FastAPI write-back")
    st.divider()
    st.subheader("Run the pipeline")
    max_budget = st.number_input("Max budget per order ($)", min_value=500, max_value=50000, value=5000, step=500)
    run_clicked = st.button("▶ Run Pipeline Now", type="primary", use_container_width=True)
    st.caption("Predicts delays across all orders, logs at-risk recommendations, simulates outcomes, and evaluates.")

if run_clicked:
    with st.spinner("Running predict → recommend → log → simulate → evaluate..."):
        results, summary = run_full_loop(max_budget=max_budget)
    st.success(f"Pipeline complete — {summary['decisions_evaluated']} decisions logged and evaluated.")
    fetch_decisions.clear()

decisions_df = fetch_decisions()
kpis = compute_kpis(decisions_df)

if not kpis:
    st.info("No decisions logged yet. Click **Run Pipeline Now** in the sidebar to get started.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Decisions Logged", f"{kpis['total_decisions']:,}")
col2.metric("Total Predicted Cost", f"${kpis['predicted_cost']:,.0f}")
col3.metric("Total Actual Cost", f"${kpis['actual_cost']:,.0f}")
col4.metric("Mean Prediction Error", f"{kpis['mean_error_pct']:.1f}%")

st.divider()

chart_col, table_col = st.columns([1.2, 1])

with chart_col:
    st.subheader("Predicted vs. Actual Cost by Action")
    closed = decisions_df.dropna(subset=["actual_cost"])
    if not closed.empty:
        by_action = closed.groupby("chosen_action")[["predicted_cost", "actual_cost"]].sum()
        by_action = by_action.rename(columns={"predicted_cost": "Predicted", "actual_cost": "Actual"})
        st.bar_chart(by_action)
    else:
        st.info("No closed-loop outcomes yet.")

with table_col:
    st.subheader("Largest Prediction Errors")
    if not closed.empty:
        closed_display = closed.copy()
        closed_display["error_pct"] = (
            (closed_display["actual_cost"] - closed_display["predicted_cost"]).abs()
            / closed_display["predicted_cost"] * 100
        )
        top5 = closed_display.nlargest(5, "error_pct")[
            ["order_id", "chosen_action", "predicted_cost", "actual_cost", "error_pct"]
        ]
        top5 = top5.rename(columns={
            "order_id": "Order", "chosen_action": "Action",
            "predicted_cost": "Predicted", "actual_cost": "Actual", "error_pct": "Error %",
        })
        st.dataframe(
            top5.style.format({"Predicted": "${:,.0f}", "Actual": "${:,.0f}", "Error %": "{:.1f}%"}),
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("No closed-loop outcomes yet.")

with st.expander("View all logged decisions"):
    st.dataframe(decisions_df, hide_index=True, use_container_width=True)