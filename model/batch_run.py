"""
Owner: Person 1 & 2 (model + solver integration)

Runs the predict-and-recommend pipeline across every order in the
dataset, not just a single example. Produces a summary: how many orders
are at risk, what the total recommended spend would be, and a
breakdown of which action type gets recommended most often.

This is the first "batch" view of the system — useful for a dashboard
later, and for the demo ("here's what the system would do across our
whole order book, right now").

Run with: python model/batch_run.py
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train_model import FEATURES
from model.pipeline import load_trained_model, predict_and_recommend

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "supply_orders.csv")
DEFAULT_MAX_BUDGET = 5000


def run_batch(df: pd.DataFrame, model, max_budget: float = DEFAULT_MAX_BUDGET) -> pd.DataFrame:
    """
    Apply predict_and_recommend to every row in df. Returns a DataFrame
    with one row per order: order_id, delay_probability, likely_delayed,
    recommended_action, recommended_cost, within_budget.
    """
    results = []
    for _, row in df.iterrows():
        order = row.to_dict()
        result = predict_and_recommend(order, model, max_budget=max_budget)

        rec = result["recommendation"]
        results.append({
            "order_id": result["order_id"],
            "delay_probability": result["delay_probability"],
            "likely_delayed": result["likely_delayed"],
            "recommended_action": rec["recommended_action"] if rec else None,
            "recommended_cost": rec["recommended_cost"] if rec else None,
            "within_budget": rec["within_budget"] if rec else None,
        })

    return pd.DataFrame(results)


def summarize(results_df: pd.DataFrame) -> dict:
    """Aggregate stats across the batch run — the kind of numbers a manager would actually want."""
    total_orders = len(results_df)
    at_risk = int(results_df["likely_delayed"].sum())
    total_recommended_spend = results_df["recommended_cost"].dropna().sum()
    action_breakdown = (
        results_df["recommended_action"].dropna().value_counts().to_dict()
    )
    over_budget_count = int((results_df["within_budget"] == False).sum())

    return {
        "total_orders": total_orders,
        "orders_at_risk": at_risk,
        "risk_rate": round(at_risk / total_orders, 3) if total_orders else 0,
        "total_recommended_spend": round(total_recommended_spend, 2),
        "action_breakdown": action_breakdown,
        "orders_over_budget": over_budget_count,
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    model = load_trained_model()

    results_df = run_batch(df, model, max_budget=DEFAULT_MAX_BUDGET)
    summary = summarize(results_df)

    print(f"Ran pipeline across {summary['total_orders']} orders (budget cap: ${DEFAULT_MAX_BUDGET:,})\n")
    print(f"Orders at risk of delay: {summary['orders_at_risk']} ({summary['risk_rate']*100:.1f}%)")
    print(f"Total recommended spend to mitigate: ${summary['total_recommended_spend']:,.2f}")
    print(f"Orders where even the cheapest option exceeds budget: {summary['orders_over_budget']}")
    print(f"\nRecommended action breakdown:")
    for action, count in summary["action_breakdown"].items():
        print(f"  {action}: {count}")

    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "batch_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nFull results saved to {output_path}")