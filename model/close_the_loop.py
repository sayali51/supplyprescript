"""
Owner: Person 1, 2 & 3 (full loop integration)

Connects the batch pipeline (model + solver) to the write-back API:
for every at-risk order, logs the recommended decision via POST
/decisions. Then simulates a real-world outcome for each (since we
don't have real fulfillment data) and records it via POST
/decisions/{id}/outcome, closing the loop.

Finally, evaluates how close predicted costs were to "actual" costs —
the core "did our recommendation work?" question from the brief.

IMPORTANT: the api server must already be running before this script
is used:
    python -m uvicorn api.main:app --port 8000

Run with: python model/close_the_loop.py
"""

import os
import sys
import random
import requests
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.batch_run import run_batch, DEFAULT_MAX_BUDGET
from model.pipeline import load_trained_model

API_BASE = os.getenv("SUPPLYPRESCRIPT_API", "http://127.0.0.1:8000")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "supply_orders.csv")

random.seed(7)


def log_decisions(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    POSTs every at-risk order's recommendation to the write-back API.
    Returns the results_df with a new 'decision_id' column, so outcomes
    can be recorded against the right rows later.
    """
    decision_ids = []
    for _, row in results_df.iterrows():
        if not row["likely_delayed"]:
            decision_ids.append(None)
            continue

        payload = {
            "order_id": row["order_id"],
            "delay_probability": row["delay_probability"],
            "chosen_action": row["recommended_action"],
            "predicted_cost": row["recommended_cost"],
        }
        response = requests.post(f"{API_BASE}/decisions", json=payload, timeout=10)
        response.raise_for_status()
        decision_ids.append(response.json()["id"])

    results_df = results_df.copy()
    results_df["decision_id"] = decision_ids
    return results_df


def simulate_and_record_outcomes(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    We don't have real fulfillment data, so we simulate a realistic
    'actual' outcome for each logged decision: the true cost is the
    predicted cost plus random variance (+/- 25%), reflecting that
    real-world costs rarely match a prediction exactly. This is
    clearly documented as simulated, not real outcome data.
    """
    actual_costs = []
    for _, row in results_df.iterrows():
        if pd.isna(row["decision_id"]):
            actual_costs.append(None)
            continue

        variance = random.uniform(-0.25, 0.35)  # costs skew slightly higher than predicted, realistically
        actual_cost = round(row["recommended_cost"] * (1 + variance), 2)

        response = requests.post(
            f"{API_BASE}/decisions/{int(row['decision_id'])}/outcome",
            json={"actual_cost": actual_cost},
            timeout=10,
        )
        response.raise_for_status()
        actual_costs.append(actual_cost)

    results_df = results_df.copy()
    results_df["actual_cost"] = actual_costs
    return results_df


def evaluate(results_df: pd.DataFrame) -> dict:
    """
    Compares predicted vs actual cost across all closed-loop decisions.
    This is the literal 'did the recommendation hold up?' evaluation
    described in the brief.
    """
    closed = results_df.dropna(subset=["actual_cost"]).copy()
    closed["error"] = closed["actual_cost"] - closed["recommended_cost"]
    closed["abs_pct_error"] = (closed["error"].abs() / closed["recommended_cost"]) * 100

    return {
        "decisions_evaluated": len(closed),
        "total_predicted_cost": round(closed["recommended_cost"].sum(), 2),
        "total_actual_cost": round(closed["actual_cost"].sum(), 2),
        "mean_absolute_pct_error": round(closed["abs_pct_error"].mean(), 2),
        "worst_prediction_order_id": closed.loc[closed["abs_pct_error"].idxmax(), "order_id"]
            if len(closed) else None,
    }


if __name__ == "__main__":
    print("Checking API is reachable...")
    try:
        requests.get(f"{API_BASE}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"ERROR: API not reachable at {API_BASE}. "
            f"Start it first with: python -m uvicorn api.main:app --port 8000")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    model = load_trained_model()

    print("Running batch predictions + recommendations...")
    results = run_batch(df, model, max_budget=DEFAULT_MAX_BUDGET)

    at_risk_count = results["likely_delayed"].sum()
    print(f"Logging {at_risk_count} at-risk decisions to the write-back API...")
    results = log_decisions(results)

    print("Simulating and recording outcomes (closing the loop)...")
    results = simulate_and_record_outcomes(results)

    summary = evaluate(results)
    print("\n--- Closed-Loop Evaluation ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "closed_loop_results.csv")
    results.to_csv(output_path, index=False)
    print(f"\nFull results saved to {output_path}")