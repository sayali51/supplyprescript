"""
Owner: Person 1 & 2 (model + solver integration)

Connects the trained XGBoost delay model to the PuLP solver: given a
single order's raw features, this returns both the predicted delay
probability AND (if delay is likely) a recommended action under budget.

This is the first real end-to-end piece of the "predict -> prescribe"
loop described in the brief.

Run with: python model/pipeline.py
"""

import os
import sys
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train_model import FEATURES, predict_delay_probability
from solver.recommend import recommend_action

MODEL_PATH = os.path.join(os.path.dirname(__file__), "delay_model.json")
DELAY_THRESHOLD = 0.5  # above this probability, we recommend an action


def load_trained_model(path: str = MODEL_PATH) -> xgb.XGBClassifier:
    """Load a previously trained and saved model."""
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def predict_and_recommend(order: dict, model: xgb.XGBClassifier, max_budget: float) -> dict:
    """
    Full pipeline for a single order:
    1. Predict delay probability from the model
    2. If likely delayed, run the solver to recommend the best action
    3. If not likely delayed, no action is needed

    `order` must contain the 5 model features (see model.train_model.FEATURES)
    plus the three cost fields the solver needs:
    cost_air_freight, cost_secondary_supplier, cost_delay_penalty
    """
    delay_probability = predict_delay_probability(model, order)
    likely_delayed = delay_probability >= DELAY_THRESHOLD

    result = {
        "order_id": order.get("order_id", "unknown"),
        "delay_probability": delay_probability,
        "likely_delayed": likely_delayed,
        "recommendation": None,
    }

    if likely_delayed:
        recommendation = recommend_action(
            cost_air_freight=order["cost_air_freight"],
            cost_secondary_supplier=order["cost_secondary_supplier"],
            cost_delay_penalty=order["cost_delay_penalty"],
            max_budget=max_budget,
        )
        result["recommendation"] = recommendation

    return result


if __name__ == "__main__":
    model = load_trained_model()

    # Example: a risky order (unreliable supplier, poor history, long distance)
    example_order = {
        "order_id": "ORD-DEMO-001",
        "lead_time_days": 5.0,
        "distance_km": 4000.0,
        "supplier_reliability": 0.55,
        "historical_delay_rate": 0.35,
        "order_value": 25000.0,
        "cost_air_freight": 9000.0,
        "cost_secondary_supplier": 3500.0,
        "cost_delay_penalty": 1200.0,
    }

    result = predict_and_recommend(example_order, model, max_budget=5000)

    print(f"Order: {result['order_id']}")
    print(f"Delay probability: {result['delay_probability']*100:.1f}%")
    print(f"Likely delayed: {result['likely_delayed']}")
    if result["recommendation"]:
        rec = result["recommendation"]
        print(f"Recommended action: {rec['recommended_action']} "
            f"(cost: {rec['recommended_cost']}, within budget: {rec['within_budget']})")
    else:
        print("No action needed — delay unlikely.")