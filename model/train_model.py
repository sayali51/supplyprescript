"""
Owner: Person 1 (model)

Trains an XGBoost classifier to predict whether an order will be delayed,
based on lead time, distance, supplier reliability, historical delay
rate, and order value.

Run with: python model/train_model.py
Outputs: model/delay_model.json (trained model, saved for reuse)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "supply_orders.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "delay_model.json")

FEATURES = [
    "lead_time_days",
    "distance_km",
    "supplier_reliability",
    "historical_delay_rate",
    "order_value",
]
TARGET = "delayed"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
    }
    cm = confusion_matrix(y_test, y_pred)

    return model, metrics, cm


def predict_delay_probability(model, order_features: dict) -> float:
    """
    Given a single order's features as a dict, return the predicted
    probability (0-1) that it will be delayed.
    """
    row = pd.DataFrame([order_features])[FEATURES]
    proba = model.predict_proba(row)[0][1]
    return round(float(proba), 3)


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} orders, {df[TARGET].sum()} delayed ({df[TARGET].mean()*100:.1f}%)")

    model, metrics, cm = train_model(df)

    print("\nModel performance on held-out test set:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nConfusion matrix:\n{cm}")

    model.save_model(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Sanity check: predict on one example order
    example = {
        "lead_time_days": 5.0,
        "distance_km": 4000.0,
        "supplier_reliability": 0.6,
        "historical_delay_rate": 0.3,
        "order_value": 20000.0,
    }
    prob = predict_delay_probability(model, example)
    print(f"\nExample prediction (unreliable supplier, long distance, short lead time): "
        f"{prob*100:.1f}% delay probability")