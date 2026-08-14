"""
Owner: Person 5 (tests)

Tests for model/batch_run.py — running the pipeline across many orders
and summarizing results.
Run with: pytest tests/test_batch_run.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from model.train_model import train_model
from model.batch_run import run_batch, summarize


@pytest.fixture
def trained_model_and_data():
    import random
    random.seed(1)
    rows = []
    for i in range(40):
        reliability = random.uniform(0.5, 0.99)
        hist_rate = random.uniform(0.0, 0.4)
        order_value = random.uniform(500, 50000)
        delayed = 1 if (reliability < 0.7 and hist_rate > 0.2) else 0
        rows.append({
            "order_id": f"ORD-{i}",
            "lead_time_days": random.uniform(2, 30),
            "distance_km": random.uniform(50, 5000),
            "supplier_reliability": reliability,
            "historical_delay_rate": hist_rate,
            "order_value": order_value,
            "delayed": delayed,
            "cost_air_freight": order_value * 0.25,
            "cost_secondary_supplier": order_value * 0.1,
            "cost_delay_penalty": order_value * 0.05,
        })
    df = pd.DataFrame(rows)
    model, metrics, cm = train_model(df)
    return model, df


def test_run_batch_returns_one_row_per_order(trained_model_and_data):
    model, df = trained_model_and_data
    results = run_batch(df, model, max_budget=5000)
    assert len(results) == len(df)
    assert set(results.columns) == {
        "order_id", "delay_probability", "likely_delayed",
        "recommended_action", "recommended_cost", "within_budget",
    }


def test_summarize_produces_sane_aggregates(trained_model_and_data):
    model, df = trained_model_and_data
    results = run_batch(df, model, max_budget=5000)
    summary = summarize(results)

    assert summary["total_orders"] == len(df)
    assert 0 <= summary["orders_at_risk"] <= summary["total_orders"]
    assert 0.0 <= summary["risk_rate"] <= 1.0
    assert summary["total_recommended_spend"] >= 0
    assert isinstance(summary["action_breakdown"], dict)


def test_summarize_handles_zero_at_risk_orders():
    """If no orders are flagged as delayed, summarize should not crash on empty recommendation data."""
    empty_results = pd.DataFrame({
        "order_id": ["A", "B"],
        "delay_probability": [0.1, 0.2],
        "likely_delayed": [False, False],
        "recommended_action": [None, None],
        "recommended_cost": [None, None],
        "within_budget": [None, None],
    })
    summary = summarize(empty_results)
    assert summary["orders_at_risk"] == 0
    assert summary["total_recommended_spend"] == 0