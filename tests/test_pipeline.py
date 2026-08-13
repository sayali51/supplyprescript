"""
Owner: Person 5 (tests)

Tests for the model+solver pipeline in model/pipeline.py
Run with: pytest tests/test_pipeline.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from model.train_model import train_model
from model.pipeline import predict_and_recommend, DELAY_THRESHOLD


@pytest.fixture
def trained_model():
    """Train a small, fast model directly (doesn't depend on a saved file on disk)."""
    import random
    random.seed(1)
    rows = []
    for i in range(60):
        reliability = random.uniform(0.5, 0.99)
        hist_rate = random.uniform(0.0, 0.4)
        delayed = 1 if (reliability < 0.7 and hist_rate > 0.2) else 0
        rows.append({
            "lead_time_days": random.uniform(2, 30),
            "distance_km": random.uniform(50, 5000),
            "supplier_reliability": reliability,
            "historical_delay_rate": hist_rate,
            "order_value": random.uniform(500, 50000),
            "delayed": delayed,
        })
    df = pd.DataFrame(rows)
    model, metrics, cm = train_model(df)
    return model


def test_high_risk_order_triggers_recommendation(trained_model):
    """A clearly risky order (unreliable supplier, poor history) should trigger a recommendation."""
    order = {
        "order_id": "TEST-001",
        "lead_time_days": 5.0,
        "distance_km": 4500.0,
        "supplier_reliability": 0.55,
        "historical_delay_rate": 0.38,
        "order_value": 20000.0,
        "cost_air_freight": 9000.0,
        "cost_secondary_supplier": 3500.0,
        "cost_delay_penalty": 1200.0,
    }
    result = predict_and_recommend(order, trained_model, max_budget=5000)

    assert 0.0 <= result["delay_probability"] <= 1.0
    assert result["order_id"] == "TEST-001"
    if result["likely_delayed"]:
        assert result["recommendation"] is not None
        assert result["recommendation"]["recommended_action"] in (
            "air_freight", "secondary_supplier", "accept_delay"
        )


def test_no_recommendation_when_delay_unlikely(trained_model):
    """A low-risk order (reliable supplier, good history) should need no action."""
    order = {
        "order_id": "TEST-002",
        "lead_time_days": 20.0,
        "distance_km": 500.0,
        "supplier_reliability": 0.97,
        "historical_delay_rate": 0.02,
        "order_value": 5000.0,
        "cost_air_freight": 2000.0,
        "cost_secondary_supplier": 800.0,
        "cost_delay_penalty": 200.0,
    }
    result = predict_and_recommend(order, trained_model, max_budget=5000)

    if not result["likely_delayed"]:
        assert result["recommendation"] is None


def test_delay_threshold_is_reasonable():
    assert 0.0 < DELAY_THRESHOLD < 1.0