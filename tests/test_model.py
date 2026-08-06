"""
Owner: Person 5 (tests)

Tests for the XGBoost delay prediction model.
Run with: pytest tests/test_model.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from model.train_model import train_model, predict_delay_probability, FEATURES, TARGET


@pytest.fixture
def sample_df():
    """A small synthetic dataset with a clear pattern, for fast/reliable testing."""
    import random
    random.seed(1)
    rows = []
    for i in range(60):
        reliability = random.uniform(0.5, 0.99)
        hist_rate = random.uniform(0.0, 0.4)
        lead_time = random.uniform(2, 30)
        distance = random.uniform(50, 5000)
        order_value = random.uniform(500, 50000)
        delayed = 1 if (reliability < 0.7 and hist_rate > 0.2) else 0
        rows.append({
            "lead_time_days": lead_time,
            "distance_km": distance,
            "supplier_reliability": reliability,
            "historical_delay_rate": hist_rate,
            "order_value": order_value,
            "delayed": delayed,
        })
    return pd.DataFrame(rows)


def test_train_model_returns_metrics(sample_df):
    model, metrics, cm = train_model(sample_df)
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_predict_delay_probability_returns_valid_range(sample_df):
    model, metrics, cm = train_model(sample_df)
    example = {
        "lead_time_days": 10.0,
        "distance_km": 2000.0,
        "supplier_reliability": 0.6,
        "historical_delay_rate": 0.3,
        "order_value": 10000.0,
    }
    prob = predict_delay_probability(model, example)
    assert 0.0 <= prob <= 1.0


def test_features_and_target_are_defined():
    assert len(FEATURES) == 5
    assert TARGET == "delayed"