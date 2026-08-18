"""
Owner: Person 5 (tests)

Tests for the evaluation logic in model/close_the_loop.py. These test
the pure evaluate() function directly with constructed data, so they
don't require the API server to be running.
Run with: pytest tests/test_close_the_loop.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from model.close_the_loop import evaluate


@pytest.fixture
def closed_loop_df():
    return pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4"],
        "likely_delayed": [True, True, True, False],
        "recommended_action": ["accept_delay", "secondary_supplier", "air_freight", None],
        "recommended_cost": [1000.0, 3000.0, 9000.0, None],
        "decision_id": [1, 2, 3, None],
        "actual_cost": [1100.0, 2800.0, 11000.0, None],
    })


def test_evaluate_counts_only_closed_decisions(closed_loop_df):
    result = evaluate(closed_loop_df)
    assert result["decisions_evaluated"] == 3


def test_evaluate_totals_are_correct(closed_loop_df):
    result = evaluate(closed_loop_df)
    assert result["total_predicted_cost"] == 13000.0
    assert result["total_actual_cost"] == 14900.0


def test_evaluate_identifies_worst_prediction(closed_loop_df):
    result = evaluate(closed_loop_df)
    # air_freight: predicted 9000, actual 11000 -> ~22% error, the largest
    assert result["worst_prediction_order_id"] == "ORD-3"


def test_evaluate_handles_no_closed_decisions():
    empty_df = pd.DataFrame({
        "order_id": ["ORD-1"],
        "likely_delayed": [False],
        "recommended_action": [None],
        "recommended_cost": [None],
        "decision_id": [None],
        "actual_cost": [None],
    })
    result = evaluate(empty_df)
    assert result["decisions_evaluated"] == 0
    assert result["worst_prediction_order_id"] is None