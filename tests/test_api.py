"""
Owner: Person 5 (tests)

Tests for the write-back API in api/main.py
Run with: pytest tests/test_api.py

Uses a separate, isolated SQLite file for testing so real decision data
is never touched by the test suite.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_decisions.db")

# Point the app at a test-only database before importing it
import api.main as api_main
api_main.DB_PATH = TEST_DB_PATH

from fastapi.testclient import TestClient

client = TestClient(api_main.app)


@pytest.fixture(autouse=True)
def clean_db():
    """Ensure a fresh database for every test."""
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    api_main.init_db()
    yield
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_decision():
    response = client.post("/decisions", json={
        "order_id": "ORD-TEST-001",
        "delay_probability": 0.72,
        "chosen_action": "secondary_supplier",
        "predicted_cost": 3500.0,
    })
    assert response.status_code == 201
    body = response.json()
    assert body["order_id"] == "ORD-TEST-001"
    assert body["id"] == 1


def test_create_decision_rejects_invalid_probability():
    """delay_probability must be between 0 and 1 — this is a real validation, not just a type check."""
    response = client.post("/decisions", json={
        "order_id": "ORD-TEST-002",
        "delay_probability": 1.5,
        "chosen_action": "air_freight",
        "predicted_cost": 9000.0,
    })
    assert response.status_code == 422


def test_list_decisions_returns_created_entries():
    client.post("/decisions", json={
        "order_id": "ORD-TEST-003", "delay_probability": 0.6,
        "chosen_action": "accept_delay", "predicted_cost": 500.0,
    })
    client.post("/decisions", json={
        "order_id": "ORD-TEST-004", "delay_probability": 0.8,
        "chosen_action": "air_freight", "predicted_cost": 8000.0,
    })
    response = client.get("/decisions")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_single_decision():
    created = client.post("/decisions", json={
        "order_id": "ORD-TEST-005", "delay_probability": 0.55,
        "chosen_action": "accept_delay", "predicted_cost": 300.0,
    }).json()

    response = client.get(f"/decisions/{created['id']}")
    assert response.status_code == 200
    assert response.json()["order_id"] == "ORD-TEST-005"


def test_get_nonexistent_decision_returns_404():
    response = client.get("/decisions/9999")
    assert response.status_code == 404


def test_record_outcome_closes_the_loop():
    """The core 'closed loop' behavior: predicted cost can be compared to actual cost afterward."""
    created = client.post("/decisions", json={
        "order_id": "ORD-TEST-006", "delay_probability": 0.7,
        "chosen_action": "air_freight", "predicted_cost": 9000.0,
    }).json()

    response = client.post(f"/decisions/{created['id']}/outcome", json={"actual_cost": 11000.0})
    assert response.status_code == 200
    body = response.json()
    assert body["actual_cost"] == 11000.0
    assert body["outcome_recorded_at"] is not None
    # Predicted vs actual is now comparable
    assert body["actual_cost"] != body["predicted_cost"]


def test_record_outcome_on_nonexistent_decision_returns_404():
    response = client.post("/decisions/9999/outcome", json={"actual_cost": 100.0})
    assert response.status_code == 404