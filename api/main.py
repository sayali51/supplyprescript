"""
Owner: Person 3 (write-back API)

FastAPI service that lets a user log which recommended action they
actually chose for an order. This is the literal "write-back" /
"closing the loop" piece from the brief: predictions and recommendations
mean nothing if the system never records what was actually decided, or
compares that decision's predicted cost against what really happened.

Run with:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /decisions        - log a decision for an order
    GET  /decisions        - list all logged decisions
    GET  /decisions/{id}   - get one decision
    POST /decisions/{id}/outcome - record the actual outcome (closes the loop)
"""

import os
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "decisions.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="SupplyPrescript Write-Back API", lifespan=lifespan)


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                delay_probability REAL NOT NULL,
                chosen_action TEXT NOT NULL,
                predicted_cost REAL NOT NULL,
                actual_cost REAL,
                decided_at TEXT NOT NULL,
                outcome_recorded_at TEXT
            )
        """)
        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class DecisionIn(BaseModel):
    order_id: str
    delay_probability: float = Field(ge=0.0, le=1.0)
    chosen_action: str
    predicted_cost: float = Field(ge=0.0)


class OutcomeIn(BaseModel):
    actual_cost: float = Field(ge=0.0)


@app.post("/decisions", status_code=201)
def create_decision(decision: DecisionIn):
    """Log which action was actually chosen for an order."""
    with get_conn() as conn:
        cursor = conn.execute(
            """INSERT INTO decisions
               (order_id, delay_probability, chosen_action, predicted_cost, decided_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                decision.order_id,
                decision.delay_probability,
                decision.chosen_action,
                decision.predicted_cost,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        new_id = cursor.lastrowid
    return {"id": new_id, **decision.model_dump()}


@app.get("/decisions")
def list_decisions():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM decisions ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


@app.get("/decisions/{decision_id}")
def get_decision(decision_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM decisions WHERE id = ?", (decision_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return dict(row)


@app.post("/decisions/{decision_id}/outcome")
def record_outcome(decision_id: int, outcome: OutcomeIn):
    """
    Closes the loop: records what actually happened, so predicted cost
    can later be compared against real cost.
    """
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM decisions WHERE id = ?", (decision_id,)
        ).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="Decision not found")

        conn.execute(
            """UPDATE decisions
               SET actual_cost = ?, outcome_recorded_at = ?
               WHERE id = ?""",
            (outcome.actual_cost, datetime.now(timezone.utc).isoformat(), decision_id),
        )
        conn.commit()

        updated = conn.execute(
            "SELECT * FROM decisions WHERE id = ?", (decision_id,)
        ).fetchone()
    return dict(updated)


@app.get("/health")
def health():
    return {"status": "ok"}