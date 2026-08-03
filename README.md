# SupplyPrescript — Closed-Loop Prescriptive Analytics

## Domain

Supply Chain Operations & Operations Research

## Problem Statement

Predictive analytics (e.g., predicting a supply chain delay) tell you
_what will happen_, but a human still has to figure out _what to do_
about it — and most systems never check whether that decision actually
worked. SupplyPrescript closes the loop: predict a delay, recommend the
best of several real alternatives under real constraints, log the
decision, and compare predicted vs. actual outcomes over time.

## Use Case

A logistics manager sees a predicted delay for an order. SupplyPrescript
doesn't just flag it — it runs an optimization and prescribes ranked
alternatives (e.g., pay for air freight, switch to a secondary supplier,
or accept the delay), each with real cost/speed tradeoffs. The manager
picks one; the system logs it. Later, actual outcomes can be compared
against what was predicted, closing the loop.

## Architecture

| Layer                 | Enterprise-scale equivalent | This implementation                               |
| --------------------- | --------------------------- | ------------------------------------------------- |
| Predictive Model      | XGBoost / LightGBM          | **XGBoost** (real, unchanged from brief)          |
| Prescriptive Solver   | SciPy / PuLP                | **PuLP** (real, unchanged from brief)             |
| Write-Back            | Snowflake + FastAPI         | **FastAPI + SQLite** (real API, lighter storage)  |
| Operational Dashboard | Retool / React              | **Streamlit** (matches skillset built this month) |

Scoped the same way MetricMind was: keep the technologies that are the
actual point of the project (a real predictive model, a real optimizer,
a real write-back API), simplify the infrastructure that's incidental to
the learning goal (no Snowflake account, no Retool setup) — same
reasoning, documented honestly, not hidden.

## Project Structure

```
supplyprescript/
├── data/           # Order dataset + generation script
├── model/          # XGBoost delay prediction
├── solver/         # PuLP optimization — recommend best action
├── api/            # FastAPI write-back endpoint + SQLite
├── ui/             # Streamlit operational dashboard
├── tests/          # Unit tests for model, solver, API
└── docs_and_demo/  # Architecture notes + demo script
```

## Dataset

`data/supply_orders.csv` — 250 synthetic but realistically-distributed
orders. Each row has:

- `lead_time_days`, `distance_km`, `supplier_reliability`,
  `historical_delay_rate`, `order_value` — features used to predict delay
- `delayed` — the actual outcome (0/1), ~26% delayed (realistic
  imbalance, not artificially 50/50)
- `cost_air_freight`, `cost_secondary_supplier`, `cost_delay_penalty` —
  the real cost of each of the three response options, used by the
  optimizer

This is clearly documented as **synthetic data**, generated with
realistic correlations (less reliable suppliers + higher historical
delay rates → higher delay probability), not real company data.

## Month plan

- **Week 1**: Data (done) + XGBoost delay prediction model
- **Week 2**: PuLP optimization solver — rank alternatives under constraints
- **Week 3**: FastAPI write-back endpoint + SQLite decision log + evaluation script
- **Week 4**: Streamlit dashboard, tests, demo script, final polish

## Status

- ✅ Day 1: Repo structure, realistic synthetic dataset generated and
  documented
- 🔄 Next: Train the XGBoost delay classifier
