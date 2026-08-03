"""
Owner: Person 1 (data)

Generates a synthetic but realistically-correlated supply chain dataset.
This is NOT real company data — documented clearly for the demo/report.

Delay probability is driven by a weighted combination of supplier
reliability, historical delay rate, distance, and lead time, plus random
noise, so the resulting labels aren't purely random but also aren't a
trivial linear giveaway for the model to learn.

Run with: python data/generate_dataset.py
"""

import random
import csv
import os

random.seed(42)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "supply_orders.csv")
N_ORDERS = 250


def generate_orders(n: int = N_ORDERS) -> list[list]:
    rows = []
    for i in range(1, n + 1):
        lead_time = round(random.uniform(2, 30), 1)
        distance = round(random.uniform(50, 5000), 1)
        supplier_reliability = round(random.uniform(0.5, 0.99), 2)
        historical_delay_rate = round(random.uniform(0.0, 0.4), 2)
        order_value = round(random.uniform(500, 50000), 2)

        delay_score = (
            (1 - supplier_reliability) * 0.4
            + historical_delay_rate * 0.35
            + (distance / 5000) * 0.15
            + (1 / max(lead_time, 1)) * 0.1
        )
        delayed = 1 if delay_score + random.uniform(-0.15, 0.15) > 0.35 else 0

        cost_air_freight = round(
            order_value * random.uniform(0.15, 0.35) + distance * random.uniform(1, 3), 2
        )
        cost_secondary_supplier = round(order_value * random.uniform(0.05, 0.15), 2)
        cost_delay_penalty = round(order_value * random.uniform(0.02, 0.08), 2)

        rows.append([
            f"ORD-{1000 + i}", lead_time, distance, supplier_reliability,
            historical_delay_rate, order_value, delayed,
            cost_air_freight, cost_secondary_supplier, cost_delay_penalty,
        ])
    return rows


def write_csv(rows: list[list], path: str = OUTPUT_PATH) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "order_id", "lead_time_days", "distance_km", "supplier_reliability",
            "historical_delay_rate", "order_value", "delayed",
            "cost_air_freight", "cost_secondary_supplier", "cost_delay_penalty",
        ])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    write_csv(generate_orders())
