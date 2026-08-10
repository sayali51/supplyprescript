"""
Owner: Person 5 (tests)

Tests for the optimization solver in solver/recommend.py
Run with: pytest tests/test_solver.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.recommend import recommend_action


def test_picks_cheapest_option_within_budget():
    result = recommend_action(
        cost_air_freight=15000,
        cost_secondary_supplier=8000,
        cost_delay_penalty=4000,
        max_budget=20000,
    )
    assert result["recommended_action"] == "accept_delay"
    assert result["recommended_cost"] == 4000
    assert result["within_budget"] is True


def test_respects_budget_constraint():
    """If the cheapest option doesn't fit, but a more expensive one that
    still respects the budget exists, the solver should still only
    consider affordable options."""
    result = recommend_action(
        cost_air_freight=15000,
        cost_secondary_supplier=8000,
        cost_delay_penalty=4000,
        max_budget=10000,
    )
    # Only secondary_supplier (8000) and accept_delay (4000) fit under 10000
    assert result["recommended_action"] in ("secondary_supplier", "accept_delay")
    assert result["recommended_cost"] <= 10000
    assert result["within_budget"] is True


def test_all_options_ranked_is_sorted_by_cost():
    result = recommend_action(
        cost_air_freight=15000,
        cost_secondary_supplier=8000,
        cost_delay_penalty=4000,
        max_budget=20000,
    )
    costs = [opt["cost"] for opt in result["all_options_ranked"]]
    assert costs == sorted(costs)
    assert len(result["all_options_ranked"]) == 3


def test_handles_budget_lower_than_every_option():
    """If no option fits the budget, the solver should still return a
    valid recommendation (the overall cheapest) and flag it as over budget."""
    result = recommend_action(
        cost_air_freight=15000,
        cost_secondary_supplier=8000,
        cost_delay_penalty=4000,
        max_budget=1000,
    )
    assert result["recommended_action"] is not None
    assert result["recommended_cost"] == 4000  # still the cheapest overall
    assert result["within_budget"] is False