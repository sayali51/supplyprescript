"""
Owner: Person 2 (solver)

Given an order predicted to be delayed, this recommends the best
alternative action (air freight, secondary supplier, or accept the
delay) using linear optimization (PuLP), subject to a real constraint:
a maximum budget the business is willing to spend to avoid the delay.

This mirrors the brief's "Prescriptive Solver" — instead of just
predicting a delay, it prescribes what to do about it, ranked by cost,
respecting hard constraints rather than just picking the cheapest option
blindly.

Run with: python solver/recommend.py
"""

import pulp


def recommend_action(
    cost_air_freight: float,
    cost_secondary_supplier: float,
    cost_delay_penalty: float,
    max_budget: float,
) -> dict:
    """
    Solves a simple linear program: choose exactly one action, minimize
    cost, subject to not exceeding max_budget. Returns the chosen action
    plus a full ranked breakdown of all options (so the UI can show
    trade-offs, not just the winner).

    This is intentionally a simple 0/1 selection problem — PuLP is used
    for real here (not simulated), even though the problem itself is
    small enough that a plain min() would give the same answer. Using
    PuLP means the constraint logic (budget, and any future constraints
    like delivery deadlines) is expressed declaratively and can grow
    without rewriting the decision logic by hand.
    """
    options = {
        "air_freight": cost_air_freight,
        "secondary_supplier": cost_secondary_supplier,
        "accept_delay": cost_delay_penalty,
    }

    prob = pulp.LpProblem("choose_best_action", pulp.LpMinimize)

    # One binary decision variable per option: 1 if chosen, 0 otherwise
    choice_vars = {
        name: pulp.LpVariable(f"choose_{name}", cat="Binary")
        for name in options
    }

    # Objective: minimize the cost of whichever option is chosen
    prob += pulp.lpSum(choice_vars[name] * cost for name, cost in options.items())

    # Constraint: exactly one option must be chosen
    prob += pulp.lpSum(choice_vars.values()) == 1

    # Constraint: the chosen option's cost cannot exceed the budget,
    # UNLESS no option fits, in which case we still need a valid answer —
    # handled by checking feasibility below before solving strictly.
    affordable = {name: cost for name, cost in options.items() if cost <= max_budget}

    if affordable:
        # Restrict the choice to only affordable options
        for name in options:
            if name not in affordable:
                prob += choice_vars[name] == 0
    # else: no option fits budget — solver will still pick the cheapest
    # overall, and the caller is told budget was exceeded.

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    chosen = None
    for name, var in choice_vars.items():
        if var.value() == 1:
            chosen = name
            break

    ranked = sorted(options.items(), key=lambda x: x[1])

    return {
        "recommended_action": chosen,
        "recommended_cost": options[chosen],
        "within_budget": options[chosen] <= max_budget,
        "all_options_ranked": [
            {"action": name, "cost": cost} for name, cost in ranked
        ],
    }


if __name__ == "__main__":
    # Example matching the brief's own use case: microchip delay,
    # air freight vs. secondary supplier vs. accepting the delay
    result = recommend_action(
        cost_air_freight=15000,
        cost_secondary_supplier=15000 * 1.10 * 0.7,  # illustrative
        cost_delay_penalty=4000,
        max_budget=16000,
    )
    print("Recommendation:")
    for k, v in result.items():
        print(f"  {k}: {v}")