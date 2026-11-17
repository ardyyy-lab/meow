"""Checks the Python model against the results in the Excel files. Run:  python test_model.py"""
from model import Expansion, Job, Problem, improve_placement, solve_placement, solve_queue


def test_prelim():
    p = Problem((Job("Technical", 60000, 20), Job("Support", 40000, 10)), 1000, 70)
    r = solve_placement(p)
    assert r.ok and r.hires == [30.0, 40.0] and round(r.revenue) == 3_400_000
    hours = next(l for l in r.limits if l["key"] == "hours")
    pool = next(l for l in r.limits if l["key"] == "pool")
    assert round(hours["shadow"] * 10) == 20000 and round(pool["shadow"]) == 20000


def test_midterm():
    p = Problem((Job("Technical", 60000, 20, 5000, 0, 45), Job("Support", 40000, 10, 3000, 10, None)),
                1000, 70, 300000)
    r = solve_placement(p)
    assert r.hires == [30.0, 40.0] and round(r.revenue) == 3_400_000
    by = {l["key"]: l for l in r.limits}
    assert (by["hours"]["lower"], by["hours"]["upper"]) == (700.0, 1150.0)
    assert (by["pool"]["lower"], by["pool"]["upper"]) == (55.0, 100.0)
    assert by["budget"]["slack"] == 30000 and not by["budget"]["binding"]
    fees = {f["name"]: (f["lower"], f["upper"]) for f in r.fee_ranges}
    assert fees["Technical"] == (40000.0, 80000.0) and fees["Support"] == (30000.0, 60000.0)


def test_queue():
    q = solve_queue(20, 25, 4)
    one, two = q["rows"][0], q["rows"][1]
    assert round(one["wq_min"], 1) == 9.6 and round(one["lq"], 1) == 3.2
    assert round(two["wq_min"], 1) == 0.5 and round(two["lq"], 2) == 0.15
    assert q["recommended"] == 2


def test_profit_and_cost():
    p = Problem((Job("Technical", 60000, 20, 5000, 0, 45), Job("Support", 40000, 10, 3000, 10, None)),
                1000, 70, 300000)
    r = solve_placement(p)
    assert round(r.sourcing_cost) == 270000 and round(r.net_profit) == 3_130_000
    assert round(r.cost_per_hire, 2) == round(270000 / 70, 2)
    q = solve_queue(20, 25, 4, 1, 0.85, 160, 150, 0.005, 1000)      # placeholder cost assumptions
    assert q["cost"]["best"] == 2 and round(q["cost"]["saving_month"]) == 122286


def test_profit_gained_vs_today():
    # Prelim case, with 20 Technical and 50 Support hires made today (fees 3,200,000 vs 3,400,000 in the best plan)
    p = Problem((Job("Technical", 60000, 20, 0, 0, None, 20), Job("Support", 40000, 10, 0, 0, None, 50)), 1000, 70)
    r = solve_placement(p)
    assert r.hires == [30.0, 40.0]
    assert round(r.current["net_profit"]) == 3_200_000 and round(r.current["profit_gain_month"]) == 200_000
    assert round(r.current["profit_gain_year"]) == 2_400_000 and not r.current["breaks"]
    # Midterm case: sourcing costs make today's cost per hire lower than the best plan's
    p = Problem((Job("Technical", 60000, 20, 5000, 0, 45, 20), Job("Support", 40000, 10, 3000, 10, None, 40)),
                1000, 70, 300000)
    r = solve_placement(p)
    assert round(r.current["net_profit"]) == 2_580_000 and round(r.current["profit_gain_month"]) == 550_000
    assert round(r.current["cost_per_hire_saved"], 2) == round(220000 / 60 - 270000 / 70, 2)
    # No current hires entered -> no comparison
    assert solve_placement(Problem((Job("A", 100, 1), Job("B", 50, 1)), 10, 10)).current is None


def test_bottleneck_solutions():
    p = Problem((Job("Technical", 60000, 20), Job("Support", 40000, 10)), 1000, 70)
    base = solve_placement(p)
    imp = improve_placement(p, Expansion(150, 300, 2000, 20), base)      # extra hours cost 150, extra candidates 2,000
    assert imp["hires"] == [40.0, 50.0] and round(imp["net_profit"]) == 4_315_000
    assert round(imp["gain_month"]) == 915_000 and round(imp["expansion_cost"]) == 85_000
    by = {f["key"]: f for f in imp["fixes"]}
    assert round(by["hours"]["add"]) == 300 and round(by["pool"]["add"]) == 20
    assert round(by["hours"]["gain_alone"]) == 555_000 and round(by["pool"]["gain_alone"]) == 360_000
    assert imp["whole_hires"] == [40, 50]
    # An expensive fix that does not pay for itself is not recommended
    none = improve_placement(p, Expansion(5000, 300, 30000, 20), base)
    assert none["fixes"] == [] and round(none["gain_month"]) == 0 and none["hires"] == [30.0, 40.0]
    # Nothing allowed to grow -> same plan, no gain
    zero = improve_placement(p, Expansion(), base)
    assert zero["fixes"] == [] and round(zero["gain_month"]) == 0


def test_bottleneck_solutions_extra_limits():
    p = Problem((Job("Technical", 60000, 20, 5000, 0, 35), Job("Support", 40000, 10, 3000, 10, None)), 1000, 70, 300000)
    base = solve_placement(p)
    imp = improve_placement(p, Expansion(150, 300, 2000, 20, 90000), base)
    assert imp["gain_month"] > 0 and imp["net_profit"] > base.net_profit
    assert sum(imp["hires"]) <= 70 + 20 + 1e-6
    for f in imp["fixes"]:                                   # only the capacity the plan needs, never the full allowance
        assert f["add"] <= f["max_add"] + 1e-6
    # The Technical cap of 35 is what holds the improved plan back next
    assert any(s["kind"] == "cap" and "Technical" in s["name"] for s in imp["still"])


def test_queue_cost_saved_and_fixes():
    q = solve_queue(20, 25, 4, 1, 0.85, 160, 150, 0.005, 1000, current_servers=1)
    assert q["cost"]["best"] == 2 and round(q["cost"]["saving_month"]) == 122286
    assert round(q["cost"]["pay_change"]) == 24000 and round(q["cost"]["lost_saved"]) == 146286
    assert q["fixes"]["meets_now"] is False and round(q["fixes"]["speedup_pct"]) == 84
    assert round(q["fixes"]["max_arrival"], 2) == 7.35
    q2 = solve_queue(20, 25, 4, 1, 0.85, 160, 150, 0.005, 1000, current_servers=2)   # already on the cheapest team
    assert round(q2["cost"]["saving_month"]) == 0 and q2["fixes"]["meets_now"] is True


if __name__ == "__main__":
    test_prelim(); test_midterm(); test_queue(); test_profit_and_cost()
    test_profit_gained_vs_today(); test_bottleneck_solutions(); test_bottleneck_solutions_extra_limits()
    test_queue_cost_saved_and_fixes()
    print("All checks passed: the Python model matches your Excel results.")
