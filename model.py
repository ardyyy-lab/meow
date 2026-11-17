"""
TalentFlow Analytics - the models behind the app (no Streamlit in this file).

1. Placement planner : Linear Programming solved with SciPy's HiGHS solver, plus
                       sensitivity analysis (shadow prices, allowable ranges, fee ranges).
                       Works for any number of job types, not just two.
2. Interview queue   : M/M/s : (FCFS / inf / inf) queueing model.

Money is in pesos everywhere in this file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# --------------------------------------------------------------------------- #
#  Data classes                                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Job:
    name: str
    fee: float                      # placement fee per hire (pesos)
    hours: float                    # recruiter hours per hire
    cost: float = 0.0               # sourcing cost per hire (pesos)
    min_hires: float = 0.0
    max_hires: Optional[float] = None
    current: Optional[float] = None  # hires per month you make today (optional, used for "profit gained")


@dataclass(frozen=True)
class Problem:
    jobs: tuple
    hours_limit: float              # recruiter hours available per month
    pool: float                     # qualified candidates available per month
    budget: Optional[float] = None  # sourcing budget (pesos); None = no budget limit
    recruiter_rate: float = 0.0     # recruiter cost per hour (pesos); 0 = leave recruiter pay out


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

TOL = 1e-7


def validate(p: Problem) -> Optional[str]:
    if not p.jobs:
        return "Add at least one job type."
    names = [j.name.strip() for j in p.jobs]
    if any(not n for n in names):
        return "Every job type needs a name."
    if len(set(n.lower() for n in names)) != len(names):
        return "Job type names must be different from each other."
    for j in p.jobs:
        if j.fee < 0 or j.cost < 0 or j.min_hires < 0:
            return f"'{j.name}': fees, costs and minimum hires cannot be negative."
        if j.hours <= 0:
            return f"'{j.name}': recruiter hours per hire must be above zero."
        if j.max_hires is not None and j.max_hires < j.min_hires:
            return f"'{j.name}': the maximum hires is below the minimum hires."
        if j.current is not None and j.current < 0:
            return f"'{j.name}': current hires cannot be negative."
    if all(j.fee <= 0 for j in p.jobs):
        return "At least one placement fee must be above zero."
    if p.hours_limit <= 0:
        return "Recruiter hours available must be above zero."
    if p.pool <= 0:
        return "The qualified candidate pool must be above zero."
    if p.budget is not None and p.budget < 0:
        return "The sourcing budget cannot be negative."
    if p.recruiter_rate < 0:
        return "The recruiter cost per hour cannot be negative."
    return None


def _rows(p: Problem):
    """Constraint rows  A x <= b  as (key, name, coefficients, rhs, unit)."""
    n = len(p.jobs)
    rows = [
        ("hours", "Recruiter hours", np.array([j.hours for j in p.jobs], float), p.hours_limit, "hours"),
        ("pool", "Qualified candidate pool", np.ones(n), p.pool, "candidates"),
    ]
    if p.budget is not None and any(j.cost > 0 for j in p.jobs):
        rows.append(("budget", "Sourcing budget", np.array([j.cost for j in p.jobs], float), p.budget, "PHP"))
    return rows


def _bounds(p: Problem):
    return [(j.min_hires, j.max_hires) for j in p.jobs]


def _lp(p: Problem):
    """Solve the LP. Returns the scipy result, or None if infeasible."""
    rows = _rows(p)
    c = -np.array([j.fee for j in p.jobs], float)           # maximise fees = minimise -fees
    A = np.vstack([r[2] for r in rows])
    b = np.array([r[3] for r in rows], float)
    res = linprog(c, A_ub=A, b_ub=b, bounds=_bounds(p), method="highs")
    return res if res.status == 0 else None


def _value(p: Problem) -> Optional[float]:
    res = _lp(p)
    return None if res is None else -res.fun


def _with_limit(p: Problem, key: str, value: float) -> Problem:
    if key == "hours":
        return replace(p, hours_limit=value)
    if key == "pool":
        return replace(p, pool=value)
    if key == "budget":
        return replace(p, budget=value)
    kind, i = key.split(":")
    i = int(i)
    jobs = list(p.jobs)
    jobs[i] = replace(jobs[i], min_hires=value) if kind == "min" else replace(jobs[i], max_hires=value)
    return replace(p, jobs=tuple(jobs))


def _with_fee(p: Problem, i: int, value: float) -> Problem:
    jobs = list(p.jobs)
    jobs[i] = replace(jobs[i], fee=value)
    return replace(p, jobs=tuple(jobs))


_OFFSETS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 100]


def _find_limit(same, r0: float, direction: int, scale: float, floor: float = 0.0):
    """Walk away from r0 until `same(r)` turns False, then bisect. None = never changes."""
    good, bad = r0, None
    for off in _OFFSETS:
        r = r0 + direction * off * scale
        if direction < 0 and r <= floor:
            r = floor
        if same(r):
            good = r
            if direction < 0 and r == floor:
                return floor
        else:
            bad = r
            break
    if bad is None:
        return None
    for _ in range(32):
        mid = (good + bad) / 2
        if same(mid):
            good = mid
        else:
            bad = mid
    return good


def _tidy(v: Optional[float]) -> Optional[float]:
    """Round range limits so solver noise (like 269,999.66) never shows."""
    if v is None:
        return None
    return float(round(v)) if abs(v) >= 1000 else round(float(v), 1)


def _fmt(n: float) -> str:
    return f"{n:,.0f}" if abs(n - round(n)) < 0.005 else f"{n:,.2f}"


# --------------------------------------------------------------------------- #
#  Placement planner                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class PlacementResult:
    ok: bool
    error: str = ""
    names: list = None
    hires: list = None
    revenue: float = 0.0
    whole_hires: Optional[list] = None
    whole_revenue: Optional[float] = None
    limits: list = None
    fee_ranges: list = None
    insights: list = None
    sourcing_cost: float = 0.0       # sum of sourcing cost per hire x hires
    recruiter_cost: float = 0.0      # recruiter hours used x cost per hour
    total_cost: float = 0.0
    net_profit: float = 0.0          # fees minus the costs above
    margin: float = 0.0              # net profit / fees
    cost_per_hire: float = 0.0       # total cost / total hires
    hours_used: float = 0.0
    current: Optional[dict] = None   # today's plan and what the best plan gains over it (None if not entered)


def _evaluate(p: Problem, hires) -> dict:
    """Fees, costs and profit of any plan (a list of hires per job type), plus any limit it breaks."""
    fees = float(sum(j.fee * h for j, h in zip(p.jobs, hires)))
    sourcing = float(sum(j.cost * h for j, h in zip(p.jobs, hires)))
    hours = float(sum(j.hours * h for j, h in zip(p.jobs, hires)))
    total = float(sum(hires))
    cost = sourcing + hours * p.recruiter_rate
    breaks = []
    if hours > p.hours_limit + 1e-6:
        breaks.append("recruiter hours")
    if total > p.pool + 1e-6:
        breaks.append("qualified candidate pool")
    if p.budget is not None and sourcing > p.budget + 1e-6:
        breaks.append("sourcing budget")
    for j, h in zip(p.jobs, hires):
        if h < j.min_hires - 1e-6:
            breaks.append(f"minimum {j.name} hires")
        if j.max_hires is not None and h > j.max_hires + 1e-6:
            breaks.append(f"maximum {j.name} hires")
    return dict(hires=[float(h) for h in hires], fees=fees, sourcing_cost=sourcing, recruiter_cost=hours * p.recruiter_rate,
                total_cost=cost, net_profit=fees - cost, total_hires=total, hours_used=hours,
                cost_per_hire=cost / total if total > 0 else 0.0, breaks=breaks)


def solve_placement(p: Problem) -> PlacementResult:
    err = validate(p)
    if err:
        return PlacementResult(False, err)
    res = _lp(p)
    if res is None:
        return PlacementResult(False, "No plan satisfies all of the limits at once. "
                                      "Check the minimum hires, the budget, and the recruiter hours.")
    x = np.where(np.abs(res.x) < 1e-9, 0.0, res.x)
    v0 = float(-res.fun)
    fee_scale = max(j.fee for j in p.jobs)
    tol_v = 1e-6 * max(1.0, abs(v0))

    # ---- every limit: usage, shadow price, allowable range ---------------------------
    rows = _rows(p)
    limits = []

    def add(key, name, kind, used, limit, shadow, unit, step):
        slack = (limit - used) if kind == "<=" else (used - limit)
        if abs(slack) < 1e-6 * max(1.0, abs(limit)):
            slack = 0.0

        def same(r):
            val = _value(_with_limit(p, key, r))
            return val is not None and abs(val - (v0 + shadow * (r - limit))) <= tol_v

        scale = max(abs(limit), 1.0)
        lo = _find_limit(same, limit, -1, scale, 0.0)
        hi = _find_limit(same, limit, +1, scale, 0.0)
        limits.append(dict(
            key=key, name=name, kind=kind, used=float(used), limit=float(limit), slack=float(slack),
            binding=slack == 0.0, shadow=float(0.0 if abs(shadow) < 1e-6 else shadow), unit=unit, step=step,
            lower=_tidy(lo), upper=_tidy(hi)))

    for k, (key, name, coef, rhs, unit) in enumerate(rows):
        step = {"hours": 10, "pool": 1, "budget": 10_000}[key]
        add(key, name, "<=", float(coef @ x), rhs, float(-res.ineqlin.marginals[k]), unit, step)
    for i, j in enumerate(p.jobs):
        if j.min_hires > 0:
            add(f"min:{i}", f"Minimum {j.name} hires", ">=", x[i], j.min_hires,
                float(-res.lower.marginals[i]), "hires", 1)
        if j.max_hires is not None:
            add(f"max:{i}", f"Maximum {j.name} hires", "<=", x[i], j.max_hires,
                float(-res.upper.marginals[i]), "hires", 1)

    # ---- how far each fee can move before the plan changes ----------------------------
    fee_ranges = []
    for i, j in enumerate(p.jobs):
        def same_fee(f, i=i):
            val = _value(_with_fee(p, i, f))
            return val is not None and abs(val - (v0 + x[i] * (f - j.fee))) <= tol_v
        scale = max(j.fee, fee_scale * 0.1, 1.0)
        lo = _find_limit(same_fee, j.fee, -1, scale, 0.0)
        hi = _find_limit(same_fee, j.fee, +1, scale, 0.0)
        fee_ranges.append(dict(name=j.name, fee=j.fee,
                               lower=0.0 if lo is None else _tidy(lo),
                               upper=_tidy(hi)))

    # ---- whole-number plan -----------------------------------------------------------
    whole, whole_rev = None, None
    try:
        A = np.vstack([r[2] for r in rows])
        b = np.array([r[3] for r in rows], float)
        lb = np.array([j.min_hires for j in p.jobs], float)
        ub = np.array([np.inf if j.max_hires is None else j.max_hires for j in p.jobs], float)
        m = milp(-np.array([j.fee for j in p.jobs], float),
                 constraints=LinearConstraint(A, -np.inf, b),
                 bounds=Bounds(lb, ub), integrality=np.ones(len(p.jobs)))
        if m.status == 0:
            whole = [int(round(v)) for v in m.x]
            whole_rev = float(-m.fun)
    except Exception:
        pass

    # ---- plain-language insights -----------------------------------------------------
    insights = []
    bottlenecks = [l["name"].lower() for l in limits if l["binding"] and abs(l["shadow"]) > 0]
    if bottlenecks:
        insights.append(f"Bottleneck{'s' if len(bottlenecks) > 1 else ''}: " + " and ".join(bottlenecks) +
                        f". Revenue cannot grow until {'these are' if len(bottlenecks) > 1 else 'this is'} increased.")
    else:
        insights.append("No single limit is holding revenue back right now.")
    adds = {"hours": "Adding 10 recruiter hours", "pool": "Adding 1 qualified candidate",
            "budget": "Adding ₱10,000 of sourcing budget"}
    for l in limits:
        if l["kind"] == "<=" and l["shadow"] > 0:
            what = adds.get(l["key"]) or f"Allowing 1 more {l['name'].split('Maximum ')[-1].replace(' hires', '')} hire"
            rng = ""
            if l["lower"] is not None or l["upper"] is not None:
                lo = "any lower value" if l["lower"] is None else _fmt(l["lower"])
                hi = "no upper limit" if l["upper"] is None else _fmt(l["upper"])
                rng = f" (valid while the limit stays between {lo} and {hi})"
            insights.append(f"{what} adds about ₱{_fmt(l['shadow'] * l['step'])} in monthly fees{rng}.")
    for l in limits:
        if not l["binding"] and l["slack"] > 0:
            amt = f"₱{_fmt(l['slack'])}" if l["unit"] == "PHP" else f"{_fmt(l['slack'])} {l['unit']}"
            if l["kind"] == "<=":
                insights.append(f"You have room to spare on {l['name'].lower()}: {amt} unused.")
            else:
                insights.append(f"The plan is {amt} above the {l['name'].lower()} of {_fmt(l['limit'])}.")

    # ---- profit view (the plan maximises fees; costs are subtracted afterwards) ---------
    hires_total = float(x.sum())
    sourcing = float(sum(j.cost * x[i] for i, j in enumerate(p.jobs)))
    hours_used = float(sum(j.hours * x[i] for i, j in enumerate(p.jobs)))
    recruiter = hours_used * p.recruiter_rate
    total_cost = sourcing + recruiter
    net = v0 - total_cost
    margin = net / v0 if v0 > 0 else 0.0
    cph = total_cost / hires_total if hires_total > 0 else 0.0

    # ---- profit gained / cost saved against today's plan (only if current hires were entered) ----
    current = None
    if any(j.current is not None for j in p.jobs):
        current = _evaluate(p, [float(j.current or 0.0) for j in p.jobs])
        current["fees_gain"] = v0 - current["fees"]
        current["profit_gain_month"] = net - current["net_profit"]
        current["profit_gain_year"] = 12 * current["profit_gain_month"]
        current["cost_per_hire_saved"] = current["cost_per_hire"] - cph if current["total_hires"] > 0 else None
        if current["profit_gain_month"] > 0.5:
            insights.append(f"Moving from today's plan to this one adds about ₱{_fmt(current['profit_gain_month'])} "
                            f"in net profit each month (₱{_fmt(current['profit_gain_year'])} a year).")
        elif current["profit_gain_month"] < -0.5:
            insights.append("Today's plan earns more net profit than this fee-maximizing plan because it spends less "
                            "on costs. Check the cost inputs.")
        else:
            insights.append("Today's plan already matches the best plan.")

    return PlacementResult(True, "", [j.name for j in p.jobs], [float(round(v, 4)) for v in x], v0,
                           whole, whole_rev, limits, fee_ranges, insights,
                           sourcing, recruiter, total_cost, net, margin, cph, hours_used, current)


# --------------------------------------------------------------------------- #
#  Bottleneck solutions: the best plan after relaxing the limits              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Expansion:
    """What it would cost, and how far you could go, to relax each limit. All amounts are per month."""
    hours_cost: float = 0.0     # extra cost of each added recruiter hour (on top of normal recruiter pay)
    hours_max: float = 0.0      # most extra recruiter hours you could add
    pool_cost: float = 0.0      # extra cost of each added qualified candidate (ads, agency fees, sourcing push)
    pool_max: float = 0.0       # most extra qualified candidates you could find
    budget_max: float = 0.0     # most extra sourcing budget you could get approved (the money itself is only
                                # counted when it is spent on hires, so adding budget has no separate price)


_FIX_NAMES = {"hours": ("Recruiter hours", "hours"), "pool": ("Qualified candidate pool", "candidates"),
              "budget": ("Sourcing budget", "PHP")}


def _expansion_lp(p: Problem, ex: Expansion, allow: set, integer: bool = False):
    """Maximise NET profit (fees - sourcing - recruiter pay - cost of the added capacity), where each limit may be
    raised by an extra amount e >= 0 up to the most you said you could add. Returns (x, extras, net, scipy result)."""
    rows = _rows(p)
    n, m = len(p.jobs), len(rows)
    caps = {"hours": ex.hours_max, "pool": ex.pool_max, "budget": ex.budget_max}
    price = {"hours": ex.hours_cost, "pool": ex.pool_cost, "budget": 0.0}
    c = np.zeros(n + m)
    for i, j in enumerate(p.jobs):
        c[i] = -(j.fee - j.cost - p.recruiter_rate * j.hours)
    A = np.zeros((m, n + m))
    for r, (key, _, coef, _, _) in enumerate(rows):
        A[r, :n] = coef
        A[r, n + r] = -1.0                       # coef . x - e <= limit   (e = capacity added)
        c[n + r] = price[key]
    b = np.array([r[3] for r in rows], float)
    lb = [j.min_hires for j in p.jobs] + [0.0] * m
    ub = [np.inf if j.max_hires is None else j.max_hires for j in p.jobs] + \
         [max(0.0, caps[r[0]]) if r[0] in allow else 0.0 for r in rows]
    if integer:
        sol = milp(c, constraints=LinearConstraint(A, -np.inf, b), bounds=Bounds(lb, ub),
                   integrality=np.array([1] * n + [0] * m))
        if sol.status != 0:
            return None
        return sol.x[:n], {r[0]: sol.x[n + k] for k, r in enumerate(rows)}, float(-sol.fun), sol
    sol = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lb, ub)), method="highs")
    if sol.status != 0:
        return None
    return sol.x[:n], {r[0]: sol.x[n + k] for k, r in enumerate(rows)}, float(-sol.fun), sol


def improve_placement(p: Problem, ex: Expansion, base: PlacementResult) -> Optional[dict]:
    """Best plan once the bottlenecks are relaxed, with a list of the fixes that pay for themselves.
    `base` is the result of solve_placement(p). Everything is per month."""
    if not base.ok:
        return None
    n = len(p.jobs)
    full = _expansion_lp(p, ex, {"hours", "pool", "budget"})
    if full is None:
        return None
    x, extras, net_model, sol = full
    x = np.where(np.abs(x) < 1e-9, 0.0, x)
    for r_ in _rows(p):                                  # report only the capacity the plan really needs
        extras[r_[0]] = min(extras[r_[0]], max(0.0, float(r_[2] @ x) - r_[3]))
    prices = {"hours": ex.hours_cost, "pool": ex.pool_cost, "budget": 0.0}
    caps = {"hours": ex.hours_max, "pool": ex.pool_max, "budget": ex.budget_max}
    ev = _evaluate(p, list(x))
    exp_cost = float(sum(prices[k] * extras[k] for k in extras))
    net = ev["fees"] - ev["total_cost"] - exp_cost
    gain = net - base.net_profit

    fixes = []
    for key, add in extras.items():
        if add < 1e-6:
            continue
        alone = _expansion_lp(p, ex, {key})
        gain_alone = (alone[2] - base.net_profit) if alone else 0.0
        name, unit = _FIX_NAMES[key]
        fixes.append(dict(key=key, name=name, unit=unit, add=float(add), cost_each=prices[key],
                          cost_total=prices[key] * float(add), max_add=float(caps[key]),
                          at_max=bool(add >= caps[key] - 1e-6), gain_alone=float(max(0.0, gain_alone))))
    fixes.sort(key=lambda f: -f["gain_alone"])

    # limits that still hold the improved plan back
    still = []
    upper = getattr(sol, "upper", None)
    keys = [r[0] for r in _rows(p)]
    for f in fixes:
        if f["at_max"]:
            worth = float(-upper.marginals[n + keys.index(f["key"])]) if upper is not None else 0.0
            worth = max(0.0, worth)
            note = f"The plan already uses all {_fmt(f['max_add'])} extra you said you could add."
            if worth > 0.5:
                note += f" Each further unit would add about ₱{_fmt(worth)} in net profit."
            still.append(dict(kind="capacity", name=f["name"], unit=f["unit"], worth_per_unit=worth, note=note))
    if upper is not None:
        for i, j in enumerate(p.jobs):
            if j.max_hires is not None and x[i] >= j.max_hires - 1e-6 and abs(upper.marginals[i]) > 1e-6:
                still.append(dict(kind="cap", name=f"Maximum {j.name} hires", unit="hires",
                                  worth_per_unit=float(-upper.marginals[i]),
                                  note=f"Each extra {j.name.lower()} hire you are allowed would add about "
                                       f"₱{_fmt(-upper.marginals[i])} in net profit."))

    # whole-number version of the improved plan
    whole, whole_net = None, None
    try:
        w = _expansion_lp(p, ex, {"hours", "pool", "budget"}, integer=True)
        if w is not None:
            whole = [int(round(v)) for v in w[0]]
            whole_net = float(w[2])
    except Exception:
        pass

    new_limits = {}
    for key, add in extras.items():
        old = {"hours": p.hours_limit, "pool": p.pool, "budget": p.budget}[key]
        new_limits[key] = dict(old=old, new=old + float(add))

    insights = []
    if not fixes:
        insights.append("No extra capacity pays for itself at the costs you entered, so the best plan stays as it is. "
                        "Lower the cost of adding capacity or check the numbers.")
    else:
        parts = [f"{_fmt(f['add'])} {f['unit'] if f['unit'] != 'PHP' else 'pesos of budget'}" for f in fixes]
        what = " and ".join([", ".join(parts[:-1]), parts[-1]] if len(parts) > 1 else parts)
        insights.append(f"Adding {what} lifts monthly net profit by about ₱{_fmt(gain)} (₱{_fmt(12 * gain)} a year), "
                        f"even after paying ₱{_fmt(exp_cost)} for the extra capacity.")
    for s_ in still:
        insights.append(f"Still limited by {s_['name'].lower()}. {s_['note']}")
    return dict(ok=True, names=base.names, hires=[float(round(v, 4)) for v in x], whole_hires=whole, whole_net=whole_net,
                fees=ev["fees"], sourcing_cost=ev["sourcing_cost"], recruiter_cost=ev["recruiter_cost"],
                expansion_cost=exp_cost, total_cost=ev["total_cost"] + exp_cost, net_profit=net,
                margin=net / ev["fees"] if ev["fees"] > 0 else 0.0, total_hires=ev["total_hires"],
                cost_per_hire=(ev["total_cost"] + exp_cost) / ev["total_hires"] if ev["total_hires"] > 0 else 0.0,
                base_net=base.net_profit, gain_month=gain, gain_year=12 * gain, fixes=fixes, still=still,
                new_limits=new_limits, insights=insights)


def region_2d(p: Problem):
    """Corner points and boundary lines for the graphical method (only for two job types)."""
    if len(p.jobs) != 2:
        return None
    hp = []                                   # half-planes  a*x + b*y <= c
    lines = []
    for key, name, coef, rhs, unit in _rows(p):
        hp.append((coef[0], coef[1], rhs))
        lines.append((coef[0], coef[1], rhs, name))
    for i, j in enumerate(p.jobs):
        e = [0.0, 0.0]
        e[i] = 1.0
        if j.min_hires > 0:
            hp.append((-e[0], -e[1], -j.min_hires))
            lines.append((e[0], e[1], j.min_hires, f"Minimum {j.name} hires"))
        else:
            hp.append((-e[0], -e[1], 0.0))
        if j.max_hires is not None:
            hp.append((e[0], e[1], j.max_hires))
            lines.append((e[0], e[1], j.max_hires, f"Maximum {j.name} hires"))
    pts = []
    for a in range(len(hp)):
        for b in range(a + 1, len(hp)):
            d = hp[a][0] * hp[b][1] - hp[b][0] * hp[a][1]
            if abs(d) < 1e-12:
                continue
            x = (hp[a][2] * hp[b][1] - hp[b][2] * hp[a][1]) / d
            y = (hp[a][0] * hp[b][2] - hp[b][0] * hp[a][2]) / d
            if all(h[0] * x + h[1] * y <= h[2] + 1e-9 * max(1.0, abs(h[2])) for h in hp):
                if not any(abs(x - q[0]) < 1e-6 and abs(y - q[1]) < 1e-6 for q in pts):
                    pts.append((float(x) + 0.0, float(y) + 0.0))
    if not pts:
        return None
    cx = sum(q[0] for q in pts) / len(pts)
    cy = sum(q[1] for q in pts) / len(pts)
    pts.sort(key=lambda q: math.atan2(q[1] - cy, q[0] - cx))
    f1, f2 = p.jobs[0].fee, p.jobs[1].fee
    return dict(vertices=[(q[0], q[1], f1 * q[0] + f2 * q[1]) for q in pts], lines=lines, fees=(f1, f2))


# --------------------------------------------------------------------------- #
#  Interview queue  (M/M/s : FCFS / inf / inf)                                #
# --------------------------------------------------------------------------- #


def mms(lam: float, mu: float, s: int) -> dict:
    """Standard M/M/s results. lam and mu are per hour; times are returned in minutes."""
    a = lam / mu
    rho = a / s
    if rho >= 1:
        return dict(servers=s, stable=False, utilization=rho)
    term, total = 1.0, 0.0
    for n in range(s):
        if n > 0:
            term *= a / n
        total += term
    last = term * a / s / (1 - rho)                  # a^s / (s! (1 - rho))
    p0 = 1 / (total + last)
    p_wait = last * p0                                # Erlang C
    lq = p_wait * rho / (1 - rho)
    wq = lq / lam
    w = wq + 1 / mu
    return dict(servers=s, stable=True, utilization=rho, p_wait=p_wait, p_empty=p0,
                lq=lq, wq_min=wq * 60, w_min=w * 60, l=lam * w)


def _meets(lam: float, mu: float, s: int, target_wait_min: float, max_util: float) -> bool:
    r = mms(lam, mu, s)
    return bool(r["stable"] and r["wq_min"] <= target_wait_min and r["utilization"] <= max_util)


def queue_fixes(lam: float, mu: float, s: int, target_wait_min: float, max_util: float) -> dict:
    """Other ways to hit the wait targets with a team of s interviewers, besides hiring more:
    interview faster (needed mu) or smooth out arrivals (largest lam the team can take)."""
    out = dict(servers=s, meets_now=_meets(lam, mu, s, target_wait_min, max_util), needed_mu=None, speedup_pct=None,
               max_arrival=None, arrival_cut_pct=None)
    if out["meets_now"]:
        return out
    hi = mu
    for _ in range(40):                                     # find a faster pace that works, then bisect down to it
        hi *= 1.5
        if _meets(lam, hi, s, target_wait_min, max_util):
            lo = hi / 1.5
            for _ in range(50):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if _meets(lam, mid, s, target_wait_min, max_util) else (mid, hi)
            out["needed_mu"], out["speedup_pct"] = hi, (hi / mu - 1) * 100
            break
        if hi > mu * 50:
            break
    lo, hi = 0.0, s * mu * 0.999999                         # largest arrival rate the team handles within targets
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _meets(mid, mu, s, target_wait_min, max_util) else (lo, mid)
    out["max_arrival"], out["arrival_cut_pct"] = lo, max(0.0, (1 - lo / lam) * 100)
    return out


def solve_queue(lam: float, mu: float, max_servers: int = 6, target_wait_min: float = 1.0,
                max_util: float = 0.85, hours_per_month: Optional[float] = None,
                cost_per_hour: Optional[float] = None, drop_per_min: Optional[float] = None,
                lost_value: Optional[float] = None, current_servers: int = 1) -> dict:
    """current_servers is the team you have today; cost savings are measured against it (default: one interviewer)."""
    if lam <= 0 or mu <= 0:
        return dict(ok=False, error="Arrival rate and interview rate must be above zero.")
    current_servers = max(1, min(int(current_servers), 20))
    max_servers = max(1, min(max(int(max_servers), current_servers), 20))
    rows = [mms(lam, mu, s) for s in range(1, max_servers + 1)]
    rec = None
    for r in rows:
        r["meets"] = bool(r["stable"] and r["wq_min"] <= target_wait_min and r["utilization"] <= max_util)
        if r["meets"] and rec is None:
            rec = r["servers"]
    min_stable = int(math.floor(lam / mu)) + 1
    insights = []
    one = rows[0]
    cur = rows[current_servers - 1]
    if one["stable"] and rec:
        if rec == 1:
            insights.append(f"One interviewer already meets your targets: average wait {one['wq_min']:.1f} minutes "
                            f"at {one['utilization']:.0%} utilization.")
        elif current_servers > 1 and cur["stable"] and rec == current_servers:
            insights.append(f"Your current {current_servers} interviewers already meet your targets: average wait "
                            f"{cur['wq_min']:.1f} minutes, each busy {cur['utilization']:.0%} of the time.")
        else:
            r = rows[rec - 1]
            was = cur if (current_servers > 1 and cur["stable"]) else one
            who = "one interviewer" if was is one else f"your current {current_servers}"
            insights.append(f"Go with {rec} interviewers. Average wait drops from {was['wq_min']:.1f} ({who}) to "
                            f"{r['wq_min']:.1f} minutes, and each interviewer is busy {r['utilization']:.0%} of the time.")
    elif not one["stable"]:
        insights.append("One interviewer cannot keep up: candidates arrive faster than they can be interviewed, "
                        f"so the line grows without end. You need at least {min_stable} interviewers for a stable queue.")
        if rec:
            insights.append(f"Go with {rec} interviewers to meet your wait and utilization targets.")
    if not rec:
        insights.append(f"No option up to {max_servers} interviewers meets both targets "
                        f"(wait of {target_wait_min:g} minutes or less and utilization of {max_util:.0%} or less). "
                        "Compare more interviewers, or relax a target.")
    cost = None
    if None not in (hours_per_month, cost_per_hour, drop_per_min, lost_value):
        # Monthly cost of each team size = interviewer pay + the fees lost when waiting candidates drop out.
        # drop_per_min is the share of candidates lost for each minute of average wait (0.005 = 0.5%).
        for r in rows:
            if r["stable"]:
                r["candidates"] = lam * hours_per_month
                r["staff_cost"] = r["servers"] * hours_per_month * cost_per_hour
                r["lost"] = r["candidates"] * min(1.0, drop_per_min * r["wq_min"])
                r["lost_value"] = r["lost"] * lost_value
                r["total_cost"] = r["staff_cost"] + r["lost_value"]
        stable = [r for r in rows if r["stable"]]
        best = min(stable, key=lambda r: r["total_cost"]) if stable else None
        base = rows[current_servers - 1]                         # today's team
        base_ok = base["stable"]
        saving = (base["total_cost"] - best["total_cost"]) if (best and base_ok) else None
        recr = rows[rec - 1] if rec else None                   # cheapest team that meets the wait/busy targets
        cost = dict(best=best["servers"] if best else None, best_total=best["total_cost"] if best else None,
                    saving_month=saving, saving_year=None if saving is None else saving * 12,
                    current=current_servers, current_total=base["total_cost"] if base_ok else None,
                    pay_change=(best["staff_cost"] - base["staff_cost"]) if (best and base_ok) else None,
                    lost_saved=(base["lost_value"] - best["lost_value"]) if (best and base_ok) else None,
                    lost_candidates_saved=(base["lost"] - best["lost"]) if (best and base_ok) else None,
                    rec_total=recr["total_cost"] if recr else None,
                    rec_saving_month=(base["total_cost"] - recr["total_cost"]) if (recr and base_ok) else None)
        if best:
            ref = "one interviewer" if current_servers == 1 else f"your current {current_servers} interviewers"
            if best["servers"] == current_servers:
                insights.append(f"On cost, {ref if current_servers > 1 else 'one interviewer'} "
                                f"{'is' if current_servers == 1 else 'are'} already the cheapest option at "
                                f"₱{best['total_cost']:,.0f} a month.")
            else:
                tail = "" if saving is None else (f" That saves ₱{saving:,.0f} a month compared with {ref} "
                                                  f"(₱{saving * 12:,.0f} a year).")
                insights.append(f"On cost, {best['servers']} interviewers is cheapest at ₱{best['total_cost']:,.0f} a month "
                                f"(interviewer pay plus the value of candidates lost to waiting).{tail}")
            if rec and rec != best["servers"] and recr is not None:
                gap = recr["total_cost"] - best["total_cost"]
                if gap > 0.5:
                    insights.append(f"On cost, meeting your wait targets with {rec} interviewers costs ₱{gap:,.0f} a month "
                                    f"more than the cheapest team ({best['servers']}). That is the price of the shorter wait.")

    # ---- ways to fix the wait for the team you have today -------------------------------------------
    fixes = queue_fixes(lam, mu, current_servers, target_wait_min, max_util)
    if fixes["meets_now"] is False and fixes["speedup_pct"] is not None:
        insights.append(f"Or keep {current_servers} interviewer{'s' if current_servers > 1 else ''} and make each interview "
                        f"about {fixes['speedup_pct']:.0f}% faster (to {fixes['needed_mu']:.1f} per hour).")
    return dict(ok=True, rows=rows, recommended=rec, min_servers=min_stable, offered_load=lam / mu,
                insights=insights, cost=cost, fixes=fixes, current_servers=current_servers)
