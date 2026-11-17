"""
TalentFlow Analytics - Streamlit app
Prescriptive Analytics | Recruitment & staffing

Run it with:   streamlit run app.py
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from model import Expansion, Job, Problem, improve_placement, region_2d, solve_placement, solve_queue

st.set_page_config(page_title="TalentFlow Analytics", page_icon=":material/insights:", layout="wide")

TEAL = "#0e7a8c"

# --------------------------------------------------------------------------- #
#  Look and feel                                                              #
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    .tf-banner{background:linear-gradient(120deg,#0f2d3a 0%,#164a5a 100%);color:#fff;
        padding:26px 32px;border-radius:14px;margin-bottom:10px}
    .tf-banner h1{color:#fff;font-size:2.1rem;margin:0 0 4px 0;padding:0;line-height:1.15}
    .tf-banner p{color:#b8cad6;margin:0;font-size:1.02rem}
    div[data-testid="stMetric"]{background:rgba(14,122,140,.08);border:1px solid rgba(14,122,140,.25);
        border-left:4px solid #0e7a8c;border-radius:10px;padding:14px 16px}
    button[data-baseweb="tab"]{font-weight:600}
    </style>
    <div class="tf-banner">
      <h1>TalentFlow Analytics</h1>
      <p>Model monthly revenue and recruiter capacity</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### :material/insights: TalentFlow Analytics")
    st.caption("Prescriptive analytics for recruitment and staffing firms")
    st.divider()
    st.markdown(
        ":material/groups: **Placement planner**  \nWhich jobs should your recruiters fill each month?\n\n"
        ":material/schedule: **Interview queue**  \nHow many interviewers keep candidates from waiting?\n\n"
        ":material/menu_book: **About**  \nHow the models work and what they assume."
    )
    st.divider()
    st.caption("Results are estimates based on the numbers you enter. Check them against your own hiring records "
               "before you act on them.")


# --------------------------------------------------------------------------- #
#  Small helpers                                                              #
# --------------------------------------------------------------------------- #
def peso(x: float) -> str:
    return f"₱{x:,.0f}"


def num(x: float) -> str:
    return f"{x:,.0f}" if abs(x - round(x)) < 0.005 else f"{x:,.2f}"


def amount(v: float, unit: str) -> str:
    return peso(v) if unit == "PHP" else f"{num(v)} {unit}"


def signed_peso(v: float) -> str:
    if abs(v) < 0.5:
        return "No change"
    return ("+" if v > 0 else "−") + peso(abs(v))


def clip_line(a, b, c, X, Y):
    """Where the line a*x + b*y = c crosses the box [0, X] x [0, Y]."""
    pts = []

    def add(x, y):
        if -1e-9 <= x <= X + 1e-9 and -1e-9 <= y <= Y + 1e-9:
            if not any(abs(x - q[0]) < 1e-6 and abs(y - q[1]) < 1e-6 for q in pts):
                pts.append((x, y))

    if abs(b) > 1e-12:
        add(0, c / b)
        add(X, (c - a * X) / b)
    if abs(a) > 1e-12:
        add(c / a, 0)
        add((c - b * Y) / a, Y)
    return pts[:2] if len(pts) >= 2 else None


# --------------------------------------------------------------------------- #
#  Examples that fill the form                                                #
# --------------------------------------------------------------------------- #
COLS = ["Job type", "Fee per hire (₱)", "Recruiter hours per hire", "Sourcing cost per hire (₱)",
        "Minimum hires", "Maximum hires", "Current hires (optional)"]

PRESETS = {
    "Basic case": dict(
        jobs=[["Technical", 60000, 20, None, None, None, 20],
              ["Support", 40000, 10, None, None, None, 50]],
        hours=1000, pool=70, budget=None),
    "With extra limits": dict(
        jobs=[["Technical", 60000, 20, 5000, None, 45, 20],
              ["Support", 40000, 10, 3000, 10, None, 40]],
        hours=1000, pool=70, budget=300000),
    "Three job types (new example)": dict(
        jobs=[["Technical", 60000, 20, 5000, None, 45, 20],
              ["Support", 40000, 10, 3000, 10, None, 30],
              ["Management", 90000, 35, 9000, None, 12, 8]],
        hours=1500, pool=90, budget=450000),
}


def clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def table_to_jobs(df: pd.DataFrame):
    """Turn the editable table into a tuple of plain tuples (hashable, so results can be cached)."""
    jobs = []
    for _, r in df.iterrows():
        name = clean(r.get(COLS[0]))
        if name is None or str(name).strip() == "":
            continue
        mx = clean(r.get(COLS[5]))
        cur = clean(r.get(COLS[6]))
        jobs.append((str(name).strip(), float(clean(r.get(COLS[1])) or 0), float(clean(r.get(COLS[2])) or 0),
                     float(clean(r.get(COLS[3])) or 0), float(clean(r.get(COLS[4])) or 0),
                     None if mx is None else float(mx), None if cur is None else float(cur)))
    return tuple(jobs)


@st.cache_data(show_spinner="Solving the model...")
def run_placement(jobs_t, hours, pool, budget, rate, fix):
    """fix = (hours cost, hours max, candidate cost, candidate max, budget max) or None to skip the bottleneck solutions."""
    p = Problem(tuple(Job(*j) for j in jobs_t), hours, pool, budget, rate)
    res = solve_placement(p)
    imp = improve_placement(p, Expansion(*fix), res) if (res.ok and fix is not None) else None
    return res, (region_2d(p) if res.ok else None), imp


@st.cache_data(show_spinner=False)
def run_queue(lam, mu, smax, target, util, costs, current):
    """costs = (hours per month, cost per interviewer hour, % lost per minute of wait, value per lost candidate) or None."""
    if costs is None:
        return solve_queue(lam, mu, smax, target, util / 100, current_servers=current)
    hrs, rate, drop_pct, value = costs
    return solve_queue(lam, mu, smax, target, util / 100, hrs, rate, drop_pct / 100, value, current_servers=current)


# --------------------------------------------------------------------------- #
#  Placement results                                                          #
# --------------------------------------------------------------------------- #
def region_figure(res, region):
    verts = region["vertices"]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    X = max(max(xs) * 1.2, 1.0)
    Y = max(max(ys) * 1.2, 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=110)
    ax.fill(xs, ys, color=TEAL, alpha=0.18, label="Feasible region", zorder=1)
    ax.plot(xs + [xs[0]], ys + [ys[0]], color=TEAL, lw=1.4, zorder=2)
    palette = ["#0e5a6b", "#c4510c", "#7b4fa3", "#2f7d5b", "#8a6d00", "#b3261e"]
    for k, (a, b, c, name) in enumerate(region["lines"]):
        seg = clip_line(a, b, c, X, Y)
        if seg:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color=palette[k % len(palette)],
                    lw=2.2, label=name, zorder=3)
    best_x, best_y = res.hires[0], res.hires[1]
    # best-fee (iso-profit) line through the optimum
    seg = clip_line(region["fees"][0], region["fees"][1], res.revenue, X, Y)
    if seg:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color="#333", lw=1.3, ls="--",
                label="Best-fee line", zorder=3)
    ax.scatter(xs, ys, s=28, color="white", edgecolor="#333", zorder=4)
    ax.scatter([best_x], [best_y], s=130, color="#f2a900", edgecolor="#333", linewidth=1.6, zorder=5, label="Best plan")
    ax.annotate(f"Best: {num(best_x)}, {num(best_y)}", (best_x, best_y), textcoords="offset points",
                xytext=(10, 10), fontsize=10, fontweight="bold", color="#222")
    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_xlabel(f"{res.names[0]} hires (x)")
    ax.set_ylabel(f"{res.names[1]} hires (y)")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
    fig.tight_layout()
    return fig


FIX_ADVICE = {
    "hours": "Add recruiter capacity: hire or contract a recruiter, approve overtime, or cut the hours per hire "
             "(screening templates, shared interview slots).",
    "pool": "Widen sourcing: more job-board postings, employee referrals, or an agency partner to bring in "
            "qualified candidates.",
    "budget": "Ask for a bigger sourcing budget: every peso is spent on hires that earn more in fees than they cost.",
}


def render_fixes(res, imp, jobs_t):
    """Solutions for the bottlenecks: what to add, what it costs, and the best plan once it is added."""
    st.subheader(":material/build_circle: Solutions for the bottlenecks")
    if imp is None:
        st.caption("Turn on \"Find the best plan after fixing the bottlenecks\" in the form above to see which limits "
                   "to raise, what it costs and what you gain.")
        return
    st.caption("The planner raised each limit only where it pays for itself, up to the most you said you could add, "
               "and re-solved for the highest net profit.")
    fees = {j[0]: j[1] for j in jobs_t}
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Net profit with fixes", peso(imp["net_profit"]), delta=signed_peso(imp["gain_month"]) + " vs best plan")
    g2.metric("Profit gained per month", peso(imp["gain_month"]))
    g3.metric("Profit gained per year", peso(imp["gain_year"]))
    g4.metric("Cost of added capacity", peso(imp["expansion_cost"]), help="Extra cost of the added hours and candidates "
              "you entered above. Normal sourcing and recruiter pay are already in net profit.")
    cur = res.current
    if cur is not None:
        total = imp["net_profit"] - cur["net_profit"]
        st.caption(f":material/info: Against today's plan, the best plan with these fixes adds {peso(total)} a month "
                   f"({peso(12 * total)} a year).")

    if imp["fixes"]:
        st.success(imp["insights"][0], icon=":material/check_circle:")
        rows = []
        for f in imp["fixes"]:
            rows.append({"What to add": f["name"], "Add each month": amount(f["add"], f["unit"]),
                         "Extra cost each month": peso(f["cost_total"]),
                         "Profit gained if done alone": peso(f["gain_alone"]),
                         "How": FIX_ADVICE[f["key"]]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("The fixes work together (more hours need more candidates, and the reverse), so the profit gained "
                   "if each is done alone does not add up to the total.")
    else:
        st.info(imp["insights"][0], icon=":material/info:")

    plan_rows = [{"Job type": n, "Best plan": num(a), "With fixes": num(b), "Change": f"{b - a:+,.0f}" if abs(b - a) >= 0.005 else "0",
                  "Fees with fixes": peso(b * fees[n])} for n, a, b in zip(res.names, res.hires, imp["hires"])]
    st.markdown("**Plan with the bottlenecks fixed**")
    st.dataframe(pd.DataFrame(plan_rows), hide_index=True)
    if imp["whole_hires"] is not None and any(abs(h - round(h)) > 1e-6 for h in imp["hires"]):
        st.caption("Counting whole hires only: " + ", ".join(f"{h} {n.lower()}" for n, h in zip(res.names, imp["whole_hires"]))
                   + f" (net profit {peso(imp['whole_net'])}).")
    if imp["new_limits"]:
        lim_rows = [{"Limit": {"hours": "Recruiter hours", "pool": "Qualified candidates", "budget": "Sourcing budget"}[k],
                     "Now": amount(v["old"], {"hours": "hours", "pool": "candidates", "budget": "PHP"}[k]),
                     "Needed": amount(v["new"], {"hours": "hours", "pool": "candidates", "budget": "PHP"}[k])}
                    for k, v in imp["new_limits"].items() if v["new"] - v["old"] > 1e-6]
        if lim_rows:
            st.markdown("**New limits to aim for**")
            st.dataframe(pd.DataFrame(lim_rows), hide_index=True)

    if imp["still"]:
        with st.container(border=True):
            st.markdown("**:material/flag: What still holds the improved plan back**")
            st.markdown("\n".join(f"- **{x['name']}**: {x['note']}" for x in imp["still"]))
            if any(x["kind"] == "cap" for x in imp["still"]):
                st.caption("A maximum-hires cap is a policy or client limit, not something the model can buy past. "
                           "Raising it is a conversation with the client or your team.")


def render_placement(res, region, jobs_t, imp):
    if not res.ok:
        st.error(res.error, icon=":material/error:")
        return

    fees = {j[0]: j[1] for j in jobs_t}
    total_hires = sum(res.hires)
    bottlenecks = [l for l in res.limits if l["binding"] and abs(l["shadow"]) > 0]
    hours_row = next(l for l in res.limits if l["key"] == "hours")

    st.subheader(":material/emoji_events: Best monthly plan")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Monthly placement fees", peso(res.revenue))
    m2.metric("Costs", peso(res.total_cost), help="Sourcing cost of the hires plus recruiter pay for the hours used.")
    m3.metric("Net profit", peso(res.net_profit), help="Placement fees minus the costs entered.")
    m4.metric("Net margin", f"{res.margin:.0%}")
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Total hires", num(total_hires))
    n2.metric("Cost per hire", peso(res.cost_per_hire) if res.total_cost > 0 else "n/a")
    n3.metric("Bottlenecks", str(len(bottlenecks)))
    n4.metric("Recruiter hours used", f"{hours_row['used'] / hours_row['limit']:.0%}")
    if res.total_cost == 0:
        st.caption(":material/info: No costs entered yet, so net profit equals the fees. Add a sourcing cost per hire "
                   "in the table or a recruiter cost per hour to see profit and cost per hire.")
    else:
        st.caption(f":material/info: The plan is chosen to maximize placement fees, then costs are subtracted "
                   f"(sourcing {peso(res.sourcing_cost)}, recruiter pay {peso(res.recruiter_cost)}). "
                   "Other overhead is not included.")

    per_row = 4
    for start in range(0, len(res.names), per_row):
        cols = st.columns(per_row)
        for col, name, h in zip(cols, res.names[start:start + per_row], res.hires[start:start + per_row]):
            col.metric(f"{name} hires", num(h))
            col.caption(f"{peso(h * fees[name])} in fees")

    plan_text = ", ".join(f"{num(h)} {n.lower()}" for n, h in zip(res.names, res.hires))
    st.success(f"Hire {plan_text} each month to earn **{peso(res.revenue)}** in placement fees.",
               icon=":material/check_circle:")
    if res.whole_hires is not None and any(abs(h - round(h)) > 1e-6 for h in res.hires):
        whole = ", ".join(f"{h} {n.lower()}" for n, h in zip(res.names, res.whole_hires))
        st.caption(f"Counting whole hires only: {whole}, for {peso(res.whole_revenue)}.")

    with st.container(border=True):
        st.markdown("**:material/lightbulb: What this means**")
        st.markdown("\n".join(f"- {t}" for t in res.insights))

    # ---- profit gained and cost saved against today's plan ----------------------------
    cur = res.current
    if cur is not None:
        st.subheader(":material/trending_up: Profit gained and cost saved")
        st.caption("The best plan compared with the hires you make today (the \"Current hires\" column).")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Profit gained per month", peso(cur["profit_gain_month"]),
                  delta=signed_peso(cur["profit_gain_month"]) if abs(cur["profit_gain_month"]) >= 0.5 else None)
        c2.metric("Profit gained per year", peso(cur["profit_gain_year"]))
        c3.metric("Fees gained per month", peso(cur["fees_gain"]))
        saved = cur["cost_per_hire_saved"]
        c4.metric("Cost per hire saved", "n/a" if saved is None or res.total_cost == 0 else peso(saved),
                  help="Today's cost per hire minus the best plan's. A negative number means the best plan spends more "
                       "per hire (for example because it shifts to hires that cost more to source).")
        d1, d2, d3 = st.columns(3)
        d1.metric("Net profit today", peso(cur["net_profit"]))
        d2.metric("Net profit, best plan", peso(res.net_profit))
        d3.metric("Total hires today", num(cur["total_hires"]),
                  delta=f"{sum(res.hires) - cur['total_hires']:+,.0f} in best plan")
        if cur["breaks"]:
            st.warning("Today's plan goes past these limits: " + ", ".join(cur["breaks"]) +
                       ". Check the current hires you entered.", icon=":material/warning:")
    else:
        st.caption(":material/info: Enter the hires you make today in the \"Current hires\" column to see the profit "
                   "gained and the cost per hire saved by switching to the best plan.")

    # ---- limits ----------------------------------------------------------------------
    st.subheader(":material/speed: Which limits are holding you back")
    st.caption("A bottleneck is a limit the plan uses completely.")
    rows = []
    for l in res.limits:
        if l["kind"] == "<=":
            usage = 0 if l["limit"] == 0 else l["used"] / l["limit"]
            plan_uses, limit_txt = amount(l["used"], l["unit"]), "≤ " + amount(l["limit"], l["unit"])
        else:
            usage = 1.0 if l["used"] == 0 else l["limit"] / l["used"]
            plan_uses, limit_txt = amount(l["used"], l["unit"]), "≥ " + amount(l["limit"], l["unit"])
        rows.append({"Limit": l["name"], "Plan uses": plan_uses, "Limit set at": limit_txt,
                     "Usage": min(100.0, max(0.0, usage * 100)),
                     "Status": "🟠 Bottleneck" if l["binding"] else "🟢 Room to spare"})
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={"Usage": st.column_config.ProgressColumn("Usage", min_value=0, max_value=100,
                                                                         format="%d%%")})

    render_fixes(res, imp, jobs_t)

    # ---- sensitivity -----------------------------------------------------------------
    st.subheader(":material/query_stats: What one more unit is worth")
    st.caption("How monthly fees change if a limit moves, and how far it can move before the best plan changes.")
    rows = []
    for l in res.limits:
        lo = "any lower value" if l["lower"] is None else amount(l["lower"], l["unit"])
        hi = "no upper limit" if l["upper"] is None else amount(l["upper"], l["unit"])
        rows.append({"Limit": l["name"], "If it goes up by": amount(l["step"], l["unit"]),
                     "Fees change by": signed_peso(l["shadow"] * l["step"]),
                     "Holds while the limit is between": f"{lo} and {hi}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader(":material/payments: How far each fee can change")
    st.caption("The best plan stays the same while a fee stays inside its range. Each fee is changed on its own.")
    rows = [{"Job type": f["name"], "Current fee": peso(f["fee"]),
             "Lowest fee that keeps the plan": peso(f["lower"]),
             "Highest fee that keeps the plan": "no upper limit" if f["upper"] is None else peso(f["upper"])}
            for f in res.fee_ranges]
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    # ---- charts ----------------------------------------------------------------------
    if region is not None:
        st.subheader(":material/show_chart: The feasible region")
        st.caption("Every plan inside the shaded area respects all your limits. The best plan sits on a corner.")
        fig = region_figure(res, region)
        st.pyplot(fig)
        plt.close(fig)
        corner = pd.DataFrame([{f"{res.names[0]} (x)": v[0], f"{res.names[1]} (y)": v[1],
                                "Monthly fees": peso(v[2]),
                                "Result": "Best" if abs(v[2] - res.revenue) < 1e-6 else "Feasible"}
                               for v in region["vertices"]])
        with st.expander("Corner points compared", icon=":material/table_chart:"):
            st.dataframe(corner, hide_index=True)
    else:
        st.subheader(":material/bar_chart: Fees by job type")
        fee_df = pd.DataFrame({"Fees (₱)": [h * fees[n] for n, h in zip(res.names, res.hires)]}, index=res.names)
        st.bar_chart(fee_df)
        st.caption("The feasible-region chart needs exactly two job types, so it is hidden for this plan.")

    # ---- download --------------------------------------------------------------------
    out = pd.DataFrame({"Job type": res.names + ["Total"], "Hires": res.hires + [total_hires],
                        "Fees (PHP)": [h * fees[n] for n, h in zip(res.names, res.hires)] + [res.revenue]})
    out.loc[len(out)] = ["Costs (PHP)", None, res.total_cost]
    out.loc[len(out)] = ["Net profit (PHP)", None, res.net_profit]
    if res.current is not None:
        out.loc[len(out)] = ["Net profit today (PHP)", None, res.current["net_profit"]]
        out.loc[len(out)] = ["Profit gained per month (PHP)", None, res.current["profit_gain_month"]]
    if imp is not None:
        out.loc[len(out)] = ["Net profit with bottleneck fixes (PHP)", None, imp["net_profit"]]
        out.loc[len(out)] = ["Profit gained by fixes per month (PHP)", None, imp["gain_month"]]
        for f in imp["fixes"]:
            out.loc[len(out)] = [f"Extra cost to add {num(f['add'])} {f['unit']} (PHP)", None, f["cost_total"]]
    st.download_button(":material/download: Download plan (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="placement_plan.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
#  Tabs                                                                       #
# --------------------------------------------------------------------------- #
tab_plan, tab_queue, tab_about = st.tabs([":material/groups: Placement planner",
                                          ":material/schedule: Interview queue",
                                          ":material/menu_book: About"])

# ============================== PLACEMENT PLANNER ============================== #
with tab_plan:
    st.subheader(":material/tune: Your numbers")
    preset_name = st.selectbox("Start from an example", list(PRESETS), key="preset",
                               help="Pick an example to fill the form, then change any number.")
    preset = PRESETS[preset_name]
    k = list(PRESETS).index(preset_name)          # new keys per example, so the form resets cleanly

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("**:material/work: Job types**")
            st.caption("One row per type of job you fill. Add or delete rows. "
                       "Leave minimum or maximum blank if there is none. Fill in current hires to see the profit "
                       "gained over today's plan (the example values are made up).")
            table = st.data_editor(
                pd.DataFrame(preset["jobs"], columns=COLS), num_rows="dynamic", hide_index=True, key=f"jobs_{k}",
                column_config={
                    COLS[0]: st.column_config.TextColumn("Job type", required=True),
                    COLS[1]: st.column_config.NumberColumn(COLS[1], min_value=0, step=1000, format="%d"),
                    COLS[2]: st.column_config.NumberColumn(COLS[2], min_value=0, step=1, format="%g"),
                    COLS[3]: st.column_config.NumberColumn(COLS[3], min_value=0, step=500, format="%d",
                                                           help="Only used if you set a sourcing budget."),
                    COLS[4]: st.column_config.NumberColumn(COLS[4], min_value=0, step=1, format="%g"),
                    COLS[5]: st.column_config.NumberColumn(COLS[5], min_value=0, step=1, format="%g"),
                    COLS[6]: st.column_config.NumberColumn(COLS[6], min_value=0, step=1, format="%g",
                                                           help="Hires you make per month today. Used to show the profit "
                                                                "gained and cost saved by switching to the best plan."),
                })
    with right:
        with st.container(border=True):
            st.markdown("**:material/event_available: Monthly limits**")
            hours = st.number_input("Recruiter hours available", min_value=0.0, value=float(preset["hours"]),
                                    step=10.0, key=f"hours_{k}")
            pool = st.number_input("Qualified candidates available", min_value=0.0, value=float(preset["pool"]),
                                   step=1.0, key=f"pool_{k}")
            use_budget = st.checkbox("Limit the sourcing budget", value=preset["budget"] is not None,
                                     key=f"usebudget_{k}")
            budget = st.number_input("Monthly sourcing budget (₱)", min_value=0.0,
                                     value=float(preset["budget"] or 300000), step=10000.0,
                                     key=f"budget_{k}", disabled=not use_budget)
            rate = st.number_input("Recruiter cost per hour (₱, optional)", min_value=0.0, value=0.0, step=10.0,
                                   key=f"rate_{k}", help="Used only to work out net profit and cost per hire.")

    with st.container(border=True):
        st.markdown("**:material/build_circle: Fix the bottlenecks (optional)**")
        st.caption("Say how much extra capacity you could add and what it would cost. The planner adds capacity only "
                   "where it pays for itself. The costs below are placeholders: replace them with your own figures.")
        find_fix = st.checkbox("Find the best plan after fixing the bottlenecks", value=True, key=f"fix_{k}")
        f1, f2, f3, f4, f5 = st.columns(5)
        h_max = f1.number_input("Most extra recruiter hours", min_value=0.0, value=float(round(0.3 * preset["hours"])),
                                step=10.0, key=f"hmax_{k}", disabled=not find_fix)
        h_cost = f2.number_input("Extra cost per added hour (₱)", min_value=0.0, value=150.0, step=10.0,
                                 key=f"hcost_{k}", disabled=not find_fix,
                                 help="On top of normal recruiter pay, for example an overtime premium or agency fee.")
        p_max = f3.number_input("Most extra candidates", min_value=0.0, value=float(round(0.3 * preset["pool"])),
                                step=1.0, key=f"pmax_{k}", disabled=not find_fix)
        p_cost = f4.number_input("Extra cost per added candidate (₱)", min_value=0.0, value=2000.0, step=100.0,
                                 key=f"pcost_{k}", disabled=not find_fix,
                                 help="Advertising, referral or agency cost of finding one more qualified candidate.")
        b_max = f5.number_input("Most extra sourcing budget (₱)", min_value=0.0,
                                value=float(round(0.3 * (preset["budget"] or 0))), step=10000.0, key=f"bmax_{k}",
                                disabled=not find_fix or not use_budget,
                                help="Only used when the sourcing budget is limited. The money is counted when it is "
                                     "spent on hires.")

    jobs_t = table_to_jobs(table)
    fix = (float(h_cost), float(h_max), float(p_cost), float(p_max), float(b_max) if use_budget else 0.0) if find_fix else None
    res, region, imp = run_placement(jobs_t, float(hours), float(pool), float(budget) if use_budget else None, float(rate), fix)
    st.divider()
    render_placement(res, region, jobs_t, imp)

# ================================ INTERVIEW QUEUE ================================ #
with tab_queue:
    st.subheader(":material/tune: Your numbers")
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        lam = c1.number_input("Candidates arriving per hour", min_value=0.0, value=20.0, step=1.0)
        mu = c2.number_input("Interviews per hour, per interviewer", min_value=0.0, value=25.0, step=1.0)
        target = c3.number_input("Longest acceptable wait (minutes)", min_value=0.0, value=1.0, step=0.5)
        util = c4.number_input("Highest acceptable busy time (%)", min_value=1.0, max_value=99.0, value=85.0, step=1.0)
        smax = c5.number_input("Compare up to (interviewers)", min_value=1, max_value=20, value=4, step=1)

    with st.container(border=True):
        st.markdown("**:material/payments: Cost assumptions (optional)**")
        st.caption("These are placeholders. Replace them with your own figures, because the cost comparison is only "
                   "as good as these numbers.")
        use_cost = st.checkbox("Compare the monthly cost of each team size", value=True)
        k0, k1, k2, k3, k4 = st.columns(5)
        c_cur = k0.number_input("Interviewers you have today", min_value=1, max_value=20, value=1, step=1,
                                help="Cost saved is measured against this team size.")
        c_hours = k1.number_input("Interview hours per month", min_value=1.0, value=160.0, step=10.0,
                                  disabled=not use_cost)
        c_rate = k2.number_input("Cost per interviewer hour (₱)", min_value=0.0, value=150.0, step=10.0,
                                 disabled=not use_cost)
        c_drop = k3.number_input("Candidates lost per minute of average wait (%)", min_value=0.0, max_value=100.0,
                                 value=0.5, step=0.1, disabled=not use_cost,
                                 help="For example 0.5 means a 4.8% loss when the average wait is 9.6 minutes.")
        c_value = k4.number_input("Fee value of each lost candidate (₱)", min_value=0.0, value=1000.0, step=100.0,
                                  disabled=not use_cost,
                                  help="A rough guide: monthly placement fees divided by candidates interviewed per month.")

    q = run_queue(float(lam), float(mu), int(smax), float(target), float(util),
                  (float(c_hours), float(c_rate), float(c_drop), float(c_value)) if use_cost else None, int(c_cur))
    st.divider()
    if not q["ok"]:
        st.error(q["error"], icon=":material/error:")
    else:
        rows, rec = q["rows"], q["recommended"]
        st.subheader(":material/emoji_events: Recommended interview team")
        best = rows[rec - 1] if rec else None
        one = rows[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Interviewers needed", str(rec) if rec else "Not yet")
        m2.metric("Average wait", f"{best['wq_min']:.1f} min" if best else "n/a")
        m3.metric("Wait with one interviewer", f"{one['wq_min']:.1f} min" if one["stable"] else "Never ends")
        m4.metric("Busy time per interviewer", f"{best['utilization']:.0%}" if best else "n/a")

        for line in q["insights"]:
            if line.startswith(("On cost", "Or keep")):
                st.info(line, icon=":material/savings:")
            else:
                (st.success if rec else st.warning)(line, icon=":material/check_circle:" if rec else ":material/warning:")
        cost = q["cost"]
        cur_n = q["current_servers"]
        cur_txt = "one interviewer" if cur_n == 1 else f"your {cur_n} interviewers"
        if cost and cost["best"]:
            st.subheader(":material/savings: Cost saved (monthly)")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Lowest-cost team", f"{cost['best']} interviewer{'s' if cost['best'] > 1 else ''}")
            s2.metric("Total monthly cost", peso(cost["best_total"]),
                      help="Interviewer pay plus the fees lost when waiting candidates drop out.")
            s3.metric(f"Cost saved vs {cur_txt}", peso(cost["saving_month"]) if cost["saving_month"] is not None else "n/a",
                      help="Monthly cost of your current team minus the lowest-cost team.")
            s4.metric("Cost saved per year", peso(cost["saving_year"]) if cost["saving_year"] is not None else "n/a")
            if cost["saving_month"] is not None:
                t1, t2, t3, t4 = st.columns(4)
                t1.metric(f"Monthly cost today ({cur_n})", peso(cost["current_total"]))
                t2.metric("Interviewer pay change", signed_peso(cost["pay_change"]),
                          help="Extra interviewers cost more in pay. Negative means pay goes down.")
                t3.metric("Lost-candidate cost saved", peso(cost["lost_saved"]),
                          help="Value of the candidates who no longer drop out while waiting.")
                t4.metric("Candidates kept each month", num(cost["lost_candidates_saved"]))
            if cost["rec_total"] is not None and q["recommended"] and q["recommended"] != cost["best"]:
                st.caption(f":material/info: The team that meets your wait targets ({q['recommended']} interviewers) costs "
                           f"{peso(cost['rec_total'])} a month"
                           + (f", which saves {peso(cost['rec_saving_month'])} against {cur_txt}." if cost["rec_saving_month"] is not None and cost["rec_saving_month"] > 0 else "."))

        fx = q["fixes"]
        st.subheader(":material/build_circle: Solutions for the wait")
        if fx["meets_now"]:
            st.success(f"Your {cur_n} interviewer{'s' if cur_n > 1 else ''} already meet your wait and busy-time targets, "
                       "so no fix is needed.", icon=":material/check_circle:")
        else:
            st.caption(f"Ways to meet your targets, starting from {cur_txt}:")
            opts = []
            if rec:
                opts.append({"Option": f"Go up to {rec} interviewers", "What changes": f"{rec - cur_n:+d} interviewer(s)" if rec != cur_n else "None",
                             "Result": f"Average wait {rows[rec - 1]['wq_min']:.1f} min, busy {rows[rec - 1]['utilization']:.0%}"})
            if fx["needed_mu"] is not None:
                opts.append({"Option": "Make each interview faster", "What changes": f"+{fx['speedup_pct']:.0f}% "
                             f"(to {fx['needed_mu']:.1f} per hour)", "Result": "Meets both targets without hiring"})
            if fx["max_arrival"] is not None:
                opts.append({"Option": "Spread out arrivals (book slots)", "What changes": f"Cap at {fx['max_arrival']:.1f} "
                             f"candidates per hour (−{fx['arrival_cut_pct']:.0f}%)", "Result": "Meets both targets without hiring"})
            st.dataframe(pd.DataFrame(opts), hide_index=True)
            if cost and cost["best"] and rec and cost["best"] != rec:
                st.caption("Hiring is not the only route. Compare its cost with the cost table below before you decide.")

        st.subheader(":material/table_chart: Interviewers compared")
        st.caption(f"Targets: average wait of {target:g} minutes or less, interviewers busy {util:.0f}% of the time "
                   f"or less. At least {q['min_servers']} interviewer{'s are' if q['min_servers'] > 1 else ' is'} "
                   "needed for the line to stay stable.")
        table_rows = []
        for r in rows:
            if r["stable"]:
                row = {"Interviewers": r["servers"], "Busy time": r["utilization"] * 100,
                       "Chance of waiting": f"{r['p_wait']:.0%}", "Average wait (min)": round(r["wq_min"], 2),
                       "Candidates in line": round(r["lq"], 2), "Time to finish (min)": round(r["w_min"], 1),
                       "Meets targets": "✅ Yes" if r["meets"] else "❌ No"}
                if cost:
                    row.update({"Interviewer pay (₱)": round(r["staff_cost"]), "Lost candidates": round(r["lost"], 1),
                                "Value of lost candidates (₱)": round(r["lost_value"]),
                                "Total monthly cost (₱)": round(r["total_cost"])})
                table_rows.append(row)
            else:
                row = {"Interviewers": r["servers"], "Busy time": min(100.0, r["utilization"] * 100),
                       "Chance of waiting": "n/a", "Average wait (min)": None,
                       "Candidates in line": None, "Time to finish (min)": None,
                       "Meets targets": "❌ Cannot keep up"}
                if cost:
                    row.update({"Interviewer pay (₱)": None, "Lost candidates": None,
                                "Value of lost candidates (₱)": None, "Total monthly cost (₱)": None})
                table_rows.append(row)
        qdf = pd.DataFrame(table_rows)
        st.dataframe(qdf, hide_index=True,
                     column_config={"Busy time": st.column_config.ProgressColumn("Busy time", min_value=0,
                                                                                  max_value=100, format="%d%%")})
        stable = qdf.dropna(subset=["Average wait (min)"])
        if len(stable):
            st.subheader(":material/bar_chart: Average wait by team size")
            st.bar_chart(stable.set_index("Interviewers")["Average wait (min)"])
            if cost:
                st.subheader(":material/stacked_bar_chart: Monthly cost by team size")
                st.caption("Interviewer pay plus the value of candidates lost to waiting. The shortest bar is the cheapest.")
                st.bar_chart(stable.set_index("Interviewers")[["Interviewer pay (₱)", "Value of lost candidates (₱)"]])

# ==================================== ABOUT ==================================== #
with tab_about:
    st.subheader(":material/menu_book: How the models work")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### :material/groups: Placement planner")
            st.markdown(
                "Uses **Linear Programming**: it chooses how many hires of each job type to make so that total "
                "placement fees are as high as possible without breaking any limit.\n\n"
                "- **Limits:** recruiter hours, qualified candidates, an optional sourcing budget, and optional "
                "minimum or maximum hires per job type.\n"
                "- **Shadow prices:** what one more hour, candidate or peso of budget is worth.\n"
                "- **Ranges:** how far a limit or a fee can change before the best plan changes.\n"
                "- **Profit gained:** the best plan compared with the hires you make today (optional column).\n"
                "- **Bottleneck solutions:** the planner raises a limit only where the extra fees beat the extra cost, "
                "up to the most you say you could add, and shows the best plan after the fix.\n"
                "- Works with **any number of job types**, solved with SciPy's HiGHS solver.")
    with c2:
        with st.container(border=True):
            st.markdown("#### :material/schedule: Interview queue")
            st.markdown(
                "Uses an **M/M/s : (FCFS / ∞ / ∞)** queueing model: candidates arrive at random, are interviewed "
                "first come, first served, and each interviewer works at a steady average pace.\n\n"
                "- **Busy time:** the share of time each interviewer is occupied.\n"
                "- **Chance of waiting:** the probability that a candidate finds every interviewer busy.\n"
                "- **Average wait:** time in line before the interview starts.\n"
                "- The line only stays stable if arrivals are below total interview capacity.\n"
                "- **Cost saved:** the lowest-cost team compared with the team you have today.\n"
                "- **Solutions for the wait:** more interviewers, faster interviews, or spreading out arrivals.")
    with st.container(border=True):
        st.markdown("#### :material/rule: What to keep in mind")
        st.markdown(
            "- Fees, hours and limits are treated as fixed within the month.\n"
            "- Fractions of a hire are allowed in the main plan. A whole-number plan is shown when they differ.\n"
            "- Real arrival and interview rates change during the day. Check the results against your own data.\n"
            "- The effect of waiting on candidate dropouts is not modeled and should be validated with company data.\n"
            "- Profit gained, cost saved and the cost of added capacity depend on the placeholder costs you enter. "
            "Replace them with your own figures before acting on them.\n"
            "- Added capacity is assumed to be available at the cost and amount you enter, and fees per hire stay the same.")
