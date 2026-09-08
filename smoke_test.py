"""
Headless smoke test — run before calling any change done.

Executes each page the way Streamlit does and fails on any exception. This
exists because eyeballing a screenshot after an edit catches a broken page only
if you happen to scroll to the broken part; a NameError below the fold looks
exactly like a page that is still rendering.

    python smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PAGES = ["tools/edge_concierge.py", "tools/destination_path.py",
         "tools/learned_edge.py", "tools/neural_edge.py", "tools/factor_edge.py"]

# What each page must actually get through. The Learned Edge stops at its
# uploader by design, so it is only asserted to reach step 1; the Neural Edge
# stops at its fit button, so it is asserted to reach the universe screen.
EXPECT = {
    "tools/edge_concierge.py": {
        "subheaders": ["1 · Panel", "2 · Sketch the edge", "3 · Universe",
                       "4 · The edge", "5 · Test"],
        "metrics": 5},
    "tools/destination_path.py": {
        "subheaders": ["1 · Panel", "2 · Coverage", "3 · The two edges", "4 · Test",
                       "5 · Unite — pace the trades"],
        "metrics": 11},   # 4 per edge + 3 agreement
    "tools/learned_edge.py": {
        "subheaders": ["1 · Holdings panel"], "metrics": 0},
    "tools/neural_edge.py": {
        "subheaders": ["1 · Target & features", "2 · Universe", "3 · Fit"],
        "metrics": 0},
    "tools/factor_edge.py": {
        "subheaders": ["1 · Target & features", "2 · Universe", "3 · Fit"],
        "metrics": 0},
}
MODULES = ["engine", "data_io", "report", "learned", "nl", "ui", "neural",
           "concepts", "autoencoder", "pace"]


def check_imports() -> list:
    import importlib
    fails = []
    for m in MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:
            fails.append(f"import {m}: {type(e).__name__}: {e}")
    return fails


def check_engine() -> list:
    """The numbers, not just the wiring: a page can render and still be wrong."""
    import pandas as pd
    from engine import (Constraint, EdgeSpec, FeatureSpec, consistency_checks,
                        run_edge)
    fails = []
    p = pd.read_parquet(ROOT / "data" / "base_panel.parquet")
    spec = EdgeSpec([FeatureSpec("fcf_yield", "rank", 1.0),
                     FeatureSpec("gpa", "rank", 0.5),
                     FeatureSpec("accruals", "rank", -0.5)],
                    constraints=[Constraint("roa", ">", "0")])
    r = run_edge(p, spec, horizon="fwd_1m")
    if not (0 <= r["composite"].min() <= r["composite"].max() <= 1):
        fails.append("composite outside [0, 1]")
    if not r["q_ann_returns"].notna().all():
        fails.append("ntile returns contain NaN")
    # sign flip must invert IC exactly — catches a broken transform silently
    a = run_edge(p, EdgeSpec([FeatureSpec("gpa", "rank", 1.0)]), horizon="fwd_1m")
    b = run_edge(p, EdgeSpec([FeatureSpec("gpa", "rank", -1.0)]), horizon="fwd_1m")
    if abs(a["mean_ic"] + b["mean_ic"]) > 1e-6:
        fails.append(f"sign flip not symmetric: {a['mean_ic']} vs {b['mean_ic']}")
    cut = p["date"].max() - pd.DateOffset(years=5)
    c = consistency_checks(run_edge(p[p.date < cut], spec, horizon="fwd_1m"),
                           run_edge(p[p.date >= cut], spec, horizon="fwd_1m"))
    if c["verdict"] not in {"consistent", "partial", "weak", "inconsistent"}:
        fails.append(f"unexpected verdict {c['verdict']}")
    return fails


def check_pace() -> list:
    """The pacing mechanism on a synthetic coverage: wholesale must sit on the
    target every month, budgeted books must respect the budget, weights must
    sum to one, and a path signal that literally is next month's return must
    make deferral pay in the direction the mechanism claims."""
    import numpy as np
    import pandas as pd
    from pace import deferral_test, run_pace
    fails = []
    rng = np.random.default_rng(1)
    dates = pd.date_range("2010-01-31", periods=60, freq="ME")
    names = [f"S{i}" for i in range(40)]
    rows = []
    for dt in dates:
        fwd = rng.normal(0.01, 0.06, len(names))
        for i, n in enumerate(names):
            rows.append({"date": dt, "ticker": n, "dest": rng.normal(),
                         "path": fwd[i] + 0.02 * rng.normal(),   # near-perfect path
                         "fwd_1m": fwd[i], "log_mcap": rng.normal(10, 1)})
    df = pd.DataFrame(rows)
    w = run_pace(df, mode="wholesale", budget=0.08, n_q=3)
    if w.distance.max() > 1e-9:
        fails.append(f"wholesale book not on target (max distance {w.distance.max():.2e})")
    for mode in ("uniform", "modulated"):
        r = run_pace(df, mode=mode, budget=0.08, n_q=3)
        if r.turnover.max() > 0.08 + 1e-9:
            fails.append(f"{mode} exceeded the turnover budget ({r.turnover.max():.4f})")
        sums = r.weights.groupby("date")["weight"].sum()
        if (sums - 1.0).abs().max() > 1e-9:
            fails.append(f"{mode} weights do not sum to one")
    m = run_pace(df, mode="modulated", budget=0.08, n_q=3)
    dt_ = deferral_test(m)
    if len(dt_) == 2 and not (dt_["mean_diff"] > 0).all():
        fails.append(f"deferral did not pay with a perfect path signal: "
                     f"{dt_[['side', 'mean_diff']].to_dict('records')}")
    return fails


def check_neural() -> list:
    """The net and the TCAV plumbing, on a small synthetic panel: the analytic
    gradient must match finite differences (TCAV is wrong everywhere if it
    doesn't), a planted concept must score aligned, and its CAV must separate."""
    import numpy as np
    import pandas as pd
    from concepts import rule_mask, QuantileRule, run_tcav
    from neural import (fit_full_history, sanity_check_grad, walk_forward_net,
                        PRED_COL)
    fails = []
    rng = np.random.default_rng(0)
    dates = pd.date_range("2005-01-31", periods=120, freq="ME")
    n_sec = 60
    rows = []
    for dt in dates:
        # f1 drives returns; the f2·f3 interaction makes the gradient vary by
        # row, so random CAV directions score near 0.5 instead of collapsing
        # onto {0, 1} the way a purely linear signal degenerately would
        f = rng.normal(size=(n_sec, 4))
        y = 0.05 * f[:, 0] + 0.04 * f[:, 1] * f[:, 2] + 0.01 * rng.normal(size=n_sec)
        for i in range(n_sec):
            rows.append({"date": dt, "ticker": f"S{i}", "f1": f[i, 0],
                         "f2": f[i, 1], "f3": f[i, 2], "f4": f[i, 3],
                         "fwd": y[i]})
    df = pd.DataFrame(rows)
    feats = ["f1", "f2", "f3", "f4"]

    res = walk_forward_net(df, feats, "fwd", arch="NN3", n_seeds=1,
                           init_train_years=4, step_years=2)
    if not res.panel[PRED_COL].notna().any():
        fails.append("walk-forward produced no predictions")
    if not res.ic_mean > 0.2:
        fails.append(f"net failed to learn a planted signal (IC {res.ic_mean:.3f})")

    # market-adjusted target: demeaning per date must not break the protocol,
    # and the per-month IC is rank-based so the planted signal must survive
    df["fwd_xmkt"] = df["fwd"] - df.groupby("date")["fwd"].transform("mean")
    res_x = walk_forward_net(df, feats, "fwd_xmkt", arch="NN3", n_seeds=1,
                             init_train_years=4, step_years=2)
    if not res_x.ic_mean > 0.2:
        fails.append(f"over-market target lost the planted signal "
                     f"(IC {res_x.ic_mean:.3f})")

    full = fit_full_history(df, feats, "fwd", arch="NN3", n_seeds=1)
    err = sanity_check_grad(full["nets"][0], full["X"], layer=2)
    if err > 1e-6:
        fails.append(f"analytic gradient off by {err:.2e} vs finite differences")

    # a concept that IS the signal must come out aligned and separable
    mask = rule_mask(df, [QuantileRule("f1", "top", 0.2)])
    rep = run_tcav(full, mask, layer=2, n_runs=8)
    if rep.scores.mean() < 0.7:
        fails.append(f"planted concept TCAV score {rep.scores.mean():.2f} — "
                     f"expected clear alignment")
    if rep.cav_accs.mean() < 0.7:
        fails.append(f"planted concept CAV accuracy {rep.cav_accs.mean():.2f}")
    if rep.p_value > 0.05:
        fails.append(f"planted concept not significant vs random (p={rep.p_value:.3f})")
    # per-row views: concept rows must express the concept more than the rest
    m = mask.to_numpy()
    if len(rep.expression) != len(df):
        fails.append("expression not aligned with the panel rows")
    elif rep.expression[m].mean() - rep.expression[~m].mean() < 0.2:
        fails.append(f"concept rows don't express the concept "
                     f"({rep.expression[m].mean():.2f} vs "
                     f"{rep.expression[~m].mean():.2f})")
    if not 0 <= rep.row_align.min() <= rep.row_align.max() <= 1:
        fails.append("row alignment outside [0, 1]")
    return fails


def check_autoencoder() -> list:
    """The CA trainer: analytic gradients must match finite differences, and
    a planted one-factor structure with a real premium must be recovered —
    factor fit (total R²), premium exploitation (IC), and row alignment (the
    write-back once scrambled tickers within months via an unstable sort)."""
    import numpy as np
    import pandas as pd
    from autoencoder import grad_check, walk_forward_ca
    fails = []
    err = grad_check()
    if err > 1e-4:
        fails.append(f"CA analytic gradient off by {err:.2e} vs finite diff")

    rng = np.random.default_rng(0)
    dates = pd.date_range("2005-01-31", periods=160, freq="ME")
    rows = []
    for dt in dates:
        z = rng.uniform(-1, 1, size=(60, 3))
        f = rng.normal(0.02, 0.03)      # one factor, detectable premium
        r = z[:, 0] * f + rng.normal(0, 0.015, 60)
        for i in range(60):
            rows.append({"date": dt, "ticker": f"S{i}", "z1": z[i, 0],
                         "z2": z[i, 1], "z3": z[i, 2], "fwd": r[i]})
    df = pd.DataFrame(rows)
    res = walk_forward_ca(df, ["z1", "z2", "z3"], "fwd", n_factors=2,
                          arch="CA1", n_seeds=1, init_train_years=4,
                          step_years=2)
    if not res.total_r2 > 0.25:
        fails.append(f"CA missed planted factor structure (total R² "
                     f"{res.total_r2:.3f})")
    if not res.ic_mean > 0.15:
        fails.append(f"CA failed to exploit the planted premium "
                     f"(IC {res.ic_mean:+.3f}) — check write-back alignment")
    return fails


def check_pages() -> list:
    from streamlit.testing.v1 import AppTest
    fails = []
    for page in PAGES:
        t0 = time.time()
        at = AppTest.from_file(str(ROOT / page), default_timeout=180)
        try:
            at.run()
        except Exception as e:
            fails.append(f"{page}: harness error {type(e).__name__}: {e}")
            continue
        if at.exception:
            for ex in at.exception:
                fails.append(f"{page}: {ex.message.splitlines()[0]}")
            continue
        # "no exception" is not the same as "rendered": a page that hits st.stop()
        # early raises nothing and looks identical to a healthy one from here.
        want = EXPECT[page]
        got = [s.value for s in at.subheader]
        missing = [w for w in want["subheaders"] if w not in got]
        if missing:
            fails.append(f"{page}: never reached {missing}")
        if len(at.metric) < want["metrics"]:
            fails.append(f"{page}: {len(at.metric)} metrics, expected "
                         f">= {want['metrics']}")
        print(f"    {page} rendered in {time.time() - t0:.1f}s "
              f"({len(got)} steps, {len(at.metric)} metrics, {len(at.tabs)} tabs)")
    return fails


def main() -> int:
    all_fails = []
    for name, fn in (("imports", check_imports), ("engine", check_engine),
                     ("pace", check_pace), ("neural", check_neural),
                     ("autoencoder", check_autoencoder), ("pages", check_pages)):
        print(f"  {name}…")
        f = fn()
        all_fails += f
        for x in f:
            print(f"    FAIL {x}")
    print()
    if all_fails:
        print(f"{len(all_fails)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
