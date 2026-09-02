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

PAGES = ["tools/edge_concierge.py", "tools/learned_edge.py"]

# What each page must actually get through. The Learned Edge stops at its
# uploader by design, so it is only asserted to reach step 1.
EXPECT = {
    "tools/edge_concierge.py": {
        "subheaders": ["1 · Panel", "2 · Sketch the edge", "3 · Universe",
                       "4 · The edge", "5 · Test"],
        "metrics": 5},
    "tools/learned_edge.py": {
        "subheaders": ["1 · Holdings panel"], "metrics": 0},
}
MODULES = ["engine", "data_io", "report", "learned", "nl", "ui"]


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
                     ("pages", check_pages)):
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
