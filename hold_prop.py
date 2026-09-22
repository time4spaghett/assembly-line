"""
Hold Propensity — an LLM-inferred investment philosophy compiled into a
deterministic factor rule, then propagated over history.

    Interpret → Compile → Propagate

  * `infer_philosophy`   one strong-LLM pass over the CURRENT holdings cross
                         section (weights + factor values). Returns a human-
                         readable philosophy and a scoring specification (JSON).
  * `validate_scoring_spec`   the contract: only approved factor columns, only
                         the operations below, sane parameters. Fails loudly.
  * `apply_scoring_spec` pure pandas/numpy. Applies the spec independently
                         inside every date cross section and writes one bounded
                         column, `hold_prop_llm` (or whatever the spec names).
                         Same DataFrame + same spec → same column, always. No
                         LLM, no holdings columns.

Scoring language (JSON). A spec is

    {"model_name": "hold_prop_llm",
     "score": <node>,                       # or "components": [...] shorthand
     "output_transform": "percentile" | "clip" | "minmax"}

and a node is one of

    {"op": "percentile",         "factor": f}                 # rank in [0,1] within date
    {"op": "inverse_percentile", "factor": f}                 # 1 − percentile
    {"op": "zscore",             "factor": f}                 # within date
    {"op": "clipped_zscore",     "factor": f, "clip": 3}      # z clipped to ±clip
    {"op": "threshold", "factor": f, "on": "percentile"|"raw", "above": x}   # 1/0
    {"op": "threshold", "factor": f, "on": "percentile"|"raw", "below": x}
    {"op": "piecewise", "factor": f, "on": "percentile"|"raw",
                        "points": [[x0, y0], [x1, y1], ...]}   # linear interpolation
    {"op": "weighted_sum", "terms": [{"weight": w, "node": <node>}, ...]}
    {"op": "multiply", "nodes": [<node>, ...]}                # interaction
    {"op": "min",      "nodes": [<node>, ...]}
    {"op": "max",      "nodes": [<node>, ...]}

The `components` shorthand — a list of {"factor", "transform", "weight"} with
transform in {percentile, inverse_percentile, zscore, clipped_zscore} — is a
weighted_sum of leaves, which is what most philosophies compile to.

Missing factor values are neutral: a NaN percentile becomes 0.5, a NaN z-score
becomes 0, so a name with a gap is neither rewarded nor punished for it.
"""
from __future__ import annotations

import json
import re
from typing import Iterable

import numpy as np
import pandas as pd

LEAF_OPS = {"percentile", "inverse_percentile", "zscore", "clipped_zscore",
            "threshold", "piecewise"}
COMBINE_OPS = {"weighted_sum", "multiply", "min", "max"}
ALLOWED_OPS = LEAF_OPS | COMBINE_OPS
OUTPUT_TRANSFORMS = {"percentile", "clip", "minmax"}
COMPONENT_TRANSFORMS = {"percentile", "inverse_percentile", "zscore", "clipped_zscore"}

# Never usable inside a spec, whatever the caller passes as factor_cols: these
# are the evidence for inference, not inputs to the historical rule.
HOLDINGS_COLS = {"fund_weight", "benchmark_weight", "active_weight",
                 "ewm_abs_weight", "ewm_active_weight", "held", "hold_prop_llm"}

MAX_DEPTH = 6
MAX_LEAVES = 24

# Plain-English definitions for the shipped panels' features, handed to the
# LLM so it reasons about economics rather than column names.
FACTOR_DEFS = {
    "ret_1m": "trailing 1-month log return", "mom_3m": "trailing 3-month return",
    "mom_3_1": "return over months 2–3 (last month excluded)",
    "mom_6_1": "return over months 2–6 (last month excluded)",
    "mom_12_1": "return over months 2–12 (last month excluded)",
    "high_52w": "price as a fraction of its 52-week high",
    "vol_1m": "daily-return volatility, trailing month",
    "vol_12m": "daily-return volatility, trailing year",
    "max_ret_1m": "largest single-day return in the trailing month (lottery-ness)",
    "beta_12m": "1-year beta to the equal-weight universe",
    "btm": "book-to-market", "earn_yield": "earnings / market cap",
    "fcf_yield": "free cash flow / market cap", "sales_yield": "revenue / market cap",
    "roa": "net income / assets", "roe": "net income / equity",
    "roic": "EBIT / (equity + debt)", "gross_margin": "gross profit / revenue",
    "op_margin": "EBIT / revenue", "gpa": "gross profit / assets (Novy-Marx)",
    "fcf_margin": "free cash flow / revenue", "leverage": "debt / assets (higher = more levered)",
    "accruals": "(net income − operating cash flow) / assets (higher = lower earnings quality)",
    "rev_gr_1y": "1-year revenue growth", "rev_gr_3y": "3-year annualised revenue growth",
    "asset_gr": "1-year asset growth (higher = investing harder)",
    "earn_mom": "1-year change in earnings / assets",
    "ebit_mom": "1-year change in EBIT / assets", "margin_mom": "1-year change in gross margin",
    "roe_mom": "1-year change in ROE", "roa_e_mom": "1-year change in EBIT / assets",
    "rev_accel": "1-year revenue growth minus 3-year", "qearn_mom": "YoY change in quarterly earnings / assets",
    "log_mcap": "log market cap",
    "vol_63d": "annualised 63-day realised volatility", "cashburn_streak_q": "consecutive quarters of negative operating cash flow",
    "loss_streak_q": "consecutive loss-making quarters", "dilution_1y": "1-year share count growth",
    "neg_book_equity": "1 if book equity is negative", "cash_margin": "operating cash flow / revenue",
    "ret_12m": "trailing 12-month return", "drawdown_52w": "drawdown from 52-week high",
}


class SpecError(ValueError):
    """A scoring specification that must not be executed."""


# ── Validation ───────────────────────────────────────────────────────────────

def _check_node(node, factor_cols: set, errors: list, depth: int, leaves: list,
                path: str = "score") -> None:
    if depth > MAX_DEPTH:
        errors.append(f"{path}: nesting deeper than {MAX_DEPTH}")
        return
    if not isinstance(node, dict):
        errors.append(f"{path}: node must be an object, got {type(node).__name__}")
        return
    op = node.get("op")
    if op not in ALLOWED_OPS:
        errors.append(f"{path}: unknown op {op!r} (allowed: {sorted(ALLOWED_OPS)})")
        return
    if op in LEAF_OPS:
        f = node.get("factor")
        if f in HOLDINGS_COLS:
            errors.append(f"{path}: {f!r} is a holdings/weight column and cannot be scored on")
        elif f not in factor_cols:
            errors.append(f"{path}: factor {f!r} is not an approved factor column")
        leaves.append(f)
        if op == "clipped_zscore":
            c = node.get("clip", 3.0)
            if not (isinstance(c, (int, float)) and np.isfinite(c) and c > 0):
                errors.append(f"{path}: clip must be a positive number")
        if op == "threshold":
            on = node.get("on", "percentile")
            if on not in ("percentile", "raw"):
                errors.append(f"{path}: threshold 'on' must be percentile or raw")
            has_above, has_below = "above" in node, "below" in node
            if has_above == has_below:
                errors.append(f"{path}: threshold needs exactly one of 'above' / 'below'")
            v = node.get("above", node.get("below"))
            if not (isinstance(v, (int, float)) and np.isfinite(v)):
                errors.append(f"{path}: threshold value must be a finite number")
            elif on == "percentile" and not 0.0 <= v <= 1.0:
                errors.append(f"{path}: a percentile threshold must lie in [0, 1]")
        if op == "piecewise":
            on = node.get("on", "percentile")
            if on not in ("percentile", "raw"):
                errors.append(f"{path}: piecewise 'on' must be percentile or raw")
            pts = node.get("points")
            ok = (isinstance(pts, list) and len(pts) >= 2
                  and all(isinstance(p, (list, tuple)) and len(p) == 2
                          and all(isinstance(v, (int, float)) and np.isfinite(v) for v in p)
                          for p in pts))
            if not ok:
                errors.append(f"{path}: piecewise needs ≥2 finite [x, y] points")
            else:
                xs = [p[0] for p in pts]
                if any(b < a for a, b in zip(xs, xs[1:])):
                    errors.append(f"{path}: piecewise x-values must be non-decreasing")
                if on == "percentile" and not all(0.0 <= x <= 1.0 for x in xs):
                    errors.append(f"{path}: piecewise on percentile needs x in [0, 1]")
        return
    if op == "weighted_sum":
        terms = node.get("terms")
        if not isinstance(terms, list) or not terms:
            errors.append(f"{path}: weighted_sum needs a non-empty 'terms' list")
            return
        for i, t in enumerate(terms):
            if not isinstance(t, dict) or "weight" not in t or "node" not in t:
                errors.append(f"{path}.terms[{i}]: each term needs 'weight' and 'node'")
                continue
            w = t["weight"]
            if not (isinstance(w, (int, float)) and np.isfinite(w)):
                errors.append(f"{path}.terms[{i}]: weight must be a finite number")
            _check_node(t["node"], factor_cols, errors, depth + 1, leaves, f"{path}.terms[{i}]")
        return
    nodes = node.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
        errors.append(f"{path}: {op} needs a 'nodes' list of at least two nodes")
        return
    for i, n in enumerate(nodes):
        _check_node(n, factor_cols, errors, depth + 1, leaves, f"{path}.nodes[{i}]")


def _components_to_node(components) -> dict:
    terms = []
    for i, c in enumerate(components):
        if not isinstance(c, dict):
            raise SpecError(f"components[{i}] must be an object")
        tr = c.get("transform", "percentile")
        if tr not in COMPONENT_TRANSFORMS:
            raise SpecError(f"components[{i}]: transform {tr!r} not in {sorted(COMPONENT_TRANSFORMS)}")
        terms.append({"weight": c.get("weight", 1.0),
                      "node": {"op": tr, "factor": c.get("factor")}})
    return {"op": "weighted_sum", "terms": terms}


def normalize_spec(spec: dict) -> dict:
    """Expand the `components` shorthand; leave an explicit `score` alone."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    out = dict(spec)
    if "score" not in out:
        if "components" not in out:
            raise SpecError("spec needs either 'score' (a node) or 'components' (a list)")
        out["score"] = _components_to_node(out["components"])
    out.setdefault("model_name", "hold_prop_llm")
    out.setdefault("output_transform", "percentile")
    return out


def validate_scoring_spec(spec: dict, factor_cols: Iterable[str]) -> dict:
    """Return the normalised spec, or raise SpecError listing every problem."""
    spec = normalize_spec(spec)
    errors: list[str] = []
    fc = set(factor_cols) - HOLDINGS_COLS
    name = spec["model_name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        errors.append(f"model_name {name!r} is not a valid column name")
    if spec["output_transform"] not in OUTPUT_TRANSFORMS:
        errors.append(f"output_transform {spec['output_transform']!r} not in {sorted(OUTPUT_TRANSFORMS)}")
    leaves: list = []
    _check_node(spec["score"], fc, errors, 1, leaves)
    if len(leaves) > MAX_LEAVES:
        errors.append(f"{len(leaves)} factor references — more than {MAX_LEAVES}; simplify")
    if errors:
        raise SpecError("Scoring spec rejected:\n  - " + "\n  - ".join(errors))
    return spec


def spec_factors(spec: dict) -> list[str]:
    """Every factor a (normalised) spec reads, in order of first appearance."""
    out: list[str] = []

    def walk(n):
        if n.get("op") in LEAF_OPS:
            if n["factor"] not in out:
                out.append(n["factor"])
        for t in n.get("terms", []):
            walk(t["node"])
        for m in n.get("nodes", []):
            walk(m)
    walk(normalize_spec(spec)["score"])
    return out


# ── Propagation ──────────────────────────────────────────────────────────────

def _pct(s: pd.Series, dates: pd.Series) -> pd.Series:
    return s.groupby(dates).rank(pct=True).fillna(0.5)


def _z(s: pd.Series, dates: pd.Series) -> pd.Series:
    g = s.groupby(dates)
    sd = g.transform("std").replace(0.0, np.nan)
    return ((s - g.transform("mean")) / sd).fillna(0.0)


def _eval(node: dict, df: pd.DataFrame, dates: pd.Series) -> pd.Series:
    op = node["op"]
    if op in LEAF_OPS:
        raw = pd.to_numeric(df[node["factor"]], errors="coerce")
        if op == "percentile":
            return _pct(raw, dates)
        if op == "inverse_percentile":
            return 1.0 - _pct(raw, dates)
        if op == "zscore":
            return _z(raw, dates)
        if op == "clipped_zscore":
            c = float(node.get("clip", 3.0))
            return _z(raw, dates).clip(-c, c)
        base = _pct(raw, dates) if node.get("on", "percentile") == "percentile" else raw
        if op == "threshold":
            if "above" in node:
                out = (base > node["above"]).astype(float)
            else:
                out = (base < node["below"]).astype(float)
            return out.where(base.notna(), 0.0)
        pts = node["points"]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        vals = np.interp(base.to_numpy(dtype=float), xs, ys)
        out = pd.Series(vals, index=df.index)
        return out.where(base.notna(), float(np.mean(ys)))
    if op == "weighted_sum":
        acc = pd.Series(0.0, index=df.index)
        for t in node["terms"]:
            acc = acc + float(t["weight"]) * _eval(t["node"], df, dates)
        return acc
    parts = [_eval(n, df, dates) for n in node["nodes"]]
    if op == "multiply":
        acc = parts[0]
        for p in parts[1:]:
            acc = acc * p
        return acc
    stack = pd.concat(parts, axis=1)
    return stack.min(axis=1) if op == "min" else stack.max(axis=1)


def apply_scoring_spec(df: pd.DataFrame, spec: dict, date_col: str = "date",
                       factor_cols: Iterable[str] | None = None,
                       out_col: str | None = None) -> pd.DataFrame:
    """
    Deterministic propagation. Returns a copy of `df` with one new column,
    bounded in [0, 1], computed independently inside each date cross section.
    `factor_cols` defaults to every column the spec references (validated).
    """
    fc = list(factor_cols) if factor_cols is not None else spec_factors(spec)
    spec = validate_scoring_spec(spec, fc)
    missing = [f for f in spec_factors(spec) if f not in df.columns]
    if missing:
        raise SpecError(f"DataFrame lacks factor columns: {missing}")
    out = df.copy()
    dates = out[date_col]
    raw = _eval(spec["score"], out, dates)
    tr = spec["output_transform"]
    if tr == "percentile":
        score = raw.groupby(dates).rank(pct=True)
    elif tr == "minmax":
        g = raw.groupby(dates)
        lo, hi = g.transform("min"), g.transform("max")
        score = ((raw - lo) / (hi - lo).replace(0.0, np.nan)).fillna(0.5)
    else:
        score = raw.clip(0.0, 1.0)
    col = out_col or spec["model_name"]
    out[col] = score.astype(float).clip(0.0, 1.0)
    return out


# ── Holdings helpers ─────────────────────────────────────────────────────────

def add_weight_columns(df: pd.DataFrame, date_col: str, id_col: str,
                       fund_col: str, bench_col: str | None = None,
                       halflife: int = 3) -> pd.DataFrame:
    """
    Fill in the standard holdings columns from what the file has. active =
    fund − benchmark; the EWM columns are exponentially-weighted means of the
    fund / active weight over each name's history (halflife in months), so a
    name only just added carries less evidence than a long-standing position.
    """
    d = df.sort_values([id_col, date_col]).copy()
    d["fund_weight"] = pd.to_numeric(d[fund_col], errors="coerce").fillna(0.0)
    if bench_col and bench_col in d.columns:
        d["benchmark_weight"] = pd.to_numeric(d[bench_col], errors="coerce").fillna(0.0)
    elif "benchmark_weight" not in d.columns:
        d["benchmark_weight"] = np.nan
    d["active_weight"] = d["fund_weight"] - d["benchmark_weight"].fillna(0.0)
    g = d.groupby(id_col, sort=False)
    d["ewm_abs_weight"] = g["fund_weight"].transform(
        lambda s: s.ewm(halflife=halflife, min_periods=1).mean())
    d["ewm_active_weight"] = g["active_weight"].transform(
        lambda s: s.ewm(halflife=halflife, min_periods=1).mean())
    return d.sort_index()


def snapshot(df: pd.DataFrame, date_col: str = "date") -> pd.Timestamp:
    """Latest date with any non-zero fund weight."""
    held = df.loc[pd.to_numeric(df["fund_weight"], errors="coerce").fillna(0) > 0, date_col]
    if held.empty:
        raise ValueError("no date has a non-zero fund weight")
    return pd.to_datetime(held).max()


def evaluate(df: pd.DataFrame, score_col: str, date_col: str = "date",
             min_names: int = 20) -> pd.DataFrame:
    """
    Where holdings exist historically: per date, AUC of the score for the
    held flag and the Spearman correlation of the score with fund weight among
    held names. The score never saw these columns, so this is a fair test.
    """
    from sklearn.metrics import roc_auc_score
    rows = []
    for dt, g in df.groupby(date_col):
        w = pd.to_numeric(g["fund_weight"], errors="coerce").fillna(0.0)
        held = (w > 0).astype(int)
        s = g[score_col]
        ok = s.notna()
        if ok.sum() < min_names or held[ok].nunique() < 2:
            continue
        auc = float(roc_auc_score(held[ok], s[ok]))
        h = ok & (w > 0)
        rho = float(s[h].corr(w[h], method="spearman")) if h.sum() >= 5 else np.nan
        rows.append({"date": dt, "auc": auc, "rho_weight": rho,
                     "n": int(ok.sum()), "n_held": int(held[ok].sum())})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


# ── Stage A: the LLM pass ────────────────────────────────────────────────────

GRAMMAR_DOC = __doc__[__doc__.index("Scoring language (JSON)"):]


def _snapshot_table(df: pd.DataFrame, date_col: str, id_col: str, factor_cols: list,
                    n_held: int = 60, n_bench: int = 40) -> str:
    """The current cross section as text: every held name (largest first,
    capped) plus a sample of benchmark names the fund does not own."""
    d = df.copy()
    d["_w"] = pd.to_numeric(d["fund_weight"], errors="coerce").fillna(0.0)
    held = d[d["_w"] > 0].sort_values("_w", ascending=False).head(n_held)
    not_held = d[d["_w"] <= 0]
    if "benchmark_weight" in d and d["benchmark_weight"].notna().any():
        not_held = not_held.sort_values("benchmark_weight", ascending=False)
    not_held = not_held.head(n_bench)
    wcols = [c for c in ("fund_weight", "benchmark_weight", "active_weight",
                         "ewm_abs_weight", "ewm_active_weight") if c in d.columns]
    cols = [id_col] + wcols + list(factor_cols)

    def fmt(sub, title):
        t = sub[cols].copy()
        for c in wcols:
            t[c] = pd.to_numeric(t[c], errors="coerce").map(lambda v: f"{v:.2%}" if pd.notna(v) else "")
        for c in factor_cols:
            t[c] = pd.to_numeric(t[c], errors="coerce").map(lambda v: f"{v:.3g}" if pd.notna(v) else "")
        return f"{title} ({len(t)} rows)\n" + t.to_csv(index=False)
    return (fmt(held, "HELD NAMES, largest fund weight first")
            + "\n" + fmt(not_held, "NOT HELD — benchmark names the fund passed on"))


def _extract_json(text: str) -> dict:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    blob = m.group(1) if m else text[text.index("{"): text.rindex("}") + 1]
    return json.loads(blob)


def infer_philosophy(df: pd.DataFrame, date_col: str, id_col: str, factor_cols: list,
                     key: str, starter_philosophy: str | None = None,
                     fund_info: str | None = None, factor_defs: dict | None = None,
                     model: str = "claude-opus-5", snapshot_date=None) -> tuple[str, dict, str]:
    """
    One LLM pass over the current cross section. Returns
    (philosophy text, validated scoring spec, raw model response).
    The spec is validated here; an invalid one is sent back to the model once
    with the errors, then raised if still invalid.
    """
    import anthropic
    snap = pd.to_datetime(snapshot_date) if snapshot_date is not None else snapshot(df, date_col)
    cs = df[pd.to_datetime(df[date_col]) == snap]
    defs = {**FACTOR_DEFS, **(factor_defs or {})}
    def_lines = "\n".join(f"  {f}: {defs.get(f, '(no definition supplied)')}" for f in factor_cols)
    system = (
        "You are a senior equity portfolio analyst. From one snapshot of a fund's "
        "holdings — weights alongside factor values, next to benchmark names the "
        "fund does not own — infer the latent investment philosophy that connects "
        "the positions, then compile it into an executable scoring rule.\n\n"
        "Reason freely about economics (secular growth, moats, unit economics, "
        "growth persistence, capital intensity, fundamental momentum, valuation "
        "discipline, balance-sheet fragility). The written philosophy may be richer "
        "than the data. The scoring rule may NOT be: it must use only the approved "
        "factor columns listed below and only the operations of the scoring "
        "language. Never reference fund_weight, benchmark_weight, active_weight, "
        "ewm_abs_weight or ewm_active_weight in the rule — they are evidence, not "
        "inputs. Prefer a rule a reader could explain in a paragraph: a handful of "
        "percentile terms, an interaction where the philosophy is genuinely "
        "conditional ('growth only when profitable'), a threshold or piecewise "
        "where there is a real cut-off. Weights should reflect what the holdings "
        "actually show, not a generic factor model.\n\n"
        f"APPROVED FACTOR COLUMNS (higher = more of the named thing):\n{def_lines}\n\n"
        f"SCORING LANGUAGE\n{GRAMMAR_DOC}\n"
        "RESPONSE FORMAT — exactly two parts:\n"
        "1. A section headed PHILOSOPHY: 2–4 short paragraphs, plain English, "
        "what the manager appears to want and to avoid, and where they trade off.\n"
        "2. A section headed SPEC: a single ```json code block containing the "
        "scoring specification, model_name \"hold_prop_llm\", output_transform "
        "\"percentile\". No prose inside the block."
    )
    user = ""
    if fund_info:
        user += f"FUND / MANAGER INFORMATION\n{fund_info}\n\n"
    if starter_philosophy:
        user += ("STARTER PHILOSOPHY (a prior to test against the holdings, "
                 f"not an answer to repeat):\n{starter_philosophy}\n\n")
    user += f"SNAPSHOT DATE {snap:%Y-%m-%d}\n\n" + _snapshot_table(cs, date_col, id_col, factor_cols)
    client = anthropic.Anthropic(api_key=key)
    msgs = [{"role": "user", "content": user}]
    last_err = None
    for attempt in range(2):
        resp = client.messages.create(model=model, max_tokens=4000, system=system,
                                      messages=msgs, output_config={"effort": "high"})
        text = "".join(getattr(b, "text", "") for b in resp.content)
        try:
            spec = validate_scoring_spec(_extract_json(text), factor_cols)
            phil = text.split("PHILOSOPHY", 1)[-1].split("SPEC", 1)[0].strip(" :\n#*")
            return phil, spec, text
        except (SpecError, ValueError, json.JSONDecodeError) as e:
            last_err = e
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"The spec was rejected:\n{e}\n\n"
                                                 "Return both parts again with a valid spec."}]
    raise SpecError(f"The model did not produce a valid spec in two attempts: {last_err}")


def infer_hold_propensity(df: pd.DataFrame, date_col: str, id_col: str, factor_cols: list,
                          key: str, fund_weight_col: str = "fund_weight",
                          benchmark_weight_col: str | None = "benchmark_weight",
                          starter_philosophy: str | None = None, fund_info: str | None = None,
                          **kw) -> tuple[pd.DataFrame, str, dict]:
    """The one-call interface: standardise weights, infer, validate, propagate."""
    d = df if {"fund_weight", "ewm_abs_weight"} <= set(df.columns) else add_weight_columns(
        df, date_col, id_col, fund_weight_col, benchmark_weight_col)
    phil, spec, _ = infer_philosophy(d, date_col, id_col, factor_cols, key,
                                     starter_philosophy, fund_info, **kw)
    return apply_scoring_spec(d, spec, date_col, factor_cols), phil, spec


# ── Demo fund for the shipped panel ──────────────────────────────────────────

# A hidden philosophy the demo fund follows, so the pipeline can be judged on
# whether the LLM recovers it from the holdings alone: profitable growth with
# an aversion to volatility.
DEMO_RULE = {
    "model_name": "demo_true_score",
    "score": {"op": "weighted_sum", "terms": [
        {"weight": 0.40, "node": {"op": "percentile", "factor": "gpa"}},
        {"weight": 0.30, "node": {"op": "percentile", "factor": "rev_gr_1y"}},
        {"weight": 0.30, "node": {"op": "inverse_percentile", "factor": "vol_12m"}},
    ]},
    "output_transform": "percentile",
}
DEMO_PHILOSOPHY = ("Profitable growers, not too volatile: 40% gross profitability, "
                   "30% revenue growth, 30% low realised volatility; top 40 names "
                   "each month, weighted by score × market cap.")


def demo_holdings(panel: pd.DataFrame, n_hold: int = 40, months: int = 36,
                  seed: int = 7) -> pd.DataFrame:
    """
    Attach synthetic holdings to the last `months` of a panel: the fund holds
    the top `n_hold` names by DEMO_RULE each month (with a little noise so the
    rule is a strong tendency, not a hard cut), weighted by score × cap;
    benchmark = cap-weighted universe. Returns date, ticker, the weight
    columns and demo_true_score for the months covered.
    """
    d = panel.sort_values("date")
    dates = sorted(d["date"].unique())[-months:]
    d = d[d["date"].isin(dates)].copy()
    d = apply_scoring_spec(d, DEMO_RULE, "date")
    rng = np.random.default_rng(seed)
    d["_noisy"] = d["demo_true_score"] + rng.normal(0, 0.08, len(d))
    cap = np.exp(pd.to_numeric(d["log_mcap"], errors="coerce")).fillna(0.0)
    d["benchmark_weight"] = cap / cap.groupby(d["date"]).transform("sum")
    rank = d.groupby("date")["_noisy"].rank(ascending=False, method="first")
    held = rank <= n_hold
    fw = (d["demo_true_score"] * cap).where(held, 0.0)
    d["fund_weight"] = fw / fw.groupby(d["date"]).transform("sum")
    out = add_weight_columns(d, "date", "ticker", "fund_weight", "benchmark_weight")
    return out[["date", "ticker", "fund_weight", "benchmark_weight", "active_weight",
                "ewm_abs_weight", "ewm_active_weight", "demo_true_score"]]
