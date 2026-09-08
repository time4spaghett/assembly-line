"""
Edge-testing engine: cross-sectional transforms, composite construction,
and IC / quantile analysis.

Math ported from base-equity-edge (analysis/ic.py, factors/zoo.py):
  - cross-sectional winsorize + percentile rank
  - Newey-West corrected t-stat on the IC series (Bartlett kernel)

Panel contract (long format, monthly):
  date, ticker, sector, industry, <feature cols...>, fwd_1m, fwd_3m, fwd_6m, fwd_12m
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

META_COLS = ["date", "ticker", "sector", "industry"]
FWD_COLS = {"fwd_1m": 1, "fwd_3m": 3, "fwd_6m": 6, "fwd_12m": 12}

TRANSFORMS = ["rank", "zscore", "raw"]

# Benchmark modes for the ntile charts
BENCH_EW = "Equal-weight universe"
BENCH_CAP = "Cap-weighted universe"
BENCH_COL = "Column"
BENCH_REF = "Reference"


@dataclass
class FeatureSpec:
    feature: str
    transform: str = "rank"   # rank | zscore | raw
    weight: float = 1.0


OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
       "!=": operator.ne, "==": operator.eq}


@dataclass
class Constraint:
    """
    A screen on raw feature values, e.g. `roa > 0` or `roa > asset_gr`.

    `right` is a number if it parses as one, otherwise the name of another
    feature — so both a threshold and a feature-vs-feature relationship are
    expressible. Applied per (date, name) on *raw* values, before any transform.
    """
    left: str
    op: str = ">"
    right: str = "0"


COMBINE_SUM = "weighted sum"
COMBINE_PRODUCT = "rank product"


@dataclass
class EdgeSpec:
    features: list[FeatureSpec] = field(default_factory=list)
    sector_neutral: bool = False   # transform within (date, sector) instead of (date)
    constraints: list[Constraint] = field(default_factory=list)
    combine: str = COMBINE_SUM


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in META_COLS and c not in FWD_COLS]


def _winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    if s.notna().sum() < 20:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


def _transform_group(s: pd.Series, how: str) -> pd.Series:
    if how == "rank":
        n = s.notna().sum()
        if n < 2:
            return s * np.nan
        return (s.rank() - 1.0) / (n - 1.0)          # [0, 1]
    w = _winsorize(s)
    if how == "zscore":
        sd = w.std()
        return (w - w.mean()) / sd if sd and sd > 0 else w * np.nan
    return w                                          # raw (winsorized)


def _as_number(text: str):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def constraint_mask(panel: pd.DataFrame,
                    constraints: list[Constraint]) -> pd.Series:
    """
    Boolean eligibility mask over panel rows. True = passes every constraint.

    A row with NaN on either side of a comparison fails: we cannot verify the
    condition, so the name is screened out rather than silently admitted.
    """
    mask = pd.Series(True, index=panel.index)
    for c in constraints or []:
        if not c.left or c.left not in panel.columns or c.op not in OPS:
            continue
        lhs = panel[c.left]
        num = _as_number(c.right)
        if num is not None:
            rhs = num
            valid = lhs.notna()
        elif c.right in panel.columns:
            rhs = panel[c.right]
            valid = lhs.notna() & rhs.notna()
        else:
            continue                       # unresolvable reference: ignore
        mask &= OPS[c.op](lhs, rhs).fillna(False) & valid
    return mask


def unresolved_constraints(panel: pd.DataFrame,
                           constraints: list[Constraint]) -> list[tuple]:
    """
    Constraints that cannot be evaluated, as (constraint, reason).

    These are skipped rather than raising, so they must be reported: a typo'd
    feature name would otherwise silently apply no screen at all.
    """
    bad = []
    for c in constraints or []:
        if not c.left or c.left not in panel.columns:
            bad.append((c, f"no feature named '{c.left}'"))
        elif c.op not in OPS:
            bad.append((c, f"unknown operator '{c.op}'"))
        elif _as_number(c.right) is None and c.right not in panel.columns:
            bad.append((c, f"'{c.right}' is neither a number nor a feature"))
    return bad


def build_composite(panel: pd.DataFrame, spec: EdgeSpec) -> pd.Series:
    """Weighted sum of cross-sectionally transformed features, re-ranked to [0,1].

    Constraints are applied *first*, so ineligible names are absent from the
    cross-section the ranks are computed over — a screen changes the peer group,
    it does not merely mark rows after the fact.
    """
    keys = ["date", "sector"] if spec.sector_neutral else ["date"]
    eligible = constraint_mask(panel, spec.constraints)
    parts = []
    for fs in spec.features:
        if fs.weight == 0 or fs.feature not in panel.columns:
            continue
        col = panel[fs.feature].where(eligible)
        t = col.groupby([panel[k] for k in keys], observed=True).transform(
            _transform_group, how=fs.transform)
        parts.append(fs.weight * t)
    if not parts:
        raise ValueError("Edge spec has no active features (all weights zero?).")

    if spec.combine == COMBINE_PRODUCT:
        # Layered composition: rank each leg, multiply. A name has to score on
        # every layer, so a zero on one is not offset by a high score on
        # another — the behaviour a weighted sum cannot express. Weight acts as
        # an exponent; a negative weight inverts the leg to (1 - rank).
        raw = None
        for fs, t in zip([f for f in spec.features
                          if f.weight != 0 and f.feature in panel.columns], parts):
            leg = t / fs.weight                      # undo the weight scaling
            leg = leg.clip(0, 1)
            if fs.weight < 0:
                leg = 1.0 - leg
            leg = leg.clip(lower=1e-6) ** abs(fs.weight)
            raw = leg if raw is None else raw * leg
    else:
        raw = sum(parts)
    # re-rank the combined score within each date so quantile cuts are stable
    comp = raw.groupby(panel["date"]).transform(
        lambda s: (s.rank() - 1.0) / max(s.notna().sum() - 1.0, 1.0))
    comp[raw.isna()] = np.nan
    return comp


# ── Newey-West t-stat (verbatim port from base-equity-edge/analysis/ic.py) ────

def newey_west_tstat(ic: np.ndarray, lag: int) -> float:
    valid = ic[np.isfinite(ic)]
    n = len(valid)
    if n < max(lag + 2, 10):
        return np.nan
    mu = float(np.mean(valid))
    centered = valid - mu
    gamma = [float(np.mean(centered ** 2))]
    for j in range(1, lag + 1):
        if j >= n:
            break
        gamma.append(float(np.mean(centered[j:] * centered[:-j])))
    var_nw = gamma[0]
    for j in range(1, len(gamma)):
        w = 1.0 - j / (lag + 1)   # Bartlett kernel
        var_nw += 2.0 * w * gamma[j]
    var_nw = max(var_nw, 1e-12)
    se = (var_nw / n) ** 0.5
    return mu / se if se > 0 else np.nan


# ── Analysis ──────────────────────────────────────────────────────────────────

def _spearman(df: pd.DataFrame, a: str, b: str, min_n: int = 30) -> float:
    sub = df[[a, b]].dropna()
    if len(sub) < min_n:
        return np.nan
    ra, rb = sub[a].rank(), sub[b].rank()
    if ra.std() < 1e-10 or rb.std() < 1e-10:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def ic_analysis(panel: pd.DataFrame, comp: pd.Series, horizon: str,
                min_n: int = 30) -> dict:
    """`min_n`: fewest names a month needs to contribute an IC. 30 suits the
    full cross-section; a single sector's coverage wants something like 10."""
    df = pd.DataFrame({"date": panel["date"], "comp": comp, "fwd": panel[horizon]})
    ic = df.groupby("date")[["comp", "fwd"]].apply(_spearman, "comp", "fwd", min_n)
    ic = ic.dropna()
    months = FWD_COLS[horizon]
    valid = ic.to_numpy()
    mean_ic = float(np.mean(valid)) if len(valid) else np.nan
    sd = float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan
    return {
        "ic_series": ic,
        "mean_ic": mean_ic,
        "ir": mean_ic / sd if sd and sd > 0 else np.nan,
        "t_stat": newey_west_tstat(valid, max(months - 1, 0)),
        "n_months": len(valid),
    }


def _h_forward(monthly: pd.Series, h: int) -> pd.Series:
    """
    Compound a monthly forward-return series into h-month forward returns.

    monthly[t] is the return over the month after t, so the h-month forward
    return at t compounds monthly[t..t+h-1]. Needed so a benchmark is measured
    the same way as the ntile bars (arithmetic mean of overlapping h-month
    returns) instead of being annualized off a 1-month base.
    """
    if h <= 1:
        return monthly
    lg = np.log1p(monthly.astype(float))
    fwd = lg[::-1].rolling(h).sum()[::-1]
    return np.expm1(fwd)


def benchmark_options(panel: pd.DataFrame) -> list[str]:
    """Benchmark choices available for this panel."""
    opts = [BENCH_EW]
    if "log_mcap" in panel.columns:
        opts.append(BENCH_CAP)
    return opts


def quantile_analysis(panel: pd.DataFrame, comp: pd.Series, horizon: str,
                      n_q: int = 5, bench_mode: str = BENCH_EW,
                      bench_col: str | None = None,
                      bench_series: pd.Series | None = None) -> dict:
    months = FWD_COLS[horizon]
    df = pd.DataFrame({"date": panel["date"], "comp": comp, "fwd": panel[horizon],
                       "fwd1": panel["fwd_1m"]})
    # benchmark inputs, carried alongside so they survive the same row filtering
    if bench_mode == BENCH_CAP and "log_mcap" in panel.columns:
        df["bw"] = np.exp(panel["log_mcap"].astype(float))
    elif bench_mode == BENCH_COL and bench_col and bench_col in panel.columns:
        df["bcol"] = panel[bench_col].astype(float)
    df = df.dropna(subset=["comp"])
    df["q"] = df.groupby("date")["comp"].transform(
        lambda s: pd.qcut(s, n_q, labels=False, duplicates="drop") + 1
        if s.notna().sum() >= n_q * 5 else pd.Series(np.nan, index=s.index))
    df = df.dropna(subset=["q"])
    df["q"] = df["q"].astype(int)

    # mean fwd return per quantile per date, then averaged across dates
    by_dq = df.groupby(["date", "q"])["fwd"].mean().unstack()
    q_mean = by_dq.mean()
    q_ann = (1.0 + q_mean) ** (12.0 / months) - 1.0

    # per-quantile monthly (non-overlapping 1m fwd) return series → cumulative curves
    by_dq1 = df.groupby(["date", "q"])["fwd1"].mean().unstack()
    q_cum = (1.0 + by_dq1.dropna(how="all")).cumprod()

    # Benchmark. Default is the equal-weight universe of *scored* names, so the
    # line sits inside the ntile fan rather than measuring a different universe.
    def _wmean(g: pd.DataFrame, col: str) -> float:
        w = g["bw"]
        m = w.notna() & g[col].notna()
        return float(np.average(g.loc[m, col], weights=w[m])) if m.any() else np.nan

    # `bench` is always a per-month series (for the cumulative curve); `bench_h`
    # is whatever the annualized figure should be derived from, together with the
    # number of months it spans — these differ, and conflating them silently
    # understates a column benchmark by the horizon multiple.
    if bench_series is not None:                 # externally supplied series
        # a forward 1-month return per date; compounded to the horizon so the
        # annualized figure matches the ntile bars' convention exactly
        bench = bench_series.reindex(q_cum.index)
        bench_h = _h_forward(bench, months)
        bench_span = months
    elif "bw" in df.columns:                     # cap-weighted universe
        bench = df.groupby("date")[["bw", "fwd1"]].apply(_wmean, "fwd1")
        bench_h = df.groupby("date")[["bw", "fwd"]].apply(_wmean, "fwd")
        bench_span = months
    elif "bcol" in df.columns:                   # user-supplied benchmark column
        # read as a per-month return, then compounded to the horizon like above
        bench = df.groupby("date")["bcol"].mean()
        bench_h = _h_forward(bench.reindex(q_cum.index), months)
        bench_span = months
    else:                                        # equal-weight universe
        bench = df.groupby("date")["fwd1"].mean()
        bench_h = df.groupby("date")["fwd"].mean()
        bench_span = months
    bench = bench.reindex(q_cum.index)
    bench_cum = (1.0 + bench).cumprod()
    bench_ann = (1.0 + float(bench_h.mean())) ** (12.0 / bench_span) - 1.0

    # long-short series on non-overlapping 1m returns (honest cumulative curve)
    ls = (by_dq1[n_q] - by_dq1[1]).dropna() if n_q in by_dq1 and 1 in by_dq1 else pd.Series(dtype=float)
    ls_cum = (1.0 + ls).cumprod()
    ann_ret = (1.0 + ls.mean()) ** 12 - 1.0 if len(ls) else np.nan
    ann_vol = ls.std() * np.sqrt(12) if len(ls) > 1 else np.nan
    def _cagr(curve: pd.Series) -> float:
        c = curve.dropna()
        if len(c) < 2:
            return np.nan
        yrs = (c.index[-1] - c.index[0]).days / 365.25
        return float(c.iloc[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else np.nan

    return {
        "q_cagr": pd.Series({q: _cagr(q_cum[q]) for q in q_cum.columns}),
        "bench_cagr": _cagr(bench_cum),
        "bench_cum": bench_cum,          # equal-weight universe, cumulative
        "bench_ann": bench_ann,          # equal-weight universe, annualized
        "q_cum": q_cum,                  # DataFrame: cumulative curve per quantile
        "q_ann_returns": q_ann,          # Series indexed by quantile 1..n_q
        "spread_ann": float(q_ann.iloc[-1] - q_ann.iloc[0]) if len(q_ann) > 1 else np.nan,
        "ls_series": ls,
        "ls_cum": ls_cum,
        "ls_sharpe": ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan,
        "ls_ann_ret": ann_ret,
        "ls_ann_vol": ann_vol,
    }


# ── In-sample vs holdout consistency ─────────────────────────────────────────
# Fixed rules, evaluated the same way every time. An LLM may narrate the result
# but never decides it: a verdict that shifts between runs on identical numbers
# is worse than no verdict, because it invites re-rolling until it reads well.

T_SIGNIFICANT = 2.0      # |Newey-West t| treated as holding up
RETENTION_GOOD = 0.50    # holdout keeps at least half the in-sample effect
RETENTION_WEAK = 0.20    # below this the effect has essentially not survived
MONOTONE_MIN = 0.80      # rank corr between ntile order and ntile return


def _ratio(new: float, old: float, floor: float) -> float:
    """
    Fraction of the in-sample effect retained; NaN when there is none to keep.

    `floor` is the smallest in-sample effect worth dividing by. Without it a
    near-zero denominator produces headline nonsense — an in-sample IC of 2e-7
    reported a "-13695% retention" during testing.
    """
    if old is None or new is None or not np.isfinite(old) or not np.isfinite(new):
        return np.nan
    return new / old if abs(old) >= floor else np.nan


def _monotonicity(q_ann: pd.Series) -> float:
    """Spearman correlation between ntile order and ntile return."""
    if len(q_ann) < 3:
        return np.nan
    a = pd.Series(range(len(q_ann))).rank()
    b = pd.Series(q_ann.to_numpy()).rank()
    return float(np.corrcoef(a, b)[0, 1])


def consistency_checks(res_is: dict, res_oos: dict) -> dict:
    """
    Compare an in-sample result against its holdout on fixed criteria.

    Returns the individual checks plus a verdict. Sign agreement is treated as
    the gate: an effect that reverses out of sample is not a weaker finding, it
    is a different one, and no amount of retained magnitude redeems it.
    """
    ic_is, ic_oos = res_is["mean_ic"], res_oos["mean_ic"]
    sp_is, sp_oos = res_is["spread_ann"], res_oos["spread_ann"]
    sr_is, sr_oos = res_is["ls_sharpe"], res_oos["ls_sharpe"]

    ic_sign = bool(np.sign(ic_is) == np.sign(ic_oos) and np.isfinite(ic_oos))
    sp_sign = bool(np.sign(sp_is) == np.sign(sp_oos) and np.isfinite(sp_oos))
    # floors: an IC below 0.001 or a Sharpe below 0.05 is no effect to retain
    ic_ret, sr_ret = _ratio(ic_oos, ic_is, 1e-3), _ratio(sr_oos, sr_is, 0.05)
    mono_is, mono_oos = _monotonicity(res_is["q_ann_returns"]), _monotonicity(res_oos["q_ann_returns"])

    if not (ic_sign and sp_sign):
        verdict, headline = "inconsistent", "The effect reverses out of sample."
    elif np.isfinite(ic_ret) and ic_ret >= RETENTION_GOOD:
        verdict, headline = "consistent", "The effect holds out of sample."
    elif np.isfinite(ic_ret) and ic_ret >= RETENTION_WEAK:
        verdict, headline = "partial", "The effect survives but weakens materially."
    else:
        verdict, headline = "weak", "Little of the effect survives out of sample."

    checks = [
        ("IC keeps its sign", ic_sign,
         f"{ic_is:+.4f} → {ic_oos:+.4f}"),
        ("Spread keeps its sign", sp_sign,
         f"{sp_is:+.1%} → {sp_oos:+.1%}"),
        ("IC retention ≥ 50%", bool(np.isfinite(ic_ret) and ic_ret >= RETENTION_GOOD),
         "—" if not np.isfinite(ic_ret) else f"{ic_ret:.0%}"),
        ("Sharpe retention ≥ 50%", bool(np.isfinite(sr_ret) and sr_ret >= RETENTION_GOOD),
         "—" if not np.isfinite(sr_ret) else f"{sr_ret:.0%}"),
        (f"Holdout |t| ≥ {T_SIGNIFICANT:g}",
         bool(np.isfinite(res_oos["t_stat"]) and abs(res_oos["t_stat"]) >= T_SIGNIFICANT),
         f"{res_oos['t_stat']:+.2f}"),
        ("Ntiles stay ordered", bool(np.isfinite(mono_oos) and mono_oos >= MONOTONE_MIN),
         "—" if not np.isfinite(mono_oos) else f"{mono_is:.2f} → {mono_oos:.2f}"),
    ]
    return {
        "verdict": verdict, "headline": headline,
        "checks": checks,
        "passed": sum(1 for _, ok, _ in checks if ok), "total": len(checks),
        "ic_retention": ic_ret, "sharpe_retention": sr_ret,
        "months_is": res_is["n_months"], "months_oos": res_oos["n_months"],
    }


def sector_ic(panel: pd.DataFrame, comp: pd.Series, horizon: str) -> pd.DataFrame:
    df = pd.DataFrame({"date": panel["date"], "sector": panel["sector"],
                       "comp": comp, "fwd": panel[horizon]})
    rows = []
    for sec, g in df.groupby("sector"):
        ic = g.groupby("date")[["comp", "fwd"]].apply(_spearman, "comp", "fwd", 15).dropna()
        if len(ic) < 12:
            continue
        arr = ic.to_numpy()
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        rows.append({"sector": sec, "mean_ic": mu,
                     "ir": mu / sd if sd > 0 else np.nan,
                     "t_stat": newey_west_tstat(arr, max(FWD_COLS[horizon] - 1, 0)),
                     "n_months": len(arr)})
    return (pd.DataFrame(rows).sort_values("mean_ic", ascending=False)
            if rows else pd.DataFrame(columns=["sector", "mean_ic", "ir", "t_stat", "n_months"]))


def screen_stats(panel: pd.DataFrame, spec: EdgeSpec) -> dict:
    """How much the constraints remove — surfaced so a screen can't quietly
    shrink the universe to a handful of names without the user noticing."""
    if not spec.constraints:
        return {}
    m = constraint_mask(panel, spec.constraints)
    per_date = m.groupby(panel["date"]).mean()
    names = panel.loc[m, "ticker"].nunique() if "ticker" in panel.columns else np.nan
    return {
        "rows_kept": int(m.sum()),
        "rows_total": int(len(m)),
        "frac_kept": float(m.mean()),
        "names_kept": int(names),
        "min_names_per_date": int(m.groupby(panel["date"]).sum().min()),
        "median_frac_per_date": float(per_date.median()),
    }


def run_edge(panel: pd.DataFrame, spec: EdgeSpec, horizon: str = "fwd_12m",
             n_q: int = 5, bench_mode: str = BENCH_EW,
             bench_col: str | None = None,
             bench_series: pd.Series | None = None, min_n: int = 30) -> dict:
    comp = build_composite(panel, spec)
    out = {"composite": comp, "screen": screen_stats(panel, spec)}
    out.update(ic_analysis(panel, comp, horizon, min_n=min_n))
    out.update(quantile_analysis(panel, comp, horizon, n_q, bench_mode, bench_col,
                                 bench_series))
    out["sector_ic"] = sector_ic(panel, comp, horizon)
    return out
