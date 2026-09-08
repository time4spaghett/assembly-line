"""
One-off builder for the base data panel shipped with the app.

Distills the base-equity-edge Sharadar caches into a single compact parquet:
one row per (date, ticker) at monthly frequency, with sector/industry metadata,
~24 raw (untransformed) features, and forward simple returns at 1/3/6/12 months.

Raw feature values are stored (not ranks) so the app's transform step
(rank / zscore / raw) is meaningful. Fundamentals are joined point-in-time
via merge_asof on Sharadar datekey (filing availability date).

Usage:
    python panel_build.py --cache <sharadar-cache-dir> --sp500 <sp500.parquet> [--out data/base_panel.parquet]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("panel_build")

FWD_HORIZONS = {"fwd_1m": 1, "fwd_3m": 3, "fwd_6m": 6, "fwd_12m": 12}

# Enforced out-of-sample boundary: the shipped base panel never contains rows
# dated on/after this, so anything after it stays untouched holdout data.
OOS_CUTOFF = pd.Timestamp("2026-01-01")


def load_universe(cache: Path, sp500_path: Path) -> pd.DataFrame:
    """
    Point-in-time S&P 500 monthly membership.

    Sharadar SP500 table 'historical' rows are quarterly full-membership
    snapshots; 'current' is today's. For each month-end trading date we take
    the most recent snapshot at or before it (no look-ahead).
    """
    snap_dates = (pd.read_parquet(cache / "universe_monthly.parquet")["snap_date"]
                  .drop_duplicates().sort_values())
    sp = pd.read_parquet(sp500_path)
    sp = sp[sp["action"].isin(["historical", "current"])]
    sp["date"] = pd.to_datetime(sp["date"])
    snaps = {d: set(g["ticker"]) for d, g in sp.groupby("date")}
    sdates = sorted(snaps)

    rows = []
    for snap in snap_dates:
        idx = np.searchsorted(sdates, snap, side="right") - 1
        if idx < 0:
            continue
        members = snaps[sdates[idx]]
        rows.append(pd.DataFrame({"snap_date": snap, "ticker": sorted(members)}))
    uni = pd.concat(rows, ignore_index=True)
    uni["month"] = uni["snap_date"].dt.to_period("M")
    log.info("SP500 PIT universe: %d rows, %d months, %d unique tickers",
             len(uni), uni["month"].nunique(), uni["ticker"].nunique())
    return uni


def load_sep(cache: Path, tickers: set[str]) -> pd.DataFrame:
    """Daily adjusted closes for the universe tickers, read once."""
    sep = pd.read_parquet(cache / "sep.parquet")
    sep = sep[sep["ticker"].isin(tickers)].copy()
    sep["date"] = pd.to_datetime(sep["date"])
    sep = sep.drop_duplicates(["ticker", "date"], keep="last")
    log.info("SEP daily: %d rows, %d tickers", len(sep), sep["ticker"].nunique())
    return sep


def monthly_prices(sep: pd.DataFrame) -> pd.DataFrame:
    """Wide monthly close panel: index=month period, columns=ticker."""
    s = sep.copy()
    s["month"] = s["date"].dt.to_period("M")
    s = s.sort_values("date").groupby(["ticker", "month"], as_index=False).last()
    wide = s.pivot(index="month", columns="ticker", values="closeadj").sort_index()
    log.info("Monthly price panel: %d months x %d tickers", *wide.shape)
    return wide


def daily_risk_features(sep: pd.DataFrame) -> pd.DataFrame:
    """
    Risk block from daily closes, sampled at month-end — the price-based
    features that dominate the Gu–Kelly–Xiu importance rankings and that a
    fundamentals-only panel lacks:

      vol_1m      std of daily returns, trailing 21 trading days
      vol_12m     std of daily returns, trailing 252 days
      max_ret_1m  largest single-day return in the trailing month (lottery)
      beta_12m    rolling 252-day beta against the equal-weight universe

    Everything derives from closeadj, so all four are split-safe. Turnover,
    dollar volume and Amihud illiquidity would belong here too, but the cache
    carries no volume column. Net share issuance is deliberately left out:
    Sharadar `sharesbas` is as-reported, so a 2-for-1 split would read as
    +100% issuance and poison the feature.
    """
    wide = (sep.pivot(index="date", columns="ticker", values="closeadj")
            .sort_index().astype(np.float32))
    ret = wide.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)

    r252 = ret.rolling(252, min_periods=126)
    m252 = mkt.rolling(252, min_periods=126)
    feats = {
        "vol_1m": ret.rolling(21, min_periods=15).std(),
        "vol_12m": r252.std(),
        "max_ret_1m": ret.rolling(21, min_periods=15).max(),
        # beta = cov(r, mkt) / var(mkt), rolling: E[rm] − E[r]E[m] over var
        "beta_12m": (ret.mul(mkt, axis=0).rolling(252, min_periods=126).mean()
                     - r252.mean().mul(m252.mean(), axis=0)
                     ).div(m252.var(), axis=0),
    }
    out = None
    for name, pf in feats.items():
        me = pf.groupby(pf.index.to_period("M")).last()
        longf = me.stack().rename(name).reset_index()
        longf.columns = ["month", "ticker", name]
        out = longf if out is None else out.merge(longf, on=["month", "ticker"],
                                                  how="outer")
    log.info("Daily risk features: %d rows", len(out))
    return out


def price_features(px: pd.DataFrame) -> pd.DataFrame:
    """Momentum + forward returns from the monthly close panel, long format."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(px.where(px > 0))

    feats = {
        "ret_1m":   logp - logp.shift(1),
        "mom_3m":   logp - logp.shift(3),
        "mom_3_1":  logp.shift(1) - logp.shift(3),   # months 2–3, reversal month out
        "mom_6_1":  logp.shift(1) - logp.shift(6),
        "mom_12_1": logp.shift(1) - logp.shift(12),
        "high_52w": px / px.rolling(12, min_periods=6).max(),
    }
    for name, h in FWD_HORIZONS.items():
        feats[name] = px.shift(-h) / px - 1.0

    out = None
    for name, panel in feats.items():
        longf = panel.stack().rename(name).reset_index()
        longf.columns = ["month", "ticker", name]
        out = longf if out is None else out.merge(longf, on=["month", "ticker"], how="outer")
    log.info("Price features: %d rows", len(out))
    return out


def asof_fundamentals(sf1: pd.DataFrame, uni: pd.DataFrame, lag_days: int = 0,
                      suffix: str = "") -> pd.DataFrame:
    """PIT as-of join: latest filing with datekey <= snap_date - lag_days."""
    left = uni[["snap_date", "ticker"]].copy()
    left["asof"] = left["snap_date"] - pd.Timedelta(days=lag_days)
    left = left.sort_values("asof")
    right = sf1.sort_values("datekey")
    merged = pd.merge_asof(left, right, left_on="asof", right_on="datekey",
                           by="ticker", tolerance=pd.Timedelta(days=270))
    merged = merged.drop(columns=["asof", "datekey"])
    if suffix:
        value_cols = [c for c in merged.columns if c not in ("snap_date", "ticker")]
        merged = merged.rename(columns={c: c + suffix for c in value_cols})
    return merged


def jitter_panel(panel: pd.DataFrame, rel: float = 0.0005,
                 seed: int = 20260901) -> pd.DataFrame:
    """
    Multiplicative Gaussian noise layer over every numeric column.

    noise_i = value_i * eps_i,  eps_i ~ N(0, rel/3) clipped to [-rel, +rel]

    sigma is rel/3 so the stated bound is a true 3-sigma hard cap: ~99.7% of
    draws are untouched by the clip, and no term ever exceeds `rel` of the
    original value (default 0.05%). Seeded, so a given panel always jitters
    the same way. NaNs stay NaN; exact zeros stay zero (noise is relative).
    """
    rng = np.random.default_rng(seed)
    out = panel.copy()
    # Shave a few float32 ulps off the cap so the bound still holds *after* the
    # float64 -> float32 cast. The cast costs up to ~6e-8 in relative error, so
    # the margin has to be absolute (~5e-7); scaling the cap by (1 - 1e-6) would
    # only shave 5e-10 and the bound would still be breached.
    cap = rel - 4.0 * float(np.finfo(np.float32).eps)
    num_cols = [c for c in out.columns if pd.api.types.is_float_dtype(out[c])]
    for c in num_cols:
        v = out[c].to_numpy(dtype=np.float64)
        eps = np.clip(rng.normal(0.0, cap / 3.0, size=v.shape), -cap, cap)
        out[c] = (v + v * eps).astype(np.float32)
    log.info("Jittered %d numeric columns at +/-%.3f%% (seed %d)",
             len(num_cols), rel * 100, seed)
    return out


def build(cache: Path, sp500_path: Path, out: Path, jitter: bool = False,
          jitter_rel: float = 0.0005, seed: int = 20260901) -> None:
    uni = load_universe(cache, sp500_path)
    tickers = set(uni["ticker"].unique())

    # ── Price block ───────────────────────────────────────────────────────────
    sep = load_sep(cache, tickers)
    px = monthly_prices(sep)
    pf = price_features(px)
    panel = uni.merge(pf, on=["month", "ticker"], how="left")
    panel = panel.merge(daily_risk_features(sep), on=["month", "ticker"], how="left")
    del sep

    # ── Fundamentals: current + 1y-lagged for growth/trend features ───────────
    art = pd.read_parquet(cache / "sf1_art.parquet")
    arq = pd.read_parquet(cache / "sf1_arq.parquet")

    cur_a = asof_fundamentals(art, uni)                       # TTM current
    lag_a = asof_fundamentals(art, uni, lag_days=365, suffix="_l1")
    lag3_a = asof_fundamentals(art[["ticker", "datekey", "revenue"]], uni,
                               lag_days=3 * 365, suffix="_l3")
    cur_q = asof_fundamentals(arq[["ticker", "datekey", "sharesbas", "assets",
                                   "equity", "debt"]], uni, suffix="_q")
    lag_q = asof_fundamentals(arq[["ticker", "datekey", "assets"]], uni,
                              lag_days=365, suffix="_q_l1")

    f = uni[["snap_date", "ticker", "month"]].copy()
    for block in (cur_a, lag_a, lag3_a, cur_q, lag_q):
        f = f.merge(block, on=["snap_date", "ticker"], how="left")

    # closeadj at snapshot for market cap
    close_long = px.stack().rename("close").reset_index()
    close_long.columns = ["month", "ticker", "close"]
    f = f.merge(close_long, on=["month", "ticker"], how="left")

    def div(num, den):
        den = den.where(den.abs() > 1e-6)
        return num / den

    mcap = f["close"] * f["sharesbas_q"]
    fcf = f["ncfo"] - f["capex"].abs()

    feat = pd.DataFrame({"snap_date": f["snap_date"], "ticker": f["ticker"]})
    # value
    feat["btm"] = div(f["equity_q"], mcap)
    feat["earn_yield"] = div(f["netinc"], mcap)
    feat["fcf_yield"] = div(fcf, mcap)
    feat["sales_yield"] = div(f["revenue"], mcap)
    # quality
    feat["roa"] = div(f["netinc"], f["assets"])
    feat["roe"] = div(f["netinc"], f["equity"])
    feat["gross_margin"] = div(f["gp"], f["revenue"])
    feat["op_margin"] = div(f["ebit"], f["revenue"])
    feat["gpa"] = div(f["gp"], f["assets"])
    feat["fcf_margin"] = div(fcf, f["revenue"])
    # safety (higher raw value = more levered / more accruals; flip with negative weight)
    feat["leverage"] = div(f["debt_q"], f["assets_q"])
    feat["accruals"] = div(f["netinc"] - f["ncfo"], f["assets"])
    # growth
    feat["rev_gr_1y"] = div(f["revenue"], f["revenue_l1"]) - 1.0
    rev3 = div(f["revenue"], f["revenue_l3"])
    feat["rev_gr_3y"] = rev3.where(rev3 > 0) ** (1 / 3) - 1.0
    feat["asset_gr"] = div(f["assets_q"], f["assets_q_l1"]) - 1.0
    # fundamental momentum
    feat["earn_mom"] = div(f["netinc"] - f["netinc_l1"], f["assets"])
    feat["margin_mom"] = div(f["gp"], f["revenue"]) - div(f["gp_l1"], f["revenue_l1"])
    feat["rev_accel"] = feat["rev_gr_1y"] - feat["rev_gr_3y"]
    # size
    feat["log_mcap"] = np.log(mcap.where(mcap > 0))

    panel = panel.merge(feat, on=["snap_date", "ticker"], how="left")

    # ── Sector / industry metadata ────────────────────────────────────────────
    t = pd.read_parquet(cache / "tickers.parquet")
    t = t[t["table"] == "SEP"][["ticker", "sector", "industry"]].drop_duplicates("ticker")
    panel = panel.merge(t, on="ticker", how="left")
    panel[["sector", "industry"]] = panel[["sector", "industry"]].fillna("Unknown")

    # ── Finalize ──────────────────────────────────────────────────────────────
    panel = panel.rename(columns={"snap_date": "date"}).drop(columns=["month"])
    panel = panel[panel["date"] < OOS_CUTOFF]

    covered = panel["ret_1m"].notna() | panel["fwd_1m"].notna()
    log.info("Price coverage: %.1f%% of universe rows (%d tickers uncovered)",
             100 * covered.mean(),
             panel.loc[~covered, "ticker"].nunique())
    data_cols = [c for c in panel.columns if c not in ("date", "ticker", "sector", "industry")]
    panel = panel.dropna(subset=data_cols, how="all")
    meta_cols = ["date", "ticker", "sector", "industry"]
    fwd_cols = list(FWD_HORIZONS)
    feature_cols = [c for c in panel.columns if c not in meta_cols + fwd_cols]
    panel = panel[meta_cols + feature_cols + fwd_cols]
    for c in feature_cols + fwd_cols:
        panel[c] = panel[c].astype(np.float32)

    if jitter:
        panel = jitter_panel(panel, rel=jitter_rel, seed=seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    log.info("Wrote %s: %d rows x %d cols (%.1f MB)", out, len(panel), panel.shape[1],
             out.stat().st_size / 1e6)
    log.info("Features: %s", feature_cols)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # No defaults: the source caches live outside this repo, wherever the
    # local Sharadar dumps are kept.
    ap.add_argument("--cache", required=True,
                    help="directory holding sep/sf1/tickers/universe parquets")
    ap.add_argument("--sp500", required=True,
                    help="path to the Sharadar SP500 actions parquet")
    ap.add_argument("--out", default="data/base_panel.parquet")
    ap.add_argument("--jitter", action="store_true",
                    help="apply the multiplicative Gaussian noise layer")
    ap.add_argument("--jitter-rel", type=float, default=0.0005,
                    help="hard cap per noise term, as a fraction (default 0.0005 = 0.05%%)")
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()
    build(Path(args.cache), Path(args.sp500), Path(args.out),
          jitter=args.jitter, jitter_rel=args.jitter_rel, seed=args.seed)
