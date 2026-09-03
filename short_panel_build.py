"""
One-off builder for the short-risk factor panel.

Factor set ported from goldenhourlabs/risk-screen — the blowup-screen research
(2026-07): what precedes a <= -40% forward-12-month return in the large-cap
universe. Kept factors only; the families that research killed (accruals,
revenue collapse, plain leverage) are deliberately absent.

All features are stored raw and oriented so HIGHER = MORE short-risk, so a
positive weight in the app means "tilt toward the risky names" and the expected
IC against forward returns is negative.

    python short_panel_build.py --cache <sharadar-cache-dir> --sp500 <sp500.parquet>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from panel_build import (FWD_HORIZONS, OOS_CUTOFF, asof_fundamentals,
                         jitter_panel, load_universe, monthly_prices,
                         price_features)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("short_panel")


def _streak(flag: pd.Series) -> pd.Series:
    """Consecutive True count as of each row (per ticker, in filing order)."""
    grp = (~flag).cumsum()
    return flag.groupby(grp).cumsum().astype(np.float32)


def quarterly_streaks(arq: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, datekey): consecutive loss quarters and cash-burn quarters."""
    q = arq.sort_values(["ticker", "datekey"]).copy()
    out = []
    for _, g in q.groupby("ticker", sort=False):
        out.append(pd.DataFrame({
            "ticker": g["ticker"].values, "datekey": g["datekey"].values,
            "loss_streak_q": _streak(g["netinc"] < 0).values,
            "cashburn_streak_q": _streak(g["ncfo"] < 0).values,
        }))
    return pd.concat(out, ignore_index=True)


def realized_vol_63d(cache: Path, tickers: set) -> pd.DataFrame:
    """Annualized 63-day realized vol from daily closes, sampled monthly."""
    sep = pd.read_parquet(cache / "sep.parquet")
    sep = sep[sep["ticker"].isin(tickers)]
    px = sep.pivot_table(index="date", columns="ticker", values="closeadj",
                         aggfunc="last").sort_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(px.where(px > 0)).diff()
    vol = lr.rolling(63, min_periods=40).std() * np.sqrt(252)
    vol.index = pd.to_datetime(vol.index)
    monthly = vol.groupby(vol.index.to_period("M")).last()
    longf = monthly.stack().rename("vol_63d").reset_index()
    longf.columns = ["month", "ticker", "vol_63d"]
    return longf


def build(cache: Path, sp500_path: Path, out: Path, jitter: bool = True) -> None:
    uni = load_universe(cache, sp500_path)
    tickers = set(uni["ticker"].unique())

    px = monthly_prices(cache, tickers)
    pf = price_features(px)                      # momentum block + fwd returns
    panel = uni.merge(pf, on=["month", "ticker"], how="left")
    panel = panel.merge(realized_vol_63d(cache, tickers),
                        on=["month", "ticker"], how="left")

    arq = pd.read_parquet(cache / "sf1_arq.parquet")
    art = pd.read_parquet(cache / "sf1_art.parquet")

    streaks = quarterly_streaks(arq)
    cur_q = asof_fundamentals(
        arq[["ticker", "datekey", "sharesbas", "assets", "equity"]], uni, suffix="_q")
    lag_q = asof_fundamentals(
        arq[["ticker", "datekey", "sharesbas", "assets"]], uni,
        lag_days=365, suffix="_q_l1")
    cur_a = asof_fundamentals(art[["ticker", "datekey", "ncfo", "revenue"]], uni)
    stk = asof_fundamentals(streaks, uni)

    f = uni[["snap_date", "ticker", "month"]].copy()
    for block in (cur_q, lag_q, cur_a, stk):
        f = f.merge(block, on=["snap_date", "ticker"], how="left")

    close_long = px.stack().rename("close").reset_index()
    close_long.columns = ["month", "ticker", "close"]
    f = f.merge(close_long, on=["month", "ticker"], how="left")

    def div(num, den):
        return num / den.where(den.abs() > 1e-6)

    feat = pd.DataFrame({"snap_date": f["snap_date"], "ticker": f["ticker"]})
    # RED families (risk-screen keeps) — higher = riskier
    feat["cashburn_streak_q"] = f["cashburn_streak_q"]
    feat["loss_streak_q"] = f["loss_streak_q"]
    feat["asset_gr"] = div(f["assets_q"], f["assets_q_l1"]) - 1.0     # doubling = 1.0
    feat["dilution_1y"] = div(f["sharesbas_q"], f["sharesbas_q_l1"]) - 1.0
    # price run: trailing 12m simple return, incl. last month (tripled = 2.0)
    # YELLOW families
    feat["neg_book_equity"] = (-div(f["equity_q"], f["assets_q"]))    # higher = worse
    # context (not screens, but let users weight/size with them)
    feat["cash_margin"] = -div(f["ncfo"], f["revenue"])               # higher = burnier
    mcap = f["close"] * f["sharesbas_q"]
    feat["log_mcap"] = np.log(mcap.where(mcap > 0))

    panel = panel.merge(feat, on=["snap_date", "ticker"], how="left")

    # 12m price run from the momentum block: mom_12_1 skips the last month, so
    # rebuild the full-window return from monthly closes directly
    with np.errstate(divide="ignore", invalid="ignore"):
        r12 = (px / px.shift(12) - 1.0).stack().rename("ret_12m").reset_index()
    r12.columns = ["month", "ticker", "ret_12m"]
    panel = panel.merge(r12, on=["month", "ticker"], how="left")
    # drawdown vs 52w high, oriented so higher = deeper drawdown
    panel["drawdown_52w"] = 1.0 - panel["high_52w"]

    t = pd.read_parquet(cache / "tickers.parquet")
    t = t[t["table"] == "SEP"][["ticker", "sector", "industry"]].drop_duplicates("ticker")
    panel = panel.merge(t, on="ticker", how="left")
    panel[["sector", "industry"]] = panel[["sector", "industry"]].fillna("Unknown")

    panel = panel.rename(columns={"snap_date": "date"}).drop(
        columns=["month", "ret_1m", "mom_3m", "mom_6_1", "mom_12_1", "high_52w"])
    panel = panel[panel["date"] < OOS_CUTOFF]

    meta = ["date", "ticker", "sector", "industry"]
    fwd = list(FWD_HORIZONS)
    feats = [c for c in panel.columns if c not in meta + fwd]
    panel = panel[meta + feats + fwd]
    for c in feats + fwd:
        panel[c] = panel[c].astype(np.float32)
    data_cols = feats + fwd
    panel = panel.dropna(subset=data_cols, how="all")

    if jitter:
        panel = jitter_panel(panel)

    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    log.info("Wrote %s: %d rows x %d cols (%.1f MB)", out, len(panel),
             panel.shape[1], out.stat().st_size / 1e6)
    log.info("Features: %s", feats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--sp500", required=True)
    ap.add_argument("--out", default="data/short_panel.parquet")
    ap.add_argument("--no-jitter", action="store_true")
    args = ap.parse_args()
    build(Path(args.cache), Path(args.sp500), Path(args.out),
          jitter=not args.no_jitter)
