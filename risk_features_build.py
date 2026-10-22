"""
One-off: the two risk-model inputs the shipped panels lack, for the top-1500
universe (and so for every S&P 500 name-date inside it):

    beta_252d        252-day beta to the equal-weight universe
    resid_vol_252d   252-day market-beta residual volatility, as a 21-day vol

Formulas copied from Chinook's build_features.price_features (lines 110-140,
tag phaseE-two-pass-optimizer): daily closeadj from the base-equity-edge
Sharadar cache, returns clipped to [-50%, +100%], rolling 252-day cov/var
beta lagged one day, residual vol = rolling 252-day std x sqrt(21), sampled
at the last trading day on or before each panel date. Chinook already has
these for the S&P 500 and the $2-15B band (data/extra_features*.parquet);
this fills the rest of the top 1500 in one pass with the same code.

    python risk_features_build.py [--out data/risk_features_1500.parquet]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("risk_features")
CACHE = Path("C:/Users/Alex/base-equity-edge/cache")


def build(panel: pd.DataFrame, cache: Path = CACHE, chunk: int = 1200) -> pd.DataFrame:
    tickers = sorted(panel.ticker.unique())
    log.info("panel %d rows, %d tickers; reading daily closes", len(panel), len(tickers))
    sep = pd.read_parquet(cache / "sep.parquet", filters=[("ticker", "in", tickers)])
    px = sep.pivot(index="date", columns="ticker", values="closeadj").sort_index()
    px = px.loc[panel.date.min() - pd.Timedelta(days=600):]
    r = px.pct_change(fill_method=None).clip(-0.5, 1.0)
    member = (panel.assign(v=1.0).pivot(index="date", columns="ticker", values="v")
              .reindex(px.index, method="ffill", limit=31).reindex(columns=px.columns))
    mkt = (r * member).mean(axis=1)  # equal-weight universe, members only
    mvar = mkt.rolling(252, min_periods=126).var()
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    pos = px.index.searchsorted(dates, side="right") - 1  # last trading day <= date
    parts = []
    for start in range(0, len(tickers), chunk):   # chunked: the full rolling cov is ~2 GB
        cols = tickers[start:start + chunk]
        rc = r[cols]
        cov = rc.rolling(252, min_periods=126).cov(mkt)
        beta = cov.div(mvar, axis=0).shift(1)
        resid = rc - beta.mul(mkt, axis=0)
        rv = resid.rolling(252, min_periods=126).std() * np.sqrt(21)
        b = beta.iloc[pos].set_axis(dates).stack().rename("beta_252d")
        v = rv.iloc[pos].set_axis(dates).stack().rename("resid_vol_252d")
        parts.append(pd.concat([b, v], axis=1).rename_axis(["date", "ticker"]).reset_index())
        log.info("  tickers %d-%d done", start, start + len(cols))
    feat = pd.concat(parts, ignore_index=True)
    out = panel[["date", "ticker"]].merge(feat, on=["date", "ticker"], how="left")
    log.info("coverage: beta %.1f%%, resid vol %.1f%%",
             100 * out.beta_252d.notna().mean(), 100 * out.resid_vol_252d.notna().mean())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/base_panel_1500.parquet")
    ap.add_argument("--out", default="data/risk_features_1500.parquet")
    a = ap.parse_args()
    p = pd.read_parquet(a.panel, columns=["date", "ticker"])
    build(p).to_parquet(a.out, index=False)
    log.info("wrote %s", a.out)
