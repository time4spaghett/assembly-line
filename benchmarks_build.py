"""
One-off builder for the reference benchmark series shipped with the app.

Fetches monthly closes for a handful of index ETFs and stores *forward* one-month
returns, aligned to the same convention as the panel's `fwd_1m`: the value at
month M is the return realized over month M+1. That way a benchmark lines up
with the ntile portfolios without an off-by-one.

Run once; the output is a few KB and needs no network at app runtime — the app
never calls out, which is what keeps it trivially hostable.

Usage:
    python benchmarks_build.py [--out data/benchmarks.parquet]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("benchmarks")

# Same out-of-sample boundary the panel enforces.
OOS_CUTOFF = pd.Timestamp("2026-01-01")

REFERENCES = {
    "SPY": "S&P 500 (cap-weighted)",
    "RSP": "S&P 500 equal-weight",
    "MDY": "S&P MidCap 400",
    "IWM": "Russell 2000 (small cap)",
    "IWD": "Russell 1000 Value",
    "IWF": "Russell 1000 Growth",
    "QQQ": "Nasdaq 100",
}


def build(out: Path) -> None:
    import yfinance as yf

    raw = yf.download(list(REFERENCES), start="1990-01-01", interval="1mo",
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    close = close.sort_index()

    # forward 1-month return, matching the panel's fwd_1m convention
    fwd = close.pct_change().shift(-1)
    fwd.index = pd.to_datetime(fwd.index).to_period("M")
    fwd = fwd[fwd.index.to_timestamp() < OOS_CUTOFF]

    tidy = fwd.reset_index().rename(columns={fwd.index.name or "index": "month"})
    tidy["month"] = tidy["month"].astype(str)
    out.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(out, index=False)

    log.info("Wrote %s (%.1f KB)", out, out.stat().st_size / 1e3)
    for t, label in REFERENCES.items():
        if t in fwd.columns:
            s = fwd[t].dropna()
            log.info("  %-4s %-26s %d months  %s -> %s",
                     t, label, len(s), s.index.min(), s.index.max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/benchmarks.parquet")
    build(Path(ap.parse_args().out))
