"""
One-off: the S&P 500 panel the Destination & Path tab needs in a single file.

    data/sp500_edges_panel.parquet  =  data/base_panel.parquet (as shipped)
                                    +  combo_12m, combo_1m      the two edges
                                    +  beta_252d, resid_vol_252d  risk-model inputs

combo_12m / combo_1m come from signal-laboratory's out/edge_factors_sp500.csv
(combo_1m starts 2009-08). That file dates rows at calendar month-ends and the
panel at the last trading day, so the join is on (ticker, month). The risk
inputs come from data/risk_features_1500.parquet (risk_features_build.py), joined
on (date, ticker). The four new columns are not jittered (user, 2026-10-22); the
base panel's own columns are kept exactly as shipped.

    python panel_build_sp500_edges.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
EDGES_CSV = Path(r"C:\Users\Alex\signal-laboratory\sharadar\out\edge_factors_sp500.csv")
NEW = ["combo_12m", "combo_1m", "beta_252d", "resid_vol_252d"]


def build() -> pd.DataFrame:
    p = pd.read_parquet(ROOT / "data" / "base_panel.parquet")
    n0, cols0 = len(p), list(p.columns)
    c = pd.read_csv(EDGES_CSV, usecols=["date", "ticker", "combo_12m", "combo_1m"], parse_dates=["date"])
    c["month"] = c["date"].dt.to_period("M")
    c = c.drop(columns="date").drop_duplicates(["ticker", "month"], keep="last")
    p["month"] = p["date"].dt.to_period("M")
    p = p.merge(c, on=["ticker", "month"], how="left").drop(columns="month")
    rf = pd.read_parquet(ROOT / "data" / "risk_features_1500.parquet")
    p = p.merge(rf, on=["date", "ticker"], how="left")
    assert len(p) == n0, "a join duplicated rows"
    for col in NEW:
        p[col] = p[col].astype(np.float32)
    # new columns sit with the features, ahead of the forward returns
    fwd = [c for c in cols0 if c.startswith("fwd_")]
    p = p[[c for c in cols0 if c not in fwd] + NEW + fwd]
    return p


if __name__ == "__main__":
    out = ROOT / "data" / "sp500_edges_panel.parquet"
    p = build()
    p.to_parquet(out, index=False)
    cov = {c: f"{p[c].notna().mean():.0%}" for c in NEW}
    print(f"wrote {out}: {len(p):,} rows x {p.shape[1]} cols ({out.stat().st_size / 1e6:.1f} MB); coverage {cov}")
