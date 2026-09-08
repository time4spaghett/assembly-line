"""
One-off builder for the top-1500 base panel — reusing base-equity-edge's
factor zoo rather than re-deriving the features.

`C:\\Users\\Alex\\base-equity-edge` already holds (a) a point-in-time top-1500-by-
market-cap monthly universe (`cache/universe_monthly.parquet`, from Sharadar
DAILY), (b) SEP prices and SF1 fundamentals for every ticker ever in it, and
(c) `factors/zoo.py::build_zoo`, the 28-factor definitions whose IC sweep is
written up in `output/CANDIDATES.md` ("next build: persist panels"). This
script is that persist step, in the app's panel format.

Two adaptations, both deliberate:

  * `build_zoo` cross-sectionally winsorizes and ranks every factor. The app
    stores RAW values so its transform step (rank / zscore / raw) and the
    raw-value constraints mean something, so `_rank_factor` is patched to
    pass the raw panel through untouched.
  * Ranking was the only cross-sectional step. Without it every factor is a
    per-ticker computation, so the zoo is built in ticker chunks and only
    month-end rows are kept — a few hundred MB peak instead of the ~4 GB the
    full daily zoo needs.

Column names are mapped onto the app's existing conventions (see RENAME) so
the S&P 500 panel's defaults work unchanged, and the zoo-only factors (roic,
ebit_mom, roe_mom, roa_e_mom, qearn_mom) come along under their own names.
The daily risk block (vol_1m, vol_12m, max_ret_1m, beta_12m) is recomputed
with panel_build's definitions, chunked the same way.

Usage:
    python panel_build_1500.py [--out data/base_panel_1500.parquet] [--jitter]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BEE = Path(r"C:\Users\Alex\base-equity-edge")
sys.path.insert(0, str(BEE))
import factors.zoo as zoo                       # noqa: E402
from run_broad import ART_FIELDS, ARQ_FIELDS, _pit_panel, build_mask  # noqa: E402
from panel_build import FWD_HORIZONS, OOS_CUTOFF, jitter_panel        # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("panel_build_1500")

CACHE = BEE / "cache"
CHUNK = 700

# zoo name -> (app name, sign). The zoo orients everything "higher = better";
# the app stores the economic quantity and lets the weight carry the sign.
RENAME = {
    "pb_inv": ("btm", 1), "ey": ("earn_yield", 1), "fcf_yield": ("fcf_yield", 1),
    "sales_yield": ("sales_yield", 1),
    "roa": ("roa", 1), "roe": ("roe", 1), "roic": ("roic", 1),
    "gross_margin": ("gross_margin", 1), "op_margin": ("op_margin", 1),
    "gpa": ("gpa", 1), "fcf_margin": ("fcf_margin", 1),
    "inv_leverage": ("leverage", -1),          # zoo: -debt/assets
    "accruals": ("accruals", -1),              # zoo: -(netinc-ncfo)/assets
    "rev_gr_1": ("rev_gr_1y", 1), "rev_gr_3": ("rev_gr_3y", 1),
    "asset_gr_inv": ("asset_gr", -1),          # zoo: -(assets/assets_l1 - 1)
    "mom_12_1": ("mom_12_1", 1), "mom_6_1": ("mom_6_1", 1), "mom_3": ("mom_3m", 1),
    "reversal_1m": ("ret_1m", -1),             # zoo: -1m log return
    "high_52w": ("high_52w", 1),
    "earn_mom": ("earn_mom", 1), "ebit_mom": ("ebit_mom", 1),
    "margin_mom": ("margin_mom", 1), "roe_mom": ("roe_mom", 1),
    "roa_e_mom": ("roa_e_mom", 1), "rev_accel": ("rev_accel", 1),
    "qearn_mom": ("qearn_mom", 1),
}


def _raw_passthrough(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, raw, np.nan).astype(np.float32)


def risk_block(px: pd.DataFrame, mkt: pd.Series) -> dict[str, pd.DataFrame]:
    """panel_build.daily_risk_features' formulas on a ticker chunk, with the
    equal-weight market series computed once over the full universe."""
    ret = px.pct_change(fill_method=None)
    r252, m252 = ret.rolling(252, min_periods=126), mkt.rolling(252, min_periods=126)
    return {
        "vol_1m": ret.rolling(21, min_periods=15).std(),
        "vol_12m": r252.std(),
        "max_ret_1m": ret.rolling(21, min_periods=15).max(),
        "beta_12m": (ret.mul(mkt, axis=0).rolling(252, min_periods=126).mean()
                     - r252.mean().mul(m252.mean(), axis=0)).div(m252.var(), axis=0),
    }


def build(out: Path, jitter: bool) -> None:
    zoo._rank_factor = _raw_passthrough     # raw values, no cross-sectional step

    log.info("Loading caches from %s", CACHE)
    sep = pd.read_parquet(CACHE / "sep.parquet")
    art = pd.read_parquet(CACHE / "sf1_art.parquet")
    arq = pd.read_parquet(CACHE / "sf1_arq.parquet")
    uni = pd.read_parquet(CACHE / "universe_monthly.parquet")
    uni["snap_date"] = pd.to_datetime(uni["snap_date"])
    sep["date"] = pd.to_datetime(sep["date"])

    trading_dates = pd.DatetimeIndex(sep["date"].unique()).sort_values()
    trading_dates = trading_dates[trading_dates >= pd.Timestamp("1998-01-01")]
    price_df = (sep.pivot_table(index="date", columns="ticker", values="closeadj",
                                aggfunc="last").reindex(trading_dates).ffill()
                .astype(np.float32))
    del sep
    tickers = price_df.columns.tolist()
    log.info("Price panel %s", price_df.shape)

    snap_dates = pd.DatetimeIndex(sorted(uni["snap_date"].unique()))
    snap_dates = snap_dates[snap_dates.isin(trading_dates)]
    snap_idx = trading_dates.get_indexer(snap_dates)
    log.info("%d month-end snapshots %s -> %s", len(snap_dates),
             snap_dates[0].date(), snap_dates[-1].date())

    log.info("PIT fundamental panels...")
    art_d = {m: _pit_panel(art, m, trading_dates).reindex(columns=tickers) for m in ART_FIELDS}
    arq_d = {m: _pit_panel(arq, m, trading_dates).reindex(columns=tickers) for m in ARQ_FIELDS}
    mask_full = build_mask(uni, trading_dates, tickers)
    mkt = price_df.pct_change(fill_method=None).mean(axis=1)

    parts = []
    for start in range(0, len(tickers), CHUNK):
        cols = tickers[start:start + CHUNK]
        j = slice(start, start + len(cols))
        z = zoo.build_zoo({m: v.iloc[:, j].to_numpy(np.float32) for m, v in art_d.items()},
                          {m: v.iloc[:, j].to_numpy(np.float32) for m, v in arq_d.items()},
                          price_df.iloc[:, j].to_numpy(np.float32), mask_full[:, j])
        rb = risk_block(price_df.iloc[:, j], mkt)
        frames = []
        for name, arr in z.items():
            app, sign = RENAME[name]
            wide = pd.DataFrame(sign * arr[snap_idx], index=snap_dates, columns=cols)
            frames.append(wide.stack().rename(app))
        for name, df in rb.items():
            wide = df.iloc[snap_idx].astype(np.float32)
            wide.index = snap_dates
            frames.append(wide.stack().rename(name))
        chunk = pd.concat(frames, axis=1)
        chunk.index.names = ["date", "ticker"]
        parts.append(chunk.dropna(how="all"))
        log.info("  chunk %d-%d: %d rows", start, start + len(cols), len(parts[-1]))
    feat = pd.concat(parts).reset_index()
    del art_d, arq_d, parts

    # Universe filter: only (date, ticker) pairs in that month's top-1500 snapshot.
    keep = uni.rename(columns={"snap_date": "date"})[["date", "ticker"]]
    feat = keep.merge(feat, on=["date", "ticker"], how="left")
    feat["mom_3_1"] = feat["mom_3m"] - feat["ret_1m"]

    # Forward simple returns from month-end closes, panel_build's convention.
    px_me = price_df.iloc[snap_idx]
    px_me.index = snap_dates
    for name, h in FWD_HORIZONS.items():
        f = (px_me.shift(-h) / px_me - 1.0).stack().rename(name).reset_index()
        f.columns = ["date", "ticker", name]
        feat = feat.merge(f, on=["date", "ticker"], how="left")
    # live market cap for log_mcap (zoo uses it only as a denominator)
    shares = arq_d_shares = _pit_panel(arq, "sharesbas", trading_dates).reindex(columns=tickers)
    mcap = (px_me * shares.iloc[snap_idx].to_numpy()).stack().rename("mcap").reset_index()
    mcap.columns = ["date", "ticker", "mcap"]
    feat = feat.merge(mcap, on=["date", "ticker"], how="left")
    feat["log_mcap"] = np.log(feat.pop("mcap").where(lambda s: s > 0))

    t = pd.read_parquet(CACHE / "tickers.parquet")
    t = t[t["table"] == "SEP"][["ticker", "sector", "industry"]].drop_duplicates("ticker")
    feat = feat.merge(t, on="ticker", how="left")
    feat[["sector", "industry"]] = feat[["sector", "industry"]].fillna("Unknown")

    feat = feat[feat["date"] < OOS_CUTOFF]
    meta = ["date", "ticker", "sector", "industry"]
    fwd = list(FWD_HORIZONS)
    features = [c for c in feat.columns if c not in meta + fwd]
    feat = feat.dropna(subset=features + fwd, how="all")
    feat = feat[meta + features + fwd]
    for c in features + fwd:
        feat[c] = feat[c].astype(np.float32)
    if jitter:
        feat = jitter_panel(feat)

    out.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out, index=False)
    g = feat.groupby("date")["ticker"].nunique()
    log.info("Wrote %s: %d rows x %d cols (%.1f MB); names/month med %d min %d",
             out, len(feat), feat.shape[1], out.stat().st_size / 1e6,
             int(g.median()), int(g.min()))
    log.info("Features: %s", features)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/base_panel_1500.parquet")
    ap.add_argument("--jitter", action="store_true")
    args = ap.parse_args()
    build(Path(args.out), jitter=args.jitter)
