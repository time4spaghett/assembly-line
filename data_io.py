"""
Data loading: bundled base panel + custom CSV normalization.

Custom CSVs are long format (one row per security per date). The user maps the
security-id and date columns; sector/industry are optional. Forward returns come
from one of three places:
  - a price column (forward returns computed from monthly closes)
  - explicit forward-return columns mapped per horizon
  - joined from the base panel by ticker + month
Every remaining numeric column becomes a selectable feature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from engine import FWD_COLS

BASE_PANEL = Path(__file__).parent / "data" / "base_panel.parquet"
BENCHMARKS = Path(__file__).parent / "data" / "benchmarks.parquet"

# Reference series shipped with the app (built once by benchmarks_build.py).
REFERENCE_LABELS = {
    "SPY": "S&P 500 (cap-weighted)",
    "RSP": "S&P 500 equal-weight",
    "MDY": "S&P MidCap 400",
    "IWM": "Russell 2000 (small cap)",
    "IWD": "Russell 1000 Value",
    "IWF": "Russell 1000 Growth",
    "QQQ": "Nasdaq 100",
}


def load_benchmarks() -> pd.DataFrame:
    """Monthly forward returns per reference ticker, indexed by month Period."""
    if not BENCHMARKS.exists():
        return pd.DataFrame()
    b = pd.read_parquet(BENCHMARKS)
    b["month"] = pd.PeriodIndex(b["month"], freq="M")
    return b.set_index("month").sort_index()


def reference_series(ticker: str, dates: pd.Series) -> pd.Series:
    """Align a shipped reference series onto the panel's own date index."""
    b = load_benchmarks()
    if b.empty or ticker not in b.columns:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(sorted(pd.Series(dates).unique()))
    vals = b[ticker].reindex(idx.to_period("M"))
    return pd.Series(vals.to_numpy(), index=idx, name=ticker)


def load_panel_1500() -> pd.DataFrame:
    """Top-1500 by market cap, base factor set — built once by
    panel_build_1500.py from base-equity-edge's factor zoo (2026-09-06)."""
    return pd.read_parquet(Path(__file__).parent / "data" / "base_panel_1500.parquet")


def load_panel_smcap() -> pd.DataFrame:
    """Mid/small-cap band of the top-1500 panel: today's $2B–$20B market-cap
    range taken as a percentile band (cap rank 438..1500 within each month's
    top-1500 snapshot) and extended backward. A row filter of base_panel_1500,
    written 2026-09-07 by the session scratchpad's make_smcap.py."""
    return pd.read_parquet(Path(__file__).parent / "data" / "base_panel_smcap.parquet")


def load_short_panel() -> pd.DataFrame:
    """Top-1500 by market cap, short-risk factor set (copied from the
    production repo 2026-09-06 — built once by short_panel_build.py there)."""
    return pd.read_parquet(Path(__file__).parent / "data" / "short_panel.parquet")


def load_si_panel() -> pd.DataFrame:
    """S&P 500 short-interest / short-volume features (free FINRA data, point-in-time),
    built by si_factor/build_features.py - see si_factor/README.md."""
    return pd.read_parquet(Path(__file__).parent / "data" / "si_panel.parquet")


def load_base_panel() -> pd.DataFrame:
    return pd.read_parquet(BASE_PANEL)


def _fwd_from_prices(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Compute forward simple returns at each horizon from monthly closes."""
    px = df.pivot_table(index="month", columns="ticker", values=price_col,
                        aggfunc="last").sort_index()
    out = df.copy()
    for name, h in FWD_COLS.items():
        fwd = (px.shift(-h) / px - 1.0).stack().rename(name).reset_index()
        fwd.columns = ["month", "ticker", name]
        out = out.merge(fwd, on=["month", "ticker"], how="left")
    return out


class AmbiguousDates(ValueError):
    """Raised when a date column has more than one defensible reading."""


def ambiguous_date(col: pd.Series) -> str | None:
    """
    Return an offending value if the column has two defensible readings, else None.

    `03/04/2005` is 4 March or 3 April depending on locale, and pandas just picks
    one — silently, no warning. On a monthly panel the wrong pick reorders the
    calendar and corrupts every forward return while the data still looks fine.
    Asking the user to choose only moves the coin flip, so the format is enforced
    instead: a column parses the same both ways or it is refused.

    ISO needs no special case — dayfirst is inert for YYYY-MM-DD, so it agrees
    with itself and passes here like any other unambiguous format.
    """
    # ISO first: a column that parses cleanly as YYYY-MM-DD (with or without a
    # time part) is unambiguous by construction, whatever dayfirst would do.
    iso = pd.to_datetime(col, errors="coerce", format="ISO8601")
    if iso.notna().sum() == col.notna().sum():
        return None
    a = pd.to_datetime(col, errors="coerce", dayfirst=False)
    b = pd.to_datetime(col, errors="coerce", dayfirst=True)
    disagree = (a != b) & a.notna() & b.notna()
    return str(col[disagree].iloc[0]) if disagree.any() else None


def normalize_csv(raw: pd.DataFrame, id_col: str, date_col: str,
                  sector_col: str | None = None, industry_col: str | None = None,
                  price_col: str | None = None,
                  fwd_map: dict[str, str] | None = None,
                  join_base_returns: bool = False) -> pd.DataFrame:
    """
    Map an uploaded CSV onto the canonical panel contract.

    Dates are read month-first (pandas' default), ISO always exact. A column
    with two defensible readings is no longer refused: `ambiguous_date` is
    advisory and the pages show it as a warning, because the check produced
    false positives on files whose dates were fine and blocked the upload.
    """
    df = raw.copy()
    df["ticker"] = df[id_col].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M")
    # one row per (ticker, month): keep the last observation
    df = df.sort_values("date").groupby(["ticker", "month"], as_index=False).last()

    df["sector"] = df[sector_col].fillna("Unknown") if sector_col else "All"
    df["industry"] = df[industry_col].fillna("Unknown") if industry_col else "All"

    used = {id_col, date_col, sector_col, industry_col} - {None}
    if fwd_map:
        df = df.rename(columns={src: h for h, src in fwd_map.items()})
        # the engine reads every horizon column (fwd_1m for the cumulative
        # curves in particular); the ones the file didn't supply exist as NaN
        for h in FWD_COLS:
            if h not in df.columns:
                df[h] = float("nan")
        used |= set()  # renamed in place
    if price_col:
        df = _fwd_from_prices(df, price_col)
        used.add(price_col)
    if join_base_returns:
        base = load_base_panel()
        base["month"] = base["date"].dt.to_period("M")
        rets = base[["ticker", "month"] + list(FWD_COLS)]
        df = df.drop(columns=[c for c in FWD_COLS if c in df.columns], errors="ignore")
        df = df.merge(rets, on=["ticker", "month"], how="left")

    for h in FWD_COLS:
        if h not in df.columns:
            df[h] = np.nan

    feature_cols = [c for c in df.columns
                    if c not in used
                    and c not in ("ticker", "date", "month", "sector", "industry")
                    and c not in FWD_COLS
                    and pd.api.types.is_numeric_dtype(df[c])
                    # a text column of "n/a" is read as all-NaN and would
                    # otherwise be offered as a perfectly useless feature
                    and df[c].notna().any()]
    keep = ["date", "ticker", "sector", "industry"] + feature_cols + list(FWD_COLS)
    out = df[keep].reset_index(drop=True)
    # match the base panel's float32 dtypes — halves memory on wide custom panels
    for c in feature_cols + list(FWD_COLS):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)
    return out
