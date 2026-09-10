"""
Destination & Path — two edges, two horizons, one universe.

The premise: where a portfolio should end up and how fast to get there are
different questions with different answers. The *destination* edge is slow
and fundamental — a ranking of the names in coverage that is expected to pay
off over a year. The *path* edge is fast — something with information about
the next month, used later to modulate how quickly the book moves toward the
destination (transact faster where both edges agree on a name, slower where
they don't). That modulation is a later step; this page builds and tests the
two base edges and shows how much they agree.

Built as a sibling of the Edge Concierge, not a mode of it: same panel, same
universe screen, same scoring engine, but two builder tables, each scored on
its own horizon, plus a cross-horizon matrix so a factor set can be judged on
whether it carries information where it is meant to — and not where it isn't.

Sector-analyst framing by default: one sector, terciles, and the IC's
minimum cross-section lowered so a 20-name coverage list still scores.

Run:  streamlit run main.py  (this is a page, not the entry point)
"""
from __future__ import annotations

import hashlib
import io
import pathlib
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_io import (AmbiguousDates, ambiguous_date, load_base_panel,
                     load_panel_1500, load_panel_smcap, load_short_panel,
                     normalize_csv)
from ui import BLUE, INK_2, RED, SURFACE, ramp, style, universe_block
from engine import (BENCH_CAP, BENCH_EW, FWD_COLS, TRANSFORMS, EdgeSpec,
                    FeatureSpec, _spearman, benchmark_options, build_composite,
                    consistency_checks, feature_columns, ic_analysis,
                    newey_west_tstat, run_edge)

BASE_PANEL_LABEL = "S&P 500 · 1998–2025 · basic factors"
PANEL_1500_LABEL = "Top 1500 · 1998–2025 · base factors"
# Streamlit renders $…$ as LaTeX, so the dollar signs are escaped.
SMCAP_LABEL = "Mid/small cap · \\$2B–\\$20B today, as a percentile band · 1998–2025"
SHORT_PANEL_LABEL = "Top 1500 · 1998–2025 · short-risk factors"
HORIZON_LABELS = {"fwd_1m": "1 month", "fwd_3m": "3 months",
                  "fwd_6m": "6 months", "fwd_12m": "12 months"}

# ── Defaults ─────────────────────────────────────────────────────────────────
# Destination: the Concierge's example edge — cash-generative, profitable,
# clean accounting, not over-expanding. Path: the textbook short-horizon
# construction — intermediate momentum (months 2–3 and 2–6) with the most
# recent month entered negatively, since it reverses. It is the *shape* of a
# path signal, not a strong one: on this panel (monthly, S&P 500, no
# revisions / surprise / short-interest / volume) it scores ≈0 at one month in
# every sector tried (Healthcare, Technology, Industrials, 2026-09-06 sweep),
# and the one price signal that does fire in-sample — bare 1m reversal, t 2–4
# — fails the 2020–25 holdout in all three. Next-month information on a
# coverage list lives in data the analyst has and this panel doesn't; bring it
# in as a column and make it the path edge.
DEST_ROWS = [
    {"feature": "fcf_yield", "transform": "rank", "weight": 1.0},
    {"feature": "gpa",       "transform": "rank", "weight": 0.5},
    {"feature": "accruals",  "transform": "rank", "weight": -0.5},
    {"feature": "asset_gr",  "transform": "rank", "weight": -0.5},
]
PATH_ROWS = [
    {"feature": "ret_1m",  "transform": "rank", "weight": -1.0},   # 1m reversal
    {"feature": "mom_3_1", "transform": "rank", "weight": 1.0},    # months 2–3
    {"feature": "mom_6_1", "transform": "rank", "weight": 1.0},    # months 2–6
]
# Short-risk panel (top 1500): every feature is oriented higher = riskier, so
# the destination is "avoid the blow-up profile" — all negative. On the top
# 1500 (2026-09-06, full sample, 12m): dilution_1y IC −0.088 (t −4.3),
# loss_streak_q −0.078 (t −3.7), cashburn_streak_q −0.063 (t −4.4), vol_63d
# −0.097 (t −2.7), and all four strengthen in the 2020–25 holdout. The same
# names also carry 1m information (cashburn t −5.1, dilution t −4.2), so the
# panel's path default is the fastest-moving pair — it is a quality signal
# read monthly, not a true price-reversal path; that needs price features
# this panel doesn't carry.
DEST_ROWS_SHORT = [
    {"feature": "dilution_1y",       "transform": "rank", "weight": -1.0},
    {"feature": "loss_streak_q",     "transform": "rank", "weight": -1.0},
    {"feature": "cashburn_streak_q", "transform": "rank", "weight": -1.0},
    {"feature": "vol_63d",           "transform": "rank", "weight": -0.5},
]
PATH_ROWS_SHORT = [
    {"feature": "vol_63d",           "transform": "rank", "weight": -1.0},
    {"feature": "cashburn_streak_q", "transform": "rank", "weight": -1.0},
]
# Top-1500 base panel. Destination = base-equity-edge CANDIDATES.md's
# Candidate 1, cheap × profitable × improving, chosen there on economics and
# already holdout-tested in that project. On this panel (2026-09-06, pre-2020
# in-sample → 2020–25 holdout, quintiles): 12m IC +0.065 (t 3.7) → +0.135
# (t 5.3), L/S Sharpe 0.96 OOS; Healthcare terciles +0.094 → +0.212. Reads
# +0.025 at 1m, so it is a slow signal, as it should be.
# Path = fade last month's move and avoid the lottery / high-vol names into
# next month (1m reversal + Bali–Cakici–Whitelaw MAX effect). The only 1m
# construction positive in-sample AND out-of-sample in both the full universe
# (+0.029 t 3.5 → +0.016) and Healthcare (+0.037 t 4.0 → +0.050 t 2.6, Sharpe
# 0.85). Intermediate momentum (−1m +3-1 +6-1) is ≈0 at one month on 1,500
# names too. Caveat, stated plainly: six path candidates were compared with
# the holdout visible, so its OOS number is a little flattered.
# Two legs added 2026-09-06 after a holdout sweep across Healthcare / Tech /
# Industrials: low vol (−½·vol_12m) and investment discipline (−½·asset_gr)
# lift the 12m IC in every sector in-sample (+0.094→+0.110, +0.053→+0.066,
# +0.042→+0.069) and in two of three out of sample. Both are textbook 12m
# effects (low-vol anomaly; asset-growth factor), which is why they are in
# and roe_mom / value-heavy / quality-heavy variants, which were not better,
# are not.
DEST_ROWS_1500 = [
    {"feature": "fcf_yield", "transform": "rank", "weight": 1.0},
    {"feature": "roic",      "transform": "rank", "weight": 1.0},
    {"feature": "qearn_mom", "transform": "rank", "weight": 0.5},
    {"feature": "vol_12m",   "transform": "rank", "weight": -0.5},
    {"feature": "asset_gr",  "transform": "rank", "weight": -0.5},
]
PATH_ROWS_1500 = [
    {"feature": "ret_1m",     "transform": "rank", "weight": -1.0},   # 1m reversal
    {"feature": "vol_1m",     "transform": "rank", "weight": -0.5},   # avoid high vol
    {"feature": "max_ret_1m", "transform": "rank", "weight": -0.5},   # avoid lottery
]
DEFAULT_SECTORS = ["Healthcare"]   # ~160 names a month on the top 1500, ~55 on the 500
DEST_HORIZON, PATH_HORIZON = "fwd_12m", "fwd_1m"


def _default_rows(rows: list[dict], features: list[str]) -> pd.DataFrame:
    keep = [r for r in rows if r["feature"] in features]
    if not keep:
        keep = [{"feature": features[0], "transform": "rank", "weight": 1.0}]
    return pd.DataFrame(keep)


# ── Figures (same shapes as the Concierge, so the two pages read alike) ──────

def fig_ntile_bars(res, colors, hl, bench_label) -> go.Figure:
    q_ann = res["q_ann_returns"]
    fig = go.Figure(go.Bar(
        x=[f"Q{int(q)}" for q in q_ann.index], y=q_ann.values,
        marker_color=colors, marker_line_color=SURFACE, marker_line_width=2,
        hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.add_hline(y=res["bench_ann"], line_dash="dash", line_color=INK_2,
                  line_width=2, annotation_text=bench_label,
                  annotation_position="top left",
                  annotation_font=dict(color=INK_2, size=12))
    fig.update_yaxes(tickformat=".0%", title=f"Mean {hl} fwd return, annualized")
    return style(fig, 320)


def fig_ntile_curves(res, colors, bench_label) -> go.Figure:
    q_cum = res["q_cum"]
    fig = go.Figure()
    bc = res["bench_cum"].dropna()
    if len(bc):
        fig.add_trace(go.Scatter(
            x=bc.index, y=bc.values, name=bench_label, mode="lines",
            line=dict(color=INK_2, width=2, dash="dash"),
            hovertemplate="Universe: %{y:.2f}×<extra></extra>"))
    for i, q in enumerate(q_cum.columns):
        fig.add_trace(go.Scatter(
            x=q_cum.index, y=q_cum[q], name=f"Q{int(q)}", mode="lines",
            line=dict(color=colors[i], width=2),
            hovertemplate="Q" + str(int(q)) + ": %{y:.2f}×<extra></extra>"))
    fig.update_yaxes(type="log", title="Growth of $1 (log)",
                     tickmode="array", tickvals=[1, 2, 5, 10, 20, 50, 100],
                     ticktext=["1×", "2×", "5×", "10×", "20×", "50×", "100×"])
    return style(fig, 320)


def fig_ic(res, hl) -> go.Figure:
    ic = res["ic_series"]
    roll = ic.rolling(12).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ic.index, y=ic.values, name="Monthly IC",
                         marker_color="#9ec5f4",
                         hovertemplate="IC %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=roll.index, y=roll.values, name="12m rolling mean",
                             line=dict(color="#104281", width=2),
                             hovertemplate="12m mean %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title=f"Spearman IC ({hl} fwd)")
    return style(fig, 300)


# ── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading base panel…")
def base_panel() -> pd.DataFrame:
    return load_base_panel()


@st.cache_data(show_spinner="Loading top-1500 panel…")
def panel_1500() -> pd.DataFrame:
    return load_panel_1500()


@st.cache_data(show_spinner="Loading mid/small-cap panel…")
def panel_smcap() -> pd.DataFrame:
    return load_panel_smcap()


@st.cache_data(show_spinner="Loading short-risk panel…")
def short_panel() -> pd.DataFrame:
    return load_short_panel()


def read_upload(data: bytes) -> pd.DataFrame:
    """Parsed once per session, in session_state rather than the process-wide
    cache, so one user's file never sits in memory shared with another's."""
    digest = hashlib.md5(data).hexdigest()
    if st.session_state.get("_upload_digest") != digest:
        with st.spinner("Reading CSV…"):
            st.session_state["_upload_df"] = pd.read_csv(io.BytesIO(data))
        st.session_state["_upload_digest"] = digest
    return st.session_state["_upload_df"]


import engine as _engine_mod
ENGINE_SIG = hashlib.md5(
    pathlib.Path(_engine_mod.__file__).read_bytes()).hexdigest()[:8]


@st.cache_data(show_spinner="Scoring…")
def cached_run(engine_sig: str, panel_key: str, sample: str, panel: pd.DataFrame,
               spec_rows: tuple, sector_neutral: bool, horizon: str, n_q: int,
               bench_mode: str, min_n: int) -> dict:
    spec = EdgeSpec(features=[FeatureSpec(*r) for r in spec_rows],
                    sector_neutral=sector_neutral)
    return run_edge(panel, spec, horizon=horizon, n_q=n_q, bench_mode=bench_mode,
                    min_n=min_n)


@st.cache_data(show_spinner=False)
def cached_ic_matrix(engine_sig: str, panel_key: str, sample: str,
                     panel: pd.DataFrame, spec_rows: tuple, sector_neutral: bool,
                     min_n: int) -> dict:
    """IC and t at every horizon for one edge — the cheap part of run_edge."""
    spec = EdgeSpec(features=[FeatureSpec(*r) for r in spec_rows],
                    sector_neutral=sector_neutral)
    comp = build_composite(panel, spec)
    out = {}
    for h in FWD_COLS:
        if panel[h].notna().sum() < 100:
            out[h] = (np.nan, np.nan, 0)
            continue
        r = ic_analysis(panel, comp, h, min_n=min_n)
        out[h] = (r["mean_ic"], r["t_stat"], r["n_months"])
    return out


def _ntile(comp: pd.Series, dates: pd.Series, n_q: int) -> pd.Series:
    """1..n_q within each date; NaN where a month is too thin to cut."""
    df = pd.DataFrame({"c": comp, "d": dates})
    return df.groupby("d")["c"].transform(
        lambda s: pd.qcut(s, n_q, labels=False, duplicates="drop") + 1
        if s.notna().sum() >= n_q * 3 else pd.Series(np.nan, index=s.index))


# ── 1 · Panel ─────────────────────────────────────────────────────────────────

st.title("Destination & Path")
st.caption("Two edges on one universe: a slow, fundamental ranking of where the "
           "book should end up, and a fast one about the next month, for "
           "deciding how quickly to get there.")

with st.container(border=True, key="step1"):
    st.subheader("1 · Panel", help="The data both edges are measured on. The shipped "
                 "panel is point-in-time S&P 500 membership; an upload replaces it. "
                 "If you already have a computed edge, bring it as a column — a "
                 "single-row spec with weight 1 scores it as-is.")
    # Only the panels actually on disk are offered: the deployed app ships the
    # S&P panel alone, the larger ones stay local.
    _data_dir = pathlib.Path(_engine_mod.__file__).parent / "data"
    _options = [lbl for lbl, fn in ((PANEL_1500_LABEL, "base_panel_1500.parquet"),
                                    (SMCAP_LABEL, "base_panel_smcap.parquet"),
                                    (BASE_PANEL_LABEL, "base_panel.parquet"),
                                    (SHORT_PANEL_LABEL, "short_panel.parquet"))
                if (_data_dir / fn).exists()] + ["Upload CSV"]
    source = st.radio("Panel", _options, key="dp_source",
                      help="Monthly panels, 1998 → Dec 2025, point-in-time. **Top "
                           "1500 · base factors**: largest 1,500 US stocks by market "
                           "cap, 34 value / quality / growth / momentum / "
                           "fundamental-momentum / risk features (the default — "
                           "a sector's coverage is 100–250 names here). **Mid/small "
                           "cap**: the same panel below today's $20B line — cap rank "
                           "438 to 1500 each month, so the band is a percentile, not "
                           "a dollar range, going back. **S&P 500**: the same "
                           "features on the 500. **Short-risk**: top 1500, the "
                           "blow-up-screen set, oriented higher = riskier. A CSV "
                           "loaded here or in the Concierge is shared.")
    panel, panel_key = None, "base"
    if source == PANEL_1500_LABEL:
        panel, panel_key = panel_1500(), "top1500"
    elif source == SMCAP_LABEL:
        panel, panel_key = panel_smcap(), "smcap"
    elif source == BASE_PANEL_LABEL:
        panel = base_panel()
    elif source == SHORT_PANEL_LABEL:
        panel, panel_key = short_panel(), "short"
    else:
        up = st.file_uploader("CSV — one row per security per date", type="csv",
                              key="dp_upload")
        st.caption("Held in memory for this session only — never written to disk.")
        if up is not None:
            raw = read_upload(up.getvalue())
            cols = list(raw.columns)

            def _guess(names, fallback=0):
                for i, c in enumerate(cols):
                    if any(n in c.lower().replace(" ", "") for n in names):
                        return i
                return fallback

            id_col = st.selectbox("Security ID column", cols, key="dp_id",
                                  index=_guess(("ticker", "symbol", "company",
                                                "id", "name", "sedol", "cusip")))
            date_col = st.selectbox("Date column", cols, key="dp_date",
                                    index=_guess(("date", "month", "period",
                                                  "asof"), min(1, len(cols) - 1)))
            _dex = ambiguous_date(raw[date_col])
            if _dex is not None:
                st.warning(f"Confirm your dates are formatted as **YYYY-MM-DD** — "
                           f"`{date_col}` has values like {_dex!r} that could read "
                           f"day-first or month-first. They are read month-first.")
            opt = ["(none)"] + cols
            sector_col = st.selectbox(
                "Sector column (optional)", opt, key="dp_sector",
                index=_guess(("sector",)) + 1
                if any("sector" in c.lower() for c in cols) else 0)
            industry_col = st.selectbox(
                "Industry column (optional)", opt, key="dp_industry",
                index=_guess(("industry",)) + 1
                if any("industry" in c.lower() for c in cols) else 0)
            # House conventions for the forward returns: `TotalReturn_USD` is the
            # 1-month forward return, and `fwd_return_3m/6m/12m` the longer ones.
            _fwd_std = {"fwd_1m": "TotalReturn_USD", "fwd_3m": "fwd_ret_3m",
                        "fwd_6m": "fwd_ret_6m", "fwd_12m": "fwd_ret_12m"}
            _fwd_alt = {"fwd_3m": ["fwd_return_3m"], "fwd_6m": ["fwd_return_6m"],
                        "fwd_12m": ["fwd_return_12m"], "fwd_1m": ["TotalReturnUSD"]}
            # tolerant match: case, spaces, underscores and punctuation ignored
            _norm = lambda s: re.sub(r"[^0-9a-z]", "", str(s).lower())
            _bynorm = {_norm(c): c for c in cols}
            _fwd_found = {}
            for h, n in _fwd_std.items():
                for cand in [n] + _fwd_alt.get(h, []):
                    if _norm(cand) in _bynorm:
                        _fwd_found[h] = _bynorm[_norm(cand)]
                        break
            _has_px = any(n in c.lower() for c in cols
                          for n in ("price", "close", "px", "adj"))
            ret_src = st.radio("Forward returns come from…", [
                "A price column (computed at 1/3/6/12m)",
                "A forward-return column",
                "Join from base panel by ticker",
            ], index=1 if _fwd_found else (0 if _has_px else 2), key="dp_retsrc")
            price_col = fwd_map = None
            join_base = False
            if ret_src.startswith("A price"):
                price_col = st.selectbox("Price column", cols, key="dp_px",
                                         index=_guess(("price", "close", "px", "adj")))
            elif ret_src.startswith("A forward"):
                # one dropdown per horizon, defaulting to the house names
                fwd_map = {}
                _fc = st.columns(4)
                for _i, (h, _std) in enumerate(_fwd_std.items()):
                    _opts = ["(none)"] + cols
                    _dflt = _fwd_found.get(h)
                    _sel = _fc[_i].selectbox(
                        f"{HORIZON_LABELS[h]} forward return", _opts, key=f"dp_fwd_{h}",
                        index=_opts.index(_dflt) if _dflt in _opts else 0,
                        help=f"Defaults to `{_std}` when the file has it.")
                    if _sel != "(none)":
                        fwd_map[h] = _sel
                if not fwd_map:
                    st.warning("Pick at least one forward-return column.")
                    fwd_map = None
            else:
                join_base = True
                st.caption("Security IDs must be US tickers for the join.")
            if st.button("Load CSV", type="primary", width="stretch", key="dp_load"):
                st.session_state["csv_panel"] = normalize_csv(
                    raw, id_col, date_col,
                    None if sector_col == "(none)" else sector_col,
                    None if industry_col == "(none)" else industry_col,
                    price_col=price_col, fwd_map=fwd_map,
                    join_base_returns=join_base)
        if "csv_panel" in st.session_state:
            panel, panel_key = st.session_state["csv_panel"], "csv"
            st.success(f"{panel['ticker'].nunique():,} securities · "
                       f"{panel['date'].nunique():,} dates · "
                       f"{panel['date'].min():%b %Y}–{panel['date'].max():%b %Y}")
    if panel is None:
        st.stop()

# A tilt handed over from another tool is an ordinary column here — which is
# the "someone already has a calced edge" case: point the destination at it.
for _state_key, _tilt_col, _src in (("learned_tilt_panel", "learned_tilt", "Learned Edge"),
                                    ("neural_tilt_panel", "neural_tilt", "Neural Edge"),
                                    ("ca_tilt_panel", "ca_tilt", "Factor Edge")):
    _tilt = st.session_state.get(_state_key)
    if _tilt is None or _tilt_col in panel.columns:
        continue
    try:
        t = _tilt.copy()
        t["date"] = pd.to_datetime(t["date"])
        t["secid"] = t["secid"].astype(str).str.strip().str.upper()
        t = t.groupby(["date", "secid"], as_index=False)[_tilt_col].mean()
        panel = panel.merge(t.rename(columns={"secid": "ticker"}),
                            on=["date", "ticker"], how="left")
        panel[_tilt_col] = panel[_tilt_col].fillna(0.5)
        st.caption(f"{_src} tilt available as `{_tilt_col}`.")
    except Exception as e:
        st.warning(f"Couldn't merge the {_src} tilt: {e}")

features = feature_columns(panel)

# ── 2 · Universe ──────────────────────────────────────────────────────────────

with st.container(border=True, key="step2"):
    st.subheader("2 · Coverage", help="The names both edges rank — the same universe "
                 "for both, by construction. Default is one sector, the analyst's "
                 "coverage list. Conditions are applied before ranking.")
    uni = universe_block(panel, "dp", features, default_sectors=DEFAULT_SECTORS)
fpanel = uni["panel"]
if fpanel.empty:
    st.info("Nothing survives the screen.")
    st.stop()
HORIZONS_AVAILABLE = [h for h in FWD_COLS
                      if h in fpanel.columns and fpanel[h].notna().sum() >= 100]
if not HORIZONS_AVAILABLE:
    st.error("This panel has no forward returns in the current screen — nothing "
             "can be scored. Upload a price column, a forward-return column, or "
             "join returns from the base panel.")
    st.stop()
if len(HORIZONS_AVAILABLE) == 1:
    st.warning(f"This panel only has **{HORIZON_LABELS[HORIZONS_AVAILABLE[0]]}** forward "
               f"returns, so both edges are scored on that horizon — the destination "
               f"cannot be tested at 12 months here. Upload a price column to get "
               f"1/3/6/12-month returns.")
if "fwd_1m" not in HORIZONS_AVAILABLE:
    st.warning("No 1-month forward returns in this panel: the pacing backtest and "
               "the alpha term structure need them and will be skipped.")

# ── 3 · The two edges ─────────────────────────────────────────────────────────

try:
    from nl import api_key as _nl_api_key
    _NL_KEY = _nl_api_key()
except Exception:
    _NL_KEY = None


def _live_rows(state_key: str, editor_key: str) -> list[tuple]:
    base = st.session_state[state_key].copy().reset_index(drop=True)
    state = st.session_state.get(editor_key) or {}
    for idx, changes in state.get("edited_rows", {}).items():
        for col, val in changes.items():
            base.loc[int(idx), col] = val
    if state.get("added_rows"):
        base = pd.concat([base, pd.DataFrame(state["added_rows"])], ignore_index=True)
    if state.get("deleted_rows"):
        base = base.drop(index=[i for i in state["deleted_rows"] if i in base.index])
    base = base.dropna(subset=["feature", "transform", "weight"])
    return [(r.feature, r.transform, float(r.weight))
            for r in base.itertuples() if r.weight != 0]


def _formula(rows: list[tuple]) -> str:
    parts = []
    for i, (feat, tr, w) in enumerate(rows):
        term = feat if tr == "raw" else f"{tr}({feat})"
        coef = "" if abs(w) == 1 else f"{abs(w):g}·"
        sign = ("−" if w < 0 else "") if i == 0 else (" − " if w < 0 else " + ")
        parts.append(f"{sign}{coef}{term}")
    return "".join(parts) or "—"


def edge_builder(name: str, key: str, default_rows: list[dict],
                 default_horizon: str, blurb: str) -> dict:
    """One builder table + its own horizon and neutrality toggle."""
    st.markdown(f"**{name}**")
    st.caption(blurb)
    # (Re)seed when first shown, or when a panel switch leaves the stored rows
    # pointing at features this panel doesn't have.
    _stored = st.session_state.get(f"{key}_rows")
    if _stored is None or not set(_stored["feature"].dropna()) <= set(features):
        st.session_state[f"{key}_rows"] = _default_rows(default_rows, features)
        st.session_state.pop(f"{key}_editor", None)
    # English sketch, same planner as the Concierge; writes only this edge's
    # table (the universe stays whatever step 2 says).
    if _NL_KEY:
        with st.expander("Describe this edge in English", expanded=True):
            desc = st.text_area("Features and weights, or just the idea", height=70,
                                key=f"{key}_desc", label_visibility="collapsed",
                                placeholder="e.g. cheap, profitable, improving; avoid "
                                            "the volatile names")
            if st.button("Draft", key=f"{key}_draft") and desc:
                try:
                    from nl import plan_edge
                    with st.status("Reading your description…", expanded=False) as _s:
                        plan = plan_edge(desc, features,
                                         sorted(panel["sector"].dropna().unique()), _NL_KEY)
                        _s.update(label="Drafted", state="complete")
                    st.session_state[f"{key}_rows"] = pd.DataFrame(
                        [r.model_dump() for r in plan.rows])
                    st.session_state.pop(f"{key}_editor", None)
                    st.session_state[f"{key}_note"] = plan.rationale
                    st.rerun()
                except Exception as e:
                    st.error(f"Couldn't draft: {e}")
            if st.session_state.get(f"{key}_note"):
                st.caption(f"→ {st.session_state[f'{key}_note']}  Adjust anything in "
                           f"the table; the table is what gets tested.")
    else:
        st.caption("English sketch needs an `ANTHROPIC_API_KEY` (env or Streamlit "
                   "Secrets).")
    edited = st.data_editor(
        st.session_state[f"{key}_rows"], num_rows="dynamic", width="stretch",
        hide_index=True, key=f"{key}_editor",
        column_config={
            "feature": st.column_config.SelectboxColumn(
                "Feature", options=features, required=True, width="medium"),
            "transform": st.column_config.SelectboxColumn(
                "Transform", options=TRANSFORMS, required=True),
            "weight": st.column_config.NumberColumn("Weight", step=0.25),
        })
    rows = [(r.feature, r.transform, float(r.weight))
            for r in edited.dropna(subset=["feature", "weight"]).itertuples()
            if r.weight != 0]
    c1, c2 = st.columns(2)
    # Only horizons this panel actually has returns for. An uploaded file
    # with a single forward-return column has one; scoring a 12m destination
    # on it produced NaN metrics and a meaningless verdict.
    if default_horizon not in HORIZONS_AVAILABLE:
        default_horizon = (HORIZONS_AVAILABLE[-1] if key.endswith("dest")
                           else HORIZONS_AVAILABLE[0])
    if st.session_state.get(f"{key}_horizon") not in HORIZONS_AVAILABLE:
        st.session_state.pop(f"{key}_horizon", None)
    horizon = c1.selectbox("Scored on", HORIZONS_AVAILABLE, key=f"{key}_horizon",
                           index=HORIZONS_AVAILABLE.index(default_horizon),
                           format_func=HORIZON_LABELS.get,
                           help="Only horizons with forward returns in this panel. "
                                "Upload a price column to get all four.")
    neutral = c2.toggle("Sector-neutral", key=f"{key}_neutral",
                        help="Rank within sector. Moot when coverage is one sector.")
    st.caption(f"`{_formula(rows)}`")
    return {"rows": rows, "horizon": horizon, "neutral": neutral}


with st.container(border=True, key="step3"):
    st.subheader("3 · The two edges", help="Each table is its own edge with its own "
                 "horizon. Destination should carry information at the long horizon; "
                 "path at the short one. The matrix below tests exactly that, in "
                 "both directions.")
    _dest_default, _path_default = {
        "top1500": (DEST_ROWS_1500, PATH_ROWS_1500),
        "smcap": (DEST_ROWS_1500, PATH_ROWS_1500),   # same edges; stronger in the band
        "short": (DEST_ROWS_SHORT, PATH_ROWS_SHORT),
    }.get(panel_key, (DEST_ROWS, PATH_ROWS))
    cd, cp = st.columns(2, gap="large")
    with cd:
        dest = edge_builder(
            "Destination — where the book should end up", "dp_dest",
            _dest_default, DEST_HORIZON,
            "Slow, fundamental. If you already have a ranking, pick that column "
            "alone with weight 1.")
    with cp:
        path = edge_builder(
            "Path — how fast to get there", "dp_path",
            _path_default, PATH_HORIZON,
            "Fast. Whatever knows something about the next month: price and "
            "estimate flow, reversal, flows.")

# ── 4 · Test ──────────────────────────────────────────────────────────────────

with st.container(border=True, key="step4"):
    st.subheader("4 · Test", help="Ntiles are shared so the two rankings can be laid "
                 "on top of each other. Terciles by default: a coverage list of "
                 "20–60 names does not support quintiles.")
    t1, t2, t3, t4 = st.columns(4)
    n_q = t1.select_slider("Ntile portfolios", options=[3, 4, 5], value=3,
                           key="dp_nq")
    holdout_yrs = t2.number_input("Holdout (years)", 0, 20, 5, 1, key="dp_holdout",
                                  help="Most recent N years held back; the "
                                       "consistency checks compare the two slices.")
    bench_mode = t3.selectbox("Benchmark", benchmark_options(panel), key="dp_bench")
    _names = fpanel.groupby("date")["ticker"].nunique()
    min_n = int(t4.number_input(
        "Min names / month", 5, 100, int(min(30, max(10, n_q * 3))), 1,
        key="dp_minn",
        help=f"A month with fewer scored names than this contributes no IC. "
             f"This coverage has {int(_names.median())} names in a typical month "
             f"(thinnest {int(_names.min())})."))
    if _names.median() < n_q * 5:
        st.warning(f"Typical month has {int(_names.median())} names — fewer than "
                   f"{n_q * 5} needed to cut {n_q} ntiles. Widen coverage or cut "
                   f"fewer ntiles.")

if not dest["rows"] or not path["rows"]:
    st.info("Both edges need at least one feature row with a non-zero weight.")
    st.stop()

# ── Samples ───────────────────────────────────────────────────────────────────
IN_SAMPLE, OUT_SAMPLE, ALL_SAMPLE = "In-sample", "Holdout", "All"
_last = fpanel["date"].max()
_cut = _last - pd.DateOffset(years=int(holdout_yrs)) if holdout_yrs else None
SAMPLES = ({ALL_SAMPLE: fpanel} if _cut is None else
           {IN_SAMPLE: fpanel[fpanel["date"] < _cut],
            OUT_SAMPLE: fpanel[fpanel["date"] >= _cut], ALL_SAMPLE: fpanel})
_opts = [k for k, v in SAMPLES.items() if v["date"].nunique() >= 12]
if not _opts:
    st.error("No sample has enough months — reduce the holdout or widen the dates.")
    st.stop()
c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
sample = c1.segmented_control("Sample", _opts, default=_opts[0], key="dp_sample") or _opts[0]
if _cut is not None:
    c2.caption(f"In-sample to {_cut:%Y-%m} · holdout {_cut:%Y-%m} – {_last:%Y-%m}")


def _score(edge: dict, sub: pd.DataFrame, sample_tag: str, horizon: str | None = None):
    return cached_run(ENGINE_SIG, panel_key, sample_tag, sub, tuple(edge["rows"]),
                      edge["neutral"], horizon or edge["horizon"], n_q, bench_mode,
                      min_n)


EDGES = {"Destination": dest, "Path": path}
res = {k: _score(e, SAMPLES[sample], sample) for k, e in EDGES.items()}
colors = ramp(n_q)

# ── Cross-horizon matrix ──────────────────────────────────────────────────────
# The framework question, not a decoration: does the destination factor set
# speak at 12 months and go quiet at 1, and does the path set do the reverse?
# A set that is strong everywhere is fine for the destination and useless as
# a path signal — it adds nothing the destination didn't already know.
st.markdown("#### Where each edge carries information")
mat = {k: cached_ic_matrix(ENGINE_SIG, panel_key, sample, SAMPLES[sample],
                           tuple(e["rows"]), e["neutral"], min_n)
       for k, e in EDGES.items()}
_tbl = pd.DataFrame({
    f"IC @ {HORIZON_LABELS[h]}": {k: f"{m[h][0]:+.3f}  (t {m[h][1]:+.1f})"
                                  if np.isfinite(m[h][0]) else "—"
                                  for k, m in mat.items()}
    for h in FWD_COLS})
st.dataframe(_tbl, width="stretch")
_dh, _ph = dest["horizon"], path["horizon"]
_d_own, _p_own = mat["Destination"][_dh][0], mat["Path"][_ph][0]
_d_cross, _p_cross = mat["Destination"][_ph][0], mat["Path"][_dh][0]
st.caption(
    f"Destination is scored at **{HORIZON_LABELS[_dh]}** (IC {_d_own:+.3f}); at the "
    f"path's horizon it reads {_d_cross:+.3f}. Path is scored at "
    f"**{HORIZON_LABELS[_ph]}** (IC {_p_own:+.3f}); at the destination's horizon "
    f"{_p_cross:+.3f}. Spearman IC per month, Newey-West t with lag = horizon − 1, "
    f"months with fewer than {min_n} names skipped.")

# ── Alpha term structure ─────────────────────────────────────────────────────
# IC against the return in month t+k alone, k = 1..12 — Blitz, Hanauer,
# Hoogteijling & Howard (2023)'s "term structure of alpha". The picture the
# whole page rests on: the destination should read flat (still paying a year
# out, so it can be held and re-ranked slowly); the path should be
# front-loaded (only next month is informed, so it should only set the pace).
@st.cache_data(show_spinner=False)
def cached_term(sig: str, sample: str, frame: pd.DataFrame, cd: pd.Series,
                cp: pd.Series, min_n: int) -> pd.DataFrame:
    from pace import term_structure
    return term_structure(frame, {"Destination": cd, "Path": cp}, min_n=min_n)


_sub0 = SAMPLES[sample].reset_index(drop=True)
HAS_1M = "fwd_1m" in HORIZONS_AVAILABLE
_ts = (cached_term(ENGINE_SIG + str(dest["rows"]) + str(path["rows"]), sample,
                   _sub0[["date", "ticker", "fwd_1m"]],
                   res["Destination"]["composite"].reset_index(drop=True),
                   res["Path"]["composite"].reset_index(drop=True), min_n)
       if HAS_1M else None)
if _ts is None:
    st.caption("Alpha term structure skipped — it needs 1-month forward returns.")
    t1 = t2 = None
else:
    t1, t2 = st.columns([3, 2])
if t1 is not None:
  with t1:
    _f = go.Figure()
    for nm, col in (("Destination", BLUE), ("Path", RED)):
        _f.add_trace(go.Scatter(x=_ts.index, y=_ts[nm], name=nm, mode="lines+markers",
                                line=dict(color=col, width=2),
                                hovertemplate=nm + " k=%{x}: IC %{y:.3f}<extra></extra>"))
    _f.add_hline(y=0, line_color=INK_2, line_width=1)
    _f.update_xaxes(title="k — return in month t+k alone", dtick=1)
    _f.update_yaxes(title="Spearman IC")
    st.plotly_chart(style(_f, 300), width="stretch")
  with t2:
    _d1, _d12 = _ts.loc[1, "Destination"], _ts.loc[12, "Destination"]
    _p1, _p6 = _ts.loc[1, "Path"], _ts.loc[6, "Path"]
    st.markdown("**Alpha term structure**")
    st.caption(
        f"IC of each edge against the return in month *t+k* on its own, not the "
        f"cumulative return. Destination: {_d1:+.3f} at k=1 → {_d12:+.3f} at k=12 — "
        f"{'still paying a year out' if _d12 > 0.5 * _d1 else 'decaying'}, so it can "
        f"be held and re-ranked slowly. Path: {_p1:+.3f} at k=1 → {_p6:+.3f} at k=6 — "
        f"{'front-loaded' if abs(_p6) < 0.5 * abs(_p1) else 'not front-loaded'}, so it "
        f"should only set the pace, never the destination. This is the term structure "
        f"of alpha in Blitz, Hanauer, Hoogteijling & Howard (2023): short-horizon "
        f"signals carry high gross alpha that costs collapse; long-horizon models load "
        f"on slow value / quality signals that survive net of costs.")

# ── Per-edge results, side by side ───────────────────────────────────────────
st.markdown("#### Each edge on its own horizon")
cols = st.columns(2, gap="large")
checks = {}
for (name, edge), col in zip(EDGES.items(), cols):
    r, hl = res[name], HORIZON_LABELS[edge["horizon"]]
    with col:
        st.markdown(f"**{name}** · {hl}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("IC", f"{r['mean_ic']:+.3f}")
        m2.metric("NW t", f"{r['t_stat']:+.2f}")
        m3.metric("Spread", f"{r['spread_ann']:+.1%}", help="Top − bottom ntile, ann.")
        m4.metric("L/S Sharpe", f"{r['ls_sharpe']:.2f}")
        if _cut is not None and IN_SAMPLE in _opts and OUT_SAMPLE in _opts:
            ck = consistency_checks(_score(edge, SAMPLES[IN_SAMPLE], IN_SAMPLE),
                                    _score(edge, SAMPLES[OUT_SAMPLE], OUT_SAMPLE))
            checks[name] = ck
            _flag = ck["total"] - ck["passed"]
            tone = {"consistent": st.success, "partial": st.warning,
                    "weak": st.warning, "inconsistent": st.error}[ck["verdict"]]
            tone(f"{_flag} of {ck['total']} checks flag · {ck['headline']}")
        tb, tc, ti = st.tabs(["Ntiles", "Growth of $1", "IC"])
        with tb:
            st.plotly_chart(fig_ntile_bars(r, colors, hl, bench_mode), width="stretch")
        with tc:
            st.plotly_chart(fig_ntile_curves(r, colors, bench_mode), width="stretch")
            st.caption("Monthly rebalance into the ntile, equal-weight, gross — "
                       "the *wholesale* rebalance the path edge will later modulate.")
        with ti:
            st.plotly_chart(fig_ic(r, hl), width="stretch")

# ── Agreement ─────────────────────────────────────────────────────────────────
# What the later modulation will act on: for each name, do the two edges point
# the same way? And does the path edge sort returns *within* a destination
# bucket — i.e. is it information the destination didn't already have?
st.markdown("#### How much the two edges agree")
sub = SAMPLES[sample].reset_index(drop=True)
cd_ = res["Destination"]["composite"].reset_index(drop=True)
cp_ = res["Path"]["composite"].reset_index(drop=True)
agree = pd.DataFrame({"date": sub["date"], "ticker": sub["ticker"],
                      "dest": cd_, "path": cp_,
                      "fwd_p": sub[path["horizon"]], "fwd_d": sub[dest["horizon"]]})
agree["q_dest"] = _ntile(agree["dest"], agree["date"], n_q)
agree["q_path"] = _ntile(agree["path"], agree["date"], n_q)
_rho = (agree.groupby("date")[["dest", "path"]]
        .apply(_spearman, "dest", "path", min_n).dropna())
_both = agree.dropna(subset=["q_dest", "q_path"])
_same = float((_both["q_dest"] == _both["q_path"]).mean()) if len(_both) else np.nan
_top_top = float(((_both["q_dest"] == n_q) & (_both["q_path"] == n_q)).mean()
                 / max(((_both["q_dest"] == n_q)).mean(), 1e-9)) if len(_both) else np.nan

a1, a2, a3 = st.columns(3)
a1.metric("Rank correlation", f"{_rho.mean():+.3f}" if len(_rho) else "—",
          help="Spearman between the two composites, per month, averaged. Near "
               "zero is what you want: the path edge is then telling you "
               "something the destination isn't.")
a2.metric("Same ntile", f"{_same:.0%}" if np.isfinite(_same) else "—",
          help=f"Share of name-months where both edges put the name in the same "
               f"of {n_q} ntiles. Chance is {1 / n_q:.0%}.")
a3.metric("Top-ntile confirmed", f"{_top_top:.0%}" if np.isfinite(_top_top) else "—",
          help="Of the destination's top-ntile names, the share the path edge "
               "also puts in its top ntile — the names to buy fastest.")

_ph_months = FWD_COLS[path["horizon"]]
_grid = (_both.groupby(["q_dest", "q_path"])["fwd_p"].mean().unstack())
_grid = ((1.0 + _grid) ** (12.0 / _ph_months) - 1.0)
_grid.index = [f"Dest Q{int(i)}" for i in _grid.index]
_grid.columns = [f"Path Q{int(c)}" for c in _grid.columns]
g1, g2 = st.columns([3, 2])
with g1:
    # No background_gradient: pandas' Styler pulls matplotlib for it, which the
    # deployed box doesn't have (it took the whole page down there).
    st.dataframe(_grid.style.format("{:+.1%}"), width="stretch")
    st.caption(
        f"Mean {HORIZON_LABELS[path['horizon']]} forward return, annualized, by "
        f"destination ntile (rows) × path ntile (columns). Read across a row: if the "
        f"path edge sorts returns *within* a destination bucket, it is information "
        f"the destination didn't have — the case for letting it set the pace. Read "
        f"down a column for the reverse.")
with g2:
    _within = []
    for qd, g in _both.groupby("q_dest"):
        ic = g.groupby("date")[["path", "fwd_p"]].apply(_spearman, "path", "fwd_p",
                                                        max(5, min_n // 3)).dropna()
        if len(ic) >= 12:
            _within.append({"Destination ntile": f"Q{int(qd)}",
                            "Path IC within": ic.mean(),
                            "NW t": newey_west_tstat(ic.to_numpy(), max(_ph_months - 1, 0)),
                            "months": len(ic)})
    if _within:
        st.dataframe(pd.DataFrame(_within).style.format(
            {"Path IC within": "{:+.3f}", "NW t": "{:+.2f}"}),
            width="stretch", hide_index=True)
        st.caption("Path edge's IC computed only among names sharing a destination "
                   "ntile. Positive and significant here means the path edge is "
                   "additive, not a restatement.")
    else:
        st.info("Too few names per destination ntile to score the path edge within it.")

from pace import MODES, coverage_benchmark, deferral_test, run_all, term_structure

if not HAS_1M:
    st.info("Pacing backtest skipped — it needs 1-month forward returns. Upload a price column to get them.")
else:
    # ── 5 · Unite: pace the trades ───────────────────────────────────────────────
    # The destination decides where the book goes; the path edge decides how fast.
    # Three books on identical inputs isolate what the path edge adds: a naive
    # monthly rebalance, the same turnover budget spent by trade size alone, and
    # the budget spent by urgency (direction × centred path rank).

    with st.container(border=True, key="step5"):
        st.subheader("5 · Unite — pace the trades", help="Target = equal weight over the "
                     "top destination ntile. Each month the book moves toward it, but "
                     "only the names whose path signal agrees with the trade — buys "
                     "about to run, sells about to drop — spend the turnover budget. "
                     "The rest wait for a better month.")
        u1, u2, u3, u4 = st.columns([2, 2, 3, 3])
        budget = u1.slider("Turnover budget / month", 0.01, 0.30, 0.08, 0.01, key="dp_budget",
                           format="%.2f", help="Fraction of NAV traded per month, buys "
                           "plus sells. Names leaving coverage are sold outside it.")
        rerank = u2.select_slider("Re-rank destination every", options=[1, 3, 6, 12], value=1,
                                  key="dp_rerank", format_func=lambda m: f"{m} mo",
                                  help="A 12-month signal re-ranked monthly mostly shuffles "
                                       "names across the ntile boundary. Holding the target "
                                       "for a quarter removes that churn at the source; the "
                                       "budget then paces what is left.")
        init = u3.radio("Starting book", ["Cap-weighted coverage", "Equal-weight coverage"],
                        key="dp_init", horizontal=True,
                        help="Where the analyst starts before migrating to the target.")
        u4.caption(f"Target: equal weight over destination ntile Q{n_q} of the coverage, "
                   f"re-ranked every {rerank} month(s). Sample: **{sample}**.")


    @st.cache_data(show_spinner="Pacing the book…")
    def cached_pace(sig: str, sample: str, frame: pd.DataFrame, budget: float, n_q: int,
                    init: str, rerank: int) -> dict:
        res = run_all(frame, budget=budget, n_q=n_q, init=init, rerank_every=rerank)
        return {m: {"cum": r.cum, "summary": r.summary(), "turnover": r.turnover,
                    "distance": r.distance, "deferral": deferral_test(r)}
                for m, r in res.items()}


    _pace_in = pd.DataFrame({
        "date": sub["date"], "ticker": sub["ticker"], "dest": cd_, "path": cp_,
        "fwd_1m": sub["fwd_1m"],
        "log_mcap": sub["log_mcap"] if "log_mcap" in sub else np.nan})
    _pace_sig = f"{ENGINE_SIG}|{panel_key}|{dest['rows']}|{path['rows']}|{dest['neutral']}|{path['neutral']}"
    paced = cached_pace(_pace_sig, sample, _pace_in, float(budget), int(n_q),
                        "cap" if init.startswith("Cap") else "ew", int(rerank))
    bench = coverage_benchmark(_pace_in)
    bench_cum = (1.0 + bench).cumprod()

    LABELS = {"wholesale": f"Wholesale rebalance (k = 1, re-rank {rerank} mo)",
              "uniform": f"Budget {budget:.0%}, by trade size",
              "modulated": f"Budget {budget:.0%}, by urgency (path edge)"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name="Coverage, equal-weight",
                             mode="lines", line=dict(color=INK_2, width=2, dash="dash"),
                             hovertemplate="%{y:.2f}×<extra></extra>"))
    for m, col in zip(MODES, ("#9ec5f4", "#5598e7", BLUE)):
        c = paced[m]["cum"]
        fig.add_trace(go.Scatter(x=c.index, y=c.values, name=LABELS[m], mode="lines",
                                 line=dict(color=col, width=2 if m != "modulated" else 3),
                                 hovertemplate="%{y:.2f}×<extra></extra>"))
    fig.update_yaxes(type="log", title="Growth of $1 (log)")
    st.plotly_chart(style(fig, 380), width="stretch")

    _sum = pd.DataFrame({LABELS[m]: paced[m]["summary"] for m in MODES}).T
    _sum = _sum.rename(columns={"ann_return": "Ann. return", "ann_vol": "Ann. vol",
                                "sharpe": "Sharpe", "max_drawdown": "Max DD",
                                "avg_turnover": "Turnover / mo", "avg_forced": "Forced / mo",
                                "avg_distance": "Dist. to target", "months": "Months"})
    _b = bench.dropna()
    _sum.loc["Coverage, equal-weight"] = {
        "Ann. return": (1 + _b.mean()) ** 12 - 1, "Ann. vol": _b.std() * np.sqrt(12),
        "Sharpe": ((1 + _b.mean()) ** 12 - 1) / (_b.std() * np.sqrt(12)),
        "Max DD": (bench_cum / bench_cum.cummax() - 1).min(), "Turnover / mo": np.nan,
        "Forced / mo": np.nan, "Dist. to target": np.nan, "Months": len(_b)}
    st.dataframe(_sum.style.format({"Ann. return": "{:+.1%}", "Ann. vol": "{:.1%}",
                                    "Sharpe": "{:.2f}", "Max DD": "{:.1%}",
                                    "Turnover / mo": "{:.1%}", "Forced / mo": "{:.1%}",
                                    "Dist. to target": "{:.1%}", "Months": "{:.0f}"},
                                   na_rep="—"), width="stretch")
    _mod, _uni, _who = (paced[m]["summary"] for m in ("modulated", "uniform", "wholesale"))
    st.caption(
        f"Gross of costs, so the budget is the cost model. Read the two gaps separately: "
        f"wholesale → by-size (Sharpe {_who['sharpe']:.2f} → {_uni['sharpe']:.2f}) is what "
        f"trading less does on its own; by-size → by-urgency ({_uni['sharpe']:.2f} → "
        f"**{_mod['sharpe']:.2f}**) is what the path edge adds at the same turnover. "
        f"*Dist. to target* is ½·Σ|x − x*| after trading — how far the paced book lags "
        f"the destination on average ({_mod['avg_distance']:.0%} by urgency vs "
        f"{_uni['avg_distance']:.0%} by size).")

    _dt = paced["modulated"]["deferral"]
    if len(_dt):
        d1, d2 = st.columns([2, 3])
        with d1:
            st.dataframe(_dt.rename(columns={"side": "Side", "claim": "Claim",
                                             "mean_diff": "Mean / mo", "t_stat": "t",
                                             "months": "Months"})
                         .style.format({"Mean / mo": "{:+.2%}", "t": "{:+.2f}"}),
                         width="stretch", hide_index=True)
        with d2:
            st.caption(
                "**Did waiting pay?** For buys: next-month return of the names bought this "
                "month minus the ones the path edge said to wait on — positive means the "
                "deferred buys were indeed cheaper a month later. For sells: deferred minus "
                "executed — positive means the names it kept holding did keep rising. This "
                "is the path edge scored only on the trades the destination actually wanted, "
                "which is the only place it matters.")

# ── Latest cross-section + handoff ───────────────────────────────────────────
st.markdown("#### Latest cross-section")
# Always the most recent month of the whole coverage, whatever sample the
# charts describe: the analyst wants today's ranks, not the end of the
# in-sample window. Composites only — no scoring — so it is cheap.
_full = fpanel.reset_index(drop=True)
_lm = _full["date"] == _full["date"].max()
_latest = pd.DataFrame({
    "date": _full.loc[_lm, "date"], "ticker": _full.loc[_lm, "ticker"],
    "dest": build_composite(_full, EdgeSpec(
        [FeatureSpec(*r) for r in dest["rows"]], sector_neutral=dest["neutral"]))[_lm],
    "path": build_composite(_full, EdgeSpec(
        [FeatureSpec(*r) for r in path["rows"]], sector_neutral=path["neutral"]))[_lm],
}).dropna(subset=["dest", "path"])
_latest["q_dest"] = _ntile(_latest["dest"], _latest["date"], n_q)
_latest["q_path"] = _ntile(_latest["path"], _latest["date"], n_q)
_latest = _latest.assign(
    agree=np.where(_latest["q_dest"] == _latest["q_path"], "✓", ""))
_show = (_latest.sort_values("dest", ascending=False)
         [["ticker", "dest", "q_dest", "path", "q_path", "agree"]]
         .rename(columns={"dest": "Destination score", "q_dest": "Dest ntile",
                          "path": "Path score", "q_path": "Path ntile",
                          "agree": "Same ntile"}))
st.dataframe(_show.style.format({"Destination score": "{:.3f}", "Path score": "{:.3f}",
                                 "Dest ntile": "{:.0f}", "Path ntile": "{:.0f}"}),
             width="stretch", hide_index=True, height=min(600, 40 + 35 * len(_show)))
st.caption(f"{_latest['date'].max():%Y-%m-%d} · {len(_show)} names in coverage, "
           f"destination-ranked. ✓ marks names both edges put in the same ntile.")

# Everything the unite step will need, kept in session for the next page.
st.session_state["destination_path"] = {
    "scores": agree[["date", "ticker", "dest", "path", "q_dest", "q_path"]],
    "dest": dest, "path": path, "n_q": n_q, "sample": sample,
    "panel_key": panel_key,
}
st.download_button(
    "Scores (CSV)", agree.to_csv(index=False), file_name="destination_path_scores.csv",
    mime="text/csv", help="Both composites and both ntiles, per name per date.")

with st.expander("How this works", expanded=False):
    st.markdown(f"""
Two edges are built on the **same coverage** and scored on **different horizons**.

**Destination** is the slow edge — where the book should be a year from now. It is
scored on the {HORIZON_LABELS[DEST_HORIZON]} forward return by default. If you
already have a ranking (an analyst score, a model output), bring it in as a column
and make it the only row with weight 1.

**Path** is the fast edge — what is known about the *next month*. It is scored on
the {HORIZON_LABELS[PATH_HORIZON]} return by default. Its job is not to pick the
portfolio; it is to say, name by name, whether now is a good month to move toward
the destination or a good month to wait.

**The matrix** scores both edges at every horizon. The pattern you want is a
destination that is strong at 12 months, and a path that is strong at 1 month
*and* close to uncorrelated with the destination. A path edge that is just the
destination again — same names, same order — cannot modulate anything.

**Agreement** is the raw material for the next step. Where both edges put a name in
the top ntile, the book moves there fast; where the destination says buy and the
path says not yet, it moves slowly. The within-ntile table is the test of whether
that distinction is worth making: the path edge's IC computed only among names the
destination already ranks alike.

**Not yet here:** the modulated backtest — wholesale monthly rebalance versus
transaction speed set by the path edge. This page fixes the two edges it will run
on; the scores are kept in session for it.
""")
