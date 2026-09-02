"""
Edge Concierge — build and test factor / edge candidates.

Select features from the panel, transform (rank / zscore / raw) and weight them
into a composite, then test it: IC with Newey-West t-stats, ntile portfolio
returns, long-short spread, sector breakdown.

Run:  streamlit run main.py  (this is a page, not the entry point)
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_io import (REFERENCE_LABELS, AmbiguousDates, ambiguous_date,
                     load_base_panel, load_benchmarks, normalize_csv,
                     reference_series)
from report import build_report as _build_report
from ui import (BASELINE, BLUE, GRID, INK, INK_2, MUTED, RED, SURFACE,
                ramp, style)
from engine import (BENCH_COL, BENCH_REF, FWD_COLS, OPS,
                    TRANSFORMS, Constraint, EdgeSpec, FeatureSpec,
                    benchmark_options, consistency_checks, feature_columns,
                    run_edge)

# Palette and chart styling live in ui.py, shared with the other tool.

BASE_PANEL_LABEL = "S&P 500 · 1998–2025 · basic factors"

HORIZON_LABELS = {"fwd_1m": "1 month", "fwd_3m": "3 months",
                  "fwd_6m": "6 months", "fwd_12m": "12 months"}

# ── The example edge the app opens with ──────────────────────────────────────
# A measured starting point, not a toy: at the default 1-month horizon this
# scores IC +0.018 (Newey-West t +3.2), a +7.8% annualized top-minus-bottom
# spread and a 0.81 long-short Sharpe; it holds at 12 months too (t +2.8) and
# stays positive in every decade of the sample.
# Economically it is cash-generative + profitable + clean-accounting + not
# over-expanding — four categories, two of them entered with negative weight.
# Communication Services is dropped because it is the one sector where the
# signal genuinely fails (IC -0.003).
EXAMPLE_ROWS = [
    {"feature": "fcf_yield", "transform": "rank", "weight": 1.0},    # value
    {"feature": "gpa",       "transform": "rank", "weight": 0.5},    # quality
    {"feature": "accruals",  "transform": "rank", "weight": -0.5},   # safety
    {"feature": "asset_gr",  "transform": "rank", "weight": -0.5},   # discipline
]
EXAMPLE_SECTORS = ["Consumer Cyclical", "Consumer Defensive"]
EXAMPLE_CONSTRAINTS = [{"left": "mom_6_1", "op": ">", "right": "0"}]
EXAMPLE_PROMPT = ("cash-generative, profitable companies with clean accounting "
                  "that aren't over-expanding — consumer names only, and only "
                  "ones with positive 6-month momentum")


def default_builder_rows(features: list[str]) -> pd.DataFrame:
    """The example spec, minus any feature this panel doesn't have."""
    rows = [r for r in EXAMPLE_ROWS if r["feature"] in features]
    if not rows:   # custom CSV with its own columns
        rows = [{"feature": features[0], "transform": "rank", "weight": 1.0}]
    return pd.DataFrame(rows)


# ── Figure builders ──────────────────────────────────────────────────────────
# Defined once and used by both the on-screen tabs and the saved HTML report,
# so the report is the same chart rather than a second implementation of it.

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
    return style(fig, 380)


def fig_ntile_curves(res, colors, bench_label) -> go.Figure:
    q_cum = res["q_cum"]
    fig = go.Figure()
    bc = res["bench_cum"].dropna()
    if len(bc):
        fig.add_trace(go.Scatter(
            x=bc.index, y=bc.values, name=bench_label, mode="lines",
            line=dict(color=INK_2, width=2, dash="dash"),
            hovertemplate="Universe: %{y:.2f}×<extra></extra>"))
    # direct-label only the extremes: the middle ntiles run too close together
    # to label without collisions, and the legend carries identity
    extremes = {q_cum.columns[0], q_cum.columns[-1]}
    for i, q in enumerate(q_cum.columns):
        last = q_cum[q].dropna()
        fig.add_trace(go.Scatter(
            x=q_cum.index, y=q_cum[q], name=f"Q{int(q)}", mode="lines",
            line=dict(color=colors[i], width=2),
            hovertemplate="Q" + str(int(q)) + ": %{y:.2f}×<extra></extra>"))
        if len(last) and q in extremes:
            fig.add_annotation(x=last.index[-1], y=np.log10(last.iloc[-1]),
                               text=f"Q{int(q)}", showarrow=False,
                               xanchor="left", xshift=4,
                               font=dict(color=colors[i], size=12))
    # plotly labels every minor decade tick on a log axis by default, which is
    # unreadable over a 25x range — pin a sparse, explicit set instead
    fig.update_yaxes(type="log", title="Growth of $1 (log)",
                     tickmode="array", tickvals=[1, 2, 5, 10, 20, 50, 100],
                     ticktext=["1×", "2×", "5×", "10×", "20×", "50×", "100×"])
    return style(fig, 380)


def fig_long_short(res) -> go.Figure | None:
    ls_cum = res["ls_cum"]
    if not len(ls_cum):
        return None
    fig = go.Figure(go.Scatter(
        x=ls_cum.index, y=ls_cum.values, mode="lines",
        line=dict(color=BLUE, width=2), name="Long–short",
        hovertemplate="%{y:.2f}×<extra></extra>"))
    fig.update_yaxes(title="Growth of $1")
    return style(fig, 360)


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
    return style(fig, 380)


def fig_sector(res) -> go.Figure | None:
    sic = res["sector_ic"]
    if not len(sic):
        return None
    sic = sic.sort_values("mean_ic")
    fig = go.Figure(go.Bar(
        x=sic["mean_ic"], y=sic["sector"], orientation="h",
        marker_color=[BLUE if v >= 0 else RED for v in sic["mean_ic"]],
        marker_line_color=SURFACE, marker_line_width=2,
        hovertemplate="%{y}: IC %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Mean IC")
    return style(fig, 400)


# ── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading base panel…")
def base_panel() -> pd.DataFrame:
    return load_base_panel()


def read_upload(data: bytes) -> pd.DataFrame:
    """
    Parse an uploaded CSV once per session, not on every widget rerun.

    Deliberately memoised in session_state rather than st.cache_data: that cache
    is process-wide, so a user's parsed data would sit in shared app memory and
    outlive their session. Session state is scoped to the uploader and dropped
    when they disconnect.
    """
    digest = hashlib.md5(data).hexdigest()
    if st.session_state.get("_upload_digest") != digest:
        with st.spinner("Reading CSV…"):
            st.session_state["_upload_df"] = pd.read_csv(io.BytesIO(data))
        st.session_state["_upload_digest"] = digest
    return st.session_state["_upload_df"]


# Cached results are keyed on the engine source too: st.cache_data only hashes
# this function's own body, so an edit to engine.py would otherwise serve stale
# results computed by the previous version.
import engine as _engine_mod
ENGINE_SIG = hashlib.md5(
    pathlib.Path(_engine_mod.__file__).read_bytes()).hexdigest()[:8]


@st.cache_data(show_spinner="Testing edge…")
def cached_run(engine_sig: str, panel_key: str, panel: pd.DataFrame,
               spec_rows: tuple, sector_neutral: bool, horizon: str, n_q: int,
               bench_mode: str, bench_col: str | None,
               bench_ref: str | None, cons_rows: tuple = ()) -> dict:
    spec = EdgeSpec(features=[FeatureSpec(*r) for r in spec_rows],
                    sector_neutral=sector_neutral,
                    constraints=[Constraint(*c) for c in cons_rows])
    # built here, not passed in, so the cache key stays a short string
    series = reference_series(bench_ref, panel["date"]) if bench_ref else None
    return run_edge(panel, spec, horizon=horizon, n_q=n_q,
                    bench_mode=bench_mode, bench_col=bench_col,
                    bench_series=series)


# ── Sidebar: data + universe ─────────────────────────────────────────────────

st.title("Edge Concierge")


with st.container(border=True, key="step1"):
    st.subheader('1 · Panel', help='The data every later step is measured on. The shipped panel is point-in-time S&P 500 membership with 24 raw features and forward returns; an upload replaces it wholesale.')
    source = st.radio("Panel", [BASE_PANEL_LABEL, "Upload CSV"],
                      help="Point-in-time S&P 500 constituents (quarterly "
                           "membership snapshots, no survivorship bias), monthly, "
                           "1998 → Dec 2025 with post-2025 held out-of-sample. "
                           "24 raw value / quality / safety / growth / momentum / "
                           "size features, plus forward returns at 1/3/6/12m.")

    panel = None
    panel_key = "base"
    if source == BASE_PANEL_LABEL:
        panel = base_panel()
    else:
        up = st.file_uploader("CSV — one row per security per date", type="csv")
        st.caption(
            "Held in memory for this session only — never written to disk, "
            "never shared between users, and dropped when you close the tab.",
            help="The file goes to the app server's memory, keyed to your "
                 "session, and is released when the session ends. Nothing is "
                 "persisted, so a restart or a closed tab leaves no copy. Note "
                 "the app itself is open-access: anyone with the link can use "
                 "it, though they cannot see your data.")
        if up is not None:
            raw = read_upload(up.getvalue())
            cols = list(raw.columns)
            if len(raw) > 1_500_000:
                st.warning(f"{len(raw):,} rows — large panels are slow to test "
                           "and memory-hungry. Consider pre-filtering.")

            def _guess(names, fallback=0):
                """Pre-select the obvious column so the common file just works."""
                for i, c in enumerate(cols):
                    if any(n in c.lower().replace(" ", "") for n in names):
                        return i
                return fallback

            st.caption(f"{len(raw):,} rows · {len(cols)} columns. Check the "
                       f"mapping below — the guesses are usually right.")
            id_col = st.selectbox("Security ID column", cols,
                                  index=_guess(("ticker", "symbol", "company",
                                                "id", "name", "sedol", "cusip")))
            date_col = st.selectbox("Date column", cols,
                                    index=_guess(("date", "month", "period",
                                                  "asof"), min(1, len(cols) - 1)))

            _dex = ambiguous_date(raw[date_col])
            if _dex is not None:
                st.error(f"`{date_col}` is ambiguous — {_dex!r} could be day-first "
                         f"or month-first. Re-save it as **YYYY-MM-DD** and upload "
                         f"again; reading it either way would silently reorder the "
                         f"calendar and corrupt every forward return.")

            opt = ["(none)"] + cols
            sector_col = st.selectbox("Sector column (optional)", opt,
                                      index=_guess(("sector",)) + 1
                                      if any("sector" in c.lower() for c in cols) else 0)
            industry_col = st.selectbox("Industry column (optional)", opt,
                                        index=_guess(("industry",)) + 1
                                        if any("industry" in c.lower() for c in cols) else 0)

            _has_px = any(n in c.lower() for c in cols
                          for n in ("price", "close", "px", "adj"))
            ret_src = st.radio("Forward returns come from…", [
                "A price column (computed at 1/3/6/12m)",
                "A forward-return column",
                "Join from base panel by ticker",
            ], index=0 if _has_px else 2,
                help="Your file needs a way to measure what happened next. If it "
                     "has neither prices nor forward returns, the last option "
                     "borrows them from the shipped S&P 500 panel — which only "
                     "works if your IDs are US tickers.")
            price_col = fwd_map = None
            join_base = False
            if ret_src.startswith("A price"):
                price_col = st.selectbox("Price column", cols, index=_guess(
                    ("price", "close", "px", "adj")))
            elif ret_src.startswith("A forward"):
                fcol = st.selectbox("Forward-return column", cols)
                fhor = st.selectbox("…measured over", list(FWD_COLS),
                                    format_func=HORIZON_LABELS.get)
                fwd_map = {fhor: fcol}
            else:
                join_base = True
                st.caption("Security IDs must be US tickers for the join.")
            st.caption("Every other numeric column becomes a selectable feature — "
                       "including a benchmark return series, which you then pick "
                       "under **Benchmark** in Test setup.")
            if st.button("Load CSV", type="primary", width="stretch",
                         disabled=_dex is not None):
                try:
                    st.session_state["csv_panel"] = normalize_csv(
                        raw, id_col, date_col,
                        None if sector_col == "(none)" else sector_col,
                        None if industry_col == "(none)" else industry_col,
                        price_col=price_col, fwd_map=fwd_map,
                        join_base_returns=join_base)
                except AmbiguousDates as e:
                    st.error(str(e))

        if "csv_panel" in st.session_state:
            panel = st.session_state["csv_panel"]
            panel_key = "csv"
            _cov = panel[["fwd_1m", "fwd_3m", "fwd_6m", "fwd_12m"]].notna().any(axis=1).mean()
            st.success(f"{panel['ticker'].nunique():,} securities · "
                       f"{panel['date'].nunique():,} dates · "
                       f"{panel['date'].min():%b %Y}–{panel['date'].max():%b %Y}")
            if _cov < 0.5:
                st.error(f"Only {_cov:.0%} of rows have a forward return. The "
                         f"return source is probably mismapped — nothing can be "
                         f"tested without it.")
            elif _cov < 0.95:
                st.warning(f"{_cov:.0%} of rows have a forward return.")
            with st.expander("Check what was read"):
                st.caption("Dates as parsed, and the features found:")
                st.dataframe(panel.head(6), width="stretch", hide_index=True)
                st.caption(", ".join(f"`{c}`" for c in feature_columns(panel))
                           or "no numeric features found")

    if panel is None:
        st.stop()


# ── Learned Edge handoff ─────────────────────────────────────────────────────
# A tilt sent over from the Learned Edge becomes an ordinary feature here, so it
# can be weighted or multiplied like any other leg. Joined on date and security
# id; names with no tilt get the neutral 0.5 rather than being dropped, so the
# layer is inert outside its coverage instead of shrinking the universe.
_tilt = st.session_state.get("learned_tilt_panel")
if _tilt is not None and "learned_tilt" not in panel.columns:
    try:
        t = _tilt.copy()
        t["date"] = pd.to_datetime(t["date"])
        t["secid"] = t["secid"].astype(str).str.strip().str.upper()
        t = t.groupby(["date", "secid"], as_index=False)["learned_tilt"].mean()
        panel = panel.merge(
            t.rename(columns={"secid": "ticker"}), on=["date", "ticker"], how="left")
        _cov = panel["learned_tilt"].notna().mean()
        panel["learned_tilt"] = panel["learned_tilt"].fillna(0.5)
        st.caption(f"Learned Edge tilt merged in as `learned_tilt` — covers "
                   f"**{_cov:.0%}** of panel rows; the rest sit at a neutral 0.5.")
    except Exception as e:
        st.warning(f"Couldn't merge the learned tilt: {e}")

features = feature_columns(panel)

if "builder" not in st.session_state:
    st.session_state["builder"] = default_builder_rows(features)


def spec_formula(rows: list[tuple], sector_neutral: bool,
                 cons: list[tuple] | None = None) -> str:
    parts = []
    for i, (feat, tr, w) in enumerate(rows):
        term = feat if tr == "raw" else f"{tr}({feat})"
        coef = "" if abs(w) == 1 else f"{abs(w):g}·"
        sign = ("−" if w < 0 else "") if i == 0 else (" − " if w < 0 else " + ")
        parts.append(f"{sign}{coef}{term}")
    out = "edge = " + "".join(parts)
    if cons:
        out += "   where " + " and ".join(f"{l} {o} {r}" for l, o, r in cons)
    if sector_neutral:
        out += "   (transforms within sector)"
    return out


def run_title(rows: list[tuple], horizon: str, n_q: int, stamp: datetime) -> str:
    """A short human title: the heaviest legs, the horizon, the ntile count."""
    top = sorted(rows, key=lambda r: -abs(r[2]))[:3]
    legs = "".join(("−" if w < 0 else ("" if i == 0 else "+")) + f
                   for i, (f, _, w) in enumerate(top))
    if len(rows) > len(top):
        legs += f"+{len(rows) - len(top)} more"
    return f"{legs} · {HORIZON_LABELS[horizon]} · {n_q} ntiles · {stamp:%Y-%m-%d %H:%M}"


def run_slug(rows: list[tuple], horizon: str, stamp: datetime) -> str:
    """Filesystem-safe filename stem. Bounded length, no collisions per second."""
    top = sorted(rows, key=lambda r: -abs(r[2]))[:3]
    parts = "-".join(re.sub(r"[^0-9a-zA-Z]+", "", f) for f, _, _ in top) or "edge"
    return f"{stamp:%Y%m%d-%H%M%S}_{parts[:60]}_{horizon.replace('fwd_', '')}"


def _jsonable(o):
    """numpy scalars and pandas types are not JSON-serialisable on their own."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (np.ndarray, pd.Series)):
        return o.tolist()
    return str(o)


def _num(x):
    """Plain float, with NaN/inf collapsed to None so the JSON stays valid."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def live_constraints() -> list[tuple]:
    """Constraint rows including uncommitted editor deltas (same trick as below)."""
    base = st.session_state.get("constraints")
    if base is None or base.empty:
        base = pd.DataFrame(columns=["left", "op", "right"])
    base = base.copy().reset_index(drop=True)
    state = st.session_state.get("cons_editor") or {}
    for idx, changes in state.get("edited_rows", {}).items():
        for col, val in changes.items():
            base.loc[int(idx), col] = val
    if state.get("added_rows"):
        base = pd.concat([base, pd.DataFrame(state["added_rows"])], ignore_index=True)
    if state.get("deleted_rows"):
        base = base.drop(index=[i for i in state["deleted_rows"] if i in base.index])
    base = base.dropna(subset=["left", "op", "right"])
    return [(r.left, r.op, str(r.right).strip()) for r in base.itertuples()
            if str(r.right).strip()]


def live_rows() -> list[tuple]:
    """Builder-table rows including uncommitted data_editor deltas.

    The builder table is the single source of truth for the spec; this only
    reads its state so the formula can render above it in script order.
    """
    base = st.session_state["builder"].copy().reset_index(drop=True)
    state = st.session_state.get("builder_editor") or {}
    for idx, changes in state.get("edited_rows", {}).items():
        for col, val in changes.items():
            base.loc[int(idx), col] = val
    if state.get("added_rows"):
        base = pd.concat([base, pd.DataFrame(state["added_rows"])],
                         ignore_index=True)
    if state.get("deleted_rows"):
        base = base.drop(index=[i for i in state["deleted_rows"]
                                if i in base.index])
    base = base.dropna(subset=["feature", "transform", "weight"])
    return [(r.feature, r.transform, float(r.weight))
            for r in base.itertuples() if r.weight != 0]


if "constraints" not in st.session_state:
    st.session_state["constraints"] = pd.DataFrame(
        [c for c in EXAMPLE_CONSTRAINTS
         if c["left"] in feature_columns(panel)],
        columns=["left", "op", "right"])

# ── Universe screen ──────────────────────────────────────────────────────────
# Sector/industry pickers and the numeric conditions are the same operation:
# both decide which names are eligible before anything is ranked. They used to
# sit on opposite sides of the screen — categorical ones hidden in the sidebar,
# numeric ones in the body — which made a sector exclusion easy to forget was
# even on. One visible section, one summary of what survives.
sectors = sorted(panel["sector"].dropna().unique())
if "sectors_w" not in st.session_state:          # open on the example universe
    _inc = [x for x in EXAMPLE_SECTORS if x in sectors]
    st.session_state["sectors_w"] = _inc
    st.session_state["grain_w"] = "Sector" if _inc else "Everything"
if "nl_pending" in st.session_state:
    plan_sectors, plan_horizon = st.session_state.pop("nl_pending")
    # Only override a filter the description actually spoke to: an empty sector
    # list means "not mentioned", not "clear what you set".
    if plan_sectors:
        st.session_state["sectors_w"] = [x for x in plan_sectors if x in sectors]
        st.session_state["grain_w"] = "Sector"   # else the list is never applied
    if plan_horizon in FWD_COLS:
        st.session_state["horizon_w"] = plan_horizon
# drop stale selections (base-panel sectors left over after a CSV load) — the
# multiselect raises if its value is not a subset of its options
st.session_state["sectors_w"] = [x for x in st.session_state["sectors_w"]
                                 if x in sectors]

with st.container(border=True, key="step2"):
    st.subheader('2 · Sketch the edge', help='Start here. Describe the whole idea in plain English — the factors and the screens together — and it fills in the universe and the edge table below, where you can correct anything. This box is a drafting aid: nothing in it is tested directly.')

    # Optional natural-language builder (needs ANTHROPIC_API_KEY)
    try:
        from nl import api_key, explain_validation, plan_edge
        _nl_key = api_key()
    except Exception:
        _nl_key = None
    if _nl_key:
        with st.expander("Describe your edge in English", expanded=True):
            # a text area, not an input: the sample prompt is a sentence and a
            # single-line box clips it, and people describing an edge tend to
            # write more than one line anyway
            desc = st.text_area(
                "Describe your edge — name the features and weights explicitly, or "
                "just say what you're after", height=86,
                placeholder=f"Current settings are this sample edge — "
                        f"{EXAMPLE_PROMPT}. Describe your own to replace it.",
            key="nl_desc",
                help="Both ends of the spectrum work. Loose: “cheap profitable "
                     "names, nothing over-leveraged.” Explicit: “fcf_yield 1.0, "
                     "gpa 0.5, accruals −0.5, asset_gr −0.5, drop financials.” "
                     "Whatever you write lands in the table below, where you can "
                     "correct it.")
            c_btn, c_formula = st.columns([1, 4], vertical_alignment="center")
            cur = live_rows()
            if cur:
                # drawn above the sector-neutral toggle, so read session state
                # rather than the not-yet-assigned name
                _neutral = st.session_state.get("neutral_w", False)
                _f = spec_formula(cur, _neutral, live_constraints())
                # the placeholder disappears the moment they type, so mark the
                # formula too — otherwise the sample silently reads as theirs
                _sample_rows = [(r["feature"], r["transform"], r["weight"])
                                for r in EXAMPLE_ROWS]
                _sample_cons = [(c["left"], c["op"], c["right"])
                                for c in EXAMPLE_CONSTRAINTS]
                _is_sample = (list(cur) == _sample_rows
                              and live_constraints() == _sample_cons)
                c_formula.markdown(
                    (":grey[sample &nbsp;]" if _is_sample else "") + f"`{_f}`")
            if c_btn.button("Draft spec") and desc:
                try:
                    # a status box rather than a bare spinner: the call runs
                    # 10-30s and a silent button reads as a hang. No progress
                    # bar — the API gives no progress to report, and a bar that
                    # invents one is a lie about how far along it is.
                    with st.status("Reading your description…", expanded=False) as _s:
                        plan = plan_edge(desc, features,
                                         sorted(panel["sector"].dropna().unique()),
                                         _nl_key)
                        _s.update(label="Spec drafted", state="complete")
                    st.session_state["builder"] = pd.DataFrame(
                        [r.model_dump() for r in plan.rows])
                    st.session_state.pop("builder_editor", None)
                    st.session_state["constraints"] = pd.DataFrame(
                        [c.model_dump() for c in plan.constraints],
                        columns=["left", "op", "right"])
                    st.session_state.pop("cons_editor", None)
                    st.session_state["nl_pending"] = (plan.sectors, plan.horizon)
                    st.session_state["nl_note"] = plan.rationale
                    st.rerun()
                except Exception as e:
                    st.error(f"Couldn't draft a spec: {e}")
            if st.session_state.get("nl_note"):
                st.caption(f"→ {st.session_state['nl_note']}")
                # its own line, not a parenthetical: this is the handoff from
                # the sketch to the thing that actually runs, and it is the one
                # instruction a first-time user needs
                st.markdown(
                    ":grey[**Adjust anything below.** Every feature, weight and "
                    "condition it picked is yours to change — the universe in "
                    "step 3, the table in step 4. This description is only how "
                    "it got there; the table is what gets tested.]")

with st.container(border=True, key="step3"):
    st.subheader('3 · Universe', help='Which names are eligible, decided before anything is ranked. Sector or industry cuts, a date range, and conditions on raw feature values. Names that fail a condition — or are missing either side of it — are dropped from that month, so the survivors are ranked only against each other.')
    # Industries nest strictly inside sectors (verified on the panel: 133
    # industries, each in exactly one of 11 sectors). Screening both at once is
    # therefore redundant when the industries sit inside the chosen sectors, and
    # yields an empty universe when they do not — so pick a granularity instead of
    # offering two filters that can contradict each other.
    _grain = st.radio(
        "Screen names by", ["Everything", "Sector", "Industry"], horizontal=True,
        key="grain_w",
        help="Industries sit inside sectors, so choose the level you want to cut "
             "at rather than filtering on both.")

    sel_sectors, sel_industries = [], []
    if _grain == "Sector":
        sel_sectors = st.multiselect(
            "Sectors to include", sectors, placeholder="All sectors",
            key="sectors_w")
    elif _grain == "Industry":
        sel_industries = st.multiselect(
            "Industries to include",
            sorted(panel["industry"].dropna().unique()),
            placeholder="All industries", key="industries_w")

    _yr_min, _yr_max = int(panel["date"].min().year), int(panel["date"].max().year)
    yr_range = st.columns([2, 3])[0].slider(
        "Date range", _yr_min, _yr_max, (_yr_min, _yr_max), key="years_w")

    st.markdown("Conditions", help="On **raw** feature values: against a number "
                "(`roa > 0`) or another feature (`roa > asset_gr`).")
    cons_edited = st.columns([5, 1])[0].data_editor(
        st.session_state["constraints"],
        num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "left": st.column_config.SelectboxColumn(
                "Feature", options=features, width="medium"),
            "op": st.column_config.SelectboxColumn("Is", options=list(OPS),
                                                   width="small"),
            "right": st.column_config.TextColumn(
                "Than", width="medium",
                help="A number (0, -0.1, 0.25) or another feature name."),
        },
        key="cons_editor",
    )

    cons = live_constraints()

    # apply the screen
    mask = panel["date"].dt.year.between(*yr_range)
    if sel_sectors:
        mask &= panel["sector"].isin(sel_sectors)
    if sel_industries:
        mask &= panel["industry"].isin(sel_industries)
    fpanel = panel.loc[mask].reset_index(drop=True)

    _kept = f"**{fpanel['ticker'].nunique():,}** securities · **{fpanel['date'].nunique():,}** months"
    _dropped = []
    if sel_sectors and len(sel_sectors) < len(sectors):
        _dropped.append(f"{len(sectors) - len(sel_sectors)} sector(s) excluded")
    if sel_industries:
        _dropped.append(f"{len(sel_industries)} industries kept")
    if (yr_range[0], yr_range[1]) != (_yr_min, _yr_max):
        _dropped.append(f"{yr_range[0]}–{yr_range[1]}")
    if cons:
        _dropped.append(f"{len(cons)} condition(s)")
    st.caption(f"After screening: {_kept}"
               + (f" — {', '.join(_dropped)}." if _dropped else " — no screen applied."))


with st.container(border=True, key="step4"):
    st.subheader('4 · The edge', help='The single source of truth. The sketch only writes into this table and the test reads only from it, so tweak weights here to search the space — what you see is exactly what runs.')

    edited = st.columns([5, 1])[0].data_editor(
        st.session_state["builder"],
        num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "feature": st.column_config.SelectboxColumn(
                "Feature", options=features, required=True, width="medium"),
            "transform": st.column_config.SelectboxColumn(
                "Transform", options=TRANSFORMS, required=True,
                help="rank: cross-sectional percentile [0,1] · zscore: winsorized z "
                     "· raw: winsorized value"),
            "weight": st.column_config.NumberColumn(
                "Weight", step=0.25, help="Negative weight flips the signal."),
        },
        key="builder_editor",
    )

    rows = [(r.feature, r.transform, float(r.weight))
            for r in edited.dropna(subset=["feature", "weight"]).itertuples()
            if r.weight != 0]

    sector_neutral = st.columns([3, 2])[0].toggle(
        "Sector-neutral (transform within sector)", key="neutral_w",
        help="Rank each name against its own sector rather than the whole "
             "cross-section — strips the sector bet out of the edge.")  # part of how the composite is built, not how it is scored



with st.container(border=True, key="step5"):
    st.subheader('5 · Test', help='How the edge is scored: the forward-return horizon, how many ntile portfolios to cut it into, and what those portfolios are measured against.')
    _t1, _t2, _t3, _t4 = st.columns(4)
    horizon = _t1.selectbox("Return horizon", list(FWD_COLS), index=0,
                            format_func=HORIZON_LABELS.get, key="horizon_w")
    n_q = _t2.select_slider("Ntile portfolios", options=[3, 4, 5, 10], value=5)
    holdout_yrs = _t4.number_input(
        "Holdout (years)", min_value=0, max_value=20, value=5, step=1,
        key="holdout_w",
        help="The most recent N years are held back as a validation sample. "
             "Build on in-sample, then check the holdout — an edge that only "
             "works in-sample is a fit, not a finding. Separate from the panel's "
             "own hard cutoff, which excludes everything from Jan 2026 onward.")

    _refs = [c for c in load_benchmarks().columns] if not load_benchmarks().empty else []
    _bench_opts = (benchmark_options(panel)
                   + [f"Reference: {t} — {REFERENCE_LABELS.get(t, t)}" for t in _refs]
                   + [f"Column: {c}" for c in feature_columns(panel)])
    bench_choice = _t3.selectbox(
        "Benchmark", _bench_opts, key="bench_w",
        help="What the ntile portfolios are compared against on the charts. "
             "The universe options are computed from the scored names in your "
             "current filter. Pick a column if your panel carries its own "
             "benchmark return series (one value repeated per date).")
    bench_ref = None
    bench_mode, bench_col = bench_choice, None
    if bench_choice.startswith("Column: "):
        bench_mode, bench_col = BENCH_COL, bench_choice[len("Column: "):]
    elif bench_choice.startswith("Reference: "):
        bench_mode = BENCH_REF
        bench_ref = bench_choice[len("Reference: "):].split(" — ")[0]




    st.button("Test edge", type="primary", disabled=not rows,
              help="Results refresh on their own as you edit — this just forces a recompute.")
if not rows:
    st.info("Add at least one feature row with a non-zero weight.")
    st.stop()
# No run gate: the app opens on the example edge already tested, so the charts
# show what the tool does before you touch anything. Repeat specs are cached.

if fpanel[horizon].notna().sum() < 1000:
    st.warning("Not enough forward-return data in this slice for the selected "
               "horizon — widen the universe or pick another horizon.")
    st.stop()

# A screen can empty the cross-section; catch that before the metrics go NaN.
if cons:
    from engine import constraint_mask, unresolved_constraints
    _bad = unresolved_constraints(fpanel, [Constraint(*c) for c in cons])
    for _c, _why in _bad:
        st.warning(f"Constraint `{_c.left} {_c.op} {_c.right}` is being ignored — "
                   f"{_why}.")
    cons = [c for c in cons
            if (c[0], c[1], c[2]) not in {(b.left, b.op, b.right) for b, _ in _bad}]

if cons:
    _m = constraint_mask(fpanel, [Constraint(*c) for c in cons])
    _per_date = _m.groupby(fpanel["date"]).sum()
    if not _m.any():
        st.error("The constraints exclude every name — nothing left to rank. "
                 "Loosen a threshold or check the feature you're comparing against.")
        st.stop()
    if _per_date.max() < n_q * 5:
        st.error(f"The constraints leave at most {int(_per_date.max())} names in "
                 f"any month, too few for {n_q} ntiles. Loosen them or reduce the "
                 f"ntile count.")
        st.stop()

# ── Validation split ─────────────────────────────────────────────────────────
# The holdout is the most recent slice, never a random one: returns are a time
# series, so sampling dates at random leaks the future into the training set.
IN_SAMPLE, OUT_SAMPLE, ALL_SAMPLE = "In-sample", "Holdout", "All"
_last = fpanel["date"].max()
_cut = _last - pd.DateOffset(years=int(holdout_yrs)) if holdout_yrs else None

if _cut is None:
    SAMPLES = {ALL_SAMPLE: fpanel}
else:
    SAMPLES = {IN_SAMPLE: fpanel[fpanel["date"] < _cut],
               OUT_SAMPLE: fpanel[fpanel["date"] >= _cut],
               ALL_SAMPLE: fpanel}

_opts = [k for k, v in SAMPLES.items() if v["date"].nunique() >= 12]
if not _opts:
    st.error("No sample has enough months to score — reduce the holdout or "
             "widen the date range.")
    st.stop()

_c1, _c2 = st.columns([2, 3], vertical_alignment="bottom")
sample = _c1.segmented_control(
    "Sample", _opts, default=_opts[0], key="sample_w",
    help="Which slice of history the metrics and charts below describe. Build "
         "on in-sample, then check the holdout: an edge that only works "
         "in-sample is a fit, not a finding.")
sample = sample or _opts[0]
if _cut is not None:
    _c2.caption(
        f"In-sample {fpanel['date'].min():%Y-%m} – {_cut:%Y-%m} "
        f"({SAMPLES[IN_SAMPLE]['date'].nunique()} months) · holdout "
        f"{_cut:%Y-%m} – {_last:%Y-%m} "
        f"({SAMPLES[OUT_SAMPLE]['date'].nunique()} months)")

def _score(sub):
    return cached_run(ENGINE_SIG, panel_key, sub, tuple(rows), sector_neutral,
                      horizon, n_q, bench_mode, bench_col, bench_ref, tuple(cons))

res = _score(SAMPLES[sample])

# Validation runs by default, not on request: the in-sample number alone is the
# one most likely to be believed and least likely to be true.
checks = None
if _cut is not None and IN_SAMPLE in _opts and OUT_SAMPLE in _opts:
    checks = consistency_checks(_score(SAMPLES[IN_SAMPLE]),
                                _score(SAMPLES[OUT_SAMPLE]))
    _flagged = checks["total"] - checks["passed"]
    _tone = {"consistent": st.success, "partial": st.warning,
             "weak": st.warning, "inconsistent": st.error}[checks["verdict"]]
    _tone(f"**{_flagged} of {checks['total']} consistency checks flag a problem.** "
          f"{checks['headline']}"
          + ("" if _flagged == 0 else
             "  Treat the in-sample figures as an upper bound, not an estimate."))
    with st.expander(f"Validation detail — in-sample vs holdout "
                     f"({checks['months_is']} vs {checks['months_oos']} months)"):
        st.dataframe(
            pd.DataFrame([{"Check": n, "": "pass" if ok else "flag",
                           "In-sample → holdout": d} for n, ok, d in checks["checks"]]),
            width="stretch", hide_index=True)
        st.caption("Fixed thresholds, applied identically every run — sign "
                   "agreement, at least half the effect retained, holdout "
                   "|t| ≥ 2, and ntiles still in order. Sign agreement is the "
                   "gate: an effect that reverses out of sample is a different "
                   "effect, not a weaker one.")
        if _nl_key:
            _vkey = (checks["verdict"], checks["passed"],
                     spec_formula(rows, sector_neutral, cons))
            if st.session_state.get("_vnote_key") != _vkey:
                try:
                    with st.spinner("Reading the validation…"):
                        st.session_state["_vnote"] = explain_validation(
                            checks, spec_formula(rows, sector_neutral, cons),
                            _nl_key)
                    st.session_state["_vnote_key"] = _vkey
                except Exception as e:
                    st.session_state["_vnote"] = ""
                    st.caption(f"(couldn't draft a note: {e})")
            if st.session_state.get("_vnote"):
                st.markdown(f":grey[{st.session_state['_vnote']}]")

# ── Results ──────────────────────────────────────────────────────────────────

hl = HORIZON_LABELS[horizon]
if bench_mode == BENCH_COL:
    bench_label = bench_col
elif bench_mode == BENCH_REF:
    bench_label = f"{bench_ref} · {REFERENCE_LABELS.get(bench_ref, bench_ref)}"
else:
    bench_label = bench_mode
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Mean IC", f"{res['mean_ic']:.4f}", help=f"Spearman rank IC vs {hl} forward return, monthly")
m2.metric("NW t-stat", f"{res['t_stat']:.2f}", help="Newey-West corrected (Bartlett kernel)")
m3.metric("IC IR", f"{res['ir']:.2f}", help="mean IC / std IC")
m4.metric("Top−bottom spread", f"{res['spread_ann']:+.1%}",
          help=f"Annualized, {hl} horizon")
m5.metric("L/S Sharpe", f"{res['ls_sharpe']:.2f}",
          help="Top minus bottom ntile, monthly rebalance, gross")

sc = res.get("screen") or {}
if sc:
    st.caption(
        f"Screen keeps **{sc['frac_kept']:.0%}** of rows "
        f"({sc['rows_kept']:,} of {sc['rows_total']:,}) and **{sc['names_kept']:,}** "
        f"distinct names; thinnest month has **{sc['min_names_per_date']}**. "
        f"Ranks are computed among survivors only, so these results describe the "
        f"screened universe — not the full one.")

colors = ramp(n_q)   # shared by the on-screen tabs and the saved report

# ── Save the run ─────────────────────────────────────────────────────────────
# Deliberately a button, not a tab: saving should be a one-click habit, and a
# tab labelled "Export" reads as somewhere to navigate rather than something to
# do — easy to skip, and you only notice once the good run is gone.
# One signature per distinct run configuration. The timestamp is pinned to it,
# so it reads as "when this run was first produced" rather than "when the page
# last redrew" — and, because it stops changing every rerun, the package below
# can actually be cached instead of rebuilt on every widget interaction.
_sig = json.dumps([rows, cons, sel_sectors, sel_industries, list(yr_range),
                   horizon, int(n_q), bench_label, bool(sector_neutral),
                   panel_key, ENGINE_SIG], default=str, sort_keys=True)
_stamp = st.session_state.setdefault("_run_stamps", {}).setdefault(
    _sig, datetime.now())
_title = run_title(rows, horizon, n_q, _stamp)
_slug = run_slug(rows, horizon, _stamp)

record = {
    "title": _title,
    "exported_at": _stamp.isoformat(timespec="seconds"),
    "panel": BASE_PANEL_LABEL if panel_key == "base" else "custom upload",
    "spec": {
        "formula": spec_formula(rows, sector_neutral, cons),
        "features": [{"feature": f, "transform": t, "weight": w} for f, t, w in rows],
        "constraints": [{"left": l, "op": o, "right": r} for l, o, r in cons],
        "sector_neutral": bool(sector_neutral),
    },
    "universe": {
        "sectors": sel_sectors or "all",
        "industries": sel_industries or "all",
        "years": [int(yr_range[0]), int(yr_range[1])],
        "securities": int(fpanel["ticker"].nunique()),
        "dates": int(fpanel["date"].nunique()),
    },
    "test": {"horizon": horizon, "ntiles": int(n_q), "benchmark": bench_label},
    "results": {
        "mean_ic": _num(res["mean_ic"]), "nw_t_stat": _num(res["t_stat"]),
        "ic_ir": _num(res["ir"]), "n_months": int(res["n_months"]),
        "spread_ann": _num(res["spread_ann"]),
        "ls_sharpe": _num(res["ls_sharpe"]),
        "ls_ann_return": _num(res["ls_ann_ret"]),
        "ls_ann_vol": _num(res["ls_ann_vol"]),
        "ntile_ann_arithmetic": {f"Q{int(q)}": _num(v)
                                 for q, v in res["q_ann_returns"].items()},
        "ntile_cagr": {f"Q{int(q)}": _num(v) for q, v in res["q_cagr"].items()},
        "benchmark_ann": _num(res["bench_ann"]),
        "benchmark_cagr": _num(res["bench_cagr"]),
    },
    "screen": {k: _num(v) if isinstance(v, float) else v
               for k, v in (res.get("screen") or {}).items()} or None,
    "validation": ({
        "verdict": checks["verdict"], "headline": checks["headline"],
        "flagged": checks["total"] - checks["passed"], "total": checks["total"],
        "months_in_sample": checks["months_is"],
        "months_holdout": checks["months_oos"],
        "ic_retention": _num(checks["ic_retention"]),
        "sharpe_retention": _num(checks["sharpe_retention"]),
        "checks": [{"name": n, "pass": ok, "detail": d}
                   for n, ok, d in checks["checks"]],
    } if checks else None),
    "sample_shown": sample,
}
record_json = json.dumps(record, indent=2, default=_jsonable)

scores = fpanel[["date", "ticker", "sector", "industry"]].copy()
scores["composite"] = res["composite"]
scores = scores.dropna(subset=["composite"])


@st.cache_data(show_spinner=False, max_entries=4)
def _scores_csv(sig: str, df) -> str:
    """Serialising ~160k rows is not free; do it once per run, not per rerun."""
    return df.to_csv(index=False)


@st.cache_data(show_spinner="Building report…", max_entries=3)
def build_report(sig: str, title: str, record: dict, figs_html: list[str],
                 sector_table: str, caveats: list[str],
                 validation: dict | None = None) -> str:
    return _build_report(sig, title, record, figs_html, sector_table, caveats,
                         validation)


_figs = [f for f in (fig_ntile_bars(res, colors, hl, bench_label),
                     fig_ntile_curves(res, colors, bench_label),
                     fig_long_short(res), fig_ic(res, hl), fig_sector(res))
         if f is not None]
def _figs_to_html(mode):
    """mode True inlines plotly.js on the first figure; "cdn" links it instead."""
    return [f.to_html(include_plotlyjs=(mode if i == 0 else False),
                      full_html=False, config={"displayModeBar": False})
            for i, f in enumerate(_figs)]

_sector_table = ""
if len(res["sector_ic"]):
    _rows = "".join(
        f"<tr><td>{r.sector}</td><td class='num'>{r.mean_ic:+.4f}</td>"
        f"<td class='num'>{r.t_stat:+.2f}</td></tr>"
        for r in res["sector_ic"].sort_values("mean_ic", ascending=False).itertuples())
    _sector_table = ("<table><tr><th>Sector</th><th>Mean IC</th><th>NW t</th></tr>"
                     + _rows + "</table>")

_caveats = [
    "Ntile bars are the <i>arithmetic</i> mean of overlapping windows — what the "
    "signal predicts. They sit above the compounded CAGR beside them by roughly "
    "half the variance, widest for the most volatile ntile.",
    "Cumulative and long–short curves always use non-overlapping 1-month returns, "
    "so they do not change with the return horizon.",
    "The Newey-West t-stat corrects for the overlap that horizons beyond one "
    "month introduce.",
    "Returns are gross of transaction costs, financing and taxes; the long–short "
    "Sharpe deducts no risk-free rate.",
]
if record.get("screen"):
    _caveats.append("A screen is active, so these results describe the screened "
                    "universe, not the full one.")

_validation = None
if checks:
    _validation = {k: v for k, v in checks.items() if k != "checks"}
    _validation["checks"] = checks["checks"]
    _validation["note"] = st.session_state.get("_vnote", "")
report_html = build_report(_sig, _title, record, _figs_to_html(True),
                           _sector_table, _caveats, _validation)

_spacer, _save, _more = st.columns([6, 2, 1], vertical_alignment="bottom")
_save.download_button(
    "Save run", report_html, file_name=f"{_slug}.html", mime="text/html",
    type="primary", width="stretch",
    help=f"{_slug}.html — one self-contained file with the spec, the metrics and "
         f"the charts. Opens in any browser, no internet needed.")
with _more.popover("⋯", width="stretch", help="Other formats"):
    st.caption(f"**{_title}**")
    st.download_button(
        "Report (small, needs internet)",
        build_report(_sig + "|cdn", _title, record, _figs_to_html("cdn"),
                     _sector_table, _caveats, _validation),
        file_name=f"{_slug}_lite.html", mime="text/html", width="stretch",
        help="Same report, ~10 KB instead of ~5 MB, but it loads the charting "
             "library from a CDN so it needs a connection to render.")
    st.download_button("Run record (JSON)", record_json, file_name=f"{_slug}.json",
                       mime="application/json", width="stretch",
                       help="The same spec and metrics, machine-readable.")
    st.download_button("Scores (CSV)", _scores_csv(_sig, scores),
                       file_name=f"{_slug}_scores.csv", mime="text/csv",
                       width="stretch",
                       help="Composite score per name per date.")

tab_nt, tab_ls, tab_ic, tab_sec, tab_rank = st.tabs(
    ["Ntile portfolios", "Long–short", "IC over time", "Sectors", "Latest ranks"])

with tab_nt:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(fig_ntile_bars(res, colors, hl, bench_label),
                        width="stretch")
        st.caption(
            f"Average {hl} forward return per ntile, annualized — Q1 = lowest "
            f"composite score, top ntile = highest. Dashed line is "
            f"**{bench_label}** at **{res['bench_ann']:+.1%}**, measured the same "
            f"way, so a ntile only adds value by clearing it. This is the "
            f"*arithmetic* mean of overlapping {hl} windows: it answers \"what "
            f"does the signal predict\", and runs above the compounded number "
            f"beside it by roughly half the variance. For what an investor "
            f"actually earns, read the curve →")
    with c2:
        st.plotly_chart(fig_ntile_curves(res, colors, bench_label), width="stretch")
        _q = res["q_cagr"]
        st.caption(
            f"Compounded growth of $1 per ntile — equal-weight within ntile, "
            f"monthly rebalance, gross of costs. Top ntile compounds at "
            f"**{_q.iloc[-1]:+.1%}/yr** vs **{_q.iloc[0]:+.1%}** for the bottom "
            f"and **{res['bench_cagr']:+.1%}** for {bench_label} — lower than the "
            f"bars because compounding penalises volatility. Always built from "
            f"non-overlapping 1-month returns, so the horizon selector does not "
            f"move this chart.")

with tab_ls:
    _f = fig_long_short(res)
    if _f is not None:
        st.plotly_chart(_f, width="stretch")
        dd = res["ls_cum"] / res["ls_cum"].cummax() - 1.0
        st.caption(f"Top ntile minus bottom ntile, monthly rebalance, gross of "
                   f"costs and financing. Ann. return **{res['ls_ann_ret']:+.1%}** · "
                   f"ann. vol **{res['ls_ann_vol']:.1%}** · max drawdown "
                   f"**{dd.min():.1%}** · Sharpe **{res['ls_sharpe']:.2f}** "
                   f"(no risk-free deducted — the legs roughly finance each "
                   f"other). Built from non-overlapping 1-month returns, so the "
                   f"horizon selector does not move this chart either.")
    else:
        st.info("Not enough data for a long–short series in this slice.")

with tab_ic:
    st.plotly_chart(fig_ic(res, hl), width="stretch")
    pos = (res["ic_series"] > 0).mean()
    st.caption(f"IC positive in **{pos:.0%}** of months ({res['n_months']} months). "
               "Overlapping observations at horizons beyond 1 month — the "
               "Newey-West t-stat corrects for this.")

with tab_sec:
    _f = fig_sector(res)
    if _f is not None:
        st.plotly_chart(_f, width="stretch")
        st.dataframe(res["sector_ic"].sort_values("mean_ic", ascending=False)
                     .style.format({"mean_ic": "{:.4f}", "ir": "{:.2f}",
                                    "t_stat": "{:.2f}"}),
                     width="stretch", hide_index=True)
    else:
        st.info("No sector metadata in this panel (or too few observations per sector).")

with tab_rank:
    latest = scores[scores["date"] == scores["date"].max()]
    st.write(f"Latest cross-section ({latest['date'].max():%Y-%m-%d}) — top ranked "
             f"by composite:")
    st.dataframe(latest.nlargest(25, "composite"), width="stretch", hide_index=True)
    st.caption("Every name and date is in `scores.csv` inside the saved package.")






# ── Reference ─────────────────────────────────────────────────────────────────────
with st.expander("How this works — features, transforms, weights, universe",
                 expanded=False):
    st.markdown("""
An **edge** here is one number per stock per month. You build it by picking
features, transforming each one, weighting them, and summing. The composite is
re-ranked each month, cut into ntile portfolios, and scored against forward
returns.

**Feature** — a raw column in the panel (`fcf_yield`, `gpa`, `mom_12_1`, …).
Raw values, not pre-ranked, so the transform below is doing real work.

**Transform** — how a feature is made comparable across stocks *within each
month*:

| | what it does | when |
|---|---|---|
| `rank` | percentile 0→1 within the month | **the default.** Outlier-proof, and puts every feature on one scale so weights mean what you think |
| `zscore` | winsorized standard deviations from the mean | when *how far* above average matters, not just the ordering |
| `raw` | winsorized value, untransformed | when the units are already comparable — mixing raw with ranked features will let the raw one dominate |

**Weight** — relative contribution. Only ratios matter (`1, 0.5` behaves like
`2, 1`). **Negative flips the feature**, which is how you express "less of
this": `accruals` and `asset_gr` are bad things, so they enter negatively.

**Universe** — which names are eligible, in the sidebar. Sectors and industries
are *include* lists (empty = everything), so you exclude a sector by leaving it
out. Useful when a signal is meaningless somewhere — leverage and accruals do
not mean the same thing for banks.

**Benchmark** — what the ntiles are measured against on the charts. Three kinds:

- *Equal-weight* / *cap-weighted universe* — computed from the names your filters
  actually leave in, so they move when you change the universe. Equal-weight is
  the honest default (your ntiles are equal-weight too); cap-weighted is the
  closer analogue of a real index.
- *Reference: …* — actual index ETFs (SPY, RSP, IWM, IWD, IWF, MDY, QQQ), shipped
  as a precomputed monthly series so the app never has to call out to a data
  provider. These are a **fixed** yardstick: they do not follow your universe
  filters, which is the point when you want to know whether an edge beats
  something investable. Note the equal-weight universe default is close to RSP by
  construction, and short histories start late (RSP only from 2003).
- *Column: …* — a benchmark return series carried in your own panel, as one value
  repeated per date. Read as a per-month return.

**Constraints** — hard screens on **raw** feature values, applied before any
ranking. Two shapes: against a number (`roa > 0`, `leverage < 0.5`) or against
another feature (`roa > asset_gr` — earning more than it is growing assets).
Names failing a constraint, *or* missing either side of it, are dropped from that
month entirely — so the survivors are ranked against each other and the peer
group itself changes. That is the difference between a constraint and a negative
weight: a weight expresses a preference and keeps everyone, a constraint discards.
Reach for a weight unless you mean a genuine requirement. Two cautions: comparing
two features only means something if they share units (`roa > asset_gr` compares
two rates and is fine; `roa > log_mcap` is nonsense), and a tight screen can
quietly shrink the universe — the app reports what survives.

**Sector-neutral** — transforms *within* each sector, so you rank a bank against
banks. Strips out the sector bet and tests whether the signal picks names.

**Horizon** — how far forward returns are measured. **Ntiles** — how many
buckets, so 5 = quintiles.

---

#### The same edge, two ways

The builder table is always the source of truth. You can fill it in directly:

| Feature | Transform | Weight |
|---|---|---|
| `fcf_yield` | rank | 1.0 |
| `gpa` | rank | 0.5 |
| `accruals` | rank | −0.5 |
| `asset_gr` | rank | −0.5 |

…with **Communication Services** left out of the sidebar sector list. Which reads as:

`edge = rank(fcf_yield) + 0.5·rank(gpa) − 0.5·rank(accruals) − 0.5·rank(asset_gr)`

Or describe it in English in the box below and let it fill the table in for you:

> *“cash-generative, profitable companies with clean accounting that aren't
> over-expanding, excluding communication services”*

Either way you land on the same four rows — and you can edit them afterwards.
Describing it in English is a starting point, not a separate mode.

---

#### Bringing your own data

Upload a CSV (sidebar → **Upload CSV**) and it replaces the panel wholesale —
**your data redefines everything downstream**:

- **The universe** is whatever rows you supply. Sector and industry filters read
  *your* labels; if you don't map those columns everything lands in one bucket
  called `All` and the sector tab has nothing to split.
- **The feature list** becomes your numeric columns. The builder table, the
  English box and this glossary all repopulate from them — the shipped feature
  names (`fcf_yield`, `gpa`, …) stop existing unless your file has them.
- **The benchmark** is recomputed from your rows, so "equal-weight universe"
  means *your* universe. Map a benchmark-return column if you want to compare
  against something external instead.
- **Forward returns** have to come from somewhere: a price column (returns get
  computed at 1/3/6/12m), a forward-return column you already have, or joined
  from the shipped panel by US ticker.

One row per security per date, and tell it which column is the security ID and
which is the date — everything else is inferred.
""")
