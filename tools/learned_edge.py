"""
Learned Edge — infer a manager's style, two ways.

Holdings: upload a panel of (date, secid, features..., held_flag), fit what
separates held names from the rest under a walk-forward protocol, and emit a
per-name style tilt the Edge Concierge can use as a layer.

Returns, two flows: upload the manager's return series, then either build one
long-short leg per factor from a stock x date panel (the shipped ones, or an
upload) or bring a wide file of factor returns (Chen-Zimmermann's predictor
zoo, or your own) and fit active return against a benchmark. Regress the
target on the legs — expanding, rolling, or a Kalman filter — and send the
loadings to the Concierge as edge-table weights.

Both are out-of-sample by construction: each fold trains only on prior history
and scores the following unseen window.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_io import load_base_panel, load_short_panel, normalize_csv
from engine import feature_columns
from learned import (INIT_TRAIN_YEARS, KALMAN_DRIFT, MARKET_LEG, PROB_COL,
                     SIZE_GROUPS, STEP_YEARS, build_learned_style_panel,
                     factor_legs, full_history_profile, kalman_style,
                     monthly_series, walk_forward_style)
from ui import BLUE, INK_2, ramp, style, universe_block

TILT_COL = "learned_tilt"
# same strings the Concierge uses, so the two tabs describe one panel one way
BASE_PANEL_LABEL = "S&P 500 · 1998–2025 · basic factors"
SHORT_PANEL_LABEL = "Top 1500 · 1998–2025 · short-risk factors"

st.title("Learned Edge")

MODE_HOLD = "Holdings — what it holds"
MODE_RET = "Returns — legs built from a stock × date panel"
MODE_LEGS = "Returns — legs you bring (Chen–Zimmermann style)"
mode = st.radio(
    "Learn the style from", [MODE_HOLD, MODE_RET, MODE_LEGS], horizontal=True,
    key="lmode",
    help="**Holdings**: a classifier on characteristics — what separates held "
         "names from the rest; the output is a per-name tilt. **Returns, legs "
         "built**: a Sharpe-style regression of the manager's return on "
         "long–short legs cut from a stock × date panel by the Concierge's own "
         "engine, plus a market leg. **Returns, legs you bring**: the same "
         "regression on a wide file of factor returns — Chen–Zimmermann's "
         "open-source predictor zoo, or your own — fitted on active return so "
         "no market leg is needed. Both return flows emit loadings that become "
         "edge-table weights.")


def _guess_in(cols, names, fallback=0):
    for i, c in enumerate(cols):
        if any(n in c.lower().replace(" ", "").replace("_", "") for n in names):
            return i
    return fallback


@st.cache_data(show_spinner="Reading…", max_entries=2)
def _read(data: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(data))


# ═════════════════════════════════════════════════════════════════════════════
# Returns-based — shared pieces
# ═════════════════════════════════════════════════════════════════════════════

# OSAP predictor -> (shipped feature, sign). Chen–Zimmermann sign every
# long–short leg so its in-sample mean is positive, so a leg on a "bad"
# characteristic is long the LOW end: AssetGrowth is long low growth, Size is
# long small. The sign flips such a loading into the Concierge's convention,
# where weight w on rank(f) is long HIGH f. Only names whose construction
# matches a shipped feature closely enough to hand a weight across; anything
# else stays a diagnostic. Signs follow OSAP's SignalDoc.
OSAP_CROSSWALK = {
    "BM": ("btm", 1), "EP": ("earn_yield", 1), "CF": ("fcf_yield", 1),
    "SP": ("sales_yield", 1), "GP": ("gpa", 1), "roaq": ("roa", 1),
    "RoE": ("roe", 1), "Leverage": ("leverage", 1), "High52": ("high_52w", 1),
    "Mom12m": ("mom_12_1", 1), "Mom6m": ("mom_6_1", 1),
    "STreversal": ("ret_1m", -1), "Accruals": ("accruals", -1),
    "AssetGrowth": ("asset_gr", -1), "Size": ("log_mcap", -1),
    "ShareIss1Y": ("dilution_1y", -1),
}

EST_EXP, EST_ROLL, EST_KAL = "Expanding window", "Rolling window", "Kalman filter"


def _parse_dates(s: pd.Series) -> pd.Series:
    """Dates as written, or yyyymm integers (OSAP's convention) — never epochs."""
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().all() and v.between(190001, 210012).all():
        return pd.to_datetime(v.astype(int).astype(str), format="%Y%m", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _manager_intake(prefix: str, with_benchmark: bool):
    """
    Step 1 of both returns flows. Returns the monthly target series (the
    manager's return, or active return against a benchmark column) or stops
    the script with guidance if nothing is uploaded yet.
    """
    with st.container(border=True, key=f"{prefix}step1"):
        st.subheader(
            "1 · Manager returns",
            help="One row per period: a date and the manager's return"
                 + (", with the benchmark's return alongside if you want to fit "
                    "active return" if with_benchmark else "")
                 + ". Monthly is the natural grain; daily rows are compounded "
                   "into months. Values above 1 in magnitude are read as percent.")
        up = st.file_uploader("Manager returns CSV", type="csv", key=f"{prefix}up")
        st.caption("Held in memory for this session only — never written to disk, "
                   "never shared between users, and dropped when you close the tab.")
        if up is None:
            st.info("Upload the manager's return series to begin — `date` and a "
                    "return column" + (", plus the benchmark's return for an "
                                       "active-return fit." if with_benchmark
                                       else ". The factor legs come from step 2."))
            st.stop()
        raw = _read(up.getvalue())
        cols = list(raw.columns)
        c1, c2, c3, c4 = st.columns(4)
        date_col = c1.selectbox("Date", cols, key=f"{prefix}date",
                                index=_guess_in(cols, ("date", "month", "period")))
        ret_col = c2.selectbox(
            "Return", cols, key=f"{prefix}ret",
            index=_guess_in(cols, ("return", "ret", "manager", "portfolio",
                                   "fund", "strategy", "pnl"), min(1, len(cols) - 1)))
        bmk_col = "(none)"
        if with_benchmark:
            bmk_col = c3.selectbox(
                "Benchmark (optional)", ["(none)"] + cols, key=f"{prefix}bmk",
                index=_guess_in(cols, ("bench", "bmk", "index"), -1) + 1,
                help="If set, the target is **active return** — fund minus "
                     "benchmark — so the market exposure cancels and the legs "
                     "need no market column. Assumes beta ≈ 1 to the benchmark; "
                     "the caption shows the realized beta so you can check.")
        month_end = c4.selectbox(
            "Each date is the…", ["end of the period it covers", "start"],
            key=f"{prefix}conv",
            help="The usual convention for a return series is a month-end date "
                 "labelling the month just finished.") == "end of the period it covers"
        dates = _parse_dates(raw[date_col])
        _mv = pd.to_numeric(raw[ret_col], errors="coerce").abs()
        scale = 100.0 if _mv.median() > 1.0 else 1.0
        fund = monthly_series(dates, raw[ret_col] / scale, month_end)
        y, note = fund, ""
        if bmk_col != "(none)":
            bmk = monthly_series(dates, raw[bmk_col] / scale, month_end)
            both = pd.DataFrame({"f": fund, "b": bmk}).dropna()
            if len(both) < 12:
                st.error("Fund and benchmark overlap on fewer than 12 months.")
                st.stop()
            beta = float(both.cov().iloc[0, 1] / both["b"].var()) if both["b"].var() else float("nan")
            y = (both["f"] - both["b"]).rename("active")
            note = (f" · **active return** vs `{bmk_col}` · realized beta "
                    f"**{beta:.2f}** · tracking error {y.std() * 12 ** 0.5:.1%}")
        if len(y) < 12:
            st.error(f"Only {len(y)} months — need a few years.")
            st.stop()
        _ann = (1 + y).prod() ** (12 / len(y)) - 1
        st.caption(
            f"{len(y):,} months · {y.index.min()} – {y.index.max()} · "
            f"{_ann:+.1%}/yr, vol {y.std() * 12 ** 0.5:.1%}" + note
            + (" · values read as **percent** and divided by 100" if scale > 1 else "")
            + (f" · {len(raw):,} rows compounded into months" if len(raw) > len(y) else ""))
    return y, bmk_col != "(none)"


def _estimator_controls(prefix: str):
    """Step 'Fit' controls shared by both returns flows."""
    est = st.radio("Estimator", [EST_EXP, EST_ROLL, EST_KAL], horizontal=True,
                   key=f"{prefix}est",
                   help="**Expanding**: each fold fits on all prior months and "
                        "replicates the next window. **Rolling**: the same, on "
                        "only the last N months — follows a style that moves, "
                        "at the cost of a noisier fit. **Kalman**: loadings "
                        "that drift month by month, each month replicated with "
                        "the loadings predicted before it was seen — a "
                        "multivariate dynamic hedge ratio.")
    f1, f2, f3 = st.columns(3)
    init_years = f1.number_input("Initial train (years)", 1, 20,
                                 min(INIT_TRAIN_YEARS, 3), 1, key=f"{prefix}init")
    window_m = step_years = None
    drift_name = "medium"
    if est == EST_ROLL:
        window_m = f2.slider("Window (months)", 24, 120, 60, 12, key=f"{prefix}win")
        step_years = f3.number_input("Step (years)", 1, 5, STEP_YEARS, 1, key=f"{prefix}stepy")
    elif est == EST_EXP:
        step_years = f2.number_input("Step (years)", 1, 5, STEP_YEARS, 1, key=f"{prefix}stepy")
    else:
        drift_name = f2.select_slider(
            "Loading drift", list(KALMAN_DRIFT), value="medium", key=f"{prefix}drift",
            help="How far the loadings may move per month. Slow ≈ a lazy "
                 "expanding regression; fast chases noise.")
    return est, int(init_years), step_years, window_m, drift_name


def _fit(X, y, est, init_years, step_years, window_m, drift_name, prog):
    if est == EST_KAL:
        return kalman_style(X, y, init_years, KALMAN_DRIFT[drift_name], prog)
    return walk_forward_style(X, y, init_years, int(step_years), window_m, prog)


def _render_style_results(res, prefix: str, active: bool, market_leg=None,
                          crosswalk=None):
    """Metrics, the four tabs, and the Concierge handoff, for either flow."""
    is_kalman = len(res.loadings_by_fold) == len(res.manager) and len(res.manager) > 0
    folds = res.folds
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pooled OOS correlation", f"{res.pooled_corr:.3f}",
              help="Replicated vs actual over every out-of-sample month at once. "
                   "0 is no relationship.")
    m2.metric("Pooled OOS R²", f"{res.pooled_r2:.3f}",
              help="Share of the month-to-month variance the legs explain out of "
                   "sample. Negative means the replication did worse than a "
                   "flat line.")
    m3.metric("Months replicated",
              f"{int(res.replicated.notna().sum()):,} / {len(res.manager):,}",
              help="The initial training block is never replicated — no fit "
                   "existed yet that had not seen it.")
    if market_leg and market_leg in res.loadings.index:
        _b = res.loadings[market_leg]
        m4.metric("Beta to universe" + (" (−1)" if active else ""), f"{_b:.2f}",
                  help=("Loading on the equal-weight universe leg. With an active-"
                        "return target this reads as beta − 1, i.e. how far the "
                        "beta-≈-1 assumption is off." if active else
                        "Loading on the equal-weight universe leg. Kept separate "
                        "so beta does not get smeared into the factor loadings; "
                        "not sent to the Concierge."))
    else:
        _alpha = folds["alpha_ann"].mean() if len(folds) else float("nan")
        m4.metric("Alpha (ann.)", f"{_alpha:+.1%}",
                  help="The intercept: return the legs don't account for. Gross; "
                       "read the sign more than the size.")

    if res.skipped:
        with st.expander(f"{len(res.skipped)} note(s) from the fit"):
            for s in res.skipped:
                st.caption(f"· {s}")

    s_fold, s_load, s_rep, s_send = st.tabs(
        ["Fit over time", "Loadings", "Replication", "Send to Concierge"])
    target = "active return" if active else "manager"

    with s_fold:
        if len(folds):
            _x = folds["train_end"].dt.to_timestamp()
            fig = go.Figure(go.Scatter(
                x=_x, y=folds["oos_corr"], mode="lines+markers",
                line=dict(color=BLUE, width=2), marker=dict(size=7),
                name="OOS correlation"))
            fig.add_hline(y=0, line_dash="dash", line_color=INK_2, line_width=2,
                          annotation_text="no relationship",
                          annotation_position="bottom left")
            fig.update_yaxes(title="Out-of-sample correlation", range=[-0.5, 1.0])
            st.plotly_chart(style(fig, 340), width="stretch")
            st.caption(("One point per calendar year of filtered months. "
                        if is_kalman else
                        "One point per fold, at the month the training set ended. ")
                       + f"A line that decays means the legs stopped explaining "
                         f"the {target} — a drift a pooled number hides.")
            st.dataframe(
                folds.assign(train_end=folds["train_end"].dt.strftime("%Y-%m"),
                             test_end=folds["test_end"].dt.strftime("%Y-%m"))
                     .rename(columns={"train_end": "from", "test_end": "to",
                                      "n_train": "months before",
                                      "n_test": "months", "oos_corr": "OOS corr",
                                      "oos_r2": "OOS R²", "alpha_ann": "alpha (ann.)"})
                     .style.format({"OOS corr": "{:.3f}", "OOS R²": "{:.3f}",
                                    "alpha (ann.)": "{:+.1%}"}),
                width="stretch", hide_index=True)
        else:
            st.info("Nothing replicated — the series may be shorter than the "
                    "initial training window.")

    with s_load:
        if len(res.loadings):
            _lab = "latest filtered" if is_kalman else "time-averaged across folds"
            ld = res.loadings.head(20).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=ld.values, y=ld.index, orientation="h",
                marker_color=[BLUE if v >= 0 else INK_2 for v in ld.values],
                marker_line_width=0))
            fig.update_xaxes(title=f"Loading ({_lab})")
            st.plotly_chart(style(fig, max(280, 22 * len(ld))), width="stretch")
            st.caption(f"A loading of 0.5 on a leg means the {target} moved half "
                       f"as much as that long–short portfolio did, holding the "
                       f"others fixed. Negative = tilted against it.")
            if len(res.loadings_by_fold) > 1:
                lbf = res.loadings_by_fold[res.loadings.head(6).index]
                _x = lbf.index.to_timestamp()
                fig2 = go.Figure()
                for i, c in enumerate(lbf.columns):
                    fig2.add_trace(go.Scatter(
                        x=_x, y=lbf[c], name=c, mode="lines",
                        line=dict(color=ramp(len(lbf.columns))[i], width=2)))
                fig2.add_hline(y=0, line_color=INK_2, line_width=1)
                fig2.update_yaxes(title="Loading over time")
                st.plotly_chart(style(fig2, 300), width="stretch")
                st.caption("The six largest loadings, "
                           + ("month by month as the filter updates them. "
                              if is_kalman else "fold by fold. ")
                           + "Lines that hold their level are the durable part "
                             "of the style; one that wanders is drift, or a leg "
                             "the sample can't pin down.")

    with s_rep:
        both = pd.DataFrame({"actual": res.manager,
                             "replicated": res.replicated}).dropna()
        if len(both) > 2:
            cum = (1 + both).cumprod()
            _x = cum.index.to_timestamp()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=_x, y=cum["actual"], name=target.capitalize(),
                                     mode="lines", line=dict(color=BLUE, width=2)))
            fig.add_trace(go.Scatter(x=_x, y=cum["replicated"],
                                     name="Factor replication (OOS)", mode="lines",
                                     line=dict(color=INK_2, width=2, dash="dash")))
            fig.update_yaxes(title="Growth of $1, replicated months only")
            st.plotly_chart(style(fig, 340), width="stretch")
            te = (both["actual"] - both["replicated"]).std() * (12 ** 0.5)
            st.caption(f"{target.capitalize()} vs what the legs alone would have "
                       f"returned, using only out-of-sample loadings. Tracking "
                       f"error **{te:.1%}** annualized. The gap is what the legs "
                       f"don't explain: selection, timing, or a factor not in "
                       f"the set.")
        else:
            st.info("Nothing replicated yet.")

    with s_send:
        fac = res.loadings.drop(market_leg, errors="ignore") if market_leg else res.loadings
        if crosswalk is not None:
            mapped = {k: (f, s) for k, (f, s) in crosswalk.items() if k in fac.index}
            unmapped = [k for k in fac.index if k not in mapped]
            st.caption("Legs whose construction matches a shipped feature are "
                       "translated — name and sign — into rows for the "
                       "Concierge's edge table, scaled so the largest is ±1. "
                       "The rest are diagnostics: real exposures, just to "
                       "characteristics the Concierge doesn't carry.")
            if unmapped:
                with st.expander(f"{len(unmapped)} leg(s) with no shipped feature"):
                    st.caption(", ".join(f"`{k}` {fac[k]:+.2f}" for k in unmapped))
            fac = pd.Series({f: fac[k] * s for k, (f, s) in mapped.items()}) \
                .sort_values(key=abs, ascending=False)
        else:
            st.caption("The factor loadings (market leg excluded), scaled so the "
                       "largest is ±1, become rows in the Concierge's edge table "
                       "— transform `rank`, joined by `+`. Edit them there before "
                       "testing. Load the same panel there and the names resolve.")
        if len(fac):
            n_send = st.slider("Legs to send", 1, len(fac), min(6, len(fac)),
                               key=f"{prefix}send_n",
                               help="Largest loadings first. Fewer is usually "
                                    "better — the tail is mostly noise.")
            top = fac.head(n_send)
            scale = top.abs().max() or 1.0
            rows_out = pd.DataFrame({
                "feature": top.index, "transform": "rank",
                "weight": (top / scale).round(2).values, "op": "+"})
            st.dataframe(rows_out, width="stretch", hide_index=True)
            if st.button("Send to Edge Concierge", type="primary", key=f"{prefix}send"):
                st.session_state["builder"] = rows_out
                st.session_state.pop("builder_editor", None)
                st.session_state["nl_note"] = (
                    f"Loaded from Learned Edge: the returns-based style "
                    f"regression's top {n_send} loading(s), scaled to ±1.")
                st.success("Sent. Open the Edge Concierge — step 4 now holds "
                           "these rows.")
        else:
            st.info("No leg maps to a shipped feature, so there is nothing to "
                    "hand across — the loadings above are the result.")


# ═════════════════════════════════════════════════════════════════════════════
# Returns — legs built from a stock × date panel
# ═════════════════════════════════════════════════════════════════════════════
if mode == MODE_RET:
    y, active = _manager_intake("r", with_benchmark=False)

    with st.container(border=True, key="rstep2"):
        st.subheader(
            "2 · Factor panel",
            help="The stock × date panel the factor legs are built from: one "
                 "long–short portfolio per feature (top ntile minus bottom, "
                 "ranked within the universe each month) plus the equal-weight "
                 "universe as a market leg. The shipped panels work as they are; "
                 "an upload needs features and a forward 1-month return column, "
                 "mapped the same way the Concierge maps one.")
        src = st.radio("Panel", [BASE_PANEL_LABEL, SHORT_PANEL_LABEL, "Upload CSV"],
                       key="rpanel", horizontal=True)
        fpanel_key, fpanel = "base", None
        if src == BASE_PANEL_LABEL:
            fpanel = load_base_panel()
        elif src == SHORT_PANEL_LABEL:
            fpanel, fpanel_key = load_short_panel(), "short"
        else:
            fup = st.file_uploader("Stock × date CSV", type="csv", key="lfactors")
            if fup is not None:
                raw_f = _read(fup.getvalue())
                fcols = list(raw_f.columns)
                c1, c2, c3, c4 = st.columns(4)
                fid = c1.selectbox("Security ID", fcols, key="fid",
                                   index=_guess_in(fcols, ("ticker", "symbol", "secid", "id")))
                fdate = c2.selectbox("Date", fcols, key="fdate",
                                     index=_guess_in(fcols, ("date", "month"), min(1, len(fcols) - 1)))
                fret = c3.selectbox(
                    "Forward 1-month return", fcols, key="fret",
                    index=_guess_in(fcols, ("totalreturn", "fwdreturn1m", "fwdreturn", "return")),
                    help="Each name's return over the month *after* `date` — the "
                         "Concierge's fwd_1m convention.")
                fsec = c4.selectbox("Sector (optional)", ["(none)"] + fcols, key="fsec",
                                    index=_guess_in(fcols, ("sector",)) + 1
                                    if any("sector" in c.lower() for c in fcols) else 0)
                fpanel = normalize_csv(raw_f, fid, fdate,
                                       None if fsec == "(none)" else fsec, None,
                                       fwd_map={"fwd_1m": fret})
                fpanel_key = f"csv:{fup.name}:{len(raw_f)}"
        if fpanel is None:
            st.info("Upload the stock × date panel, or pick a shipped one.")
            st.stop()
        all_feats = feature_columns(fpanel)
        c5, c6, c7 = st.columns([3, 1, 1])
        legs_sel = c5.multiselect("Factors", all_feats, default=all_feats, key="rlegs",
                                  help="One long–short leg per factor. Fewer, "
                                       "less-collinear legs give steadier loadings.")
        n_q = c6.selectbox("Ntiles", [3, 4, 5, 10], index=2, key="rnq",
                           help="Top ntile minus bottom ntile, equal-weight.")
        neutral = c7.toggle("Sector-neutral", key="rneutral",
                            help="Rank within sector before cutting the legs — the "
                                 "same toggle as the Concierge.")
        if not legs_sel:
            st.warning("Pick at least one factor.")
            st.stop()

    with st.container(border=True, key="rstep3"):
        st.subheader("3 · Universe",
                     help="The names the legs are cut from. Blank by default: the "
                          "whole panel unless you narrow it.")
        uni = universe_block(fpanel, "lret", all_feats)
    fit_panel = uni["panel"]

    with st.container(border=True, key="rstep4"):
        st.subheader("4 · Fit", help="Regress the manager on the legs, out of sample.")
        est, init_years, step_years, window_m, drift_name = _estimator_controls("r")
        run = st.button("Fit", type="primary", key="rrun")
        st.caption(f"{len(legs_sel)} legs from {fit_panel['ticker'].nunique():,} names · "
                   f"{fit_panel['date'].nunique():,} panel months · "
                   f"{len(y):,} manager months. Building the legs takes about a "
                   f"second per factor the first time; after that it's instant.")

    _legs_key = (fpanel_key, tuple(legs_sel), int(n_q), bool(neutral),
                 tuple(uni["constraints"]), tuple(uni["sectors"]),
                 tuple(uni.get("industries", ())), uni["years"], len(fit_panel))
    _key = (_legs_key, est, init_years, step_years, window_m, drift_name,
            len(y), float(y.sum()))
    if run:
        bar = st.progress(0.0, "Fitting…")
        try:
            if st.session_state.get("style_legs_key") != _legs_key:
                X = factor_legs(fit_panel, legs_sel, int(n_q), bool(neutral),
                                progress=lambda f, m: bar.progress(min(f * 0.8, 0.8), m))
                st.session_state["style_legs"] = X
                st.session_state["style_legs_key"] = _legs_key
            X = st.session_state["style_legs"]
            res = _fit(X, y, est, init_years, step_years, window_m, drift_name,
                       lambda f, m: bar.progress(min(0.8 + f * 0.2, 1.0), m))
            bar.progress(1.0, "Done")
            st.session_state["style_res"] = res
            st.session_state["style_key"] = _key
        except Exception as e:
            bar.empty()
            st.error(f"Fit failed: {e}")

    res = st.session_state.get("style_res")
    if res is None:
        st.stop()
    if st.session_state.get("style_key") != _key:
        st.info("Settings changed since the last fit — press **Fit** to refresh. "
                "The results below are from the previous run.")
    _render_style_results(res, "r", active, market_leg=MARKET_LEG)
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# Factor returns — bring the legs (Chen–Zimmermann style)
# ═════════════════════════════════════════════════════════════════════════════
if mode == MODE_LEGS:
    y, active = _manager_intake("z", with_benchmark=True)

    with st.container(border=True, key="zstep2"):
        st.subheader(
            "2 · Factor returns",
            help="A wide file of long–short factor returns: one date column, one "
                 "column per leg. Chen–Zimmermann's `PredictorLSretWide` is the "
                 "canonical one — ~200 published predictors, monthly, in percent, "
                 "`yyyymm` dates, each signed so its mean is positive. Nothing is "
                 "built here; the columns you bring are the regressors you get. "
                 "There is no market leg in such a file — fit **active return** "
                 "(a benchmark column in step 1) so beta cancels.")
        fup = st.file_uploader("Factor returns CSV", type="csv", key="zlegs_up")
        if fup is None:
            st.info("Upload the wide factor-return file. With no benchmark in "
                    "step 1 the fit is on total return, and the market exposure "
                    "will land in the intercept and any market-correlated leg.")
            st.stop()
        raw_z = _read(fup.getvalue())
        zcols = list(raw_z.columns)
        c1, c2 = st.columns([1, 1])
        zdate = c1.selectbox("Date", zcols, key="zlegs_date",
                             index=_guess_in(zcols, ("date", "month", "yyyymm", "period")))
        _zn = [c for c in zcols if c != zdate and pd.api.types.is_numeric_dtype(raw_z[c])
               and raw_z[c].notna().any()]
        _zv = pd.to_numeric(raw_z[_zn].stack(), errors="coerce").abs() if _zn else pd.Series(dtype=float)
        pct = c2.toggle("Values are in percent", value=bool(len(_zv) and _zv.median() > 1.0),
                        key="zpct", help="OSAP files are. Divides by 100.")
        legs_sel = st.multiselect(
            "Legs", _zn, default=_zn, key="zlegs",
            help="Every numeric column. With 200 legs and ~200 months even ridge "
                 "is thin — prefer a chosen subset, or lean on the Kalman "
                 "estimator's shrinkage.")
        if not legs_sel:
            st.warning("Pick at least one leg.")
            st.stop()
        zdates = _parse_dates(raw_z[zdate])
        X = pd.DataFrame({c: monthly_series(zdates, raw_z[c] / (100.0 if pct else 1.0), True)
                          for c in legs_sel})
        overlap = X.dropna(how="all").index.intersection(y.index)
        n_map = sum(1 for c in legs_sel if c in OSAP_CROSSWALK)
        st.caption(f"{len(legs_sel)} legs · {len(X):,} months · overlap with the "
                   f"manager **{len(overlap):,}** months · {n_map} leg(s) map to "
                   f"shipped features for the Concierge handoff.")
        if len(overlap) < 24:
            st.error("Fewer than 24 overlapping months — check the date "
                     "conventions on both files.")
            st.stop()

    with st.container(border=True, key="zstep3"):
        st.subheader("3 · Fit",
                     help="Regress the target on the legs, out of sample.")
        est, init_years, step_years, window_m, drift_name = _estimator_controls("z")
        run = st.button("Fit", type="primary", key="zrun")
        st.caption(f"{len(legs_sel)} legs · {len(overlap):,} overlapping months · "
                   f"target: {'active return' if active else 'total return'}.")

    _key = (fup.name, len(raw_z), tuple(legs_sel), pct, est, init_years,
            step_years, window_m, drift_name, len(y), float(y.sum()))
    if run:
        bar = st.progress(0.0, "Fitting…")
        try:
            res = _fit(X, y, est, init_years, step_years, window_m, drift_name,
                       lambda f, m: bar.progress(min(f, 1.0), m))
            bar.progress(1.0, "Done")
            st.session_state["legs_res"] = res
            st.session_state["legs_key"] = _key
        except Exception as e:
            bar.empty()
            st.error(f"Fit failed: {e}")

    res = st.session_state.get("legs_res")
    if res is None:
        st.stop()
    if st.session_state.get("legs_key") != _key:
        st.info("Settings changed since the last fit — press **Fit** to refresh. "
                "The results below are from the previous run.")
    _render_style_results(res, "z", active, crosswalk=OSAP_CROSSWALK)
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# Holdings-based
# ═════════════════════════════════════════════════════════════════════════════

# ── 1 · Holdings panel ───────────────────────────────────────────────────────
with st.container(border=True, key="lstep1"):
    st.subheader(
        "1 · Holdings panel",
        help="One row per security per date: the index columns, the fundamental "
             "features, and a 0/1 flag for whether the name was held. Held rates "
             "around 3% are normal and expected. Uploads stay in memory for your "
             "session only.")
    up = st.file_uploader("Holdings CSV", type="csv", key="lholdings")
    st.caption("Held in memory for this session only — never written to disk, "
               "never shared between users, and dropped when you close the tab.")

if up is None:
    st.info("Upload a holdings panel to begin. Expected shape: `date`, `secid`, "
            "fundamental factor columns (size, bp, grpr, tdta, …), optional "
            "sector, and a binary held flag.")
    st.stop()

raw = _read(up.getvalue())
cols = list(raw.columns)


def _guess(names, fallback=0):
    return _guess_in(cols, names, fallback)


# ── 2 · Columns ──────────────────────────────────────────────────────────────
with st.container(border=True, key="lstep2"):
    st.subheader("2 · Columns",
                 help="Which column is what. Everything numeric that is not an "
                      "index, the target, or explicitly excluded becomes a "
                      "feature the model may key on.")
    c1, c2, c3 = st.columns(3)
    date_col = c1.selectbox("Date", cols, index=_guess(("date", "month", "period")))
    id_col = c2.selectbox("Security ID", cols,
                          index=_guess(("secid", "ticker", "symbol", "id")))
    target_col = c3.selectbox("Held flag (target)", cols,
                              index=_guess(("held", "flag", "target", "y")))

    c4, c5 = st.columns(2)
    _opt = ["(none)"] + cols
    sector_col = c4.selectbox("Sector (optional)", _opt,
                              index=_guess(("sector",)) + 1
                              if any("sector" in c.lower() for c in cols) else 0)
    size_col = c5.selectbox(
        "Size column (for cohorts)", _opt,
        index=_guess(("size", "mcap", "marketcap")) + 1
        if any(n in c.lower() for c in cols for n in ("size", "mcap")) else 0,
        help="If set, the whole walk-forward runs separately inside the bottom "
             "20%, middle 60% and top 20% by size. A manager who tolerates "
             "richer valuation in large caps than small is expressing a size × "
             "style interaction that one global fit would average away.")

    _reserved = {date_col, id_col, target_col} | (
        {sector_col} if sector_col != "(none)" else set())
    _numeric = [c for c in cols if c not in _reserved
                and pd.api.types.is_numeric_dtype(raw[c]) and raw[c].notna().any()]
    features = st.multiselect("Features", _numeric, default=_numeric,
                              help="Drop any column that encodes the answer — a "
                                   "weight, a portfolio flag, anything derived "
                                   "from the holding itself.")

    _t = pd.to_numeric(raw[target_col], errors="coerce")
    if _t.dropna().nunique() != 2:
        st.error(f"`{target_col}` has {_t.dropna().nunique()} distinct values — "
                 f"the target must be binary (0/1).")
        st.stop()
    if not features:
        st.warning("Select at least one feature.")
        st.stop()
    st.caption(f"{len(raw):,} rows · {len(features)} features · held rate "
               f"**{_t.mean():.2%}** · {pd.to_datetime(raw[date_col]).dt.year.min()}"
               f"–{pd.to_datetime(raw[date_col]).dt.year.max()}")

work = raw.copy()
work["date"] = pd.to_datetime(work[date_col], errors="coerce")
work = work.dropna(subset=["date"])
work["_target"] = pd.to_numeric(work[target_col], errors="coerce")
_before = len(work)
work = work.dropna(subset=["_target"] + features)
_kept_frac = len(work) / _before if _before else 1.0
if _kept_frac < 0.95:
    # a listwise dropna across every selected feature at once compounds fast —
    # 20 features each 90% populated independently keeps only ~12% of rows,
    # and with nothing surfacing that, a bad fit reads as "no signal" rather
    # than "most of the panel just got dropped before fitting ever started"
    _miss = (raw[features].isna().mean().sort_values(ascending=False) * 100).round(1)
    _worst = ", ".join(f"`{f}` {p:.0f}% missing" for f, p in _miss.head(5).items() if p > 0)
    st.warning(
        f"Selected features dropped **{_before - len(work):,} of {_before:,} rows** "
        f"({1 - _kept_frac:.0%}) to missing values before any fit runs — only "
        f"**{len(work):,}** remain. Sparsest features: {_worst or 'none'}. "
        f"Deselect a sparse one above if this is more attrition than you expect; "
        f"a fit on a starved, possibly biased survivor sample can look like "
        f"'no signal' when the real problem is the feature list.")
if sector_col != "(none)":
    work["sector"] = work[sector_col]

# ── 3 · Universe ─────────────────────────────────────────────────────────────
with st.container(border=True, key="lstep3"):
    st.subheader("3 · Universe",
                 help="Fit the style on a slice rather than everything — one "
                      "sector, one size band, one era. Blank by default: the "
                      "whole panel is used unless you narrow it.")
    uni = universe_block(work, "learned", features,
                         sector_col="sector" if sector_col != "(none)" else "_none")
fit_panel = uni["panel"]

# ── 4 · Fit ──────────────────────────────────────────────────────────────────
with st.container(border=True, key="lstep4"):
    st.subheader(
        "4 · Fit",
        help="Expanding-window walk-forward: train on all history before a "
             "cut, score the next unseen window, then fold that window into "
             "the training set and repeat. Every score comes from a model that "
             "never saw the row.")
    f1, f2, f3, f4 = st.columns(4)
    kind = f1.selectbox("Model", ["forest", "logit"], index=0,
                        format_func=lambda k: {"forest": "Random forest",
                                               "logit": "Logistic"}[k],
                        help="The forest can express size × style interactions; "
                             "the logit runs the identical protocol as a "
                             "linear comparison.")
    init_years = f2.number_input("Initial train (years)", 1, 20,
                                 min(INIT_TRAIN_YEARS, 3), 1)
    step_years = f3.number_input("Step (years)", 1, 5, STEP_YEARS, 1)
    use_cohorts = f4.toggle("Size cohorts", value=size_col != "(none)",
                            disabled=size_col == "(none)")
    run = st.button("Fit walk-forward", type="primary")
    st.caption(f"{len(fit_panel):,} rows in scope. A cohort run fits "
               f"{len(SIZE_GROUPS)}× as many models and takes correspondingly "
               f"longer.")

_key = (len(fit_panel), tuple(features), target_col, kind, int(init_years),
        int(step_years), bool(use_cohorts), size_col,
        tuple(uni["constraints"]), tuple(uni["sectors"]), uni["years"])
if run:
    bar = st.progress(0.0, "Fitting…")
    try:
        res = build_learned_style_panel(
            fit_panel, features, "_target",
            size_col=(size_col if (use_cohorts and size_col != "(none)") else None),
            date_col="date", kind=kind, init_train_years=int(init_years),
            step_years=int(step_years),
            progress=lambda f, m: bar.progress(min(f, 1.0), m))
        bar.progress(1.0, "Done")
        st.session_state["learned_res"] = res
        st.session_state["learned_key"] = _key
        st.session_state["learned_profile"] = None
    except Exception as e:
        bar.empty()
        st.error(f"Fit failed: {e}")

res = st.session_state.get("learned_res")
if res is None:
    st.stop()
if st.session_state.get("learned_key") != _key:
    st.info("Settings changed since the last fit — press **Fit walk-forward** to "
            "refresh. The results below are from the previous run.")

# ── Results ──────────────────────────────────────────────────────────────────
folds = res.folds
m1, m2, m3, m4 = st.columns(4)
m1.metric("Pooled OOS AUC", f"{res.pooled_auc:.3f}",
          help="Every out-of-sample score at once. 0.5 is a coin flip.")
m2.metric("Folds", f"{len(folds):,}")
m3.metric("Rows scored", f"{res.panel[PROB_COL].notna().sum():,}",
          help="Rows in some test window. The initial training block is never "
               "scored — no model existed yet that had not seen it.")
m4.metric("Held rate", f"{res.panel['_target'].mean():.2%}")

if res.skipped:
    with st.expander(f"{len(res.skipped)} fold(s) skipped"):
        for s in res.skipped:
            st.caption(f"· {s}")

t_auc, t_imp, t_int, t_tilt = st.tabs(
    ["Fold AUC", "What it keys on", "Interactions", "Style tilt"])

with t_auc:
    if len(folds):
        fig = go.Figure()
        for i, (ch, g) in enumerate(folds.groupby("cohort")):
            fig.add_trace(go.Scatter(
                x=g["train_end"], y=g["auc"], name=str(ch), mode="lines+markers",
                line=dict(color=ramp(folds["cohort"].nunique())[i], width=2),
                marker=dict(size=7)))
        fig.add_hline(y=0.5, line_dash="dash", line_color=INK_2, line_width=2,
                      annotation_text="coin flip", annotation_position="bottom left")
        fig.update_yaxes(title="Out-of-sample AUC", range=[0.4, 1.0])
        st.plotly_chart(style(fig, 340), width="stretch")
        st.caption("One point per fold, at the date the training set ended. A "
                   "flat line near 0.5 means the style is not learnable from "
                   "these features; a line that decays means it stopped "
                   "generalising, which a pooled number would hide.")
        st.dataframe(
            folds.assign(train_end=folds["train_end"].dt.strftime("%Y-%m"),
                         test_end=folds["test_end"].dt.strftime("%Y-%m"))
                 .rename(columns={"pos_rate_train": "held rate (train)"})
                 .style.format({"auc": "{:.3f}", "held rate (train)": "{:.2%}"}),
            width="stretch", hide_index=True)
    else:
        st.info("No folds completed — the panel may be too short for the initial "
                "training window.")

with t_imp:
    if len(res.importances):
        imp = res.importances.head(20).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                               marker_color=BLUE, marker_line_width=0))
        fig.update_xaxes(title="Importance (time-averaged across folds)")
        st.plotly_chart(style(fig, max(280, 22 * len(imp))), width="stretch")
        st.caption("Averaged across folds so each period counts once, rather "
                   "than pooled — otherwise a fold with more rows would speak "
                   "louder than a fold with more information.")

with t_int:
    st.caption("Fitted once on the whole history. Descriptive only — this fit "
               "has seen every row, so it is never used for scoring.")
    if st.button("Analyse full history"):
        with st.spinner("Fitting on all history…"):
            st.session_state["learned_profile"] = full_history_profile(
                fit_panel, features, "_target",
                size_col=(size_col if size_col != "(none)" else None),
                date_col="date")
    prof = st.session_state.get("learned_profile")
    if prof:
        pairs = prof["interactions"]
        if len(pairs):
            p = pairs.head(10).sort_values("strength")
            fig = go.Figure(go.Bar(x=p["strength"], y=p["pair"], orientation="h",
                                   marker_color=BLUE, marker_line_width=0))
            fig.update_xaxes(title="Relative interaction strength")
            st.plotly_chart(style(fig, max(260, 26 * len(p))), width="stretch")
            st.caption("Feature pairs that repeatedly split along the same "
                       "branch — the tree conditioning one on the other. "
                       "Directional evidence of where to look, not a formal test.")
        bc = prof["by_cohort"]
        if len(bc):
            st.markdown("**How the style differs by size**")
            st.dataframe(bc.head(12).style.format("{:.4f}"), width="stretch")
            st.caption("Importance of each feature fitted separately within each "
                       "size cohort. A large spread is the size × style "
                       "interaction stated directly: that feature matters in "
                       "some size bands and not others.")

with t_tilt:
    panel = res.panel.copy()
    # Rank within date, missing → 0.5. Neutral rather than dropped, so the tilt
    # is inert on dates before coverage instead of silently shrinking the
    # universe the edge is built on.
    panel[TILT_COL] = panel.groupby("date")[PROB_COL].rank(pct=True)
    n_missing = int(panel[TILT_COL].isna().sum())
    panel[TILT_COL] = panel[TILT_COL].fillna(0.5)

    scored = panel[panel[PROB_COL].notna()]
    s1, s2, s3 = st.columns(3)
    s1.metric("Mean tilt, held", f"{scored.loc[scored['_target'] == 1, TILT_COL].mean():.3f}")
    s2.metric("Mean tilt, not held", f"{scored.loc[scored['_target'] == 0, TILT_COL].mean():.3f}")
    s3.metric("Neutral (pre-coverage)", f"{n_missing:,}",
              help="Rows with no out-of-sample score — inside the initial "
                   "training block. Set to 0.5 so the layer is inert there.")

    out = panel[["date", id_col, TILT_COL]].rename(columns={id_col: "secid"})
    if "cohort" in panel.columns:
        out["cohort"] = panel["cohort"].values
    st.dataframe(out.head(20), width="stretch", hide_index=True)

    d1, d2 = st.columns([1, 3])
    d1.download_button("Download tilt (CSV)", out.to_csv(index=False),
                       file_name="learned_tilt.csv", mime="text/csv",
                       type="primary", width="stretch")
    if d2.button("Send to Edge Concierge", width="content"):
        st.session_state["learned_tilt_panel"] = out
        st.success("Available in the Edge Concierge as the feature "
                   f"`{TILT_COL}` — join is on date and security id.")
    st.caption("Rank of the out-of-sample hold probability within each date. "
               "Use it as a layer in the Edge Concierge: multiply it into the "
               "edge to tilt toward names the manager's revealed style favours, "
               "or weight it like any other feature.")
