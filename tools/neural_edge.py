"""
Neural Edge — learn a return forecast the GKX way, then ask what it believes.

Two halves. The first fits a Gu–Kelly–Xiu-style net (NN3 by default) to the
panel under the same expanding walk-forward protocol as the Learned Edge, and
emits an out-of-sample prediction rank the Edge Concierge can use as a layer.

The second half is the Concept Concierge: a TCAV flow that measures how
aligned the fitted net's forecasts are with a *concept*. Concepts are defined
by examples, three ways — a preset, quantile rules built on the panel's own
features, or an uploaded list of company–date pairs (a manager's actual
holdings, i.e. an investment philosophy stated by demonstration).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from concepts import (MIN_CONCEPT_ROWS, PRESETS, QuantileRule, match_uploaded,
                      rule_mask, run_tcav)
from data_io import load_base_panel
from engine import FWD_COLS, feature_columns
from neural import (ARCHS, INIT_TRAIN_YEARS, N_SEEDS, PRED_COL, STEP_YEARS,
                    fit_full_history, walk_forward_net)
from ui import BLUE, GRID, INK_2, MUTED, RED, ramp, style, universe_block

TILT_COL = "neural_tilt"
ARCH_LABELS = {
    "NN1": "NN1 — one layer (32)",
    "NN2": "NN2 — two layers (32·16)",
    "NN3": "NN3 — three layers (32·16·8) · GKX best",
    "NN4": "NN4 — four layers (32·16·8·4)",
    "NN5": "NN5 — five layers (32·16·8·4·2)",
}

st.title("Neural Edge")


@st.cache_data(show_spinner="Loading base panel…")
def base_panel() -> pd.DataFrame:
    return load_base_panel()


panel = base_panel()
all_features = feature_columns(panel)

# ── 1 · Target & features ────────────────────────────────────────────────────
with st.container(border=True, key="step1"):
    st.subheader(
        "1 · Target & features",
        help="The net learns to predict a forward return from per-month "
             "cross-sectional feature ranks mapped to [−1, 1] — the Gu, Kelly "
             "& Xiu (2020) characteristic transform. Missing values sit at 0, "
             "the median, so they are inert rather than informative.")
    c1, c2, c3 = st.columns([1, 1, 2])
    horizon = c1.selectbox("Predict", list(FWD_COLS), index=3, key="nn_horizon",
                           format_func=lambda h: f"{FWD_COLS[h]}-month forward return",
                           help="GKX forecast the 1-month return, but their "
                                "1-month results ride on price-based features; "
                                "on this fundamentals-heavy panel the signal "
                                "lives at longer horizons (12m tested ~8× the "
                                "1m IC), so 12 months is the default. Longer "
                                "horizons overlap: the t-stats correct for it "
                                "within the test window, and the last "
                                "`horizon` months before each training cut are "
                                "purged so no training label reaches into the "
                                "test window.")
    tgt_mode = c2.selectbox(
        "Target", ["Return over market", "Raw return"], index=0, key="nn_tgt",
        help="Chen, Hanauer & Kalsbach (2024) find the target variable is one "
             "of the biggest design levers: predicting the return OVER the "
             "market strips the common component no cross-sectional model can "
             "time, so capacity goes to ranking stocks. Market = equal-weight "
             "mean of the panel that month.")
    features = c3.multiselect("Features", all_features, default=all_features,
                              key="nn_features")
    st.caption(f"{len(panel):,} rows · S&P 500 point-in-time · "
               f"{panel['date'].dt.year.min()}–{panel['date'].dt.year.max()}")

# The demeaned target is built AFTER the feature list, so it can never be
# offered — or leak — as a feature. Demeaning is per-date over the full panel
# (the market), not the screened universe, so narrowing the universe doesn't
# quietly redefine the benchmark.
if tgt_mode == "Return over market":
    target_col = f"_{horizon}_xmkt"
    panel[target_col] = (panel[horizon]
                         - panel.groupby("date")[horizon].transform("mean"))
else:
    target_col = horizon

if not features:
    st.warning("Select at least one feature.")
    st.stop()

# ── 2 · Universe ─────────────────────────────────────────────────────────────
with st.container(border=True, key="step2"):
    st.subheader("2 · Universe",
                 help="Fit on a slice rather than everything. Blank by default.")
    uni = universe_block(panel, "neural", features)
fit_panel = uni["panel"]

# ── 3 · Fit ──────────────────────────────────────────────────────────────────
with st.container(border=True, key="step3"):
    st.subheader(
        "3 · Fit",
        help="Expanding-window walk-forward, refit from scratch each fold, an "
             "ensemble of nets averaged per fold — every prediction comes from "
             "an ensemble that never saw the row. Early stopping holds out a "
             "random 15% of each fold's training window.")
    f1, f2, f3, f4, f5 = st.columns([2, 1, 1, 1, 1])
    arch = f1.selectbox("Architecture", list(ARCHS), index=2, key="nn_arch",
                        format_func=lambda a: ARCH_LABELS[a],
                        help="GKX's geometric pyramid. Their headline result: "
                             "shallow nets win, NN3 wins among them.")
    n_seeds = f2.number_input("Ensemble", 1, 10, N_SEEDS, 1, key="nn_seeds",
                              help="Nets trained from different seeds and "
                                   "averaged — GKX's variance-reduction trick.")
    init_years = f3.number_input("Initial train (yrs)", 3, 20,
                                 INIT_TRAIN_YEARS, 1, key="nn_init")
    step_years = f4.number_input("Step (yrs)", 1, 5, STEP_YEARS, 1, key="nn_step")
    alpha = f5.select_slider("L2 penalty", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                             value=1e-2, key="nn_alpha",
                             format_func=lambda a: f"{a:g}",
                             help="Tuned on this panel: 1e-2 won both halves "
                                  "of the sample at 5 seeds. Wider nets "
                                  "preferred 1e-1 — more capacity wants more "
                                  "shrinkage.")
    run = st.button("Fit walk-forward", type="primary", key="nn_run")
    n_folds_est = max(0, (fit_panel["date"].dt.year.max()
                          - fit_panel["date"].dt.year.min()
                          - int(init_years)) // int(step_years) + 1)
    st.caption(f"{len(fit_panel):,} rows in scope · ~{n_folds_est} folds × "
               f"{int(n_seeds)} net(s) = ~{n_folds_est * int(n_seeds)} fits. "
               f"A few minutes at defaults.")

_key = (len(fit_panel), tuple(features), horizon, tgt_mode, arch, int(n_seeds),
        int(init_years), int(step_years), float(alpha),
        tuple(uni["constraints"]), tuple(uni["sectors"]), uni["years"])
if run:
    bar = st.progress(0.0, "Fitting…")
    try:
        res = walk_forward_net(
            fit_panel, features, target_col, arch=arch, n_seeds=int(n_seeds),
            alpha=float(alpha), init_train_years=int(init_years),
            step_years=int(step_years), nw_lag=FWD_COLS[horizon] - 1,
            # Purge: a row dated just before the cut has a label realised
            # inside the test window. Without dropping the last `horizon`
            # months of training, the net scores that overlap, not the future
            # (2026-09-06: unpurged NN3 +0.038 -> purged +0.008; the wide
            # nets' "capacity gains" collapsed the same way).
            embargo_months=FWD_COLS[horizon],
            progress=lambda f, m: bar.progress(min(f, 1.0), m))
        bar.progress(1.0, "Done")
        st.session_state["nn_res"] = res
        st.session_state["nn_key"] = _key
        st.session_state["nn_full_fit"] = None      # net changed → refit for TCAV
        st.session_state["nn_tcav"] = None
    except Exception as e:
        bar.empty()
        st.error(f"Fit failed: {e}")

res = st.session_state.get("nn_res")
if res is None:
    st.info("Fit the walk-forward to see out-of-sample performance and unlock "
            "the concept alignment step.")
    st.stop()
if st.session_state.get("nn_key") != _key:
    st.info("Settings changed since the last fit — press **Fit walk-forward** "
            "to refresh. The results below are from the previous run.")

# ── Results ──────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("OOS R²", f"{res.pooled_r2:.2%}",
          help="Gu–Kelly–Xiu out-of-sample R², benchmarked against a zero "
               "forecast rather than the mean. Monthly stock returns are "
               "almost all noise: GKX's best models score ~0.4% here, so "
               "small positive numbers are the win condition. With the "
               "over-market target, the zero forecast IS the market — a "
               "positive number means the net beats 'every stock is average'.")
m2.metric("Mean monthly IC", f"{res.ic_mean:+.3f}",
          help="Spearman rank correlation of prediction vs realised return, "
               "per month, averaged.")
m3.metric("IC t-stat (NW)", f"{res.ic_t:+.2f}",
          help="Newey-West with lag = horizon − 1, matching the rest of the app.")
m4.metric("Rows scored", f"{res.panel[PRED_COL].notna().sum():,}",
          help="Rows in some test window. The initial training block is never "
               "scored — no model existed yet that had not seen it.")

if res.skipped:
    with st.expander(f"{len(res.skipped)} fold(s) skipped"):
        for s in res.skipped:
            st.caption(f"· {s}")

t_folds, t_ic, t_tilt = st.tabs(["Fold performance", "Monthly IC", "Neural tilt"])

with t_folds:
    folds = res.folds
    if len(folds):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=folds["train_end"], y=folds["r2_oos"],
                             name="OOS R²", marker_color=BLUE,
                             marker_line_width=0))
        fig.add_trace(go.Scatter(x=folds["train_end"], y=folds["mean_ic"],
                                 name="mean IC", yaxis="y2", mode="lines+markers",
                                 line=dict(color=RED, width=2),
                                 marker=dict(size=7)))
        fig.update_layout(yaxis=dict(title="OOS R² (per fold)", tickformat=".1%"),
                          yaxis2=dict(title="mean monthly IC", overlaying="y",
                                      side="right", showgrid=False))
        st.plotly_chart(style(fig, 340), width="stretch")
        st.caption("One bar per fold, at the date its training set ended. A "
                   "fold with negative R² but positive IC ranks stocks in the "
                   "right order while missing the level — common, and fine for "
                   "a long-short use of the signal.")
        st.dataframe(
            folds.assign(train_end=folds["train_end"].dt.strftime("%Y-%m"),
                         test_end=folds["test_end"].dt.strftime("%Y-%m"))
                 .style.format({"r2_oos": "{:.2%}", "mean_ic": "{:+.3f}"}),
            width="stretch", hide_index=True)
    else:
        st.info("No folds completed — the panel may be too short for the "
                "initial training window.")

with t_ic:
    ic = res.monthly_ic.dropna()
    if len(ic):
        colors = np.where(ic.to_numpy() >= 0, BLUE, RED)
        fig = go.Figure(go.Bar(x=ic.index, y=ic.values, marker_color=colors,
                               marker_line_width=0))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        roll = ic.rolling(12).mean()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, name="12m mean",
                                 line=dict(color=INK_2, width=2)))
        fig.update_yaxes(title="Monthly OOS IC")
        st.plotly_chart(style(fig, 320), width="stretch")
        st.caption("Every month's out-of-sample rank correlation. The 12-month "
                   "mean is the line to read — single months are noise.")

with t_tilt:
    p = res.panel.copy()
    p[TILT_COL] = p.groupby("date")[PRED_COL].rank(pct=True)
    n_missing = int(p[TILT_COL].isna().sum())
    p[TILT_COL] = p[TILT_COL].fillna(0.5)
    out = p[["date", "ticker", TILT_COL]].rename(columns={"ticker": "secid"})
    st.dataframe(out.head(20), width="stretch", hide_index=True)
    d1, d2 = st.columns([1, 3])
    d1.download_button("Download tilt (CSV)", out.to_csv(index=False),
                       file_name="neural_tilt.csv", mime="text/csv",
                       type="primary", width="stretch", key="nn_dl")
    if d2.button("Send to Edge Concierge", width="content", key="nn_send"):
        st.session_state["neural_tilt_panel"] = out
        st.success(f"Available in the Edge Concierge as the feature "
                   f"`{TILT_COL}` — join is on date and security id.")
    st.caption(f"Rank of the out-of-sample prediction within each date; "
               f"{n_missing:,} pre-coverage rows sit at a neutral 0.5. Use it "
               f"as a layer in the Edge Concierge like any other feature.")

# ── 4 · Concept alignment ────────────────────────────────────────────────────
with st.container(border=True, key="step4"):
    st.subheader(
        "4 · Concept alignment (TCAV)",
        help="Testing with Concept Activation Vectors (Kim et al., 2018). "
             "Define a concept by example rows; a linear probe finds the "
             "direction 'toward the concept' inside a hidden layer; the score "
             "is the fraction of stock-months where nudging the representation "
             "that way raises the predicted return. Runs on a full-history "
             "fit of the same architecture — descriptive, never scored from.")

    sizes = ARCHS[arch]
    n_layers = len(sizes)

    # Optional natural-language concept sketcher (needs ANTHROPIC_API_KEY),
    # gated the same way as the Edge Concierge's English box.
    try:
        from nl import api_key, sketch_concept
        _nl_key = api_key()
    except Exception:
        _nl_key = None

    _sources = ["Preset", "Panel rules", "Company–date list"] + (
        ["Describe in English"] if _nl_key else [])
    lc1, lc2, lc3 = st.columns([2, 1, 1])
    src = lc1.radio("Concept from", _sources,
                    horizontal=True, key="nn_src",
                    help="Presets and rules carve the concept out of the "
                         "panel's own features. A company–date list defines it "
                         "by demonstration — e.g. a manager's actual holdings. "
                         "The English box drafts rules for you and shows its "
                         "working; correct them in the rule builder.")
    _layer_opts = list(range(0, n_layers))      # 0 = inputs
    layer = lc2.selectbox(
        "Probe layer", _layer_opts, index=len(_layer_opts) - 1, key="nn_layer",
        format_func=lambda l: (f"inputs ({len(features)} ranks)" if l == 0
                               else f"hidden {l} ({sizes[l - 1]} units)"),
        help="Where the CAV lives. Deeper layers hold more abstract, more "
             "prediction-shaped representations. The final hidden layer is "
             "excluded: with only a linear readout above it, the directional "
             "derivative points the same way for every row and the score "
             "degenerates to 0 or 1.")
    n_runs = lc3.slider("Resamples", 4, 20, 8, 2, key="nn_nruns",
                        help="CAVs refit against fresh random negatives. More "
                             "resamples → a tighter score distribution and a "
                             "more trustworthy significance test.")

    pos_mask, concept_name = None, ""
    fp = fit_panel.reset_index(drop=True)

    if src == "Preset":
        preset = st.selectbox("Concept", list(PRESETS), key="nn_preset")
        rules = [r for r in PRESETS[preset] if r.feature in fp.columns]
        st.caption(" AND ".join(f"`{r.feature}` in the {r.side} {r.pct:.0%} "
                                f"of each month" for r in rules))
        pos_mask, concept_name = rule_mask(fp, rules), preset

    elif src == "Panel rules":
        if "nn_rules" not in st.session_state:
            st.session_state["nn_rules"] = pd.DataFrame(
                [{"feature": "gpa", "side": "top", "pct": 0.2}])
        edited = st.data_editor(
            st.session_state["nn_rules"], num_rows="dynamic", hide_index=True,
            key="nn_rules_editor", width="stretch",
            column_config={
                "feature": st.column_config.SelectboxColumn(
                    "Feature", options=features, width="medium"),
                "side": st.column_config.SelectboxColumn(
                    "Side", options=["top", "bottom"], width="small"),
                "pct": st.column_config.NumberColumn(
                    "Fraction", min_value=0.05, max_value=0.5, step=0.05,
                    format="%.2f",
                    help="0.2 = quintile. Within each month; rows AND together."),
            })
        rules = [QuantileRule(r.feature, r.side, float(r.pct))
                 for r in edited.dropna().itertuples() if r.feature in fp.columns]
        if rules:
            pos_mask = rule_mask(fp, rules)
            concept_name = " & ".join(f"{r.side} {r.pct:.0%} {r.feature}"
                                      for r in rules)

    elif src == "Describe in English":
        desc = st.text_area(
            "Describe the concept", height=86, key="nn_nl_desc",
            placeholder="e.g. companies earning returns above their cost of "
                        "capital with a large opportunity to reinvest",
            help="Loose or precise both work: “quality compounders”, or “gpa "
                 "top quintile and asset growth above median”. The mapping "
                 "onto panel features is shown below with its caveats, and "
                 "you can copy it into the rule builder to correct by hand.")
        if st.button("Shape the concept", key="nn_nl_go",
                     disabled=not desc.strip()):
            try:
                with st.spinner("Mapping the idea onto the panel…"):
                    st.session_state["nn_nl_plan"] = (desc, sketch_concept(
                        desc, features, _nl_key))
            except Exception as e:
                st.error(f"Couldn't sketch the concept: {e}")

        _plan = st.session_state.get("nn_nl_plan")
        if _plan is not None:
            plan_desc, plan = _plan
            if plan_desc != desc:
                st.info("The description changed since this mapping was "
                        "drafted — press **Shape the concept** to redo it.")
            # normalise before trusting: features must exist, sides be valid,
            # fractions sit in the range the rule builder itself enforces
            rules = [QuantileRule(r.feature,
                                  "top" if r.side.lower().startswith("t") else "bottom",
                                  float(min(max(r.pct, 0.05), 0.5)))
                     for r in plan.rules if r.feature in fp.columns]
            dropped = len(plan.rules) - len(rules)
            if rules:
                st.markdown(f"**{plan.name}** — " + " AND ".join(
                    f"`{r.feature}` in the {r.side} {r.pct:.0%} of each month"
                    for r in rules))
                st.caption(plan.rationale)
                if plan.caveats.strip():
                    st.warning(f"What the panel can't express: {plan.caveats}")
                if dropped:
                    st.warning(f"{dropped} rule(s) named features not in the "
                               f"panel and were dropped.")
                pos_mask, concept_name = rule_mask(fp, rules), plan.name
                if st.button("Copy to rule builder", key="nn_nl_copy",
                             help="Puts these rows in the Panel rules editor "
                                  "so you can tweak them by hand."):
                    st.session_state["nn_rules"] = pd.DataFrame(
                        [{"feature": r.feature, "side": r.side, "pct": r.pct}
                         for r in rules])
                    st.session_state.pop("nn_rules_editor", None)
                    st.success("Copied — switch **Concept from** to "
                               "**Panel rules** to edit.")
            else:
                st.error("None of the drafted rules matched panel features — "
                         "try describing the idea differently.")

    else:
        up = st.file_uploader(
            "Company–date CSV", type="csv", key="nn_concept_csv",
            help="One row per (security, date) that expresses the concept — "
                 "e.g. every holding of a fund you admire, quarter by quarter. "
                 "Matched to the panel on ticker + month. Held in memory for "
                 "this session only.")
        if up is not None:
            up_df = pd.read_csv(up)
            uc = list(up_df.columns)

            def _g(names, fb=0):
                for i, c in enumerate(uc):
                    if any(n in c.lower() for n in names):
                        return i
                return fb

            u1, u2 = st.columns(2)
            udate = u1.selectbox("Date column", uc, index=_g(("date", "period")),
                                 key="nn_udate")
            uid = u2.selectbox("Security column", uc,
                               index=_g(("ticker", "secid", "symbol", "id"), 1 % len(uc)),
                               key="nn_uid")
            pos_mask = match_uploaded(fp, up_df, udate, uid)
            concept_name = up.name
            st.caption(f"{len(up_df):,} uploaded rows → **{int(pos_mask.sum()):,}** "
                       f"matched panel rows (ticker + month). Unmatched rows "
                       f"are usually names outside the S&P 500 panel or dates "
                       f"outside {fp['date'].dt.year.min()}–"
                       f"{fp['date'].dt.year.max()}.")

    if pos_mask is not None:
        n_pos = int(pos_mask.sum())
        ok = n_pos >= MIN_CONCEPT_ROWS
        st.caption(f"Concept **{concept_name or '—'}**: {n_pos:,} example rows "
                   f"out of {len(fp):,}."
                   + ("" if ok else f" Needs at least {MIN_CONCEPT_ROWS}."))
        go_tcav = st.button("Measure alignment", type="primary",
                            disabled=not ok, key="nn_tcav_btn")
    else:
        go_tcav = False
        st.caption("Define a concept to measure against.")

if go_tcav and pos_mask is not None:
    full = st.session_state.get("nn_full_fit")
    if full is None or full.get("_key") != _key:
        with st.spinner(f"Fitting {arch} ensemble on all history — the "
                        f"descriptive model TCAV interrogates…"):
            full = fit_full_history(fp, features, target_col, arch=arch,
                                    n_seeds=int(n_seeds), alpha=float(alpha))
            full["_key"] = _key
            st.session_state["nn_full_fit"] = full
    bar = st.progress(0.0, "Fitting concept vectors…")
    try:
        rep = run_tcav(full, pos_mask, layer=int(layer), n_runs=int(n_runs),
                       progress=lambda f, m: bar.progress(min(f, 1.0), m))
        bar.empty()
        st.session_state["nn_tcav"] = (concept_name, int(layer), rep,
                                       pos_mask.to_numpy())
    except ValueError as e:
        bar.empty()
        st.error(str(e))

if st.session_state.get("nn_tcav"):
    cname, clayer, rep, pmask = st.session_state["nn_tcav"]
    st.markdown(f"#### Alignment with “{cname}” · layer {clayer}")

    score, spread = rep.scores.mean(), rep.scores.std()
    acc = rep.cav_accs.mean()
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("TCAV score", f"{score:.2f} ± {spread:.2f}",
              help="Fraction of stock-months where moving the representation "
                   "toward the concept raises the predicted return. 0.5 = "
                   "orthogonal; 1.0 = uniformly aligned; 0.0 = anti-aligned. "
                   "Mean ± sd across ensemble members × negative resamples.")
    a2.metric("CAV accuracy", f"{acc:.0%}",
              help="Held-out accuracy of the linear probe separating concept "
                   "from random activations. Near 50% means the net does not "
                   "encode the concept at this layer — and the TCAV score "
                   "above is then noise, whatever it says.")
    a3.metric("p vs random", f"{rep.p_value:.1e}",
              help="Two-sample t-test of the concept's scores against "
                   "same-size random pseudo-concepts run through the "
                   "identical pipeline.")
    a4.metric("Concept rows", f"{rep.n_pos:,}")

    for note in rep.notes:
        st.warning(note)
    sig = rep.p_value < 0.01 and acc >= 0.6
    if sig and score > 0.5:
        st.success(f"The forecast is **aligned** with {cname}: in "
                   f"{score:.0%} of stock-months, looking more like the "
                   f"concept raises the predicted return — and the effect "
                   f"clearly separates from random concepts.")
    elif sig and score < 0.5:
        st.error(f"The forecast is **anti-aligned** with {cname}: looking "
                 f"more like the concept *lowers* the predicted return in "
                 f"{1 - score:.0%} of stock-months.")
    else:
        st.info(f"No reliable alignment between the forecast and {cname} — "
                f"the score does not separate from random concepts at this "
                f"layer, so read it as orthogonal rather than weakly aligned.")

    v1, v2 = st.columns(2)
    with v1:
        fig = go.Figure()
        fig.add_trace(go.Box(y=rep.random_scores, name="random concepts",
                             marker_color=MUTED, boxpoints="all", jitter=0.4,
                             pointpos=0))
        fig.add_trace(go.Box(y=rep.scores, name=cname[:24] or "concept",
                             marker_color=BLUE, boxpoints="all", jitter=0.4,
                             pointpos=0))
        fig.add_hline(y=0.5, line_dash="dash", line_color=INK_2, line_width=1)
        fig.update_yaxes(title="TCAV score", range=[-0.02, 1.02])
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 320), width="stretch")
        st.caption("Every CAV as a point. The concept only means something if "
                   "its cloud sits clear of the random one.")
    with v2:
        by = rep.by_year
        fig = go.Figure(go.Scatter(x=by.index, y=by.values, mode="lines+markers",
                                   line=dict(color=BLUE, width=2),
                                   marker=dict(size=6)))
        fig.add_hline(y=0.5, line_dash="dash", line_color=INK_2, line_width=1,
                      annotation_text="orthogonal",
                      annotation_position="bottom left")
        fig.update_yaxes(title="Alignment by year", range=[-0.02, 1.02])
        st.plotly_chart(style(fig, 320), width="stretch")
        st.caption("The same measure grouped by the evaluation rows' year: "
                   "whether the concept helps a stock's forecast in that era. "
                   "The model is one full-history fit — what varies is where "
                   "the stocks sit in its representation space.")

    # ── Best ideas within the concept ────────────────────────────────────────
    # Three per-name signals, one honest split. The return leg (neural tilt)
    # is the walk-forward OOS prediction rank — the only number here produced
    # under the no-lookahead protocol. Expression and alignment come from the
    # descriptive full-history fit: they say how strongly a name embodies the
    # concept and whether its forecast runs THROUGH the concept, not whether
    # to buy it.
    st.markdown("#### Best ideas within the concept")
    if rep.expression is None or len(rep.expression) != len(fp):
        st.info("The universe changed since this alignment run — press "
                "**Measure alignment** again to refresh the idea list.")
        st.stop()
    ideas = fp[["date", "ticker"] +
               (["sector"] if "sector" in fp.columns else [])].copy()
    ideas["expression"] = rep.expression
    ideas["align"] = rep.row_align
    ideas["member"] = pmask
    _tp = res.panel.copy()
    _tp["tilt"] = _tp.groupby("date")[PRED_COL].rank(pct=True)
    ideas = ideas.merge(_tp[["date", "ticker", "tilt"]],
                        on=["date", "ticker"], how="left")

    m_dates = sorted(ideas.loc[ideas["member"], "date"].unique())
    if not m_dates:
        st.info("No concept members in the fitted universe.")
    else:
        b1, b2 = st.columns([1, 3])
        asof = b1.selectbox(
            "As of", m_dates, index=len(m_dates) - 1, key="nn_asof",
            format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m"),
            help="Dates where the concept has members. Defaults to the most "
                 "recent — the closest thing to a live idea list this panel "
                 "can produce.")
        snap = ideas[ideas["date"] == asof].copy()
        mem = snap[snap["member"]]
        b2.caption(f"{len(mem):,} concept members of {len(snap):,} names. "
                   f"Strong ideas sit upper-right: expressing the philosophy "
                   f"AND ranked high by the out-of-sample forecast.")

        fig = go.Figure()
        rest = snap[~snap["member"]]
        fig.add_trace(go.Scatter(
            x=rest["expression"], y=rest["tilt"], mode="markers",
            name="rest of universe", marker=dict(color=GRID, size=5),
            text=rest["ticker"], hovertemplate="%{text}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=mem["expression"], y=mem["tilt"], mode="markers",
            name="concept members", marker=dict(color=BLUE, size=8),
            text=mem["ticker"],
            hovertemplate="%{text}<br>expression %{x:.2f} · tilt %{y:.2f}"
                          "<extra></extra>"))
        fig.update_xaxes(title="Concept expression (pooled pct-rank)")
        fig.update_yaxes(title="Neural tilt (OOS forecast rank)")
        st.plotly_chart(style(fig, 380), width="stretch")

        if mem["tilt"].isna().all():
            st.warning("No out-of-sample forecasts at this date — it falls "
                       "inside the initial training block. Pick a later date.")
        else:
            top = (mem.dropna(subset=["tilt"])
                      .sort_values("tilt", ascending=False))
            show_cols = [c for c in ("ticker", "sector", "tilt", "expression",
                                     "align") if c in top.columns]
            st.dataframe(
                top[show_cols].head(15).style.format(
                    {"tilt": "{:.2f}", "expression": "{:.2f}",
                     "align": "{:.0%}"}),
                width="stretch", hide_index=True)
            st.caption(
                "Members ranked by the OOS forecast. `expression`: how "
                "strongly the net sees the concept in the name. `align`: the "
                "share of concept vectors under which pushing this name "
                "further toward the concept raises its forecast — near 100% "
                "means the model likes it *because of* the philosophy; near "
                "0% means it likes it despite it, so the idea leans on "
                "something else.")
            st.download_button(
                "Download members (CSV)",
                top[show_cols].to_csv(index=False),
                file_name=f"concept_ideas_{pd.Timestamp(asof):%Y%m}.csv",
                mime="text/csv", key="nn_ideas_dl")
