"""
Learned Edge — infer a manager's style from revealed holdings.

Upload a panel of (date, secid, features..., held_flag), fit what separates
held names from the rest under a walk-forward protocol, and emit a per-name
style tilt the Edge Concierge can use as a layer.

The scores are produced out-of-sample by construction: each fold trains only on
prior history and scores the following unseen window. What the model *learned*
is shown separately, from a full-history fit that is never scored from.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from learned import (INIT_TRAIN_YEARS, PROB_COL, SIZE_GROUPS, STEP_YEARS,
                     StyleResult, build_learned_style_panel,
                     full_history_profile, walk_forward_style)
from ui import BLUE, INK_2, ramp, style, universe_block

TILT_COL = "learned_tilt"

st.title("Learned Edge")

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


@st.cache_data(show_spinner="Reading holdings…", max_entries=1)
def _read(data: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(data))


raw = _read(up.getvalue())
cols = list(raw.columns)


def _guess(names, fallback=0):
    for i, c in enumerate(cols):
        if any(n in c.lower().replace(" ", "").replace("_", "") for n in names):
            return i
    return fallback


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

    c6, c7 = st.columns(2)
    _has_ret = any(n in c.lower().replace("_", "") for c in cols
                   for n in ("totalreturn", "fwdreturn", "return"))
    ret_col = c6.selectbox(
        "Return column (for returns-based mode)", _opt,
        index=_guess(("totalreturn", "fwdreturn1m", "fwdreturn", "return", "ret")) + 1
        if _has_ret else 0,
        help="Each name's return over the month *after* `date` — a forward "
             "1-month return, the same convention the Concierge uses. Needed "
             "only for the returns-based style regression.")
    weight_col = c7.selectbox(
        "Holding weight (optional)", _opt,
        index=_guess(("weight", "wgt")) + 1
        if any(n in c.lower() for c in cols for n in ("weight", "wgt")) else 0,
        help="Portfolio weight of each held name. Left blank, the manager's "
             "return is the equal-weight average of held names.")

    _reserved = {date_col, id_col, target_col} | {
        c for c in (sector_col, ret_col, weight_col) if c != "(none)"}
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
    MODE_HOLD, MODE_RET = "Holdings — what it holds", "Returns — what explains its returns"
    mode = st.radio(
        "Learn the style from", [MODE_HOLD, MODE_RET], horizontal=True, key="lmode",
        help="**Holdings**: a classifier on characteristics — what separates "
             "held names from the rest. **Returns**: a Sharpe-style regression "
             "— the manager's monthly return on each factor's long–short return "
             "within the universe; the loadings are the factor mix that best "
             "replicates the manager. Same question, two kinds of evidence.")
    returns_mode = mode == MODE_RET
    if returns_mode and ret_col == "(none)":
        st.error("Returns-based mode needs a return column — map one in step 2.")
        st.stop()

    f1, f2, f3, f4 = st.columns(4)
    if returns_mode:
        kind = "style"
        n_q = f1.selectbox("Ntiles for the L/S legs", [3, 4, 5, 10], index=2,
                           help="Each factor's return is its top ntile minus its "
                                "bottom ntile, equal-weight, ranked within the "
                                "universe each month.")
    else:
        n_q = 5
        kind = f1.selectbox("Model", ["forest", "logit"], index=0,
                            format_func=lambda k: {"forest": "Random forest",
                                                   "logit": "Logistic"}[k],
                            help="The forest can express size × style interactions; "
                                 "the logit runs the identical protocol as a "
                                 "linear comparison.")
    init_years = f2.number_input("Initial train (years)", 1, 20,
                                 min(INIT_TRAIN_YEARS, 3), 1)
    step_years = f3.number_input("Step (years)", 1, 5, STEP_YEARS, 1)
    use_cohorts = f4.toggle("Size cohorts",
                            value=size_col != "(none)" and not returns_mode,
                            disabled=size_col == "(none)" or returns_mode,
                            help="Holdings mode only.")
    run = st.button("Fit walk-forward", type="primary")
    if returns_mode:
        st.caption(f"{len(fit_panel):,} rows in scope · "
                   f"{fit_panel['date'].nunique():,} months. Each fold is one "
                   f"ridge regression on the months before the cut — fast.")
    else:
        st.caption(f"{len(fit_panel):,} rows in scope. A cohort run fits "
                   f"{len(SIZE_GROUPS)}× as many models and takes correspondingly "
                   f"longer.")

_key = (len(fit_panel), tuple(features), target_col, kind, int(init_years),
        int(step_years), bool(use_cohorts), size_col, ret_col, weight_col, n_q,
        tuple(uni["constraints"]), tuple(uni["sectors"]), uni["years"])
if run:
    bar = st.progress(0.0, "Fitting…")
    try:
        if returns_mode:
            res = walk_forward_style(
                fit_panel, features, "_target", ret_col, date_col="date",
                weight_col=(weight_col if weight_col != "(none)" else None),
                n_q=int(n_q), init_train_years=int(init_years),
                step_years=int(step_years),
                progress=lambda f, m: bar.progress(min(f, 1.0), m))
        else:
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
# the stored result decides which view renders, not the radio — otherwise
# flipping the mode before refitting would try to read classifier fields
# off a regression result
returns_mode = isinstance(res, StyleResult)

# ── Results: returns-based ───────────────────────────────────────────────────
if returns_mode:
    folds = res.folds
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pooled OOS correlation", f"{res.pooled_corr:.3f}",
              help="Replicated vs actual manager return over every out-of-sample "
                   "month at once. 0 is no relationship.")
    m2.metric("Pooled OOS R²", f"{res.pooled_r2:.3f}",
              help="Share of the manager's month-to-month variance the factor "
                   "mix explains out of sample. Can go negative when the "
                   "replication is worse than a flat line.")
    m3.metric("Months replicated", f"{res.replicated.notna().sum():,} / {len(res.manager):,}",
              help="The initial training block is never replicated — no fit "
                   "existed yet that had not seen it.")
    _alpha = folds["alpha_ann"].mean() if len(folds) else float("nan")
    m4.metric("Alpha (ann., mean of folds)", f"{_alpha:+.1%}",
              help="The intercept: return the factor legs don't account for. "
                   "Gross, and on the L/S-leg convention, so read the sign more "
                   "than the size.")

    if res.skipped:
        with st.expander(f"{len(res.skipped)} note(s) from the fit"):
            for s in res.skipped:
                st.caption(f"· {s}")

    s_fold, s_load, s_rep, s_send = st.tabs(
        ["Fit by fold", "Loadings", "Replication", "Send to Concierge"])

    with s_fold:
        if len(folds):
            fig = go.Figure(go.Scatter(
                x=folds["train_end"], y=folds["oos_corr"], mode="lines+markers",
                line=dict(color=BLUE, width=2), marker=dict(size=7),
                name="OOS correlation"))
            fig.add_hline(y=0, line_dash="dash", line_color=INK_2, line_width=2,
                          annotation_text="no relationship",
                          annotation_position="bottom left")
            fig.update_yaxes(title="Out-of-sample correlation", range=[-0.5, 1.0])
            st.plotly_chart(style(fig, 340), width="stretch")
            st.caption("One point per fold, at the date the training set ended. "
                       "A line that decays means the factor mix stopped "
                       "explaining the manager — a style drift a pooled number "
                       "would hide.")
            st.dataframe(
                folds.assign(train_end=folds["train_end"].dt.strftime("%Y-%m"),
                             test_end=folds["test_end"].dt.strftime("%Y-%m"))
                     .rename(columns={"n_train": "train months",
                                      "n_test": "test months",
                                      "oos_corr": "OOS corr", "oos_r2": "OOS R²",
                                      "alpha_ann": "alpha (ann.)"})
                     .style.format({"OOS corr": "{:.3f}", "OOS R²": "{:.3f}",
                                    "alpha (ann.)": "{:+.1%}"}),
                width="stretch", hide_index=True)
        else:
            st.info("No folds completed — the panel may be too short for the "
                    "initial training window, or too few months had enough "
                    "names to cut ntiles.")

    with s_load:
        if len(res.loadings):
            ld = res.loadings.head(20).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=ld.values, y=ld.index, orientation="h",
                marker_color=[BLUE if v >= 0 else INK_2 for v in ld.values],
                marker_line_width=0))
            fig.update_xaxes(title="Loading on the factor's L/S return "
                                   "(time-averaged across folds)")
            st.plotly_chart(style(fig, max(280, 22 * len(ld))), width="stretch")
            st.caption("A loading of 0.5 on `bp` means the manager moved half as "
                       "much as a top-minus-bottom `bp` portfolio would, holding "
                       "the other legs fixed. Negative = tilted against it. "
                       "Averaged across folds so each period counts once.")
            if len(res.loadings_by_fold) > 1:
                lbf = res.loadings_by_fold[res.loadings.head(6).index]
                fig2 = go.Figure()
                for i, c in enumerate(lbf.columns):
                    fig2.add_trace(go.Scatter(
                        x=lbf.index, y=lbf[c], name=c, mode="lines",
                        line=dict(color=ramp(len(lbf.columns))[i], width=2)))
                fig2.add_hline(y=0, line_color=INK_2, line_width=1)
                fig2.update_yaxes(title="Loading, by fold")
                st.plotly_chart(style(fig2, 300), width="stretch")
                st.caption("The six largest loadings, fold by fold. Lines that "
                           "hold their level are the durable part of the style; "
                           "one that wanders is either drift or a factor the "
                           "sample can't pin down.")

    with s_rep:
        both = pd.DataFrame({"manager": res.manager, "replicated": res.replicated}).dropna()
        if len(both) > 2:
            cum = (1 + both).cumprod()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cum.index, y=cum["manager"], name="Manager",
                                     mode="lines", line=dict(color=BLUE, width=2)))
            fig.add_trace(go.Scatter(x=cum.index, y=cum["replicated"],
                                     name="Factor replication (OOS)", mode="lines",
                                     line=dict(color=INK_2, width=2, dash="dash")))
            fig.update_yaxes(title="Growth of $1, replicated months only")
            st.plotly_chart(style(fig, 340), width="stretch")
            te = (both["manager"] - both["replicated"]).std() * (12 ** 0.5)
            st.caption(f"Manager vs what the factor mix alone would have "
                       f"returned, using only out-of-sample loadings. Tracking "
                       f"error **{te:.1%}** annualized. The gap is what the "
                       f"factors don't explain: selection, timing, or a factor "
                       f"not in the list.")
        else:
            st.info("Nothing replicated yet.")

    with s_send:
        st.caption("The time-averaged loadings, scaled so the largest is ±1, "
                   "become rows in the Concierge's edge table — transform "
                   "`rank`, joined by `+`. You can edit them there before testing.")
        if len(res.loadings):
            n_send = st.slider("Legs to send", 1, len(res.loadings),
                               min(6, len(res.loadings)),
                               help="Largest loadings first. Fewer is usually "
                                    "better — the tail is mostly noise.")
            top = res.loadings.head(n_send)
            scale = top.abs().max() or 1.0
            rows_out = pd.DataFrame({
                "feature": top.index, "transform": "rank",
                "weight": (top / scale).round(2).values, "op": "+"})
            st.dataframe(rows_out, width="stretch", hide_index=True)
            _csv_cols = set()
            if st.session_state.get("csv_panel") is not None:
                _csv_cols = set(st.session_state["csv_panel"].columns)
            if _csv_cols and not set(top.index) <= _csv_cols:
                st.warning("Some of these feature names aren't in the panel "
                           "currently loaded in the Concierge — it will only "
                           "keep the ones that match.")
            if st.button("Send to Edge Concierge", type="primary"):
                st.session_state["builder"] = rows_out
                st.session_state.pop("builder_editor", None)
                st.session_state["nl_note"] = (
                    f"Loaded from Learned Edge: the returns-based style "
                    f"regression's top {n_send} loading(s), scaled to ±1.")
                st.success("Sent. Open the Edge Concierge — step 4 now holds "
                           "these rows. Load the same panel there so the feature "
                           "names resolve.")
    st.stop()

# ── Results: holdings-based ──────────────────────────────────────────────────
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
