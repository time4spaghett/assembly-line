"""
Factor Edge — a conditional autoencoder (Gu–Kelly–Xiu 2021) on the panel.

Where the Neural Edge fits an unconstrained forecast g(z), this tool imposes
factor structure: r = beta(z)' f, characteristics-driven loadings times latent
factors distilled from the month's own cross-section of returns. Expected
returns can only arise as compensation for factor exposure, which regularizes
hard — the reason this family tests well on small panels.

Same protocol as everything else here: expanding walk-forward, out-of-sample
predictions, a descriptive full-history fit (never scored from) for what the
factors and loadings look like, and a tilt the Edge Concierge can take.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from autoencoder import (BETA_ARCHS, INIT_TRAIN_YEARS, N_FACTORS, N_SEEDS,
                         PRED_COL, STEP_YEARS, beta_profile, full_history_ca,
                         walk_forward_ca)
from data_io import load_base_panel
from engine import FWD_COLS, feature_columns
from ui import BLUE, GRID, INK_2, MUTED, RED, ramp, style, universe_block

TILT_COL = "ca_tilt"
ARCH_LABELS = {"CA0": "CA0 — linear betas",
               "CA1": "CA1 — one hidden layer (32) · GKX workhorse",
               "CA2": "CA2 — two hidden layers (32·16)"}

st.title("Factor Edge")


@st.cache_data(show_spinner="Loading base panel…")
def base_panel() -> pd.DataFrame:
    return load_base_panel()


panel = base_panel()
all_features = feature_columns(panel)

# ── 1 · Target & features ────────────────────────────────────────────────────
with st.container(border=True, key="step1"):
    st.subheader(
        "1 · Target & features",
        help="The model factors the RAW forward return — no market demeaning "
             "needed, because the first managed portfolio is the equal-weight "
             "market and a market-like latent factor absorbs the common "
             "component on its own. Inputs are the GKX [−1, 1] per-month "
             "feature ranks.")
    c1, c2 = st.columns([1, 3])
    horizon = c1.selectbox("Factor", list(FWD_COLS), index=3, key="ca_horizon",
                           format_func=lambda h: f"{FWD_COLS[h]}-month forward return",
                           help="GKX run the model monthly; 12 months is the "
                                "default here because that is where this "
                                "panel's signal lives. Longer horizons overlap "
                                "and the t-stats correct for it.")
    features = c2.multiselect("Features", all_features, default=all_features,
                              key="ca_features")
    st.caption(f"{len(panel):,} rows · S&P 500 point-in-time · "
               f"{panel['date'].dt.year.min()}–{panel['date'].dt.year.max()}")

if not features:
    st.warning("Select at least one feature.")
    st.stop()

# ── 2 · Universe ─────────────────────────────────────────────────────────────
with st.container(border=True, key="step2"):
    st.subheader("2 · Universe",
                 help="Fit on a slice rather than everything. Blank by default.")
    uni = universe_block(panel, "ca", features)
fit_panel = uni["panel"]

# ── 3 · Fit ──────────────────────────────────────────────────────────────────
with st.container(border=True, key="step3"):
    st.subheader(
        "3 · Fit",
        help="Joint training of the beta network and the factor weights by "
             "minibatch Adam (one month per step), with the LAST 15% of each "
             "fold's training months as a chronological early-stopping set. "
             "Expanding walk-forward, refit from scratch, seed ensemble — "
             "every prediction comes from models that never saw the row.")
    f1, f2, f3, f4, f5, f6 = st.columns([1, 2, 1, 1, 1, 1])
    n_k = f1.number_input("Factors (K)", 1, 8, N_FACTORS, 1, key="ca_k",
                          help="Latent factors. GKX find little beyond 5–6; "
                               "one of them will act like the market.")
    arch = f2.selectbox("Beta network", list(BETA_ARCHS), index=1, key="ca_arch",
                        format_func=lambda a: ARCH_LABELS[a],
                        help="CA0 is conditional PCA/IPCA-flavoured; hidden "
                             "layers let loadings depend nonlinearly on "
                             "characteristics.")
    n_seeds = f3.number_input("Ensemble", 1, 10, N_SEEDS, 1, key="ca_seeds")
    init_years = f4.number_input("Initial train (yrs)", 3, 20,
                                 INIT_TRAIN_YEARS, 1, key="ca_init")
    step_years = f5.number_input("Step (yrs)", 1, 5, STEP_YEARS, 1, key="ca_step")
    alpha = f6.select_slider("L2 penalty", [1e-5, 1e-4, 1e-3, 1e-2],
                             value=1e-4, key="ca_alpha",
                             format_func=lambda a: f"{a:g}")
    use_huber = st.toggle(
        "Huber loss (robust)", value=True, key="ca_huber",
        help="Clips each residual's influence at 2× the target's pooled "
             "standard deviation, so a handful of extreme months can't steer "
             "the fit — the GKX robustness device, free in a hand-rolled "
             "trainer.")
    run = st.button("Fit walk-forward", type="primary", key="ca_run")
    n_folds_est = max(0, (fit_panel["date"].dt.year.max()
                          - fit_panel["date"].dt.year.min()
                          - int(init_years)) // int(step_years) + 1)
    st.caption(f"{len(fit_panel):,} rows in scope · ~{n_folds_est} folds × "
               f"{int(n_seeds)} model(s). The nets are tiny — this runs "
               f"faster than the Neural Edge.")

_key = (len(fit_panel), tuple(features), horizon, int(n_k), arch, int(n_seeds),
        int(init_years), int(step_years), float(alpha), bool(use_huber),
        tuple(uni["constraints"]), tuple(uni["sectors"]), uni["years"])
_huber_thr = (2.0 * float(pd.to_numeric(fit_panel[horizon], errors="coerce").std())
              if use_huber else None)
if run:
    bar = st.progress(0.0, "Fitting…")
    try:
        res = walk_forward_ca(
            fit_panel, features, horizon, n_factors=int(n_k), arch=arch,
            n_seeds=int(n_seeds), alpha=float(alpha), huber=_huber_thr,
            init_train_years=int(init_years), step_years=int(step_years),
            nw_lag=FWD_COLS[horizon] - 1,
            progress=lambda f, m: bar.progress(min(f, 1.0), m))
        bar.progress(1.0, "Done")
        st.session_state["ca_res"] = res
        st.session_state["ca_key"] = _key
        st.session_state["ca_full"] = None
    except Exception as e:
        bar.empty()
        st.error(f"Fit failed: {e}")

res = st.session_state.get("ca_res")
if res is None:
    st.info("Fit the walk-forward to see out-of-sample performance, the "
            "latent factors and their premia.")
    st.stop()
if st.session_state.get("ca_key") != _key:
    st.info("Settings changed since the last fit — press **Fit walk-forward** "
            "to refresh. The results below are from the previous run.")

# ── Results ──────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total R² (OOS)", f"{res.total_r2:.1%}",
          help="Fit against each month's REALIZED factors — observable ex "
               "post, since factors are portfolio returns. This is the "
               "asset-pricing test: how much of returns is factor structure. "
               "GKX report ~10–20% monthly on all-CRSP.")
m2.metric("Pred R² (OOS)", f"{res.pred_r2:.2%}",
          help="Forecast using estimated factor premia instead of realized "
               "factors — the honest prediction number. Small positives win, "
               "as with the Neural Edge.")
m3.metric("Mean IC", f"{res.ic_mean:+.3f}",
          help="Spearman of β·λ vs realised return, per month, averaged. "
               f"Newey-West t = {res.ic_t:+.2f} at lag = horizon − 1.")
m4.metric("Rows scored", f"{res.panel[PRED_COL].notna().sum():,}")

if res.skipped:
    with st.expander(f"{len(res.skipped)} fold(s) skipped"):
        for s in res.skipped:
            st.caption(f"· {s}")

t_folds, t_prem, t_load, t_tilt = st.tabs(
    ["Fold performance", "Factors & premia", "Loadings", "Factor tilt"])

with t_folds:
    folds = res.folds
    if len(folds):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=folds["train_end"], y=folds["total_r2"],
                             name="total R²", marker_color=BLUE,
                             marker_line_width=0))
        fig.add_trace(go.Scatter(x=folds["train_end"], y=folds["mean_ic"],
                                 name="mean IC", yaxis="y2",
                                 mode="lines+markers",
                                 line=dict(color=RED, width=2),
                                 marker=dict(size=7)))
        fig.update_layout(yaxis=dict(title="OOS total R²", tickformat=".0%"),
                          yaxis2=dict(title="mean monthly IC", overlaying="y",
                                      side="right", showgrid=False))
        st.plotly_chart(style(fig, 340), width="stretch")
        st.caption("Total R² is the pricing test (does factor structure "
                   "describe returns); IC is the forecasting test (do the "
                   "premia rank stocks). A model can pass the first and fail "
                   "the second — that gap IS the difference between risk and "
                   "alpha.")
        st.dataframe(
            folds.assign(train_end=folds["train_end"].dt.strftime("%Y-%m"),
                         test_end=folds["test_end"].dt.strftime("%Y-%m"))
                 .style.format({"total_r2": "{:.1%}", "pred_r2": "{:.2%}",
                                "mean_ic": "{:+.3f}"}),
            width="stretch", hide_index=True)

with t_prem:
    st.caption("From a full-history fit — descriptive, never scored from, "
               "mirroring the Learned Edge convention.")
    if st.button("Fit full history", key="ca_full_btn"):
        with st.spinner(f"Fitting {res.arch} × {res.n_seeds} on all history…"):
            st.session_state["ca_full"] = full_history_ca(
                fit_panel, features, horizon, n_factors=int(n_k), arch=arch,
                n_seeds=int(n_seeds), alpha=float(alpha), huber=_huber_thr)
    full = st.session_state.get("ca_full")
    if full:
        F, lam = full["factors"], full["premia"]
        months = FWD_COLS[horizon]
        ann = (1 + lam) ** (12 / months) - 1
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=ann.index, y=ann.values,
                marker_color=[BLUE if v >= 0 else RED for v in ann.values],
                marker_line_width=0))
            fig.update_yaxes(title="Estimated premium (annualized)",
                             tickformat=".1%")
            st.plotly_chart(style(fig, 300), width="stretch")
            st.caption("Mean factor realization, annualized. Signs are "
                       "arbitrary (latent factors identify up to rotation) — "
                       "the products β·λ are what's pinned down.")
        with c2:
            cum = (1 + F).cumprod()
            fig = go.Figure()
            for i, k in enumerate(cum.columns):
                fig.add_trace(go.Scatter(x=cum.index, y=cum[k], name=k,
                                         line=dict(color=ramp(len(cum.columns))[i],
                                                   width=2)))
            fig.update_yaxes(title="Cumulative factor value", type="log")
            st.plotly_chart(style(fig, 300), width="stretch")
            st.caption("Latent factor paths (log scale). At horizons past one "
                       "month the realizations overlap, so read levels, not "
                       "wiggles. One factor typically tracks the market.")

with t_load:
    full = st.session_state.get("ca_full")
    if not full:
        st.info("Fit the full history (previous tab) to inspect loadings.")
    else:
        feat = st.selectbox("Characteristic", features, key="ca_load_feat",
                            help="Sweep this feature's rank over [−1, 1] with "
                                 "every other input at its median; each line "
                                 "is one factor's loading response. A slice "
                                 "of a nonlinear surface — the marginal "
                                 "shape, not a decomposition.")
        prof = beta_profile(full["models"], features.index(feat), len(features))
        zs = np.linspace(-1, 1, prof.shape[0])
        fig = go.Figure()
        for k in range(prof.shape[1]):
            fig.add_trace(go.Scatter(x=zs, y=prof[:, k], name=f"F{k + 1}",
                                     line=dict(color=ramp(prof.shape[1])[k],
                                               width=2)))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_xaxes(title=f"{feat} (cross-sectional rank, −1…1)")
        fig.update_yaxes(title="Factor loading β")
        st.plotly_chart(style(fig, 340), width="stretch")
        spans = []
        for j, fname in enumerate(features):
            pj = beta_profile(full["models"], j, len(features), grid=5)
            spans.append({"feature": fname,
                          "span": float(np.abs(pj[-1] - pj[0]).max())})
        top = (pd.DataFrame(spans).sort_values("span", ascending=False)
               .head(10).reset_index(drop=True))
        st.dataframe(top.style.format({"span": "{:.3f}"}),
                     width="stretch", hide_index=True)
        st.caption("Characteristics ranked by how much they move some "
                   "loading across their range — the model's own view of "
                   "which features define exposure.")

with t_tilt:
    p = res.panel.copy()
    p[TILT_COL] = p.groupby("date")[PRED_COL].rank(pct=True)
    n_missing = int(p[TILT_COL].isna().sum())
    p[TILT_COL] = p[TILT_COL].fillna(0.5)
    out = p[["date", "ticker", TILT_COL]].rename(columns={"ticker": "secid"})
    st.dataframe(out.head(20), width="stretch", hide_index=True)
    d1, d2 = st.columns([1, 3])
    d1.download_button("Download tilt (CSV)", out.to_csv(index=False),
                       file_name="ca_tilt.csv", mime="text/csv",
                       type="primary", width="stretch", key="ca_dl")
    if d2.button("Send to Edge Concierge", width="content", key="ca_send"):
        st.session_state["ca_tilt_panel"] = out
        st.success(f"Available in the Edge Concierge as the feature "
                   f"`{TILT_COL}` — join is on date and security id.")
    st.caption(f"Rank of the out-of-sample β·λ forecast within each date; "
               f"{n_missing:,} pre-coverage rows sit at a neutral 0.5. Note "
               f"what this tilt IS: expected return earned as factor-risk "
               f"compensation, not alpha — buying it is buying exposure.")
