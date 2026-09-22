"""
Hold Propensity — infer an investment philosophy from holdings with an LLM,
compile it to a factor rule, propagate the rule over history.

    Interpret → Compile → Propagate

The LLM sees one cross section (the latest with holdings) and writes two
things: a philosophy in English, and a scoring spec in a small JSON language.
Python validates the spec and applies it identically inside every historical
date, using factor columns only — never the holdings — to produce
`hold_prop_llm` in [0, 1]: how much each name, at each date, resembles what
this process appears to prefer today.

Run:  streamlit run main.py  (this is a page, not the entry point)
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_io import load_base_panel
from engine import feature_columns
from hold_prop import (DEMO_PHILOSOPHY, DEMO_RULE, FACTOR_DEFS, HOLDINGS_COLS,
                       SpecError, add_weight_columns, apply_returns_leg,
                       apply_scoring_spec, composite, demo_holdings, evaluate,
                       evaluate_weighted, fit_returns_leg, implied_fund_returns,
                       infer_philosophy, snapshot, spec_factors,
                       style_scores, style_sleeve_returns, validate_scoring_spec)
from ui import BLUE, INK_2, RED, style

try:
    from nl import api_key as _api_key
    NL_KEY = _api_key()
except Exception:
    NL_KEY = None

OUT_COL = "hold_prop_llm"
BASE_LABEL = "S&P 500 panel + a demo fund"

st.title("Hold Propensity")
st.caption("One LLM pass reads the current holdings and writes an investment "
           "philosophy plus an executable factor rule. Deterministic Python then "
           "scores every name at every date on that rule alone — the LLM never "
           "scores history, and the holdings never enter the historical score.")


@st.cache_data(show_spinner="Loading base panel…")
def base_panel() -> pd.DataFrame:
    return load_base_panel()


@st.cache_data(show_spinner="Building the demo fund…")
def demo_frame(months: int) -> pd.DataFrame:
    p = base_panel()
    h = demo_holdings(p, months=months)
    return p.merge(h, on=["date", "ticker"], how="inner")


# ── 1 · Data ──────────────────────────────────────────────────────────────────
with st.container(border=True, key="hstep1"):
    st.subheader("1 · Holdings and factors", help="A security × date panel with factor "
                 "columns and fund weights. The shipped panel has no fund, so it "
                 "comes with a synthetic one whose philosophy is known — a test of "
                 "whether the pipeline recovers it. Uploads stay in session memory.")
    src = st.radio("Data", [BASE_LABEL, "Upload CSV"], key="hp_src", horizontal=True)
    df = None
    if src == BASE_LABEL:
        months = st.select_slider("Months of demo holdings", [12, 24, 36, 60, 120], 36,
                                  key="hp_months")
        df = demo_frame(int(months))
        date_col, id_col = "date", "ticker"
        factor_cols = [c for c in feature_columns(base_panel())]
        st.caption(f"{df[id_col].nunique():,} names · {df[date_col].nunique()} months · "
                   f"{len(factor_cols)} factors · the demo fund holds 40 names a month. "
                   f"Its rule is hidden until after inference.")
    else:
        up = st.file_uploader("CSV — one row per security per date", type="csv", key="hp_up")
        if up is not None:
            raw = pd.read_csv(io.BytesIO(up.getvalue()))
            cols = list(raw.columns)

            def _guess(names, fallback=0):
                for i, c in enumerate(cols):
                    if any(n in c.lower().replace(" ", "").replace("_", "") for n in names):
                        return i
                return fallback
            c1, c2, c3, c4 = st.columns(4)
            id_col = c1.selectbox("Security ID", cols, index=_guess(("companyid", "ticker", "secid", "id")), key="hp_id")
            date_col = c2.selectbox("Date", cols, index=_guess(("date",), min(1, len(cols) - 1)), key="hp_date")
            fund_col = c3.selectbox("Fund weight", cols, index=_guess(("fundweight", "weight"), 0), key="hp_fw")
            bopts = ["(none)"] + cols
            bench_col = c4.selectbox("Benchmark weight", bopts, key="hp_bw",
                                     index=(_guess(("benchmarkweight", "benchweight", "bmweight"), -1) + 1))
            reserved = {id_col, date_col, fund_col, bench_col} | HOLDINGS_COLS
            num = [c for c in cols if c not in reserved and pd.api.types.is_numeric_dtype(raw[c])]
            factor_cols = st.multiselect("Factor columns", num, default=num, key="hp_factors",
                                         help="Every other numeric column, by default. "
                                              "Holdings and weight columns are never offered.")
            raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
            raw = raw.dropna(subset=[date_col])
            df = add_weight_columns(raw, date_col, id_col, fund_col,
                                    None if bench_col == "(none)" else bench_col)
            st.caption(f"{df[id_col].nunique():,} names · {df[date_col].nunique()} dates · "
                       f"{len(factor_cols)} factors. active / EWM weights computed where "
                       f"the file didn't carry them.")
    if df is None or not factor_cols:
        st.stop()

snap = snapshot(df, date_col)
n_held = int((df.loc[df[date_col] == snap, "fund_weight"] > 0).sum())

# ── 2 · Infer ─────────────────────────────────────────────────────────────────
with st.container(border=True, key="hstep2"):
    st.subheader("2 · Interpret and compile", help="The LLM reads the latest cross "
                 "section with holdings — every held name with its weights and "
                 "factors, next to benchmark names not held — and returns a "
                 "philosophy and a spec. Or paste a spec to skip the LLM entirely.")
    st.caption(f"Snapshot: **{snap:%Y-%m-%d}**, {n_held} names held.")
    c1, c2 = st.columns(2)
    starter = c1.text_area("Starter philosophy (optional)", height=90, key="hp_starter",
                           placeholder="A prior for the model to test against the holdings.")
    info = c2.text_area("Fund / manager information (optional)", height=90, key="hp_info",
                        placeholder="Mandate, style, anything the holdings alone won't say.")
    b1, b2 = st.columns([1, 3])
    run = b1.button("Infer philosophy", type="primary", disabled=not NL_KEY, key="hp_run")
    if not NL_KEY:
        b2.caption("Needs an `ANTHROPIC_API_KEY` (environment or Streamlit Secrets). "
                   "Pasting a spec below works without one.")
    with st.expander("Or paste a scoring spec (JSON)"):
        pasted = st.text_area("Spec", height=160, key="hp_paste", label_visibility="collapsed")
        if st.button("Use this spec", key="hp_use_paste") and pasted.strip():
            try:
                st.session_state["hp_spec"] = validate_scoring_spec(json.loads(pasted), factor_cols)
                st.session_state["hp_phil"] = "(spec supplied directly — no philosophy text)"
                st.session_state["hp_raw"] = ""
            except (SpecError, ValueError) as e:
                st.error(str(e))
    if run:
        try:
            with st.status("Reading the holdings…", expanded=False) as s:
                phil, spec, raw_text = infer_philosophy(
                    df, date_col, id_col, factor_cols, NL_KEY,
                    starter_philosophy=starter or None, fund_info=info or None,
                    factor_defs=FACTOR_DEFS)
                s.update(label="Philosophy inferred, spec validated", state="complete")
            st.session_state["hp_spec"], st.session_state["hp_phil"] = spec, phil
            st.session_state["hp_raw"] = raw_text
        except Exception as e:
            st.error(f"Inference failed: {e}")

spec = st.session_state.get("hp_spec")
if spec is None:
    st.info("Infer a philosophy, or paste a spec, to continue.")
    st.stop()

p1, p2 = st.columns(2)
with p1:
    st.markdown("**Philosophy**")
    st.markdown(st.session_state.get("hp_phil", ""))
with p2:
    st.markdown("**Scoring spec** — validated; the only thing the history sees")
    st.code(json.dumps(spec, indent=2), language="json")
    st.caption(f"Reads {len(spec_factors(spec))} factors: {', '.join(spec_factors(spec))}.")

# ── 3 · Propagate ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Propagating the rule over history…")
def propagate(frame: pd.DataFrame, spec_json: str, date_col: str, fcs: tuple) -> pd.DataFrame:
    return apply_scoring_spec(frame, json.loads(spec_json), date_col, list(fcs))


scored = propagate(df, json.dumps(spec, sort_keys=True), date_col, tuple(factor_cols))
ev = evaluate(scored, OUT_COL, date_col)

with st.container(border=True, key="hstep3"):
    st.subheader("3 · Propagate and check", help="The spec applied inside every date. "
                 "Where holdings exist historically they are the ground truth: AUC "
                 "of the score for held-vs-not, and the rank correlation of the score "
                 "with fund weight among held names. The score never saw them.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Names scored", f"{scored[OUT_COL].notna().sum():,}")
    m2.metric("Dates", f"{scored[date_col].nunique():,}")
    if len(ev):
        m3.metric("AUC, mean over dates", f"{ev['auc'].mean():.3f}",
                  help="0.5 = the rule says nothing about what is held; 1 = perfect.")
        m4.metric("AUC at snapshot", f"{ev['auc'].iloc[-1]:.3f}")
    else:
        m3.metric("AUC", "—", help="No historical holdings to check against.")
    if len(ev) > 1:
        f = go.Figure()
        f.add_trace(go.Scatter(x=ev.index, y=ev["auc"], name="AUC held vs not", mode="lines",
                               line=dict(color=BLUE, width=2)))
        f.add_trace(go.Scatter(x=ev.index, y=ev["rho_weight"], name="Spearman vs fund weight (held)",
                               mode="lines", line=dict(color=RED, width=2)))
        f.add_hline(y=0.5, line_color=INK_2, line_dash="dash", line_width=1)
        f.update_yaxes(range=[-0.2, 1.02])
        st.plotly_chart(style(f, 300), width="stretch")
        st.caption("Static-philosophy approximation: the rule is today's; earlier dates "
                   "measure how much those names resembled what the process prefers now, "
                   "not what the manager actually wanted then. A falling AUC into the "
                   "past is the philosophy drifting.")
    if src == BASE_LABEL:
        truth = scored[[OUT_COL, "demo_true_score"]].dropna()
        rho = float(truth[OUT_COL].corr(truth["demo_true_score"], method="spearman"))
        st.success(f"**The demo fund's hidden rule was:** {DEMO_PHILOSOPHY}  \n"
                   f"Rank correlation between `hold_prop_llm` and that true score, all "
                   f"name-dates: **{rho:+.3f}**. Reads: {', '.join(spec_factors(DEMO_RULE))}.")

    latest = scored[scored[date_col] == snap].copy()
    latest["held"] = latest["fund_weight"] > 0
    show = (latest.sort_values(OUT_COL, ascending=False)
            [[id_col, OUT_COL, "fund_weight", "benchmark_weight", "held"] + spec_factors(spec)[:6]]
            .head(40))
    st.dataframe(show.style.format({OUT_COL: "{:.3f}", "fund_weight": "{:.2%}",
                                    "benchmark_weight": "{:.2%}"}, na_rep="—",
                                   precision=3),
                 width="stretch", hide_index=True, height=420)
    st.caption(f"Top 40 by `hold_prop_llm` at {snap:%Y-%m-%d}: names the rule likes, "
               f"whether or not the fund owns them — the not-held ones are the rule's "
               f"suggestions, the held ones with a low score are where the fund departs "
               f"from its own apparent philosophy.")

    if st.session_state.get("hp_raw"):
        with st.expander("Raw model response"):
            st.text(st.session_state["hp_raw"])

# ── 4 · Returns leg and composite ────────────────────────────────────────────
# A second, independent estimate from the fund's realised returns; the
# composite is the plain average of the two legs.
with st.container(border=True, key="hstep4"):
    st.subheader("4 · Returns leg and composite", help="What factor mix best explains "
                 "the fund's monthly returns, projected onto names. RBSA reads as "
                 "'57% market + 32% low-risk + 11% quality' — the market sleeve absorbs "
                 "beta, size is deliberately not a style. The composite averages this "
                 "with the LLM leg.")
    r1, r2, r3 = st.columns([2, 2, 3])
    mode = r1.radio("Model", ["rbsa", "ridge"], key="hp_rmode", horizontal=True,
                    format_func=lambda m: {"rbsa": "Sharpe RBSA (default)", "ridge": "Ridge on spreads"}[m],
                    help="RBSA: non-negative weights summing to one over the market and "
                         "long-only style sleeves. Ridge: signed loadings on long-short "
                         "family spreads; needs 60+ months to be stable.")
    rsrc = r2.radio("Fund returns", ["Implied by holdings", "Upload monthly NAV returns"],
                    key="hp_rsrc",
                    help="Implied: each date's book held for the next period, from the "
                         "panel's own forward returns (carried forward up to 4 months "
                         "between holdings snapshots). Better: the fund's actual NAV "
                         "return series.")
    fund_ret = None
    if rsrc.startswith("Upload"):
        rup = r3.file_uploader("CSV: date, return (monthly)", type="csv", key="hp_rup")
        if rup is not None:
            rr = pd.read_csv(io.BytesIO(rup.getvalue()))
            rr.iloc[:, 0] = pd.to_datetime(rr.iloc[:, 0], errors="coerce")
            rr = rr.dropna()
            # snap each NAV month to the panel's nearest date at or after month end
            pdates = pd.DatetimeIndex(sorted(scored[date_col].unique()))
            snapd = [pdates[pdates.searchsorted(d - pd.Timedelta(days=6))] if pdates.searchsorted(d - pd.Timedelta(days=6)) < len(pdates) else pd.NaT for d in rr.iloc[:, 0]]
            fund_ret = pd.Series(pd.to_numeric(rr.iloc[:, 1], errors="coerce").to_numpy(), index=snapd).dropna()
            fund_ret = fund_ret[~fund_ret.index.isna()]
    else:
        if "fwd_1m" in scored.columns:
            fund_ret = implied_fund_returns(scored, date_col)
        else:
            r3.warning("The panel has no 1-month forward returns, so returns can't be implied.")


@st.cache_data(show_spinner="Fitting the returns leg…")
def cached_returns(frame: pd.DataFrame, fcs: tuple, date_col: str, fund_ret: pd.Series, mode: str):
    sty = style_scores(frame, date_col, list(fcs))
    sleeves = style_sleeve_returns(frame, sty, date_col)
    fit = fit_returns_leg(fund_ret, sleeves, mode)
    return apply_returns_leg(frame, sty, fit, date_col), fit


if fund_ret is not None and len(fund_ret) >= 12 and "fwd_1m" in scored.columns:
    try:
        scored, rfit = cached_returns(scored, tuple(factor_cols), date_col, fund_ret, mode)
        scored = composite(scored, [OUT_COL, "hold_prop_returns"])
        st.markdown(f"**Returns leg ({rfit['mode']}, {rfit['n_months']} months, R² {rfit['r2']:.2f}):** "
                    f"{rfit['text']}")
        legs = {"LLM leg": OUT_COL, "Returns leg": "hold_prop_returns", "Composite": "hold_prop_ensemble"}
        rows = []
        for lbl, c in legs.items():
            e = evaluate_weighted(scored, c, date_col)
            if len(e):
                rows.append({"": lbl, "AUC (membership)": e["auc"].mean(), "Weighted AUC": e["weighted_auc"].mean(),
                             "Fund $ in score's top 100": e["weight_in_top"].mean(), "Worst-quarter AUC": e["auc"].min()})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("").style.format({
                "AUC (membership)": "{:.3f}", "Weighted AUC": "{:.3f}",
                "Fund $ in score's top 100": "{:.0%}", "Worst-quarter AUC": "{:.3f}"}), width="stretch")
            st.caption("Means over the dates that have holdings. *Weighted AUC* counts each held "
                       "name by its weight; *fund $ in top 100* is the share of the book the "
                       "score would have put in its 100 favourite names (random ≈ 100 / names "
                       "in coverage). For a diffuse book membership alone mostly says 'large "
                       "cap' — judge it on where the money sits. The returns leg is fit on "
                       "every month available, including the ones scored here.")
        sc_ = scored[["hold_prop_llm", "hold_prop_returns"]].corr(method="spearman").iloc[0, 1]
        st.caption(f"The two legs' scores are {sc_:+.2f} rank-correlated — the further from 1, "
                   f"the more the composite can add.")
    except SpecError as e:
        st.warning(str(e))
elif fund_ret is not None:
    st.info(f"Only {len(fund_ret)} months of fund returns — the returns leg needs 12 or more.")

HANDOFF_COL = "hold_prop_ensemble" if "hold_prop_ensemble" in scored.columns else OUT_COL
d1, d2, d3, d4 = st.columns(4)
d1.download_button("Philosophy (txt)", st.session_state.get("hp_phil", ""),
                   "philosophy.txt", "text/plain", width="stretch")
d2.download_button("Scoring spec (json)", json.dumps(spec, indent=2),
                   "scoring_spec.json", "application/json", width="stretch")
d3.download_button("Scored panel (csv)", scored.to_csv(index=False),
                   "hold_prop.csv", "text/csv", width="stretch",
                   help="The input panel unchanged, plus every hold_prop_* column computed.")
if d4.button("Send to Edge Concierge", width="stretch",
             help=f"`{HANDOFF_COL}` becomes a feature there, joined on date and ID."):
    st.session_state["hold_prop_llm_panel"] = scored[[date_col, id_col, HANDOFF_COL]].rename(
        columns={date_col: "date", id_col: "secid", HANDOFF_COL: OUT_COL})
    st.toast(f"{HANDOFF_COL} is available in the Concierge and Destination & Path as hold_prop_llm.")

with st.expander("How this works", expanded=False):
    st.markdown("""
**Stage A — one LLM pass.** The model sees the latest cross section that has
holdings: every held name with fund, benchmark, active and EWM weights and its
factor values, beside benchmark names the fund passed on. It infers the
philosophy connecting them — it may reason about moats, growth persistence,
unit economics, anything — and writes it in English. Then it compiles that
into a **scoring spec**: a JSON rule in a small language (percentiles,
z-scores, thresholds, piecewise maps, weighted sums, interactions, min / max)
over the approved factor columns only.

**Validation.** Every factor must exist and be approved, every op must be in
the language, parameters must be sane, and no holdings column may appear.
An invalid spec fails loudly; nothing is scored.

**Stage B — propagation.** Pure pandas. The spec is applied independently
inside every date cross section and mapped to [0, 1]. Same panel + same spec
= same `hold_prop_llm`, every time, with no LLM. Holdings columns are never
read. So a single current snapshot is all this leg needs; where historical
holdings exist they serve only as a check.

**Reading the number.** "How much does this name, at this date, resemble the
kind of company this process appears to prefer *today*?" — a static-philosophy
approximation, one leg of an eventual ensemble with `hold_prop_ewm`,
`hold_prop_ml` and `hold_prop_returns`.
""")
