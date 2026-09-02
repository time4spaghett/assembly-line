"""
Shared UI pieces, so both tools screen a universe the same way.

The universe block was written for the Edge Concierge; the Learned Edge needs
the identical thing (sector or industry cut, date range, conditions on raw
values) and duplicating it would guarantee the two drift apart. It lives here
instead, parameterised by a key prefix so two pages can hold independent state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import OPS, Constraint, constraint_mask, unresolved_constraints

# ── Palette (dataviz reference instance, light mode) ─────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#c4302b"
ORDINAL_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def ramp(n: int) -> list:
    idx = np.linspace(0, len(ORDINAL_RAMP) - 1, max(n, 1)).round().astype(int)
    return [ORDINAL_RAMP[i] for i in idx]


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(family='system-ui, "Segoe UI", sans-serif', color=INK_2, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, zerolinecolor=BASELINE,
                     tickfont=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, zerolinecolor=BASELINE,
                     tickfont=dict(color=MUTED), automargin=True)
    return fig


def live_constraints(key: str) -> list:
    """Constraint rows including edits not yet committed by the data editor."""
    base = st.session_state.get(f"{key}_cons")
    if base is None or base.empty:
        base = pd.DataFrame(columns=["left", "op", "right"])
    base = base.copy().reset_index(drop=True)
    state = st.session_state.get(f"{key}_cons_editor") or {}
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


def universe_block(panel: pd.DataFrame, key: str, features: list,
                   sector_col: str = "sector", industry_col: str = "industry",
                   date_col: str = "date",
                   default_sectors=None, default_constraints=None) -> dict:
    """
    Render the universe screen and return what survives it.

    Industries nest inside sectors, so the granularity is a choice rather than
    two filters that can contradict each other. Conditions are applied on raw
    values before anything downstream ranks or fits.
    """
    has_sector = sector_col in panel.columns
    sectors = sorted(panel[sector_col].dropna().unique()) if has_sector else []

    if f"{key}_cons" not in st.session_state:
        st.session_state[f"{key}_cons"] = pd.DataFrame(
            default_constraints or [], columns=["left", "op", "right"])
    if has_sector and f"{key}_sectors" not in st.session_state:
        st.session_state[f"{key}_sectors"] = [s for s in (default_sectors or [])
                                              if s in sectors]
        st.session_state[f"{key}_grain"] = "Sector" if default_sectors else "Everything"
    if has_sector:
        st.session_state[f"{key}_sectors"] = [
            s for s in st.session_state[f"{key}_sectors"] if s in sectors]

    sel_sectors, sel_industries = [], []
    if has_sector:
        grain = st.radio("Screen names by", ["Everything", "Sector", "Industry"],
                         horizontal=True, key=f"{key}_grain",
                         help="Industries sit inside sectors — pick the level to "
                              "cut at rather than filtering on both.")
        if grain == "Sector":
            sel_sectors = st.multiselect("Sectors to include", sectors,
                                         placeholder="All sectors",
                                         key=f"{key}_sectors")
        elif grain == "Industry" and industry_col in panel.columns:
            sel_industries = st.multiselect(
                "Industries to include",
                sorted(panel[industry_col].dropna().unique()),
                placeholder="All industries", key=f"{key}_industries")

    yr_min = int(pd.to_datetime(panel[date_col]).dt.year.min())
    yr_max = int(pd.to_datetime(panel[date_col]).dt.year.max())
    yr_range = (yr_min, yr_max)
    if yr_max > yr_min:
        yr_range = st.columns([2, 3])[0].slider(
            "Date range", yr_min, yr_max, (yr_min, yr_max), key=f"{key}_years")

    st.markdown("Conditions", help="On **raw** feature values: against a number "
                                   "(`roa > 0`) or another feature "
                                   "(`roa > asset_gr`). Rows that fail — or are "
                                   "missing either side — are dropped.")
    st.columns([5, 1])[0].data_editor(
        st.session_state[f"{key}_cons"], num_rows="dynamic", width="stretch",
        hide_index=True, key=f"{key}_cons_editor",
        column_config={
            "left": st.column_config.SelectboxColumn("Feature", options=features,
                                                     width="medium"),
            "op": st.column_config.SelectboxColumn("Is", options=list(OPS),
                                                   width="small"),
            "right": st.column_config.TextColumn(
                "Than", width="medium",
                help="A number (0, -0.1) or another feature name."),
        })
    cons = live_constraints(key)

    mask = pd.to_datetime(panel[date_col]).dt.year.between(*yr_range)
    if sel_sectors:
        mask &= panel[sector_col].isin(sel_sectors)
    if sel_industries:
        mask &= panel[industry_col].isin(sel_industries)
    out = panel.loc[mask]

    if cons:
        bad = unresolved_constraints(out, [Constraint(*c) for c in cons])
        for c, why in bad:
            st.warning(f"Constraint `{c.left} {c.op} {c.right}` ignored — {why}.")
        cons = [c for c in cons
                if (c[0], c[1], c[2]) not in {(b.left, b.op, b.right) for b, _ in bad}]
        if cons:
            out = out[constraint_mask(out, [Constraint(*c) for c in cons])]

    out = out.reset_index(drop=True)
    bits = []
    if sel_sectors:
        bits.append(f"{len(sel_sectors)} sector(s)")
    if sel_industries:
        bits.append(f"{len(sel_industries)} industries")
    if yr_range != (yr_min, yr_max):
        bits.append(f"{yr_range[0]}–{yr_range[1]}")
    if cons:
        bits.append(f"{len(cons)} condition(s)")
    st.caption(f"After screening: **{len(out):,}** rows"
               + (f" — {', '.join(bits)}." if bits else " — no screen applied."))
    return {"panel": out, "sectors": sel_sectors, "industries": sel_industries,
            "years": yr_range, "constraints": cons}
