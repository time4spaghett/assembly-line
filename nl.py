"""
Optional natural-language edge builder.

Turns a plain-English description ("cheap quality names with good momentum,
ex-financials") into an edge spec: feature/transform/weight rows plus universe
filters. Active only when an Anthropic API key is available.
"""
from __future__ import annotations

import os
from typing import List, Optional

import anthropic
from pydantic import BaseModel


class SpecRow(BaseModel):
    feature: str
    transform: str          # rank | zscore | raw
    weight: float


class PlanConstraint(BaseModel):
    left: str               # feature name
    op: str                 # > >= < <= == !=
    right: str              # a number as text, or another feature name


class EdgePlan(BaseModel):
    rows: List[SpecRow]
    constraints: List[PlanConstraint]
    sectors: List[str]      # empty = all sectors
    sector_neutral: bool
    horizon: str            # fwd_1m | fwd_3m | fwd_6m | fwd_12m
    rationale: str          # one sentence explaining the mapping


def api_key() -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:  # Streamlit secrets, if configured
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def plan_edge(description: str, features: list[str], sectors: list[str],
              key: str) -> EdgePlan:
    client = anthropic.Anthropic(api_key=key)
    response = client.messages.parse(
        # Opus: the rationale it writes is the point of this box, not a
        # by-product, and the reasoning quality shows in it. The wait is made
        # legible in the UI rather than traded away here. Effort is medium —
        # raise to "high" (or drop the line) if the mapping ever looks thin.
        model="claude-opus-5",
        max_tokens=2000,
        output_config={"effort": "medium"},
        system=(
            "You translate a plain-English factor/edge idea into a spec for a "
            "quant backtester. Available features (raw values, higher = more of "
            f"the named thing): {', '.join(features)}. "
            "Note: 'leverage', 'accruals' and 'asset_gr' are typically *bad* — "
            "use negative weights for them when the user wants safety/quality. "
            "'log_mcap' with negative weight = small-cap tilt. "
            f"Available sectors: {', '.join(sectors)}. "
            "Use transform 'rank' unless the user asks for z-scores or raw values. "
            "'sectors' is the list to INCLUDE: if the user excludes sectors, "
            "list all remaining ones; leave empty for no restriction. "
            "Default horizon fwd_1m unless the user implies a longer holding period. "
            "'constraints' are hard screens on RAW feature values, dropping names "
            "that fail: use them when the user states a threshold or a relationship "
            "('only profitable names' -> roa > 0; 'earning more than they are "
            "growing assets' -> roa > asset_gr). 'right' is either a number as a "
            "string or another feature name. Prefer weights over constraints for "
            "soft preferences — a constraint discards names entirely, so reach for "
            "one only when the user means a genuine requirement. Empty list if none."
        ),
        messages=[{"role": "user", "content": description}],
        output_format=EdgePlan,
    )
    return response.parsed_output


class ValidationNote(BaseModel):
    note: str


def explain_validation(checks: dict, spec_formula: str, key: str) -> str:
    """
    Narrate a validation result that has already been decided.

    The verdict, the thresholds and the pass/fail of every check are computed
    deterministically in engine.consistency_checks. This only puts them into
    English and says what to do next. It is given the outcome, not the freedom
    to reach one — otherwise the same numbers could read as a pass on Monday
    and a fail on Tuesday, and the check would be worthless.
    """
    client = anthropic.Anthropic(api_key=key)
    lines = chr(10).join(f"- {n}: {'PASS' if ok else 'FLAG'} ({d})"
                         for n, ok, d in checks["checks"])
    flagged = checks["total"] - checks["passed"]
    response = client.messages.parse(
        model="claude-opus-5",
        max_tokens=300,          # a hard ceiling on the urge to elaborate
        output_config={"effort": "low"},
        system=(
            "You explain a validation result that has already been decided. "
            "Two sentences, three at the very most. "
            "Say what the checks show and what it means for how far to trust "
            "the numbers. Nothing else. "
            "Do not speculate about causes (regimes, cycles, crowding). Do not "
            "propose further analysis, next steps, or things to check. Do not "
            "raise questions. Do not re-judge the verdict or add thresholds of "
            "your own. No preamble, no headings, no congratulation. Plain "
            "declarative sentences about what is in front of you."
        ),
        messages=[{"role": "user", "content": chr(10).join([
            f"Edge: {spec_formula}",
            f"Verdict: {checks['verdict']} — {checks['headline']}",
            f"{flagged} of {checks['total']} checks flagged.",
            f"In-sample {checks['months_is']} months, holdout "
            f"{checks['months_oos']} months.",
            "Checks:", lines])}],
        output_format=ValidationNote,
    )
    return response.parsed_output.note
