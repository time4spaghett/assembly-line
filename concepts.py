"""
Concept alignment — TCAV over the Neural Edge net.

Kim et al. (2018, "Interpretability Beyond Feature Attribution") test whether a
*concept* — defined only by examples — has a directional influence on a model's
output. The recipe, adapted here from image classifiers to a return forecaster:

  1. A concept is a set of panel rows (stock-months): "high profitability" is
     the top quintile of `gpa` each month; "Manager X's philosophy" is the
     rows they actually held. Negatives are random rows sampled from the same
     dates, so the CAV cannot learn "the concept is the 2010s".
  2. At a chosen hidden layer, fit a linear classifier separating concept
     activations from negative activations. The unit normal of its boundary
     is the Concept Activation Vector — the direction "toward the concept"
     in the net's own representation space.
  3. For every evaluation row, take the exact gradient of the predicted
     return with respect to that layer and dot it with the CAV. The TCAV
     score is the fraction of rows where the derivative is positive: how
     often nudging a stock toward the concept raises its predicted return.
     0.5 means orthogonal; 1.0 means the forecast is uniformly aligned.

Two honesty guards, both from the paper. The CAV must actually separate the
classes — a held-out accuracy near 0.5 means the concept is not encoded at
that layer and the score above it is noise. And the whole procedure is
repeated against *random* pseudo-concepts of the same size; a two-sample
t-test against that null is what separates alignment from wishful reading.
Everything runs across the ensemble members and several negative resamples,
so the score arrives as a distribution, not a point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from neural import grad_wrt_layer, hidden_activations

N_RUNS = 8            # negative resamples per ensemble member
MIN_CONCEPT_ROWS = 40


# ── Concept definition ───────────────────────────────────────────────────────

@dataclass
class QuantileRule:
    """`top`/`bottom` `pct` of `feature` within each date. Rules AND together."""
    feature: str
    side: str          # "top" | "bottom"
    pct: float         # e.g. 0.2 for a quintile


PRESETS = {
    "High profitability": [QuantileRule("gpa", "top", 0.2),
                           QuantileRule("roa", "top", 0.4)],
    "Deep value": [QuantileRule("btm", "top", 0.2),
                   QuantileRule("earn_yield", "top", 0.4)],
    "Momentum winners": [QuantileRule("mom_12_1", "top", 0.2)],
    "Conservative balance sheet": [QuantileRule("leverage", "bottom", 0.3),
                                   QuantileRule("accruals", "bottom", 0.5)],
    "Aggressive expansion": [QuantileRule("asset_gr", "top", 0.2),
                             QuantileRule("rev_gr_1y", "top", 0.4)],
    "Large-cap quality": [QuantileRule("log_mcap", "top", 0.3),
                          QuantileRule("gpa", "top", 0.5)],
}


def rule_mask(panel: pd.DataFrame, rules: list, date_col: str = "date") -> pd.Series:
    """Boolean mask of rows satisfying every rule, quantiles within each date."""
    mask = pd.Series(True, index=panel.index)
    g = panel.groupby(date_col)
    for r in rules:
        pr = g[r.feature].rank(pct=True)
        cut = float(r.pct)
        this = pr >= 1.0 - cut if r.side == "top" else pr <= cut
        mask &= this.fillna(False)
    return mask


def match_uploaded(panel: pd.DataFrame, uploaded: pd.DataFrame,
                   date_col: str, id_col: str,
                   panel_date: str = "date", panel_id: str = "ticker") -> pd.Series:
    """
    Mask of panel rows named by an uploaded (date, security) list.

    Matching is on (id, month): holdings files carry mid-month report dates
    that would never hit the panel's month-end stamps exactly.
    """
    up = uploaded[[date_col, id_col]].copy()
    up["_m"] = pd.to_datetime(up[date_col], errors="coerce").dt.to_period("M")
    up["_id"] = up[id_col].astype(str).str.strip().str.upper()
    keys = set(zip(up["_id"], up["_m"].astype(str)))
    pm = pd.to_datetime(panel[panel_date]).dt.to_period("M").astype(str)
    pid = panel[panel_id].astype(str).str.strip().str.upper()
    return pd.Series(list(zip(pid, pm)), index=panel.index).isin(keys)


# ── CAV + TCAV ───────────────────────────────────────────────────────────────

def _date_matched_sample(dates: pd.Series, pos_idx: np.ndarray, k: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Sample k non-concept rows with the same date distribution as the
    concept rows, so the CAV cannot separate the classes on era alone."""
    pos_set = set(pos_idx.tolist())
    pos_dates = dates.iloc[pos_idx]
    by_date = {}
    for dt, g in dates.groupby(dates):
        cand = np.array([i for i in g.index if i not in pos_set])
        if len(cand):
            by_date[dt] = cand
    want = pos_dates.value_counts()
    picks = []
    for dt, n in want.items():
        cand = by_date.get(dt)
        if cand is None:
            continue
        picks.append(rng.choice(cand, size=min(n, len(cand)), replace=False))
    out = np.concatenate(picks) if picks else np.array([], dtype=int)
    if len(out) > k:
        out = rng.choice(out, size=k, replace=False)
    return out


def fit_cav(acts_pos: np.ndarray, acts_neg: np.ndarray,
            seed: int) -> tuple[np.ndarray, float]:
    """Linear boundary between concept and negative activations. Returns the
    unit normal pointing toward the concept, and held-out accuracy."""
    Xc = np.vstack([acts_pos, acts_neg])
    yc = np.r_[np.ones(len(acts_pos)), np.zeros(len(acts_neg))]
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.33,
                                          random_state=seed, stratify=yc)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    v = clf.coef_.ravel().astype(np.float64)
    norm = np.linalg.norm(v) or 1.0
    return v / norm, acc


@dataclass
class ConceptReport:
    scores: np.ndarray            # one TCAV score per (net, resample)
    cav_accs: np.ndarray          # held-out CAV accuracy, same shape
    random_scores: np.ndarray     # TCAV scores of size-matched random concepts
    p_value: float                # two-sample t-test, concept vs random
    by_year: pd.Series            # mean alignment per year
    n_pos: int
    n_eval: int
    layer: int
    # Per-row views, aligned with the fit panel's row order:
    expression: np.ndarray = None  # how strongly each row expresses the
                                   # concept: pooled pct-rank of its projection
                                   # onto the CAV, averaged over (net, resample)
    row_align: np.ndarray = None   # share of CAVs for which nudging THIS row
                                   # toward the concept raises its forecast
    notes: list = field(default_factory=list)


def run_tcav(fit: dict, pos_mask: pd.Series, layer: int,
             n_runs: int = N_RUNS, seed: int = 0,
             progress=None) -> ConceptReport:
    """
    The full flow over an ensemble: for every net and every negative resample,
    fit a CAV and score the directional derivative over every panel row; run
    the identical procedure on random pseudo-concepts for the null.

    Beyond the aggregate score, two per-row series come back. `expression`
    (projection onto the CAV, pct-ranked) says how strongly a given stock-month
    looks like the concept to the net; `row_align` says whether that particular
    stock's forecast rises when nudged toward the concept. Both are averaged
    across every CAV fit, so a row's value is stable to the negative draw.

    `fit` is the dict from neural.fit_full_history. `pos_mask` is a boolean
    Series over the same panel rows.
    """
    rng = np.random.default_rng(seed)
    nets, X = fit["nets"], fit["X"]
    dates = fit["dates"]
    pos_idx = np.flatnonzero(pos_mask.to_numpy())
    if len(pos_idx) < MIN_CONCEPT_ROWS:
        raise ValueError(f"Concept has {len(pos_idx)} rows — needs at least "
                         f"{MIN_CONCEPT_ROWS} to fit a CAV worth reading.")
    n_neg = len(pos_idx)
    n = len(X)

    scores, accs, rand_scores = [], [], []
    align_frac = np.zeros(n)       # running mean of 1[S>0] per row
    expr_pct = np.zeros(n)         # running mean of pct-rank(a · cav) per row
    n_cavs = 0
    total = len(nets) * n_runs
    for ni, net in enumerate(nets):
        acts = hidden_activations(net, X, layer)       # once per net
        grads = grad_wrt_layer(net, X, layer)
        acts_pos = acts[pos_idx]
        for r in range(n_runs):
            if progress:
                progress((ni * n_runs + r) / total, f"net {ni + 1}, resample {r + 1}")
            neg_idx = _date_matched_sample(dates, pos_idx, n_neg, rng)
            cav, acc = fit_cav(acts_pos, acts[neg_idx], seed=r)
            s = grads @ cav
            scores.append(float(np.mean(s > 0)))
            accs.append(acc)
            align_frac += (s > 0).astype(float)
            proj = acts @ cav
            expr_pct += pd.Series(proj).rank(pct=True).to_numpy()
            n_cavs += 1

            # the null: a "concept" of the same size with no meaning
            fake_pos = rng.choice(n, size=n_neg, replace=False)
            fake_neg = _date_matched_sample(dates, fake_pos, n_neg, rng)
            fcav, _ = fit_cav(acts[fake_pos], acts[fake_neg], seed=r)
            rand_scores.append(float(np.mean(grads @ fcav > 0)))

    scores, accs = np.array(scores), np.array(accs)
    rand_scores = np.array(rand_scores)
    import warnings
    with warnings.catch_warnings():
        # a cleanly aligned concept scores near-identically across resamples;
        # scipy warns about the tiny variance, which here is the finding
        warnings.simplefilter("ignore", RuntimeWarning)
        p = float(ttest_ind(scores, rand_scores, equal_var=False).pvalue)

    align_frac /= max(n_cavs, 1)
    expr_pct /= max(n_cavs, 1)
    yrs = pd.to_datetime(dates).dt.year.to_numpy()
    by_year = (pd.Series(align_frac, index=yrs).groupby(level=0).mean()
               .rename("alignment"))

    notes = []
    if accs.mean() < 0.6:
        notes.append(
            f"Mean CAV accuracy is {accs.mean():.0%} — the concept is barely "
            f"separable at this layer, so the score is mostly noise.")
    return ConceptReport(scores=scores, cav_accs=accs, random_scores=rand_scores,
                         p_value=p, by_year=by_year, n_pos=len(pos_idx),
                         n_eval=n, layer=layer, expression=expr_pct,
                         row_align=align_frac, notes=notes)
