"""
Neural Edge — a Gu–Kelly–Xiu-style return-forecasting net.

Gu, Kelly & Xiu (2020, RFS, "Empirical Asset Pricing via Machine Learning")
find that shallow feed-forward nets beat everything else they try, with the
three-hidden-layer "NN3" their best performer. This module reproduces that
recipe at this repo's scale:

  * Inputs are per-date cross-sectional ranks mapped to [-1, 1], missing set
    to 0 (the cross-sectional median) — their characteristic transform.
  * Architectures follow their geometric pyramid: NN1 (32) … NN5 (32,16,8,4,2),
    ReLU throughout, NN3 (32,16,8) as the default.
  * An ensemble over random seeds, averaged — their variance-reduction trick.
  * Expanding-window walk-forward with from-scratch refits, exactly the
    protocol `learned.py` uses, so every score is out-of-sample.

Differences worth stating rather than hiding: sklearn's MLPRegressor gives L2
(not L1) penalty, and its early stopping holds out a random (not chronological)
validation slice of the *training window only* — the OOS protocol is untouched.
Batch normalisation is dropped; at 24 features and three layers it is not
load-bearing.

The second half of the module is a small pure-numpy view into the fitted nets:
forward passes that expose hidden-layer activations, and exact gradients of the
predicted return with respect to any hidden layer. Those two pieces are what
TCAV (concepts.py) needs, and hand-rolling them over sklearn's weight matrices
is what lets the app avoid a torch dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from engine import newey_west_tstat

# GKX's pyramid. NN3 is their headline model.
ARCHS = {"NN1": (32,), "NN2": (32, 16), "NN3": (32, 16, 8),
         "NN4": (32, 16, 8, 4), "NN5": (32, 16, 8, 4, 2)}

PRED_COL = "nn_pred"
INIT_TRAIN_YEARS = 8
STEP_YEARS = 2
# 5 seeds and alpha=1e-2, from a walk-forward sweep on this panel (2026-09):
# ensemble size mattered more than expected (5 seeds lifted both eras over 3),
# and the stronger L2 won both halves of the sample at 5 seeds. Capacity was
# swept 2 neurons -> 128·64·32: monotone up to NN3, flat-to-worse beyond.
N_SEEDS = 5
DEFAULT_ALPHA = 1e-2

NET_KWARGS = dict(
    activation="relu", solver="adam", learning_rate_init=1e-3,
    batch_size=1024, max_iter=100, shuffle=True,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=5,
    tol=1e-5)


def rank_signed(df: pd.DataFrame, features: list, date_col: str = "date") -> np.ndarray:
    """
    GKX characteristic transform: per-date cross-sectional percentile rank
    mapped to [-1, 1]; missing → 0 (the median, hence inert).

    Ranks are within-date only, so computing this once over the whole panel
    leaks nothing across time.
    """
    g = df.groupby(date_col)
    X = np.empty((len(df), len(features)), dtype=np.float64)
    for j, f in enumerate(features):
        r = g[f].rank(pct=True)          # NaN rank stays NaN
        X[:, j] = (2.0 * r - 1.0).fillna(0.0).to_numpy()
    return X


def make_net(arch: str = "NN3", seed: int = 0, alpha: float = DEFAULT_ALPHA) -> MLPRegressor:
    """A fresh, unfitted net. Called once per (fold, seed) — never reused."""
    return MLPRegressor(hidden_layer_sizes=ARCHS[arch], alpha=alpha,
                        random_state=seed, **NET_KWARGS)


def _fit_ensemble(X: np.ndarray, y: np.ndarray, arch: str, n_seeds: int,
                  alpha: float) -> list:
    import warnings
    nets = []
    for s in range(n_seeds):
        net = make_net(arch, seed=s, alpha=alpha)
        with warnings.catch_warnings():
            # hitting max_iter is a budget, not a bug
            warnings.simplefilter("ignore", ConvergenceWarning)
            net.fit(X, y)
        nets.append(net)
    return nets


def _predict_ensemble(nets: list, X: np.ndarray) -> np.ndarray:
    return np.mean([n.predict(X) for n in nets], axis=0)


def _monthly_ic(dates: pd.Series, pred: np.ndarray, y: np.ndarray,
                min_n: int = 30) -> pd.Series:
    """Spearman IC of prediction vs realised return, one value per date."""
    d = pd.DataFrame({"date": dates.to_numpy(), "p": pred, "y": y})
    out = {}
    for dt, g in d.groupby("date"):
        if len(g) >= min_n:
            out[dt] = g["p"].rank().corr(g["y"].rank())
    return pd.Series(out).sort_index()


def _r2_oos(y: np.ndarray, pred: np.ndarray) -> float:
    """GKX's out-of-sample R²: benchmarked against a zero forecast, not the
    mean — the historical mean is itself a noisy in-sample estimate."""
    denom = float(np.sum(y ** 2))
    return 1.0 - float(np.sum((y - pred) ** 2)) / denom if denom > 0 else np.nan


@dataclass
class NetFold:
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    r2_oos: float
    mean_ic: float


@dataclass
class NeuralResult:
    panel: pd.DataFrame                 # input rows + out-of-sample PRED_COL
    folds: pd.DataFrame
    pooled_r2: float                    # every OOS prediction at once
    monthly_ic: pd.Series               # pooled OOS IC per month
    ic_mean: float
    ic_t: float                         # Newey-West, lag = horizon − 1
    arch: str = "NN3"
    n_seeds: int = N_SEEDS
    features: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def _fold_windows(dates: pd.Series, init_train_years: int, step_years: int):
    start, end = dates.min(), dates.max()
    train_end = start + pd.DateOffset(years=init_train_years)
    while train_end <= end:
        yield train_end, train_end + pd.DateOffset(years=step_years)
        train_end = train_end + pd.DateOffset(years=step_years)


def walk_forward_net(df: pd.DataFrame, features: list, target: str,
                     date_col: str = "date", arch: str = "NN3",
                     n_seeds: int = N_SEEDS, alpha: float = DEFAULT_ALPHA,
                     init_train_years: int = INIT_TRAIN_YEARS,
                     step_years: int = STEP_YEARS,
                     nw_lag: int = 0, progress=None,
                     embargo_months: int = 0) -> NeuralResult:
    """
    Expanding-window, out-of-sample refit — the `learned.py` protocol with a
    net ensemble in place of the forest.

    For each fold: fit `n_seeds` nets on everything strictly before
    `train_end`, average their predictions over the following `step_years`
    window, then absorb that window and repeat. Every prediction comes from an
    ensemble that never saw the row.

    `embargo_months` purges training rows dated within that many months of
    `train_end`. With a 12-month forward target, a row dated the month before
    the cut has a label realised almost entirely inside the test window and
    near-identical rank features to the test rows that follow it — without a
    12-month purge, part of what any high-capacity model "predicts" is that
    overlap. Set it to the target horizon for an honest score.
    """
    d = df.sort_values(date_col).reset_index(drop=True).copy()
    d[date_col] = pd.to_datetime(d[date_col])
    X = rank_signed(d, features, date_col)
    y_all = pd.to_numeric(d[target], errors="coerce").to_numpy(dtype=np.float64)
    has_y = np.isfinite(y_all)

    out = np.full(len(d), np.nan)
    rows, skipped = [], []
    windows = list(_fold_windows(d[date_col], init_train_years, step_years))
    for i, (train_end, test_end) in enumerate(windows):
        if progress:
            progress(i / max(len(windows), 1),
                     f"fold to {train_end:%Y} ({n_seeds} net(s))")
        cut = train_end - pd.DateOffset(months=embargo_months)
        tr = (d[date_col] < cut).to_numpy() & has_y
        te = ((d[date_col] >= train_end) & (d[date_col] < test_end)).to_numpy()
        if not te.any():
            skipped.append(f"{train_end:%Y}: no rows in the test window")
            continue
        if tr.sum() < 1000:
            skipped.append(f"{train_end:%Y}: only {tr.sum()} training rows")
            continue

        nets = _fit_ensemble(X[tr], y_all[tr], arch, n_seeds, alpha)
        out[te] = _predict_ensemble(nets, X[te])

        te_y = te & has_y
        ic = _monthly_ic(d.loc[te_y, date_col], out[te_y], y_all[te_y])
        rows.append(NetFold(train_end, test_end, int(tr.sum()), int(te.sum()),
                            _r2_oos(y_all[te_y], out[te_y]), float(ic.mean())))

    d[PRED_COL] = out
    scored = np.isfinite(out) & has_y
    ic_all = (_monthly_ic(d.loc[scored, date_col], out[scored], y_all[scored])
              if scored.any() else pd.Series(dtype=float))
    ic_arr = ic_all.dropna().to_numpy()
    return NeuralResult(
        panel=d,
        folds=pd.DataFrame([f.__dict__ for f in rows]),
        pooled_r2=_r2_oos(y_all[scored], out[scored]) if scored.any() else np.nan,
        monthly_ic=ic_all,
        ic_mean=float(ic_arr.mean()) if len(ic_arr) else np.nan,
        ic_t=newey_west_tstat(ic_arr, nw_lag) if len(ic_arr) > 2 else np.nan,
        arch=arch, n_seeds=n_seeds, features=list(features), skipped=skipped)


# ── The descriptive fit TCAV interrogates ────────────────────────────────────
# Same convention as learned.py: the walk-forward exists to produce honest
# scores and is a poor lens on what the model learned, because each fold is a
# different model. For interpretation, fit one ensemble on all history. That
# fit is explicitly never scored from — it has seen every row.

def fit_full_history(df: pd.DataFrame, features: list, target: str,
                     date_col: str = "date", arch: str = "NN3",
                     n_seeds: int = N_SEEDS, alpha: float = DEFAULT_ALPHA) -> dict:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    X = rank_signed(d, features, date_col)
    y = pd.to_numeric(d[target], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(y)
    nets = _fit_ensemble(X[ok], y[ok], arch, n_seeds, alpha)
    return {"nets": nets, "X": X, "mask": ok, "features": list(features),
            "arch": arch, "dates": d[date_col].reset_index(drop=True)}


# ── Pure-numpy internals for TCAV ────────────────────────────────────────────
# An MLPRegressor with ReLU hidden layers is just matrix products with masks.
# coefs_[0] maps input → h1, coefs_[k] maps h_k → h_{k+1}, coefs_[-1] → output.

def _forward(net: MLPRegressor, X: np.ndarray):
    """Post-ReLU activations and active-unit masks for every hidden layer."""
    acts, masks = [], []
    a = X
    n_hidden = len(net.coefs_) - 1
    for k in range(n_hidden):
        z = a @ net.coefs_[k] + net.intercepts_[k]
        m = z > 0
        a = np.where(m, z, 0.0)
        acts.append(a)
        masks.append(m)
    return acts, masks


def n_hidden_layers(net: MLPRegressor) -> int:
    return len(net.coefs_) - 1


def hidden_activations(net: MLPRegressor, X: np.ndarray, layer: int) -> np.ndarray:
    """Activations at hidden layer `layer` (1-indexed from the input side).
    Layer 0 is the input itself — a CAV there lives in feature-rank space."""
    if layer == 0:
        return X
    acts, _ = _forward(net, X)
    return acts[layer - 1]


def grad_wrt_layer(net: MLPRegressor, X: np.ndarray, layer: int) -> np.ndarray:
    """
    Exact ∂(predicted return)/∂(activations at hidden layer `layer`), per row.

    Backprop by hand: start from the output weights, and walk back through
    each intermediate layer masking by which ReLUs were active for that row.
    Returns an array shaped (rows, units at `layer`); layer 0 gives the
    gradient with respect to the inputs.

    Note the degenerate top: at the LAST hidden layer only the linear readout
    sits above, so the gradient is the same vector for every row and any
    directional-derivative sign test collapses to all-0 or all-1. TCAV should
    probe a layer with at least one nonlinearity above it — 0..H-1.
    """
    acts, masks = _forward(net, X)
    H = len(acts)
    if not 0 <= layer <= H:
        raise ValueError(f"layer must be in 0..{H}")
    g = np.tile(net.coefs_[H].ravel(), (len(X), 1))     # ∂ŷ/∂a_H
    for k in range(H - 1, layer - 1, -1):               # a_{k+1} → a_k
        g = (masks[k] * g) @ net.coefs_[k].T
    return g


def sanity_check_grad(net: MLPRegressor, X: np.ndarray, layer: int,
                      eps: float = 1e-5) -> float:
    """Max abs error of the analytic gradient vs finite differences on the
    first row. Used by the smoke test, not the app."""
    acts, masks = _forward(net, X[:1])
    a0 = acts[layer - 1].copy()

    def out_from(a):
        v = a
        for k in range(layer, len(net.coefs_) - 1):
            z = v @ net.coefs_[k] + net.intercepts_[k]
            v = np.maximum(z, 0.0)
        return (v @ net.coefs_[-1] + net.intercepts_[-1]).item()

    g = grad_wrt_layer(net, X[:1], layer)[0]
    err = 0.0
    for j in range(a0.shape[1]):
        ap, am = a0.copy(), a0.copy()
        ap[0, j] += eps
        am[0, j] -= eps
        num = (out_from(ap) - out_from(am)) / (2 * eps)
        err = max(err, abs(num - g[j]))
    return err
