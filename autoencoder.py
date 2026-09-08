"""
Factor Edge — Gu–Kelly–Xiu's conditional autoencoder, in plain numpy.

Gu, Kelly & Xiu (2021, J. Econometrics, "Autoencoder Asset Pricing Models")
replace the unconstrained return forecast g(z) with a *factor structure*:

    r_{i,t} = beta(z_{i,t-1})' f_t + eps_{i,t}

The beta network maps a stock's characteristics to K factor loadings; the
factor network distills the month's K latent factor returns from
characteristic-managed portfolio returns x_t = Z'r / N (plus an equal-weight
market portfolio). Expected returns can only arise as compensation for factor
exposure — a near-no-arbitrage constraint that acts as a powerful regularizer.
That structure is why the CA tests so much better than unconstrained nets on
small panels: structure substitutes for data.

Why hand-rolled numpy rather than torch: the whole model is a 28->32->K MLP
and one linear layer. Autograd is the only thing torch would contribute, and
the backward pass for a bilinear-over-ReLU model is a page of matrix algebra
that `grad_check` verifies against finite differences (the same discipline the
smoke test applies to neural.py). In exchange the app keeps its dependency
footprint, and Huber loss and chronological early stopping — two things
sklearn's MLP could not give the Neural Edge — come for free.

Protocol matches the rest of the repo: expanding walk-forward, refit from
scratch per fold, a seed ensemble averaged, every score out-of-sample.
Within each fold the last 15% of training months are the early-stopping
validation set — chronological, not random.

Conventions:
  * Inputs are the GKX [-1, 1] per-month cross-sectional ranks (neural.rank_signed).
  * Prediction for an OOS month: beta(z)' lambda, with lambda the mean of the
    fitted factors over training months — the estimated risk premia.
  * Total R^2 uses the month's *realized* factors (observable ex post — the
    factors are portfolio returns); predictive R^2 uses lambda. Both are
    benchmarked against a zero forecast, GKX-style.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine import newey_west_tstat
from neural import _monthly_ic, rank_signed

BETA_ARCHS = {"CA0": (), "CA1": (32,), "CA2": (32, 16)}

PRED_COL = "ca_pred"
INIT_TRAIN_YEARS = 8
STEP_YEARS = 2
N_SEEDS = 3
N_FACTORS = 5

LR = 5e-3
MAX_EPOCHS = 200
PATIENCE = 10
VAL_FRAC = 0.15
MIN_MONTH_N = 30


# ── Data plumbing ────────────────────────────────────────────────────────────

def month_blocks(df: pd.DataFrame, features: list, target: str,
                 date_col: str = "date") -> list:
    """
    One block per month: (date, Z, r, x). Z are the signed ranks, r the target
    returns, and x the managed-portfolio returns [1'r/N ; Z'r/N] — the
    equal-weight market first, then one rank-weighted portfolio per feature.
    x is built within the month from its own cross-section, so nothing leaks
    across months.

    Deliberately does NOT re-sort: block rows follow the caller's frame in the
    caller's row order, so positional write-back stays aligned. (A re-sort here
    once scrambled tickers within months — pandas' default sort is unstable —
    which silently destroyed the IC while leaving fold-internal R² intact.)
    """
    d = df.reset_index(drop=True)
    Z_all = rank_signed(d, features, date_col)
    y = pd.to_numeric(d[target], errors="coerce").to_numpy(dtype=np.float64)
    blocks = []
    for dt, idx in d.groupby(date_col).indices.items():
        r = y[idx]
        ok = np.isfinite(r)
        if ok.sum() < MIN_MONTH_N:
            continue
        Z, r = Z_all[idx][ok], r[ok]
        x = np.concatenate(([r.mean()], Z.T @ r / len(r)))
        blocks.append((pd.Timestamp(dt), Z, r, x))
    blocks.sort(key=lambda b: b[0])       # chronological, for the val split
    return blocks


# ── The model: parameters, forward, backward ─────────────────────────────────

def init_params(P: int, K: int, hidden: tuple, rng: np.random.Generator) -> dict:
    """He-initialised beta net + linear factor net. Wf keys 'beta:' / 'fact:'."""
    params = {}
    sizes = (P,) + hidden + (K,)
    for i in range(len(sizes) - 1):
        params[f"beta:W{i}"] = (rng.standard_normal((sizes[i], sizes[i + 1]))
                                * np.sqrt(2.0 / sizes[i]))
        params[f"beta:b{i}"] = np.zeros(sizes[i + 1])
    params["fact:W"] = (rng.standard_normal((P + 1, K))
                        * np.sqrt(1.0 / (P + 1)))
    return params


def _n_beta_layers(params: dict) -> int:
    return sum(1 for k in params if k.startswith("beta:W"))


def betas_of(params: dict, Z: np.ndarray, keep_cache: bool = False):
    """Beta-network forward pass: (N, P) -> (N, K). ReLU hidden layers."""
    L = _n_beta_layers(params)
    a, cache = Z, []
    for i in range(L):
        z = a @ params[f"beta:W{i}"] + params[f"beta:b{i}"]
        if i < L - 1:
            m = z > 0
            a_next = np.where(m, z, 0.0)
            if keep_cache:
                cache.append((a, m))
            a = a_next
        else:
            if keep_cache:
                cache.append((a, None))
            a = z                     # linear output layer
    return (a, cache) if keep_cache else a


def factors_of(params: dict, x: np.ndarray) -> np.ndarray:
    return x @ params["fact:W"]


def month_loss_grad(params: dict, Z: np.ndarray, r: np.ndarray,
                    x: np.ndarray, huber: float | None = None):
    """
    Loss and exact gradients for one month.

    y_hat = B f  with B = beta(Z) (N, K) and f = Wf' x (K,).
    dL/dB = e f' ; dL/df = B' e ; both then propagate into their networks.
    `huber` (a threshold in return units) clips the residual's influence — the
    robustness device GKX use against the heavy tails of monthly returns.
    """
    B, cache = betas_of(params, Z, keep_cache=True)
    f = factors_of(params, x)
    yhat = B @ f
    e = yhat - r
    n = len(r)
    if huber:
        # gradient of the Huber loss: linear in the tails
        loss = float(np.mean(np.where(np.abs(e) <= huber, 0.5 * e ** 2,
                                      huber * (np.abs(e) - 0.5 * huber))))
        de = np.clip(e, -huber, huber) / n
    else:
        loss = float(np.mean(0.5 * e ** 2))
        de = e / n

    grads = {"fact:W": np.outer(x, B.T @ de)}
    g = np.outer(de, f)                       # dL/dB, (N, K)
    L = _n_beta_layers(params)
    for i in range(L - 1, -1, -1):
        a_in, _ = cache[i]
        grads[f"beta:W{i}"] = a_in.T @ g
        grads[f"beta:b{i}"] = g.sum(axis=0)
        if i > 0:                             # descend through the ReLU below
            g = g @ params[f"beta:W{i}"].T
            g = g * cache[i - 1][1]           # mask of the layer below's output
    return loss, grads


def _adam_step(params, grads, state, lr, alpha, t):
    for k, gr in grads.items():
        # L2 on weights only — shrinking biases would distort the level
        gr = gr + alpha * params[k] if ":W" in k else gr
        m = state.setdefault("m:" + k, np.zeros_like(gr))
        v = state.setdefault("v:" + k, np.zeros_like(gr))
        m[:] = 0.9 * m + 0.1 * gr
        v[:] = 0.999 * v + 0.001 * gr ** 2
        mh = m / (1 - 0.9 ** t)
        vh = v / (1 - 0.999 ** t)
        params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


def _val_loss(params, blocks) -> float:
    tot, n = 0.0, 0
    for _, Z, r, x in blocks:
        e = betas_of(params, Z) @ factors_of(params, x) - r
        tot += float(e @ e)
        n += len(r)
    return tot / max(n, 1)


def fit_ca(blocks: list, K: int, arch: str = "CA1", alpha: float = 1e-4,
           seed: int = 0, huber: float | None = None,
           lr: float = LR, max_epochs: int = MAX_EPOCHS) -> dict:
    """
    Fit one CA on a list of month blocks. The last VAL_FRAC of months (by
    date — the blocks arrive sorted) is the chronological early-stopping set:
    train on the rest, keep the parameters from the best validation epoch.
    """
    rng = np.random.default_rng(seed)
    P = blocks[0][1].shape[1]
    n_val = max(int(len(blocks) * VAL_FRAC), 2)
    train, val = blocks[:-n_val], blocks[-n_val:]

    params = init_params(P, K, BETA_ARCHS[arch], rng)
    state, t = {}, 0
    best, best_params, since = np.inf, None, 0
    order = np.arange(len(train))
    for _epoch in range(max_epochs):
        rng.shuffle(order)
        for j in order:
            _, Z, r, x = train[j]
            t += 1
            _, grads = month_loss_grad(params, Z, r, x, huber)
            _adam_step(params, grads, state, lr, alpha, t)
        vl = _val_loss(params, val)
        if vl < best - 1e-12:
            best, since = vl, 0
            best_params = {k: v.copy() for k, v in params.items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    return best_params if best_params is not None else params


# ── Walk-forward protocol ────────────────────────────────────────────────────

@dataclass
class CAFold:
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    n_train_months: int
    n_test: int
    total_r2: float
    pred_r2: float
    mean_ic: float


@dataclass
class CAResult:
    panel: pd.DataFrame                 # input rows + out-of-sample PRED_COL
    folds: pd.DataFrame
    total_r2: float                     # pooled OOS, realized factors
    pred_r2: float                      # pooled OOS, estimated premia
    monthly_ic: pd.Series
    ic_mean: float
    ic_t: float
    arch: str = "CA1"
    n_factors: int = N_FACTORS
    n_seeds: int = N_SEEDS
    features: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def _fold_windows(dates, init_train_years, step_years):
    start, end = min(dates), max(dates)
    train_end = start + pd.DateOffset(years=init_train_years)
    while train_end <= end:
        yield train_end, train_end + pd.DateOffset(years=step_years)
        train_end = train_end + pd.DateOffset(years=step_years)


def walk_forward_ca(df: pd.DataFrame, features: list, target: str,
                    date_col: str = "date", n_factors: int = N_FACTORS,
                    arch: str = "CA1", n_seeds: int = N_SEEDS,
                    alpha: float = 1e-4, huber: float | None = None,
                    init_train_years: int = INIT_TRAIN_YEARS,
                    step_years: int = STEP_YEARS,
                    nw_lag: int = 0, progress=None) -> CAResult:
    """
    Expanding-window walk-forward for the CA — the repo protocol. Per fold:
    fit `n_seeds` models on all months strictly before train_end, then for
    each test month predict beta(z)'lambda (lambda from training months only)
    and record the total-R^2 fit against the month's realized factors.
    """
    d = df.sort_values(date_col).reset_index(drop=True).copy()
    d[date_col] = pd.to_datetime(d[date_col])
    blocks = month_blocks(d, features, target, date_col)
    if not blocks:
        raise ValueError("No months with enough stocks to fit on.")
    dates = [b[0] for b in blocks]

    # positional index of each panel row inside its month block, for writeback
    d["_pred"] = np.nan
    row_index = {}
    y_all = pd.to_numeric(d[target], errors="coerce").to_numpy(dtype=np.float64)
    for dt, idx in d.groupby(date_col).indices.items():
        ok = np.isfinite(y_all[idx])
        row_index[pd.Timestamp(dt)] = np.asarray(idx)[ok]

    rows, skipped = [], []
    sse_tot = sse_pred = ss = 0.0
    windows = list(_fold_windows(dates, init_train_years, step_years))
    for i, (train_end, test_end) in enumerate(windows):
        if progress:
            progress(i / max(len(windows), 1),
                     f"fold to {train_end:%Y} ({n_seeds} model(s))")
        train = [b for b in blocks if b[0] < train_end]
        test = [b for b in blocks if train_end <= b[0] < test_end]
        if not test:
            skipped.append(f"{train_end:%Y}: no test months")
            continue
        if len(train) < 24:
            skipped.append(f"{train_end:%Y}: only {len(train)} training months")
            continue

        models = [fit_ca(train, n_factors, arch, alpha, seed=s, huber=huber)
                  for s in range(n_seeds)]
        lambdas = [np.mean([factors_of(m, b[3]) for b in train], axis=0)
                   for m in models]

        f_sse_t = f_sse_p = f_ss = 0.0
        n_test = 0
        for dt, Z, r, x in test:
            pred = np.mean([betas_of(m, Z) @ lam
                            for m, lam in zip(models, lambdas)], axis=0)
            fit = np.mean([betas_of(m, Z) @ factors_of(m, x)
                           for m in models], axis=0)
            d.loc[row_index[dt], "_pred"] = pred
            f_sse_p += float(np.sum((r - pred) ** 2))
            f_sse_t += float(np.sum((r - fit) ** 2))
            f_ss += float(np.sum(r ** 2))
            n_test += len(r)
        sse_pred += f_sse_p
        sse_tot += f_sse_t
        ss += f_ss

        te_mask = d["_pred"].notna() & (d[date_col] >= train_end) \
            & (d[date_col] < test_end)
        ic = _monthly_ic(d.loc[te_mask, date_col],
                         d.loc[te_mask, "_pred"].to_numpy(),
                         y_all[te_mask.to_numpy()])
        rows.append(CAFold(train_end, test_end, len(train), n_test,
                           1 - f_sse_t / f_ss if f_ss else np.nan,
                           1 - f_sse_p / f_ss if f_ss else np.nan,
                           float(ic.mean())))

    d = d.rename(columns={"_pred": PRED_COL})
    scored = d[PRED_COL].notna() & np.isfinite(y_all)
    ic_all = (_monthly_ic(d.loc[scored, date_col],
                          d.loc[scored, PRED_COL].to_numpy(),
                          y_all[scored.to_numpy()])
              if scored.any() else pd.Series(dtype=float))
    ic_arr = ic_all.dropna().to_numpy()
    return CAResult(
        panel=d,
        folds=pd.DataFrame([f.__dict__ for f in rows]),
        total_r2=1 - sse_tot / ss if ss else np.nan,
        pred_r2=1 - sse_pred / ss if ss else np.nan,
        monthly_ic=ic_all,
        ic_mean=float(ic_arr.mean()) if len(ic_arr) else np.nan,
        ic_t=newey_west_tstat(ic_arr, nw_lag) if len(ic_arr) > 2 else np.nan,
        arch=arch, n_factors=n_factors, n_seeds=n_seeds,
        features=list(features), skipped=skipped)


# ── Descriptive full-history fit, for interpretation ─────────────────────────

def full_history_ca(df: pd.DataFrame, features: list, target: str,
                    date_col: str = "date", n_factors: int = N_FACTORS,
                    arch: str = "CA1", n_seeds: int = N_SEEDS,
                    alpha: float = 1e-4, huber: float | None = None) -> dict:
    """Fit once on everything — never scored from. Returns the ensemble, the
    per-month factor realizations, and the estimated premia."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    blocks = month_blocks(d, features, target, date_col)
    models = [fit_ca(blocks, n_factors, arch, alpha, seed=s, huber=huber)
              for s in range(n_seeds)]
    F = pd.DataFrame(
        {b[0]: np.mean([factors_of(m, b[3]) for m in models], axis=0)
         for b in blocks}).T
    F.columns = [f"F{k + 1}" for k in range(n_factors)]
    return {"models": models, "factors": F, "premia": F.mean(),
            "features": list(features), "arch": arch}


def beta_profile(models: list, feature_idx: int, P: int,
                 grid: int = 21) -> np.ndarray:
    """How loadings respond to ONE characteristic: sweep it over [-1, 1] with
    every other input at its median (0), average over the ensemble. Returns
    (grid, K). The one-at-a-time sweep is a slice of a nonlinear surface —
    read it as the marginal shape, not a full decomposition."""
    zs = np.linspace(-1, 1, grid)
    Z = np.zeros((grid, P))
    Z[:, feature_idx] = zs
    return np.mean([betas_of(m, Z) for m in models], axis=0)


def grad_check(seed: int = 0, eps: float = 1e-6) -> float:
    """Max relative error of the analytic gradient vs finite differences on a
    tiny synthetic month. Used by the smoke test, not the app."""
    rng = np.random.default_rng(seed)
    P, K, N = 5, 3, 40
    params = init_params(P, K, (8,), rng)
    Z = rng.standard_normal((N, P))
    r = rng.standard_normal(N) * 0.05
    x = np.concatenate(([r.mean()], Z.T @ r / N))
    _, grads = month_loss_grad(params, Z, r, x)
    worst = 0.0
    for k, g in grads.items():
        flat = params[k].ravel()
        for j in rng.choice(flat.size, size=min(10, flat.size), replace=False):
            old = flat[j]
            flat[j] = old + eps
            lp, _ = month_loss_grad(params, Z, r, x)
            flat[j] = old - eps
            lm, _ = month_loss_grad(params, Z, r, x)
            flat[j] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g.ravel()[j]) / max(abs(num), 1e-8))
    return worst
