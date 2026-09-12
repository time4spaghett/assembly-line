"""
Learned Edge — infer a manager's implicit style from revealed holdings.

Given a panel of (date, secid, features..., held_flag), fit what distinguishes
held names from the rest. The output is not the fitted model but an
out-of-sample probability on every row: `hold_prob`, produced by a walk-forward
refit so that no score is ever informed by data from its own period or later.

Panel contract (long, one row per security per date):
    date, secid, <feature columns...>, <target column, 0/1>

Design notes worth keeping:

  * The window EXPANDS rather than rolls. A manager's style accumulates; the
    2010 decisions are still evidence in 2020, so discarding them would throw
    away signal to control for a drift we have not shown exists.
  * Each fold refits FROM SCRATCH. Warm-starting would let the first fold's fit
    leak forward through every later one, which is exactly what the protocol
    exists to prevent.
  * Forest over logit by default, to capture size x style interactions — a
    manager who tolerates richer valuation or higher beta in large caps than in
    small ones is expressing an interaction a linear model cannot see. A logit
    runs the identical protocol for comparison.
  * The held class is typically ~3% of rows. Shallow depth and fat leaves are
    the regularisers against that imbalance; class_weight rebalances it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Defaults from the reference implementation.
INIT_TRAIN_YEARS = 5
STEP_YEARS = 1
FOREST_KWARGS = dict(n_estimators=400, max_depth=4, min_samples_leaf=200,
                     max_features="sqrt", class_weight="balanced_subsample",
                     random_state=0, n_jobs=-1)
# Size cohorts: the walk-forward runs separately inside each, so the style
# profile is allowed to differ by size bucket.
SIZE_GROUPS = {"bottom": (0.00, 0.20), "mid_20_80": (0.20, 0.80),
               "top_quintile": (0.80, 1.01)}

PROB_COL = "hold_prob"


def make_model(kind: str = "forest"):
    """A fresh, unfitted estimator. Called once per fold — never reused."""
    if kind == "forest":
        return RandomForestClassifier(**FOREST_KWARGS)
    return Pipeline([("scale", StandardScaler()),
                     ("logit", LogisticRegression(max_iter=2000,
                                                  class_weight="balanced"))])


@dataclass
class Fold:
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    pos_rate_train: float
    auc: float
    cohort: str = "all"


@dataclass
class LearnedResult:
    panel: pd.DataFrame                    # input rows + out-of-sample PROB_COL
    folds: pd.DataFrame                    # one row per fold, with OOS AUC
    importances: pd.Series                 # time-averaged across folds
    pooled_auc: float                      # AUC over all OOS rows at once
    features: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def _fold_windows(dates: pd.Series, init_train_years: int, step_years: int):
    """Yield (train_end, test_end) for an expanding window."""
    start, end = dates.min(), dates.max()
    train_end = start + pd.DateOffset(years=init_train_years)
    while train_end <= end:
        yield train_end, train_end + pd.DateOffset(years=step_years)
        train_end = train_end + pd.DateOffset(years=step_years)


def _importances(model, features: list) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    coef = model.named_steps["logit"].coef_[0]      # logit: magnitude of effect
    denom = np.abs(coef).sum() or 1.0
    return np.abs(coef) / denom


def walk_forward(df: pd.DataFrame, features: list, target: str,
                 date_col: str = "date", kind: str = "forest",
                 init_train_years: int = INIT_TRAIN_YEARS,
                 step_years: int = STEP_YEARS,
                 cohort: str = "all",
                 progress=None) -> LearnedResult:
    """
    Expanding-window, out-of-sample refit.

    For each fold: train on everything strictly before `train_end`, score the
    following `step_years` window, then absorb that window into the next
    training set. Every row is scored exactly once, by a model that never saw
    it.
    """
    d = df.sort_values(date_col).reset_index(drop=True).copy()
    d[date_col] = pd.to_datetime(d[date_col])
    out = pd.Series(np.nan, index=d.index, dtype=float)

    rows, imps, skipped = [], [], []
    windows = list(_fold_windows(d[date_col], init_train_years, step_years))
    for i, (train_end, test_end) in enumerate(windows):
        if progress:
            progress(i / max(len(windows), 1), f"{cohort}: fold to {train_end:%Y}")
        tr = d[date_col] < train_end
        te = (d[date_col] >= train_end) & (d[date_col] < test_end)
        if not te.any():
            skipped.append(f"{train_end:%Y}: no rows in the test window")
            continue
        y_tr = d.loc[tr, target]
        if y_tr.nunique() < 2:
            skipped.append(f"{train_end:%Y}: training slice has one class only")
            continue

        model = make_model(kind)                    # from scratch, every fold
        model.fit(d.loc[tr, features], y_tr)
        p = model.predict_proba(d.loc[te, features])[:, 1]
        out.loc[d.index[te]] = p

        y_te = d.loc[te, target]
        auc = float(roc_auc_score(y_te, p)) if y_te.nunique() == 2 else np.nan
        rows.append(Fold(train_end, test_end, int(tr.sum()), int(te.sum()),
                         float(y_tr.mean()), auc, cohort))
        imps.append(pd.Series(_importances(model, features), index=features))

    d[PROB_COL] = out
    scored = d[PROB_COL].notna()
    pooled = (float(roc_auc_score(d.loc[scored, target], d.loc[scored, PROB_COL]))
              if scored.any() and d.loc[scored, target].nunique() == 2 else np.nan)
    return LearnedResult(
        panel=d,
        folds=pd.DataFrame([f.__dict__ for f in rows]),
        # time-averaged, not pooled: each fold counts once regardless of how
        # many rows it happened to contain
        importances=(pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
                     if imps else pd.Series(dtype=float)),
        pooled_auc=pooled, features=list(features), skipped=skipped)


def size_pctile(df: pd.DataFrame, size_col: str, date_col: str = "date") -> pd.Series:
    """Cross-sectional size percentile within each date, in [0, 1]."""
    return df.groupby(date_col)[size_col].rank(pct=True)


def build_learned_style_panel(
        df: pd.DataFrame, features: list, target: str,
        size_col=None, date_col: str = "date",
        kind: str = "forest", init_train_years: int = INIT_TRAIN_YEARS,
        step_years: int = STEP_YEARS, progress=None) -> LearnedResult:
    """
    Run the walk-forward separately inside each size cohort, then union.

    One global model would average a style that genuinely differs by size
    bucket. Fitting per cohort lets the profile differ; the cost is a smaller
    training set in each, which is why the cohorts are coarse.
    """
    if not size_col:
        return walk_forward(df, features, target, date_col, kind,
                            init_train_years, step_years, "all", progress)

    d = df.copy()
    d["_size_pctile"] = size_pctile(d, size_col, date_col)
    parts, folds, imps, skipped = [], [], [], []
    for i, (name, (lo, hi)) in enumerate(SIZE_GROUPS.items()):
        sub = d[(d["_size_pctile"] >= lo) & (d["_size_pctile"] < hi)]
        if sub.empty:
            skipped.append(f"{name}: no rows")
            continue
        sub_progress = None
        if progress:
            def sub_progress(f, m, i=i):
                progress((i + f) / len(SIZE_GROUPS), m)
        r = walk_forward(sub.drop(columns="_size_pctile"), features, target,
                         date_col, kind, init_train_years, step_years, name,
                         sub_progress)
        parts.append(r.panel.assign(cohort=name))
        folds.append(r.folds)
        if len(r.importances):
            imps.append(r.importances.rename(name))
        skipped += [f"{name}: {s}" for s in r.skipped]

    panel = pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0]
    scored = (panel[PROB_COL].notna() if PROB_COL in panel.columns
              else pd.Series(False, index=panel.index))
    pooled = (float(roc_auc_score(panel.loc[scored, target], panel.loc[scored, PROB_COL]))
              if scored.any() and panel.loc[scored, target].nunique() == 2 else np.nan)
    return LearnedResult(
        panel=panel,
        folds=pd.concat(folds, ignore_index=True) if folds else pd.DataFrame(),
        importances=(pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
                     if imps else pd.Series(dtype=float)),
        pooled_auc=pooled, features=list(features), skipped=skipped)


# ── Returns-based style analysis (Sharpe-style) ──────────────────────────────
# The same question as the classifier — what is this manager's style — but
# inferred from returns instead of characteristics. Each month the manager's
# implied return (held names) is regressed on each factor's long-short return
# within the universe; the loadings are the factor mix that best replicates
# the manager. Ridge rather than OLS: factor L/S series are collinear, and
# with ~200 months and a dozen factors unconstrained betas swing fold to fold
# and read as noise. Sharpe's sum-to-one / non-negative constraints do not
# apply here — the regressors are L/S legs, not long-only asset classes, so a
# negative loading is a real answer.

from sklearn.linear_model import RidgeCV

RIDGE_ALPHAS = np.logspace(-3, 2, 24)
MIN_TRAIN_MONTHS = 24


def factor_ls_returns(df: pd.DataFrame, features: list, ret_col: str,
                      date_col: str = "date", n_q: int = 5) -> pd.DataFrame:
    """
    Top-ntile minus bottom-ntile return per feature per date, equal-weight.

    A date needs at least n_q*5 names with both the feature and the return
    present (the same floor the Concierge's ntile cutter uses); otherwise that
    factor is NaN on that date rather than cut from too few names.
    """
    out = {}
    g = df.groupby(date_col)
    for f in features:
        vals = {}
        for d, sub in g:
            s = sub[[f, ret_col]].dropna()
            if len(s) < n_q * 5:
                vals[d] = np.nan
                continue
            q = pd.qcut(s[f].rank(method="first"), n_q, labels=False)
            vals[d] = float(s.loc[q == n_q - 1, ret_col].mean()
                            - s.loc[q == 0, ret_col].mean())
        out[f] = pd.Series(vals)
    return pd.DataFrame(out).sort_index()


def manager_returns(df: pd.DataFrame, target: str, ret_col: str,
                    date_col: str = "date", weight_col=None) -> pd.Series:
    """Return of the held names each date — weighted if a weight column is given."""
    held = df[df[target] == 1].dropna(subset=[ret_col])
    if weight_col:
        held = held.dropna(subset=[weight_col])
        w = held[weight_col].astype(float).clip(lower=0)
        num = (held[ret_col] * w).groupby(held[date_col]).sum()
        den = w.groupby(held[date_col]).sum()
        return (num / den.replace(0, np.nan)).sort_index()
    return held.groupby(date_col)[ret_col].mean().sort_index()


@dataclass
class StyleFold:
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int          # months
    n_test: int
    oos_corr: float
    oos_r2: float
    alpha_ann: float      # intercept, annualized — return the factors don't explain


@dataclass
class StyleResult:
    manager: pd.Series                # actual manager return per month
    replicated: pd.Series             # OOS replicated return, NaN before coverage
    factor_returns: pd.DataFrame      # the L/S legs, months × features
    folds: pd.DataFrame
    loadings_by_fold: pd.DataFrame    # folds × features (raw-unit betas)
    loadings: pd.Series               # time-averaged across folds
    pooled_corr: float
    pooled_r2: float
    features: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def _ridge_fit(X: pd.DataFrame, y: pd.Series):
    """Ridge on standardized regressors; returns raw-unit betas + intercept."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    m = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xs, y)
    beta = pd.Series(m.coef_ / scaler.scale_, index=X.columns)
    intercept = float(m.intercept_ - (beta * scaler.mean_).sum())
    return beta, intercept


def _design(df, features, target, ret_col, date_col, weight_col, n_q):
    fr = factor_ls_returns(df, features, ret_col, date_col, n_q)
    mr = manager_returns(df, target, ret_col, date_col, weight_col)
    skipped = []
    # a factor missing on many dates would drag every other factor's months
    # down with it in a listwise drop — cut it instead, and say so
    sparse = fr.columns[fr.isna().mean() > 0.25].tolist()
    for f in sparse:
        skipped.append(f"{f}: L/S leg missing on "
                       f"{fr[f].isna().mean():.0%} of months, dropped")
    fr = fr.drop(columns=sparse)
    X = fr.dropna()
    y = mr.reindex(X.index).dropna()
    X = X.loc[y.index]
    return X, y, fr, mr, skipped


def walk_forward_style(df: pd.DataFrame, features: list, target: str,
                       ret_col: str, date_col: str = "date", weight_col=None,
                       n_q: int = 5, init_train_years: int = INIT_TRAIN_YEARS,
                       step_years: int = STEP_YEARS, progress=None) -> StyleResult:
    """
    Expanding-window, out-of-sample style regression.

    Each fold: fit loadings on all months strictly before train_end, then
    replicate the following step_years window with those loadings. Every
    replicated month comes from a fit that never saw it.
    """
    X, y, fr, mr, skipped = _design(df, features, target, ret_col, date_col,
                                    weight_col, n_q)
    replicated = pd.Series(np.nan, index=y.index, dtype=float)
    rows, betas = [], []
    windows = list(_fold_windows(pd.Series(y.index), init_train_years, step_years))
    for i, (train_end, test_end) in enumerate(windows):
        if progress:
            progress(i / max(len(windows), 1), f"fold to {train_end:%Y}")
        tr = y.index < train_end
        te = (y.index >= train_end) & (y.index < test_end)
        if tr.sum() < MIN_TRAIN_MONTHS:
            skipped.append(f"{train_end:%Y}: only {int(tr.sum())} training months")
            continue
        if not te.any():
            skipped.append(f"{train_end:%Y}: no months in the test window")
            continue
        beta, b0 = _ridge_fit(X[tr], y[tr])
        yhat = X[te] @ beta + b0
        replicated[te] = yhat.values
        yt = y[te]
        corr = float(np.corrcoef(yt, yhat)[0, 1]) if len(yt) > 2 else np.nan
        sst = float(((yt - yt.mean()) ** 2).sum())
        r2 = float(1 - ((yt - yhat) ** 2).sum() / sst) if sst > 0 else np.nan
        rows.append(StyleFold(train_end, test_end, int(tr.sum()), int(te.sum()),
                              corr, r2, b0 * 12))
        betas.append(beta.rename(train_end))

    scored = replicated.notna()
    if scored.sum() > 2:
        ys, yh = y[scored], replicated[scored]
        pooled_corr = float(np.corrcoef(ys, yh)[0, 1])
        pooled_r2 = float(1 - ((ys - yh) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())
    else:
        pooled_corr = pooled_r2 = np.nan
    lb = pd.DataFrame(betas) if betas else pd.DataFrame(columns=X.columns)
    return StyleResult(
        manager=y, replicated=replicated, factor_returns=fr,
        folds=pd.DataFrame([f.__dict__ for f in rows]),
        loadings_by_fold=lb,
        loadings=(lb.mean().sort_values(key=abs, ascending=False)
                  if len(lb) else pd.Series(dtype=float)),
        pooled_corr=pooled_corr, pooled_r2=pooled_r2,
        features=list(X.columns), skipped=skipped)


def full_history_style(df: pd.DataFrame, features: list, target: str,
                       ret_col: str, date_col: str = "date", weight_col=None,
                       n_q: int = 5) -> dict:
    """Fit once on every month, to describe the style. In-sample; never scored from."""
    X, y, _, _, skipped = _design(df, features, target, ret_col, date_col,
                                  weight_col, n_q)
    if len(y) < MIN_TRAIN_MONTHS:
        return {"loadings": pd.Series(dtype=float), "r2": np.nan,
                "alpha_ann": np.nan, "n_months": int(len(y)), "skipped": skipped}
    beta, b0 = _ridge_fit(X, y)
    yhat = X @ beta + b0
    r2 = float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return {"loadings": beta.sort_values(key=abs, ascending=False), "r2": r2,
            "alpha_ann": b0 * 12, "n_months": int(len(y)), "skipped": skipped}


# ── What the model actually keys on, over the whole history ──────────────────
# The walk-forward exists to produce honest scores; it is a poor lens on what
# the model learned, because each fold sees a different slice. For a picture of
# the style itself, fit once on everything. That fit is explicitly NOT used for
# scoring — it has seen every row, so its probabilities would be worthless.

def _pair_interactions(forest, features: list, top: int = 12) -> pd.DataFrame:
    """
    Feature pairs that repeatedly split along the same root-to-leaf path.

    A tree expresses an interaction by conditioning one feature on another —
    splitting on `bp` only inside a branch already split on `size` is the tree
    saying value matters differently by size. Counting co-occurrences along a
    path, weighted by the impurity each split removed, surfaces those pairs.
    Cheap and directional, not a formal H-statistic: read it as a pointer to
    where to look, not a test.
    """
    from collections import defaultdict
    strength = defaultdict(float)
    for est in forest.estimators_:
        t = est.tree_
        # walk down, carrying the ancestors of each node
        stack = [(0, ())]
        while stack:
            node, ancestors = stack.pop()
            if t.children_left[node] == -1:
                continue
            f = t.feature[node]
            gain = float(t.impurity[node] * t.weighted_n_node_samples[node])
            for a in ancestors:
                if a != f:
                    strength[tuple(sorted((a, f)))] += gain
            nxt = ancestors + (f,)
            stack.append((t.children_left[node], nxt))
            stack.append((t.children_right[node], nxt))
    if not strength:
        return pd.DataFrame(columns=["pair", "strength"])
    total = sum(strength.values()) or 1.0
    rows = [{"pair": f"{features[a]} x {features[b]}", "strength": v / total}
            for (a, b), v in strength.items()]
    return (pd.DataFrame(rows).sort_values("strength", ascending=False)
            .head(top).reset_index(drop=True))


def full_history_profile(df: pd.DataFrame, features: list, target: str,
                         size_col=None, date_col: str = "date") -> dict:
    """
    Fit once on all history to describe the style, and report:
      * importances over the whole sample
      * the strongest feature interactions the trees express
      * how importances differ by size cohort — which IS the size x style
        interaction, stated directly rather than inferred

    In-sample by construction. Descriptive only; never scored from.
    """
    d = df.dropna(subset=features + [target])
    model = make_model("forest")
    model.fit(d[features], d[target])
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    pairs = _pair_interactions(model, features)

    by_cohort = pd.DataFrame()
    if size_col:
        pc = size_pctile(d, size_col, date_col)
        cols = {}
        for name, (lo, hi) in SIZE_GROUPS.items():
            sub = d[(pc >= lo) & (pc < hi)]
            if len(sub) < 500 or sub[target].nunique() < 2:
                continue
            m = make_model("forest")
            m.fit(sub[features], sub[target])
            cols[name] = pd.Series(m.feature_importances_, index=features)
        by_cohort = pd.DataFrame(cols)
        if not by_cohort.empty:
            # the spread across cohorts is the size-conditionality of each feature
            by_cohort["spread"] = by_cohort.max(axis=1) - by_cohort.min(axis=1)
            by_cohort = by_cohort.sort_values("spread", ascending=False)
    return {"importances": imp, "interactions": pairs, "by_cohort": by_cohort,
            "n_rows": int(len(d)), "pos_rate": float(d[target].mean())}
