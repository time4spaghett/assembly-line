"""
Two-pass optimizer — Chinook's mean-variance construction, carried into the
dashboard so the Destination & Path tab can ask: given this pair of edges, how
would the two-pass approach do on this coverage?

Copied from C:\\Users\\Alex\\Chinook at tag phaseE-two-pass-optimizer (commit
b6c460e): chinook/risk.py (the factor risk model, whole), chinook/optimize.py
(mvo_destination, mvo_pass2 and their helpers, whole), and the small pieces
of core.py / panel.py / backtest.py they rest on (group_bands, active_share,
one_way_turnover, merge_small_groups, drift). Function bodies are verbatim;
what is new is TwoPassParams (the mvo fields of Chinook's Params, same
defaults) and run_two_pass, which replaces Chinook's backtest loop with one
that takes the tab's coverage frame and edge columns.

The problem (optimize.py's docstring, Chinook):

  Pass 1 chooses portfolio weights w to
      maximize    alpha' (w - bench)  -  (risk_aversion / 2) * (w - bench)' V (w - bench)
                  -  hold_hurdle * |w - held|_1
      subject to  w >= 0, sum(w) = 1, w <= name cap
                  |sector weight - benchmark sector weight| <= band
  Pass 2, from the held portfolio toward that destination,
      maximize    alpha_short' (w - held)  -  (risk_aversion / 2) * (w - dest)' V (w - dest)
      subject to  pass 1's constraints, one-way turnover <= budget,
                  and (toward_destination) each w_i between held_i and dest_i
  V = X F X' + D is the factor risk model, monthly units. Alpha is the edge's own
  rank of ranks across the universe, centred, in return units as
  ic * specific vol * standardized rank.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── chinook/risk.py ──────────────────────────────────────────────────────────
from sklearn.covariance import LedoitWolf

# style -> [(panel column, sign)]; a style is the mean of its available z-scored legs
STYLES = {
    "size": [("log_mcap", 1)],
    "beta": [("beta_252d", 1), ("beta_12m", 1)],
    "momentum": [("mom_12_1", 1)],
    "value": [("btm", 1), ("earn_yield", 1)],
    "volatility": [("resid_vol_252d", 1), ("vol_12m", 1)],
    "quality": [("roa", 1), ("gpa", 1)],
    "growth": [("rev_gr_1y", 1), ("asset_gr", 1)],
    "leverage": [("leverage", 1)],
    "reversal": [("ret_1m", 1)],
}


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    s = s.clip(s.quantile(0.01), s.quantile(0.99))
    sd = s.std()
    return ((s - s.mean()) / sd).clip(-3, 3) if sd and sd > 0 else s * 0.0


def exposures(frame: pd.DataFrame, group_col: str = "sector", extra: tuple = ()) -> pd.DataFrame:
    """Sector dummies plus style z-scores for one date frame (indexed by ticker).
    `extra` names further columns to use as style factors, e.g. the alpha itself."""
    X = pd.get_dummies(frame[group_col].fillna("Unknown")).astype(float)
    for col in extra:
        if col in frame.columns:
            X[col] = _z(frame[col]).fillna(0.0)
    for style, legs in STYLES.items():
        zs = [sign * _z(frame[col]) for col, sign in legs if col in frame.columns]
        if zs:
            X[style] = _z(pd.concat(zs, axis=1).mean(axis=1)).fillna(0.0)
    return X


@dataclass
class Snapshot:
    """The model at one date, for that date's names. Monthly variance units."""
    X: pd.DataFrame  # names x factors
    F: pd.DataFrame  # factors x factors
    spec_var: pd.Series

    def variance(self, w: pd.Series) -> float:
        w = w.reindex(self.X.index).fillna(0.0)
        b = self.X.T @ w
        return float(b @ self.F @ b + (w ** 2 * self.spec_var).sum())

    def vol(self, w: pd.Series, periods: int = 12) -> float:
        return float(np.sqrt(self.variance(w) * periods))

    def covariance(self) -> np.ndarray:
        X = self.X.to_numpy()
        return X @ self.F.to_numpy() @ X.T + np.diag(self.spec_var.to_numpy())


class RiskModel:
    def __init__(self, panel: pd.DataFrame, fwd_col: str = "fwd_1m", group_col: str = "sector",
                 halflife: int = 36, min_months: int = 24, daily_vol_col: str = "resid_vol_252d",
                 regime_halflife: int | None = 12):
        self.group_col, self.halflife, self.min_months = group_col, halflife, min_months
        self.regime_halflife = regime_halflife
        self._raw_cache, self._bias = {}, None
        self.daily_vol_col = daily_vol_col if daily_vol_col in panel.columns else None
        self.frames = {d: f.set_index("ticker") for d, f in panel.groupby("date")}
        self.dates = pd.DatetimeIndex(sorted(self.frames))
        self._fit(fwd_col)

    def _fit(self, fwd_col):
        f_rows, resid = {}, {}
        for d in self.dates:
            fr = self.frames[d]
            ok = fr[fwd_col].notna() & fr["log_mcap"].notna()
            if ok.sum() < 100:
                continue
            fr = fr[ok]
            X = exposures(fr, self.group_col)
            y = fr[fwd_col].astype(float).clip(-0.5, 0.5)
            sw = np.exp(fr["log_mcap"].astype(float) / 4.0)  # sqrt of the sqrt(cap) weight
            coef, *_ = np.linalg.lstsq(X.to_numpy() * sw.to_numpy()[:, None],
                                       y.to_numpy() * sw.to_numpy(), rcond=None)
            f_rows[d] = pd.Series(coef, index=X.columns)
            resid[d] = y - X.to_numpy() @ coef
        self.factor_returns = pd.DataFrame(f_rows).T.sort_index().fillna(0.0)
        self.residuals = pd.DataFrame(resid).T.sort_index()  # dates x tickers

    def first_date(self) -> pd.Timestamp:
        # Deviation from Chinook (`self.dates[self.min_months + 1]`): months with under
        # 100 priced names are skipped by _fit, and the top-1500 panel has such months
        # early on, so count fitted months rather than panel months.
        fitted = self.factor_returns.index
        for d in self.dates:
            if (fitted < d).sum() >= self.min_months:
                return d
        raise ValueError(f"fewer than {self.min_months} months of factor returns in the panel")

    def at(self, date) -> Snapshot:
        """The model as of `date`, with the volatility regime adjustment."""
        date = pd.Timestamp(date)
        raw = self._raw(date)
        if not self.regime_halflife:
            return raw
        if self._bias is None:
            self._fit_bias()
        past = self._bias[self._bias.index < date]  # month s is realized at s + 1 <= date
        if len(past) < 6:
            return raw
        w = 0.5 ** (np.arange(len(past))[::-1] / self.regime_halflife)
        lam = (past.mul(w, axis=0).sum() / w.sum()).clip(0.25, 4.0)
        return Snapshot(raw.X, raw.F * lam["factor"], raw.spec_var * lam["specific"])

    def _fit_bias(self):
        """Per month: mean squared standardized factor return and residual, raw model."""
        rows = {}
        for d in self.factor_returns.index:
            if d < self.first_date():
                continue
            snap = self._raw(d)
            f = self.factor_returns.loc[d, snap.F.index]
            e = self.residuals.loc[d].reindex(snap.spec_var.index)
            cap = np.exp(self.frames[d]["log_mcap"].astype(float)).reindex(e.index)
            z2 = (e ** 2 / snap.spec_var).clip(upper=25.0)
            ok = z2.notna()
            rows[d] = {"factor": float((f ** 2 / np.diag(snap.F)).clip(upper=25.0).mean()),
                       "specific": float((z2[ok] * cap[ok]).sum() / cap[ok].sum())}
        self._bias = pd.DataFrame(rows).T.sort_index()

    def _raw(self, date) -> Snapshot:
        if date in self._raw_cache:
            return self._raw_cache[date]
        # the forward month of frame s ends at frame s + 1, so s must be before `date`
        hist = self.factor_returns.index[self.factor_returns.index < date]
        if len(hist) < self.min_months:
            raise ValueError(f"{date.date()}: only {len(hist)} months of factor returns")
        age = np.arange(len(hist))[::-1]
        w = 0.5 ** (age / self.halflife)
        w = w / w.sum()

        fr = self.frames[date]
        fr = fr[fr["log_mcap"].notna()]
        X = exposures(fr, self.group_col)
        cols = [c for c in X.columns if c in self.factor_returns.columns]
        X = X[cols]
        f = self.factor_returns.loc[hist, cols].to_numpy()
        lw = LedoitWolf(assume_centered=True).fit(f * np.sqrt(w * len(w))[:, None])
        F = pd.DataFrame(lw.covariance_, index=cols, columns=cols)

        e = self.residuals.loc[hist].reindex(columns=X.index).to_numpy()
        seen = ~np.isnan(e)
        wsum = (seen * w[:, None]).sum(axis=0)
        own = np.where(seen.sum(axis=0) >= 12,
                       np.nansum(e ** 2 * w[:, None], axis=0) / np.where(wsum > 0, wsum, np.nan), np.nan)
        own = pd.Series(own, index=X.index)
        if self.daily_vol_col:
            daily = fr[self.daily_vol_col].astype(float) ** 2  # already a 21-day variance
            # daily residuals are net of the market only, so they still hold sector and
            # style risk the factor block already counts: rescale their level to the
            # model's own residuals (as of this date) and keep their cross-section
            daily = daily * (own / daily).median()
            spec = pd.concat([own, daily], axis=1).mean(axis=1)  # whichever exist
        else:
            spec = own
        spec = spec.clip(spec.quantile(0.02), spec.quantile(0.98))
        self._raw_cache[date] = Snapshot(X, F, spec.fillna(spec.median()))
        return self._raw_cache[date]


# ── chinook/core.py, chinook/panel.py, chinook/backtest.py (helpers) ─────────

def group_bands(bench_g: pd.Series, band_abs, band_rel, band_cap) -> pd.Series:
    return np.minimum(np.maximum(band_abs, band_rel * bench_g), band_cap * bench_g)


def active_share(w, bench) -> float:
    names = w.index.union(bench.index)
    return 0.5 * (w.reindex(names, fill_value=0.0)
                  - bench.reindex(names, fill_value=0.0)).abs().sum()


def one_way_turnover(trades: pd.Series) -> float:
    return 0.5 * trades.abs().sum()


def merge_small_groups(groups: pd.Series, min_size: int = 8,
                       other: str = "Other") -> pd.Series:
    """Pool groups with fewer than min_size names; fold a thin pool into the
    smallest remaining group."""
    counts = groups.value_counts()
    small = counts.index[counts < min_size]
    if len(small) == 0:
        return groups
    out = groups.where(~groups.isin(small), other)
    counts = out.value_counts()
    if counts[other] < min_size and len(counts) > 1:
        out = out.where(out != other, counts.drop(other).idxmin())
    return out


def drift(w, period_return):
    v = w * (1 + period_return.reindex(w.index).fillna(0.0))
    return v / v.sum()


# ── chinook/config.py (the mvo fields, same defaults) ────────────────────────

@dataclass
class TwoPassParams:
    band_abs: float = 0.02
    band_rel: float = 0.25
    band_cap: float = 0.5
    cap_abs: float = 0.05
    cap_mult: float = 3.0
    max_names: float | None = None
    min_per_group: int = 2
    hold_hurdle: float = 0.001  # mvo pass 1: monthly return a move away from the held portfolio must clear
    dest_every: int = 1  # steps between long-alpha refreshes (1 = every month)
    risk_aversion: float = 30.0  # mvo only: penalty on predicted active variance (monthly units)
    ic_long: float = 0.02  # mvo only: monthly IC that puts the rank alpha in return units
    te_target: float | None = None  # mvo only: optional hard cap on predicted tracking error, annual
    max_active: float | None = None  # mvo only: limit on any name's active weight, e.g. 0.01
    ic_short: float = 0.015  # mvo_pass2: monthly IC that puts the short rank alpha in return units
    toward_destination: bool = True  # mvo_pass2: trades only move toward the destination, never past it
    turnover_budget: float | None = 0.03  # one-way, per step


# ── chinook/optimize.py ──────────────────────────────────────────────────────
import cvxpy as cp  # noqa: E402

RANK_SD = 12 ** -0.5  # standard deviation of a uniform rank on [-0.5, 0.5]


def _alpha(rank: pd.Series, risk, names, ic: float) -> np.ndarray:
    score = rank.reindex(names).fillna(0.0) / RANK_SD
    return (ic * score * np.sqrt(risk.spec_var.reindex(names))).to_numpy()


def _setup(bench, groups, risk, p):
    names = bench.index.intersection(risk.X.index)
    b = (bench.reindex(names) / bench.reindex(names).sum()).to_numpy()
    g = groups.reindex(names)
    X = risk.X.reindex(names).to_numpy()
    L = np.linalg.cholesky(risk.F.to_numpy() + 1e-12 * np.eye(len(risk.F)))
    sd = np.sqrt(risk.spec_var.reindex(names).to_numpy())
    w = cp.Variable(len(names))
    active_var = cp.sum_squares(L.T @ (X.T @ (w - b))) + cp.sum_squares(cp.multiply(sd, w - b))
    cons = [w >= 0, w <= np.maximum(p.cap_abs, p.cap_mult * b)]
    if p.max_active:
        cons += [w - b <= p.max_active, w - b >= -p.max_active]
    if p.te_target:
        cons += [active_var <= p.te_target ** 2 / 12.0]
    bench_g = pd.Series(b, index=names).groupby(g).sum()
    band = group_bands(bench_g, p.band_abs, p.band_rel, p.band_cap)
    for sector in bench_g.index:
        mask = (g == sector).to_numpy().astype(float)
        cons += [mask @ (w - b) <= band[sector], mask @ (w - b) >= -band[sector]]
    return names, b, w, active_var, cons


def _solve(objective, cons) -> bool:
    prob = cp.Problem(cp.Maximize(objective), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        prob.solve(solver=cp.SCS)
    return prob.status in ("optimal", "optimal_inaccurate")


def mvo_destination(bench, groups, long_rank, risk, p, held=None) -> pd.Series | None:
    """Pass 1: the mean-variance portfolio for the long alpha, starting from what is held.

    With `held`, moving a weight away from the held portfolio costs `hold_hurdle` of
    monthly expected return per unit moved, so the destination only leaves the held
    portfolio where the long alpha (net of risk) clears that hurdle. At hold_hurdle 0
    the solve ignores holdings. Returns None when the solver fails, so the caller can
    fall back to the heuristic."""
    names, b, w, active_var, cons = _setup(bench, groups, risk, p)
    alpha = _alpha(long_rank, risk, names, p.ic_long)
    objective = alpha @ (w - b) - 0.5 * p.risk_aversion * active_var
    if held is not None and p.hold_hurdle:
        h = held.reindex(names, fill_value=0.0)
        objective = objective - p.hold_hurdle * cp.norm1(w - (h / h.sum()).to_numpy())
    if not _solve(objective, cons + [cp.sum(w) == 1.0]):
        warnings.warn("mvo_destination: no solution")
        return None
    if p.max_names:
        # names cap: keep the solution's largest weights (at least min_per_group in each
        # sector, so the sector bands stay reachable) and solve again on those names only.
        # The solve already starts from the held portfolio, so held names are favoured.
        first = pd.Series(np.maximum(w.value, 0.0), index=names)
        n = int(round(p.max_names * len(names))) if p.max_names < 1 else int(p.max_names)
        g = groups.reindex(names)
        keep = set(first.groupby(g, group_keys=False).nlargest(p.min_per_group).index)
        keep |= set(first.drop(list(keep)).nlargest(max(n - len(keep), 0)).index)
        out_mask = (~names.isin(list(keep))).astype(float)
        if not _solve(objective, cons + [cp.sum(w) == 1.0, cp.multiply(out_mask, w) == 0]):
            warnings.warn("mvo_destination: names cap infeasible, using the uncapped solution")
            _solve(objective, cons + [cp.sum(w) == 1.0])
    out = pd.Series(np.maximum(w.value, 0.0), index=names)
    out[out < 1e-7] = 0.0
    return (out / out.sum()).reindex(bench.index, fill_value=0.0)


def mvo_pass2(held, dest, bench, groups, short_rank, risk, p) -> pd.Series | None:
    """Pass 2 of the two-pass optimizer. Same constraint set as pass 1, plus the
    turnover budget:

        maximize    alpha_short' (w - held)  -  (risk_aversion / 2) * (w - dest)' V (w - dest)
        subject to  pass 1's constraints, one-way turnover <= budget,
                    and (toward_destination) each w_i between held_i and dest_i

    The second term is what pass 1's objective loses by not being at its optimum
    `dest` (the objective is quadratic with curvature risk_aversion * V), so an
    unspent gap costs the most where it carries the most risk. The short alpha
    then shifts the budget to the trades it likes. With toward_destination the
    short alpha only prioritizes: it never reverses a trade or pushes one past
    the destination. Exits are sold first, as in the heuristic.
    """
    names, b, w, _, cons = _setup(bench, groups, risk, p)
    outside = held[~held.index.isin(names) & (held > 0)]
    sold = min(outside.sum(), p.turnover_budget)
    leftover = outside.sum() - sold
    h = held.reindex(names, fill_value=0.0).to_numpy()
    d = dest.reindex(names, fill_value=0.0).to_numpy()
    X = risk.X.reindex(names).to_numpy()
    L = np.linalg.cholesky(risk.F.to_numpy() + 1e-12 * np.eye(len(risk.F)))
    sd = np.sqrt(risk.spec_var.reindex(names).to_numpy())
    distance = cp.sum_squares(L.T @ (X.T @ (w - d))) + cp.sum_squares(cp.multiply(sd, w - d))
    budget = [cp.sum(w) == 1.0 - leftover, cp.norm1(w - h) <= 2.0 * p.turnover_budget - sold]
    box = [w >= np.minimum(h, d), w <= np.maximum(h, d)] if p.toward_destination else []
    objective = _alpha(short_rank, risk, names, p.ic_short) @ (w - h) - 0.5 * p.risk_aversion * distance
    # a drifted held portfolio can sit outside pass 1's constraints with no way back inside
    # the budget; then only the path to the destination is constrained
    if not _solve(objective, cons + budget + box) and not _solve(objective, [w >= 0] + budget + box):
        warnings.warn("mvo_pass2: no solution")
        return None
    trades = pd.Series(np.maximum(w.value, 0.0) - h, index=names)
    trades[trades.abs() < 1e-7] = 0.0
    if sold > 0:
        trades = pd.concat([trades, -outside * (sold / outside.sum())])
    buys, sells = trades.clip(lower=0.0), trades.clip(upper=0.0)
    return buys * (-sells.sum() / buys.sum()) + sells if buys.sum() > 0 else trades * 0.0


def attach_risk_features(panel: pd.DataFrame, path=None) -> pd.DataFrame:
    """Join beta_252d / resid_vol_252d (risk_features_build.py) when the file is
    present; without it the risk model runs on the panel's own style legs."""
    from pathlib import Path
    path = Path(path) if path else Path(__file__).parent / "data" / "risk_features_1500.parquet"
    if not path.exists() or "beta_252d" in panel.columns:
        return panel
    rf = pd.read_parquet(path)
    return panel.merge(rf, on=["date", "ticker"], how="left")


# ── The tab's loop (new; replaces chinook/backtest.backtest for one coverage) ─

VARIANTS = ("destination held directly", "pass 2 without the path edge", "two-pass (full)")


def centred_rank(s: pd.Series) -> pd.Series:
    """Chinook's alpha for the optimizer: rank of ranks across the universe, centred (panel.py:157)."""
    return (s.rank(pct=True) - 0.5).fillna(0.0)


def run_two_pass(coverage: pd.DataFrame, risk: RiskModel, dest_col: str, path_col: str,
                 p: TwoPassParams, init: str = "cap", date_col: str = "date",
                 id_col: str = "ticker", progress=None) -> dict:
    """
    Three books on the tab's coverage, every month the risk model covers:
      destination held directly   held = pass 1's destination each month (no budget)
      pass 2 without the path edge   the same two passes with the short alpha zeroed:
                                   the budget is spent purely by the risk-distance term
      two-pass (full)             pass 1 from the held book, pass 2 steered by the path edge
    Returns {variant: DataFrame[date, port, bench, turnover, active_share, gap, failed]}.
    Starting book: the cap-weighted coverage ('cap') or equal-weight ('ew').
    """
    frames = {d: f.set_index(id_col) for d, f in coverage.groupby(date_col)}
    dates = [d for d in sorted(frames) if d >= risk.first_date()]
    out = {}
    for vi, variant in enumerate(VARIANTS):
        held, plan_rank, rows, failed = None, None, [], 0
        for i, t in enumerate(dates):
            if progress:
                progress((vi * len(dates) + i) / (3 * len(dates)), f"{variant}: {t:%Y-%m}")
            f = frames[t]
            cap = np.exp(f["log_mcap"].astype(float)).dropna()
            f = f.loc[cap.index]
            bench = cap / cap.sum()
            groups = merge_small_groups(f["sector"].fillna("Unknown"))
            if plan_rank is None or i % max(int(p.dest_every), 1) == 0:
                plan_rank = centred_rank(f[dest_col].astype(float))
            short_rank = centred_rank(f[path_col].astype(float))
            if variant == "pass 2 without the path edge":
                short_rank = short_rank * 0.0
            snap = risk.at(t)
            if held is None:
                held = bench.copy() if init == "cap" else pd.Series(1.0 / len(bench), index=bench.index)
            dest = mvo_destination(bench, groups, plan_rank.reindex(bench.index).fillna(0.0),
                                   snap, p, held=held)
            if dest is None:
                failed += 1
                dest = held.reindex(bench.index, fill_value=0.0)
                dest = dest / dest.sum()
            if variant == "destination held directly":
                new = dest
                trades = new.reindex(held.index.union(new.index), fill_value=0.0) \
                    - held.reindex(held.index.union(new.index), fill_value=0.0)
            else:
                trades = mvo_pass2(held, dest, bench, groups, short_rank, snap, p)
                if trades is None:
                    failed += 1
                    trades = held * 0.0
                new = held.reindex(trades.index, fill_value=0.0) + trades
            r = f["fwd_1m"].astype(float).reindex(new.index).fillna(0.0)
            bench_r = f["fwd_1m"].astype(float).reindex(bench.index).fillna(0.0)
            rows.append({"date": t, "port": float((new * r).sum()), "bench": float((bench * bench_r).sum()),
                         "turnover": one_way_turnover(trades), "active_share": active_share(new, bench),
                         "gap": active_share(new, dest), "names": int((new > 1e-6).sum())})
            held = drift(new[new > 1e-12], r)
        df = pd.DataFrame(rows).set_index("date")
        df.attrs["failed"] = failed
        out[variant] = df
    return out


def summarize(res: pd.DataFrame, periods_per_year: int = 12) -> dict:
    port, bench = res["port"], res["bench"]
    active = port - bench
    te = active.std() * periods_per_year ** 0.5
    ar = active.mean() * periods_per_year
    cum = (1 + port).cumprod()
    vol = port.std() * periods_per_year ** 0.5
    ann = (1 + port.mean()) ** periods_per_year - 1
    return {"ann_return": ann, "ann_vol": vol, "sharpe": ann / vol if vol > 0 else np.nan,
            "max_drawdown": float((cum / cum.cummax() - 1).min()),
            "active_return": ar, "tracking_error": te,
            "information_ratio": ar / te if te > 0 else np.nan,
            "annual_turnover": float(res["turnover"].mean() * periods_per_year),
            "avg_active_share": float(res["active_share"].mean()),
            "avg_gap": float(res["gap"].mean()), "months": int(len(res)),
            "solver_failures": int(res.attrs.get("failed", 0))}


def problem_statement(p: TwoPassParams, budget_two_way: float, rerank: int) -> str:
    """The problem as it will be solved, with the numbers filled in."""
    return f"""
**Pass 1 — destination** (every month, from the held book; long alpha refreshed every {rerank} month(s))

    maximize   α_dest'(w − bench) − ({p.risk_aversion:g}/2)·(w − bench)' V (w − bench) − {p.hold_hurdle:g}·|w − held|₁
    s.t.       w ≥ 0,  Σw = 1,  w_i ≤ max({p.cap_abs:g}, {p.cap_mult:g}·bench_i)
               |sector active weight| ≤ max({p.band_abs:g}, {p.band_rel:g}·bench_sector), capped at {p.band_cap:g}·bench_sector

**Pass 2 — trading step** (toward that destination, path alpha steering)

    maximize   α_path'(w − held) − ({p.risk_aversion:g}/2)·(w − dest)' V (w − dest)
    s.t.       pass 1's constraints,  one-way turnover ≤ {p.turnover_budget:.1%} (the tab's {budget_two_way:.0%} buys + sells),
               {"held_i ≤ w_i ≤ dest_i or dest_i ≤ w_i ≤ held_i (never past the destination)" if p.toward_destination else "no direction box"}

**Alpha**: each edge's rank across the coverage, centred, × IC × specific vol — IC_dest {p.ic_long:g}, IC_path {p.ic_short:g} per month.
**V**: factor risk model (sectors + styles, Ledoit-Wolf EWMA factor covariance, blended specific risk, volatility-regime adjusted), fit on the whole selected panel, monthly units.
"""
