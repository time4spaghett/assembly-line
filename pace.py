"""
Pace — unite a slow destination edge with a fast path edge.

The destination edge says where the book should be; the path edge says, name
by name, whether this is the month to move there. Every month:

  1. Target x*: equal weight over the top ntile of the destination score
     among the names in coverage; zero elsewhere.
  2. Desired change Δ*ᵢ = x*ᵢ − xᵢ,ₜ₋₁ and its direction dᵢ = sign(Δ*ᵢ).
  3. Path signal as a centred rank rᵢ = 2·pct_rankᵢ − 1 ∈ [−1, 1]: the median
     name is 0, positive = expected to beat the coverage next month.
  4. Urgency sᵢ = dᵢ·rᵢ. Positive when the path agrees with the trade —
     buying a name about to run, selling one about to drop. Negative when
     it says wait: buying something about to be cheaper, selling something
     still going up. Only sᵢ > 0 trades this month; the rest carry forward.
  5. Turnover budget B (fraction of NAV, buys + sells) is spread over the
     urgent names in proportion to urgency and capped by need:
         Δexecᵢ = min(|Δ*ᵢ|, λ·sᵢ⁺),   λ s.t. Σ Δexecᵢ = min(B, Σ_{s>0}|Δ*ᵢ|)
     — the water-filling form of "allocate max(s,0)/Σmax(s,0), complete
     k = min(1, |Δexec|/|Δ*|)", with what a saturated name doesn't need
     flowing to the next most urgent instead of being lost.
  6. xᵢ,ₜ = xᵢ,ₜ₋₁ + kᵢ·Δ*ᵢ, then renormalised to sum to one (partial fills
     leave the book slightly over- or under-invested; this parks or draws the
     residual pro-rata).

Names that drop out of coverage are liquidated that month outside the budget
(they have left the analyst's universe — usually a delisting or a cap change),
and their turnover is reported separately.

Three books are run on identical inputs so the path edge's contribution can
be isolated:
  wholesale  — k = 1 every month: the naive monthly rebalance to target.
  uniform    — the same budget B, allocated by trade size alone (no path).
  modulated  — the mechanism above.
If modulated beats uniform, the short-horizon signal is paying; if uniform
already beats wholesale, that part is just lower turnover.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MODES = ("wholesale", "uniform", "modulated")


@dataclass
class PaceResult:
    returns: pd.Series          # monthly portfolio return, indexed by date
    turnover: pd.Series         # budgeted turnover used each month (Σ|Δexec|)
    forced: pd.Series           # turnover from names leaving coverage
    distance: pd.Series         # ½ Σ|x − x*| after trading: how far from target
    n_traded: pd.Series         # names with k > 0
    deferral: pd.DataFrame      # per month: mean fwd return of executed vs deferred buys / sells
    weights: pd.DataFrame       # long: date, ticker, weight (post-trade)

    @property
    def cum(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    def summary(self) -> dict:
        r = self.returns.dropna()
        ann = (1.0 + r.mean()) ** 12 - 1.0 if len(r) else np.nan
        vol = r.std() * np.sqrt(12) if len(r) > 1 else np.nan
        cum = self.cum
        dd = (cum / cum.cummax() - 1.0).min() if len(cum) else np.nan
        return {"ann_return": ann, "ann_vol": vol,
                "sharpe": ann / vol if vol and vol > 0 else np.nan,
                "max_drawdown": float(dd),
                "avg_turnover": float(self.turnover.mean()),
                "avg_forced": float(self.forced.mean()),
                "avg_distance": float(self.distance.mean()),
                "months": int(len(r))}


def _water_fill(need: np.ndarray, urg: np.ndarray, budget: float) -> np.ndarray:
    """Δexec = min(need, λ·urg) with λ chosen so Σ Δexec = min(budget, Σ need[urg>0])."""
    live = urg > 0
    cap = min(budget, float(need[live].sum()))
    if cap <= 0 or not live.any():
        return np.zeros_like(need)
    lo, hi = 0.0, float(need[live].sum() / urg[live].min()) + 1.0
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        tot = np.minimum(need, lam * np.where(live, urg, 0.0)).sum()
        if tot < cap:
            lo = lam
        else:
            hi = lam
    return np.minimum(need, hi * np.where(live, urg, 0.0))


def _cap_weights(g: pd.DataFrame) -> pd.Series:
    if "log_mcap" in g and g["log_mcap"].notna().sum() >= 2:
        w = np.exp(g["log_mcap"].astype(float)).fillna(0.0)
        if w.sum() > 0:
            return w / w.sum()
    return pd.Series(1.0 / len(g), index=g.index)


def run_pace(df: pd.DataFrame, mode: str = "modulated", budget: float = 0.08,
             n_q: int = 3, init: str = "cap", min_names: int = 6,
             rerank_every: int = 1) -> PaceResult:
    """
    df columns: date, ticker, dest, path, fwd_1m [, log_mcap].
    `dest`/`path` are the two composites; rows with NaN dest are out of coverage.

    `rerank_every`: months between destination re-ranks. A 12-month signal
    re-ranked monthly mostly shuffles names across the ntile boundary; holding
    the target for a quarter removes that churn at the source. Between
    re-ranks the target is carried forward (names that leave coverage drop
    out of it and the rest are rescaled to sum to one).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    x: pd.Series | None = None
    target: pd.Series | None = None
    month = 0
    rets, turn, forced, dist, ntr, defer, wrows = {}, {}, {}, {}, {}, [], []

    for t, g in d.sort_values("date").groupby("date"):
        g = g.drop_duplicates("ticker").set_index("ticker")
        cov = g.dropna(subset=["dest"])
        if len(cov) < min_names:
            continue
        # 1. target: equal weight over the top ntile of the destination score,
        #    refreshed every `rerank_every` months and carried forward between
        if target is None or month % max(int(rerank_every), 1) == 0:
            q = pd.qcut(cov["dest"].rank(method="first"), n_q, labels=False)
            top = cov.index[q == q.max()]
            target = pd.Series(0.0, index=cov.index)
            target[top] = 1.0 / len(top)
        else:
            target = target.reindex(cov.index).fillna(0.0)
            target = target / target.sum() if target.sum() > 0 else target
        month += 1

        if x is None:
            x = _cap_weights(cov) if init == "cap" else pd.Series(1.0 / len(cov), index=cov.index)

        # names that left coverage: liquidated outside the budget
        gone = x.index.difference(cov.index)
        forced[t] = float(x[gone].abs().sum()) if len(gone) else 0.0
        x = x.reindex(cov.index).fillna(0.0)
        if x.sum() > 0:
            x = x / x.sum()

        # 2–4. desired change, direction, centred path rank, urgency
        delta = target - x
        need = delta.abs().to_numpy()
        pr = cov["path"].rank(method="average")
        r = (2.0 * (pr - 1.0) / max(len(pr) - 1, 1) - 1.0).fillna(0.0)
        s = np.sign(delta) * r
        urg = np.clip(s.to_numpy(), 0.0, None)

        # 5. execution size
        if mode == "wholesale":
            ex = need
        elif mode == "uniform":
            tot = need.sum()
            ex = need * min(1.0, budget / tot) if tot > 0 else need * 0.0
        else:
            ex = _water_fill(need, urg, budget)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(need > 0, ex / need, 0.0)

        # 6. update and renormalise
        x_new = (x + k * delta).clip(lower=0.0)
        if x_new.sum() > 0:
            x_new = x_new / x_new.sum()

        fwd = cov["fwd_1m"].astype(float).fillna(0.0)
        rets[t] = float((x_new * fwd).sum())
        turn[t] = float(ex.sum())
        dist[t] = float(0.5 * (x_new - target).abs().sum())
        ntr[t] = int((k > 0).sum())

        # deferral diagnostic: did waiting pay?
        buys, sells = delta > 0, delta < 0
        done = k > 0
        defer.append({
            "date": t,
            "buy_exec": fwd[buys & done].mean() if (buys & done).any() else np.nan,
            "buy_wait": fwd[buys & ~done].mean() if (buys & ~done).any() else np.nan,
            "sell_exec": fwd[sells & done].mean() if (sells & done).any() else np.nan,
            "sell_wait": fwd[sells & ~done].mean() if (sells & ~done).any() else np.nan,
        })
        wrows.append(pd.DataFrame({"date": t, "ticker": x_new.index, "weight": x_new.values}))
        x = x_new[x_new > 0]

    idx = pd.DatetimeIndex(sorted(rets))
    return PaceResult(
        returns=pd.Series(rets).reindex(idx), turnover=pd.Series(turn).reindex(idx),
        forced=pd.Series(forced).reindex(idx), distance=pd.Series(dist).reindex(idx),
        n_traded=pd.Series(ntr).reindex(idx),
        deferral=pd.DataFrame(defer).set_index("date") if defer else pd.DataFrame(),
        weights=pd.concat(wrows, ignore_index=True) if wrows else pd.DataFrame())


def run_all(df: pd.DataFrame, budget: float = 0.08, n_q: int = 3,
            init: str = "cap", rerank_every: int = 1) -> dict[str, PaceResult]:
    return {m: run_pace(df, mode=m, budget=budget, n_q=n_q, init=init,
                        rerank_every=rerank_every) for m in MODES}


def term_structure(df: pd.DataFrame, comps: dict[str, pd.Series],
                   ks: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                   min_n: int = 10) -> pd.DataFrame:
    """
    Alpha term structure: for each composite, the Spearman IC against the
    return in month t+k ALONE (not cumulative), k = 1..12. A slow edge reads
    flat — its information is still paying a year out; a fast edge is
    front-loaded and dies within a few months. Columns: k, one per composite.
    `df` needs date, ticker, fwd_1m; consecutive calendar months per ticker.
    """
    from engine import _spearman
    d = df[["date", "ticker", "fwd_1m"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    for name, c in comps.items():
        d[name] = c.to_numpy()
    d = d.sort_values(["ticker", "date"]).reset_index(drop=True)
    d["_m"] = d["date"].dt.year * 12 + d["date"].dt.month
    g = d.groupby("ticker")
    rows = []
    for k in ks:
        r = g["fwd_1m"].shift(-(k - 1))
        m = g["_m"].shift(-(k - 1))
        d["_r"] = r.where(m == d["_m"] + (k - 1))
        row = {"k": k}
        for name in comps:
            sub = d[["date", name, "_r"]].dropna()
            ic = sub.groupby("date")[[name, "_r"]].apply(_spearman, name, "_r", min_n).dropna()
            row[name] = float(ic.mean()) if len(ic) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")


def coverage_benchmark(df: pd.DataFrame) -> pd.Series:
    """Equal-weight monthly return of everything in coverage."""
    d = df.dropna(subset=["dest"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.groupby("date")["fwd_1m"].mean()


def deferral_test(res: PaceResult) -> pd.DataFrame:
    """Executed-minus-deferred next-month return, buys and sells, with a t-stat.

    The mechanism claims deferred buys are about to be cheaper (executed >
    deferred) and deferred sells are still rising (deferred > executed)."""
    dfr = res.deferral
    rows = []
    for side, a, b, claim in (("Buys", "buy_exec", "buy_wait", "executed − deferred > 0"),
                              ("Sells", "sell_exec", "sell_wait", "deferred − executed > 0")):
        if dfr.empty or a not in dfr:
            continue
        diff = (dfr[a] - dfr[b]) if side == "Buys" else (dfr[b] - dfr[a])
        diff = diff.dropna()
        if len(diff) < 12:
            continue
        rows.append({"side": side, "claim": claim,
                     "mean_diff": float(diff.mean()),
                     "t_stat": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))),
                     "months": int(len(diff))})
    return pd.DataFrame(rows)
