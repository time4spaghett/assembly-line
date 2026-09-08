# Edge Concierge

Build and test factor / edge candidates in the browser. Select features from a
panel, transform (rank / z-score / raw) and weight them into a composite, then
test it: Spearman IC with Newey-West t-stats, ntile portfolio returns
(longitudinal cumulative curves + annualized bars), long-short spread, and
per-sector breakdown.

The analysis engine is ported from `base-equity-edge` (numpy/pandas IC sweep
with Bartlett-kernel Newey-West correction) — no alphalens dependency.

## Run

```bash
pip install -r requirements.txt
streamlit run main.py
```

Hosting: push this repo (including `data/base_panel.parquet`, ~20 MB) to GitHub
and deploy on [Streamlit Community Cloud](https://share.streamlit.io) — no other
infrastructure needed. To enable the natural-language edge builder, set
`ANTHROPIC_API_KEY` in the environment or in `.streamlit/secrets.toml`.

## The edge it opens on

The app loads a worked example rather than an empty form:

```
edge = rank(fcf_yield) + 0.5·rank(gpa) − 0.5·rank(accruals) − 0.5·rank(asset_gr)
```

excluding Communication Services. Cash-generative, profitable, clean-accounting,
not over-expanding. At the default 1-month horizon it scores IC **+0.018**
(Newey-West t **+3.23**), a **+7.8%** annualized top-minus-bottom spread and a
**0.81** long-short Sharpe; the quintile ladder runs **7.0% → 14.8%** (bars) or
**5.3% → 12.9%** compounded. It holds at 12 months too (IC +0.052, t +2.79), and
stays positive in every era of the sample — weakest post-2016, as value generally
has been. Communication Services is dropped because it is the only sector where
the signal fails outright (IC −0.003).

Two notes the app states on the charts. The bars are an *arithmetic* mean of
overlapping windows — what the signal predicts — and sit above the compounded
CAGR beside them by roughly half the variance; that gap is widest for the most
volatile ntile. And the cumulative and long-short curves are always built from
non-overlapping 1-month returns, so the horizon selector moves the bars, IC and
t-stat but deliberately not those two charts.

Clear the rows and build your own; the table is the source of truth.

## Benchmarks

The ntile charts compare against a benchmark you choose in the sidebar:
equal-weight universe (default), cap-weighted universe, one of seven index ETFs
(SPY, RSP, MDY, IWM, IWD, IWF, QQQ), or a column in your own panel.

The universe options are computed live from whatever your filters leave in. The
index references are **precomputed** into `data/benchmarks.parquet` (~28 KB) by
`benchmarks_build.py`, which is the only thing that touches the network — the app
itself never calls a data provider, which is what keeps it trivially hostable.
Refresh them with:

```bash
python benchmarks_build.py
```

Series are stored as *forward* one-month returns to match the panel's `fwd_1m`
convention, and truncated at the same Jan-2026 out-of-sample boundary.

## Saving a run

**Save run** sits beside the results tabs and produces **one self-contained HTML
file** — the spec, the universe it was measured over, the metrics and all five
charts, in a single document that opens in any browser with no internet. The
charting library is inlined, which is most of its ~5 MB; the run's own data is a
few KB.

Under **⋯** beside it: the same report in a ~10 KB variant that loads the library
from a CDN, the run record as JSON, and composite scores as CSV.

Filenames are timestamped and derived from the heaviest legs, e.g.
`20260901-141530_fcfyield-gpa-accruals_1m.html`. The timestamp is pinned to the
run configuration rather than the render, so it reads as when the run was
produced and only changes when an input does.

## Neural Edge

The third tool fits a **Gu–Kelly–Xiu (2020)**-style return-forecasting net to
the base panel — their characteristic transform (per-month cross-sectional
ranks mapped to [−1, 1], missing → 0), their architecture pyramid with **NN3
(32·16·8, ReLU)** as the default and best performer, and a seed ensemble —
under the same expanding walk-forward protocol as the Learned Edge, so every
prediction is out-of-sample. The default target is the **return over the
equal-weight market** rather than the raw return — Chen, Hanauer & Kalsbach
(2024, "Design choices, machine learning, and the cross-section of stock
returns") find the target variable is among the largest of the design levers,
and market-adjusting strips the common component a cross-sectional model
cannot time. A toggle restores the raw-return target. Reported the GKX way: OOS R² against a zero
forecast (their best models score ~0.4% monthly; small positive numbers are
the win) plus the app's usual monthly IC with Newey-West t. The prediction
rank exports to the Edge Concierge as `neural_tilt`, exactly like the learned
tilt.

Divergences from the paper, stated rather than hidden: L2 rather than L1
penalty, a random rather than chronological early-stopping split (within the
training window only), and no batch norm.

Defaults are tuned on this panel under the walk-forward (selection on the
2007–16 OOS half, confirmation on 2017–25): horizon 12m (fundamentals-heavy
features carry ~8× the 1-month IC), NN3 (capacity swept from 2 neurons to
128·64·32 — monotone up to NN3, flat-to-worse beyond), α = 1e-2, 5 seeds. At
those defaults the net scores IC +0.042 (NW t +2.0), a +6.2% annualized
quintile spread and 0.65 long-short Sharpe — edging the app's hand-built
default edge, and blending with it lifts the pair to IC +0.047 / Sharpe 0.71,
i.e. the net contributes orthogonal information rather than diluting.

### Concept alignment (TCAV)

The tab's second half asks how aligned the net's forecasts are with a
*concept*, via Testing with Concept Activation Vectors (Kim et al., 2018). A
concept is a set of stock-months, defined four ways:

- a **preset** (high profitability, deep value, momentum winners, …),
- **panel rules** — quantile conditions built on the panel's own features
  (`gpa` in the top 20% of each month AND …),
- an uploaded **company–date CSV** — the concept by demonstration, e.g. every
  quarterly holding of a fund whose philosophy you want to test against, or
- **plain English** (needs `ANTHROPIC_API_KEY`, like the Edge Concierge's
  sketch box) — "returns above cost of capital with room to reinvest" is
  drafted into quantile rules, with the proxy mapping and what the panel
  *cannot* express stated alongside, and a copy-to-rule-builder button so the
  draft can be corrected by hand.

A linear probe on a hidden layer's activations separates concept rows from
date-matched random rows; its unit normal is the CAV. The exact gradient of
the predicted return w.r.t. that layer (hand-rolled numpy over the sklearn
weights — no torch) is dotted with the CAV; the **TCAV score** is the fraction
of stock-months where leaning toward the concept raises the forecast. 0.5 is
orthogonal. Guards, both from the paper: the CAV's held-out accuracy must beat
chance (else the concept isn't encoded at that layer and the score is noise),
and the score distribution must separate from same-size random pseudo-concepts
(two-sample t-test). The probe runs at the inputs or any hidden layer *except
the last* — above the last sits only a linear readout, so the gradient there
is the same for every row and the score degenerates to 0 or 1.

TCAV interrogates a full-history fit, mirroring the Learned Edge convention:
walk-forward for honest scores, one descriptive fit — never scored from — for
what the model believes.

An alignment run also produces two per-name series, combined in the **Best
ideas within the concept** view: `expression` (how strongly a name embodies
the concept — its projection onto the CAVs, pct-ranked) and `align` (the share
of CAVs under which pushing the name further toward the concept raises its
forecast). Crossed with the walk-forward `neural_tilt`, the view ranks concept
members by the out-of-sample forecast at a chosen date: the return leg is the
only OOS-protocol number, and `align` diagnoses whether the model likes a name
*because of* the philosophy or despite it.

## Factor Edge

The fourth tool is Gu–Kelly–Xiu's **conditional autoencoder** (2021, *Journal
of Econometrics*): returns are constrained to a factor structure
`r = β(z)′f`, where a small beta network maps characteristics to K factor
loadings and the K latent factors are distilled linearly from the month's
characteristic-managed portfolio returns (equal-weight market first, then one
rank-weighted portfolio per feature). Expected returns can only arise as
compensation for factor exposure — a near-no-arbitrage constraint that
regularizes far harder than anything in the unconstrained Neural Edge, which
is why this family tests well on small panels.

The trainer is ~200 lines of hand-rolled numpy (`autoencoder.py`): joint
minibatch Adam over both networks (one month per step), **chronological**
early stopping on the last 15% of training months, optional Huber loss
(residual influence clipped at 2× the target's sd), seed ensembling, and the
repo's expanding walk-forward. The smoke test verifies the analytic gradients
against finite differences and that a planted factor structure is recovered.
No torch: the networks are tiny enough that autograd is the only thing it
would add.

Reported OOS: **total R²** (fit against the month's realized factors —
observable ex post, since factors are portfolio returns; the asset-pricing
test) and **predictive R² / IC** (forecast via estimated premia β·λ; the
honest prediction test). The gap between the two is the difference between
describing risk and finding alpha. Interpretation from a full-history fit:
factor premia and cumulative paths, plus per-characteristic loading curves.
The β·λ rank exports to the Edge Concierge as `ca_tilt` — labeled for what it
is: expected return earned as factor-risk compensation, not alpha.

## Base panel

Point-in-time S&P 500 constituents (quarterly membership snapshots,
no survivorship bias), monthly frequency, **1998 → Dec 2025**. Data on/after
**Jan 2026 is deliberately excluded** as an enforced out-of-sample holdout
(`OOS_CUTOFF` in `panel_build.py`).

28 raw features across the standard categories:

| Category | Features |
|---|---|
| Value | `btm`, `earn_yield`, `fcf_yield`, `sales_yield` |
| Quality | `roa`, `roe`, `gross_margin`, `op_margin`, `gpa`, `fcf_margin` |
| Safety | `leverage`, `accruals` (higher = worse — use negative weights) |
| Growth | `rev_gr_1y`, `rev_gr_3y`, `asset_gr` |
| Momentum | `mom_12_1`, `mom_6_1`, `mom_3m`, `ret_1m`, `high_52w` |
| Fundamental momentum | `earn_mom`, `margin_mom`, `rev_accel` |
| Risk | `vol_1m`, `vol_12m`, `beta_12m`, `max_ret_1m` |
| Size | `log_mcap` |

The risk block is computed from daily adjusted closes (std of daily returns
over 21 and 252 trading days, rolling 252-day beta against the equal-weight
universe, and the largest single-day return in the trailing month), sampled at
month-end. These are the price-based features that dominate the GKX importance
rankings; turnover and Amihud illiquidity would join them but the source cache
carries no volume data, and net share issuance is left out because as-reported
share counts read splits as issuance.

Forward simple returns at 1/3/6/12-month horizons are precomputed in the panel.
Raw values (not ranks) are stored so the app's transform step is meaningful.
Fundamentals join point-in-time on Sharadar `datekey` (filing availability date).

### Jitter layer

The shipped panel carries a multiplicative Gaussian noise layer: every numeric
value is perturbed by `v * eps`, with `eps ~ N(0, sigma)` clipped to a hard
**±0.05%** of the original value (sigma = cap/3, so the cap is a true 3-sigma
bound and ~99.7% of draws are unclipped). It is **seeded** (`20260901`), so the
same input always jitters the same way; NaNs stay NaN and exact zeros stay zero.

Effect on results is nil at this magnitude — the default edge scores
IC +0.0515 / t +2.79 / spread +4.5% / Sharpe 0.81 both with and without it. The
pristine panel is kept alongside as `data/base_panel_clean.parquet`.

Regenerate with `--jitter` (optionally `--jitter-rel` / `--seed`):

```bash
python panel_build.py --jitter --jitter-rel 0.0005 --seed 20260901
```

Rebuild with `python panel_build.py` (requires the local Sharadar caches — see
the argparse defaults for paths).

## Data handling

Uploaded CSVs go to the app server's **memory only**, keyed to the uploading
session (Streamlit holds them in `dict[session_id][file_id]`; the parsed frame
lives in that session's state). Nothing is written to disk, so nothing survives
a closed tab, a restart, or reaches the repo. Users cannot see each other's data.

The app itself is open-access when deployed on Streamlit Community Cloud —
anyone with the link can use it. Fine for public or your own data; for client
data, run locally or self-host behind your own auth.

## Custom CSV

Upload a long-format CSV (one row per security per date) and map the security-id
and date columns; sector/industry columns are optional. Forward returns come from
one of: a price column (computed at 1/3/6/12m from monthly closes), an explicit
forward-return column, or joined from the base panel by ticker. Every other
numeric column becomes a selectable feature. `data/sample_custom.csv` is an
example.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point and tool navigation |
| `tools/edge_concierge.py` | Build and test an edge by hand or from English |
| `tools/learned_edge.py` | Learn a style from holdings, emit a tilt |
| `tools/neural_edge.py` | GKX-style net + TCAV concept alignment |
| `tools/factor_edge.py` | Conditional autoencoder: latent factors, premia, loadings |
| `autoencoder.py` | Numpy CA trainer: joint Adam, Huber, walk-forward |
| `ui.py` | Shared universe/constraint screen and chart styling |
| `learned.py` | Walk-forward classifier, cohorts, interaction scan |
| `neural.py` | GKX net, walk-forward protocol, numpy activations/gradients |
| `concepts.py` | Concept definitions, CAVs, TCAV scoring and significance |
| `engine.py` | Transforms, composite construction, IC / ntile / sector analysis |
| `data_io.py` | Base panel loading + custom CSV normalization |
| `nl.py` | Optional natural-language → edge-spec (Claude API) |
| `report.py` | Self-contained HTML run report (pure rendering, no Streamlit) |
| `panel_build.py` | One-off base panel builder (not needed to run the app) |
| `benchmarks_build.py` | One-off reference-ETF fetch (the only network call) |

## Methodology notes

- Composite = weighted sum of per-date cross-sectional transforms, re-ranked to
  [0, 1] per date. Sector-neutral mode transforms within (date, sector).
- IC = Spearman rank correlation per month; t-stat is Newey-West with lag =
  horizon months − 1 (overlapping-observation correction).
- Ntile cumulative curves and the long-short curve use non-overlapping 1-month
  forward returns (equal-weight, monthly rebalance, gross of costs). Ntile
  annualized bars use the selected horizon.
