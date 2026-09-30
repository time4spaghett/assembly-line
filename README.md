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

restricted to the two consumer sectors, with positive 6-month momentum as a
screen. Cash-generative, profitable, clean-accounting, not over-expanding. At the
default 1-month horizon it scores IC **+0.021** (Newey-West t **+2.1**), a
**+5.4%** annualized top-minus-bottom spread and a **0.41** long-short Sharpe.
It is a 1-month signal: at 12 months the IC holds (+0.023) but t falls to +1.2.

Two notes the app states on the charts. The bars are a *geometric* mean of
overlapping windows, annualized — what the signal predicts at that horizon —
and track the compounded CAGR beside them closely at 1 month, diverging
somewhat at longer horizons because they answer a different question. And the
cumulative and long-short curves are always built from
non-overlapping 1-month returns, so the horizon selector moves the bars, IC and
t-stat but deliberately not those two charts.

Clear the rows and build your own; the table is the source of truth.

## Benchmarks

The ntile charts compare against a benchmark you choose in step 5 (Test):
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

## Base panel

Point-in-time S&P 500 constituents (quarterly membership snapshots,
no survivorship bias), monthly frequency, **1998 → Dec 2025**. Data on/after
**Jan 2026 is deliberately excluded** as an enforced out-of-sample holdout
(`OOS_CUTOFF` in `panel_build.py`).

24 raw features across the standard categories:

| Category | Features |
|---|---|
| Value | `btm`, `earn_yield`, `fcf_yield`, `sales_yield` |
| Quality | `roa`, `roe`, `gross_margin`, `op_margin`, `gpa`, `fcf_margin` |
| Safety | `leverage`, `accruals` (higher = worse — use negative weights) |
| Growth | `rev_gr_1y`, `rev_gr_3y`, `asset_gr` |
| Momentum | `mom_12_1`, `mom_6_1`, `mom_3m`, `ret_1m`, `high_52w` |
| Fundamental momentum | `earn_mom`, `margin_mom`, `rev_accel` |
| Size | `log_mcap` |

Forward simple returns at 1/3/6/12-month horizons are precomputed in the panel.
Raw values (not ranks) are stored so the app's transform step is meaningful.
Fundamentals join point-in-time on Sharadar `datekey` (filing availability date).

### Jitter layer

The shipped panel carries a multiplicative Gaussian noise layer: every numeric
value is perturbed by `v * eps`, with `eps ~ N(0, sigma)` clipped to a hard
**±0.05%** of the original value (sigma = cap/3, so the cap is a true 3-sigma
bound and ~99.7% of draws are unclipped). It is **seeded** (`20260901`), so the
same input always jitters the same way; NaNs stay NaN and exact zeros stay zero.

Effect on results is nil at this magnitude — a ±0.05% perturbation almost never
swaps two names' ranks, so the rank-based metrics are effectively unchanged.
Build without `--jitter` for the pristine panel.

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
| `ui.py` | Shared universe/constraint screen and chart styling |
| `learned.py` | Walk-forward classifier, cohorts, interaction scan |
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
