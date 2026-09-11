"""
Self-contained HTML report for a single edge run.

Kept out of the page because it is pure rendering — no Streamlit, no globals — so
it can be generated and inspected outside the app.
"""
from __future__ import annotations

import numpy as np

HORIZON_LABELS = {"fwd_1m": "1 month", "fwd_3m": "3 months",
                  "fwd_6m": "6 months", "fwd_12m": "12 months"}


def _fmt_pct(x, dp=1):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{dp}%}"


def _validation_html(v: dict | None) -> str:
    """The holdout result, stated before the charts rather than after them."""
    if not v:
        return ""
    flagged = v["total"] - v["passed"]
    tone = {"consistent": "#0ca30c", "partial": "#e3b505",
            "weak": "#e3b505", "inconsistent": "#c4302b"}[v["verdict"]]
    rows = "".join(
        f"<tr><td>{n}</td><td style='color:{'#0ca30c' if ok else '#c4302b'}'>"
        f"{'pass' if ok else 'flag'}</td><td class='num'>{d}</td></tr>"
        for n, ok, d in v["checks"])
    note = f"<p class='note'>{v['note']}</p>" if v.get("note") else ""
    return f"""<h2>Validation</h2>
<p class="verdict" style="border-left:3px solid {tone}">
<b>{flagged} of {v['total']} consistency checks flag a problem.</b>
{v['headline']} In-sample {v['months_is']} months, holdout {v['months_oos']} months.</p>
{note}
<table><tr><th>Check</th><th></th><th>In-sample &rarr; holdout</th></tr>{rows}</table>
<p class="note">Fixed thresholds, applied identically every run. Sign agreement
is the gate: an effect that reverses out of sample is a different effect, not a
weaker one.</p>"""


def build_report(sig: str, title: str, record: dict, figs_html: list[str],
                 sector_table: str, caveats: list[str],
                 validation: dict | None = None) -> str:
    """
    One self-contained HTML file: spec, metrics, charts.

    plotly.js is inlined once rather than pulled from a CDN, so the report still
    renders years later on a machine with no network — which is the point of
    saving it. That is most of the file size; the run's own data is a few KB.
    """
    r, sp, un, te = record["results"], record["spec"], record["universe"], record["test"]
    rows = "".join(
        f"<tr><td><code>{f['feature']}</code></td><td>{f['transform']}</td>"
        f"<td class='num'>{f['weight']:g}</td></tr>" for f in sp["features"])
    cons = "".join(
        f"<tr><td><code>{c['left']}</code></td><td>{c['op']}</td>"
        f"<td><code>{c['right']}</code></td></tr>" for c in sp["constraints"])
    ntiles = "".join(
        f"<tr><td>{q}</td><td class='num'>{_fmt_pct(v)}</td>"
        f"<td class='num'>{_fmt_pct(r['ntile_cagr'].get(q))}</td></tr>"
        for q, v in r["ntile_ann_geometric"].items())
    screen = record.get("screen")
    screen_html = ""
    if screen:
        screen_html = (
            f"<p class='note'><b>Screen</b> keeps {screen['frac_kept']:.0%} of rows, "
            f"{screen['names_kept']:,} names; thinnest month "
            f"{screen['min_names_per_date']}. Ranks are computed among survivors "
            f"only.</p>")
    # Universe note sits with the formula, not buried in the spec table below:
    # a result is meaningless without knowing what it was measured over.
    _sect = (un["sectors"] if isinstance(un["sectors"], str)
             else f"{len(un['sectors'])} sectors")
    _ind = (un["industries"] if isinstance(un["industries"], str)
            else f"{len(un['industries'])} industries")
    universe_bits = [
        f"<b>{record['panel']}</b>",
        f"{un['securities']:,} securities · {un['dates']:,} months",
        f"{un['years'][0]}–{un['years'][1]}",
        _sect if _sect != "all" else "all sectors",
    ]
    if _ind != "all":
        universe_bits.append(_ind)
    universe_bits += [
        f"{HORIZON_LABELS.get(te['horizon'], te['horizon'])} horizon",
        f"{te['ntiles']} ntiles",
        f"vs {te['benchmark']}",
    ]
    if sp["sector_neutral"]:
        universe_bits.append("sector-neutral")
    universe_html = ("<p class='universe'>Universe &nbsp;"
                     + " &nbsp;·&nbsp; ".join(universe_bits) + "</p>")

    kpis = [("Mean IC", f"{r['mean_ic']:.4f}"), ("Newey-West t", f"{r['nw_t_stat']:.2f}"),
            ("IC IR", f"{r['ic_ir']:.2f}"),
            ("Top−bottom spread", _fmt_pct(r["spread_ann"])),
            ("L/S Sharpe", f"{r['ls_sharpe']:.2f}")]
    kpi_html = "".join(f"<div class='kpi'><span>{k}</span><strong>{v}</strong></div>"
                       for k, v in kpis)
    caveat_html = "".join(f"<li>{c}</li>" for c in caveats)
    css = """
    :root{--s:#fcfcfb;--p:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;
          --grid:#e1e0d9;--line:#c3c2b7;--blue:#2a78d6}
    *{box-sizing:border-box}
    body{margin:0;background:var(--p);color:var(--ink);
         font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
    .wrap{max-width:1080px;margin:0 auto;padding:40px 24px 64px}
    h1{font-size:28px;margin:0 0 4px} h2{font-size:17px;margin:36px 0 10px}
    .sub{color:var(--ink2);margin:0 0 24px}
    code{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:13px}
    .formula{background:var(--s);border:1px solid var(--grid);border-radius:8px;
             padding:14px 16px;margin:0 0 22px;font-family:ui-monospace,monospace;
             font-size:14px;overflow-x:auto;white-space:nowrap}
    .kpis{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 8px}
    .kpi{background:var(--s);border:1px solid var(--grid);border-radius:8px;
         padding:12px 16px;min-width:132px}
    .kpi span{display:block;color:var(--mut);font-size:12px}
    .kpi strong{font-size:24px;font-weight:600}
    table{border-collapse:collapse;width:100%;background:var(--s);
          border:1px solid var(--grid);border-radius:8px;overflow:hidden}
    th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--grid);
          font-size:13px}
    th{color:var(--mut);font-weight:500} tr:last-child td{border-bottom:0}
    td.num{text-align:right;font-variant-numeric:tabular-nums}
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
    .note{color:var(--ink2);font-size:13px}
    .verdict{background:var(--s);border:1px solid var(--grid);padding:11px 14px;
             margin:0 0 12px;font-size:14px}
    .universe{color:var(--ink2);font-size:13px;margin:-12px 0 20px;
              padding:10px 14px;background:var(--s);border:1px solid var(--grid);
              border-radius:8px}
    .universe b{color:var(--ink)}
    ul.caveats{color:var(--ink2);font-size:13px;padding-left:18px}
    .chart{background:var(--s);border:1px solid var(--grid);border-radius:8px;
           padding:6px;margin:0 0 18px}
    footer{color:var(--mut);font-size:12px;margin-top:40px;
           border-top:1px solid var(--grid);padding-top:14px}
    @media(max-width:820px){.cols{grid-template-columns:1fr}}
    """
    charts = "".join(f"<div class='chart'>{h}</div>" for h in figs_html)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{css}</style></head><body><div class="wrap">
<h1>Edge Concierge</h1>
<p class="sub">{title}</p>
<div class="formula">{sp['formula']}</div>
{universe_html}
<div class="kpis">{kpi_html}</div>
{screen_html}
{_validation_html(validation)}
<h2>Charts</h2>{charts}
<h2>Specification</h2>
<div class="cols">
  <div><table><tr><th>Feature</th><th>Transform</th><th>Weight</th></tr>
  {rows}</table>
  {"<table style='margin-top:12px'><tr><th>Constraint</th><th></th><th></th></tr>"
   + cons + "</table>" if cons else ""}</div>
  <div><table>
    <tr><th>Panel</th><td>{record['panel']}</td></tr>
    <tr><th>Sectors</th><td>{un['sectors'] if isinstance(un['sectors'], str)
                             else ', '.join(un['sectors'])}</td></tr>
    <tr><th>Industries</th><td>{un['industries'] if isinstance(un['industries'], str)
                                else ', '.join(un['industries'])}</td></tr>
    <tr><th>Years</th><td>{un['years'][0]}–{un['years'][1]}</td></tr>
    <tr><th>Universe</th><td>{un['securities']:,} securities · {un['dates']:,} months</td></tr>
    <tr><th>Horizon</th><td>{HORIZON_LABELS.get(te['horizon'], te['horizon'])}</td></tr>
    <tr><th>Ntiles</th><td>{te['ntiles']}</td></tr>
    <tr><th>Benchmark</th><td>{te['benchmark']}</td></tr>
    <tr><th>Sector-neutral</th><td>{'yes' if sp['sector_neutral'] else 'no'}</td></tr>
  </table></div>
</div>
<h2>Ntile returns</h2>
<div class="cols"><div><table>
<tr><th>Ntile</th><th>Mean fwd (ann.)</th><th>Compounded CAGR</th></tr>{ntiles}
<tr><td><b>{te['benchmark']}</b></td><td class="num">{_fmt_pct(r['benchmark_ann'])}</td>
<td class="num">{_fmt_pct(r['benchmark_cagr'])}</td></tr>
</table></div><div>{sector_table}</div></div>
<h2>Reading this</h2><ul class="caveats">{caveat_html}</ul>
<footer>Generated {record['exported_at']} · gross of costs and taxes ·
research output, not investment advice.</footer>
</div></body></html>"""
