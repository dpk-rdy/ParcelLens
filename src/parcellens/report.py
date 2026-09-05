"""Findings write-up and the HTML review dashboard."""

from __future__ import annotations

import pandas as pd

TOKENS = """
:root{
  --paper:#E9EBE7; --card:#F8F9F7; --ink:#1A1D1A; --muted:#6E736C;
  --rule:#C9CDC6; --primary:#1F4E79; --leak:#8E2436; --save:#3F6E4E;
  --kraft:#8C6D3F;
}
@media (prefers-color-scheme: dark){
  :root{ --paper:#16191A; --card:#1E2223; --ink:#E7EAE6; --muted:#9AA39C;
         --rule:#343A3A; --primary:#7FAAD4; --leak:#D77E8C; --save:#8CBE9C;
         --kraft:#C2A272; }
}
"""


def _money(v, k=True):
    if k and abs(v) >= 1_000_000:
        return f"${v/1_000_000:,.2f}M"
    if k and abs(v) >= 1000:
        return f"${v/1000:,.0f}k"
    return f"${v:,.0f}"


def hero_svg(spend, steps):
    """Invoiced spend, with the recoverable slice magnified underneath.

    The opportunity is a single-digit percentage of the bill, so splitting the
    full-width bar into its four sources produces four unreadable slivers. The
    top band keeps the true proportion, which is the honest picture, and the
    band below expands that slice to full width so the composition is legible.
    The trapezoid between them is what makes the magnification explicit rather
    than a second chart at a silently different scale.
    """
    W = 1000
    top_y, top_h = 40, 30
    bot_y, bot_h = 128, 40
    total = float(spend)
    opp = sum(v for _, v in steps)
    kept = total - opp

    w_kept = kept / total * W
    parts = [
        f'<rect x="0" y="{top_y}" width="{w_kept:.2f}" height="{top_h}" '
        f'fill="var(--primary)" opacity="0.88"/>',
        f'<rect x="{w_kept:.2f}" y="{top_y}" width="{W-w_kept:.2f}" '
        f'height="{top_h}" fill="var(--leak)"/>',
        # magnification bridge
        f'<path d="M {w_kept:.2f} {top_y+top_h} L {W} {top_y+top_h} '
        f'L {W} {bot_y} L 0 {bot_y} Z" fill="var(--leak)" opacity="0.10"/>',
        f'<line x1="{w_kept:.2f}" y1="{top_y+top_h}" x2="0" y2="{bot_y}" '
        f'stroke="var(--leak)" stroke-width="1" opacity="0.45"/>',
        f'<line x1="{W}" y1="{top_y+top_h}" x2="{W}" y2="{bot_y}" '
        f'stroke="var(--leak)" stroke-width="1" opacity="0.45"/>',
    ]

    head = (
        f'<text x="0" y="26" class="wf-head">{_money(total)} invoiced</text>'
        f'<text x="{W}" y="26" text-anchor="end" class="wf-head-r">'
        f'{_money(opp)} recoverable or avoidable, {opp/total:.1%}</text>')

    palette = ["var(--leak)", "var(--leak)", "var(--save)", "var(--kraft)"]
    x = 0.0
    labels = []
    for i, (name, val) in enumerate(steps):
        w = val / opp * W
        parts.append(
            f'<rect x="{x:.2f}" y="{bot_y}" width="{max(w-1.5,1.5):.2f}" '
            f'height="{bot_h}" fill="{palette[i % len(palette)]}"/>')
        # Alternate label rows so narrow segments still get a readable name.
        drop = 26 if i % 2 == 0 else 62
        cx = x + w / 2
        anchor, tx = "middle", cx
        if cx < 40:
            anchor, tx = "start", 0
        elif cx > W - 40:
            anchor, tx = "end", W
        labels.append(
            f'<line x1="{cx:.2f}" y1="{bot_y+bot_h}" x2="{cx:.2f}" '
            f'y2="{bot_y+bot_h+drop-20:.2f}" stroke="var(--rule)" stroke-width="1"/>'
            f'<text x="{tx:.2f}" y="{bot_y+bot_h+drop:.2f}" text-anchor="{anchor}" '
            f'class="wf-val">{_money(val)}</text>'
            f'<text x="{tx:.2f}" y="{bot_y+bot_h+drop+15:.2f}" text-anchor="{anchor}" '
            f'class="wf-lab">{name}</text>')
        x += w

    return (f'<svg viewBox="0 0 {W} 268" role="img" class="waterfall" '
            f'aria-label="Invoiced parcel spend with the recoverable share '
            f'magnified into its four sources">'
            f'{head}{"".join(parts)}{"".join(labels)}</svg>')


def _table(df: pd.DataFrame, cols, headers, fmts, align_right=None):
    align_right = align_right or set(range(1, len(cols)))
    head = "".join(
        f'<th class="{"num" if i in align_right else ""}">{h}</th>'
        for i, h in enumerate(headers))
    rows = []
    for _, r in df.iterrows():
        cells = []
        for i, (c, f) in enumerate(zip(cols, fmts)):
            v = f(r[c]) if f else r[c]
            cells.append(f'<td class="{"num" if i in align_right else ""}">{v}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def build_dashboard(ctx: dict) -> str:
    s = ctx["stats"]

    steps = [
        ("Billing errors recovered", s["recovery_usd"]),
        ("Service failure refunds", s["gsr_usd"]),
        ("Carrier and service mix", s["rateshop_savings"]),
        ("Box right-sizing", s["packaging_savings"]),
    ]
    hero = hero_svg(s["invoiced_spend"], steps)

    det_table = _table(
        ctx["detection"], ["exception_type", "injected", "recall", "precision"],
        ["Rule", "Seeded errors", "Recall", "Precision"],
        [lambda v: v.replace("_", " ").capitalize(), lambda v: f"{v:,}",
         lambda v: f"{v:.1%}" if pd.notna(v) else "n/a",
         lambda v: f"{v:.1%}" if pd.notna(v) else "n/a"])

    client_table = _table(
        ctx["clients"],
        ["client", "shipments", "cost_per_package", "accessorial_share",
         "on_time", "total_opportunity", "opportunity_pct_of_spend"],
        ["Client", "Shipments", "Cost / pkg", "Accessorials", "On time",
         "Opportunity", "% of spend"],
        [None, lambda v: f"{v:,}", lambda v: f"${v:,.2f}",
         lambda v: f"{v:.1%}", lambda v: f"{v:.1%}",
         lambda v: f"${v:,.0f}", lambda v: f"{v:.1%}"])

    mig_table = _table(
        ctx["migration"].head(6),
        ["current_service", "optimal_service", "shipments", "savings"],
        ["From", "To", "Shipments", "Savings"],
        [lambda v: v.replace("_", " "), lambda v: v.replace("_", " "),
         lambda v: f"{v:,}", lambda v: f"${v:,.0f}"])

    div = ctx["divisors"]
    div_rows = "".join(
        f"<li><strong>{k.replace('/', 'on')}</strong> — billed weights match a "
        f"divisor near {v['divisor_applied']} against a contracted "
        f"{v['contract_divisor']}, across {v['lines']:,} lines</li>"
        for k, v in sorted(div.items(), key=lambda kv: -kv[1]["lines"])[:4])

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parcel spend review {ctx['year']}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&display=swap" rel="stylesheet">
<style>
{TOKENS}
*{{box-sizing:border-box}}
html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans","Segoe UI",Helvetica,Arial,sans-serif;
  font-size:16px;line-height:1.55;font-feature-settings:"tnum" 1}}
.wrap{{max-width:1120px;margin:0 auto;padding:56px 28px 96px}}
header{{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:36px}}
h1{{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
  font-weight:600;font-size:clamp(30px,4.4vw,46px);line-height:1.06;margin:0 0 8px;
  letter-spacing:-0.012em;max-width:22ch}}
.sub{{color:var(--muted);font-size:15px;margin:0;max-width:72ch}}
section{{margin:0 0 52px}}
h2{{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
  font-weight:600;font-size:22px;margin:0 0 4px;letter-spacing:-0.004em}}
.lede{{color:var(--muted);font-size:14.5px;margin:0 0 20px;max-width:74ch}}
p{{max-width:74ch}}
.rule{{height:1px;background:var(--rule);border:0;margin:0 0 22px}}
.waterfall{{width:100%;height:auto;display:block;margin:6px 0 10px}}
.wf-head{{font-family:"IBM Plex Sans Condensed";font-size:19px;font-weight:600;fill:var(--ink)}}
.wf-head-r{{font-family:"IBM Plex Sans Condensed";font-size:19px;font-weight:600;fill:var(--primary)}}
.wf-val{{font-size:13px;font-weight:600;fill:var(--ink)}}
.wf-lab{{font-size:11.5px;fill:var(--muted)}}
figure{{margin:0 0 8px}}
figure svg{{width:100%;height:auto;display:block}}
figcaption{{color:var(--muted);font-size:13px;margin-top:6px;max-width:74ch}}
table{{width:100%;border-collapse:collapse;font-size:14px;margin:4px 0 8px}}
th{{text-align:left;font-weight:600;font-size:12.5px;color:var(--muted);
  padding:0 12px 7px 0;border-bottom:1px solid var(--ink)}}
td{{padding:8px 12px 8px 0;border-bottom:1px solid var(--rule)}}
td.num,th.num{{text-align:right;padding-right:0;font-variant-numeric:tabular-nums}}
th.num{{padding-right:0}}
tbody tr:last-child td{{border-bottom:0}}
.cols{{display:flex;gap:38px;align-items:flex-start}}
.cols>*{{flex:1 1 0;min-width:0}}
.note{{border-left:3px solid var(--primary);padding:2px 0 2px 16px;
  color:var(--muted);font-size:14px;max-width:70ch}}
ul{{padding-left:18px;max-width:74ch;font-size:14.5px;color:var(--muted)}}
li{{margin-bottom:6px}}
li strong{{color:var(--ink);font-weight:500}}
footer{{border-top:1px solid var(--rule);padding-top:18px;color:var(--muted);font-size:13px}}
@media (max-width:820px){{.cols{{display:block}}.cols>*+*{{margin-top:28px}}
  .wrap{{padding:36px 18px 64px}}}}
@media (prefers-reduced-motion:no-preference){{
  .waterfall rect{{transform-origin:left center;animation:grow .62s cubic-bezier(.2,.7,.3,1) both}}
  @keyframes grow{{from{{transform:scaleX(0)}}to{{transform:scaleX(1)}}}}
}}
</style></head>
<body><div class="wrap">

<header>
<h1>Where {_money(s['invoiced_spend'])} of parcel spend actually went</h1>
<p class="sub">{s['shipments']:,} shipments and {s['invoice_lines']:,} carrier invoice
lines across {s['carriers']} carriers and {s['clients']} clients, {ctx['year']}.
Every invoice line reconciled against the contracted rate, then priced for
recovery and re-optimized at a constant delivery promise.</p>
</header>

<section>
<h2>The gap between what was billed and what was owed</h2>
<p class="lede">Part of the bill is wrong, and part of it is right but avoidable.
Neither part needs a rate renegotiation to fix.</p>
{hero}
</section>

<section>
<h2>Billing integrity</h2>
<p class="lede">{s['exception_lines']:,} invoice lines failed reconciliation,
{s['exception_rate']:.2%} of all lines. Recoveries are priced at the disputable
amount, not the full line, except where the whole line is invalid.</p>
<figure>{ctx['svg']['recovery']}</figure>
</section>

<section>
<h2>How the exception rules hold up</h2>
<p class="lede">Errors were seeded into the invoice file at known rates, so the
detection logic can be scored rather than trusted. Recall is the share of
seeded errors the rules caught; precision is the share of flagged lines that
were real, which is what determines how much time an analyst wastes on
disputes that get rejected.</p>
<div class="cols">
<div>{det_table}</div>
<div>
<p class="note">Dimensional and scale-weight overcharges are the one pair the
rules cannot cleanly separate line by line. Both produce a billed weight above
contract, and on a large box some divisor reproduces almost any inflated
weight. The classifier resolves them at cluster level instead, learning the
divisor a carrier is actually applying:</p>
<ul>{div_rows}</ul>
<p class="note">Recovery dollars are unaffected by the split, since both types
are recovered. The distinction matters for the fix: a divisor problem is a
contract conversation, a scale problem is a calibration ticket.</p>
</div>
</div>
</section>

<section>
<h2>What a package costs, by carrier</h2>
<p class="lede">Cost per package is mostly a function of mix, so it is split
into transportation, accessorials and fuel. Accessorial share is the number
worth watching: it is the part of the bill that has nothing to do with moving
the box further.</p>
<figure>{ctx['svg']['carriers']}</figure>
</section>

<section>
<h2>Spend and leakage through the year</h2>
<figure>{ctx['svg']['trend']}</figure>
<figcaption>Leakage climbs into peak season, when volume, surcharges and
service failures all rise together and audit attention is thinnest.</figcaption>
</section>

<section>
<h2>Carrier and service mix</h2>
<p class="lede">Every shipment repriced across all seven services, holding the
delivery promise the customer already received. The unconstrained figure is
arithmetic; the executable plan caps the regional carrier at
{s['regional_cap']:.0%} of volume, which is what its network can absorb.</p>
<figure>{ctx['svg']['rateshop']}</figure>
<div class="cols">
<div>{mig_table}
<figcaption>Largest volume movements under the executable plan.</figcaption></div>
<div><p class="note">{s['air_to_ground_pkgs']:,} packages moved on guaranteed
air to a lane where regional ground already delivered within the same promised
transit. That is {_money(s['air_to_ground_savings'])} of air premium bought for
no delivery-date benefit, and it is a service-selection default rather than a
pricing problem.</p></div>
</div>
</section>

<section>
<h2>Dimensional weight</h2>
<p class="lede">{s['dim_billed_share']:.0%} of shipments are billed on
dimensional rather than scale weight, at a cost of
{_money(s['dim_penalty_spend'])}. Average boxes leave
{1-s['avg_cube_utilization']:.0%} of their cube empty. Right-sizing to the
smallest catalog box that still fits the item recovers
{_money(s['packaging_savings'])} without touching a carrier contract.</p>
<figure>{ctx['svg']['dim']}</figure>
</section>

<section>
<h2>Next year's rate card on this year's volume</h2>
<p class="lede">Repricing the same {s['shipments']:,} shipments on the announced
increases raises spend {s['gri_pct']:.1%}, against a headline ground increase of
{s['gri_headline']:.1%}. The gap is where the money is: accessorials and the
divisor move faster than the base rate, and neither appears in the headline.</p>
<figure>{ctx['svg']['gri']}</figure>
</section>

<section>
<h2>Method</h2>
<hr class="rule">
<ul>
<li><strong>Reconciliation</strong> — each invoice line repriced from the
manifest against the contracted card: billable weight, zone, discount,
accessorials, then fuel on the fuel-bearing subtotal.</li>
<li><strong>Tolerances</strong> — 2% or 25 cents on base, 0.4 points on the
fuel index. Below that, a variance is rounding and is not worth a dispute.</li>
<li><strong>Recovery pricing</strong> — the disputable delta, not the line
total, except for duplicates and never-shipped labels where the whole line is
invalid. Service failure refunds are held separate from billing errors because
they are valid charges that became refundable.</li>
<li><strong>Optimization</strong> — delivery promise held constant, so no
saving is bought with slower service. Capacity caps applied greedily on regret,
so a capped carrier gets the volume where its advantage is largest.</li>
<li><strong>Data</strong> — {s['shipments']:,} shipments generated with
carrier-realistic rate structures and billing errors seeded at known rates.
The rate values are synthetic; the tariff mechanics are not.</li>
</ul>
</section>

<footer>ParcelLens, generated {ctx['generated']}. Figures in USD.</footer>
</div></body></html>"""


def build_findings(ctx: dict) -> str:
    s = ctx["stats"]
    rec = ctx["recovery"]
    top = rec.iloc[0]
    return f"""# Parcel spend review, {ctx['year']}

## Summary

{s['shipments']:,} shipments moved for {s['clients']} clients across
{s['carriers']} carriers, invoiced at {_money(s['invoiced_spend'])} over
{s['invoice_lines']:,} invoice lines.

Reconciling every line against contract and repricing every shipment against
the full carrier set identifies **{_money(s['total_opportunity'])},
{s['opportunity_pct']:.1%} of invoiced spend**, split four ways:

| Source | Value | Nature |
| --- | ---: | --- |
| Billing errors | {_money(s['recovery_usd'])} | Recoverable from the carrier |
| Service failure refunds | {_money(s['gsr_usd'])} | Claimable, on a filing clock |
| Carrier and service mix | {_money(s['rateshop_savings'])} | Avoidable, executable within capacity |
| Box right-sizing | {_money(s['packaging_savings'])} | Avoidable, packaging change |

None of it depends on renegotiating a rate.

## What changed, why, and what it is worth

**Billing accuracy is {1-s['exception_rate']:.2%}.**
{s['exception_lines']:,} of {s['invoice_lines']:,} lines failed
reconciliation. The largest single exception is
{top['exception_type'].replace('_', ' ')} at {_money(top['recovery_usd'])}
across {top['lines']:,} lines.

**The divisor is being applied tighter than contract.** Billed weights on the
largest carrier cluster match a dimensional divisor well below the contracted
value. This is systemic rather than random, which makes it a contract
conversation rather than a dispute queue.

**Air is being bought where ground already arrives on time.**
{s['air_to_ground_pkgs']:,} packages shipped on guaranteed air into lanes where
regional ground meets the same promised date, worth
{_money(s['air_to_ground_savings'])}. This is a service-selection default, not
a rate problem.

**Dimensional weight costs {_money(s['dim_penalty_spend'])}.**
{s['dim_billed_share']:.0%} of volume bills on cube rather than scale weight,
and the average box is {1-s['avg_cube_utilization']:.0%} empty.

**Next year's card adds {s['gri_pct']:.1%} to unchanged volume** against a
{s['gri_headline']:.1%} headline. Accessorials and the divisor change carry
the difference.

## Recommendations

1. Work the exception queue in recovery order. The top three exception types
   carry most of the dollars and are the least ambiguous to dispute.
2. Raise the divisor discrepancy at contract level with supporting line counts
   rather than filing individual disputes.
3. Change the service-selection default so guaranteed air is not selected on
   lanes where ground meets the same date.
4. Add the two box sizes that close most of the right-sizing gap and set a
   cube-utilization floor on the pack line.
5. Model the rate change on your own mix before the effective date, and budget
   the effective rate rather than the headline.

## Reproducing

```
pip install -r requirements.txt
python run_pipeline.py --shipments {s['shipments']}
```

Outputs land in `outputs/`: this write-up, the review dashboard, the exception
register, both scorecards, and the optimization detail.
"""
