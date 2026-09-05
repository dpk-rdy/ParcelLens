# ParcelLens

**Parcel spend audit and transportation optimization engine.**

Reconciles 250,000 parcel shipments against carrier invoices, prices the
recoverable billing errors, re-optimizes carrier and service selection at a
constant delivery promise, and models next year's rate card against the current
mix.

On the reference dataset it identifies **$410K of recoverable or avoidable cost
on $5.20M of invoiced spend (7.9%)**, and answers the four questions a parcel
analyst is actually paid to answer: what changed, why it changed, what it cost,
and what to do about it.

```
Invoiced spend              5,196,216
  Billing errors recovered     73,533     1.41%   audit
  Service failure refunds      17,145     0.33%   guaranteed-service claims
  Carrier and service mix     234,033     4.50%   rate shop within capacity
  Box right-sizing             85,011     1.64%   dimensional weight
  ------------------------------------------------------------------
  Total opportunity           409,723     7.89%

Billing accuracy               95.90%
Shipments billed on DIM         53.5%
Next-year rate impact           7.30%   against a 5.90% headline increase
```

---

## What it does

**Reconciliation.** Every invoice line is repriced from the shipment manifest
against the contracted rate card: billable weight, zone, negotiated discount,
accessorials, then fuel on the fuel-bearing subtotal. The variance is
classified into ten exception types and priced at the disputable amount rather
than the line total.

| Exception | What it is |
| --- | --- |
| `duplicate_billing` | Same tracking number billed more than once |
| `manifested_not_shipped` | Label billed with no outbound scan |
| `dim_misapplication` | Billed on a divisor tighter than contract |
| `weight_discrepancy` | Billed above the pack-line scale weight |
| `rate_variance` | Negotiated discount not fully applied |
| `unearned_residential` | Residential fee on a commercial delivery |
| `unearned_address_correction` | Correction fee with no correction on file |
| `unearned_peak_surcharge` | Peak fee outside the peak window |
| `fuel_percentage_error` | Fuel billed above the published index |
| `service_failure_refund` | Guaranteed service delivered late |

**Rate shop.** Each shipment is repriced across all seven services and moved to
the cheapest one that still meets the delivery promise the customer already
received. A materiality floor stops the optimizer proposing carrier moves worth
pennies, and capacity caps keep the regional carrier inside what its network
can absorb, so the output is a plan rather than arithmetic.

**Packaging.** Measures how much volume is billed on cube rather than scale
weight, how full the boxes are, and what right-sizing to the smallest catalog
box that still fits the item would save.

**Rate change model.** Reprices the same volume on next year's card and splits
the increase into base rate, dimensional divisor, accessorials and fuel, so the
effective increase on the real mix is separable from the headline number.

---

## Results worth reading

**Detection is scored, not asserted.** Billing errors are seeded into the
invoice file at known rates, so the exception rules can be measured. Seven of
nine rules run at 100% recall and 100% precision.

The two that do not are the interesting ones. A dimensional overcharge and a
scale-weight overcharge both produce a billed weight above contract, and on a
large box some divisor reproduces almost any inflated weight by coincidence, so
they cannot be separated line by line. Testing each line against every
plausible divisor finds a match roughly 60% of the time whether or not one
exists.

The classifier resolves them at cluster level instead. For each carrier and
service it counts how many overbilled lines each candidate divisor would
explain, estimates the coincidence background from the quiet part of the sweep,
and treats the resulting plateau as one hypothesis. On the reference data it
recovers the seeded divisor correctly:

```
NationalExpress / NEX_GROUND     divisor ~107 applied against a contracted 139
ContinentalParcel / CPX_GROUND   divisor ~107 applied against a contracted 139
```

Joint recall across both exception types is above 99% — no overcharge escapes —
while line-level attribution between them tops out near 77%. The recovery
dollars are unaffected, because both types are recovered. The distinction
matters for the fix: a divisor problem is a contract conversation, a scale
problem is a calibration ticket.

**Air is bought where ground already arrives on time.** Thousands of packages
ship on guaranteed air into lanes where regional ground meets the same promised
date. That is a service-selection default, not a pricing problem, and it is the
single largest line in the optimization.

**The headline rate increase understates the bill.** Accessorials and the
divisor change move faster than the base rate, so 5.90% announced lands at
7.30% on this mix. Neither driver appears in the announcement.

---

## Running it

```bash
pip install -r requirements.txt

python run_pipeline.py                    # full run, ~10 seconds
python run_pipeline.py --shipments 50000  # smaller
python run_queries.py                     # the SQL analysis pack
python run_queries.py 03                  # one query
pytest tests -q
```

Outputs land in `outputs/`:

| File | Contents |
| --- | --- |
| `dashboard.html` | Review dashboard, self-contained |
| `findings.md` | Written findings and recommendations |
| `exception_register.csv` | Every flagged line, ranked by recovery |
| `carrier_scorecard.csv` | Cost per package, accessorial share, on-time, billing accuracy |
| `client_scorecard.csv` | Cost per package and opportunity by client |
| `service_migration.csv` | Where volume moves under the executable plan |
| `detection_scorecard.csv` | Recall and precision per exception rule |
| `charts/` | Every figure as SVG and PNG |

---

## Layout

```
src/parcellens/
  ratecards.py   rate tables, accessorials, fuel index, DIM rules, GRI
  rating.py      billable weight and charge construction, vectorized
  generate.py    shipment manifest and invoice file with seeded errors
  audit.py       reconciliation, exception rules, recovery pricing, scoring
  rateshop.py    least-cost service selection under SLA and capacity
  packaging.py   cube utilization and box right-sizing
  scorecard.py   carrier and client scorecards, rate change model
  charts.py      figures
  report.py      dashboard and findings
sql/             analysis queries, run against the parquet output via DuckDB
tests/           22 tests over rating, audit, optimization and rate modeling
```

The rating engine is fully vectorized: rate-shopping 250,000 shipments across
seven services is 1.75 million quotes, and the whole pipeline runs in about ten
seconds.

---

## On the data

The shipments, invoices and rate values are **synthetic**. They are generated
to reproduce the structure of real parcel tariffs — zone-banded base charges,
escalating per-pound cost, dimensional divisors, percentage fuel surcharge on a
fuel-bearing subtotal, accessorial schedules, negotiated discount tiers — and
no carrier's published or contracted rates are reproduced.

The tariff mechanics, the exception taxonomy, and the analytical methods are
real. Generating the data is also what makes the detection logic measurable: on
a real invoice file nobody knows the true error count, so an audit engine can
only be trusted, never scored.
