# Parcel spend review, 2025

## Summary

250,000 shipments moved for 6 clients across
4 carriers, invoiced at $5.20M over
250,879 invoice lines.

Reconciling every line against contract and repricing every shipment against
the full carrier set identifies **$410k,
7.9% of invoiced spend**, split four ways:

| Source | Value | Nature |
| --- | ---: | --- |
| Billing errors | $74k | Recoverable from the carrier |
| Service failure refunds | $17k | Claimable, on a filing clock |
| Carrier and service mix | $234k | Avoidable, executable within capacity |
| Box right-sizing | $85k | Avoidable, packaging change |

None of it depends on renegotiating a rate.

## What changed, why, and what it is worth

**Billing accuracy is 95.90%.**
10,279 of 250,879 lines failed
reconciliation. The largest single exception is
unearned address correction at $24k
across 1,159 lines.

**The divisor is being applied tighter than contract.** Billed weights on the
largest carrier cluster match a dimensional divisor well below the contracted
value. This is systemic rather than random, which makes it a contract
conversation rather than a dispute queue.

**Air is being bought where ground already arrives on time.**
6,375 packages shipped on guaranteed air into lanes where
regional ground meets the same promised date, worth
$128k. This is a service-selection default, not
a rate problem.

**Dimensional weight costs $323k.**
53% of volume bills on cube rather than scale weight,
and the average box is 46% empty.

**Next year's card adds 7.3% to unchanged volume** against a
5.9% headline. Accessorials and the divisor change carry
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
python run_pipeline.py --shipments 250000
```

Outputs land in `outputs/`: this write-up, the review dashboard, the exception
register, both scorecards, and the optimization detail.
