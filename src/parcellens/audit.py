"""
Invoice audit engine.

Reconciles every carrier invoice line against the contracted charge implied by
the manifest, classifies the variance into a named exception type, and prices
the recovery. Detection uses only fields an analyst would actually hold: the
manifest, the rate card, the fuel table and the delivery scan. The injected
ground-truth label is used *afterwards*, and only to score the rules.

Exception taxonomy
------------------
duplicate_billing            same tracking number billed more than once
manifested_not_shipped       label billed with no outbound scan
dim_misapplication           billed on a tighter divisor than contract
weight_discrepancy           billed weight above scale weight, not DIM-explained
rate_variance                negotiated discount not fully applied to base
unearned_residential         residential fee on a commercial delivery
unearned_address_correction  correction fee with no correction on file
unearned_peak_surcharge      peak fee outside the peak window
fuel_percentage_error        fuel billed above the published index
service_failure_refund       guaranteed service delivered late (GSR claim)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ratecards as rc

# Tolerances keep rounding noise out of the exception queue. A parcel audit
# that flags eight-cent variances buys nothing and destroys carrier goodwill.
BASE_TOL = 0.02          # 2% on discounted base
BASE_TOL_ABS = 0.25      # or 25 cents, whichever is larger
FUEL_TOL_PP = 0.004      # 0.4 percentage points on the fuel index

# Priority when a single line trips several rules. Whole-line recoveries come
# first because they supersede any component-level variance on that line.
PRIORITY = [
    "duplicate_billing",
    "manifested_not_shipped",
    "dim_misapplication",
    "weight_discrepancy",
    "rate_variance",
    "fuel_percentage_error",
    "unearned_residential",
    "unearned_address_correction",
    "unearned_peak_surcharge",
]


CANDIDATE_DIVISORS = np.arange(100, 139)


def _systemic_divisor_match(cube, billed_w, manifest_w, contract_divisor,
                            group_key, overbilled):
    """Separate a divisor problem from a scale problem.

    A tightened divisor is systemic: one wrong number explains a whole cluster
    of overcharges on the same carrier and service. A scale problem is random.
    Testing each line against every plausible divisor cannot tell them apart,
    because on a large box some divisor will reproduce almost any billed
    weight by coincidence.

    So the divisor is learned rather than guessed. For each carrier and service
    the routine counts how many overbilled lines each candidate divisor would
    explain, takes the single divisor that explains the most, and classifies
    only the lines that divisor accounts for. Everything else stays a weight
    discrepancy. This also yields the number an analyst actually needs in a
    carrier conversation: the divisor being applied against contract.
    """
    n = len(cube)
    match = np.zeros(n, dtype=bool)
    learned = {}

    for key in pd.unique(group_key[overbilled]):
        sel = overbilled & (group_key == key)
        if sel.sum() == 0:
            continue
        c, b, a = cube[sel], billed_w[sel], np.ceil(manifest_w[sel])
        contract_d = contract_divisor[sel][0]

        cands = CANDIDATE_DIVISORS[CANDIDATE_DIVISORS < contract_d]
        if len(cands) < 5:
            continue

        masks, counts = [], []
        for d in cands:
            dim_at_d = np.ceil(c / d)
            hit = (dim_at_d == b) & (dim_at_d > a)
            masks.append(hit)
            counts.append(int(hit.sum()))
        counts = np.array(counts, dtype=float)

        # Divisors near the true one explain many of the same lines, because
        # ceil() is a step function: the count curve shows a plateau around
        # the systemic divisor, not a single spike. A local baseline would
        # read that plateau as its own background, so the baseline is taken
        # globally, from the quiet part of the sweep.
        baseline = float(np.median(counts[counts <= np.percentile(counts, 40)]))
        j = int(np.argmax(counts))

        if counts[j] < max(10.0, 5.0 * baseline, 0.03 * sel.sum()):
            continue

        # Within the plateau the data cannot resolve a single divisor, so all
        # of it is treated as one hypothesis: a line is a divisor problem if
        # any divisor in the plateau explains it.
        keep = counts >= 0.75 * counts[j]
        plateau = cands[keep]
        union = np.zeros(sel.sum(), dtype=bool)
        for k in np.where(keep)[0]:
            union |= masks[k]
        idx = np.where(sel)[0][union]
        match[idx] = True
        learned[key] = {
            "divisor_applied": int(cands[j]),
            "plateau": (int(plateau.min()), int(plateau.max())),
            "contract_divisor": int(contract_d),
            "lines": int(union.sum()),
            "coincidence_baseline": round(baseline, 1),
        }

    return match, learned


def run_audit(ship: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    df = inv.merge(
        ship[[
            "tracking_id", "client", "ship_date", "month", "zone", "mode", "tier",
            "discount_pct", "manifest_weight_lb", "length_in", "width_in", "height_in",
            "is_residential", "address_correction", "billable_lb", "expected_base",
            "expected_resi", "expected_das", "expected_ah", "expected_lp",
            "expected_addr", "expected_peak", "expected_fuel_pct",
            "expected_accessorials", "expected_fuel", "expected_total",
            "delivered_late", "service_guaranteed",
        ]],
        on="tracking_id", how="left", suffixes=("", "_m"),
    )

    svc_idx = np.array([rc.SERVICE_IDX[s] for s in df["service"]])
    contract_divisor = rc.DIM_DIVISOR[svc_idx]
    cube = (df["length_in"] * df["width_in"] * df["height_in"]).values

    flags = {}

    # ---- whole-line recoveries -------------------------------------------
    dup_rank = df.groupby("tracking_id").cumcount()
    flags["duplicate_billing"] = (dup_rank > 0).values
    flags["manifested_not_shipped"] = (~df["shipped_scan"]).values

    # ---- weight and dimensional -----------------------------------------
    over_weight = df["billed_weight_lb"].values > df["billable_lb"].values
    group_key = (df["carrier"] + " / " + df["service"]).values
    dim_explained, learned_divisors = _systemic_divisor_match(
        cube, df["billed_weight_lb"].values, df["manifest_weight_lb"].values,
        contract_divisor, group_key, over_weight)
    flags["dim_misapplication"] = over_weight & dim_explained
    flags["weight_discrepancy"] = over_weight & ~dim_explained

    # ---- base rate --------------------------------------------------------
    # Only meaningful once billed weight matches, otherwise the base gap is a
    # weight problem already captured above.
    weight_match = df["billed_weight_lb"].values == df["billable_lb"].values
    base_gap = df["billed_base"].values - df["expected_base"].values
    tol = np.maximum(df["expected_base"].values * BASE_TOL, BASE_TOL_ABS)
    flags["rate_variance"] = weight_match & (base_gap > tol)

    # ---- fuel index -------------------------------------------------------
    fuel_bearing = (df["billed_base"].values
                    + df["billed_accessorials"].values
                    - df["billed_address_correction"].values * rc.ACCESSORIALS["address_correction"])
    implied_fuel_pct = np.divide(
        df["billed_fuel"].values, fuel_bearing,
        out=np.zeros(len(df)), where=fuel_bearing > 0)
    flags["fuel_percentage_error"] = (
        implied_fuel_pct > (df["expected_fuel_pct"].values + FUEL_TOL_PP))

    # ---- unearned accessorials -------------------------------------------
    flags["unearned_residential"] = (
        (df["billed_residential"].values == 1) & (~df["is_residential"].values))
    flags["unearned_address_correction"] = (
        (df["billed_address_correction"].values == 1) & (~df["address_correction"].values))
    in_peak = ((df["month"] >= 11)
               | ((df["month"] == 10) & (df["ship_date"].dt.day >= 15))
               | ((df["month"] == 1) & (df["ship_date"].dt.day <= 15))).values
    flags["unearned_peak_surcharge"] = (df["billed_peak"].values == 1) & ~in_peak

    for k, v in flags.items():
        df["flag_" + k] = v

    # ---- primary classification ------------------------------------------
    exception = np.full(len(df), "none", dtype=object)
    for rule in reversed(PRIORITY):
        exception = np.where(flags[rule], rule, exception)
    df["exception_type"] = exception

    # ---- recovery pricing -------------------------------------------------
    whole_line = flags["duplicate_billing"] | flags["manifested_not_shipped"]
    fuel_pct = df["expected_fuel_pct"].values

    resi_fee = np.where(df["mode"].values == "Air",
                        rc.ACCESSORIALS["residential"]["Air"],
                        rc.ACCESSORIALS["residential"]["Ground"])
    peak_fee = np.where(df["billable_lb"].values <= 5, rc.PEAK_RESIDENTIAL["light"],
                        np.where(df["billable_lb"].values <= 20,
                                 rc.PEAK_RESIDENTIAL["mid"], rc.PEAK_RESIDENTIAL["heavy"]))

    base_delta = np.maximum(df["billed_base"].values - df["expected_base"].values, 0)
    fuel_only = np.maximum(
        df["billed_fuel"].values - (fuel_bearing * fuel_pct), 0)

    recovery = np.select(
        [
            whole_line,
            np.isin(exception, ["dim_misapplication", "weight_discrepancy", "rate_variance"]),
            exception == "fuel_percentage_error",
            exception == "unearned_residential",
            exception == "unearned_address_correction",
            exception == "unearned_peak_surcharge",
        ],
        [
            df["billed_total"].values,
            base_delta * (1 + fuel_pct),
            fuel_only,
            resi_fee * (1 + fuel_pct),
            np.full(len(df), rc.ACCESSORIALS["address_correction"], dtype=float),
            peak_fee,
        ],
        default=0.0,
    )
    # A claim can never exceed what was actually billed on the line.
    recovery = np.clip(recovery, 0, df["billed_total"].values)
    df["recovery_usd"] = np.round(recovery, 2)

    # ---- guaranteed service refunds --------------------------------------
    # Priced separately: this is a valid charge that becomes refundable on a
    # service failure, not a billing error, and it is claimed on a clock.
    gsr = (df["service_guaranteed"].values & df["delivered_late"].values
           & ~whole_line)
    df["gsr_eligible"] = gsr
    df["gsr_refund_usd"] = np.where(gsr, df["billed_total"].values, 0.0).round(2)

    df["variance_usd"] = (df["billed_total"] - df["expected_total"]).round(2)
    df.attrs["learned_divisors"] = learned_divisors
    return df


def score_detection(audited: pd.DataFrame) -> pd.DataFrame:
    """Score each rule against the injected ground truth.

    An audit program lives or dies on precision. Every false positive is a
    dispute an analyst files, a carrier rep rejects, and a client sees.
    """
    whole_line = (audited["flag_duplicate_billing"].values
                  | audited["flag_manifested_not_shipped"].values)
    rows = []
    for rule in PRIORITY:
        # A duplicated or never-shipped line is recovered in full, so the
        # component rules are never run against it and are not scored on it.
        universe = (np.ones(len(audited), bool)
                    if rule in ("duplicate_billing", "manifested_not_shipped")
                    else ~whole_line)
        truth = ((audited["error_label"] == rule).values & universe)
        pred = (audited["flag_" + rule].values & universe)
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        rows.append({
            "exception_type": rule,
            "injected": int(truth.sum()),
            "detected": int(pred.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "recall": tp / truth.sum() if truth.sum() else np.nan,
            "precision": tp / pred.sum() if pred.sum() else np.nan,
        })
    return pd.DataFrame(rows)


def recovery_summary(audited: pd.DataFrame) -> pd.DataFrame:
    grp = (audited[audited["exception_type"] != "none"]
           .groupby("exception_type")
           .agg(lines=("invoice_line_id", "count"),
                recovery_usd=("recovery_usd", "sum"))
           .reset_index()
           .sort_values("recovery_usd", ascending=False))
    gsr = audited[audited["gsr_eligible"]]
    if len(gsr):
        grp = pd.concat([grp, pd.DataFrame([{
            "exception_type": "service_failure_refund",
            "lines": len(gsr),
            "recovery_usd": gsr["gsr_refund_usd"].sum(),
        }])], ignore_index=True)
    return grp.sort_values("recovery_usd", ascending=False).reset_index(drop=True)
