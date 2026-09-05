"""
Tests for the rating, audit and optimization logic.

The audit tests matter most. A reconciliation engine that silently drifts from
the rate card produces a plausible-looking exception register full of variances
that are not real, and nobody notices until a carrier rejects the disputes.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from parcellens import audit, generate, packaging, rateshop  # noqa: E402
from parcellens import ratecards as rc  # noqa: E402
from parcellens import rating, scorecard  # noqa: E402

N = 12_000


@pytest.fixture(scope="module")
def data():
    ship, inv = generate.generate(N, seed=7)
    return ship, inv


@pytest.fixture(scope="module")
def audited(data):
    return audit.run_audit(*data)


# ---------------------------------------------------------------- rating
def test_billable_weight_takes_the_greater_of_scale_and_dim():
    svc = np.array([rc.SERVICE_IDX["NEX_GROUND"]] * 2)
    # 18x14x12 = 3024 cu in / 139 = 21.8 -> 22 lb dim
    bw = rating.billable_weight(
        np.array([5.0, 40.0]), np.array([18.0, 18.0]),
        np.array([14.0, 14.0]), np.array([12.0, 12.0]), svc)
    assert bw[0] == 22          # dim wins
    assert bw[1] == 40          # scale wins


def test_billable_weight_rounds_up_to_the_pound():
    svc = np.array([rc.SERVICE_IDX["NEX_GROUND"]])
    bw = rating.billable_weight(np.array([4.01]), np.array([2.0]),
                                np.array([2.0]), np.array([2.0]), svc)
    assert bw[0] == 5


def test_a_tighter_divisor_can_only_raise_billable_weight():
    svc = np.array([rc.SERVICE_IDX["NEX_GROUND"]] * 200)
    L = np.random.default_rng(1).uniform(6, 24, 200)
    at_contract = rating.billable_weight(
        np.full(200, 3.0), L, L, L, svc, dim_divisor=np.full(200, 139.0))
    at_tighter = rating.billable_weight(
        np.full(200, 3.0), L, L, L, svc, dim_divisor=np.full(200, 110.0))
    assert (at_tighter >= at_contract).all()


def test_discount_applies_to_base_but_not_to_accessorials():
    kw = dict(
        svc_idx=np.array([rc.SERVICE_IDX["NEX_GROUND"]]), zone=np.array([5]),
        actual_lb=np.array([10.0]), length_in=np.array([12.0]),
        width_in=np.array([9.0]), height_in=np.array([6.0]),
        is_residential=np.array([True]), das_type=np.array([1]),
        month=np.array([6]), day=np.array([10]))
    full = rating.rate_shipments(discount_pct=np.array([0.0]), **kw)
    disc = rating.rate_shipments(discount_pct=np.array([0.40]), **kw)
    assert disc["base"][0] == pytest.approx(full["base"][0] * 0.60)
    assert disc["accessorials"][0] == pytest.approx(full["accessorials"][0])


def test_fuel_excludes_the_address_correction_fee():
    kw = dict(
        svc_idx=np.array([rc.SERVICE_IDX["NEX_GROUND"]]), zone=np.array([4]),
        actual_lb=np.array([8.0]), length_in=np.array([10.0]),
        width_in=np.array([8.0]), height_in=np.array([6.0]),
        is_residential=np.array([False]), das_type=np.array([0]),
        month=np.array([3]), day=np.array([2]), discount_pct=np.array([0.30]))
    without = rating.rate_shipments(address_correction=np.array([False]), **kw)
    with_fee = rating.rate_shipments(address_correction=np.array([True]), **kw)
    assert with_fee["fuel"][0] == pytest.approx(without["fuel"][0])
    assert with_fee["total"][0] == pytest.approx(
        without["total"][0] + rc.ACCESSORIALS["address_correction"])


def test_large_package_absorbs_the_additional_handling_fee():
    # 40x30x30: longest 40, girth 120, length+girth 160 -> large package
    priced = rating.rate_shipments(
        svc_idx=np.array([rc.SERVICE_IDX["NEX_GROUND"]]), zone=np.array([4]),
        actual_lb=np.array([55.0]), length_in=np.array([40.0]),
        width_in=np.array([30.0]), height_in=np.array([30.0]),
        is_residential=np.array([False]), das_type=np.array([0]),
        month=np.array([5]), day=np.array([1]), discount_pct=np.array([0.0]))
    assert priced["acc_large_package"][0] == rc.ACCESSORIALS["large_package"]
    assert priced["acc_additional_handling"][0] == 0.0


def test_service_coverage_rejects_out_of_zone_and_overweight():
    n = 3
    ok = rating.service_available(
        np.array([rc.SERVICE_IDX["RGX_REGIONAL"]] * n),
        np.array([3, 7, 3]), np.array([10.0, 10.0, 60.0]),
        np.full(n, 8.0), np.full(n, 6.0), np.full(n, 4.0))
    assert list(ok) == [True, False, False]   # zone 7 and 60 lb are out


# ---------------------------------------------------------------- audit
def test_a_clean_invoice_produces_no_exceptions(data):
    """Reconciliation must be an identity on a correctly billed file."""
    ship, inv = data
    clean = inv[inv["error_label"] == "none"].copy()
    result = audit.run_audit(ship, clean)
    flagged = result[result["exception_type"] != "none"]
    assert len(flagged) == 0, flagged["exception_type"].value_counts().to_dict()


def test_every_seeded_error_type_is_detected(audited):
    scores = audit.score_detection(audited).set_index("exception_type")
    floors = {
        "duplicate_billing": 0.99, "manifested_not_shipped": 0.99,
        "rate_variance": 0.99, "unearned_residential": 0.99,
        "unearned_address_correction": 0.99, "unearned_peak_surcharge": 0.99,
        "fuel_percentage_error": 0.85, "weight_discrepancy": 0.80,
    }
    for rule, floor in floors.items():
        assert scores.loc[rule, "recall"] >= floor, rule


def test_overbillings_are_caught_even_when_the_subtype_is_ambiguous(audited):
    """DIM and scale overcharges cannot always be told apart line by line.

    What must not happen is an overcharge escaping entirely, so the two rules
    are held to a joint standard rather than individually.
    """
    truth = audited["error_label"].isin(["dim_misapplication", "weight_discrepancy"])
    caught = (audited["flag_dim_misapplication"] | audited["flag_weight_discrepancy"])
    assert (truth & caught).sum() / truth.sum() >= 0.99


def test_recovery_never_exceeds_the_amount_billed(audited):
    assert (audited["recovery_usd"] <= audited["billed_total"] + 0.01).all()
    assert (audited["recovery_usd"] >= 0).all()


def test_whole_line_recoveries_claim_the_whole_line(audited):
    whole = audited[audited["exception_type"].isin(
        ["duplicate_billing", "manifested_not_shipped"])]
    assert (whole["recovery_usd"] == whole["billed_total"]).all()


def test_service_refunds_are_kept_separate_from_billing_errors(audited):
    """A late guaranteed shipment is a valid charge, not a billing error."""
    gsr = audited[audited["gsr_eligible"]]
    assert gsr["service_guaranteed"].all()
    assert gsr["delivered_late"].all()
    assert not (audited["gsr_refund_usd"] > 0).equals(
        audited["recovery_usd"] > 0)


def test_the_learned_divisor_is_below_contract(audited):
    learned = audited.attrs["learned_divisors"]
    assert learned, "no systemic divisor identified"
    for key, v in learned.items():
        assert v["divisor_applied"] < v["contract_divisor"], key


# ---------------------------------------------------------- optimization
def test_optimization_never_slows_the_delivery_promise(data):
    ship, _ = data
    shop = rateshop.optimize(ship)
    assert (shop["optimal_transit"] <= shop["current_transit"]).all()


def test_savings_are_never_negative_and_switches_are_material(data):
    ship, _ = data
    shop = rateshop.optimize(ship, min_saving_per_shipment=0.50)
    assert (shop["savings"] >= 0).all()
    switched = shop[shop["switched"]]
    assert (switched["savings"] >= 0.50).all()


def test_capacity_caps_are_respected(data):
    ship, _ = data
    cap = 0.10
    shop = rateshop.optimize(ship, capacity_caps={"RegionalGrid": cap})
    share = (shop["optimal_carrier"] == "RegionalGrid").mean()
    assert share <= cap + 1e-9


def test_capping_a_carrier_cannot_increase_savings(data):
    ship, _ = data
    free = rateshop.optimize(ship)["savings"].sum()
    capped = rateshop.optimize(
        ship, capacity_caps={"RegionalGrid": 0.05})["savings"].sum()
    assert capped <= free + 0.01


# ------------------------------------------------------------- packaging
def test_right_sizing_never_costs_more_than_the_current_box(data):
    ship, _ = data
    detail, _ = packaging.analyze(ship)
    assert (detail["packaging_savings"] >= 0).all()
    assert (detail["rightsized_box_cube"] <= ship["box_cube_in3"].values + 1e-6).all()


def test_dim_penalty_is_zero_when_scale_weight_governs(data):
    ship, _ = data
    detail, _ = packaging.analyze(ship)
    scale_billed = detail["billable_lb"] <= detail["actual_lb"]
    assert (detail.loc[scale_billed, "dim_penalty_lb"] == 0).all()


# ------------------------------------------------------------ rate change
def test_a_zero_increase_reprices_to_the_current_spend(data, monkeypatch):
    ship, _ = data
    monkeypatch.setitem(rc.GRI, "base_pct", {"Ground": 0.0, "Air": 0.0, "Postal": 0.0})
    monkeypatch.setitem(rc.GRI, "accessorial_pct", {k: 0.0 for k in rc.GRI["accessorial_pct"]})
    monkeypatch.setitem(rc.GRI, "fuel_shift_pp", 0.0)
    monkeypatch.setitem(rc.GRI, "dim_divisor_change", {})
    _, result = scorecard.gri_impact(ship)
    # Not exactly zero: the stored current cost is rounded to the cent per
    # line, and the reprice is not.
    assert abs(result["summary"]["increase_pct"]) < 1e-4


def test_the_effective_increase_exceeds_the_headline(data):
    """Accessorials and the divisor move faster than the base rate, so the
    effective increase on a real mix is always above the announced number."""
    ship, _ = data
    _, result = scorecard.gri_impact(ship)
    assert result["summary"]["increase_pct"] > rc.GRI["base_pct"]["Ground"]
