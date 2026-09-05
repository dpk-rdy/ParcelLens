"""
Carrier and client scorecards, and forward rate-change modeling.

Scorecards answer the standing questions: what does a package cost with each
carrier, how much of that is surcharge rather than transportation, is service
holding, and how clean is the billing. The GRI model answers the annual one:
what does next year's rate card do to this exact volume, and how much of it
can be absorbed by moving the mix rather than paying it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ratecards as rc
from . import rating


def carrier_scorecard(ship: pd.DataFrame, audited: pd.DataFrame) -> pd.DataFrame:
    base = (ship.groupby("carrier")
            .agg(shipments=("tracking_id", "count"),
                 spend=("expected_total", "sum"),
                 base_spend=("expected_base", "sum"),
                 accessorial_spend=("expected_accessorials", "sum"),
                 fuel_spend=("expected_fuel", "sum"),
                 billable_lb=("billable_lb", "sum"),
                 on_time=("delivered_late", lambda s: 1 - s.mean()),
                 avg_zone=("zone", "mean"))
            .reset_index())
    base["cost_per_package"] = base["spend"] / base["shipments"]
    base["cost_per_lb"] = base["spend"] / base["billable_lb"]
    base["accessorial_share"] = base["accessorial_spend"] / base["spend"]
    base["fuel_share"] = base["fuel_spend"] / base["spend"]

    ex = (audited.assign(is_exc=audited["exception_type"] != "none")
          .groupby("carrier")
          .agg(invoice_lines=("invoice_line_id", "count"),
               exception_rate=("is_exc", "mean"),
               recovery_usd=("recovery_usd", "sum"),
               gsr_usd=("gsr_refund_usd", "sum"))
          .reset_index())
    out = base.merge(ex, on="carrier", how="left")
    out["billing_accuracy"] = 1 - out["exception_rate"]
    out["recovery_share_of_spend"] = (out["recovery_usd"] + out["gsr_usd"]) / out["spend"]
    return out.sort_values("spend", ascending=False)


def client_scorecard(ship, audited, shop, pack) -> pd.DataFrame:
    base = (ship.groupby("client")
            .agg(shipments=("tracking_id", "count"),
                 spend=("expected_total", "sum"),
                 accessorial_spend=("expected_accessorials", "sum"),
                 avg_billable_lb=("billable_lb", "mean"),
                 avg_zone=("zone", "mean"),
                 residential_share=("is_residential", "mean"),
                 on_time=("delivered_late", lambda s: 1 - s.mean()))
            .reset_index())
    base["cost_per_package"] = base["spend"] / base["shipments"]
    base["accessorial_share"] = base["accessorial_spend"] / base["spend"]

    rec = (audited.groupby("client")
           .agg(recovery_usd=("recovery_usd", "sum"),
                gsr_usd=("gsr_refund_usd", "sum")).reset_index())
    shp = (shop.groupby("client")
           .agg(rateshop_savings=("savings", "sum")).reset_index())
    pck = (pack.groupby("client")
           .agg(packaging_savings=("packaging_savings", "sum")).reset_index())

    out = (base.merge(rec, on="client", how="left")
           .merge(shp, on="client", how="left")
           .merge(pck, on="client", how="left"))
    out["total_opportunity"] = (out["recovery_usd"] + out["gsr_usd"]
                                + out["rateshop_savings"] + out["packaging_savings"])
    out["opportunity_pct_of_spend"] = out["total_opportunity"] / out["spend"]
    return out.sort_values("total_opportunity", ascending=False)


# --------------------------------------------------------------------------
# Forward rate change
# --------------------------------------------------------------------------
def _next_year_table() -> np.ndarray:
    table = rc.RATE_TABLE.copy()
    for name, spec in rc.SERVICES.items():
        i = rc.SERVICE_IDX[name]
        table[i] *= (1 + rc.GRI["base_pct"][spec["mode"]])
    return table


def gri_impact(ship: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reprice the current year's exact volume on next year's rate card.

    Splits the increase into its causes, because "spend is up 7%" is not
    actionable and "the divisor change is 31% of the increase and only touches
    low-density freight" is.
    """
    svc_idx = np.array([rc.SERVICE_IDX[s] for s in ship["service"]])
    args = dict(
        svc_idx=svc_idx, zone=ship["zone"].values,
        actual_lb=ship["manifest_weight_lb"].values,
        length_in=ship["length_in"].values, width_in=ship["width_in"].values,
        height_in=ship["height_in"].values,
        is_residential=ship["is_residential"].values,
        das_type=ship["das_type"].values, month=ship["month"].values,
        day=ship["ship_date"].dt.day.values,
        discount_pct=ship["discount_pct"].values,
        address_correction=ship["address_correction"].values,
    )

    new_table = _next_year_table()
    new_divisor = rc.DIM_DIVISOR[svc_idx].copy()
    for svc, d in rc.GRI["dim_divisor_change"].items():
        new_divisor = np.where(svc_idx == rc.SERVICE_IDX[svc], d, new_divisor)

    current = ship["expected_total"].values

    # Isolate each lever by applying it on its own against the base year.
    only_base = rating.rate_shipments(rate_table=new_table, **args)["total"]
    only_dim = rating.rate_shipments(dim_divisor=new_divisor, **args)["total"]
    only_acc = rating.rate_shipments(
        accessorial_multiplier=rc.GRI["accessorial_pct"], **args)["total"]
    only_fuel = rating.rate_shipments(fuel_shift=rc.GRI["fuel_shift_pp"], **args)["total"]

    combined = rating.rate_shipments(
        rate_table=new_table, dim_divisor=new_divisor,
        accessorial_multiplier=rc.GRI["accessorial_pct"],
        fuel_shift=rc.GRI["fuel_shift_pp"], **args)["total"]

    levers = pd.DataFrame([
        {"lever": "Base rate increase", "impact_usd": (only_base - current).sum()},
        {"lever": "DIM divisor tightening", "impact_usd": (only_dim - current).sum()},
        {"lever": "Accessorial increases", "impact_usd": (only_acc - current).sum()},
        {"lever": "Fuel index shift", "impact_usd": (only_fuel - current).sum()},
    ])
    total_increase = combined.sum() - current.sum()
    levers["share_of_increase"] = levers["impact_usd"] / levers["impact_usd"].sum()

    detail = ship[["tracking_id", "client", "carrier", "service", "zone"]].copy()
    detail["current_cost"] = current
    detail["next_year_cost"] = np.round(combined, 2)
    detail["increase"] = (detail["next_year_cost"] - detail["current_cost"]).round(2)

    summary = {
        "current_spend": float(current.sum()),
        "next_year_spend": float(combined.sum()),
        "increase_usd": float(total_increase),
        "increase_pct": float(total_increase / current.sum()),
        "effective_vs_headline": float(
            total_increase / current.sum() - rc.GRI["base_pct"]["Ground"]),
    }
    return levers, {"summary": summary, "detail": detail}
