"""
Packaging and dimensional weight analysis.

Dimensional weight is the quietest line item in parcel spend: nothing on the
invoice says "you shipped air." A package is DIM-billed whenever cube divided
by the contractual divisor exceeds the scale weight, and the fix is a box
change on the pack line rather than a carrier negotiation.

This module measures how much volume is DIM-billed, how full the boxes are,
and what right-sizing to the smallest catalog box that still fits the item
would be worth at current rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ratecards as rc
from . import rating
from .generate import BOX_CATALOG

VOID_ALLOWANCE = 1.25   # 25% void space is a fair packing target


def analyze(ship: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    svc_idx = np.array([rc.SERVICE_IDX[s] for s in ship["service"]])
    divisor = rc.DIM_DIVISOR[svc_idx]
    cube = ship["box_cube_in3"].values.astype(float)
    dim_lb = np.ceil(cube / divisor)
    actual_lb = np.ceil(ship["manifest_weight_lb"].values)
    dim_billed = dim_lb > actual_lb

    # Smallest catalog box that still holds the item plus void allowance.
    vols = np.array([l * w * h for _, l, w, h in BOX_CATALOG], dtype=float)
    order = np.argsort(vols)
    sorted_vols = vols[order]
    need = ship["item_cube_in3"].values * VOID_ALLOWANCE
    pick = np.clip(np.searchsorted(sorted_vols, need), 0, len(BOX_CATALOG) - 1)
    right_box = order[pick]
    dims = np.array([[l, w, h] for _, l, w, h in BOX_CATALOG], dtype=float)
    rL, rW, rH = dims[right_box, 0], dims[right_box, 1], dims[right_box, 2]

    # Only ever shrink; never propose a box smaller than what fits.
    shrink = (rL * rW * rH) < cube
    nL = np.where(shrink, rL, ship["length_in"].values)
    nW = np.where(shrink, rW, ship["width_in"].values)
    nH = np.where(shrink, rH, ship["height_in"].values)

    priced = rating.rate_shipments(
        svc_idx, ship["zone"].values, ship["manifest_weight_lb"].values,
        nL, nW, nH, ship["is_residential"].values, ship["das_type"].values,
        ship["month"].values, ship["ship_date"].dt.day.values,
        ship["discount_pct"].values,
        address_correction=ship["address_correction"].values)

    out = pd.DataFrame({
        "tracking_id": ship["tracking_id"].values,
        "client": ship["client"].values,
        "carrier": ship["carrier"].values,
        "box_id": ship["box_id"].values,
        "cube_utilization": np.clip(ship["item_cube_in3"].values / cube, 0, 1),
        "dim_billed": dim_billed,
        "billable_lb": ship["billable_lb"].values,
        "actual_lb": actual_lb,
        "dim_penalty_lb": np.maximum(dim_lb - actual_lb, 0),
        "current_cost": ship["expected_total"].values,
        "rightsized_box_cube": (nL * nW * nH),
        "rightsized_billable_lb": priced["billable_lb"],
        "rightsized_cost": np.round(priced["total"], 2),
    })
    out["packaging_savings"] = np.maximum(
        out["current_cost"] - out["rightsized_cost"], 0).round(2)

    summary = {
        "dim_billed_share": float(dim_billed.mean()),
        "avg_cube_utilization": float(out["cube_utilization"].mean()),
        "dim_penalty_spend": float(_dim_penalty_spend(ship, svc_idx, dim_lb, actual_lb)),
        "packaging_savings": float(out["packaging_savings"].sum()),
        "shipments_improvable": int((out["packaging_savings"] > 0).sum()),
    }
    return out, summary


def _dim_penalty_spend(ship, svc_idx, dim_lb, actual_lb):
    """Spend attributable purely to DIM: cost at billable weight minus cost at
    scale weight, everything else held constant."""
    zone = ship["zone"].values
    at_dim = rc.RATE_TABLE[svc_idx, zone, np.clip(dim_lb, 1, rc.MAX_WEIGHT).astype(int)]
    at_actual = rc.RATE_TABLE[svc_idx, zone, np.clip(actual_lb, 1, rc.MAX_WEIGHT).astype(int)]
    delta = np.maximum(np.nan_to_num(at_dim) - np.nan_to_num(at_actual), 0)
    delta = delta * (1 - ship["discount_pct"].values) * (1 + ship["expected_fuel_pct"].values)
    return delta.sum()


def by_client(detail: pd.DataFrame) -> pd.DataFrame:
    g = (detail.groupby("client")
         .agg(shipments=("tracking_id", "count"),
              dim_billed_rate=("dim_billed", "mean"),
              avg_cube_utilization=("cube_utilization", "mean"),
              avg_dim_penalty_lb=("dim_penalty_lb", "mean"),
              packaging_savings=("packaging_savings", "sum"))
         .reset_index()
         .sort_values("packaging_savings", ascending=False))
    return g
