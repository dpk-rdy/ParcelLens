"""
Rate-shop optimizer.

For every shipment, prices all seven services and picks the cheapest one that
still meets the delivery promise the customer was already given. Holding the
promise constant matters: a rate shop that quietly slows delivery to look good
on cost is not a saving, it is a service cut priced as a saving.

A second scenario relaxes the promise by one day to size what a deliberate
service-policy change would be worth, so the two decisions stay separable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ratecards as rc
from . import rating

N_SVC = len(rc.SERVICE_LIST)


def _price_all_services(ship: pd.DataFrame):
    """Return (cost[n, n_svc], feasible[n, n_svc], transit[n, n_svc])."""
    n = len(ship)
    zone = ship["zone"].values
    w = ship["manifest_weight_lb"].values
    L = ship["length_in"].values
    W = ship["width_in"].values
    H = ship["height_in"].values
    resi = ship["is_residential"].values
    das = ship["das_type"].values
    month = ship["month"].values
    day = ship["ship_date"].dt.day.values
    addr = ship["address_correction"].values
    tier = ship["tier"].values

    cost = np.full((n, N_SVC), np.inf)
    feasible = np.zeros((n, N_SVC), dtype=bool)
    transit = np.full((n, N_SVC), 99, dtype=int)

    for j, svc in enumerate(rc.SERVICE_LIST):
        idx = np.full(n, j)
        mode = rc.SERVICES[svc]["mode"]
        disc = np.array([rc.discount_for(t, mode) for t in tier])
        ok = rating.service_available(idx, zone, w, L, W, H)
        if not ok.any():
            continue
        priced = rating.rate_shipments(
            idx, zone, w, L, W, H, resi, das, month, day, disc,
            address_correction=addr)
        cost[:, j] = np.where(ok, priced["total"], np.inf)
        feasible[:, j] = ok
        transit[:, j] = rating.transit_days(idx, zone)
    return cost, feasible, transit


def optimize(ship: pd.DataFrame, sla_slack_days: int = 0,
             capacity_caps: dict[str, float] | None = None,
             min_saving_per_shipment: float = 0.50) -> pd.DataFrame:
    """Least-cost feasible service per shipment within the delivery promise.

    min_saving_per_shipment is a materiality floor. Moving tens of thousands of
    packages to another carrier to save a quarter each is not an optimization,
    it is a rounding difference with an implementation project attached, and it
    will not survive contact with an operations team.

    capacity_caps maps a carrier to the maximum share of total volume it may
    carry. Without it the optimizer will hand a regional carrier every package
    it can price cheaply, which is arithmetic rather than a plan: regional
    networks have finite injection capacity and finite coverage. Caps are
    applied greedily on regret, so the volume a capped carrier does get is the
    volume where it beats the next-best alternative by the most.
    """
    cost, feasible, transit = _price_all_services(ship)
    promise = ship["promised_days"].values[:, None] + sla_slack_days
    eligible = feasible & (transit <= promise)

    # A shipment always remains eligible for the service it actually used.
    n = len(ship)
    cur_idx = np.array([rc.SERVICE_IDX[s] for s in ship["service"]])
    eligible[np.arange(n), cur_idx] = True

    masked = np.where(eligible, cost, np.inf)
    best = np.argmin(masked, axis=1)
    best_cost = masked[np.arange(n), best]
    current_cost = ship["expected_total"].values

    for carrier, share in (capacity_caps or {}).items():
        svc_cols = np.where(rc.SERVICE_CARRIER == carrier)[0]
        assigned = np.isin(best, svc_cols)
        cap = int(share * n)
        if assigned.sum() <= cap:
            continue

        # Cost of the best alternative that avoids this carrier.
        alt = masked.copy()
        alt[:, svc_cols] = np.inf
        alt_best = np.argmin(alt, axis=1)
        alt_cost = alt[np.arange(n), alt_best]

        # Volume already flowing to this carrier is inside its capacity by
        # definition, so it is kept first. The cap governs how much *new*
        # volume the network can absorb, and that goes to the lanes where the
        # carrier beats the next-best alternative by the most.
        incumbent = np.isin(cur_idx, svc_cols) & assigned
        regret = np.where(assigned, alt_cost - best_cost, -np.inf)
        rank = np.lexsort((-regret, ~incumbent))   # incumbents first, then regret
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[rank[:cap]] = True

        bump = assigned & ~keep_mask & np.isfinite(alt_cost)
        best = np.where(bump, alt_best, best)
        best_cost = np.where(bump, alt_cost, best_cost)

    out = pd.DataFrame({
        "tracking_id": ship["tracking_id"].values,
        "client": ship["client"].values,
        "zone": ship["zone"].values,
        "current_service": ship["service"].values,
        "current_carrier": ship["carrier"].values,
        "current_cost": current_cost,
        "optimal_service": np.array(rc.SERVICE_LIST)[best],
        "optimal_carrier": rc.SERVICE_CARRIER[best],
        "optimal_cost": np.round(best_cost, 2),
        "current_transit": ship["promised_days"].values,
        "optimal_transit": transit[np.arange(len(ship)), best],
    })
    out["savings"] = (out["current_cost"] - out["optimal_cost"]).round(2)
    out["switched"] = out["current_service"] != out["optimal_service"]

    # Below the materiality floor, or where the current service already wins,
    # nothing moves.
    immaterial = out["savings"] < min_saving_per_shipment
    out.loc[immaterial, "optimal_service"] = out.loc[immaterial, "current_service"]
    out.loc[immaterial, "optimal_carrier"] = out.loc[immaterial, "current_carrier"]
    out.loc[immaterial, "optimal_cost"] = out.loc[immaterial, "current_cost"]
    out.loc[immaterial, "savings"] = 0.0
    out.loc[immaterial, "switched"] = False
    return out


def savings_by(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    g = (frame.groupby(key)
         .agg(shipments=("tracking_id", "count"),
              current_spend=("current_cost", "sum"),
              optimized_spend=("optimal_cost", "sum"),
              savings=("savings", "sum"),
              switch_rate=("switched", "mean"))
         .reset_index())
    g["savings_pct"] = g["savings"] / g["current_spend"]
    return g.sort_values("savings", ascending=False)


def migration_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Where volume moves from and to, in shipments and dollars."""
    m = (frame[frame["switched"]]
         .groupby(["current_service", "optimal_service"])
         .agg(shipments=("tracking_id", "count"), savings=("savings", "sum"))
         .reset_index()
         .sort_values("savings", ascending=False))
    return m
