"""
Shipment and invoice generator.

Produces two tables that mirror what a parcel analyst actually receives:

  shipments.parquet  - the manifest from the TMS/WMS: what was tendered,
                       what it weighed and measured on the pack line, which
                       service was selected, and how it was delivered.
  invoices.parquet   - what the carrier billed, at line-item granularity,
                       including the errors carriers make in practice.

The invoice is *not* generated from the manifest verbatim. Billing errors are
injected at realistic rates and every error carries a ground-truth label, so
the audit engine can be scored on recall and precision rather than just
producing a number nobody can check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ratecards as rc
from . import rating

RNG_SEED = 20260905

# --------------------------------------------------------------------------
# Client book of business
# --------------------------------------------------------------------------
CLIENTS = {
    "Northwind Apparel": dict(
        share=0.26, tier="T1", resi_rate=0.93,
        weight_lognorm=(0.35, 0.55), density=13.0,
        primary="NEX_GROUND", secondary="PSD_ECONOMY", sec_share=0.24,
        air_share=0.030, dc="Columbus",
    ),
    "Cascade Supplements": dict(
        share=0.19, tier="T2", resi_rate=0.88,
        weight_lognorm=(1.05, 0.48), density=26.0,
        primary="CPX_GROUND", secondary="NEX_GROUND", sec_share=0.18,
        air_share=0.021, dc="Reno",
    ),
    "Ironline Industrial": dict(
        share=0.13, tier="T2", resi_rate=0.11,
        weight_lognorm=(2.85, 0.62), density=32.0,
        primary="NEX_GROUND", secondary="CPX_GROUND", sec_share=0.30,
        air_share=0.055, dc="Dallas",
    ),
    "Lumen Electronics": dict(
        share=0.14, tier="T3", resi_rate=0.71,
        weight_lognorm=(1.35, 0.52), density=22.0,
        primary="CPX_GROUND", secondary="CPX_SAVER", sec_share=0.16,
        air_share=0.085, dc="Memphis",
    ),
    "Verity Beauty": dict(
        share=0.16, tier="T3", resi_rate=0.96,
        weight_lognorm=(-0.15, 0.50), density=16.0,
        primary="PSD_ECONOMY", secondary="NEX_GROUND", sec_share=0.34,
        air_share=0.018, dc="Columbus",
    ),
    "Harbor Home Goods": dict(
        share=0.12, tier="T4", resi_rate=0.90,
        weight_lognorm=(1.55, 0.60), density=7.5,   # low density: DIM exposure
        primary="NEX_GROUND", secondary="CPX_GROUND", sec_share=0.22,
        air_share=0.024, dc="Dallas",
    ),
}

# Zone mix by origin DC. Zones 2-8.
DC_ZONE_MIX = {
    "Columbus": [0.14, 0.20, 0.22, 0.18, 0.13, 0.08, 0.05],
    "Reno":     [0.11, 0.15, 0.17, 0.17, 0.16, 0.14, 0.10],
    "Dallas":   [0.13, 0.18, 0.20, 0.19, 0.15, 0.09, 0.06],
    "Memphis":  [0.16, 0.21, 0.21, 0.17, 0.12, 0.08, 0.05],
}

# Standard box catalog used on the pack lines (L x W x H inches).
BOX_CATALOG = [
    ("B01", 6, 4, 3), ("B02", 9, 6, 4), ("B03", 12, 9, 4), ("B04", 12, 12, 8),
    ("B05", 16, 12, 10), ("B06", 18, 14, 12), ("B07", 20, 16, 14),
    ("B08", 24, 18, 16), ("B09", 30, 20, 16),
]

# Injected billing error rates (share of invoice lines)
WRONG_DIM_DIVISOR = 110.0  # divisor the carrier misapplies

ERROR_RATES = {
    "rate_variance": 0.0110,
    "dim_misapplication": 0.0080,
    "duplicate_billing": 0.0035,
    "unearned_residential": 0.0140,
    "unearned_address_correction": 0.0050,
    "weight_discrepancy": 0.0160,
    "fuel_percentage_error": 0.0040,
    "unearned_peak_surcharge": 0.0060,
    "manifested_not_shipped": 0.0030,
}


def _sample_dims(rng, weight_lb, density):
    """Pick the smallest catalog box that fits the item, with packer error.

    Density (lb per cubic foot) converts weight to the item's cube. Packers
    pick up a box size at random on roughly one shipment in five, which is
    where dimensional-weight exposure comes from in the real world.
    """
    n = weight_lb.shape[0]
    cubic_ft = weight_lb / density
    cubic_in_needed = cubic_ft * 1728.0

    box_vols = np.array([l * w * h for _, l, w, h in BOX_CATALOG], dtype=float)
    order = np.argsort(box_vols)
    sorted_vols = box_vols[order]

    # smallest box whose volume covers the item plus 25% void
    idx = np.searchsorted(sorted_vols, cubic_in_needed * 1.18)
    idx = np.clip(idx, 0, len(BOX_CATALOG) - 1)

    bump = rng.random(n)
    idx = np.where(bump < 0.11, np.clip(idx + 1, 0, len(BOX_CATALOG) - 1), idx)
    idx = np.where(bump > 0.978, np.clip(idx + 2, 0, len(BOX_CATALOG) - 1), idx)
    chosen = order[idx]

    dims = np.array([[l, w, h] for _, l, w, h in BOX_CATALOG], dtype=float)
    picked = dims[chosen]
    return (picked[:, 0], picked[:, 1], picked[:, 2],
            np.array([BOX_CATALOG[i][0] for i in chosen]), cubic_in_needed)


def generate(n_shipments: int = 250_000, year: int = 2025, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)

    # ---------------- client / date / geography ----------------
    names = list(CLIENTS)
    probs = np.array([CLIENTS[c]["share"] for c in names])
    probs = probs / probs.sum()
    client = rng.choice(names, size=n_shipments, p=probs)

    # Volume seasonality: Q4 peak, February trough.
    month_weight = np.array([0.070, 0.062, 0.075, 0.076, 0.080, 0.078,
                             0.077, 0.079, 0.084, 0.095, 0.118, 0.106])
    month = rng.choice(np.arange(1, 13), size=n_shipments, p=month_weight)
    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    day = (rng.integers(0, 1_000_000, n_shipments) % days_in_month[month - 1]) + 1
    ship_date = pd.to_datetime(
        dict(year=year, month=month, day=day)
    )

    dc = np.array([CLIENTS[c]["dc"] for c in client])
    zone = np.empty(n_shipments, dtype=int)
    for d, mix in DC_ZONE_MIX.items():
        sel = dc == d
        zone[sel] = rng.choice(np.arange(2, 9), size=sel.sum(), p=mix)

    # ---------------- package characteristics ----------------
    mu = np.array([CLIENTS[c]["weight_lognorm"][0] for c in client])
    sigma = np.array([CLIENTS[c]["weight_lognorm"][1] for c in client])
    weight = np.clip(rng.lognormal(mu, sigma), 0.3, 68.0)

    density = np.array([CLIENTS[c]["density"] for c in client])
    L, W, H, box_id, item_cube = _sample_dims(rng, weight, density)

    resi_p = np.array([CLIENTS[c]["resi_rate"] for c in client])
    is_residential = rng.random(n_shipments) < resi_p

    # Delivery area surcharge exposure rises with zone.
    das_roll = rng.random(n_shipments)
    das_p = 0.10 + 0.035 * (zone - 2)
    ext_p = 0.015 + 0.011 * (zone - 2)
    das_type = np.where(
        das_roll < ext_p, 2,
        np.where(das_roll < ext_p + das_p, 1, 0))
    das_type = np.where((das_type > 0) & ~is_residential, 3, das_type)

    # ---------------- service selection ----------------
    # Air is bought mostly where ground is slow, so its share rises with zone
    # rather than being flat. Some short-zone air remains, which is exactly the
    # over-service the rate shop is meant to find.
    zone_air_factor = np.array([0.30, 0.45, 0.70, 1.05, 1.45, 1.85, 2.15])
    air_factor = zone_air_factor[zone - 2]

    svc = np.empty(n_shipments, dtype=object)
    roll = rng.random(n_shipments)
    for c in names:
        spec = CLIENTS[c]
        sel = client == c
        r = roll[sel]
        air_p = np.clip(spec["air_share"] * air_factor[sel], 0, 0.45)
        chosen = np.where(
            r < air_p, "NEX_2DAY",
            np.where(r < air_p + spec["sec_share"],
                     spec["secondary"], spec["primary"]))
        svc[sel] = chosen

    # A slice of volume is tendered to the regional carrier on short zones.
    regional_ok = (zone <= 4) & (weight <= 45) & (rng.random(n_shipments) < 0.12)
    svc = np.where(regional_ok & (svc != "NEX_2DAY"), "RGX_REGIONAL", svc)

    svc_idx = np.array([rc.SERVICE_IDX[s] for s in svc])

    # Fall back to ground when the selected service cannot carry the package.
    ok = rating.service_available(svc_idx, zone, weight, L, W, H)
    fallback = rc.SERVICE_IDX["NEX_GROUND"]
    svc_idx = np.where(ok, svc_idx, fallback)
    svc = np.array(rc.SERVICE_LIST)[svc_idx]

    # ---------------- contracted cost ----------------
    tier = np.array([CLIENTS[c]["tier"] for c in client])
    mode = rc.SERVICE_MODE[svc_idx]
    discount = np.array([rc.discount_for(t, m) for t, m in zip(tier, mode)])

    addr_correction = rng.random(n_shipments) < 0.011
    expected = rating.rate_shipments(
        svc_idx, zone, weight, L, W, H, is_residential, das_type,
        month, day, discount, address_correction=addr_correction)

    # ---------------- delivery outcome ----------------
    promised = rating.transit_days(svc_idx, zone)
    base_delay_p = np.where(mode == "Air", 0.021, np.where(mode == "Postal", 0.086, 0.043))
    peak_bump = np.where((month >= 11) | (month == 12), 0.030, 0.0)
    late = rng.random(n_shipments) < (base_delay_p + peak_bump)
    delay_days = np.where(late, rng.integers(1, 4, n_shipments), 0)
    actual_days = promised + delay_days
    guaranteed = mode == "Air"

    ship = pd.DataFrame({
        "tracking_id": np.arange(1, n_shipments + 1),
        "ship_date": ship_date,
        "month": month,
        "client": client,
        "origin_dc": dc,
        "zone": zone,
        "service": svc,
        "carrier": rc.SERVICE_CARRIER[svc_idx],
        "mode": mode,
        "tier": tier,
        "discount_pct": discount,
        "manifest_weight_lb": np.round(weight, 2),
        "length_in": L, "width_in": W, "height_in": H,
        "box_id": box_id,
        "item_cube_in3": np.round(item_cube, 1),
        "box_cube_in3": L * W * H,
        "is_residential": is_residential,
        "das_type": das_type,
        "address_correction": addr_correction,
        "billable_lb": expected["billable_lb"],
        "expected_base": np.round(expected["base"], 2),
        "expected_resi": np.round(expected["acc_residential"], 2),
        "expected_das": np.round(expected["acc_das"], 2),
        "expected_ah": np.round(expected["acc_additional_handling"], 2),
        "expected_lp": np.round(expected["acc_large_package"], 2),
        "expected_addr": np.round(expected["acc_address_correction"], 2),
        "expected_peak": np.round(expected["acc_peak"], 2),
        "expected_fuel_pct": expected["fuel_pct"],
        "expected_accessorials": np.round(expected["accessorials"], 2),
        "expected_fuel": np.round(expected["fuel"], 2),
        "expected_total": np.round(expected["total"], 2),
        "promised_days": promised,
        "actual_days": actual_days,
        "delivered_late": late,
        "service_guaranteed": guaranteed,
    })
    ship["tracking_id"] = "1Z" + ship["tracking_id"].astype(str).str.zfill(9)

    invoices = _build_invoices(ship, rng)
    return ship, invoices


def _build_invoices(ship: pd.DataFrame, rng) -> pd.DataFrame:
    """Create the carrier invoice file, injecting billing errors with labels."""
    n = len(ship)
    inv = pd.DataFrame({
        "tracking_id": ship["tracking_id"].values,
        "invoice_date": ship["ship_date"] + pd.to_timedelta(
            rng.integers(2, 9, n), unit="D"),
        "carrier": ship["carrier"].values,
        "service": ship["service"].values,
        "zone_billed": ship["zone"].values,
        "billed_weight_lb": ship["billable_lb"].values.astype(float),
        "billed_base": ship["expected_base"].values.astype(float),
        "billed_accessorials": ship["expected_accessorials"].values.astype(float),
        "billed_fuel": ship["expected_fuel"].values.astype(float),
        "billed_residential": np.where(ship["is_residential"], 1, 0),
        "billed_address_correction": np.where(ship["address_correction"], 1, 0),
        "billed_peak": np.where(ship["expected_peak"].values > 0, 1, 0),
    })
    inv["error_label"] = "none"

    def pick(rate, exclude_labeled=True):
        pool = (inv["error_label"] == "none").values if exclude_labeled else np.ones(n, bool)
        draw = rng.random(n) < rate
        return draw & pool

    # 1. Rate variance: contracted discount partially unapplied.
    sel = pick(ERROR_RATES["rate_variance"])
    shortfall = rng.uniform(0.05, 0.16, sel.sum())
    inv.loc[sel, "billed_base"] = (
        inv.loc[sel, "billed_base"].values
        / (1 - ship.loc[sel, "discount_pct"].values)
        * (1 - ship.loc[sel, "discount_pct"].values + shortfall))
    inv.loc[sel, "error_label"] = "rate_variance"

    # 2. DIM misapplication: carrier bills on a tighter divisor than contract.
    sel = pick(ERROR_RATES["dim_misapplication"])
    if sel.sum():
        sub = ship.loc[sel]
        wrong_dim = np.ceil(
            (sub["length_in"] * sub["width_in"] * sub["height_in"]) / WRONG_DIM_DIVISOR)
        new_w = np.maximum(np.ceil(sub["manifest_weight_lb"]), wrong_dim)
        # A carrier cannot bill above the service's weight ceiling, so the
        # misapplied divisor tops out there rather than pricing off the card.
        cap = np.array([rc.SERVICES[s]["max_weight"] for s in sub["service"]])
        new_w = np.clip(new_w, 1, cap)
        changed = new_w > sub["billable_lb"].values
        svc_i = np.array([rc.SERVICE_IDX[s] for s in sub["service"]])
        new_base = (rc.RATE_TABLE[svc_i, sub["zone"].values, new_w.astype(int)]
                    * (1 - sub["discount_pct"].values))
        idx = sub.index[changed]
        inv.loc[idx, "billed_weight_lb"] = new_w.values[changed]
        inv.loc[idx, "billed_base"] = new_base[changed]
        inv.loc[idx, "error_label"] = "dim_misapplication"

    # 3. Unearned residential surcharge on commercial addresses.
    sel = pick(ERROR_RATES["unearned_residential"]) & (~ship["is_residential"].values)
    resi_fee = np.where(ship.loc[sel, "mode"].values == "Air",
                        rc.ACCESSORIALS["residential"]["Air"],
                        rc.ACCESSORIALS["residential"]["Ground"])
    inv.loc[sel, "billed_accessorials"] += resi_fee
    inv.loc[sel, "billed_residential"] = 1
    inv.loc[sel, "error_label"] = "unearned_residential"

    # 4. Address correction fee with no correction on the manifest.
    sel = pick(ERROR_RATES["unearned_address_correction"]) & (~ship["address_correction"].values)
    inv.loc[sel, "billed_accessorials"] += rc.ACCESSORIALS["address_correction"]
    inv.loc[sel, "billed_address_correction"] = 1
    inv.loc[sel, "error_label"] = "unearned_address_correction"

    # 5. Weight discrepancy: billed above the scale weight on the pack line.
    #    Restricted to packages billed on scale weight rather than DIM.
    scale_billed = (np.ceil(ship["manifest_weight_lb"].values)
                    >= ship["billable_lb"].values)
    sel = pick(ERROR_RATES["weight_discrepancy"]) & scale_billed
    if sel.sum():
        sub = ship.loc[sel]
        bump = rng.integers(1, 5, sel.sum())
        new_w = np.clip(sub["billable_lb"].values + bump, 1, rc.MAX_WEIGHT)
        svc_i = np.array([rc.SERVICE_IDX[s] for s in sub["service"]])
        cap = np.array([rc.SERVICES[s]["max_weight"] for s in sub["service"]])
        new_w = np.minimum(new_w, cap)
        new_base = (rc.RATE_TABLE[svc_i, sub["zone"].values, new_w.astype(int)]
                    * (1 - sub["discount_pct"].values))
        moved = new_w > sub["billable_lb"].values
        idx = sub.index[moved]
        inv.loc[idx, "billed_weight_lb"] = new_w[moved]
        inv.loc[idx, "billed_base"] = new_base[moved]
        inv.loc[idx, "error_label"] = "weight_discrepancy"

    # 6. Fuel surcharge applied at the wrong index.
    sel = pick(ERROR_RATES["fuel_percentage_error"]) & (ship["mode"].values != "Postal")
    inv.loc[sel, "billed_fuel"] *= rng.uniform(1.10, 1.35, sel.sum())
    inv.loc[sel, "error_label"] = "fuel_percentage_error"

    # 7. Peak surcharge billed outside the peak window.
    in_peak = ((ship["month"] >= 11)
               | ((ship["month"] == 10) & (ship["ship_date"].dt.day >= 15))
               | ((ship["month"] == 1) & (ship["ship_date"].dt.day <= 15))).values
    sel = pick(ERROR_RATES["unearned_peak_surcharge"]) & ~in_peak & ship["is_residential"].values
    fee = np.where(ship.loc[sel, "billable_lb"].values <= 5, rc.PEAK_RESIDENTIAL["light"],
                   np.where(ship.loc[sel, "billable_lb"].values <= 20,
                            rc.PEAK_RESIDENTIAL["mid"], rc.PEAK_RESIDENTIAL["heavy"]))
    inv.loc[sel, "billed_accessorials"] += fee
    inv.loc[sel, "billed_peak"] = 1
    inv.loc[sel, "error_label"] = "unearned_peak_surcharge"

    # 8. Manifested but never shipped: label created and billed, no movement.
    sel = pick(ERROR_RATES["manifested_not_shipped"])
    inv.loc[sel, "error_label"] = "manifested_not_shipped"
    mns_ids = set(inv.loc[sel, "tracking_id"])

    # Recompute fuel on the lines whose base moved, so the invoice stays
    # internally consistent the way a real carrier file does.
    touched = inv["error_label"].isin(
        ["rate_variance", "dim_misapplication", "weight_discrepancy"]).values
    fuel_pct = np.where(
        ship["mode"].values == "Ground",
        ship["month"].map(rc.FUEL_GROUND).values,
        np.where(ship["mode"].values == "Air",
                 ship["month"].map(rc.FUEL_AIR).values, 0.0))
    fuel_bearing = (inv["billed_base"].values + inv["billed_accessorials"].values
                    - inv["billed_address_correction"].values * rc.ACCESSORIALS["address_correction"])
    inv.loc[touched, "billed_fuel"] = (fuel_bearing * fuel_pct)[touched]

    inv["billed_total"] = (inv["billed_base"] + inv["billed_accessorials"]
                           + inv["billed_fuel"]).round(2)

    # 9. Duplicate billing: append a second line for the same tracking number.
    dup_sel = rng.random(n) < ERROR_RATES["duplicate_billing"]
    dups = inv.loc[dup_sel].copy()
    dups["invoice_date"] = dups["invoice_date"] + pd.to_timedelta(
        rng.integers(6, 21, len(dups)), unit="D")
    dups["error_label"] = "duplicate_billing"
    inv = pd.concat([inv, dups], ignore_index=True)

    inv.insert(0, "invoice_line_id", np.arange(1, len(inv) + 1))
    inv["shipped_scan"] = ~inv["tracking_id"].isin(mns_ids)
    for c in ["billed_base", "billed_accessorials", "billed_fuel", "billed_total"]:
        inv[c] = inv[c].round(2)
    return inv
