"""
Carrier rate cards, accessorial schedules, fuel tables and DIM rules.

The rate structures here are synthetic. They are built to reproduce the *shape*
of published parcel tariffs (zone-banded base charge, escalating per-pound cost,
dimensional weight divisors, percentage fuel surcharge applied to base plus
selected accessorials) without reproducing any carrier's proprietary rates.
Every downstream module treats these as the contractual source of truth, which
is exactly how a rate card behaves in a real audit.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Services
# --------------------------------------------------------------------------
# Each service: carrier, mode, zone coverage, weight ceiling, transit days by
# zone, DIM divisor, and the coefficients that generate the base rate table.
#
# base(zone, weight) = a + b*(zone-2) + weight * (c + d*(zone-2))
# --------------------------------------------------------------------------

SERVICES = {
    "NEX_GROUND": dict(
        carrier="NationalExpress", mode="Ground", min_zone=2, max_zone=8,
        max_weight=70, dim_divisor=139, residential_ok=True,
        a=8.76, b=1.10, c=0.420, d=0.0850,
        transit={2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 4, 8: 5},
    ),
    "NEX_2DAY": dict(
        carrier="NationalExpress", mode="Air", min_zone=2, max_zone=8,
        max_weight=70, dim_divisor=139, residential_ok=True,
        a=22.40, b=2.30, c=1.050, d=0.2200,
        transit={2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
    ),
    "NEX_OVERNIGHT": dict(
        carrier="NationalExpress", mode="Air", min_zone=2, max_zone=8,
        max_weight=70, dim_divisor=139, residential_ok=True,
        a=38.50, b=4.10, c=1.900, d=0.4200,
        transit={2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1},
    ),
    "CPX_GROUND": dict(
        carrier="ContinentalParcel", mode="Ground", min_zone=2, max_zone=8,
        max_weight=70, dim_divisor=139, residential_ok=True,
        a=8.20, b=1.28, c=0.455, d=0.0810,
        transit={2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5},
    ),
    "CPX_SAVER": dict(
        carrier="ContinentalParcel", mode="Air", min_zone=2, max_zone=8,
        max_weight=70, dim_divisor=139, residential_ok=True,
        a=17.80, b=1.90, c=0.780, d=0.1700,
        transit={2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3},
    ),
    "PSD_ECONOMY": dict(
        carrier="PostalDirect", mode="Postal", min_zone=2, max_zone=8,
        max_weight=10, dim_divisor=166, residential_ok=True,
        a=7.20, b=0.62, c=0.360, d=0.0500,
        transit={2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 8},
    ),
    "RGX_REGIONAL": dict(
        carrier="RegionalGrid", mode="Ground", min_zone=2, max_zone=4,
        max_weight=50, dim_divisor=166, residential_ok=True,
        a=7.10, b=0.95, c=0.350, d=0.0700,
        transit={2: 1, 3: 1, 4: 2},
    ),
}

SERVICE_LIST = list(SERVICES.keys())
SERVICE_IDX = {s: i for i, s in enumerate(SERVICE_LIST)}
CARRIERS = sorted({v["carrier"] for v in SERVICES.values()})

MAX_ZONE = 8
MAX_WEIGHT = 70

# --------------------------------------------------------------------------
# Accessorial schedule
# --------------------------------------------------------------------------
ACCESSORIALS = {
    "residential": {"Ground": 5.95, "Air": 5.20, "Postal": 0.00},
    "das": 4.85,                 # delivery area surcharge, residential
    "das_extended": 9.10,        # extended / remote
    "das_commercial": 3.95,
    "additional_handling": 17.50,
    "large_package": 145.00,
    "address_correction": 21.00,
}

# Peak-season residential surcharge, applied Oct 15 - Jan 15 by weight band.
PEAK_RESIDENTIAL = {"light": 1.65, "mid": 3.10, "heavy": 6.00}

# Additional-handling triggers
AH_LONGEST_IN = 48.0
AH_SECOND_IN = 30.0
AH_WEIGHT_LB = 50.0
LARGE_PACKAGE_LG_IN = 105.0  # length + girth

# --------------------------------------------------------------------------
# Fuel surcharge: percentage of (base + fuel-bearing accessorials), by month.
# Ground and air indices move together but at different levels, as they do in
# practice because they are pegged to different fuel benchmarks.
# --------------------------------------------------------------------------
FUEL_GROUND = {
    1: 0.1550, 2: 0.1585, 3: 0.1620, 4: 0.1675, 5: 0.1710, 6: 0.1665,
    7: 0.1590, 8: 0.1545, 9: 0.1580, 10: 0.1640, 11: 0.1725, 12: 0.1780,
}
FUEL_AIR = {
    1: 0.2050, 2: 0.2095, 3: 0.2140, 4: 0.2210, 5: 0.2265, 6: 0.2200,
    7: 0.2110, 8: 0.2055, 9: 0.2100, 10: 0.2175, 11: 0.2280, 12: 0.2350,
}
FUEL_POSTAL = {m: 0.0 for m in range(1, 13)}

FUEL_BY_MODE = {"Ground": FUEL_GROUND, "Air": FUEL_AIR, "Postal": FUEL_POSTAL}

# --------------------------------------------------------------------------
# Next-year general rate increase (GRI) assumptions, used by the GRI module.
# --------------------------------------------------------------------------
GRI = {
    "base_pct": {"Ground": 0.0590, "Air": 0.0620, "Postal": 0.0480},
    "accessorial_pct": {
        "residential": 0.072, "das": 0.095, "das_extended": 0.110,
        "das_commercial": 0.085, "additional_handling": 0.088,
        "large_package": 0.065, "address_correction": 0.060,
    },
    "fuel_shift_pp": 0.008,          # fuel index up 0.8 points across the year
    "dim_divisor_change": {"NEX_GROUND": 130, "CPX_GROUND": 130},
}


# --------------------------------------------------------------------------
# Rate table construction
# --------------------------------------------------------------------------
def build_rate_table() -> np.ndarray:
    """Return array [service, zone, weight_lb] of published base charges.

    Index with rate_table[svc_idx, zone, billable_weight]. Zone 0/1 and weight 0
    rows exist only so the array can be indexed directly without offsetting.
    Cells outside a service's coverage are NaN, which makes an out-of-coverage
    quote fail loudly instead of silently pricing at zero.
    """
    table = np.full((len(SERVICE_LIST), MAX_ZONE + 1, MAX_WEIGHT + 1), np.nan)
    zones = np.arange(MAX_ZONE + 1)
    weights = np.arange(MAX_WEIGHT + 1)
    zz, ww = np.meshgrid(zones, weights, indexing="ij")

    for name, spec in SERVICES.items():
        i = SERVICE_IDX[name]
        z_off = np.clip(zz - 2, 0, None)
        rates = spec["a"] + spec["b"] * z_off + ww * (spec["c"] + spec["d"] * z_off)
        # Light-weight floor: sub-2 lb packages do not price below the minimum.
        rates = np.maximum(rates, spec["a"] * 0.92)
        valid = (
            (zz >= spec["min_zone"]) & (zz <= spec["max_zone"])
            & (ww >= 1) & (ww <= spec["max_weight"])
        )
        table[i] = np.where(valid, rates, np.nan)
    return table


RATE_TABLE = build_rate_table()

SERVICE_MODE = np.array([SERVICES[s]["mode"] for s in SERVICE_LIST])
SERVICE_CARRIER = np.array([SERVICES[s]["carrier"] for s in SERVICE_LIST])
DIM_DIVISOR = np.array([SERVICES[s]["dim_divisor"] for s in SERVICE_LIST], dtype=float)
MAX_WEIGHT_BY_SVC = np.array([SERVICES[s]["max_weight"] for s in SERVICE_LIST], dtype=float)

TRANSIT_TABLE = np.full((len(SERVICE_LIST), MAX_ZONE + 1), 99, dtype=int)
for _name, _spec in SERVICES.items():
    for _z, _d in _spec["transit"].items():
        TRANSIT_TABLE[SERVICE_IDX[_name], _z] = _d


# --------------------------------------------------------------------------
# Negotiated discounts
# --------------------------------------------------------------------------
# Tier is a function of annual volume with the carrier. Air is discounted less
# than ground, and the postal consolidator prices off a flat published card.
DISCOUNT_TIERS = {
    "T1": {"Ground": 0.42, "Air": 0.31, "Postal": 0.06},
    "T2": {"Ground": 0.36, "Air": 0.26, "Postal": 0.05},
    "T3": {"Ground": 0.28, "Air": 0.19, "Postal": 0.04},
    "T4": {"Ground": 0.19, "Air": 0.12, "Postal": 0.02},
}


def discount_for(tier: str, mode: str) -> float:
    return DISCOUNT_TIERS[tier][mode]
