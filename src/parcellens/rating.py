"""
Rating engine.

Given shipment characteristics and a service, this reproduces what the carrier
*should* charge under contract. Everything else in the project is built on it:
the audit compares invoices against this, the rate-shop optimizer calls it once
per candidate service, and the GRI module calls it against a forward rate card.

The engine is fully vectorized. Rating 250k shipments across 7 candidate
services is 1.75M quotes, which needs to run in seconds, not minutes.
"""

from __future__ import annotations

import numpy as np

from . import ratecards as rc


def billable_weight(actual_lb, length_in, width_in, height_in, svc_idx, dim_divisor=None):
    """Billable weight = max(actual, dimensional), rounded up to the pound.

    Dimensional weight is cubic inches / divisor. A lower divisor means the
    carrier bills more for the same box, which is why divisor changes at GRI
    time are one of the most expensive line items in a parcel contract.
    """
    divisor = rc.DIM_DIVISOR[svc_idx] if dim_divisor is None else dim_divisor
    cubic = length_in * width_in * height_in
    dim_lb = cubic / divisor
    billable = np.maximum(np.ceil(actual_lb), np.ceil(dim_lb))
    return np.clip(billable, 1, rc.MAX_WEIGHT).astype(int)


def additional_handling_flag(length_in, width_in, height_in, actual_lb):
    dims = np.sort(np.stack([length_in, width_in, height_in]), axis=0)
    longest, second = dims[2], dims[1]
    return (
        (longest > rc.AH_LONGEST_IN)
        | (second > rc.AH_SECOND_IN)
        | (actual_lb > rc.AH_WEIGHT_LB)
    )


def large_package_flag(length_in, width_in, height_in):
    dims = np.sort(np.stack([length_in, width_in, height_in]), axis=0)
    longest = dims[2]
    girth = 2 * (dims[0] + dims[1])
    return (longest + girth) > rc.LARGE_PACKAGE_LG_IN


def peak_surcharge(month, day, is_residential, billable_lb):
    """Peak residential surcharge, Oct 15 through Jan 15, by weight band."""
    in_peak = ((month >= 11) | ((month == 10) & (day >= 15))
               | ((month == 1) & (day <= 15)))
    band = np.where(billable_lb <= 5, rc.PEAK_RESIDENTIAL["light"],
                    np.where(billable_lb <= 20, rc.PEAK_RESIDENTIAL["mid"],
                             rc.PEAK_RESIDENTIAL["heavy"]))
    return np.where(in_peak & is_residential, band, 0.0)


def rate_shipments(
    svc_idx,
    zone,
    actual_lb,
    length_in,
    width_in,
    height_in,
    is_residential,
    das_type,          # 0 none, 1 standard, 2 extended, 3 commercial
    month,
    day,
    discount_pct,
    address_correction=None,
    rate_table=None,
    dim_divisor=None,
    fuel_shift=0.0,
    accessorial_multiplier=None,
):
    """Return a dict of cost components plus the total expected charge.

    Charge construction, in the order a carrier applies it:
      1. published base for (service, zone, billable weight)
      2. negotiated discount off base
      3. accessorials, which are not discounted
      4. fuel surcharge, as a percentage of discounted base plus the
         fuel-bearing accessorials (residential and DAS travel with fuel;
         address correction does not)
    """
    table = rc.RATE_TABLE if rate_table is None else rate_table
    n = np.asarray(svc_idx).shape[0]
    address_correction = (np.zeros(n, dtype=bool) if address_correction is None
                          else address_correction)
    amult = {} if accessorial_multiplier is None else accessorial_multiplier

    def acc(key, value):
        return value * (1.0 + amult.get(key, 0.0))

    bw = billable_weight(actual_lb, length_in, width_in, height_in, svc_idx,
                         dim_divisor=dim_divisor)
    published = table[svc_idx, zone, bw]
    base = published * (1.0 - discount_pct)

    mode = rc.SERVICE_MODE[svc_idx]
    is_ground = mode == "Ground"
    is_air = mode == "Air"
    is_postal = mode == "Postal"

    resi_rate = np.where(is_ground, rc.ACCESSORIALS["residential"]["Ground"],
                         np.where(is_air, rc.ACCESSORIALS["residential"]["Air"], 0.0))
    resi = np.where(is_residential, acc("residential", resi_rate), 0.0)

    das = np.select(
        [das_type == 1, das_type == 2, das_type == 3],
        [acc("das", rc.ACCESSORIALS["das"]),
         acc("das_extended", rc.ACCESSORIALS["das_extended"]),
         acc("das_commercial", rc.ACCESSORIALS["das_commercial"])],
        default=0.0,
    )
    das = np.where(is_postal, 0.0, das)

    ah = np.where(
        additional_handling_flag(length_in, width_in, height_in, actual_lb) & ~is_postal,
        acc("additional_handling", rc.ACCESSORIALS["additional_handling"]), 0.0)
    lp = np.where(
        large_package_flag(length_in, width_in, height_in) & ~is_postal,
        acc("large_package", rc.ACCESSORIALS["large_package"]), 0.0)
    # A large package absorbs the additional-handling fee rather than stacking.
    ah = np.where(lp > 0, 0.0, ah)

    addr = np.where(address_correction & ~is_postal,
                    acc("address_correction", rc.ACCESSORIALS["address_correction"]), 0.0)
    peak = peak_surcharge(month, day, is_residential & ~is_postal, bw)

    fuel_pct = np.zeros(n)
    for m in range(1, 13):
        sel = month == m
        fuel_pct = np.where(
            sel & is_ground, rc.FUEL_GROUND[m] + fuel_shift,
            np.where(sel & is_air, rc.FUEL_AIR[m] + fuel_shift, fuel_pct))
    fuel_base = base + resi + das + ah + lp
    fuel = fuel_base * fuel_pct

    accessorials = resi + das + ah + lp + addr + peak
    total = base + accessorials + fuel

    return {
        "billable_lb": bw,
        "published_base": published,
        "base": base,
        "acc_residential": resi,
        "acc_das": das,
        "acc_additional_handling": ah,
        "acc_large_package": lp,
        "acc_address_correction": addr,
        "acc_peak": peak,
        "accessorials": accessorials,
        "fuel_pct": fuel_pct,
        "fuel": fuel,
        "total": total,
    }


def transit_days(svc_idx, zone):
    return rc.TRANSIT_TABLE[svc_idx, zone]


def service_available(svc_idx, zone, actual_lb, length_in, width_in, height_in):
    """Coverage test: zone in range, weight under ceiling, dims transportable."""
    zone_ok = ~np.isnan(rc.RATE_TABLE[svc_idx, zone, 1])
    weight_ok = np.ceil(actual_lb) <= rc.MAX_WEIGHT_BY_SVC[svc_idx]
    dim_lb = (length_in * width_in * height_in) / rc.DIM_DIVISOR[svc_idx]
    dim_ok = np.ceil(dim_lb) <= rc.MAX_WEIGHT_BY_SVC[svc_idx]
    postal = rc.SERVICE_MODE[svc_idx] == "Postal"
    dims = np.sort(np.stack([length_in, width_in, height_in]), axis=0)
    postal_ok = ~postal | ((dims[2] <= 22) & (dims[2] + 2 * (dims[0] + dims[1]) <= 84))
    return zone_ok & weight_ok & dim_ok & postal_ok
