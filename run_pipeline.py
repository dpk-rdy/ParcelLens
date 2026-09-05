"""
ParcelLens pipeline.

    python run_pipeline.py --shipments 250000

Generates the shipment and invoice data, audits it, optimizes carrier and
service selection, analyzes packaging, builds scorecards, models next year's
rate change, and writes the dashboard and findings.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from parcellens import (audit, charts, generate, packaging, rateshop,  # noqa: E402
                        report, scorecard)

DATA = os.path.join(os.path.dirname(__file__), "data")
OUT = os.path.join(os.path.dirname(__file__), "outputs")
CHARTS = os.path.join(OUT, "charts")

REGIONAL_CAP = 0.18   # share of volume the regional carrier can absorb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipments", type=int, default=250_000)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--seed", type=int, default=generate.RNG_SEED)
    args = ap.parse_args()

    for d in (DATA, OUT, CHARTS):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()

    def step(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

    step(f"generating {args.shipments:,} shipments")
    ship, inv = generate.generate(args.shipments, year=args.year, seed=args.seed)
    ship.to_parquet(f"{DATA}/shipments.parquet", index=False)
    inv.to_parquet(f"{DATA}/invoices.parquet", index=False)

    step("auditing invoices")
    audited = audit.run_audit(ship, inv)
    recovery = audit.recovery_summary(audited)
    detection = audit.score_detection(audited)
    divisors = audited.attrs["learned_divisors"]

    step("optimizing carrier and service mix")
    uncon = rateshop.optimize(ship)
    con = rateshop.optimize(ship, capacity_caps={"RegionalGrid": REGIONAL_CAP})
    migration = rateshop.migration_matrix(con)

    step("analyzing packaging and dimensional weight")
    pack_detail, pack_summary = packaging.analyze(ship)
    pack_client = packaging.by_client(pack_detail)

    step("building scorecards")
    carriers = scorecard.carrier_scorecard(ship, audited)
    clients = scorecard.client_scorecard(ship, audited, con, pack_detail)

    step("modeling next year's rate change")
    levers, gri = scorecard.gri_impact(ship)

    # ---------------- headline numbers ----------------
    air_services = [k for k, v in generate.rc.SERVICES.items() if v["mode"] == "Air"]
    ground_services = [k for k, v in generate.rc.SERVICES.items() if v["mode"] != "Air"]
    air_to_ground = con[con["current_service"].isin(air_services)
                        & con["optimal_service"].isin(ground_services)
                        & con["switched"]]
    stats = {
        "shipments": len(ship),
        "invoice_lines": len(audited),
        "carriers": ship["carrier"].nunique(),
        "clients": ship["client"].nunique(),
        "invoiced_spend": float(audited["billed_total"].sum()),
        "contracted_spend": float(ship["expected_total"].sum()),
        "recovery_usd": float(audited["recovery_usd"].sum()),
        "gsr_usd": float(audited["gsr_refund_usd"].sum()),
        "rateshop_savings": float(con["savings"].sum()),
        "rateshop_savings_uncapped": float(uncon["savings"].sum()),
        "packaging_savings": float(pack_summary["packaging_savings"]),
        "exception_lines": int((audited["exception_type"] != "none").sum()),
        "exception_rate": float((audited["exception_type"] != "none").mean()),
        "dim_billed_share": pack_summary["dim_billed_share"],
        "avg_cube_utilization": pack_summary["avg_cube_utilization"],
        "dim_penalty_spend": pack_summary["dim_penalty_spend"],
        "gri_pct": gri["summary"]["increase_pct"],
        "gri_usd": gri["summary"]["increase_usd"],
        "gri_headline": 0.0590,
        "regional_cap": REGIONAL_CAP,
        "air_to_ground_pkgs": int(len(air_to_ground)),
        "air_to_ground_savings": float(air_to_ground["savings"].sum()),
    }
    stats["total_opportunity"] = (stats["recovery_usd"] + stats["gsr_usd"]
                                  + stats["rateshop_savings"]
                                  + stats["packaging_savings"])
    stats["opportunity_pct"] = stats["total_opportunity"] / stats["invoiced_spend"]

    step("rendering charts")
    svg = {
        "recovery": charts.recovery_by_exception(recovery, CHARTS),
        "trend": charts.monthly_trend(ship, audited, CHARTS),
        "carriers": charts.carrier_costs(carriers, CHARTS),
        "rateshop": charts.rateshop_savings(uncon, con, CHARTS),
        "dim": charts.dim_exposure(pack_detail, pack_client, CHARTS),
        "gri": charts.gri_levers(levers, gri["summary"], CHARTS),
    }

    ctx = dict(stats=stats, recovery=recovery, detection=detection,
               clients=clients, carriers=carriers, migration=migration,
               divisors=divisors, svg=svg, year=args.year,
               generated=date.today().isoformat())

    step("writing outputs")
    with open(f"{OUT}/dashboard.html", "w") as fh:
        fh.write(report.build_dashboard(ctx))
    with open(f"{OUT}/findings.md", "w") as fh:
        fh.write(report.build_findings(ctx))

    (audited[audited["exception_type"] != "none"]
     [["invoice_line_id", "tracking_id", "client", "carrier", "service",
       "invoice_date", "billed_total", "expected_total", "variance_usd",
       "exception_type", "recovery_usd"]]
     .sort_values("recovery_usd", ascending=False)
     .to_csv(f"{OUT}/exception_register.csv", index=False))

    recovery.to_csv(f"{OUT}/recovery_summary.csv", index=False)
    detection.to_csv(f"{OUT}/detection_scorecard.csv", index=False)
    carriers.to_csv(f"{OUT}/carrier_scorecard.csv", index=False)
    clients.to_csv(f"{OUT}/client_scorecard.csv", index=False)
    migration.to_csv(f"{OUT}/service_migration.csv", index=False)
    pack_client.to_csv(f"{OUT}/packaging_by_client.csv", index=False)
    levers.to_csv(f"{OUT}/rate_change_levers.csv", index=False)
    pd.DataFrame([stats]).T.rename(columns={0: "value"}).to_csv(f"{OUT}/headline_stats.csv")

    step("done")
    print()
    print(f"  Invoiced spend        {stats['invoiced_spend']:>14,.0f}")
    print(f"  Billing recovery      {stats['recovery_usd']:>14,.0f}")
    print(f"  Service refunds       {stats['gsr_usd']:>14,.0f}")
    print(f"  Mix optimization      {stats['rateshop_savings']:>14,.0f}"
          f"   (uncapped {stats['rateshop_savings_uncapped']:,.0f})")
    print(f"  Packaging             {stats['packaging_savings']:>14,.0f}")
    print(f"  {'Total opportunity':<21} {stats['total_opportunity']:>14,.0f}"
          f"   ({stats['opportunity_pct']:.2%} of spend)")
    print()
    print(f"  Billing accuracy      {1-stats['exception_rate']:>14.2%}")
    print(f"  DIM-billed share      {stats['dim_billed_share']:>14.1%}")
    print(f"  Next-year increase    {stats['gri_pct']:>14.2%}"
          f"   (headline {stats['gri_headline']:.2%})")


if __name__ == "__main__":
    main()
