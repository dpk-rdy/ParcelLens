"""
Run the SQL analysis pack against the generated data.

    python run_queries.py            # run all
    python run_queries.py 03         # run one by number prefix

The shipment and invoice files are parquet, so DuckDB reads them directly with
no load step. The audited view is rebuilt in Python because the audit logic is
not expressible in SQL, and re-implementing it there would give two versions of
the same rules that quietly drift apart.
"""

from __future__ import annotations

import glob
import os
import sys

import duckdb
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from parcellens import audit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SQL = os.path.join(HERE, "sql")


def connect() -> duckdb.DuckDBPyConnection:
    for f in ("shipments.parquet", "invoices.parquet"):
        if not os.path.exists(os.path.join(DATA, f)):
            raise SystemExit("No data yet. Run: python run_pipeline.py")

    con = duckdb.connect()
    con.execute(f"CREATE VIEW shipments AS SELECT * FROM "
                f"'{DATA}/shipments.parquet'")
    con.execute(f"CREATE VIEW invoices AS SELECT * FROM "
                f"'{DATA}/invoices.parquet'")

    ship = con.execute("SELECT * FROM shipments").df()
    inv = con.execute("SELECT * FROM invoices").df()
    audited = audit.run_audit(ship, inv)  # noqa: F841 - registered below
    con.register("audited", audited)
    return con


def main():
    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    con = connect()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    for path in sorted(glob.glob(f"{SQL}/*.sql")):
        name = os.path.basename(path)
        if wanted and not name.startswith(wanted):
            continue
        sql = open(path).read()
        header = [ln for ln in sql.splitlines() if ln.startswith("--")]
        print("\n" + "=" * 78)
        print(name)
        for ln in header[:3]:
            print(" " + ln.lstrip("- "))
        print("=" * 78)
        print(con.execute(sql).df().to_string(index=False))


if __name__ == "__main__":
    main()
