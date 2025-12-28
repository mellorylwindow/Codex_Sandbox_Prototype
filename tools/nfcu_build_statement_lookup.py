#!/usr/bin/env python3
"""
Build NFCU statement lookup JSON from nfcu_statements_index.csv.

Writes:
  out/tax_text/nfcu/2025/nfcu_statements_lookup.json

Lookup keys:
  - by_sha[sha] -> row
  - by_period["YYYY-MM-DD__YYYY-MM-DD"] -> row
  - by_end_date["YYYY-MM-DD"] -> row (end dates are unique for statements)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def main() -> int:
    csv_path = Path("out/tax_text/nfcu/2025/nfcu_statements_index.csv")
    out_path = Path("out/tax_text/nfcu/2025/nfcu_statements_lookup.json")

    if not csv_path.exists():
        print(f"ERROR: missing: {csv_path}")
        print("Run: python tools/nfcu_build_statement_index.py")
        return 2

    rows: List[Dict[str, Any]] = list(
        csv.DictReader(csv_path.open("r", encoding="utf-8", newline=""))
    )
    if not rows:
        print("ERROR: CSV has no rows")
        return 2

    by_sha: Dict[str, Any] = {}
    by_period: Dict[str, Any] = {}
    by_end_date: Dict[str, Any] = {}

    for r in rows:
        sha = (r.get("sha256") or "").strip()
        start = (r.get("statement_period_start") or "").strip()
        end = (r.get("statement_period_end") or "").strip()

        if sha:
            by_sha[sha] = r

        if start and end:
            by_period[f"{start}__{end}"] = r

        if end:
            if end in by_end_date:
                print(f"WARN: duplicate end_date {end} for sha {sha}")
            else:
                by_end_date[end] = r

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_from": str(csv_path).replace("\\", "/"),
        "count": len(rows),
        "by_sha": by_sha,
        "by_period": by_period,
        "by_end_date": by_end_date,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    print("WROTE:", out_path)
    print("Rows:", len(rows))
    print("Lookup keys:")
    print(" - by_sha:", len(by_sha))
    print(" - by_period:", len(by_period))
    print(" - by_end_date:", len(by_end_date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
