#!/usr/bin/env python3
"""
Build NFCU statement index CSV from out/tax_text/nfcu/2025/_meta/*.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def main() -> int:
    meta_dir = Path("out/tax_text/nfcu/2025/_meta")
    out_csv = Path("out/tax_text/nfcu/2025/nfcu_statements_index.csv")

    if not meta_dir.exists():
        print(f"ERROR: meta folder not found: {meta_dir}")
        print("Did you run tax_extract_text.py already?")
        return 2

    rows: List[Dict[str, Any]] = []
    for p in sorted(meta_dir.glob("*.json")):
        try:
            m = read_json(p)
        except Exception as e:
            print(f"WARN: could not read {p}: {e}")
            continue

        rows.append(
            {
                "sha256": m.get("sha256", ""),
                "source_name": m.get("source_name", ""),
                "statement_file_date": m.get("statement_file_date", ""),
                "statement_tag": m.get("statement_tag", ""),
                "statement_period_start": m.get("statement_period_start", ""),
                "statement_period_end": m.get("statement_period_end", ""),
                "statement_period_raw": m.get("statement_period_raw", ""),
                "resolved_path": m.get("resolved_path", ""),
                "src_rel": m.get("src_rel", ""),
                "ok": m.get("ok", ""),
                "mode": m.get("mode", ""),
            }
        )

    if not rows:
        print(f"ERROR: no meta json files found in: {meta_dir}")
        return 2

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sha256",
        "source_name",
        "statement_file_date",
        "statement_tag",
        "statement_period_start",
        "statement_period_end",
        "statement_period_raw",
        "resolved_path",
        "src_rel",
        "ok",
        "mode",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("WROTE:", out_csv)
    print("ROWS:", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
