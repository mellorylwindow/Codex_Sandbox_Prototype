#!/usr/bin/env python3
"""
Patch Step 12 output for known OCR quirks, without touching earlier steps.

Fixes:
- direction missing on "Zelle GR ..." (OCR for "Zelle CR") -> credit

Input:
  notes/tax/work/parsed/2025/nfcu_transactions.final.csv

Output:
  notes/tax/work/parsed/2025/nfcu_transactions.final.csv (optionally with --backup)
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from decimal import Decimal
from datetime import datetime
from pathlib import Path


def D(x: str) -> Decimal:
    x = (x or "").strip()
    if not x:
        return Decimal("0.00")
    return Decimal(x)


def fmt(x: Decimal) -> str:
    return f"{x:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="notes/tax/work/parsed/2025/nfcu_transactions.final.csv")
    ap.add_argument("--out", dest="out_csv", default="notes/tax/work/parsed/2025/nfcu_transactions.final.csv")
    ap.add_argument("--backup", action="store_true", help="Write a timestamped .bak copy of the original output")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)

    if not in_csv.exists():
        print(f"ERROR: missing input: {in_csv}")
        return 2

    if args.backup and out_csv.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = out_csv.with_suffix(out_csv.suffix + f".{ts}.bak")
        shutil.copy2(out_csv, bak)
        print("BACKUP:", bak)

    rows = list(csv.DictReader(in_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        print("ERROR: input has no rows")
        return 2

    patched = 0
    for r in rows:
        direction = (r.get("direction") or "").strip()
        desc = (r.get("description") or "").strip()

        if direction:
            continue

        # OCR quirk: "Zelle CR ..." sometimes becomes "Zelle GR ..."
        if re.search(r"\bzelle\s+gr\b", desc, re.IGNORECASE):
            amt = D(r.get("amount") or "0.00")
            r["direction"] = "credit"
            r["signed_amount"] = fmt(amt)

            # external_signed_amount should be same unless it's an internal transfer
            is_transfer = (r.get("is_transfer") or "").strip().lower() in ("1", "true", "yes")
            if is_transfer:
                r["external_signed_amount"] = "0.00"
            else:
                r["external_signed_amount"] = fmt(amt)

            patched += 1

    # write
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("---- Step 14: Patch final directions ----")
    print("IN :", in_csv)
    print("OUT:", out_csv)
    print("Patched rows:", patched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
