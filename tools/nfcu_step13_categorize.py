#!/usr/bin/env python3
"""
Step 13: Categorize + Summarize

Input:
  notes/tax/work/parsed/2025/nfcu_transactions.final.csv

Outputs:
  notes/tax/work/parsed/2025/nfcu_transactions.categorized.csv
  notes/tax/work/parsed/2025/nfcu_summary_by_month_category.csv
  notes/tax/work/parsed/2025/nfcu_flags_step13.csv
"""

from __future__ import annotations

import csv
import re
from decimal import Decimal, InvalidOperation
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List


def dec(s: str) -> Decimal:
    s = (s or "").strip()
    if not s:
        return Decimal("0.00")
    try:
        return Decimal(s)
    except InvalidOperation:
        return Decimal("0.00")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def guess_counterparty(desc: str) -> str:
    d = desc.lower()

    # Zelle: "Zelle*name"
    m = re.search(r"zelle\*([a-z0-9 .'-]+)", d)
    if m:
        return norm(m.group(1)).title()

    # Zelle: "Zelle CR/DR/GR Name"  (GR shows up via OCR)
    m = re.search(r"zelle\s+(?:cr|dr|gr)\s+([a-z0-9 .'-]+)", d)
    if m:
        return norm(m.group(1)).title()

    # ACH Paid To/From
    m = re.search(r"ach\s+paid\s+to\s+([a-z0-9 .'-]+)", d)
    if m:
        return norm(m.group(1)).title()

    m = re.search(r"ach\s+paid\s+from\s+([a-z0-9 .'-]+)", d)
    if m:
        return norm(m.group(1)).title()

    if "visa direct" in d:
        return "Visa Direct"

    return ""


def categorize(desc: str, cat_guess: str) -> str:
    d = (desc or "").lower().strip()
    cg = (cat_guess or "").lower().strip()

    # mortgage first
    if "mortgage" in d or cg == "mortgage":
        return "mortgage"

    # internal transfers
    if cg == "transfer" or d.startswith("transfer "):
        return "transfer_internal"

    # mobile deposits (this fixes your 6 "other" flags)
    if "edeposit" in d or "scan/mobile" in d or "mobile" in d and "deposit" in d:
        return "mobile_deposit"

    # zelle (including OCR "GR")
    if "zelle" in d:
        return "zelle"

    # ACH / EFT
    if d.startswith("ach ") or " ach " in d:
        return "ach"

    # fees
    if "fee" in d or "service charge" in d:
        return "fees"

    # interest/dividends
    if "dividend" in d or "interest" in d:
        return "interest"

    # card / POS
    if d.startswith("pos ") or d.startswith("debit card") or "debit card" in d:
        return "card_pos"

    # adjustments
    if "adjustment" in d or "credit adjustment" in d or "debit adjustment" in d:
        return "adjustment"

    return "other"


def main() -> int:
    in_csv = Path("notes/tax/work/parsed/2025/nfcu_transactions.final.csv")
    out_csv = Path("notes/tax/work/parsed/2025/nfcu_transactions.categorized.csv")
    out_sum = Path("notes/tax/work/parsed/2025/nfcu_summary_by_month_category.csv")
    out_flags = Path("notes/tax/work/parsed/2025/nfcu_flags_step13.csv")

    if not in_csv.exists():
        print(f"ERROR: missing: {in_csv}")
        print("Run Step 12 first.")
        return 2

    rows: List[Dict[str, Any]] = list(csv.DictReader(in_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        print("ERROR: final CSV has no rows")
        return 2

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    flagged: List[Dict[str, Any]] = []
    summary = defaultdict(lambda: {"count": 0, "external_net": Decimal("0.00")})

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        desc = (r.get("description") or "").strip()
        month = (r.get("month") or "").strip()
        cat_guess = (r.get("category_guess") or "").strip()

        cat_final = categorize(desc, cat_guess)
        cp = guess_counterparty(desc)
        ex = dec(r.get("external_signed_amount") or "0.00")

        if not (r.get("direction") or "").strip():
            flagged.append({
                "flag": "direction_missing",
                "date": r.get("date",""),
                "account": r.get("account",""),
                "amount": r.get("amount",""),
                "description": desc,
                "sha256": r.get("sha256",""),
            })

        if cat_final == "other":
            flagged.append({
                "flag": "category_other",
                "date": r.get("date",""),
                "account": r.get("account",""),
                "amount": r.get("amount",""),
                "description": desc,
                "sha256": r.get("sha256",""),
            })

        out = dict(r)
        out["category_final"] = cat_final
        out["counterparty_guess"] = cp
        out_rows.append(out)

        if month:
            key = (month, cat_final)
            summary[key]["count"] += 1
            summary[key]["external_net"] += ex

    fieldnames = list(out_rows[0].keys())
    for extra in ("category_final", "counterparty_guess"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    with out_sum.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["month", "category_final", "count", "external_net"])
        w.writeheader()
        for (m, c) in sorted(summary.keys()):
            w.writerow({
                "month": m,
                "category_final": c,
                "count": summary[(m, c)]["count"],
                "external_net": f"{summary[(m, c)]['external_net']:.2f}",
            })

    with out_flags.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["flag", "date", "account", "amount", "description", "sha256"])
        w.writeheader()
        w.writerows(flagged)

    print("---- Step 15: Re-categorize ----")
    print("IN :", in_csv)
    print("OUT:", out_csv)
    print("OUT:", out_sum)
    print("OUT:", out_flags)
    print("Rows:", len(out_rows))
    print("Flags:", len(flagged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
