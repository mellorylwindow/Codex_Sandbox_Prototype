#!/usr/bin/env python3
"""
Step 12:
- Input : notes/tax/work/parsed/2025/nfcu_transactions.enriched.csv
- Output:
    notes/tax/work/parsed/2025/nfcu_transactions.final.csv
    notes/tax/work/parsed/2025/nfcu_transactions.summary_external_by_month.csv
    notes/tax/work/parsed/2025/nfcu_transactions.summary_external_by_month_account.csv

What it does:
- Infers missing direction from description (DR/CR, Paid To, Zelle CR, etc.)
- Recomputes signed_amount when needed
- Reclassifies "Transfer To Mortgage" as mortgage (NOT an internal transfer)
- Cancels internal transfers:
    - If category is "transfer" and NOT mortgage, then external_signed_amount = 0
    - Also attempts to pair transfers (credit/debit) for reporting only
"""

from __future__ import annotations

import csv
import re
from decimal import Decimal, InvalidOperation
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple


def d(s: str) -> Decimal | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        x = Decimal(s)
    except InvalidOperation:
        return None
    if x == Decimal("-0.00"):
        return Decimal("0.00")
    return x


def infer_direction(desc: str) -> str:
    dl = (desc or "").lower()

    # strongest explicit markers
    if re.search(r"\bdr\b", dl):  # "Adjustment DR"
        return "debit"
    if re.search(r"\bcr\b", dl):  # "Zelle CR ..."
        return "credit"

    # common banking phrases
    if "paid to" in dl:
        return "debit"
    if "paid from" in dl:
        return "credit"

    # zelle without DR/CR (fallback)
    if "zelle" in dl:
        # many statements label credits as "credit adjustment" etc
        if "credit" in dl:
            return "credit"
        if "debit" in dl:
            return "debit"

    # transfers
    if dl.startswith("transfer from"):
        return "credit"
    if dl.startswith("transfer to"):
        return "debit"

    return ""


def is_mortgage(desc: str) -> bool:
    return "mortgage" in (desc or "").lower()


def main() -> int:
    in_csv = Path("notes/tax/work/parsed/2025/nfcu_transactions.enriched.csv")
    out_final = Path("notes/tax/work/parsed/2025/nfcu_transactions.final.csv")
    out_m = Path("notes/tax/work/parsed/2025/nfcu_transactions.summary_external_by_month.csv")
    out_ma = Path("notes/tax/work/parsed/2025/nfcu_transactions.summary_external_by_month_account.csv")

    if not in_csv.exists():
        print(f"ERROR: missing input: {in_csv}")
        print("Run Step 11 first.")
        return 2

    rows: List[Dict[str, Any]] = list(csv.DictReader(in_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        print("ERROR: enriched CSV has no rows")
        return 2

    out_final.parent.mkdir(parents=True, exist_ok=True)

    inferred = 0
    still_unknown = 0
    transfers_total = 0
    transfers_canceled = 0
    mortgage_transfers = 0

    # For pairing report (optional)
    transfer_buckets = defaultdict(list)  # (date, amount) -> list of indices

    final_rows: List[Dict[str, Any]] = []

    for r in rows:
        desc = (r.get("description") or "").strip()
        direction = (r.get("direction") or "").strip().lower()
        amount = d(r.get("amount") or "")

        cat = (r.get("category_guess") or "").strip()
        # fix classification: mortgage should win even if it starts with "Transfer To"
        if is_mortgage(desc):
            if cat == "transfer":
                mortgage_transfers += 1
            cat = "mortgage"

        if not direction:
            guess = infer_direction(desc)
            if guess:
                direction = guess
                inferred += 1
            else:
                still_unknown += 1

        signed = d(r.get("signed_amount") or "")
        if amount is not None and (signed is None or r.get("signed_amount") in ("", None)):
            # recompute signed_amount from amount + direction
            if direction == "debit":
                signed = -amount
            elif direction == "credit":
                signed = amount
            else:
                signed = None

        # external view: cancel internal transfers (but NOT mortgage)
        external_signed = signed
        if cat == "transfer" and not is_mortgage(desc):
            transfers_total += 1
            external_signed = Decimal("0.00") if signed is not None else Decimal("0.00")
            transfers_canceled += 1

            # bucket for pairing stats (date+amount)
            date = (r.get("date") or "").strip()
            amt_key = f"{amount:.2f}" if amount is not None else ""
            transfer_buckets[(date, amt_key)].append(r)

        # month convenience
        date = (r.get("date") or "").strip()
        month = date[:7] if len(date) >= 7 else (r.get("month") or "").strip()

        out = dict(r)
        out["month"] = month
        out["category_guess"] = cat
        out["direction"] = direction
        out["signed_amount"] = f"{signed:.2f}" if isinstance(signed, Decimal) else (out.get("signed_amount") or "")
        out["external_signed_amount"] = f"{external_signed:.2f}" if isinstance(external_signed, Decimal) else ""
        final_rows.append(out)

    # Write final CSV
    fieldnames = list(final_rows[0].keys()) if final_rows else []
    if "external_signed_amount" not in fieldnames:
        fieldnames.append("external_signed_amount")

    with out_final.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(final_rows)

    # Summaries (external only)
    by_month = defaultdict(lambda: {"count": 0, "net": Decimal("0.00")})
    by_month_acct = defaultdict(lambda: {"count": 0, "net": Decimal("0.00")})

    for r in final_rows:
        month = (r.get("month") or "").strip()
        acct = (r.get("account") or "").strip()
        ex = d(r.get("external_signed_amount") or "")
        if not month or ex is None:
            continue
        by_month[month]["count"] += 1
        by_month[month]["net"] += ex

        by_month_acct[(month, acct)]["count"] += 1
        by_month_acct[(month, acct)]["net"] += ex

    with out_m.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["month", "count", "external_net"])
        w.writeheader()
        for m in sorted(by_month.keys()):
            w.writerow({
                "month": m,
                "count": by_month[m]["count"],
                "external_net": f"{by_month[m]['net']:.2f}",
            })

    with out_ma.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["month", "account", "count", "external_net"])
        w.writeheader()
        for (m, a) in sorted(by_month_acct.keys()):
            w.writerow({
                "month": m,
                "account": a,
                "count": by_month_acct[(m, a)]["count"],
                "external_net": f"{by_month_acct[(m, a)]['net']:.2f}",
            })

    print("---- Step 12: Finalize ----")
    print("IN :", in_csv)
    print("OUT:", out_final)
    print("OUT:", out_m)
    print("OUT:", out_ma)
    print("Rows:", len(final_rows))
    print("Direction inferred:", inferred)
    print("Still unknown direction:", still_unknown)
    print("Transfers canceled (non-mortgage):", transfers_canceled, "of", transfers_total)
    print("Mortgage reclassified from transfer:", mortgage_transfers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
