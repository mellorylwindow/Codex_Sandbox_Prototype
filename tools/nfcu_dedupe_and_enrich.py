#!/usr/bin/env python3
"""
Step 11:
- Input:  notes/tax/work/parsed/2025/nfcu_transactions.cleaned.csv
- Output:
    notes/tax/work/parsed/2025/nfcu_transactions.enriched.csv
    notes/tax/work/parsed/2025/nfcu_transactions.flags.csv
    notes/tax/work/parsed/2025/nfcu_transactions.summary_by_month_account.csv

What it does:
- Dedupes exact repeats (date+account+desc+amount+direction)
- Adds:
    month
    txn_key (stable key for dedupe/debug)
    is_transfer
    merchant_guess
    category_guess
- Flags:
    duplicates removed
    unknown direction
    zero amounts
    suspiciously generic descriptions (e.g. just "POS")
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from hashlib import sha1
from pathlib import Path
from typing import Dict, Any, List, Tuple


def norm_space(s: str) -> str:
    return " ".join((s or "").strip().split())


def dec_or_none(s: str) -> Decimal | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    # collapse -0.00 -> 0.00
    if d == Decimal("-0.00") or d == Decimal("0.00"):
        return Decimal("0.00")
    return d


ZELLE_RE = re.compile(r"\bzelle\*([a-z0-9 .'\-]+)", re.IGNORECASE)


def merchant_guess(desc: str) -> str:
    d = desc.strip()

    # Zelle*Name ...
    m = ZELLE_RE.search(d)
    if m:
        return norm_space(m.group(1)).title()

    # "POS Something" (if present)
    if d.lower().startswith("pos "):
        rest = d[4:].strip()
        # if rest is just a number or empty, return blank
        if rest and not re.fullmatch(r"[-\d.,]+", rest):
            return rest.title()
        return ""

    # "Transfer ... <name>" sometimes: "Transfer From Checking Mallow B Stone"
    if d.lower().startswith("transfer "):
        # last chunk sometimes is a person name; keep it mild/optional
        parts = d.split()
        if len(parts) >= 4:
            tail = " ".join(parts[3:])
            # if tail looks like an account word, ignore
            if tail.lower() in {"checking", "savings", "mortgage"}:
                return ""
            return tail.title()
        return ""

    return ""


def category_guess(desc: str) -> Tuple[str, bool]:
    dl = desc.lower().strip()

    # transfers
    if dl.startswith("transfer from") or dl.startswith("transfer to"):
        return "transfer", True

    # zelle
    if "zelle*" in dl:
        return "zelle", False

    # mortgage
    if "mortgage" in dl:
        return "mortgage", False

    # POS adjustments
    if dl.startswith("pos credit adjustment"):
        return "pos_adjustment", False

    # POS purchases
    if dl.startswith("pos"):
        return "pos_purchase", False

    return "unknown", False


def build_txn_key(r: Dict[str, Any]) -> str:
    # stable-ish dedupe key
    base = "|".join([
        (r.get("date") or "").strip(),
        (r.get("account") or "").strip().lower(),
        norm_space(r.get("description") or "").lower(),
        (r.get("amount") or "").strip(),
        (r.get("direction") or "").strip().lower(),
    ])
    # short hash so it’s readable
    return sha1(base.encode("utf-8", errors="replace")).hexdigest()[:16]


def main() -> int:
    in_csv = Path("notes/tax/work/parsed/2025/nfcu_transactions.cleaned.csv")
    out_enriched = Path("notes/tax/work/parsed/2025/nfcu_transactions.enriched.csv")
    out_flags = Path("notes/tax/work/parsed/2025/nfcu_transactions.flags.csv")
    out_summary = Path("notes/tax/work/parsed/2025/nfcu_transactions.summary_by_month_account.csv")

    if not in_csv.exists():
        print(f"ERROR: missing input: {in_csv}")
        print("Run Step 10 first (nfcu_clean_and_summarize.py).")
        return 2

    rows = list(csv.DictReader(in_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        print("ERROR: cleaned CSV has no rows")
        return 2

    out_enriched.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    enriched: List[Dict[str, Any]] = []
    flags: List[Dict[str, Any]] = []

    duplicates = 0
    unknown_dir = 0
    zero_amt = 0
    generic_desc = 0

    for r in rows:
        date = (r.get("date") or "").strip()
        account = (r.get("account") or "").strip()
        desc = norm_space(r.get("description") or "")
        amount_s = (r.get("amount") or "").strip()
        direction = (r.get("direction") or "").strip().lower()
        signed_s = (r.get("signed_amount") or "").strip()

        amt = dec_or_none(amount_s)
        month = date[:7] if len(date) >= 7 else ""

        cat, is_xfer = category_guess(desc)
        merch = merchant_guess(desc)

        txn_key = build_txn_key({"date": date, "account": account, "description": desc, "amount": amount_s, "direction": direction})

        # Deduping on txn_key
        if txn_key in seen:
            duplicates += 1
            flags.append({
                "flag": "duplicate_dropped",
                "txn_key": txn_key,
                "date": date,
                "account": account,
                "amount": amount_s,
                "direction": direction,
                "description": desc,
            })
            continue
        seen.add(txn_key)

        # Flags
        if not direction:
            unknown_dir += 1
            flags.append({
                "flag": "unknown_direction",
                "txn_key": txn_key,
                "date": date,
                "account": account,
                "amount": amount_s,
                "direction": direction,
                "description": desc,
            })

        if amt is not None and amt == Decimal("0.00"):
            zero_amt += 1
            flags.append({
                "flag": "zero_amount",
                "txn_key": txn_key,
                "date": date,
                "account": account,
                "amount": amount_s,
                "direction": direction,
                "description": desc,
            })

        if desc.lower() in {"pos", "transfer", "transfer to", "transfer from"}:
            generic_desc += 1
            flags.append({
                "flag": "generic_description",
                "txn_key": txn_key,
                "date": date,
                "account": account,
                "amount": amount_s,
                "direction": direction,
                "description": desc,
            })

        enriched.append({
            "txn_key": txn_key,
            "month": month,

            "txn_id": r.get("txn_id", ""),
            "date": date,
            "account": account,
            "description": desc,

            "amount": amount_s,
            "direction": direction,
            "signed_amount": signed_s,

            "category_guess": cat,
            "is_transfer": "1" if is_xfer else "0",
            "merchant_guess": merch,

            "statement_period_start": r.get("statement_period_start", ""),
            "statement_period_end": r.get("statement_period_end", ""),
            "source_name": r.get("source_name", ""),
            "sha256": r.get("sha256", ""),
        })

    # Write enriched
    if enriched:
        fieldnames = list(enriched[0].keys())
    else:
        fieldnames = [
            "txn_key","month","txn_id","date","account","description",
            "amount","direction","signed_amount","category_guess","is_transfer","merchant_guess",
            "statement_period_start","statement_period_end","source_name","sha256"
        ]

    with out_enriched.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(enriched)

    # Write flags
    if flags:
        fn = list(flags[0].keys())
        with out_flags.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(flags)
    else:
        # still create empty flags file with headers
        with out_flags.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["flag","txn_key","date","account","amount","direction","description"])
            w.writeheader()

    # Monthly summary by month+account
    agg = defaultdict(lambda: {"count": 0, "credits": Decimal("0.00"), "debits": Decimal("0.00"), "net": Decimal("0.00")})

    for r in enriched:
        month = (r.get("month") or "").strip()
        account = (r.get("account") or "").strip()
        direction = (r.get("direction") or "").strip().lower()
        amt = dec_or_none(r.get("amount") or "")
        if not month or not account or amt is None:
            continue

        key = (month, account)
        agg[key]["count"] += 1

        if direction == "credit":
            agg[key]["credits"] += amt
            agg[key]["net"] += amt
        elif direction == "debit":
            agg[key]["debits"] += amt
            agg[key]["net"] -= amt
        else:
            # unknown direction: ignore totals
            pass

    with out_summary.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["month","account","count","credits","debits","net"])
        w.writeheader()
        for (month, account) in sorted(agg.keys()):
            w.writerow({
                "month": month,
                "account": account,
                "count": agg[(month, account)]["count"],
                "credits": f"{agg[(month, account)]['credits']:.2f}",
                "debits": f"{agg[(month, account)]['debits']:.2f}",
                "net": f"{agg[(month, account)]['net']:.2f}",
            })

    print("---- Step 11: Dedupe + Enrich ----")
    print("IN :", in_csv)
    print("OUT:", out_enriched)
    print("OUT:", out_flags)
    print("OUT:", out_summary)
    print("Input rows:", len(rows))
    print("Enriched rows:", len(enriched))
    print("Duplicates dropped:", duplicates)
    print("Flags: unknown_direction:", unknown_dir)
    print("Flags: zero_amount:", zero_amt)
    print("Flags: generic_description:", generic_desc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
