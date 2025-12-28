#!/usr/bin/env python3
"""
Normalize NFCU raw transactions into a clean CSV for downstream tax tooling.

Input:
  out/tax_text/nfcu/2025/nfcu_transactions_raw.csv

Output:
  notes/tax/work/parsed/2025/nfcu_transactions.normalized.csv
  notes/tax/work/parsed/2025/nfcu_transactions.normalized.jsonl (optional audit trail)

Rules:
- amount_raw -> amount (Decimal-ish string "123.45")
- signed_amount:
    credit -> +amount
    debit  -> -amount
    blank/unknown -> blank
- stable txn_id (sha256 + date + account + description + amount_raw) hashed
- keep statement period fields

Safe:
- refuses to run if input missing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Any


def norm_space(s: str) -> str:
    return " ".join((s or "").strip().split())


def parse_amount(s: str) -> str:
    """
    Normalize things like:
      "1,234.56" -> "1234.56"
      "$12.00"   -> "12.00"
      ""         -> ""
    """
    s = (s or "").strip()
    if not s:
        return ""
    s = s.replace("$", "").replace(",", "")
    return s


def sign_amount(amount: str, direction: str) -> str:
    amount = (amount or "").strip()
    direction = (direction or "").strip().lower()
    if not amount:
        return ""
    if direction == "credit":
        return amount
    if direction == "debit":
        # avoid "--" if already negative for any reason
        return amount if amount.startswith("-") else f"-{amount}"
    return ""


def make_txn_id(row: Dict[str, Any]) -> str:
    base = "|".join([
        (row.get("sha256") or "").strip(),
        (row.get("statement_date") or "").strip(),
        norm_space(row.get("account") or ""),
        norm_space(row.get("description") or ""),
        (row.get("amount_raw") or "").strip(),
    ])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]  # short but collision-resistant enough


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="out/tax_text/nfcu/2025/nfcu_transactions_raw.csv")
    ap.add_argument("--out-csv", default="notes/tax/work/parsed/2025/nfcu_transactions.normalized.csv")
    ap.add_argument("--out-jsonl", default="notes/tax/work/parsed/2025/nfcu_transactions.normalized.jsonl")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    out_jsonl = Path(args.out_jsonl)

    if not in_path.exists():
        print(f"ERROR: missing input: {in_path}")
        print("Did you run: python tools/nfcu_extract_transactions_raw.py ?")
        return 2

    rows = list(csv.DictReader(in_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        print("ERROR: input CSV has no rows")
        return 2

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "txn_id",
        "date",
        "account",
        "description",
        "amount",
        "direction",
        "signed_amount",
        "statement_period_start",
        "statement_period_end",
        "source_name",
        "sha256",
        "txt_rel",
        "meta_rel",
    ]

    wrote = 0
    with out_csv.open("w", encoding="utf-8", newline="") as f_csv, out_jsonl.open("w", encoding="utf-8", newline="\n") as f_j:
        w = csv.DictWriter(f_csv, fieldnames=fieldnames)
        w.writeheader()

        for r in rows:
            amount = parse_amount(r.get("amount_raw", ""))
            direction = (r.get("direction_guess") or "").strip().lower()

            out = {
                "txn_id": make_txn_id(r),
                "date": (r.get("statement_date") or "").strip(),
                "account": norm_space(r.get("account") or ""),
                "description": norm_space(r.get("description") or ""),
                "amount": amount,
                "direction": direction,
                "signed_amount": sign_amount(amount, direction),
                "statement_period_start": (r.get("statement_period_start") or "").strip(),
                "statement_period_end": (r.get("statement_period_end") or "").strip(),
                "source_name": (r.get("source_name") or "").strip(),
                "sha256": (r.get("sha256") or "").strip(),
                "txt_rel": (r.get("txt_rel") or "").strip(),
                "meta_rel": (r.get("meta_rel") or "").strip(),
            }

            w.writerow(out)
            f_j.write(json.dumps(out, ensure_ascii=False) + "\n")
            wrote += 1

    print("---- NFCU Normalize ----")
    print("IN :", in_path)
    print("OUT:", out_csv)
    print("ROWS:", wrote)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
