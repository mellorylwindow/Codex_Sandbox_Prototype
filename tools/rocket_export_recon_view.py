#!/usr/bin/env python
"""
Export a recon view from Rocket Money export CSV.

Input:  Rocket export CSV (e.g. notes/tax/inbox/rocket_money_export.csv)
Output: notes/tax/work/parsed/<YEAR>/rocket_money.recon.csv

Adds deterministic keys (for forgiving->strict matching later):
- match_key_mk0: YYYY-MM|abs_cents
- match_key_mk1: YYYY-MM-DD|abs_cents
- match_key_mk1m1 / mk1p1: date +/- 1 day | abs_cents
- match_key_mk2: YYYY-MM-DD|abs_cents|merchant12
- txn_key: sha1(date|amount|merchant|desc)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

Q = Decimal("0.01")
_ws = re.compile(r"\s+")
_bad = re.compile(r"[^a-z0-9 ]+")

def q2(x: Decimal) -> str:
    return str(x.quantize(Q, rounding=ROUND_HALF_UP))

def d(x) -> Decimal:
    try:
        s = (x or "").strip()
        if s == "":
            return Decimal("0")
        # remove $ and commas
        s = s.replace("$", "").replace(",", "")
        return Decimal(s)
    except Exception:
        return Decimal("0")

def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = _bad.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    return s

def parse_date_any(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    # Try common Rocket formats first
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    # Last resort: try to coerce YYYY/MM/DD or similar
    s2 = s.replace(".", "/").replace("-", "/")
    parts = s2.split("/")
    if len(parts) == 3:
        a, b, c = parts
        # guess order: if first part has 4 digits -> YYYY/MM/DD else MM/DD/YYYY
        try:
            if len(a) == 4:
                return date(int(a), int(b), int(c))
            return date(int(c), int(a), int(b))
        except Exception:
            return None
    return None

def abs_cents(x: Decimal) -> int:
    cents = (x.copy_abs() * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(cents)

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocket", required=True, help="Rocket Money export CSV path")
    ap.add_argument("--year", default="", help="Force output year folder (else derived from row dates)")
    ap.add_argument("--out", default="", help="Optional explicit output CSV path")
    args = ap.parse_args()

    in_path = Path(args.rocket)

    # Read first pass to determine year if not provided
    derived_year = None
    if not args.year:
        with in_path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                dt = parse_date_any(row.get("Original Date") or row.get("Date") or "")
                if dt:
                    derived_year = dt.year
                    break
    year = str(args.year or derived_year or "unknown")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(f"notes/tax/work/parsed/{year}/rocket_money.recon.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_fields = [
        "source",
        "year",
        "month",
        "date",
        "original_date",
        "account_type",
        "account_name",
        "institution_name",
        "name",
        "custom_name",
        "merchant_norm",
        "amount",
        "abs_amount_cents",
        "category",
        "description",
        "note",
        "match_key_mk0",
        "match_key_mk1",
        "match_key_mk1m1",
        "match_key_mk1p1",
        "match_key_mk2",
        "txn_key",
    ]

    n = 0
    with in_path.open("r", encoding="utf-8-sig", newline="") as f_in, out_path.open("w", encoding="utf-8", newline="\n") as f_out:
        r = csv.DictReader(f_in)
        w = csv.DictWriter(f_out, fieldnames=out_fields)
        w.writeheader()

        for row in r:
            dt = parse_date_any(row.get("Original Date") or row.get("Date") or "")
            if not dt:
                continue

            amt = d(row.get("Amount"))
            cents = abs_cents(amt)

            name = (row.get("Custom Name") or row.get("Name") or "").strip()
            desc = (row.get("Description") or "").strip()
            merch = norm(name) or norm(desc)

            iso = dt.isoformat()
            month = iso[:7]

            mk0 = f"{month}|{cents}"
            mk1 = f"{iso}|{cents}"
            mk1m1 = f"{(dt - timedelta(days=1)).isoformat()}|{cents}"
            mk1p1 = f"{(dt + timedelta(days=1)).isoformat()}|{cents}"
            mk2 = f"{iso}|{cents}|{merch[:12]}"

            txn_key = sha1_hex(f"{iso}|{q2(amt)}|{merch}|{norm(desc)}")

            out = {
                "source": "ROCKET",
                "year": str(dt.year),
                "month": month,
                "date": (row.get("Date") or "").strip(),
                "original_date": (row.get("Original Date") or "").strip(),
                "account_type": (row.get("Account Type") or "").strip(),
                "account_name": (row.get("Account Name") or "").strip(),
                "institution_name": (row.get("Institution Name") or "").strip(),
                "name": (row.get("Name") or "").strip(),
                "custom_name": (row.get("Custom Name") or "").strip(),
                "merchant_norm": merch,
                "amount": q2(amt),
                "abs_amount_cents": str(cents),
                "category": (row.get("Category") or "").strip(),
                "description": desc,
                "note": (row.get("Note") or "").strip(),
                "match_key_mk0": mk0,
                "match_key_mk1": mk1,
                "match_key_mk1m1": mk1m1,
                "match_key_mk1p1": mk1p1,
                "match_key_mk2": mk2,
                "txn_key": txn_key,
            }
            w.writerow(out)
            n += 1

    print("---- Rocket recon export ----")
    print("IN :", in_path)
    print("OUT:", out_path)
    print("rows:", n)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
