#!/usr/bin/env python
"""
Export NFCU recon view from categorized CSV.

Adds:
- external_amount (zeros transfer_internal)
- abs_amount_cents
- deterministic match keys:
  - match_key_loose      = posted_date|abs_amount_cents
  - match_key_loose_m1   = (posted_date-1)|abs_amount_cents
  - match_key_loose_p1   = (posted_date+1)|abs_amount_cents
  - match_key_strict     = posted_date|abs_amount_cents|merchant12
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

Q = Decimal("0.01")

def d(x) -> Decimal:
    try:
        return Decimal((x or "").strip() or "0")
    except Exception:
        return Decimal("0")

def q2(x: Decimal) -> str:
    return str(x.quantize(Q, rounding=ROUND_HALF_UP))

_ws = re.compile(r"\s+")
_bad = re.compile(r"[^a-z0-9 ]+")

def norm_merchant(s: str) -> str:
    s = (s or "").lower().strip()
    s = _bad.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    return s

def parse_ymd(s: str) -> date:
    # expects YYYY-MM-DD
    y, m, d_ = s.split("-")
    return date(int(y), int(m), int(d_))

def ymd(dt: date) -> str:
    return dt.isoformat()

def abs_cents(x: Decimal) -> int:
    # deterministic, avoid float
    cents = (x.copy_abs() * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(cents)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2025")
    args = ap.parse_args()
    year = str(args.year)

    in_path = Path(f"notes/tax/work/parsed/{year}/nfcu_transactions.categorized.csv")
    out_path = Path(f"notes/tax/work/parsed/{year}/nfcu_transactions.recon.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise SystemExit(f"Missing input: {in_path}")

    # Output schema (stable + friendly for matching)
    out_fields = [
        "source",
        "year",
        "month",
        "posted_date",
        "account",
        "description",
        "merchant_norm",
        "category_final",
        "signed_amount",
        "external_amount",
        "abs_amount_cents",
        "match_key_loose",
        "match_key_loose_m1",
        "match_key_loose_p1",
        "match_key_strict",
        "txn_key",
    ]

    n = 0
    with in_path.open("r", encoding="utf-8", newline="") as f_in, out_path.open("w", encoding="utf-8", newline="\n") as f_out:
        r = csv.DictReader(f_in)
        w = csv.DictWriter(f_out, fieldnames=out_fields)
        w.writeheader()

        for row in r:
            posted = (row.get("posted_date") or row.get("date") or "").strip()
            if not posted:
                # skip malformed rows
                continue

            cat = (row.get("category_final") or row.get("category") or "").strip()
            signed = d(row.get("signed_amount") or row.get("amount"))
            external = signed if cat != "transfer_internal" else Decimal("0")

            acct = (row.get("account") or "").strip()
            desc = (row.get("description") or "").strip()

            merch = (row.get("merchant_norm") or "").strip()
            if not merch:
                merch = norm_merchant(desc)

            dt = parse_ymd(posted)
            cents = abs_cents(external)

            mk_loose = f"{posted}|{cents}"
            mk_m1 = f"{ymd(dt - timedelta(days=1))}|{cents}"
            mk_p1 = f"{ymd(dt + timedelta(days=1))}|{cents}"
            mk_strict = f"{posted}|{cents}|{merch[:12]}"

            out = {
                "source": row.get("source") or "NFCU",
                "year": (row.get("year") or year),
                "month": row.get("month") or posted[:7],
                "posted_date": posted,
                "account": acct,
                "description": desc,
                "merchant_norm": merch,
                "category_final": cat,
                "signed_amount": q2(signed),
                "external_amount": q2(external),
                "abs_amount_cents": str(cents),
                "match_key_loose": mk_loose,
                "match_key_loose_m1": mk_m1,
                "match_key_loose_p1": mk_p1,
                "match_key_strict": mk_strict,
                "txn_key": row.get("txn_key") or row.get("transaction_id") or "",
            }
            w.writerow(out)
            n += 1

    print("---- NFCU recon export ----")
    print("IN :", in_path)
    print("OUT:", out_path)
    print("rows:", n)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
