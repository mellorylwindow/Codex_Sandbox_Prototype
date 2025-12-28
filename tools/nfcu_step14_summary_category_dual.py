#!/usr/bin/env python
"""
Step 14: Month+Category summary with BOTH nets:
- ledger_net: includes transfer_internal
- external_net: zeros out transfer_internal (what you actually spent/earned externally)

We keep the first 4 columns compatible with the existing file:
month, category_final, count, external_net
...then add:
ledger_net, transfer_internal_net
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

Q = Decimal("0.01")


def q2(x: Decimal) -> str:
    return str(x.quantize(Q, rounding=ROUND_HALF_UP))


def d(x) -> Decimal:
    try:
        return Decimal((x or "").strip() or "0")
    except Exception:
        return Decimal("0")


@dataclass
class Agg:
    count: int = 0
    ledger_net: Decimal = Decimal("0")
    external_net: Decimal = Decimal("0")
    transfer_internal_net: Decimal = Decimal("0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2025")
    ap.add_argument(
        "--in",
        dest="inp",
        default=None,
        help="Optional explicit input categorized.csv path",
    )
    ap.add_argument(
        "--out",
        dest="outp",
        default=None,
        help="Optional explicit output summary path",
    )
    args = ap.parse_args()

    year = str(args.year)
    out_dir = Path(f"notes/tax/work/parsed/{year}")

    categorized = Path(args.inp) if args.inp else (out_dir / "nfcu_transactions.categorized.csv")
    summary_out = Path(args.outp) if args.outp else (out_dir / "nfcu_summary_by_month_category.csv")

    if not categorized.exists():
        raise SystemExit(f"missing input: {categorized}")

    agg: dict[tuple[str, str], Agg] = defaultdict(Agg)

    with categorized.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            month = (row.get("month") or "").strip()
            cat = (row.get("category_final") or "").strip()
            if not month or not cat:
                continue

            amt = d(row.get("signed_amount"))

            a = agg[(month, cat)]
            a.count += 1
            a.ledger_net += amt

            if cat == "transfer_internal":
                a.transfer_internal_net += amt
                # external_net contributes 0 for internal transfers
            else:
                a.external_net += amt

    keys = sorted(agg.keys())

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        # Keep old shape FIRST for compatibility:
        w.writerow(["month", "category_final", "count", "external_net", "ledger_net", "transfer_internal_net"])
        for (m, c) in keys:
            a = agg[(m, c)]
            w.writerow([m, c, a.count, q2(a.external_net), q2(a.ledger_net), q2(a.transfer_internal_net)])

    ledger_total = sum(agg[k].ledger_net for k in keys)
    external_total = sum(agg[k].external_net for k in keys)
    transfers_total = sum(agg[k].transfer_internal_net for k in keys)

    print("---- Step 14: Month+Category (dual-net) ----")
    print("IN :", categorized)
    print("OUT:", summary_out)
    print("ledger_total  =", q2(ledger_total))
    print("external_total=", q2(external_total))
    print("transfer_internal_total=", q2(transfers_total))
    print("delta (ledger - external)=", q2(ledger_total - external_total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
