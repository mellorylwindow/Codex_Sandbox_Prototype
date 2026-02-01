from __future__ import annotations

import argparse
import csv
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser(description="Make pnc.recon.csv using Rocket recon header as canonical schema.")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--rocket", default=None, help="Rocket recon CSV (used for header/schema)")
    ap.add_argument("--pnc-in", default=None, help="PNC input CSV (enriched/categorized)")
    ap.add_argument("--out", default=None, help="Output recon CSV")
    ap.add_argument("--account-name", default="Spend")
    ap.add_argument("--institution-name", default="PNC")
    ap.add_argument("--account-type", default="Cash")
    args = ap.parse_args()

    year = args.year
    rocket = Path(args.rocket or f"notes/tax/work/parsed/{year}/rocket_money.recon.csv")
    pnc_in = Path(args.pnc_in or f"notes/tax/work/parsed/{year}/pnc_spend_transactions.categorized.csv")
    out = Path(args.out or f"notes/tax/work/parsed/{year}/pnc.recon.csv")

    if not rocket.exists():
        raise SystemExit(f"Rocket recon not found: {rocket}")
    if not pnc_in.exists():
        raise SystemExit(f"PNC input not found: {pnc_in}")

    # Canonical header
    with rocket.open(newline="", encoding="utf-8") as f:
        rocket_header = next(csv.reader(f))

    def set_if(d: dict, key: str, val: str) -> None:
        if key in d:
            d[key] = val

    # Read PNC input
    rows_out = []
    with pnc_in.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            dt = (row.get("date") or "").strip()
            if not dt or len(dt) < 10:
                continue
            if int(dt[:4]) != year:
                continue

            out_row = {h: "" for h in rocket_header}

            # Required-ish fields you showed
            set_if(out_row, "source", "PNC")
            set_if(out_row, "year", str(year))
            set_if(out_row, "month", dt[:7])
            set_if(out_row, "date", dt[:10])
            set_if(out_row, "original_date", dt[:10])
            set_if(out_row, "account_type", args.account_type)
            set_if(out_row, "account_name", args.account_name)
            set_if(out_row, "institution_name", args.institution_name)

            # Common money fields (only fill if those cols exist in rocket header)
            amt = (row.get("amount") or "").strip()
            if amt:
                for k in ("amount", "signed_amount", "transaction_amount"):
                    set_if(out_row, k, amt)

            desc = (row.get("merchant") or row.get("description") or "").strip()
            if desc:
                for k in ("description", "merchant", "payee", "name"):
                    set_if(out_row, k, desc)

            # Category if available
            cat = (row.get("category") or "").strip()
            if cat:
                for k in ("category", "rocket_category"):
                    set_if(out_row, k, cat)

            rows_out.append(out_row)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rocket_header)
        w.writeheader()
        w.writerows(rows_out)

    print("wrote:", out)
    print("rows:", len(rows_out))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
