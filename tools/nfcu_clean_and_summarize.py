#!/usr/bin/env python3
"""
Step 10: Clean + Summarize (NFCU)

Inputs:
  notes/tax/work/parsed/<year>/nfcu_transactions.normalized.csv

Outputs:
  notes/tax/work/parsed/<year>/nfcu_transactions.cleaned.csv
  notes/tax/work/parsed/<year>/nfcu_transactions.summary_by_month.csv

Key fixes in this rewrite:
  1) date repair:
     - ensure `date` is populated
     - repair missing posted_date/transaction_date when possible
     - derive `month` (YYYY-MM)
  2) glued-header trimming:
     - remove statement header junk glued into `description`
       (Average Daily Balance / Items Paid / Statement Period / Access No. etc)
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple


CUT_MARKERS = [
    " Average Daily Balance",
    " Items Paid",
    " Statement Period",
    " Access No.",
    " Statement of Account",
    " (Continued from previous page)",
    " Joint Owner(s):",
]

# Useful for parsing dates inside descriptions (e.g., "01-03-25", "01/03/25", sometimes "01-03-2025")
RE_MMDDYY = re.compile(r"\b(\d{2})[/-](\d{2})[/-](\d{2,4})\b")

# File name pattern like: 2025-01-21_STMSSCM.pdf
RE_FILE_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

Q2 = Decimal("0.01")


def d2(x: Decimal) -> str:
    return str(x.quantize(Q2, rounding=ROUND_HALF_UP))


def to_dec(s: str | None) -> Decimal:
    if s is None:
        return Decimal("0")
    s = s.strip()
    if not s:
        return Decimal("0")
    # strip commas / $ just in case
    s = s.replace(",", "").replace("$", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return Decimal("0")


def parse_yyyy_mm_dd(s: str | None) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    # accept YYYY-MM-DD only
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    return None


def parse_desc_date(desc: str) -> Optional[str]:
    m = RE_MMDDYY.search(desc or "")
    if not m:
        return None
    mm, dd, yy = m.group(1), m.group(2), m.group(3)
    if len(yy) == 2:
        yyyy = f"20{yy}"
    else:
        yyyy = yy
    # basic sanity
    try:
        Date(int(yyyy), int(mm), int(dd))
    except Exception:
        return None
    return f"{yyyy}-{mm}-{dd}"


def parse_source_file_date(source_file: str | None) -> Optional[str]:
    if not source_file:
        return None
    m = RE_FILE_DATE.search(source_file)
    if not m:
        return None
    yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
    try:
        Date(int(yyyy), int(mm), int(dd))
    except Exception:
        return None
    return f"{yyyy}-{mm}-{dd}"


def clean_desc(s: str) -> str:
    s = (s or "").strip()
    for marker in CUT_MARKERS:
        i = s.find(marker)
        if i != -1:
            s = s[:i].rstrip()
    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_direction_from_desc(desc: str) -> Optional[str]:
    d = (desc or "").lower()

    # strongest signals first
    if "adjustment - dr" in d:
        return "debit"
    if "adjustment - cr" in d:
        return "credit"

    if "pos debit" in d:
        return "debit"
    if "pos credit" in d:
        return "credit"

    if "transfer to" in d:
        return "debit"
    if "transfer from" in d:
        return "credit"

    if "withdrawal" in d:
        return "debit"
    if "deposit" in d:
        return "credit"

    # zelle can be either; leave to amounts unless we have clear text
    return None


def compute_month(yyyy_mm_dd: str) -> str:
    return yyyy_mm_dd[:7]


@dataclass
class CleanStats:
    input_rows: int = 0
    clean_rows: int = 0
    dropped_rows: int = 0
    recovered_amounts: int = 0
    inferred_direction_desc: int = 0
    unknown_direction_kept: int = 0


def pick_year_dir(base: Path, year: Optional[int]) -> Path:
    if year is not None:
        return base / str(year)

    # auto-pick most recent year folder that contains nfcu_transactions.normalized.csv
    candidates = []
    if base.exists():
        for p in base.iterdir():
            if p.is_dir() and re.fullmatch(r"\d{4}", p.name):
                if (p / "nfcu_transactions.normalized.csv").exists():
                    candidates.append(p)
    if not candidates:
        # fall back to base/2025 if present, else base
        if (base / "2025" / "nfcu_transactions.normalized.csv").exists():
            return base / "2025"
        return base
    return sorted(candidates, key=lambda p: p.name)[-1]


def read_normalized_rows(inp: Path) -> Iterable[Dict[str, str]]:
    with inp.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def write_cleaned(out: Path, rows: Iterable[Dict[str, str]]) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "account",
        "description",
        "amount",
        "direction",
        "credit",
        "currency",
        "debit",
        "fit_id",
        "memo",
        "month",
        "posted_date",
        "raw_category",
        "raw_type",
        "source_file",
        "source_row",
        "transaction_date",
    ]
    n = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
            n += 1
    return n


def write_summary(summary: Path, cleaned_rows: Iterable[Dict[str, str]]) -> None:
    agg = defaultdict(lambda: {"count": 0, "debit": Decimal("0"), "credit": Decimal("0")})

    for r in cleaned_rows:
        m = (r.get("month") or "").strip()
        if not m:
            continue
        agg[m]["count"] += 1
        agg[m]["debit"] += to_dec(r.get("debit"))
        agg[m]["credit"] += to_dec(r.get("credit"))

    months = sorted(agg.keys())
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "count", "net", "debit", "credit"])
        for m in months:
            d = agg[m]["debit"]
            c = agg[m]["credit"]
            w.writerow([m, agg[m]["count"], d2(c - d), d2(d), d2(c)])


def clean_pipeline(inp: Path, out_clean: Path, out_summary: Path) -> CleanStats:
    stats = CleanStats()
    cleaned_rows = []

    for row in read_normalized_rows(inp):
        stats.input_rows += 1

        account = (row.get("account") or "").strip()

        # Some normalized files use different keys; be tolerant
        desc_raw = (row.get("description") or row.get("memo") or "").strip()
        desc = clean_desc(desc_raw)

        memo = (row.get("memo") or "").strip()
        currency = (row.get("currency") or "USD").strip() or "USD"

        fit_id = (row.get("fit_id") or "").strip()
        raw_category = (row.get("raw_category") or "").strip()
        raw_type = (row.get("raw_type") or "").strip()

        source_file = (row.get("source_file") or "").strip()
        source_row = (row.get("source_row") or "").strip()

        posted_date = parse_yyyy_mm_dd(row.get("posted_date"))
        transaction_date = parse_yyyy_mm_dd(row.get("transaction_date"))

        # --- date repair ---
        # Prefer transaction_date, then posted_date, then parse from description, then from source filename.
        best_date = transaction_date or posted_date
        if not best_date:
            best_date = parse_desc_date(desc) or parse_desc_date(desc_raw)
        if not best_date:
            best_date = parse_source_file_date(source_file)

        if not posted_date and best_date:
            posted_date = best_date
        if not transaction_date and best_date:
            transaction_date = best_date

        month = (row.get("month") or "").strip()
        if not month and best_date:
            month = compute_month(best_date)

        # amounts
        debit_raw = to_dec(row.get("debit"))
        credit_raw = to_dec(row.get("credit"))

        # normalize negatives (parser noise): treat magnitude as value; sign handled by direction
        debit_mag = abs(debit_raw)
        credit_mag = abs(credit_raw)

        # infer direction primarily from numeric columns
        direction = (row.get("direction") or "").strip().lower()
        direction = direction if direction in ("debit", "credit") else ""

        inferred_by_desc = False

        if debit_mag == Decimal("0") and credit_mag == Decimal("0"):
            # attempt amount recovery (rare): allow "$12.34" in description
            m_amt = re.search(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)\b", desc)
            if m_amt:
                amt = to_dec(m_amt.group(1))
                if amt != Decimal("0"):
                    stats.recovered_amounts += 1
                    # direction guess from desc if possible
                    gd = infer_direction_from_desc(desc)
                    if gd:
                        direction = gd
                        inferred_by_desc = True
                    # otherwise leave unknown
                    if direction == "debit":
                        debit_mag, credit_mag = amt, Decimal("0")
                    elif direction == "credit":
                        credit_mag, debit_mag = amt, Decimal("0")
                    else:
                        # unknown; keep amount but no side assignment
                        pass

        # If both sides present, resolve with description hint; else fall back to net sign.
        if debit_mag != Decimal("0") and credit_mag != Decimal("0"):
            gd = infer_direction_from_desc(desc)
            if gd:
                direction = gd
                inferred_by_desc = True
            else:
                # choose by larger side, else by (credit - debit)
                if credit_mag > debit_mag:
                    direction = "credit"
                elif debit_mag > credit_mag:
                    direction = "debit"
                else:
                    # equal magnitude; ambiguous (likely internal transfer noise)
                    # keep as debit by default; downstream transfer cancel handles these anyway
                    direction = direction or "debit"

        # If direction still empty but one side present, set it.
        if not direction:
            if debit_mag != Decimal("0") and credit_mag == Decimal("0"):
                direction = "debit"
            elif credit_mag != Decimal("0") and debit_mag == Decimal("0"):
                direction = "credit"

        # If still empty, try desc inference.
        if not direction:
            gd = infer_direction_from_desc(desc)
            if gd:
                direction = gd
                inferred_by_desc = True

        if inferred_by_desc:
            stats.inferred_direction_desc += 1

        # Construct amount and enforce one-sided credit/debit for downstream sanity.
        if direction == "debit":
            amount = debit_mag if debit_mag != Decimal("0") else credit_mag
            debit = amount
            credit = Decimal("0")
        elif direction == "credit":
            amount = credit_mag if credit_mag != Decimal("0") else debit_mag
            credit = amount
            debit = Decimal("0")
        else:
            # unknown direction: keep magnitudes as-is; amount = max
            amount = max(debit_mag, credit_mag)
            debit = debit_mag
            credit = credit_mag
            stats.unknown_direction_kept += 1

        # Drop true zero rows
        if amount == Decimal("0"):
            stats.dropped_rows += 1
            continue

        # Final description must not be empty; if it is, drop
        if not desc:
            stats.dropped_rows += 1
            continue

        cleaned_rows.append(
            {
                "date": best_date or "",
                "account": account,
                "description": desc,
                "amount": d2(amount),
                "direction": direction or "",
                "credit": d2(credit),
                "currency": currency,
                "debit": d2(debit),
                "fit_id": fit_id,
                "memo": memo,
                "month": month,
                "posted_date": posted_date or "",
                "raw_category": raw_category,
                "raw_type": raw_type,
                "source_file": source_file,
                "source_row": source_row,
                "transaction_date": transaction_date or "",
            }
        )

    stats.clean_rows = len(cleaned_rows)

    # write cleaned
    write_cleaned(out_clean, cleaned_rows)

    # write summary (read back cleaned to ensure we summarize exactly what we wrote)
    with out_clean.open("r", encoding="utf-8", newline="") as f:
        cleaned_iter = csv.DictReader(f)
        write_summary(out_summary, cleaned_iter)

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="NFCU Step 10: clean + summarize (rewrite)")
    ap.add_argument("--base", default="notes/tax/work/parsed", help="Base parsed directory (default: notes/tax/work/parsed)")
    ap.add_argument("--year", type=int, default=None, help="Year folder to use (e.g., 2025). Default: auto-detect latest.")
    args = ap.parse_args()

    base = Path(args.base)
    year_dir = pick_year_dir(base, args.year)

    inp = year_dir / "nfcu_transactions.normalized.csv"
    out_clean = year_dir / "nfcu_transactions.cleaned.csv"
    out_summary = year_dir / "nfcu_transactions.summary_by_month.csv"

    if not inp.exists():
        print("ERROR: missing input:", inp)
        return 2

    print("---- Step 10: Clean + Summarize ----")
    print(f"IN : {inp}")
    print(f"OUT: {out_clean}")
    print(f"OUT: {out_summary}")

    stats = clean_pipeline(inp, out_clean, out_summary)

    print(f"Input rows: {stats.input_rows}")
    print(f"Clean rows: {stats.clean_rows}")
    print(f"Dropped   : {stats.dropped_rows}")
    print(f"Recovered amounts from description: {stats.recovered_amounts}")
    print(f"Inferred direction (desc rules): {stats.inferred_direction_desc}")
    print(f"Kept with unknown direction: {stats.unknown_direction_kept}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
