#!/usr/bin/env python3
"""
Step 10: NFCU — Clean + Summarize (v2)

Goals:
- Keep real transactions; drop statement noise (beginning/ending balance blocks, avg daily balance chatter, etc.)
- Produce:
    notes/tax/work/parsed/2025/nfcu_transactions.cleaned.csv
    notes/tax/work/parsed/2025/nfcu_transactions.summary_by_month.csv

Key behaviors:
- Amount selection precedence: amount_raw -> amount -> recover from description (last resort)
- Direction inference precedence:
    direction_guess -> description rules (tight patterns, incl. Zelle GR) -> sign from amount token (if any)
- Do NOT drop rows merely because direction is unknown.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_WS_RE = re.compile(r"\s+")
_MONEY_RE = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})*|\d+)\.\d{2}(?!\d)")

_DIRECTION_VALUES = {"debit", "credit"}


# ----------------------------
# Normalizers
# ----------------------------

def _clean_desc(desc: str) -> str:
    return _WS_RE.sub(" ", (desc or "").replace("\u00a0", " ")).strip()


def _lower(s: str) -> str:
    return (s or "").strip().lower()


def _safe_get(row: Dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def _normalize_direction(v: str) -> str:
    s = _lower(v)
    if s in _DIRECTION_VALUES:
        return s
    if s == "dr":
        return "debit"
    if s == "cr":
        return "credit"
    return ""


def _choose_date(row: Dict[str, str]) -> str:
    # normalized may use "date" or "statement_date"
    d = _safe_get(row, "date")
    if d:
        return d
    return _safe_get(row, "statement_date")


def _choose_account(row: Dict[str, str]) -> str:
    a = _safe_get(row, "account")
    if a:
        return a
    return _safe_get(row, "acct")


def _month_key(date_str: str) -> str:
    return date_str[:7] if date_str and len(date_str) >= 7 else ""


# ----------------------------
# Noise filters (keep tight)
# ----------------------------

def should_drop_row(desc_clean: str) -> bool:
    """
    Only drop things we are confident are statement noise.
    """
    s = _lower(desc_clean)
    if not s:
        return True

    # Obvious noise headers
    if s.startswith("beginning balance"):
        return True
    if s.startswith("ending balance"):
        return True

    # Common statement chatter (not transactions)
    if "average daily balance" in s:
        return True
    if "joint owner" in s:
        return True
    if "everyday checking" in s:
        return True
    if "items paid" in s and "ach paid to" not in s:
        # prevent false positives on ACH Paid To
        return True

    return False


# ----------------------------
# Amount parsing
# ----------------------------

def _parse_amount_token(token: str) -> Tuple[Optional[Decimal], Optional[str]]:
    """
    Parse token like:
      125.00-
      -125.00
      (125.00)
      1,234.56
    Returns (abs_amount, sign_hint) where sign_hint is debit/credit if detectable.
    """
    s = (token or "").strip()
    if not s:
        return None, None

    neg = False
    if s.endswith("-"):
        neg = True
        s = s[:-1].strip()
    if s.startswith("-"):
        neg = True
        s = s[1:].strip()
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    s = s.replace(",", "")

    try:
        amt = Decimal(s)
    except InvalidOperation:
        return None, None

    if amt == 0:
        return Decimal("0.00"), None

    return abs(amt), ("debit" if neg else "credit")


def parse_amount_from_row(row: Dict[str, str], desc_clean: str) -> Tuple[Optional[Decimal], Optional[str], bool]:
    """
    Returns (amount, sign_hint, recovered_from_desc)
    """
    # 1) amount_raw
    amount_raw = _safe_get(row, "amount_raw")
    if amount_raw:
        amt, sign_hint = _parse_amount_token(amount_raw)
        if amt is not None:
            return amt, sign_hint, False

    # 2) amount (some pipelines may already compute this)
    amount = _safe_get(row, "amount")
    if amount:
        amt, sign_hint = _parse_amount_token(amount)
        if amt is not None:
            return amt, sign_hint, False

    # 3) recover from description (last resort)
    tokens = desc_clean.split()
    last_money = None
    for t in tokens:
        tt = t.strip()
        if tt.endswith("-"):
            core = tt[:-1]
            if _MONEY_RE.fullmatch(core):
                last_money = tt
        else:
            if _MONEY_RE.fullmatch(tt):
                last_money = tt

    if last_money:
        amt, sign_hint = _parse_amount_token(last_money)
        if amt is not None:
            return amt, sign_hint, True

    return None, None, False


# ----------------------------
# Direction inference (tight rules)
# ----------------------------

def infer_direction_from_desc(desc_clean: str) -> Optional[str]:
    s = _lower(desc_clean)

    # Zelle (NFCU shorthand we observed)
    if s.startswith("zelle gr "):
        return "credit"
    if s.startswith("zelle to "):
        return "debit"

    # Card / POS
    if s.startswith("pos "):
        return "debit"
    if s.startswith("debit card "):
        return "debit"
    if s.startswith("visa "):
        return "debit"

    # Transfers
    if s.startswith("transfer to "):
        return "debit"
    if s.startswith("transfer from "):
        return "credit"

    # ACH
    if s.startswith("ach paid to "):
        return "debit"
    if s.startswith("ach deposit"):
        return "credit"
    if s.startswith("ach credit"):
        return "credit"
    if s.startswith("ach debit"):
        return "debit"

    # Adjustments with DR/CR markers
    if s.startswith("adjustment"):
        if re.search(r"\bdr\b", s):
            return "debit"
        if re.search(r"\bcr\b", s):
            return "credit"

    # Fees
    if re.search(r"\bfee\b", s):
        return "debit"

    # Dividends / interest
    if "dividend" in s or "interest" in s:
        return "credit"

    return None


# ----------------------------
# Core
# ----------------------------

@dataclass
class Stats:
    input_rows: int = 0
    clean_rows: int = 0
    dropped_rows: int = 0
    recovered_amounts: int = 0
    inferred_direction: int = 0
    unknown_direction_kept: int = 0


def process_rows(rows: Iterable[Dict[str, str]], stats: Stats) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    for r in rows:
        stats.input_rows += 1

        desc = _clean_desc(_safe_get(r, "description"))

        if should_drop_row(desc):
            stats.dropped_rows += 1
            continue

        amt, sign_hint, recovered = parse_amount_from_row(r, desc)
        if amt is None:
            # No usable amount = not usable transaction row
            stats.dropped_rows += 1
            continue

        if recovered:
            stats.recovered_amounts += 1

        # Direction precedence:
        direction = _normalize_direction(_safe_get(r, "direction_guess"))

        if not direction:
            d2 = infer_direction_from_desc(desc)
            if d2:
                direction = d2
                stats.inferred_direction += 1

        if not direction and sign_hint in _DIRECTION_VALUES:
            # Only use sign if it truly exists (trailing '-' etc.)
            direction = sign_hint

        if not direction:
            # Keep it; downstream can flag it.
            stats.unknown_direction_kept += 1

        r2 = dict(r)
        r2["date"] = _choose_date(r)
        r2["account"] = _choose_account(r)
        r2["description"] = desc
        r2["amount"] = f"{amt:.2f}"
        r2["direction"] = direction

        out.append(r2)
        stats.clean_rows += 1

    return out


def write_clean_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "account", "description", "amount", "direction"])
        return

    preferred = ["date", "account", "description", "amount", "direction"]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    tail = [k for k in sorted(all_keys) if k not in preferred]
    fieldnames = preferred + tail

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_by_month(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    agg: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: {
        "count": Decimal(0),
        "debit_count": Decimal(0),
        "credit_count": Decimal(0),
        "debit_total": Decimal("0.00"),
        "credit_total": Decimal("0.00"),
        "net": Decimal("0.00"),
        "unknown_dir_count": Decimal(0),
    })

    for r in rows:
        m = _month_key(_safe_get(r, "date"))
        if not m:
            continue

        direction = _normalize_direction(_safe_get(r, "direction"))
        amt_s = _safe_get(r, "amount")
        try:
            amt = Decimal(amt_s) if amt_s else Decimal("0.00")
        except InvalidOperation:
            amt = Decimal("0.00")

        a = agg[m]
        a["count"] += 1

        if direction == "debit":
            a["debit_count"] += 1
            a["debit_total"] += amt
            a["net"] -= amt
        elif direction == "credit":
            a["credit_count"] += 1
            a["credit_total"] += amt
            a["net"] += amt
        else:
            a["unknown_dir_count"] += 1

    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "month",
            "count",
            "debit_count",
            "credit_count",
            "unknown_dir_count",
            "debit_total",
            "credit_total",
            "net",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for month in sorted(agg.keys()):
            a = agg[month]
            w.writerow({
                "month": month,
                "count": int(a["count"]),
                "debit_count": int(a["debit_count"]),
                "credit_count": int(a["credit_count"]),
                "unknown_dir_count": int(a["unknown_dir_count"]),
                "debit_total": f"{a['debit_total']:.2f}",
                "credit_total": f"{a['credit_total']:.2f}",
                "net": f"{a['net']:.2f}",
            })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_path",
        default="notes/tax/work/parsed/2025/nfcu_transactions.normalized.csv",
        help="Input normalized CSV",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        default="notes/tax/work/parsed/2025/nfcu_transactions.cleaned.csv",
        help="Output cleaned CSV",
    )
    ap.add_argument(
        "--summary",
        dest="summary_path",
        default="notes/tax/work/parsed/2025/nfcu_transactions.summary_by_month.csv",
        help="Output summary-by-month CSV",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path)

    print("---- Step 10: Clean + Summarize ----")
    print(f"IN : {in_path}")
    print(f"OUT: {out_path}")
    print(f"OUT: {summary_path}")

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows_in = list(reader)

    stats = Stats()
    rows_out = process_rows(rows_in, stats)

    write_clean_csv(out_path, rows_out)
    write_summary_by_month(summary_path, rows_out)

    print(f"Input rows: {stats.input_rows}")
    print(f"Clean rows: {stats.clean_rows}")
    print(f"Dropped   : {stats.dropped_rows}")
    print(f"Recovered amounts from description: {stats.recovered_amounts}")
    print(f"Inferred direction (desc rules): {stats.inferred_direction}")
    print(f"Kept with unknown direction: {stats.unknown_direction_kept}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
