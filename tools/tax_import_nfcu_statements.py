#!/usr/bin/env python3
"""
NFCU statement importer (CSV + PDF).

PDF format observed (important):
- Table header: "Date Transaction Detail Amount($) Balance($)"
- Transaction date format: "MM-DD" (no year)
- Debits show trailing minus: "85.00-"
- Extra payee lines may appear on following line (no date)

Outputs (by year):
- nfcu_transactions.normalized.csv
- nfcu_transactions.summary_by_month.csv
- nfcu_transactions.import_log.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass, asdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional PDF dependency
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None


# ----------------------------
# Normalized output schema
# ----------------------------

@dataclass
class Txn:
    account: str
    source_file: str
    source_row: str
    transaction_date: str  # YYYY-MM-DD
    posted_date: str       # YYYY-MM-DD (same for statements)
    description: str
    memo: str
    amount: str            # signed string (e.g., "-12.34")
    debit: str
    credit: str
    currency: str
    month: str             # YYYY-MM
    raw_type: str
    raw_category: str
    fit_id: str


# ----------------------------
# Regex / parsing helpers
# ----------------------------

_DATE_RANGE_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{2})\s*-\s*(\d{1,2}/\d{1,2}/\d{2})")
_MMDDYY_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2,4})\s*$")
_MM_DD_RE = re.compile(r"^\s*(\d{2})-(\d{2})\b")  # MM-DD at line start

# A money token like:
#  140.00
#  2,510.00
#  85.00-
#  (85.00)
_MONEY_TOKEN_RE = re.compile(r"\(?-?\$?\s*[\d,]+\.\d{2}\)?-?")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_in_dir(repo_root: Path) -> Path:
    # Your real “inbox”
    return repo_root / "notes" / "tax" / "inbox" / "nfcu"


def _default_out_dir(repo_root: Path, year: int) -> Path:
    return repo_root / "notes" / "tax" / "work" / "parsed" / str(year)


def _iter_csv_files(in_dir: Path) -> List[Path]:
    if not in_dir.exists():
        return []
    return sorted([p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"])


def _iter_pdf_files(in_dir: Path) -> List[Path]:
    if not in_dir.exists():
        return []
    return sorted([p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"])


def _parse_mmddyy(s: str) -> Optional[date]:
    s = s.strip()
    m = _MMDDYY_RE.match(s)
    if not m:
        return None
    mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if yy < 100:
        yy += 2000 if yy < 80 else 1900
    try:
        return date(yy, mm, dd)
    except Exception:
        return None


def _to_month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _money_to_decimal(s: str) -> Optional[Decimal]:
    """
    Handles:
      "85.00-"  (trailing minus)
      "-85.00"
      "(85.00)"
      "$1,234.56"
    """
    if not s:
        return None
    s = s.strip()

    neg = False
    if s.endswith("-"):
        neg = True
        s = s[:-1].strip()

    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    s = s.replace("$", "").replace(",", "").strip()
    if not s:
        return None

    try:
        v = Decimal(s)
        return -v if neg else v
    except InvalidOperation:
        return None


def _infer_year_for_mmdd(mm: int, dd: int, stmt_start: date, stmt_end: date) -> int:
    """
    Infer year for MM-DD given statement period.
    Handles typical cross-year Dec->Jan statements.
    """
    if stmt_start.year == stmt_end.year:
        return stmt_end.year

    # Common NFCU cross-year case: 12/22/24 - 01/21/25
    if stmt_start.month == 12 and stmt_end.month == 1:
        return stmt_start.year if mm == 12 else stmt_end.year

    # Generic fallback: choose year that places date inside the range, else closest.
    def safe_date(y: int) -> Optional[date]:
        try:
            return date(y, mm, dd)
        except Exception:
            return None

    d1 = safe_date(stmt_start.year)
    d2 = safe_date(stmt_end.year)
    if d1 and stmt_start <= d1 <= stmt_end:
        return stmt_start.year
    if d2 and stmt_start <= d2 <= stmt_end:
        return stmt_end.year
    # closest endpoint heuristic
    if d1 and d2:
        score1 = abs((d1 - stmt_start).days) + abs((stmt_end - d1).days)
        score2 = abs((d2 - stmt_start).days) + abs((stmt_end - d2).days)
        return stmt_start.year if score1 <= score2 else stmt_end.year
    return stmt_end.year


def _stable_fit_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


# ----------------------------
# PDF parsing (line-based for NFCU layout)
# ----------------------------

def _extract_statement_period(pdf) -> Tuple[Optional[date], Optional[date]]:
    """
    Looks for 'MM/DD/YY - MM/DD/YY' anywhere on page 1.
    """
    try:
        txt = pdf.pages[0].extract_text() or ""
    except Exception:
        return (None, None)

    m = _DATE_RANGE_RE.search(txt.replace(" ", ""))
    if not m:
        m = _DATE_RANGE_RE.search(txt)
    if not m:
        return (None, None)

    d1 = _parse_mmddyy(m.group(1))
    d2 = _parse_mmddyy(m.group(2))
    return (d1, d2)


def _looks_like_table_header(line: str) -> bool:
    s = re.sub(r"\s+", " ", line).strip().lower()
    # Exact-ish header you pasted:
    # "Date Transaction Detail Amount($) Balance($)"
    return (
        "date" in s
        and "transaction" in s
        and "detail" in s
        and "amount" in s
        and "balance" in s
    )


def _parse_txn_line(line: str) -> Optional[Tuple[int, int, str, Optional[Decimal], Optional[Decimal]]]:
    """
    Parse NFCU transaction line format like:
      "12-23 Transfer To Mortgage 84.77- 100.46"
      "12-22 Beginning Balance 400.23"
    Returns (mm, dd, detail, amount, balance)
      - amount may be None when only a balance is present (Beginning/Ending Balance)
    """
    line = re.sub(r"\s+", " ", line).strip()
    m = _MM_DD_RE.match(line)
    if not m:
        return None
    mm, dd = int(m.group(1)), int(m.group(2))
    rest = line[m.end():].strip()

    # Grab all money-ish tokens, from right
    money = _MONEY_TOKEN_RE.findall(rest)
    money = [re.sub(r"\s+", "", x) for x in money]  # normalize whitespace

    if not money:
        return None

    # Balance is almost always the last money token
    bal = _money_to_decimal(money[-1])

    # If we have 2+ money tokens, second-to-last is amount
    amt = _money_to_decimal(money[-2]) if len(money) >= 2 else None

    # Detail is rest with the last 1 or 2 money tokens removed
    # Do a conservative right-strip by removing the last tokens in order
    detail = rest
    # remove last balance token
    detail = re.sub(re.escape(money[-1]) + r"\s*$", "", detail).strip()
    if amt is not None and len(money) >= 2:
        detail = re.sub(re.escape(money[-2]) + r"\s*$", "", detail).strip()

    return (mm, dd, detail.strip(), amt, bal)


def _parse_pdf_transactions(pdf_path: Path, year_filter: Optional[int], verbose: bool) -> Tuple[List[Txn], List[str]]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed (pip install pdfplumber).")

    rows: List[Txn] = []
    log: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        stmt_start, stmt_end = _extract_statement_period(pdf)
        if stmt_start and stmt_end and verbose:
            log.append(f"    [meta] statement_period: {stmt_start.isoformat()} -> {stmt_end.isoformat()}")

        current_account = "NFCU"
        in_table = False
        last_txn: Optional[Txn] = None

        for page_idx, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]

            for li, ln in enumerate(lines, start=1):
                s = re.sub(r"\s+", " ", ln).strip()

                # Detect account section lines like: "EveryDay Checking - 7103999459"
                m_acct = re.match(r"^(.*?)-\s*(\d{10})\s*$", s)
                if m_acct:
                    acct_name = m_acct.group(1).strip()
                    acct_num = m_acct.group(2).strip()
                    current_account = f"NFCU:{acct_name}:{acct_num}"
                    in_table = False
                    last_txn = None
                    continue

                # Detect header line
                if _looks_like_table_header(s):
                    in_table = True
                    last_txn = None
                    continue

                if not in_table:
                    continue

                # Parse a transaction row
                parsed = _parse_txn_line(s)
                if parsed:
                    mm, dd, detail, amt, bal = parsed

                    # Skip non-transaction balance markers if you don’t want them:
                    # (They never matter for tax categories)
                    if re.search(r"\b(beginning balance|ending balance|average daily balance)\b", detail, flags=re.I):
                        continue

                    # Determine year
                    if stmt_start and stmt_end:
                        y = _infer_year_for_mmdd(mm, dd, stmt_start, stmt_end)
                    else:
                        y = date.today().year

                    try:
                        txn_dt = date(y, mm, dd)
                    except Exception:
                        continue

                    if year_filter is not None and txn_dt.year != year_filter:
                        continue

                    # Amount sign rules:
                    # - If amt is None: treat as 0 (shouldn’t happen after skipping balances)
                    # - If amt is negative: debit
                    # - If amt is positive: credit
                    if amt is None:
                        amt = Decimal("0.00")

                    debit = abs(amt) if amt < 0 else Decimal("0.00")
                    credit = amt if amt > 0 else Decimal("0.00")

                    txn_date_s = txn_dt.isoformat()
                    month_s = _to_month_key(txn_dt)

                    source_row = f"p{page_idx}:l{li}"
                    fit_id = _stable_fit_id("NFCU", pdf_path.name, source_row, txn_date_s, detail, f"{amt:.2f}")

                    t = Txn(
                        account=current_account,
                        source_file=str(pdf_path.as_posix()),
                        source_row=source_row,
                        transaction_date=txn_date_s,
                        posted_date=txn_date_s,
                        description=detail,
                        memo="",
                        amount=f"{amt:.2f}",
                        debit=f"{debit:.2f}",
                        credit=f"{credit:.2f}",
                        currency="USD",
                        month=month_s,
                        raw_type="",
                        raw_category="",
                        fit_id=fit_id,
                    )
                    rows.append(t)
                    last_txn = t
                    continue

                # Continuation line (e.g. "Mallow B Stone") after a txn
                if last_txn:
                    # Avoid obvious footer junk
                    if not re.search(r"\b(page \d+ of \d+|statement period|access no\.)\b", s, flags=re.I):
                        last_txn.description = (last_txn.description + " " + s).strip()

    return rows, log


# ----------------------------
# CSV parsing (best-effort)
# ----------------------------

def _parse_csv_transactions(csv_path: Path, account_label: str) -> List[Txn]:
    out: List[Txn] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            d = (row.get("Date") or row.get("Transaction Date") or row.get("Posted Date") or "").strip()
            desc = (row.get("Description") or row.get("Payee") or row.get("Name") or "").strip()
            amt_s = (row.get("Amount") or row.get("Transaction Amount") or "").strip()
            if not d or not desc or not amt_s:
                continue

            txn_dt = None
            for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
                try:
                    txn_dt = datetime.strptime(d, fmt).date()
                    break
                except Exception:
                    pass
            if not txn_dt:
                continue

            amt = _money_to_decimal(amt_s)
            if amt is None:
                continue

            debit = abs(amt) if amt < 0 else Decimal("0.00")
            credit = amt if amt > 0 else Decimal("0.00")

            fit_id = _stable_fit_id("NFCU", csv_path.name, str(idx), txn_dt.isoformat(), desc, f"{amt:.2f}")
            out.append(
                Txn(
                    account=account_label,
                    source_file=str(csv_path.as_posix()),
                    source_row=str(idx),
                    transaction_date=txn_dt.isoformat(),
                    posted_date=txn_dt.isoformat(),
                    description=desc,
                    memo="",
                    amount=f"{amt:.2f}",
                    debit=f"{debit:.2f}",
                    credit=f"{credit:.2f}",
                    currency="USD",
                    month=_to_month_key(txn_dt),
                    raw_type=row.get("Type", "") or "",
                    raw_category=row.get("Category", "") or "",
                    fit_id=fit_id,
                )
            )
    return out


# ----------------------------
# Writers
# ----------------------------

def _write_normalized(path: Path, rows: List[Txn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "account","source_file","source_row","transaction_date","posted_date",
        "description","memo","amount","debit","credit","currency","month",
        "raw_type","raw_category","fit_id"
    ]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _write_summary_by_month(path: Path, rows: List[Txn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sums: Dict[str, Dict[str, Decimal]] = {}
    for r in rows:
        m = r.month
        if not m:
            continue
        net = _money_to_decimal(r.amount) or Decimal("0.00")
        debit = _money_to_decimal(r.debit) or Decimal("0.00")
        credit = _money_to_decimal(r.credit) or Decimal("0.00")
        if m not in sums:
            sums[m] = {"count": Decimal(0), "net": Decimal(0), "debit": Decimal(0), "credit": Decimal(0)}
        sums[m]["count"] += Decimal(1)
        sums[m]["net"] += net
        sums[m]["debit"] += debit
        sums[m]["credit"] += credit

    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["month", "count", "net", "debit", "credit"])
        for m in sorted(sums.keys()):
            d = sums[m]
            w.writerow([m, int(d["count"]), f"{d['net']:.2f}", f"{d['debit']:.2f}", f"{d['credit']:.2f}"])


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    repo_root = _repo_root()

    ap = argparse.ArgumentParser(description="Import NFCU statements (CSV + PDF) into normalized CSV.")
    ap.add_argument("--in", dest="in_dir", default=str(_default_in_dir(repo_root)), help="Input folder (recursive).")
    ap.add_argument("--year", default="ALL", help="Year to import (e.g. 2025) or ALL.")
    ap.add_argument("--account", default="NFCU", help="Account label for CSV imports.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    year_filter: Optional[int] = None
    if str(args.year).upper() != "ALL":
        year_filter = int(args.year)

    out_year = year_filter if year_filter is not None else date.today().year
    out_dir = _default_out_dir(repo_root, out_year)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = []
    log_lines.append("🧾 NFCU Import")
    log_lines.append(f"- repo_root: {repo_root}")
    log_lines.append(f"- in_dir:    {in_dir}")
    log_lines.append(f"- out_dir:   {out_dir}")
    log_lines.append(f"- year:      {args.year}")
    log_lines.append(f"- account:   {args.account}")
    log_lines.append("")

    csv_files = _iter_csv_files(in_dir)
    pdf_files = _iter_pdf_files(in_dir)

    log_lines.append(f"- found_csv: {len(csv_files)}")
    log_lines.append(f"- found_pdf: {len(pdf_files)}")
    log_lines.append("")

    rows: List[Txn] = []

    # CSV import
    for f in csv_files:
        if args.verbose:
            print(f"[csv] {f}")
        rows.extend(_parse_csv_transactions(f, account_label=args.account))

    # PDF import
    for f in pdf_files:
        try:
            txns, meta = _parse_pdf_transactions(f, year_filter=year_filter, verbose=args.verbose)
            rows.extend(txns)
            if args.verbose:
                for ln in meta:
                    print(ln)
                print(f"[pdf] {f.name}: {len(txns)} rows")
            log_lines.extend(meta)
            log_lines.append(f"[pdf] {f.name}: {len(txns)} rows")
        except Exception as e:
            msg = f"[pdf][ERROR] {f.name}: {e}"
            if args.verbose:
                print(msg)
            log_lines.append(msg)

    # sort stable
    rows.sort(key=lambda t: (t.transaction_date, t.account, t.source_file, t.source_row))

    normalized_path = out_dir / "nfcu_transactions.normalized.csv"
    summary_path = out_dir / "nfcu_transactions.summary_by_month.csv"
    log_path = out_dir / "nfcu_transactions.import_log.txt"

    _write_normalized(normalized_path, rows)
    _write_summary_by_month(summary_path, rows)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8", newline="\n")

    print("wrote:")
    print(f"  {normalized_path} rows: {len(rows)}")
    print(f"  {summary_path}")
    print(f"  {log_path}")

    if len(rows) == 0:
        print("[WARN] 0 rows extracted.")
        print("[HINT] Confirm a transaction line begins with MM-DD (e.g. 12-23 ...) and contains amount/balance tokens.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
