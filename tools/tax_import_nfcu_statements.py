#!/usr/bin/env python3
"""
NFCU statement extractor → normalized transactions CSV + monthly summary.

Goal:
- Ingest Navy Federal Credit Union exports (CSV) from one or more folders
- Normalize to a consistent schema
- Output:
  - nfcu_transactions.normalized.csv
  - nfcu_transactions.summary_by_month.csv
  - nfcu_transactions.import_log.txt

Design notes:
- Robust header detection (NFCU exports vary)
- Defensive parsing (dates, amounts, blanks, weird currency formatting)
- Deterministic outputs (stable column order, stable sorting)
- Meaningful logging (what was found, what parsed, what got skipped)

Typical usage:
  python tools/tax_import_nfcu_statements.py --year 2025

Optional:
  python tools/tax_import_nfcu_statements.py --in notes/tax/statements/nfcu --year 2025 --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---- Output schema (stable order) -------------------------------------------

NORMALIZED_COLUMNS: Tuple[str, ...] = (
    "account",
    "source_file",
    "source_row",
    "transaction_date",
    "posted_date",
    "description",
    "memo",
    "amount",
    "debit",
    "credit",
    "currency",
    "month",
    "raw_type",
    "raw_category",
    "fit_id",
)

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "month",
    "tx_count",
    "total_credit",
    "total_debit",
    "net",
)

# ---- Parsing helpers ---------------------------------------------------------

_DATE_PATTERNS = (
    "%m/%d/%Y",
    "%m/%d/%y",
    "%Y-%m-%d",
    "%m-%d-%Y",
)

_CURRENCY_RE = re.compile(r"[,\s$]")

def _parse_date(value: str) -> Optional[date]:
    v = (value or "").strip()
    if not v:
        return None
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    # Some exports include timestamps
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).date()
    except Exception:
        return None

def _parse_amount(value: str) -> Optional[float]:
    v = (value or "").strip()
    if not v:
        return None
    # normalize parentheses negatives: (12.34) => -12.34
    neg = False
    if v.startswith("(") and v.endswith(")"):
        neg = True
        v = v[1:-1]
    v = _CURRENCY_RE.sub("", v)
    # Handle stray plus signs
    v = v.replace("+", "")
    try:
        amt = float(v)
        return -amt if neg else amt
    except ValueError:
        return None

def _month_key(d: Optional[date]) -> str:
    if not d:
        return ""
    return f"{d.year:04d}-{d.month:02d}"

def _safe_str(x: object) -> str:
    if x is None:
        return ""
    return str(x).strip()

def _norm_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (h or "").strip().lower()).strip("_")

# ---- NFCU CSV mapping --------------------------------------------------------

# NFCU exports vary; these are the most common header names across variants.
HEADER_ALIASES: Dict[str, Sequence[str]] = {
    "transaction_date": ("transaction_date", "date", "trans_date", "transactiondate"),
    "posted_date": ("posted_date", "post_date", "posted", "posting_date"),
    "description": ("description", "desc", "merchant", "payee", "name"),
    "memo": ("memo", "notes", "details"),
    "amount": ("amount", "amt", "transaction_amount", "value"),
    "debit": ("debit", "withdrawal", "outflow"),
    "credit": ("credit", "deposit", "inflow"),
    "raw_type": ("type", "transaction_type"),
    "raw_category": ("category", "transaction_category"),
    "fit_id": ("fit_id", "fitid", "id", "transaction_id"),
    "currency": ("currency", "ccy"),
}

@dataclass(frozen=True)
class NormalizedTx:
    account: str
    source_file: str
    source_row: int
    transaction_date: str
    posted_date: str
    description: str
    memo: str
    amount: str
    debit: str
    credit: str
    currency: str
    month: str
    raw_type: str
    raw_category: str
    fit_id: str

    def as_row(self) -> Dict[str, str]:
        return {
            "account": self.account,
            "source_file": self.source_file,
            "source_row": str(self.source_row),
            "transaction_date": self.transaction_date,
            "posted_date": self.posted_date,
            "description": self.description,
            "memo": self.memo,
            "amount": self.amount,
            "debit": self.debit,
            "credit": self.credit,
            "currency": self.currency,
            "month": self.month,
            "raw_type": self.raw_type,
            "raw_category": self.raw_category,
            "fit_id": self.fit_id,
        }

def _build_header_index(fieldnames: Sequence[str]) -> Dict[str, int]:
    """
    Returns canonical_field -> column index mapping, based on aliases.
    """
    normed = [_norm_header(h) for h in fieldnames]
    idx: Dict[str, int] = {}

    for canonical, aliases in HEADER_ALIASES.items():
        for a in aliases:
            a_norm = _norm_header(a)
            if a_norm in normed:
                idx[canonical] = normed.index(a_norm)
                break

    return idx

def _get_cell(row: Sequence[str], idx: Dict[str, int], key: str) -> str:
    i = idx.get(key)
    if i is None or i < 0 or i >= len(row):
        return ""
    return _safe_str(row[i])

# ---- File discovery ----------------------------------------------------------

def _repo_root_from_script() -> Path:
    # tools/tax_import_nfcu_statements.py -> repo root is two parents up
    return Path(__file__).resolve().parents[1]

def _default_in_dir(repo_root: Path) -> Path:
    # You can change this default without breaking CLI flags.
    # Keep it opinionated but overridable.
    candidates = [
        repo_root / "notes" / "tax" / "statements" / "nfcu",
        repo_root / "notes" / "tax" / "work" / "statements" / "nfcu",
        repo_root / "notes" / "tax" / "statements",
        repo_root / "notes" / "tax" / "work" / "statements",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    # fall back to a sensible default even if it doesn't exist yet
    return repo_root / "notes" / "tax" / "statements" / "nfcu"

def _default_out_dir(repo_root: Path, year: int) -> Path:
    return repo_root / "notes" / "tax" / "work" / "parsed" / str(year)

def _iter_csv_files(in_dir: Path) -> List[Path]:
    if not in_dir.exists():
        return []
    files = sorted([p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"])
    return files

# ---- Core ingest -------------------------------------------------------------

def ingest_nfcu_csv(
    csv_path: Path,
    account: str,
    year_filter: Optional[int],
    log_lines: List[str],
) -> List[NormalizedTx]:
    """
    Read one NFCU CSV and return normalized transactions.
    """
    out: List[NormalizedTx] = []
    rel = str(csv_path)

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        log_lines.append(f"[ERROR] Failed to read CSV: {rel} :: {e}")
        return out

    if not rows:
        log_lines.append(f"[WARN] Empty CSV: {rel}")
        return out

    # Find the header row: first row with at least 2 non-empty cells
    header_row_i = None
    for i, r in enumerate(rows[:50]):  # don't scan forever
        nonempty = sum(1 for c in r if _safe_str(c))
        if nonempty >= 2:
            header_row_i = i
            break

    if header_row_i is None:
        log_lines.append(f"[WARN] No header row detected: {rel}")
        return out

    header = rows[header_row_i]
    idx = _build_header_index(header)

    # Require at minimum a date and description and *some* money field
    has_date = "transaction_date" in idx or "posted_date" in idx
    has_desc = "description" in idx
    has_money = ("amount" in idx) or ("debit" in idx) or ("credit" in idx)

    if not (has_date and has_desc and has_money):
        log_lines.append(
            "[WARN] Unrecognized NFCU CSV headers (missing required fields). "
            f"file={rel} headers={[_norm_header(h) for h in header]}"
        )
        return out

    data_rows = rows[header_row_i + 1 :]
    parsed = 0
    kept = 0
    skipped = 0

    for row_num, r in enumerate(data_rows, start=header_row_i + 2):  # 1-based line-ish
        parsed += 1

        tx_date = _parse_date(_get_cell(r, idx, "transaction_date")) or _parse_date(_get_cell(r, idx, "posted_date"))
        posted = _parse_date(_get_cell(r, idx, "posted_date"))
        desc = _get_cell(r, idx, "description")
        memo = _get_cell(r, idx, "memo")
        currency = _get_cell(r, idx, "currency") or "USD"
        raw_type = _get_cell(r, idx, "raw_type")
        raw_category = _get_cell(r, idx, "raw_category")
        fit_id = _get_cell(r, idx, "fit_id")

        # Money: prefer explicit amount; else compute credit - debit
        amt = _parse_amount(_get_cell(r, idx, "amount"))
        debit = _parse_amount(_get_cell(r, idx, "debit"))
        credit = _parse_amount(_get_cell(r, idx, "credit"))

        if amt is None:
            # If only one side exists, treat it as signed
            if credit is not None and debit is None:
                amt = abs(credit)
            elif debit is not None and credit is None:
                amt = -abs(debit)
            elif credit is not None and debit is not None:
                # Some exports provide both as positives
                amt = abs(credit) - abs(debit)
            else:
                skipped += 1
                continue

        # Normalize debit/credit fields
        if amt < 0:
            debit_val = abs(amt)
            credit_val = 0.0
        else:
            debit_val = 0.0
            credit_val = abs(amt)

        if year_filter is not None and tx_date is not None and tx_date.year != year_filter:
            skipped += 1
            continue

        tx_date_s = tx_date.isoformat() if tx_date else ""
        posted_s = posted.isoformat() if posted else ""
        month = _month_key(tx_date or posted)

        out.append(
            NormalizedTx(
                account=account,
                source_file=str(csv_path.name),
                source_row=row_num,
                transaction_date=tx_date_s,
                posted_date=posted_s,
                description=desc,
                memo=memo,
                amount=f"{amt:.2f}",
                debit=f"{debit_val:.2f}",
                credit=f"{credit_val:.2f}",
                currency=currency,
                month=month,
                raw_type=raw_type,
                raw_category=raw_category,
                fit_id=fit_id,
            )
        )
        kept += 1

    log_lines.append(f"[OK] {rel}: parsed={parsed} kept={kept} skipped={skipped}")
    return out

# ---- Output writers ----------------------------------------------------------

def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def summarize_by_month(txs: Sequence[NormalizedTx]) -> List[Dict[str, str]]:
    agg: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for t in txs:
        m = t.month or ""
        if not m:
            continue
        counts[m] = counts.get(m, 0) + 1
        a = agg.setdefault(m, {"credit": 0.0, "debit": 0.0})
        a["credit"] += float(t.credit or "0")
        a["debit"] += float(t.debit or "0")

    out: List[Dict[str, str]] = []
    for m in sorted(counts.keys()):
        credit = agg[m]["credit"]
        debit = agg[m]["debit"]
        net = credit - debit
        out.append(
            {
                "month": m,
                "tx_count": str(counts[m]),
                "total_credit": f"{credit:.2f}",
                "total_debit": f"{debit:.2f}",
                "net": f"{net:.2f}",
            }
        )
    return out

# ---- CLI --------------------------------------------------------------------

def build_arg_parser(repo_root: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tax_import_nfcu_statements",
        description="Extract NFCU CSV exports → normalized transactions + monthly summary.",
    )
    p.add_argument("--in", dest="in_dir", default=str(_default_in_dir(repo_root)), help="Input folder (recursive).")
    p.add_argument("--out", dest="out_dir", default="", help="Output folder (defaults to notes/tax/work/parsed/<year>).")
    p.add_argument("--year", type=int, default=0, help="Filter to this year (0 = no filter).")
    p.add_argument("--account", default="NFCU", help="Account label to stamp on each transaction.")
    p.add_argument("--verbose", action="store_true", help="Print log lines to stdout.")
    p.add_argument("--dry-run", action="store_true", help="Parse but do not write outputs.")
    return p

def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = _repo_root_from_script()
    args = build_arg_parser(repo_root).parse_args(argv)

    in_dir = Path(args.in_dir).expanduser().resolve()
    year_filter = args.year if args.year and args.year > 0 else None
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir(repo_root, year_filter or date.today().year)

    log_lines: List[str] = []
    log_lines.append("🧾 NFCU Import")
    log_lines.append(f"- repo_root: {repo_root}")
    log_lines.append(f"- in_dir:    {in_dir}")
    log_lines.append(f"- out_dir:   {out_dir}")
    log_lines.append(f"- year:      {year_filter if year_filter is not None else 'ALL'}")
    log_lines.append(f"- account:   {args.account}")
    log_lines.append("")

    files = _iter_csv_files(in_dir)
    if not files:
        log_lines.append("[WARN] No CSV files found under input dir.")
        _emit_log(out_dir, log_lines, verbose=args.verbose, dry_run=args.dry_run)
        # Still write empty outputs (deterministic) unless dry-run
        if not args.dry_run:
            _write_empty_outputs(out_dir, log_lines, verbose=args.verbose)
        return 0

    all_txs: List[NormalizedTx] = []
    for f in files:
        all_txs.extend(ingest_nfcu_csv(f, account=args.account, year_filter=year_filter, log_lines=log_lines))

    # Stable sort: date, description, amount, source
    def sort_key(t: NormalizedTx) -> Tuple[str, str, str, str, int]:
        return (t.transaction_date or t.posted_date or "", t.description, t.amount, t.source_file, int(t.source_row))

    all_txs = sorted(all_txs, key=sort_key)

    normalized_rows = [t.as_row() for t in all_txs]
    summary_rows = summarize_by_month(all_txs)

    normalized_path = out_dir / "nfcu_transactions.normalized.csv"
    summary_path = out_dir / "nfcu_transactions.summary_by_month.csv"
    log_path = out_dir / "nfcu_transactions.import_log.txt"

    log_lines.append("")
    log_lines.append(f"rows: {len(all_txs)}")
    log_lines.append(f"normalized: {normalized_path}")
    log_lines.append(f"summary:    {summary_path}")
    log_lines.append(f"log:        {log_path}")

    if args.dry_run:
        _emit_log(out_dir, log_lines, verbose=True, dry_run=True)
        return 0

    write_csv(normalized_path, NORMALIZED_COLUMNS, normalized_rows)
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if args.verbose:
        print("\n".join(log_lines))

    return 0

def _write_empty_outputs(out_dir: Path, log_lines: List[str], verbose: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = out_dir / "nfcu_transactions.normalized.csv"
    summary_path = out_dir / "nfcu_transactions.summary_by_month.csv"
    log_path = out_dir / "nfcu_transactions.import_log.txt"

    write_csv(normalized_path, NORMALIZED_COLUMNS, [])
    write_csv(summary_path, SUMMARY_COLUMNS, [])
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if verbose:
        print("\n".join(log_lines))
        print(f"[OK] wrote empty outputs to {out_dir}")

def _emit_log(out_dir: Path, log_lines: List[str], verbose: bool, dry_run: bool) -> None:
    if verbose:
        print("\n".join(log_lines))
    if dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nfcu_transactions.import_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    raise SystemExit(main())
