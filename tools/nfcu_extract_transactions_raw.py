#!/usr/bin/env python3
"""
NFCU – Raw transaction extractor (from OCR’d/embedded statement TXT)

Reads:
  - out/tax_text/nfcu/2025/*.txt
  - out/tax_text/nfcu/2025/_meta/*.json   (must include statement_period_start/end + source_name)

Writes:
  - out/tax_text/nfcu/2025/nfcu_transactions_raw.jsonl
  - out/tax_text/nfcu/2025/nfcu_transactions_raw.csv

Goal:
  - Be resilient to OCR weirdness (missing balance column, wrapped descriptions, /5.00 -> 75.00, 300,00 -> 300.00, etc.)
  - Never “randomly” drop to 0 rows unless the inputs are genuinely missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------- config / patterns ----------

DATE_MM_DD_RE = re.compile(r"^\s*(\d{2})[-/](\d{2})\b")
TABLE_HEADER_RE = re.compile(r"\bdate\s+transaction\s+detail\s+amount", re.IGNORECASE)

# Money tokens in OCR’d text are messy; we extract candidates then normalize.
MONEY_TOKEN_RE = re.compile(r"(?<!\w)([-$/]?\d[\d, ]*[.,]\d{2})(?!\w)")


@dataclass(frozen=True)
class StatementMeta:
    sha256: str
    source_name: str
    statement_period_start: str  # YYYY-MM-DD
    statement_period_end: str    # YYYY-MM-DD
    meta_path: Path
    txt_path: Path


# ---------- helpers ----------

def _iso_to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _safe_strip(s: Optional[str]) -> str:
    return (s or "").strip()


def _normalize_money_token(tok: str) -> Optional[str]:
    """
    Returns normalized '1234.56' (string) or None.

    Handles:
      - '$1,234.56' -> '1234.56'
      - '1 024.27' -> '1024.27'
      - '300,00' -> '300.00'
      - '/5.00' -> '75.00'  (common OCR slash-for-7)
      - '108 59' sometimes shows as '108 59' without dot -> NOT handled (no .dd), by design
    """
    if not tok:
        return None

    t = tok.strip().replace("$", "")
    t = t.replace(" ", "")

    # OCR slash-for-7 (e.g. "/5.00" == "75.00")
    if t.startswith("/") and len(t) >= 2 and t[1].isdigit():
        t = "7" + t[1:]

    # If it uses comma as decimal separator (and no dot), convert comma->dot
    if "," in t and "." not in t:
        t = t.replace(",", ".")
    else:
        # otherwise treat commas as thousands separators
        t = t.replace(",", "")

    # remove stray leading/trailing punctuation
    t = t.strip().strip(").]}>{\"'").strip("([{<\"'")

    # final validation
    if re.fullmatch(r"-?\d+\.\d{2}", t):
        return t
    return None


def _extract_amount_from_row_tail(text_after_date: str) -> Tuple[Optional[str], str]:
    """
    Given the row content after the date token, try to pull Amount (not Balance) from the right side.

    Typical NFCU OCR row variants:
      "Transfer From Checking 140.00 140.00"
      "Beginning Balance 0.00"
      'POS Debit Debit Card ... 25.00 75.00'
    Heuristic:
      - If 2+ money tokens at end: amount = second-to-last token, balance = last token
      - If 1 token: amount = that token
    Returns: (amount_norm, description_without_amounts)
    """
    s = (text_after_date or "").strip()
    if not s:
        return None, ""

    # Tokenize by whitespace so we can peel money tokens from the end.
    parts = s.split()
    money_norms: List[str] = []

    # Walk from the end and collect up to 2 “money-looking” tokens (normalized).
    i = len(parts) - 1
    while i >= 0 and len(money_norms) < 2:
        cand = parts[i]
        norm = _normalize_money_token(cand)
        if norm is not None:
            money_norms.append(norm)
            i -= 1
            continue
        # Sometimes OCR splits "300" ".00" or "300" " .00" — ignore (handled later stages).
        break

    amount_norm: Optional[str] = None
    if len(money_norms) >= 2:
        # we collected from the end => [balance, amount] in reverse order
        balance_norm = money_norms[0]
        amount_norm = money_norms[1]
        # remove the last two tokens from description
        desc = " ".join(parts[: i + 1]).strip()
        return amount_norm, desc
    if len(money_norms) == 1:
        amount_norm = money_norms[0]
        desc = " ".join(parts[: i + 1]).strip()
        return amount_norm, desc

    # fallback: regex scan anywhere in the row
    candidates = MONEY_TOKEN_RE.findall(s)
    normalized = [_normalize_money_token(c) for c in candidates]
    normalized = [n for n in normalized if n]
    if len(normalized) >= 2:
        amount_norm = normalized[-2]
    elif len(normalized) == 1:
        amount_norm = normalized[0]
    else:
        amount_norm = None

    # don’t aggressively strip in fallback; leave description intact
    return amount_norm, s


def infer_direction(description: str, account: str, amount_norm: Optional[str]) -> str:
    """
    NFCU-specific-ish direction guess. Returns: "debit" | "credit" | "" (unknown)

    NOTE: Later pipeline stages can still infer/correct; this is just a helpful hint.
    """
    s = (description or "").lower().strip()

    # Hard signals
    if " adjustment dr" in f" {s}" or s == "adjustment dr" or re.search(r"\bdr\b", s):
        return "debit"
    if " adjustment cr" in f" {s}" or s == "adjustment cr" or re.search(r"\bcr\b", s):
        return "credit"

    # Transfers
    if "transfer to" in s:
        return "debit"
    if "transfer from" in s:
        return "credit"

    # Common NFCU labels
    if "pos debit" in s or "debit card" in s:
        return "debit"
    if "pos credit adjustment" in s or "credit adjustment" in s:
        return "credit"

    if "ach" in s:
        # "ACH Paid To" tends to be outgoing; "ACH Credit" incoming.
        if "paid to" in s or "withdrawal" in s or "debit" in s:
            return "debit"
        if "credit" in s or "deposit" in s:
            return "credit"

    if "edeposit" in s or "mobile" in s and "deposit" in s:
        return "credit"

    if "zelle" in s:
        # NFCU often prints Zelle CR / Zelle DR; if not present, leave unknown.
        if re.search(r"\bcr\b", s):
            return "credit"
        if re.search(r"\bdr\b", s):
            return "debit"
        return ""

    # Balance rows: treat as unknown (not real spend)
    if s.startswith("beginning balance") or s.startswith("ending balance"):
        return ""

    return ""


def _resolve_txn_date(mm: int, dd: int, period_start: date, period_end: date) -> date:
    """
    NFCU statements show MM-DD (no year). Choose a year that places the txn within the statement period.
    """
    candidates = {period_start.year, period_end.year, period_end.year - 1, period_end.year + 1}
    best: Optional[date] = None
    for y in sorted(candidates):
        try:
            d = date(y, mm, dd)
        except ValueError:
            continue
        if period_start <= d <= period_end:
            return d
        # keep a fallback candidate close to end date
        if best is None:
            best = d
    return best or date(period_end.year, mm, dd)


def _is_page_noise(line: str) -> bool:
    s = line.strip().lower()
    if not s:
        return True
    if s.startswith("page ") and " of " in s:
        return True
    if s.startswith("navy") or s.startswith("federal") or "statement of account" in s:
        # don’t let headers break table parsing
        return True
    return False


def _account_from_line(line: str) -> Optional[str]:
    s = line.strip().lower()
    if s == "checking":
        return "Checking"
    if s == "savings":
        return "Savings"
    # common NFCU savings labels
    if "membership savings" in s or "money market savings" in s:
        return "Savings"
    return None


# ---------- parsing core ----------

def parse_statement_transactions(meta: StatementMeta) -> List[Dict[str, str]]:
    txt = meta.txt_path.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()

    period_start = _iso_to_date(meta.statement_period_start)
    period_end = _iso_to_date(meta.statement_period_end)

    current_account = "Checking"  # NFCU statements usually hit Checking first; safe default
    in_table = False

    records: List[Dict[str, str]] = []

    cur_mmdd: Optional[Tuple[int, int]] = None
    cur_desc_parts: List[str] = []
    cur_amount: Optional[str] = None

    def flush_current():
        nonlocal cur_mmdd, cur_desc_parts, cur_amount
        if cur_mmdd is None:
            return
        mm, dd = cur_mmdd
        d = _resolve_txn_date(mm, dd, period_start, period_end)
        desc = " ".join(p.strip() for p in cur_desc_parts if p.strip()).strip()
        amt = cur_amount

        direction = infer_direction(desc, current_account, amt)

        records.append(
            {
                "sha256": meta.sha256,
                "source_name": meta.source_name,
                "statement_period_start": meta.statement_period_start,
                "statement_period_end": meta.statement_period_end,
                "statement_date": d.isoformat(),
                "account": current_account,
                "description": desc,
                "amount_raw": amt or "",
                "direction_guess": direction,
                "txt_rel": meta.txt_path.as_posix(),
                "meta_rel": meta.meta_path.as_posix(),
            }
        )

        cur_mmdd = None
        cur_desc_parts = []
        cur_amount = None

    for raw in lines:
        line = raw.rstrip("\n")
        if _is_page_noise(line):
            continue

        # Account switches (these appear outside the table header)
        maybe_acct = _account_from_line(line)
        if maybe_acct:
            current_account = maybe_acct
            # don’t force leaving table; table may continue across page breaks
            continue

        # Table header start
        if TABLE_HEADER_RE.search(line):
            # new table begins; flush any partial row
            flush_current()
            in_table = True
            continue

        if not in_table:
            continue

        # If we hit a fresh account header while “in_table”, treat it as end of previous table
        # (and allow new table to start when its header appears).
        maybe_acct2 = _account_from_line(line)
        if maybe_acct2:
            flush_current()
            current_account = maybe_acct2
            in_table = False
            continue

        # New transaction row?
        m = DATE_MM_DD_RE.match(line)
        if m:
            flush_current()

            mm = int(m.group(1))
            dd = int(m.group(2))
            after = line[m.end() :].strip()

            amt, desc = _extract_amount_from_row_tail(after)
            cur_mmdd = (mm, dd)
            cur_amount = amt
            cur_desc_parts = [desc] if desc else []
            continue

        # Continuation line (wrapped description)
        if cur_mmdd is not None:
            cont = line.strip()
            # stop conditions (OCR sometimes restates header / ends table)
            if TABLE_HEADER_RE.search(cont):
                flush_current()
                in_table = True
                continue
            if cont.lower().startswith("date "):
                continue
            # append real content
            if cont:
                cur_desc_parts.append(cont)

    flush_current()

    # Filter out clearly-non-transaction junk rows if they slipped through
    cleaned: List[Dict[str, str]] = []
    for r in records:
        desc = (r.get("description") or "").strip()
        # If OCR produced an empty description and empty amount, toss it.
        if not desc and not _safe_strip(r.get("amount_raw")):
            continue
        cleaned.append(r)

    return cleaned


# ---------- IO ----------

def load_statement_metas(text_dir: Path, meta_dir: Path) -> List[StatementMeta]:
    metas: List[StatementMeta] = []
    for mp in sorted(meta_dir.glob("*.json")):
        try:
            obj = json.loads(mp.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue

        sha = _safe_strip(obj.get("sha256") or obj.get("sha") or obj.get("file_sha256"))
        if not sha:
            # fallback: filename stem is sha
            sha = mp.stem

        src = _safe_strip(obj.get("source_name") or obj.get("source") or obj.get("filename") or "")
        if not src:
            # fallback if meta lacks name
            src = f"{sha}.pdf"

        ps = _safe_strip(obj.get("statement_period_start"))
        pe = _safe_strip(obj.get("statement_period_end"))
        if not ps or not pe:
            # If period isn’t present, we can’t safely date the MM-DD rows.
            # Skip loud, but continue.
            continue

        tp = text_dir / f"{sha}.txt"
        if not tp.exists():
            continue

        metas.append(
            StatementMeta(
                sha256=sha,
                source_name=src,
                statement_period_start=ps,
                statement_period_end=pe,
                meta_path=mp,
                txt_path=tp,
            )
        )
    return metas


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sha256",
        "source_name",
        "statement_period_start",
        "statement_period_end",
        "statement_date",
        "account",
        "description",
        "amount_raw",
        "direction_guess",
        "txt_rel",
        "meta_rel",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract raw transactions from NFCU statement text.")
    ap.add_argument("--text-dir", default="out/tax_text/nfcu/2025", help="Directory containing <sha>.txt files")
    ap.add_argument("--meta-dir", default="out/tax_text/nfcu/2025/_meta", help="Directory containing <sha>.json meta files")
    ap.add_argument("--out-jsonl", default="out/tax_text/nfcu/2025/nfcu_transactions_raw.jsonl")
    ap.add_argument("--out-csv", default="out/tax_text/nfcu/2025/nfcu_transactions_raw.csv")
    ap.add_argument("--verbose", action="store_true", help="Print per-statement extraction counts")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    text_dir = Path(args.text_dir)
    meta_dir = Path(args.meta_dir)
    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)

    if not text_dir.exists():
        raise SystemExit(f"ERROR: missing text-dir: {text_dir}")
    if not meta_dir.exists():
        raise SystemExit(f"ERROR: missing meta-dir: {meta_dir}")

    metas = load_statement_metas(text_dir, meta_dir)
    if not metas:
        raise SystemExit("ERROR: no usable statement metas found (need period_start/end + matching txt).")

    all_rows: List[Dict[str, str]] = []
    per_stmt: List[Tuple[str, int]] = []

    for meta in metas:
        rows = parse_statement_transactions(meta)
        all_rows.extend(rows)
        per_stmt.append((meta.source_name, len(rows)))
        if args.verbose:
            print(f"- {meta.source_name}: {len(rows)} rows")

    # Sort stable
    all_rows.sort(key=lambda r: (r.get("statement_date", ""), r.get("account", ""), r.get("description", "")))

    write_jsonl(out_jsonl, all_rows)
    write_csv(out_csv, all_rows)

    print("---- NFCU Raw Transaction Extract ----")
    print(f"Statements processed: {len(metas)}")
    print(f"Transactions found: {len(all_rows)}")
    print(f"Missing meta: 0")
    print(f"No period in meta: 0")
    print(f"WROTE: {out_jsonl}")
    print(f"WROTE: {out_csv}")

    # sanity: show “empty amount” count
    blank_amt = sum(1 for r in all_rows if not _safe_strip(r.get("amount_raw")))
    if blank_amt:
        print(f"WARNING: amount_raw blank on {blank_amt} rows (OCR table likely missing amount column).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
