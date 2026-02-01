# tools/tax_parse_pnc_statements.py
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional


# ----------------------------
# Regex + section definitions
# ----------------------------

period_re = re.compile(
    r"For the period\s+(?P<s>\d{2}/\d{2}/\d{4})\s+to\s+(?P<e>\d{2}/\d{2}/\d{4})",
    re.I,
)

section_markers = [
    ("deposits", re.compile(r"^Deposits and Other Additions", re.I)),
    ("withdrawals", re.compile(r"^Banking/Debit Card Withdrawals and Purchases", re.I)),
    ("checks", re.compile(r"^Checks Paid", re.I)),
    ("fees", re.compile(r"^Service Charges and Fees", re.I)),
    ("other", re.compile(r"^Other Deductions", re.I)),
]

tx_line_re = re.compile(r"^(?P<mmdd>\d{2}/\d{2})\s+(?P<amt>-?\d[\d,]*\.\d{2})\s+(?P<desc>.+)$")


# ----------------------------
# Helpers
# ----------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_mmdd(mmdd: str) -> tuple[int, int]:
    m, d = mmdd.split("/")
    return int(m), int(d)

def pick_full_date(mmdd: str, start: date, end: date) -> Optional[date]:
    """Pick a full date for an MM/DD that best fits inside the statement period."""
    m, d = parse_mmdd(mmdd)
    candidates: list[date] = []
    for y in (end.year, start.year, end.year - 1, end.year + 1, start.year - 1, start.year + 1):
        try:
            candidates.append(date(y, m, d))
        except ValueError:
            pass

    if not candidates:
        return None

    inside = [c for c in candidates if start <= c <= end]
    if inside:
        # choose the inside candidate closest to statement end (usually posted near end)
        return min(inside, key=lambda c: abs((end - c).days))

    # otherwise pick closest overall
    return min(candidates, key=lambda c: abs((end - c).days))

def infer_tx_type(section: str, raw_amount: float) -> str:
    # PNC statements usually list deposits as positive credits,
    # withdrawals/purchases as positive numbers (but effectively debits).
    if section == "deposits":
        return "credit"
    if section in ("withdrawals", "fees", "checks", "other"):
        return "debit"
    # fallback: if it looks negative, treat as credit reversal
    return "credit" if raw_amount < 0 else "debit"

def normalize_whitespace(s: str) -> str:
    return " ".join(s.split()).strip()

def clean_merchant(desc: str) -> str:
    d = normalize_whitespace(desc)

    # strip common PNC prefixes
    for prefix in (
        "6157 Debit Card Purchase ",
        "6157 Recurring Debit Card ",
        "6157 Debit Card/Bankcard ",
        "Zel To ",
        "Zel From ",
        "Zelle To ",
        "Zelle From ",
        "Corporate ACH Payroll ",
        "Web Pmt- Deposit ",
        "Direct Payment - ",
    ):
        if d.startswith(prefix):
            d = d[len(prefix):].strip()
            break

    return d[:80].strip()


# ----------------------------
# PDF extraction (optional)
# ----------------------------

def extract_pdf_to_text(pdf_path: Path) -> str:
    """
    Best-effort PDF -> text. Uses pypdf if available.
    If your extraction pipeline already creates .txt files, you can skip extraction.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PDF extraction requires 'pypdf'. Install it (pip install pypdf) or run with --no-extract and provide .txt files."
        ) from e

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts)


# ----------------------------
# Parsing
# ----------------------------

@dataclass(frozen=True)
class ParseStats:
    emitted: int
    skipped_consecutive_dups: int
    start: date
    end: date
    min_date: Optional[date]
    max_date: Optional[date]

def parse_one(txt_path: Path) -> tuple[list[dict], ParseStats]:
    """
    Parse one extracted text statement file.
    Drops only *consecutive* duplicate transaction lines (common OCR artifact).
    """
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # statement period
    start = end = None
    for line in lines[:200]:
        m = period_re.search(line)
        if m:
            start = datetime.strptime(m.group("s"), "%m/%d/%Y").date()
            end   = datetime.strptime(m.group("e"), "%m/%d/%Y").date()
            break
    if not (start and end):
        # fallback: a wide net; caller will filter by year anyway
        start = date(1900, 1, 1)
        end   = date(2100, 12, 31)

    in_activity = False
    section: Optional[str] = None
    out: list[dict] = []

    last_key = None
    skipped_dups = 0
    min_dt: Optional[date] = None
    max_dt: Optional[date] = None

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()

        if "Activity Detail" in line:
            in_activity = True
            i += 1
            continue
        if not in_activity:
            i += 1
            continue

        # section switching
        for name, rx in section_markers:
            if rx.match(line):
                section = name
                break

        m = tx_line_re.match(line)
        if m and section:
            mmdd = m.group("mmdd")
            raw_amt = float(m.group("amt").replace(",", ""))
            desc = m.group("desc").strip()

            # merge wrapped description lines
            j = i + 1
            while j < n:
                nxt_strip = lines[j].strip()
                if not nxt_strip:
                    break
                if tx_line_re.match(nxt_strip):
                    break
                if any(rx.match(nxt_strip) for _, rx in section_markers):
                    break
                desc += " " + nxt_strip
                j += 1

            full_dt = pick_full_date(mmdd, start, end)
            tx_type = infer_tx_type(section, raw_amt)

            signed_amt = raw_amt
            # Normalize sign conventions: debits are negative; credits are positive
            if tx_type == "debit" and signed_amt > 0:
                signed_amt = -signed_amt
            if tx_type == "credit" and signed_amt < 0:
                signed_amt = -signed_amt

            norm_desc = normalize_whitespace(desc)

            # Drop only consecutive exact duplicates (same file + section + line essentials)
            key = (txt_path.stem, section, mmdd, f"{signed_amt:.2f}", norm_desc)
            if key == last_key:
                skipped_dups += 1
                i = j
                continue
            last_key = key

            if full_dt:
                min_dt = full_dt if (min_dt is None or full_dt < min_dt) else min_dt
                max_dt = full_dt if (max_dt is None or full_dt > max_dt) else max_dt

            out.append({
                "sha": txt_path.stem,
                "statement_start": start.isoformat(),
                "statement_end": end.isoformat(),
                "date": full_dt.isoformat() if full_dt else "",
                "mmdd": mmdd,
                "section": section,
                "tx_type": tx_type,
                "amount": f"{signed_amt:.2f}",
                "merchant": clean_merchant(desc),
                "description": norm_desc,
            })

            i = j
            continue

        i += 1

    stats = ParseStats(
        emitted=len(out),
        skipped_consecutive_dups=skipped_dups,
        start=start,
        end=end,
        min_date=min_dt,
        max_date=max_dt,
    )
    return out, stats


# ----------------------------
# CLI
# ----------------------------

def iter_input_files(in_path: Path) -> tuple[list[Path], list[Path]]:
    """
    Returns (pdfs, txts).
    """
    if in_path.is_file():
        if in_path.suffix.lower() == ".pdf":
            return [in_path], []
        if in_path.suffix.lower() == ".txt":
            return [], [in_path]
        return [], []

    pdfs = sorted(in_path.glob("*.pdf"))
    txts = sorted(in_path.glob("*.txt"))
    return pdfs, txts

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parse PNC Spend statements into a normalized transaction CSV (with OCR de-dupe)."
    )
    ap.add_argument(
        "--in",
        dest="in_dir",
        default="notes/tax/intake/bank/2025/PNC_Documents",
        help="Input directory (PDFs and/or extracted .txt).",
    )
    ap.add_argument(
        "--year",
        type=int,
        required=True,
        help="Tax year to keep (filters parsed transactions by transaction date year).",
    )
    ap.add_argument(
        "--extracted-dir",
        default=None,
        help="Where to write extracted text (if PDFs are provided). Default: notes/tax/work/extracted_text/<year>/",
    )
    ap.add_argument(
        "--no-extract",
        action="store_true",
        help="Do not extract PDFs; only parse existing .txt files.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: notes/tax/work/parsed/<year>/pnc_spend_transactions.enriched2.csv",
    )

    args = ap.parse_args()
    year: int = args.year

    in_dir = Path(args.in_dir)
    extracted_dir = Path(args.extracted_dir) if args.extracted_dir else Path(f"notes/tax/work/extracted_text/{year}")
    out_csv = Path(args.out) if args.out else Path(f"notes/tax/work/parsed/{year}/pnc_spend_transactions.enriched2.csv")

    extracted_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pdfs, txts = iter_input_files(in_dir)

    # If PDFs exist and extraction is allowed, extract them to extracted_dir as <sha>.txt
    if pdfs and not args.no_extract:
        for pdf in pdfs:
            sha = sha256_file(pdf)
            txt_path = extracted_dir / f"{sha}.txt"
            if txt_path.exists() and txt_path.stat().st_size > 0:
                continue
            text = extract_pdf_to_text(pdf)
            txt_path.write_text(text, encoding="utf-8", newline="\n")

    # Determine which .txt files to parse:
    # - if input dir directly has txts, parse those
    # - else parse extracted_dir/*.txt (the canonical work area)
    if txts:
        parse_txts = txts
    else:
        parse_txts = sorted(extracted_dir.glob("*.txt"))

    all_rows: list[dict] = []
    total_skipped_dups = 0
    coverage_min: Optional[date] = None
    coverage_max: Optional[date] = None

    for txt in parse_txts:
        rows, stats = parse_one(txt)

        total_skipped_dups += stats.skipped_consecutive_dups
        if stats.min_date:
            coverage_min = stats.min_date if (coverage_min is None or stats.min_date < coverage_min) else coverage_min
        if stats.max_date:
            coverage_max = stats.max_date if (coverage_max is None or stats.max_date > coverage_max) else coverage_max

        # Filter to requested year based on parsed transaction date
        for r in rows:
            dt = r.get("date", "")
            if not dt or len(dt) < 4:
                continue
            if int(dt[:4]) != year:
                continue
            all_rows.append(r)

    # Stable sort for sanity/debugging
    def sort_key(r: dict):
        return (
            r.get("date", ""),
            float(r.get("amount", "0") or 0),
            r.get("description", ""),
            r.get("sha", ""),
            r.get("mmdd", ""),
        )

    all_rows.sort(key=sort_key)

    fieldnames = ["sha","statement_start","statement_end","date","mmdd","section","tx_type","amount","merchant","description"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print("wrote:", out_csv)
    print("rows:", len(all_rows))
    if coverage_min and coverage_max:
        print(f"PNC statement coverage: {coverage_min.isoformat()} -> {coverage_max.isoformat()}")
    if total_skipped_dups:
        print("skipped consecutive OCR dups:", total_skipped_dups)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
