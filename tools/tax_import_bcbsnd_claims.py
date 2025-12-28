# tools/tax_import_bcbsnd_claims.py
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Where your BCBSND intake lives
IN_DIR = Path("notes/tax/intake/insurance/2025/BCBSND")
OUT_DIR = Path("notes/tax/work/parsed/2025")

# Expected filenames (you can tweak later)
OVERVIEW_XLSX = "Swain_Claims_Overview.xlsx"
EXPORT_XLS = "claimsReportExport.xls"


def _log(msg: str) -> None:
    print(msg)


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None


def _coerce_date_iso(val) -> str:
    """
    Best-effort normalize dates to YYYY-MM-DD string.
    Handles pandas Timestamp, python datetime/date, or already-string.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if not s or s.lower() in {"nan", "nat"}:
        return ""
    # If pandas timestamp prints with time, keep only date
    if " " in s and s[:4].isdigit() and s[4] == "-":
        return s.split(" ", 1)[0]
    return s


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def import_overview_xlsx(pd) -> Tuple[Path, int]:
    src = IN_DIR / OVERVIEW_XLSX
    if not src.exists():
        raise FileNotFoundError(f"Missing: {src}")

    xls = pd.ExcelFile(src)
    # Most files like this have the data on the first sheet
    df = xls.parse(xls.sheet_names[0])

    # Normalize column names lightly
    cols = [c.strip() for c in df.columns.tolist()]
    df.columns = cols

    # Expected (from your screenshot): Date, Service, Provider/Facility, Family Member, Category, Your Responsibility
    # But we’ll be defensive.
    def colpick(*names: str) -> Optional[str]:
        lower = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    c_date = colpick("Date")
    c_service = colpick("Service")
    c_provider = colpick("Provider/Facility", "Provider", "Facility")
    c_member = colpick("Family Member", "Member", "Patient")
    c_cat = colpick("Category")
    c_resp = colpick("Your Responsibility", "Responsibility", "Member Responsibility", "Patient Responsibility")

    if not c_date:
        raise RuntimeError("BCBSND overview: could not find a Date column.")

    out_rows: List[Dict] = []
    for i, row in df.iterrows():
        r = {
            "source": "BCBSND",
            "source_file": OVERVIEW_XLSX,
            "source_sheet": xls.sheet_names[0],
            "row_index": int(i),
            "service_date": _coerce_date_iso(row.get(c_date)),
            "service": str(row.get(c_service, "")).strip() if c_service else "",
            "provider": str(row.get(c_provider, "")).strip() if c_provider else "",
            "member": str(row.get(c_member, "")).strip() if c_member else "",
            "bcbs_category": str(row.get(c_cat, "")).strip() if c_cat else "",
            "member_responsibility": row.get(c_resp, "") if c_resp else "",
        }
        # Drop obviously empty rows
        if not r["service_date"] and not r["service"] and not r["provider"]:
            continue
        out_rows.append(r)

    out_path = OUT_DIR / "bcbsnd_claims_overview.normalized.csv"
    fieldnames = [
        "source", "source_file", "source_sheet", "row_index",
        "service_date", "service", "provider", "member", "bcbs_category",
        "member_responsibility",
    ]
    _write_csv(out_path, fieldnames, out_rows)
    return out_path, len(out_rows)


def import_export_xls(pd) -> Tuple[Optional[Path], int, str]:
    """
    claimsReportExport.xls needs xlrd installed (pandas uses xlrd for .xls).
    If unavailable, we emit a friendly instruction and keep going.
    """
    src = IN_DIR / EXPORT_XLS
    if not src.exists():
        return None, 0, f"Missing: {src}"

    try:
        xls = pd.ExcelFile(src)  # will fail if xlrd not installed
        sheet = xls.sheet_names[0]
        df = xls.parse(sheet)
    except Exception as e:
        msg = (
            f"Could not read {EXPORT_XLS} (likely missing xlrd).\n"
            f"Fix:\n"
            f"  ./.venv_tax/Scripts/python -m pip install xlrd==2.0.1\n"
            f"Then re-run this importer.\n"
            f"Error was: {type(e).__name__}: {e}"
        )
        return None, 0, msg

    # Save a raw dump and a lightly-normalized version.
    raw_path = OUT_DIR / "bcbsnd_claims_export.raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8")

    # Light normalization: keep original columns but add a few common ones if we can detect them.
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]

    def find_col_contains(*needles: str) -> Optional[str]:
        for c in df2.columns:
            lc = c.lower()
            if all(n.lower() in lc for n in needles):
                return c
        return None

    # Heuristic “usual suspects”
    c_service_from = find_col_contains("service", "from") or find_col_contains("service", "date")
    c_service_to = find_col_contains("service", "to")
    c_provider = find_col_contains("provider") or find_col_contains("facility")
    c_claim = find_col_contains("claim")
    c_status = find_col_contains("status")
    c_paid = find_col_contains("paid")
    c_resp = find_col_contains("respons")

    norm_rows: List[Dict] = []
    for i, row in df2.iterrows():
        norm_rows.append({
            "source": "BCBSND",
            "source_file": EXPORT_XLS,
            "source_sheet": xls.sheet_names[0],
            "row_index": int(i),
            "claim_id": str(row.get(c_claim, "")).strip() if c_claim else "",
            "status": str(row.get(c_status, "")).strip() if c_status else "",
            "service_from": _coerce_date_iso(row.get(c_service_from)) if c_service_from else "",
            "service_to": _coerce_date_iso(row.get(c_service_to)) if c_service_to else "",
            "provider": str(row.get(c_provider, "")).strip() if c_provider else "",
            "paid_amount": row.get(c_paid, "") if c_paid else "",
            "member_responsibility": row.get(c_resp, "") if c_resp else "",
        })

    norm_path = OUT_DIR / "bcbsnd_claims_export.normalized.csv"
    _write_csv(
        norm_path,
        [
            "source", "source_file", "source_sheet", "row_index",
            "claim_id", "status", "service_from", "service_to",
            "provider", "paid_amount", "member_responsibility",
        ],
        norm_rows,
    )
    return norm_path, len(norm_rows), "OK"


def main() -> int:
    _ensure_dirs()

    if not IN_DIR.exists():
        _log(f"Missing intake folder: {IN_DIR}")
        _log("Create it and drop BCBSND exports there.")
        return 2

    pd = _try_import_pandas()
    if pd is None:
        _log("Missing dependency: pandas")
        _log("Fix:")
        _log("  ./.venv_tax/Scripts/python -m pip install pandas openpyxl xlrd==2.0.1")
        return 2

    wrote: List[Tuple[Path, int]] = []

    # 1) Overview xlsx (high value, usually easy)
    try:
        p1, n1 = import_overview_xlsx(pd)
        wrote.append((p1, n1))
    except Exception as e:
        _log(f"ERROR importing {OVERVIEW_XLSX}: {type(e).__name__}: {e}")

    # 2) Detailed xls (nice to have; may require xlrd)
    p2, n2, msg = import_export_xls(pd)
    if p2 is not None:
        wrote.append((p2, n2))
    else:
        _log(msg)

    _log("\nwrote:")
    for p, n in wrote:
        _log(f"  {p} rows: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
