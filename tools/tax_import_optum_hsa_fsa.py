# tools/tax_import_optum_hsa_fsa.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OUT_DIR = Path("notes/tax/work/parsed/2025")

# Accept both layouts:
#   notes/tax/intake/hsa_fsa/2025/Optum/optumFSAHSAclaimsReportExport.xls
#   notes/tax/intake/hsa_fsa/2025/optumFSAHSAclaimsReportExport.xls
CANDIDATE_PATHS = [
    Path("notes/tax/intake/hsa_fsa/2025/Optum/optumFSAHSAclaimsReportExport.xls"),
    Path("notes/tax/intake/hsa_fsa/2025/optumFSAHSAclaimsReportExport.xls"),
    Path("notes/tax/intake/hsa_fsa/2025/Optum"),
    Path("notes/tax/intake/hsa_fsa/2025"),
]

def _log(msg: str) -> None:
    print(msg)

def _try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None

def _coerce_date_iso(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if not s or s.lower() in {"nan", "nat"}:
        return ""
    # if "YYYY-MM-DD HH:MM:SS" -> date
    if len(s) >= 10 and s[4:5] == "-" and " " in s:
        return s.split(" ", 1)[0]
    return s

def _find_files() -> List[Path]:
    files: List[Path] = []
    for p in CANDIDATE_PATHS:
        if p.is_file() and p.suffix.lower() in {".xls", ".xlsx"}:
            files.append(p)
        elif p.is_dir():
            files.extend(sorted([x for x in p.glob("*.xls")]))
            files.extend(sorted([x for x in p.glob("*.xlsx")]))
    # de-dupe
    uniq = []
    seen = set()
    for f in files:
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq

def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _colpick_exact(df_cols: List[str], *names: str) -> Optional[str]:
    lower = {c.lower(): c for c in df_cols}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def _colpick_contains(df_cols: List[str], *needles: str) -> Optional[str]:
    for c in df_cols:
        lc = c.lower()
        if all(n.lower() in lc for n in needles):
            return c
    return None

def _normalize_one_file(pd, src: Path) -> Tuple[Path, int]:
    xls = pd.ExcelFile(src)
    all_rows: List[Dict] = []

    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        df.columns = [str(c).strip() for c in df.columns]

        cols = df.columns.tolist()

        # We’ll be flexible: Optum exports vary.
        c_date = (
            _colpick_exact(cols, "Date", "Transaction Date", "Service Date", "Paid Date")
            or _colpick_contains(cols, "date")
        )
        c_amount = (
            _colpick_exact(cols, "Amount", "Transaction Amount", "Paid Amount", "Claim Amount")
            or _colpick_contains(cols, "amount")
        )
        c_merchant = (
            _colpick_exact(cols, "Merchant", "Provider", "Payee", "Service Provider", "Facility")
            or _colpick_contains(cols, "merchant")
            or _colpick_contains(cols, "provider")
            or _colpick_contains(cols, "payee")
        )
        c_desc = _colpick_exact(cols, "Description", "Transaction Description") or _colpick_contains(cols, "description")
        c_status = _colpick_exact(cols, "Status", "Claim Status") or _colpick_contains(cols, "status")
        c_member = _colpick_exact(cols, "Member", "Patient", "Participant") or _colpick_contains(cols, "member") or _colpick_contains(cols, "patient")
        c_claim = _colpick_exact(cols, "Claim", "Claim ID", "Claim Number") or _colpick_contains(cols, "claim")

        for i, row in df.iterrows():
            r = {
                "source": "Optum",
                "source_file": src.name,
                "source_sheet": sheet,
                "row_index": int(i),

                "date": _coerce_date_iso(row.get(c_date)) if c_date else "",
                "amount": row.get(c_amount, "") if c_amount else "",
                "merchant": str(row.get(c_merchant, "")).strip() if c_merchant else "",
                "description": str(row.get(c_desc, "")).strip() if c_desc else "",
                "status": str(row.get(c_status, "")).strip() if c_status else "",
                "member": str(row.get(c_member, "")).strip() if c_member else "",
                "claim_id": str(row.get(c_claim, "")).strip() if c_claim else "",
            }

            # Skip totally blank rows
            if not r["date"] and not r["merchant"] and not r["description"]:
                continue

            all_rows.append(r)

    out = OUT_DIR / "optum_fsa_hsa_claims.normalized.csv"
    _write_csv(
        out,
        ["source","source_file","source_sheet","row_index","date","amount","merchant","description","status","member","claim_id"],
        all_rows,
    )
    return out, len(all_rows)

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pd = _try_import_pandas()
    if pd is None:
        _log("Missing dependency: pandas")
        _log("Fix: ./.venv_tax/Scripts/python -m pip install pandas openpyxl xlrd==2.0.1")
        return 2

    files = _find_files()
    if not files:
        _log("No Optum export files found under:")
        for p in CANDIDATE_PATHS:
            _log(f"  - {p}")
        return 2

    # If multiple exports exist, we’ll process the newest (or you can extend later).
    # For now: pick the largest file as a simple heuristic.
    files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    src = files[0]
    _log(f"Using Optum export: {src}")

    out, n = _normalize_one_file(pd, src)
    _log("\nwrote:")
    _log(f"  {out} rows: {n}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
