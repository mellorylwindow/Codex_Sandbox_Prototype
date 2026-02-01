#!/usr/bin/env python3
"""
Medical prep: inventory medical docs + extract Optum HSA/FSA payments into CSV/XLSX.

Run:
  python tools/med_prep.py --root tax_intake --tax-year 2025
Outputs:
  <root>/40_reports/medical_docs_<year>.csv
  <root>/40_reports/medical_payments_optum_<year>.csv
  <root>/40_reports/medical_master_<year>.xlsx
  <root>/index/medical_docs_<year>.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd

DATE_RE = re.compile(r"(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})")  # 20260103
DATE_DASH_RE = re.compile(r"(?P<y>20\d{2})-(?P<m>\d{2})-(?P<d>\d{2})")  # 2025-01-21


@dataclass
class DocRow:
    sha256: str
    relpath: str
    ext: str
    size_bytes: int
    mtime_iso: str
    provider_bucket: str
    doc_type: str
    capture_date_guess: Optional[str]
    tax_year: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def guess_date_from_name(name: str) -> Optional[str]:
    m = DATE_RE.search(name)
    if m:
        y, mo, d = m.group("y"), m.group("m"), m.group("d")
        return f"{y}-{mo}-{d}"
    m2 = DATE_DASH_RE.search(name)
    if m2:
        y, mo, d = m2.group("y"), m2.group("m"), m2.group("d")
        return f"{y}-{mo}-{d}"
    return None


def infer_doc_type(relpath: str) -> str:
    p = relpath.replace("\\", "/").lower()

    if "/10_sources/insurance/" in p and "/eob_images/" in p:
        return "EOB_IMAGE"

    if "/10_sources/hsa_fsa/" in p and "/optum/" in p and p.endswith((".xls", ".xlsx")):
        return "HSA_EXPORT"

    if "/10_sources/providers/" in p and p.endswith((".jpg", ".jpeg", ".png", ".pdf")):
        return "PROVIDER_DOC"

    # fallbacks
    if "eob" in p:
        return "EOB"
    return "UNKNOWN"


def infer_provider_bucket(relpath: str) -> str:
    parts = Path(relpath).parts
    lowered = [x.lower() for x in parts]

    # providers/<bucket>/...
    for i, part in enumerate(lowered):
        if part == "providers" and i + 1 < len(parts):
            return parts[i + 1]

    # insurance/bcbsnd
    if "bcbsnd" in lowered:
        return "BCBSND"

    # optum exports
    if "optum" in lowered:
        return "Optum"

    return "UNKNOWN"


def walk_docs(base: Path) -> List[Path]:
    exts = {".pdf", ".jpg", ".jpeg", ".png", ".xls", ".xlsx"}
    paths: List[Path] = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return paths


def load_optum_exports(optum_exports_dir: Path) -> pd.DataFrame:
    files = sorted([p for p in optum_exports_dir.glob("*.xls*") if p.is_file()])
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_excel(f)
            df["source_file"] = f.name
            frames.append(df)
        except Exception as e:
            frames.append(pd.DataFrame([{"source_file": f.name, "error": str(e)}]))

    out = pd.concat(frames, ignore_index=True)

    # Normalize common columns if found (keeps original columns too)
    rename_map = {}
    for c in out.columns:
        lc = str(c).strip().lower()
        if lc in {"date", "service date", "servicedate", "transaction date"}:
            rename_map[c] = "service_date"
        elif lc in {"amount", "paid amount", "payment amount", "total"}:
            rename_map[c] = "amount"
        elif lc in {"provider", "merchant", "merchant name", "payee"}:
            rename_map[c] = "provider_or_merchant"
        elif lc in {"description", "service description", "details"}:
            rename_map[c] = "description"
        elif lc in {"category", "expense category", "type"}:
            rename_map[c] = "category"
    if rename_map:
        out = out.rename(columns=rename_map)

    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tax_intake", help="Root tax intake folder")
    ap.add_argument("--tax-year", default="2025", help="Tax year label (string)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    year_dir = root / args.tax_year

    if not year_dir.exists():
        raise SystemExit(f"Year dir not found: {year_dir}")

    reports_dir = root / "40_reports"
    index_dir = root / "index"
    ensure_dir(reports_dir)
    ensure_dir(index_dir)

    # Only scan 2025/10_sources (medical sources), not the whole repo
    sources_dir = year_dir / "10_sources"
    if not sources_dir.exists():
        raise SystemExit(f"Sources dir not found: {sources_dir}")

    docs = walk_docs(sources_dir)
    rows: List[DocRow] = []

    for p in docs:
        relpath = str(p.relative_to(root))  # keep stable relative paths from tax_intake/
        st = p.stat()
        rows.append(
            DocRow(
                sha256=sha256_file(p),
                relpath=relpath,
                ext=p.suffix.lower(),
                size_bytes=st.st_size,
                mtime_iso=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                provider_bucket=infer_provider_bucket(relpath),
                doc_type=infer_doc_type(relpath),
                capture_date_guess=guess_date_from_name(p.name),
                tax_year=args.tax_year,
            )
        )

    docs_df = pd.DataFrame([asdict(r) for r in rows]).sort_values(
        ["doc_type", "provider_bucket", "capture_date_guess", "relpath"]
    )

    # JSONL manifest
    manifest_path = index_dir / f"medical_docs_{args.tax_year}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    docs_csv = reports_dir / f"medical_docs_{args.tax_year}.csv"
    docs_df.to_csv(docs_csv, index=False)

    # Optum exports → payments
    optum_exports_dir = year_dir / "10_sources" / "hsa_fsa" / "optum" / "exports"
    optum_df = load_optum_exports(optum_exports_dir) if optum_exports_dir.exists() else pd.DataFrame()
    optum_csv = reports_dir / f"medical_payments_optum_{args.tax_year}.csv"
    optum_df.to_csv(optum_csv, index=False)

    # Master workbook
    xlsx_path = reports_dir / f"medical_master_{args.tax_year}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        docs_df.to_excel(xw, index=False, sheet_name="Docs")
        optum_df.to_excel(xw, index=False, sheet_name="OptumPayments")

        link_cols = [
            "claim_id",
            "patient",
            "provider_bucket",
            "service_date",
            "eob_sha256",
            "provider_doc_sha256",
            "optum_payment_row_ref",
            "status",
            "notes",
        ]
        pd.DataFrame(columns=link_cols).to_excel(xw, index=False, sheet_name="ClaimLink")

    print("OK Medical docs manifest:", manifest_path)
    print("OK Docs CSV:", docs_csv)
    print("OK Optum payments CSV:", optum_csv)
    print("OK Medical master workbook:", xlsx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
