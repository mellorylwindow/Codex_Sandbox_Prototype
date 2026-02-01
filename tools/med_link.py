#!/usr/bin/env python3
"""
tools/med_link.py

Phase 2: Link Optum payments (FSA/HSA export) to your local medical docs by:
- Vendor/Provider -> provider_bucket matching (rules + aliases + fuzzy)
- Date proximity (Date of Service vs doc capture_date_guess)
- Doc type bonus (e.g., provider docs / receipts slightly preferred)

Outputs:
- <root>/40_reports/medical_link_candidates_<tax_year>.csv
- Sheet "LinkCandidates" (replaced) inside <root>/40_reports/medical_master_<tax_year>.xlsx

Notes:
- capture_date_guess is based on filename metadata (often scan/import date). That can be far from DOS.
  This script avoids over-penalizing date when the vendor->bucket match is very strong.

Typical run:
  PYTHONUTF8=1 python tools/med_link.py --root tax_intake --tax-year 2025 --max-days 365 --top-n 8
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from difflib import SequenceMatcher


# -----------------------------
# Normalization helpers
# -----------------------------
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def norm(s: object) -> str:
    """Lower, keep alnum, collapse whitespace."""
    if s is None:
        return ""
    s2 = str(s).lower().strip()
    if not s2 or s2.lower() == "nan":
        return ""
    s2 = _NON_ALNUM_RE.sub(" ", s2)
    s2 = _WS_RE.sub(" ", s2).strip()
    return s2


def sim(a: object, b: object) -> float:
    """SequenceMatcher similarity on normalized strings."""
    a2, b2 = norm(a), norm(b)
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()


def parse_mmddyyyy(s: object) -> Optional[date]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    s2 = str(s).strip()
    if not s2 or s2.lower() == "nan":
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s2, fmt).date()
        except ValueError:
            continue
    return None


def parse_yyyy_mm_dd(s: object) -> Optional[date]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    s2 = str(s).strip()
    if not s2 or s2.lower() == "nan":
        return None
    try:
        return datetime.strptime(s2, "%Y-%m-%d").date()
    except ValueError:
        return None


def days_diff(a: Optional[date], b: Optional[date]) -> Optional[int]:
    if not a or not b:
        return None
    return abs((a - b).days)


def is_blank(x: object) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


# -----------------------------
# Provider bucket matching rules
# -----------------------------
@dataclass(frozen=True)
class BucketRule:
    """
    A bucket can match a vendor/provider via:
    - contains_any: token containment in vendor string (strong match)
    - aliases: extra strings to fuzz-match against (weak/medium match)
    """
    contains_any: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()


# Edit freely over time. Keys MUST match your provider_bucket folder names.
# Keep tokens short and normalized-ish (no punctuation); they are tested against norm(vendor_provider).
BUCKET_RULES: dict[str, BucketRule] = {
    # --- Dental
    "dentist_dr_le": BucketRule(
        contains_any=("trang t le", "dds", "vienna"),
        aliases=("trang t le dds ltd", "trang t le", "dr le dds", "vienna dental"),
    ),

    # --- Radiology
    "community_radiology_associates": BucketRule(
        contains_any=("community radiology", "radiology assoc", "radiology associates"),
        aliases=("community radiology associates", "community radiology", "cra radiology"),
    ),

    # --- Labs
    "labcorp": BucketRule(
        contains_any=("labcorp", "laboratory corporation of america", "laboratory corporation of ameri"),
        aliases=("labcorp norcross", "laboratory corporation of america"),
    ),
    "cbl_path": BucketRule(
        contains_any=("cbl path", "cblpath", "cbl path inc"),
        aliases=("cbl path", "cbl path inc"),
    ),

    # --- Primary / urgent care
    "healthworks": BucketRule(
        contains_any=("healthworks",),
        aliases=("healthworks for northe", "healthworks", "health works"),
    ),
    "receivables_management_aka_patient_first": BucketRule(
        contains_any=("patient first", "receivables management"),
        aliases=("patient first", "med patient first", "receivables management"),
    ),

    # --- Inova / hospital systems
    "inova": BucketRule(
        contains_any=("inova", "inova phys", "inova physician", "inova health"),
        aliases=("inova phys ptnrs", "inova phys ptnrs-mycha", "inova physician partners"),
    ),
    "privia": BucketRule(
        contains_any=("privia",),
        aliases=("privia", "privia medical group"),
    ),

    # --- Psychiatry
    "psychiatrist_dr_blanchfield": BucketRule(
        contains_any=("blanchfield", "colleen a blanchfield"),
        aliases=("colleen blanchfield", "colleen a blanchfield", "dr blanchfield"),
    ),

    # --- Pharmacy / retailers commonly used for eligible items
    "pharmacy": BucketRule(
        contains_any=("cvs", "walgreens", "rite aid", "giant pharmacy", "costco pharmacy", "walmart pharmacy", "target"),
        aliases=("genoa qol", "genoa", "cvs", "walgreens", "rite aid", "target t", "target", "pharmacy"),
    ),
}


# Doc type bonuses: tweak as you like (small nudges only).
DOC_TYPE_BONUS: dict[str, float] = {
    "PROVIDER_DOC": 0.06,
    "RECEIPT_IMAGE": 0.06,
    "EOB_DOC": 0.02,
}


# -----------------------------
# Optum column detection
# -----------------------------
def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    want = {norm(c) for c in candidates}
    for c in df.columns:
        if norm(c) in want:
            return c
    return None


# -----------------------------
# Bucket scoring
# -----------------------------
def bucket_match(vendor_provider: str, bucket: str) -> Tuple[float, str]:
    """
    Return (bucket_score, why_bucket).

    bucket_score is in [0, 0.99] where 0.99 is a strong "contains token" match.
    """
    v = norm(vendor_provider)
    if not v:
        return 0.0, "no_vendor"

    rule = BUCKET_RULES.get(bucket)

    # Strong: containment rules
    if rule and rule.contains_any:
        for tok in rule.contains_any:
            t = norm(tok)
            if t and t in v:
                return 0.99, f"contains:{t}"

    # Medium/weak: fuzzy across bucket name + aliases
    best = sim(v, bucket)
    why = f"fuzzy:bucket={bucket}"

    if rule and rule.aliases:
        for a in rule.aliases:
            sc = sim(v, a)
            if sc > best:
                best = sc
                why = f"fuzzy:alias={a}"

    # Clamp to 0.98 for fuzzy so it never beats "contains" purely by chance.
    best = float(min(best, 0.98))
    return best, why


# -----------------------------
# Scoring candidates
# -----------------------------
def date_score(ddays: Optional[int], max_days: int) -> float:
    """
    1.0 when ddays == 0, decreasing linearly to 0 at max_days.
    If ddays is unknown, return a neutral 0.50.
    """
    if ddays is None:
        return 0.50
    if ddays <= 0:
        return 1.0
    if ddays >= max_days:
        return 0.0
    return max(0.0, (max_days - float(ddays)) / float(max_days))


def final_score(
    bucket_sc: float,
    dsc: float,
    dtype_bonus: float,
    bucket_strong_ignore_date: float,
) -> Tuple[float, str]:
    """
    Blend score in a way that:
    - rewards strong bucket match heavily
    - date helps, but doesn't kill a strong match when docs were scanned months later
    """
    # If the bucket match is very strong, treat date as "nice to have"
    if bucket_sc >= bucket_strong_ignore_date:
        # 90% bucket, 10% date, plus bonus
        sc = 0.90 * bucket_sc + 0.10 * dsc + dtype_bonus
        return float(min(sc, 0.9999)), "blend=strong_bucket(0.90/0.10)+bonus"
    # Otherwise, weight date a bit more
    sc = 0.80 * bucket_sc + 0.20 * dsc + dtype_bonus
    return float(min(sc, 0.9999)), "blend=normal(0.80/0.20)+bonus"


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tax_intake")
    ap.add_argument("--tax-year", default="2025")
    ap.add_argument("--top-n", type=int, default=8, help="Top N doc candidates per Optum row")
    ap.add_argument("--max-days", type=int, default=365, help="Max days between DOS and doc capture date")
    ap.add_argument("--min-bucket-score", type=float, default=0.20, help="Ignore buckets below this match score")
    ap.add_argument("--max-buckets", type=int, default=4, help="Only consider top K buckets per Optum row")
    ap.add_argument("--include-voided", action="store_true", help="Include rows where status is Voided")
    ap.add_argument("--include-unpaid", action="store_true", help="Include rows where paid_amount is blank/NaN")
    ap.add_argument("--bucket-strong-ignore-date", type=float, default=0.90, help="If bucket_score >= this, date has minimal impact")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    reports = root / "40_reports"

    docs_csv = reports / f"medical_docs_{args.tax_year}.csv"
    optum_csv = reports / f"medical_payments_optum_{args.tax_year}.csv"
    master_xlsx = reports / f"medical_master_{args.tax_year}.xlsx"

    if not docs_csv.exists():
        raise SystemExit(f"Missing: {docs_csv}")
    if not optum_csv.exists():
        raise SystemExit(f"Missing: {optum_csv}")
    if not master_xlsx.exists():
        raise SystemExit(f"Missing: {master_xlsx}")

    docs = pd.read_csv(docs_csv)
    opt = pd.read_csv(optum_csv)

    # ---- validate docs columns
    need_docs_cols = ["sha256", "relpath", "provider_bucket", "doc_type", "capture_date_guess"]
    missing_docs = [c for c in need_docs_cols if c not in docs.columns]
    if missing_docs:
        raise SystemExit(f"Docs CSV missing columns: {missing_docs}\nCols={list(docs.columns)}")

    # Keep only provider docs under 10_sources/providers
    provider_mask = docs["relpath"].astype(str).str.contains(r"[\\/]+10_sources[\\/]+providers[\\/]+", regex=True, na=False)
    provider_docs = docs[provider_mask].copy()
    if provider_docs.empty:
        raise SystemExit("No docs found under 10_sources/providers (relpath filter).")

    provider_docs["doc_date"] = provider_docs["capture_date_guess"].apply(parse_yyyy_mm_dd)
    provider_docs["doc_type"] = provider_docs["doc_type"].astype(str)

    # Group docs by bucket for faster lookup
    docs_by_bucket: dict[str, pd.DataFrame] = {}
    for bucket, g in provider_docs.groupby("provider_bucket", dropna=False):
        b = str(bucket) if not is_blank(bucket) else ""
        docs_by_bucket[b] = g.copy()

    # ---- detect Optum columns (your export uses "My Amount Paid")
    c_claim_id = find_col(opt, ["Claim ID", "ClaimID"])
    c_service_for = find_col(opt, ["Service For", "ServiceFor"])
    c_vendor = find_col(opt, ["Vendor / Provider", "Vendor/Provider", "Vendor Provider", "Provider", "Vendor"])
    c_dos = find_col(opt, ["Date of Service", "Service Date"])
    c_dop = find_col(opt, ["Date of Payment", "Payment Date"])
    c_claim_amt = find_col(opt, ["Claim Amount", "Amount"])
    c_paid = find_col(opt, ["My Amount Paid", "My Amount Paid From Account", "Amount Paid", "Paid Amount"])
    c_pay_to = find_col(opt, ["Pay To", "PayTo"])
    c_status = find_col(opt, ["Status"])
    c_source_file = find_col(opt, ["source_file", "Source File", "Source"])

    required = [("Claim ID", c_claim_id), ("Vendor / Provider", c_vendor), ("Date of Service", c_dos)]
    missing = [name for name, col in required if col is None]
    if missing:
        raise SystemExit(f"Optum CSV missing required columns: {missing}\nCols={list(opt.columns)}")

    # Normalize optum fields into internal columns
    opt2 = opt.copy()
    opt2["_optum_row"] = opt2.index.astype(int)
    opt2["_claim_id"] = opt2[c_claim_id].astype(str)
    opt2["_service_for"] = opt2[c_service_for].astype(str) if c_service_for else ""
    opt2["_vendor"] = opt2[c_vendor].astype(str)
    opt2["_dos"] = opt2[c_dos].apply(parse_mmddyyyy)
    opt2["_dop"] = opt2[c_dop].apply(parse_mmddyyyy) if c_dop else None
    opt2["_claim_amt"] = opt2[c_claim_amt] if c_claim_amt else ""
    opt2["_paid_amt"] = opt2[c_paid] if c_paid else ""
    opt2["_pay_to"] = opt2[c_pay_to].astype(str) if c_pay_to else ""
    opt2["_status"] = opt2[c_status].astype(str) if c_status else ""
    opt2["_source_file"] = opt2[c_source_file].astype(str) if c_source_file else ""

    # Filter rows
    rows_written = 0
    out_rows: List[dict] = []

    for _, r in opt2.iterrows():
        status = str(r["_status"]).strip()
        paid_amt = r["_paid_amt"]

        if (not args.include_voided) and status.lower() == "voided":
            continue
        if (not args.include_unpaid) and is_blank(paid_amt):
            continue

        vendor = str(r["_vendor"])
        dos = r["_dos"]

        # Score buckets
        bucket_scores: List[Tuple[str, float, str]] = []
        for bucket in docs_by_bucket.keys():
            if not bucket:
                continue
            bsc, bwhy = bucket_match(vendor, bucket)
            if bsc >= args.min_bucket_score:
                bucket_scores.append((bucket, bsc, bwhy))

        if not bucket_scores:
            # Nothing matched; still emit one tracking row
            out_rows.append({
                "optum_row": int(r["_optum_row"]),
                "claim_id": r["_claim_id"],
                "service_for": r["_service_for"],
                "vendor_provider": vendor,
                "date_of_service": dos.isoformat() if isinstance(dos, date) else "",
                "date_of_payment": r["_dop"].isoformat() if isinstance(r["_dop"], date) else "",
                "claim_amount": r["_claim_amt"],
                "paid_amount": r["_paid_amt"],
                "pay_to": r["_pay_to"],
                "status": status,
                "source_file": r["_source_file"],
                "doc_sha256": "",
                "doc_relpath": "",
                "doc_provider_bucket": "",
                "doc_capture_date": "",
                "days_from_service": "",
                "provider_similarity": "",
                "score": "",
                "why": "no_bucket_match",
            })
            rows_written += 1
            continue

        # Only consider top K buckets
        bucket_scores.sort(key=lambda t: t[1], reverse=True)
        bucket_scores = bucket_scores[: max(1, int(args.max_buckets))]

        # Build doc candidates across those buckets
        cand_rows: List[Tuple[float, dict]] = []

        for bucket, bsc, bwhy in bucket_scores:
            g = docs_by_bucket.get(bucket)
            if g is None or g.empty:
                continue

            dtype_bonus_default = 0.0

            for _, d in g.iterrows():
                d_doc_date: Optional[date] = d.get("doc_date", None)
                dd = days_diff(dos, d_doc_date) if isinstance(dos, date) else None
                dsc = date_score(dd, args.max_days)

                dtype = str(d.get("doc_type", "")).strip()
                dtype_bonus = float(DOC_TYPE_BONUS.get(dtype, dtype_bonus_default))

                sc, blend_why = final_score(
                    bucket_sc=float(bsc),
                    dsc=float(dsc),
                    dtype_bonus=dtype_bonus,
                    bucket_strong_ignore_date=float(args.bucket_strong_ignore_date),
                )

                why_parts = [
                    f"bucket={bucket}",
                    bwhy,
                    f"bucket_score={float(bsc):.3f}",
                    f"ddays={dd if dd is not None else 'NA'}",
                    f"date_score={float(dsc):.3f}",
                    f"dtype={dtype}",
                    f"dtype_bonus={dtype_bonus:+.2f}",
                    blend_why,
                ]
                why = " ".join(why_parts)

                row = {
                    "optum_row": int(r["_optum_row"]),
                    "claim_id": r["_claim_id"],
                    "service_for": r["_service_for"],
                    "vendor_provider": vendor,
                    "date_of_service": dos.isoformat() if isinstance(dos, date) else "",
                    "date_of_payment": r["_dop"].isoformat() if isinstance(r["_dop"], date) else "",
                    "claim_amount": r["_claim_amt"],
                    "paid_amount": r["_paid_amt"],
                    "pay_to": r["_pay_to"],
                    "status": status,
                    "source_file": r["_source_file"],
                    "doc_sha256": d.get("sha256", ""),
                    "doc_relpath": d.get("relpath", ""),
                    "doc_provider_bucket": bucket,
                    "doc_capture_date": d_doc_date.isoformat() if isinstance(d_doc_date, date) else "",
                    "days_from_service": dd if dd is not None else "",
                    "provider_similarity": round(float(bsc), 4),
                    "score": round(float(sc), 4),
                    "why": why,
                }
                cand_rows.append((float(sc), row))

        if not cand_rows:
            out_rows.append({
                "optum_row": int(r["_optum_row"]),
                "claim_id": r["_claim_id"],
                "service_for": r["_service_for"],
                "vendor_provider": vendor,
                "date_of_service": dos.isoformat() if isinstance(dos, date) else "",
                "date_of_payment": r["_dop"].isoformat() if isinstance(r["_dop"], date) else "",
                "claim_amount": r["_claim_amt"],
                "paid_amount": r["_paid_amt"],
                "pay_to": r["_pay_to"],
                "status": status,
                "source_file": r["_source_file"],
                "doc_sha256": "",
                "doc_relpath": "",
                "doc_provider_bucket": "",
                "doc_capture_date": "",
                "days_from_service": "",
                "provider_similarity": "",
                "score": "",
                "why": "no_doc_candidates",
            })
            rows_written += 1
            continue

        # Sort candidates by score desc, then prefer closer dates when score ties
        def dd_sort_key(row: dict) -> int:
            v = row.get("days_from_service", "")
            try:
                return int(v)
            except Exception:
                return 10**9

        cand_rows.sort(key=lambda t: (t[0], -dd_sort_key(t[1])), reverse=True)
        top = cand_rows[: max(1, int(args.top_n))]
        for _, row in top:
            out_rows.append(row)
            rows_written += 1

    out = pd.DataFrame(out_rows)

    # Stable column order
    cols = [
        "optum_row",
        "claim_id",
        "service_for",
        "vendor_provider",
        "date_of_service",
        "date_of_payment",
        "claim_amount",
        "paid_amount",
        "pay_to",
        "status",
        "source_file",
        "doc_sha256",
        "doc_relpath",
        "doc_provider_bucket",
        "doc_capture_date",
        "days_from_service",
        "provider_similarity",
        "score",
        "why",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out = out[cols]

    out_csv = reports / f"medical_link_candidates_{args.tax_year}.csv"
    out.to_csv(out_csv, index=False)

    with pd.ExcelWriter(master_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
        out.to_excel(xw, index=False, sheet_name="LinkCandidates")

    print("OK Link candidates CSV:", out_csv)
    print("OK Updated workbook sheet: LinkCandidates ->", master_xlsx)
    print(
        f"OK rows_written={rows_written} "
        f"include_voided={bool(args.include_voided)} include_unpaid={bool(args.include_unpaid)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
