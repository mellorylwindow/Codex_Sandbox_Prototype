#!/usr/bin/env python3
"""
Phase 3: Apply best LinkCandidates into ClaimLink sheet in medical_master_<year>.xlsx

- Reads: 40_reports/medical_link_candidates_<year>.csv
- Writes: updates 40_reports/medical_master_<year>.xlsx (ClaimLink sheet replaced)

Logic:
- Choose best candidate per claim_id by score desc, provider_similarity desc, days_from_service asc
- Skip rows where status is 'Voided' OR paid_amount is blank/NaN
- Set status:
    - AUTO_LINKED_CONFIDENT if score >= threshold
    - AUTO_LINKED_NEEDS_REVIEW otherwise
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


def canon(col: str) -> str:
    c = str(col).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def is_blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tax_intake")
    ap.add_argument("--tax-year", default="2025")
    ap.add_argument("--threshold", type=float, default=0.62)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    reports = root / "40_reports"
    cand_csv = reports / f"medical_link_candidates_{args.tax_year}.csv"
    master_xlsx = reports / f"medical_master_{args.tax_year}.xlsx"

    cand = pd.read_csv(cand_csv)
    cand.columns = [canon(c) for c in cand.columns]

    # expected canonical column names after canon()
    required = [
        "claim_id",
        "optum_row",
        "date_of_service",
        "vendor_provider",
        "status",
        "paid_amount",
        "doc_sha256",
        "doc_provider_bucket",
        "score",
        "provider_similarity",
        "days_from_service",
    ]
    missing = [c for c in required if c not in cand.columns]
    if missing:
        raise SystemExit(f"Candidate CSV missing columns after normalization: {missing}\nCols={list(cand.columns)}")

    # Remove voided + unpaid
    cand = cand[~cand["status"].astype(str).str.lower().eq("voided")].copy()
    cand = cand[~cand["paid_amount"].apply(is_blank)].copy()

    # Ensure numeric sorts
    cand["score_num"] = pd.to_numeric(cand["score"], errors="coerce")
    cand["sim_num"] = pd.to_numeric(cand["provider_similarity"], errors="coerce")
    cand["days_num"] = pd.to_numeric(cand["days_from_service"], errors="coerce")

    # Only rows that actually have a candidate doc
    cand = cand[~cand["doc_sha256"].apply(is_blank)].copy()

    cand = cand.sort_values(
        ["claim_id", "score_num", "sim_num", "days_num"],
        ascending=[True, False, False, True],
        na_position="last",
    )
    best = cand.groupby("claim_id", as_index=False).first()

    # service_for column if present
    service_for = best["service_for"] if "service_for" in best.columns else ""

    claimlink = pd.DataFrame({
        "claim_id": best["claim_id"],
        "patient": service_for,
        "provider_bucket": best["doc_provider_bucket"],
        "service_date": best["date_of_service"],
        "eob_sha256": "",
        "provider_doc_sha256": best["doc_sha256"],
        "optum_payment_row_ref": best["optum_row"],
        "status": best["score_num"].apply(lambda s: "AUTO_LINKED_CONFIDENT" if (not pd.isna(s) and s >= args.threshold) else "AUTO_LINKED_NEEDS_REVIEW"),
        "notes": best.apply(lambda r: f"vendor={r['vendor_provider']} score={r['score']} sim={r['provider_similarity']} days={r['days_from_service']}", axis=1),
    })

    with pd.ExcelWriter(master_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
        claimlink.to_excel(xw, index=False, sheet_name="ClaimLink")

    confident = int((claimlink["status"] == "AUTO_LINKED_CONFIDENT").sum())
    needs = int((claimlink["status"] == "AUTO_LINKED_NEEDS_REVIEW").sum())
    print("OK ClaimLink updated:", master_xlsx)
    print(f"OK rows={len(claimlink)} confident={confident} needs_review={needs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
