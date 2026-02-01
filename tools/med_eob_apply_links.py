#!/usr/bin/env python3
"""
Phase: EOB linking

Reads:
- <root>/40_reports/medical_master_<year>.xlsx (sheet: ClaimLink)
- <root>/40_reports/medical_eob_docs_<year>.csv (from med_eob_prep.py)

Writes:
- <root>/40_reports/medical_eob_link_candidates_<year>.csv
- Updates workbook:
    - sheet "EOBLinkCandidates" (replaced)
    - sheet "ClaimLink" (replaced) with EOB link columns populated

IMPORTANT:
- This script DOES NOT assume Optum Claim ID == Insurance Claim Number.
  It primarily matches by provider + date proximity + (optional) amount proximity + patient name.
- It ALWAYS writes the output CSV with headers, even if no candidates exist.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any

import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_blank(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


def sim(a: Any, b: Any) -> float:
    a2, b2 = norm(a), norm(b)
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()


def parse_iso_date(s: Any) -> date | None:
    if is_blank(s):
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def to_money(x: Any) -> float | None:
    """
    Converts "$1,052.88" or "1052.88" to float.
    """
    if is_blank(x):
        return None
    s = str(x).strip()
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def abs_days(a: date | None, b: date | None) -> int | None:
    if not a or not b:
        return None
    return abs((a - b).days)


def date_score(ddays: int | None, max_days: int) -> float:
    if ddays is None:
        return 0.0
    if ddays <= 0:
        return 1.0
    if ddays >= max_days:
        return 0.0
    return 1.0 - (ddays / float(max_days))


# ----------------------------
# Provider bucket hinting
# ----------------------------

BUCKET_ALIASES: dict[str, list[str]] = {
    # bucket_name: list of substrings / aliases that should match vendor/provider strings
    "dentist_dr_le": ["trang t le", "dds", "vienna"],
    "community_radiology_associates": ["community radiology", "radiology assoc", "radiology associates"],
    "healthworks": ["healthworks"],
    "labcorp": ["labcorp", "laboratory corporation"],
    "inova": ["inova"],
    "privia": ["privia"],
    "psychiatrist_dr_blanchfield": ["blanchfield", "psychiatrist", "psychiatry", "colleen", "blanch"],
    "receivables_management_aka_patient_first": ["patient first", "receivables management"],
    "patient_first": ["patient first"],
    "pharmacy": ["cvs", "walgreens", "rite aid", "giant pharmacy", "pharmacy", "genoa"],
    "genoa_qol": ["genoa", "qol"],
}

def provider_bucket_match_score(bucket: str, provider_name: str) -> tuple[float, str]:
    """
    Returns (score, reason).
    Uses:
      - direct similarity bucket vs provider_name (weak)
      - alias substring hits (strong)
    """
    b = norm(bucket)
    p = norm(provider_name)

    if not p:
        return (0.0, "no_provider_name")

    # Strong: alias contains
    for alias in BUCKET_ALIASES.get(bucket, []):
        a = norm(alias)
        if a and a in p:
            # strong signal
            return (0.99, f"bucket={bucket} contains:{a}")

    # Medium: similarity
    s = sim(bucket, provider_name)
    return (s, f"bucket={bucket} sim={s:.4f}")


# ----------------------------
# Candidate scoring
# ----------------------------

@dataclass
class Weights:
    provider: float = 0.50
    date: float = 0.35
    amount: float = 0.15


def compute_amount_score(
    eob_amounts: list[float],
    target_amount: float | None,
    tol: float,
) -> tuple[float, str]:
    """
    Score 1.0 if any eob_amount is within tol of target_amount,
    else 0.0. If target missing or no eob amounts, neutral 0.0.
    """
    if target_amount is None:
        return (0.0, "no_target_amount")
    if not eob_amounts:
        return (0.0, "no_eob_amounts")

    best_delta = None
    for a in eob_amounts:
        d = abs(a - target_amount)
        if best_delta is None or d < best_delta:
            best_delta = d

    if best_delta is None:
        return (0.0, "no_amount_delta")
    if best_delta <= tol:
        return (1.0, f"amount_match delta={best_delta:.2f} tol={tol:.2f}")
    return (0.0, f"amount_no_match delta={best_delta:.2f} tol={tol:.2f}")


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tax_intake")
    ap.add_argument("--tax-year", default="2025")
    ap.add_argument("--max-days", type=int, default=45, help="Max days between claim service_date and EOB service date (min/max).")
    ap.add_argument("--amount-tol", type=float, default=3.0, help="Dollar tolerance for amount matching.")
    ap.add_argument("--threshold", type=float, default=0.72, help="Auto-link threshold.")
    ap.add_argument("--top-n", type=int, default=8, help="Candidates per claim.")
    ap.add_argument("--min-score", type=float, default=0.0, help="Filter out candidates below this score.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    reports = root / "40_reports"

    master_xlsx = reports / f"medical_master_{args.tax_year}.xlsx"
    eob_docs_csv = reports / f"medical_eob_docs_{args.tax_year}.csv"
    out_csv = reports / f"medical_eob_link_candidates_{args.tax_year}.csv"

    if not master_xlsx.exists():
        raise SystemExit(f"Missing: {master_xlsx}")
    if not eob_docs_csv.exists():
        raise SystemExit(f"Missing: {eob_docs_csv}")

    claimlink = pd.read_excel(master_xlsx, sheet_name="ClaimLink")
    eob = pd.read_csv(eob_docs_csv)

    # Normalize / parse dates
    if "service_date" in claimlink.columns:
        claimlink["service_date_parsed"] = claimlink["service_date"].apply(parse_iso_date)
    elif "service_date" not in claimlink.columns and "service_date" in claimlink.columns:
        claimlink["service_date_parsed"] = claimlink["service_date"].apply(parse_iso_date)
    else:
        # Your ClaimLink has service_date already
        claimlink["service_date_parsed"] = claimlink["service_date"].apply(parse_iso_date)

    for col in ["service_date_min", "service_date_max"]:
        if col in eob.columns:
            eob[col + "_parsed"] = eob[col].apply(parse_iso_date)
        else:
            eob[col + "_parsed"] = None

    # Ensure key columns exist in EOB docs
    for need in ["sha256", "relpath", "filename"]:
        if need not in eob.columns:
            raise SystemExit(f"EOB docs CSV missing column: {need}")

    # Optional columns in EOB docs (may be empty)
    if "patient_name" not in eob.columns:
        eob["patient_name"] = ""
    if "provider_name" not in eob.columns:
        eob["provider_name"] = ""
    if "amount_you_owe_provider" not in eob.columns:
        eob["amount_you_owe_provider"] = ""
    if "total_provider_responsibility" not in eob.columns:
        eob["total_provider_responsibility"] = ""
    if "total_billed" not in eob.columns:
        eob["total_billed"] = ""

    # Parse money-ish fields into floats
    eob["_amt_you_owe"] = eob["amount_you_owe_provider"].apply(to_money)
    eob["_amt_provider_resp"] = eob["total_provider_responsibility"].apply(to_money)
    eob["_amt_total_billed"] = eob["total_billed"].apply(to_money)

    # Prepare outputs
    COLUMNS = [
        "claimlink_row",
        "claim_id",
        "patient",
        "provider_bucket",
        "service_date",
        "eob_sha256",
        "eob_relpath",
        "eob_filename",
        "eob_patient_name",
        "eob_provider_name",
        "eob_service_date_min",
        "eob_service_date_max",
        "days_from_service",
        "provider_score",
        "date_score",
        "amount_score",
        "score",
        "link_status",
        "why",
    ]
    out_rows: list[dict[str, Any]] = []

    w = Weights()

    # Link columns in ClaimLink (create if absent)
    for col in ["eob_sha256", "eob_link_status", "eob_link_score", "eob_link_notes"]:
        if col not in claimlink.columns:
            claimlink[col] = ""

    auto_linked = 0
    needs_review = 0
    no_match = 0
    no_service_date = 0

    for i, r in claimlink.iterrows():
        claim_id = r.get("claim_id", "")
        patient = r.get("patient", r.get("service_for", ""))
        bucket = r.get("provider_bucket", "")
        svc_date = r.get("service_date_parsed", None)

        if not isinstance(svc_date, date):
            no_service_date += 1
            claimlink.loc[i, "eob_link_status"] = "NO_MATCH"
            claimlink.loc[i, "eob_link_score"] = ""
            claimlink.loc[i, "eob_link_notes"] = "no_service_date"
            continue

        # Optional: amount to compare — try "paid_amount" / "claim_amount" if present
        target_amount = None
        for amt_col in ["paid_amount", "claim_amount", "amount"]:
            if amt_col in r and not is_blank(r.get(amt_col)):
                target_amount = to_money(r.get(amt_col))
                if target_amount is not None:
                    break

        # Build candidates from EOB docs by date window first
        cand = eob.copy()

        # Use min/max date from EOB; if missing, treat as unknown
        def row_ddays(eob_min: date | None, eob_max: date | None) -> int | None:
            # If EOB has a range, use nearest endpoint distance
            if eob_min and eob_max:
                if eob_min <= svc_date <= eob_max:
                    return 0
                return min(abs((svc_date - eob_min).days), abs((svc_date - eob_max).days))
            if eob_min:
                return abs((svc_date - eob_min).days)
            if eob_max:
                return abs((svc_date - eob_max).days)
            return None

        cand["_ddays"] = cand.apply(lambda x: row_ddays(x["service_date_min_parsed"], x["service_date_max_parsed"]), axis=1)
        cand = cand[cand["_ddays"].isna() | (cand["_ddays"] <= args.max_days)].copy()

        if cand.empty:
            no_match += 1
            claimlink.loc[i, "eob_link_status"] = "NO_MATCH"
            claimlink.loc[i, "eob_link_score"] = ""
            claimlink.loc[i, "eob_link_notes"] = f"no_candidates_within_{args.max_days}_days"
            continue

        # Score candidates
        scored = []
        for _, e in cand.iterrows():
            prov_s, prov_reason = provider_bucket_match_score(str(bucket), str(e.get("provider_name", "")))
            dscore = date_score(e["_ddays"], args.max_days)

            eob_amounts = []
            for v in [e.get("_amt_you_owe"), e.get("_amt_provider_resp"), e.get("_amt_total_billed")]:
                if isinstance(v, float) and not math.isnan(v):
                    eob_amounts.append(float(v))
                elif isinstance(v, (int,)):
                    eob_amounts.append(float(v))

            ascore, areason = compute_amount_score(eob_amounts, target_amount, args.amount_tol)

            # Patient bonus (small)
            pscore = sim(patient, e.get("patient_name", ""))
            patient_bonus = 0.05 if pscore >= 0.65 else 0.0

            score = (w.provider * prov_s) + (w.date * dscore) + (w.amount * ascore) + patient_bonus

            why = f"{prov_reason} | ddays={e['_ddays']} date_score={dscore:.3f} | {areason} | patient_sim={pscore:.3f} bonus=+{patient_bonus:.2f}"

            if score >= args.min_score:
                scored.append((score, prov_s, dscore, ascore, why, e))

        if not scored:
            no_match += 1
            claimlink.loc[i, "eob_link_status"] = "NO_MATCH"
            claimlink.loc[i, "eob_link_score"] = ""
            claimlink.loc[i, "eob_link_notes"] = f"all_candidates_below_min_score={args.min_score}"
            continue

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[: args.top_n]

        # Emit candidates rows
        for (score, prov_s, dscore, ascore, why, e) in top:
            out_rows.append({
                "claimlink_row": i,
                "claim_id": claim_id,
                "patient": patient,
                "provider_bucket": bucket,
                "service_date": svc_date.isoformat(),
                "eob_sha256": e.get("sha256", ""),
                "eob_relpath": e.get("relpath", ""),
                "eob_filename": e.get("filename", ""),
                "eob_patient_name": e.get("patient_name", ""),
                "eob_provider_name": e.get("provider_name", ""),
                "eob_service_date_min": (e.get("service_date_min_parsed").isoformat() if isinstance(e.get("service_date_min_parsed"), date) else ""),
                "eob_service_date_max": (e.get("service_date_max_parsed").isoformat() if isinstance(e.get("service_date_max_parsed"), date) else ""),
                "days_from_service": (int(e["_ddays"]) if e["_ddays"] is not None and not (isinstance(e["_ddays"], float) and math.isnan(e["_ddays"])) else ""),
                "provider_score": round(float(prov_s), 4),
                "date_score": round(float(dscore), 4),
                "amount_score": round(float(ascore), 4),
                "score": round(float(score), 4),
                "link_status": "",
                "why": why,
            })

        # Decide best link
        best_score, best_prov_s, best_dscore, best_ascore, best_why, best_e = top[0]
        if best_score >= args.threshold:
            status = "AUTO_LINKED"
            auto_linked += 1
        else:
            status = "NEEDS_REVIEW"
            needs_review += 1

        claimlink.loc[i, "eob_sha256"] = best_e.get("sha256", "")
        claimlink.loc[i, "eob_link_status"] = status
        claimlink.loc[i, "eob_link_score"] = float(best_score)
        claimlink.loc[i, "eob_link_notes"] = best_why

    # Build output DF with fixed columns (so header always writes)
    out_df = pd.DataFrame(out_rows, columns=COLUMNS)

    # Always write CSV (even if empty)
    out_df.to_csv(out_csv, index=False)

    # Write into workbook sheets
    with pd.ExcelWriter(master_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
        out_df.to_excel(xw, index=False, sheet_name="EOBLinkCandidates")
        claimlink.drop(columns=["service_date_parsed"], errors="ignore").to_excel(xw, index=False, sheet_name="ClaimLink")

    print("OK EOB link candidates CSV:", out_csv)
    print("OK Updated workbook sheets: EOBLinkCandidates + ClaimLink ->", master_xlsx)
    print(
        f"OK claimlink_rows={len(claimlink)} auto_linked={auto_linked} needs_review={needs_review} "
        f"no_match={no_match} no_service_date={no_service_date} eob_docs_rows={len(eob)} candidates_rows={len(out_df)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
