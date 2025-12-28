# tools/tax_match_bcbsnd_to_optum_and_pnc.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Paths
# -----------------------------
YEAR = 2025
OUT_DIR = Path(f"notes/tax/work/parsed/{YEAR}")

BCBS_OVERVIEW = OUT_DIR / "bcbsnd_claims_overview.normalized.csv"
BCBS_EXPORT = OUT_DIR / "bcbsnd_claims_export.normalized.csv"
OPTUM_NORM = OUT_DIR / "optum_fsa_hsa_claims.normalized.csv"
PNC_CATEG = OUT_DIR / "pnc_spend_transactions.categorized.csv"

OUT_BCBS_UNIFIED = OUT_DIR / "bcbsnd_claims.unified.csv"
OUT_BCBS_X_OPTUM_MATCHED = OUT_DIR / "bcbsnd_x_optum.matched.csv"
OUT_BCBS_X_OPTUM_REVIEW = OUT_DIR / "bcbsnd_x_optum.review_candidates.csv"
OUT_BCBS_X_PNC_MATCHED = OUT_DIR / "bcbsnd_x_pnc_patient_pay.matched.csv"
OUT_BCBS_X_PNC_REVIEW = OUT_DIR / "bcbsnd_x_pnc_patient_pay.review_candidates.csv"
OUT_BCBS_UNMATCHED = OUT_DIR / "bcbsnd.unmatched_claims.csv"


# -----------------------------
# Tunables
# -----------------------------
# BCBS export -> Optum: claim_id exact is king
OPTUM_MATCH_DATE_WINDOW_DAYS = 14  # used only for non-claim-id fallbacks
OPTUM_MATCH_MIN_SCORE = 0.92
OPTUM_REVIEW_MIN_SCORE = 0.70

# BCBS overview patient-pay -> PNC withdrawals
PNC_LAG_DAYS_MIN = 0
PNC_LAG_DAYS_MAX = 75
PNC_MATCH_MIN_SCORE = 0.86
PNC_REVIEW_MIN_SCORE = 0.60

# Categories that boost medical-ness in PNC
PNC_MEDICALISH_PREFIXES = (
    "Medical:",
    "Health:",
    "Shopping: Pharmacy",
)

# Categories that are usually *not* a direct patient pay swipe
PNC_TRANSFERISH_PREFIXES = (
    "Transfer:",
    "ATM",
    "Cash",
)


# -----------------------------
# Helpers
# -----------------------------
_money_re = re.compile(r"[-+]?\$?\s*[\d,]+(?:\.\d{1,2})?\s*-?\s*$")


def _money_to_float(x) -> Optional[float]:
    """
    Accepts:
      "$1,052.88"  -> 1052.88
      "84.77 -"    -> -84.77
      -40.0        -> -40.0
      ""/NaN       -> None
    """
    if x is None:
        return None
    if isinstance(x, (int, float)) and pd.notna(x):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None

    neg = False
    # trailing "-" means negative in some statement exports
    if s.endswith("-"):
        neg = True
        s = s[:-1].strip()

    # remove currency and commas and spaces
    s = s.replace("$", "").replace(",", "").strip()

    try:
        val = float(s)
    except ValueError:
        return None

    return -val if neg else val


def _to_dt(series: pd.Series) -> pd.Series:
    # strict parsing is default now; don't use infer_datetime_format
    return pd.to_datetime(series, errors="coerce")


def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


STOP_TOKENS = {
    "pos", "purchase", "debit", "card", "visa", "mc", "mastercard", "amex",
    "md", "va", "dc", "usa",
    "www", "com", "llc", "inc", "co", "corp",
    "online", "electronic", "banking", "payment", "payments",
    "transfer", "zelle",
}


def _tokenize(s: str) -> List[str]:
    toks = [t for t in _norm_text(s).split(" ") if t and t not in STOP_TOKENS]
    return toks


def _jaccard(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta and not tb:
        return 0.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _write_csv(df: pd.DataFrame, path: Path, columns: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        df = df.reindex(columns=columns)
    # Always write headers, even if empty
    df.to_csv(path, index=False, encoding="utf-8", newline="\n")


def _is_medicalish_category(cat: str) -> int:
    c = (cat or "").strip()
    return 1 if any(c.startswith(p) for p in PNC_MEDICALISH_PREFIXES) else 0


def _is_transferish_category(cat: str) -> int:
    c = (cat or "").strip()
    return 1 if any(c.startswith(p) for p in PNC_TRANSFERISH_PREFIXES) else 0


# -----------------------------
# Load + unify BCBS
# -----------------------------
BCBS_UNIFIED_COLS = [
    "source",
    "source_file",
    "source_sheet",
    "row_index",
    "service_date",
    "service",
    "provider",
    "member",
    "bcbs_category",
    "member_responsibility",
    "paid_amount",
    "status",
    "claim_id",
    "_bcbs_source",
    "_bcbs_processed_date",
]


def load_bcbs_overview(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=BCBS_UNIFIED_COLS)

    df = pd.read_csv(path)
    # expected cols:
    # source,source_file,source_sheet,row_index,service_date,service,provider,member,bcbs_category,member_responsibility
    o = pd.DataFrame()
    o["source"] = df.get("source", "BCBSND")
    o["source_file"] = df.get("source_file", path.name)
    o["source_sheet"] = df.get("source_sheet", "Swain_Claims_Overview")
    o["row_index"] = df.get("row_index")
    o["service_date"] = _to_dt(df.get("service_date"))
    o["service"] = df.get("service")
    o["provider"] = df.get("provider")
    o["member"] = df.get("member")
    o["bcbs_category"] = df.get("bcbs_category")
    o["member_responsibility"] = df.get("member_responsibility").apply(_money_to_float)
    o["paid_amount"] = None
    o["status"] = None
    o["claim_id"] = None
    o["_bcbs_source"] = "BCBS_OVERVIEW"
    o["_bcbs_processed_date"] = None

    # types
    o["member_responsibility"] = pd.to_numeric(o["member_responsibility"], errors="coerce").fillna(0.0).astype("float64")
    return o.reindex(columns=BCBS_UNIFIED_COLS)


def load_bcbs_export(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=BCBS_UNIFIED_COLS)

    df = pd.read_csv(path)
    # cols:
    # source,source_file,source_sheet,row_index,claim_id,status,service_from,service_to,provider,paid_amount,member_responsibility
    e = pd.DataFrame()
    e["source"] = df.get("source", "BCBSND")
    e["source_file"] = df.get("source_file", path.name)
    e["source_sheet"] = df.get("source_sheet", "Claim Detail Report")
    e["row_index"] = df.get("row_index")

    # Export uses service_from/service_to; we’ll treat service_from as service_date
    e["service_date"] = _to_dt(df.get("service_from"))
    e["service"] = None
    e["provider"] = df.get("provider")
    e["member"] = None
    e["bcbs_category"] = None

    e["member_responsibility"] = df.get("member_responsibility").apply(_money_to_float)
    e["paid_amount"] = df.get("paid_amount").apply(_money_to_float)
    e["status"] = df.get("status")
    e["claim_id"] = df.get("claim_id")
    e["_bcbs_source"] = "BCBS_EXPORT"
    e["_bcbs_processed_date"] = None

    e["member_responsibility"] = pd.to_numeric(e["member_responsibility"], errors="coerce").fillna(0.0).astype("float64")
    e["paid_amount"] = pd.to_numeric(e["paid_amount"], errors="coerce")
    return e.reindex(columns=BCBS_UNIFIED_COLS)


def unify_bcbs(overview: pd.DataFrame, export: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if overview is not None and not overview.empty:
        frames.append(overview.reindex(columns=BCBS_UNIFIED_COLS))
    if export is not None and not export.empty:
        frames.append(export.reindex(columns=BCBS_UNIFIED_COLS))

    if not frames:
        return pd.DataFrame(columns=BCBS_UNIFIED_COLS)

    bcbs = pd.concat(frames, ignore_index=True)
    # normalize obvious dtypes
    bcbs["service_date"] = _to_dt(bcbs["service_date"])
    bcbs["member_responsibility"] = pd.to_numeric(bcbs["member_responsibility"], errors="coerce").fillna(0.0).astype("float64")
    return bcbs


# -----------------------------
# Load Optum + PNC
# -----------------------------
def load_optum(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    o = df.copy()
    o["optum_date"] = _to_dt(o.get("date"))
    o["optum_amount"] = o.get("amount").apply(_money_to_float)
    o["optum_merchant"] = o.get("merchant")
    o["optum_status"] = o.get("status")
    o["optum_claim_id"] = o.get("claim_id")
    return o


def load_pnc(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    p = df.copy()
    p["pnc_date"] = _to_dt(p.get("date"))
    p["pnc_amount"] = pd.to_numeric(p.get("amount"), errors="coerce")
    p["pnc_section"] = p.get("section")
    p["pnc_category"] = p.get("category")
    p["pnc_merchant"] = p.get("merchant")
    p["pnc_description"] = p.get("description")
    return p


# -----------------------------
# Matching: BCBS Export -> Optum
# -----------------------------
def match_bcbs_to_optum(bcbs: pd.DataFrame, optum: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strategy:
      1) Exact claim_id join (BCBS_EXPORT only) => matched, score 1.00
      2) Fallback (rare): amount/date/provider~merchant heuristic => review or matched depending score
    """
    if bcbs.empty or optum.empty:
        return (
            pd.DataFrame(columns=[]),
            pd.DataFrame(columns=[]),
        )

    # Focus on export rows for claim-id matching
    bcbs_export = bcbs[bcbs["_bcbs_source"] == "BCBS_EXPORT"].copy()
    bcbs_export["claim_id_str"] = bcbs_export["claim_id"].astype(str).where(bcbs_export["claim_id"].notna(), None)
    optum["optum_claim_id_str"] = optum["optum_claim_id"].astype(str).where(optum["optum_claim_id"].notna(), None)

    joined = bcbs_export.merge(
        optum,
        left_on="claim_id_str",
        right_on="optum_claim_id_str",
        how="left",
        suffixes=("", "_opt"),
    )

    exact = joined[joined["optum_claim_id_str"].notna()].copy()
    exact["_match_score"] = 1.0
    exact["_match_reason"] = "bcbs->optum claim_id_exact"

    # For export rows that did not claim-id match, try heuristics
    leftovers = joined[joined["optum_claim_id_str"].isna()].copy()
    heur_rows: List[dict] = []

    if not leftovers.empty:
        # build candidate pool: optum paid only is most useful
        opt = optum.copy()
        opt = opt[opt["optum_date"].notna()].copy()
        opt["optum_amount_num"] = pd.to_numeric(opt["optum_amount"], errors="coerce")

        for _, r in leftovers.iterrows():
            sd = r.get("service_date")
            if pd.isna(sd):
                continue

            prov = str(r.get("provider") or "")
            paid = r.get("paid_amount")
            paid = float(paid) if pd.notna(paid) else None

            # candidate window
            lo = sd - timedelta(days=OPTUM_MATCH_DATE_WINDOW_DAYS)
            hi = sd + timedelta(days=OPTUM_MATCH_DATE_WINDOW_DAYS)
            cands = opt[(opt["optum_date"] >= lo) & (opt["optum_date"] <= hi)].copy()

            # if paid_amount exists, filter close
            if paid is not None:
                cands = cands[cands["optum_amount_num"].between(paid - 0.02, paid + 0.02)]

            if cands.empty:
                continue

            best = None
            best_score = -1.0
            best_reason = ""

            for _, o in cands.iterrows():
                sim = _jaccard(prov, str(o.get("optum_merchant") or ""))
                daydiff = abs((o["optum_date"] - sd).days)
                date_score = max(0.0, 1.0 - (daydiff / (OPTUM_MATCH_DATE_WINDOW_DAYS + 1)))
                amt_ok = 1.0 if paid is not None else 0.0

                score = (0.65 * sim) + (0.30 * date_score) + (0.05 * amt_ok)
                reason = f"bcbs->optum heuristic sim={sim:.2f} daydiff={daydiff} date_score={date_score:.2f} amt_ok={amt_ok:.0f}"

                if score > best_score:
                    best_score = score
                    best = o
                    best_reason = reason

            if best is None:
                continue

            out = dict(r)
            out["optum_date"] = best.get("optum_date")
            out["optum_amount"] = best.get("optum_amount")
            out["optum_merchant"] = best.get("optum_merchant")
            out["optum_status"] = best.get("optum_status")
            out["optum_claim_id"] = best.get("optum_claim_id")
            out["_match_score"] = float(best_score)
            out["_match_reason"] = best_reason
            heur_rows.append(out)

    heur = pd.DataFrame(heur_rows) if heur_rows else pd.DataFrame()

    # Split heuristic into matched vs review
    heur_matched = heur[heur["_match_score"] >= OPTUM_MATCH_MIN_SCORE].copy() if not heur.empty else pd.DataFrame()
    heur_review = heur[
        (heur["_match_score"] >= OPTUM_REVIEW_MIN_SCORE) & (heur["_match_score"] < OPTUM_MATCH_MIN_SCORE)
    ].copy() if not heur.empty else pd.DataFrame()

    # Exact always matched
    matched = pd.concat([exact, heur_matched], ignore_index=True) if not exact.empty or not heur_matched.empty else pd.DataFrame()
    review = heur_review

    return matched, review


# -----------------------------
# Matching: BCBS overview patient pay -> PNC withdrawals
# -----------------------------
def match_patient_pay_to_pnc(bcbs: pd.DataFrame, pnc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Only BCBS_OVERVIEW rows with member_responsibility > 0 are considered patient-pay claims.
    We try to find PNC withdrawal candidates with same absolute amount within lag window.
    Then score by:
      - amount exact (hard gate)
      - lag_days score (prefer shorter lag)
      - provider~merchant similarity
      - medicalish boost
      - transferish penalty
    """
    if bcbs.empty or pnc.empty:
        return (pd.DataFrame(), pd.DataFrame())

    bc = bcbs[(bcbs["_bcbs_source"] == "BCBS_OVERVIEW") & (bcbs["member_responsibility"] > 0)].copy()
    bc = bc[bc["service_date"].notna()].copy()
    if bc.empty:
        return (pd.DataFrame(), pd.DataFrame())

    pn = pnc[pnc["pnc_date"].notna()].copy()
    pn = pn[pn["pnc_amount"].notna()].copy()
    # withdrawals only
    pn = pn[pn["pnc_amount"] < 0].copy()
    if pn.empty:
        return (pd.DataFrame(), pd.DataFrame())

    # work in cents for exactness
    bc["amt_cents"] = (bc["member_responsibility"].round(2) * 100).astype(int)
    pn["amt_cents"] = (pn["pnc_amount"].abs().round(2) * 100).astype(int)

    # join on amount cents
    cand = bc.merge(
        pn,
        on="amt_cents",
        how="left",
        suffixes=("_bcbs", "_pnc"),
    )

    # lag window: PNC date within [min,max] days after service_date
    cand["lag_days"] = (cand["pnc_date"] - cand["service_date"]).dt.days
    cand = cand[(cand["lag_days"] >= PNC_LAG_DAYS_MIN) & (cand["lag_days"] <= PNC_LAG_DAYS_MAX)].copy()
    if cand.empty:
        return (pd.DataFrame(), pd.DataFrame())

    # score candidates per BCBS row
    scored_rows: List[dict] = []
    for _, r in cand.iterrows():
        prov = str(r.get("provider") or "")
        merch = str(r.get("pnc_merchant") or "")
        cat = str(r.get("pnc_category") or "")

        sim = _jaccard(prov, merch)
        lag = int(r.get("lag_days"))
        lag_score = max(0.0, 1.0 - (lag / (PNC_LAG_DAYS_MAX + 1)))

        medish = _is_medicalish_category(cat)
        transferish = _is_transferish_category(cat)

        # weighting
        score = (0.55 * lag_score) + (0.35 * sim) + (0.15 * medish) - (0.25 * transferish)
        score = float(max(-1.0, min(1.0, score)))

        r_out = dict(r)
        r_out["_match_score"] = score
        r_out["_match_reason"] = f"bcbs->pnc amt_ok lag={lag} lag_score={lag_score:.2f} sim={sim:.2f} medish={medish} transferish={transferish}"
        scored_rows.append(r_out)

    scored = pd.DataFrame(scored_rows)

    # pick best candidate per BCBS row_index (keep top 1)
    scored = scored.sort_values(["row_index", "_match_score"], ascending=[True, False])
    best = scored.groupby("row_index", as_index=False).head(1).copy()

    matched = best[best["_match_score"] >= PNC_MATCH_MIN_SCORE].copy()
    review = best[(best["_match_score"] >= PNC_REVIEW_MIN_SCORE) & (best["_match_score"] < PNC_MATCH_MIN_SCORE)].copy()

    return matched, review


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    overview = load_bcbs_overview(BCBS_OVERVIEW)
    export = load_bcbs_export(BCBS_EXPORT)
    bcbs = unify_bcbs(overview, export)

    optum = load_optum(OPTUM_NORM)
    pnc = load_pnc(PNC_CATEG)

    _write_csv(bcbs, OUT_BCBS_UNIFIED, columns=BCBS_UNIFIED_COLS)

    # BCBS->Optum
    opt_matched, opt_review = match_bcbs_to_optum(bcbs, optum)

    opt_out_cols = BCBS_UNIFIED_COLS + [
        "optum_date",
        "optum_amount",
        "optum_merchant",
        "optum_status",
        "optum_claim_id",
        "_match_score",
        "_match_reason",
    ]
    _write_csv(opt_matched, OUT_BCBS_X_OPTUM_MATCHED, columns=opt_out_cols)
    _write_csv(opt_review, OUT_BCBS_X_OPTUM_REVIEW, columns=opt_out_cols)

    # Patient-pay -> PNC
    pnc_matched, pnc_review = match_patient_pay_to_pnc(bcbs, pnc)

    pnc_out_cols = BCBS_UNIFIED_COLS + [
        "pnc_date",
        "pnc_amount",
        "pnc_section",
        "pnc_category",
        "pnc_merchant",
        "pnc_description",
        "lag_days",
        "_match_score",
        "_match_reason",
    ]
    _write_csv(pnc_matched, OUT_BCBS_X_PNC_MATCHED, columns=pnc_out_cols)
    _write_csv(pnc_review, OUT_BCBS_X_PNC_REVIEW, columns=pnc_out_cols)

    # Unmatched BCBS (not in optum matched AND not in patient-pay matched)
    matched_keys = set()
    if not opt_matched.empty:
        matched_keys |= set(opt_matched["row_index"].astype(int).tolist())
    if not pnc_matched.empty:
        matched_keys |= set(pnc_matched["row_index"].astype(int).tolist())

    bcbs_unmatched = bcbs[~bcbs["row_index"].astype("Int64").isin(list(matched_keys))].copy()
    _write_csv(bcbs_unmatched, OUT_BCBS_UNMATCHED, columns=BCBS_UNIFIED_COLS + ["_bcbs_source"])

    print("wrote:")
    print(f"  {OUT_BCBS_UNIFIED} rows: {len(bcbs)}")
    print(f"  {OUT_BCBS_X_OPTUM_MATCHED} rows: {len(opt_matched)}")
    print(f"  {OUT_BCBS_X_OPTUM_REVIEW} rows: {len(opt_review)}")
    print(f"  {OUT_BCBS_X_PNC_MATCHED} rows: {len(pnc_matched)}")
    print(f"  {OUT_BCBS_X_PNC_REVIEW} rows: {len(pnc_review)}")
    print(f"  {OUT_BCBS_UNMATCHED} rows: {len(bcbs_unmatched)}")
    print("")
    print(f"BCBS rows loaded: {len(bcbs)} (overview+export)")
    print(f"Optum rows loaded: {len(optum)}")
    print(f"PNC rows loaded: {len(pnc)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
