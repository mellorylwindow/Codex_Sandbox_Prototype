from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OPTUM_NORM = Path("notes/tax/work/parsed/2025/optum_fsa_hsa_claims.normalized.csv")
PNC_CATEG  = Path("notes/tax/work/parsed/2025/pnc_spend_transactions.categorized.csv")
OUT_DIR    = Path("notes/tax/work/parsed/2025")

# Expense window: card swipes usually same/next day
EXPENSE_WINDOW_DAYS = 3

# Reimbursement window: deposits can be much later than service/claim date
REIMB_MIN_DAYS = 0
REIMB_MAX_DAYS = 45

AMT_TOL = 0.02  # cents tolerance

STOP_TOKENS = {
    "tst","sq","py","pos","purchase","debit","card","visa","vis",
    "md","va","dc","ca","oh","nj","wa",
    "www","com","llc","inc","co",
    "n","x","xxxx","xxxxxxxx","xxxxxxxxxxxx",
}

def parse_optum_date(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None

def parse_pnc_date(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None

def parse_money(val) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    s = s.replace("$", "").replace(",", "").strip()
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()
    try:
        return float(s)
    except ValueError:
        return None

def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(?:tst|sq|py|pos)\*?\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str) -> List[str]:
    out: List[str] = []
    for w in norm_text(s).split():
        if w in STOP_TOKENS:
            continue
        if len(w) <= 2:
            continue
        if w.isdigit() and len(w) >= 4:
            continue
        out.append(w)
    return out

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def days_between(a: datetime, b: datetime) -> int:
    return (b.date() - a.date()).days

@dataclass
class OptumRow:
    raw: Dict[str, str]
    dt: Optional[datetime]
    amt: Optional[float]  # positive
    merchant: str
    status: str
    claim_id: str

@dataclass
class PncRow:
    raw: Dict[str, str]
    dt: Optional[datetime]
    amt: Optional[float]  # signed: withdrawals negative, deposits positive
    abs_amt: Optional[float]
    merchant: str
    desc: str
    cat: str
    section: str
    tx_type: str

def pnc_id(pr: PncRow) -> Tuple[str, str, str, str]:
    return (
        pr.raw.get("date",""),
        pr.raw.get("amount",""),
        pr.raw.get("merchant",""),
        pr.raw.get("description",""),
    )

def load_optum() -> List[OptumRow]:
    out: List[OptumRow] = []
    with open(OPTUM_NORM, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            dt = parse_optum_date(row.get("date",""))
            amt = parse_money(row.get("amount",""))
            merchant = (row.get("merchant") or "").strip()
            status = (row.get("status") or "").strip()
            claim_id = str(row.get("claim_id") or "").strip()
            out.append(OptumRow(row, dt, amt, merchant, status, claim_id))
    return out

def load_pnc_all() -> List[PncRow]:
    out: List[PncRow] = []
    with open(PNC_CATEG, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            dt = parse_pnc_date(row.get("date",""))
            amt = parse_money(row.get("amount",""))
            if amt is None:
                continue
            section = (row.get("section") or "").strip().lower()
            tx_type = (row.get("tx_type") or "").strip().lower()
            merchant = (row.get("merchant") or "").strip()
            desc = (row.get("description") or "").strip()
            cat = (row.get("category") or "").strip()
            out.append(PncRow(row, dt, amt, abs(amt), merchant, desc, cat, section, tx_type))
    return out

def best_expense_match(o: OptumRow, pnc: List[PncRow], used: set) -> Optional[Tuple[PncRow, float, str]]:
    """Match Optum claim amount to a PNC withdrawal around same date."""
    if o.dt is None or o.amt is None:
        return None
    ot = tokens(o.merchant)
    best = None
    for pr in pnc:
        if pnc_id(pr) in used:
            continue
        if pr.dt is None:
            continue
        if pr.section != "withdrawals":
            continue
        # amount must match (abs)
        if abs(pr.abs_amt - abs(o.amt)) > AMT_TOL:
            continue
        dd = abs(days_between(o.dt, pr.dt))
        if dd > EXPENSE_WINDOW_DAYS:
            continue
        pt = tokens(pr.merchant + " " + pr.desc)
        sim = jaccard(ot, pt)
        score = sim - (dd * 0.05)
        reason = f"expense strict amt date±{EXPENSE_WINDOW_DAYS} sim={sim:.2f} daydiff={dd}"
        if best is None or score > best[1]:
            best = (pr, score, reason)
    return best

def best_reimb_match(o: OptumRow, pnc: List[PncRow], used: set) -> Optional[Tuple[PncRow, float, str]]:
    """Match Optum claim amount to a PNC deposit after claim date."""
    if o.dt is None or o.amt is None:
        return None
    ot = tokens(o.merchant)
    best = None
    for pr in pnc:
        if pnc_id(pr) in used:
            continue
        if pr.dt is None:
            continue
        if pr.section != "deposits":
            continue
        # amount must match
        if abs(pr.amt - abs(o.amt)) > AMT_TOL:
            continue
        lag = days_between(o.dt, pr.dt)  # positive means deposit after claim date
        if lag < REIMB_MIN_DAYS or lag > REIMB_MAX_DAYS:
            continue
        pt = tokens(pr.merchant + " " + pr.desc)
        sim = jaccard(ot, pt)
        score = sim - (lag * 0.01)  # don't punish lag too hard
        reason = f"reimb strict amt lag[{REIMB_MIN_DAYS},{REIMB_MAX_DAYS}] sim={sim:.2f} lag_days={lag}"
        if best is None or score > best[1]:
            best = (pr, score, reason)
    return best

def suggest_review_candidates(o: OptumRow, pnc: List[PncRow], used: set, max_rows: int = 5) -> List[Tuple[PncRow, float, str]]:
    """Merchant-only suggestions for human review (NOT counted as matches)."""
    if o.dt is None:
        return []
    ot = tokens(o.merchant)
    scored = []
    for pr in pnc:
        if pnc_id(pr) in used:
            continue
        if pr.dt is None:
            continue
        # keep it near-ish in time
        dd = abs(days_between(o.dt, pr.dt))
        if dd > 7:
            continue
        pt = tokens(pr.merchant + " " + pr.desc)
        sim = jaccard(ot, pt)
        if sim < 0.70:
            continue
        score = sim - (dd * 0.05)
        reason = f"review merchant_only sim={sim:.2f} daydiff={dd}"
        scored.append((pr, score, reason))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_rows]

def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    optum = load_optum()
    pnc = load_pnc_all()
    used = set()

    matched_expense: List[Dict[str, str]] = []
    matched_reimb: List[Dict[str, str]] = []
    review: List[Dict[str, str]] = []
    unmatched: List[Dict[str, str]] = []

    for o in optum:
        status_l = (o.status or "").strip().lower()
        if status_l == "voided":
            unmatched.append({**o.raw, "_match_reason": "optum_voided"})
            continue
        if o.dt is None or o.amt is None:
            unmatched.append({**o.raw, "_match_reason": "missing_amt_or_date"})
            continue

        hit = best_expense_match(o, pnc, used)
        if hit is not None:
            pr, score, reason = hit
            used.add(pnc_id(pr))
            out = dict(o.raw)
            out.update({
                "pnc_date": pr.raw.get("date",""),
                "pnc_amount": pr.raw.get("amount",""),
                "pnc_section": pr.raw.get("section",""),
                "pnc_category": pr.raw.get("category",""),
                "pnc_merchant": pr.raw.get("merchant",""),
                "pnc_description": pr.raw.get("description",""),
                "_match_score": f"{score:.3f}",
                "_match_reason": reason,
            })
            matched_expense.append(out)
            continue

        hit = best_reimb_match(o, pnc, used)
        if hit is not None:
            pr, score, reason = hit
            used.add(pnc_id(pr))
            out = dict(o.raw)
            out.update({
                "pnc_date": pr.raw.get("date",""),
                "pnc_amount": pr.raw.get("amount",""),
                "pnc_section": pr.raw.get("section",""),
                "pnc_category": pr.raw.get("category",""),
                "pnc_merchant": pr.raw.get("merchant",""),
                "pnc_description": pr.raw.get("description",""),
                "_match_score": f"{score:.3f}",
                "_match_reason": reason,
            })
            matched_reimb.append(out)
            continue

        # Not matched: emit review candidates (merchant-only suggestions)
        suggestions = suggest_review_candidates(o, pnc, used)
        if suggestions:
            for pr, score, reason in suggestions:
                out = dict(o.raw)
                out.update({
                    "pnc_date": pr.raw.get("date",""),
                    "pnc_amount": pr.raw.get("amount",""),
                    "pnc_section": pr.raw.get("section",""),
                    "pnc_category": pr.raw.get("category",""),
                    "pnc_merchant": pr.raw.get("merchant",""),
                    "pnc_description": pr.raw.get("description",""),
                    "_match_score": f"{score:.3f}",
                    "_match_reason": reason,
                })
                review.append(out)
            unmatched.append({**o.raw, "_match_reason": "no_strict_match_but_review_candidates"})
        else:
            unmatched.append({**o.raw, "_match_reason": "no_candidate"})

    # Unmatched PNC rows (all sections) that weren’t used by strict matches
    unmatched_pnc: List[Dict[str, str]] = []
    for pr in pnc:
        if pnc_id(pr) in used:
            continue
        unmatched_pnc.append(pr.raw)

    opt_fields = list(optum[0].raw.keys()) if optum else []
    out_fields = opt_fields + [
        "pnc_date","pnc_amount","pnc_section","pnc_category","pnc_merchant","pnc_description",
        "_match_score","_match_reason"
    ]

    out_exp  = OUT_DIR / "optum_x_pnc.matched_expense.csv"
    out_reim = OUT_DIR / "optum_x_pnc.matched_reimb.csv"
    out_rev  = OUT_DIR / "optum_x_pnc.review_candidates.csv"
    out_un_o = OUT_DIR / "optum_x_pnc.unmatched_optum.csv"
    out_un_p = OUT_DIR / "optum_x_pnc.unmatched_pnc_all.csv"

    write_csv(out_exp,  out_fields, matched_expense)
    write_csv(out_reim, out_fields, matched_reimb)
    write_csv(out_rev,  out_fields, review)
    write_csv(out_un_o, opt_fields + ["_match_reason"], unmatched)

    if unmatched_pnc:
        pnc_fields = list(unmatched_pnc[0].keys())
    else:
        with open(PNC_CATEG, newline="", encoding="utf-8") as f:
            pnc_fields = csv.DictReader(f).fieldnames or []
    write_csv(out_un_p, pnc_fields, unmatched_pnc)

    print("wrote:")
    print(f"  {out_exp} rows: {len(matched_expense)}")
    print(f"  {out_reim} rows: {len(matched_reimb)}")
    print(f"  {out_rev} rows: {len(review)}")
    print(f"  {out_un_o} rows: {len(unmatched)}")
    print(f"  {out_un_p} rows: {len(unmatched_pnc)}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
