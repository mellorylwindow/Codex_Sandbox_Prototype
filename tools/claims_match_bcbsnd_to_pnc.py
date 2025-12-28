from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CLAIMS = Path("notes/tax/work/parsed/2025/bcbsnd_claims.normalized.csv")
PNC    = Path("notes/tax/work/parsed/2025/pnc_spend_transactions.categorized.csv")
OUTDIR = Path("notes/tax/work/parsed/2025")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_MATCHED     = OUTDIR / "bcbsnd_x_pnc.matched.csv"
OUT_UNCLAIMED   = OUTDIR / "bcbsnd_x_pnc.unmatched_claims.csv"
OUT_UNPNC       = OUTDIR / "bcbsnd_x_pnc.unmatched_pnc_medicalish.csv"
OUT_AMBIG       = OUTDIR / "bcbsnd_x_pnc.ambiguities.csv"

# Tuning knobs
DAY_WINDOW = 30              # +/- days around claim dates to search
AMT_TOL = 0.01               # exact cents match
SCORE_THRESHOLD = 0.62       # accept match if >= this
MAX_CANDIDATES_LOG = 12      # for ambiguity output

MEDICALISH_CATS = {
    "Medical: Doctors",
    "Medical: Dental",
    "Medical: Mental Health",
    "Health: Labs",
    "Shopping: Pharmacy",
}

STOP = {
    "the","and","of","for","to","a","an","llc","inc","co","corp",
    "hospital","medical","center","clinic","associates","group","pc",
    "services","service","health","care",
    "md","va","dc","ny","nj","ca",
    "tst","sq","py","pos","purchase","debit","card","visa","vis",
}

def parse_date(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except Exception:
        return None

def ffloat(s: str) -> float:
    try:
        return float((s or "").strip())
    except Exception:
        return 0.0

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def tokenize(s: str) -> List[str]:
    s = norm_text(s).lower()
    parts = re.split(r"[^a-z0-9]+", s)
    toks = []
    for p in parts:
        if not p:
            continue
        if p in STOP:
            continue
        if len(p) <= 2:
            continue
        # kill obvious numeric noise except useful phone-like patterns
        if p.isdigit() and len(p) >= 4:
            continue
        toks.append(p)
    return toks

def token_overlap(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / max(1, min(len(ta), len(tb)))

@dataclass
class Claim:
    idx: int
    claim_id: str
    provider: str
    dos_from: Optional[datetime]
    dos_to: Optional[datetime]
    received: Optional[datetime]
    processed: Optional[datetime]
    status: str
    patient_resp: float
    billed: float
    raw: Dict[str, str]

    def amount_candidates(self) -> List[float]:
        # prefer patient responsibility; fall back to billed if resp missing/zero
        cands = []
        if abs(self.patient_resp) > 0.0001:
            cands.append(abs(self.patient_resp))
        if abs(self.billed) > 0.0001:
            cands.append(abs(self.billed))
        # de-dupe
        out = []
        for x in cands:
            if not any(abs(x - y) <= AMT_TOL for y in out):
                out.append(x)
        return out

    def best_date(self) -> Optional[datetime]:
        # matching charge date is often around service date, sometimes near processed date
        return self.dos_from or self.processed or self.received

@dataclass
class PncTx:
    idx: int
    date: datetime
    amount: float
    merchant: str
    description: str
    category: str
    raw: Dict[str, str]

    @property
    def abs_amt(self) -> float:
        return abs(self.amount)

def load_claims() -> List[Claim]:
    rows = list(csv.DictReader(CLAIMS.open(newline="", encoding="utf-8")))
    out: List[Claim] = []
    for i, r in enumerate(rows, start=1):
        out.append(
            Claim(
                idx=i,
                claim_id=norm_text(r.get("claim_id","")),
                provider=norm_text(r.get("provider","")),
                dos_from=parse_date(r.get("service_date_from","")),
                dos_to=parse_date(r.get("service_date_to","")),
                received=parse_date(r.get("received_date","")),
                processed=parse_date(r.get("processed_date","")),
                status=norm_text(r.get("status","")),
                patient_resp=ffloat(r.get("patient_responsibility","")),
                billed=ffloat(r.get("billed_amount","")),
                raw=r,
            )
        )
    return out

def load_pnc() -> List[PncTx]:
    rows = list(csv.DictReader(PNC.open(newline="", encoding="utf-8")))
    out: List[PncTx] = []
    for i, r in enumerate(rows, start=1):
        d = parse_date(r.get("date",""))
        if not d:
            continue
        out.append(
            PncTx(
                idx=i,
                date=d,
                amount=ffloat(r.get("amount","")),
                merchant=norm_text(r.get("merchant","")),
                description=norm_text(r.get("description","")),
                category=norm_text(r.get("category","")),
                raw=r,
            )
        )
    return out

def score_match(cl: Claim, tx: PncTx) -> float:
    # Amount score
    cands = cl.amount_candidates()
    if not cands:
        return 0.0

    amt_ok = any(abs(tx.abs_amt - a) <= AMT_TOL for a in cands)
    if not amt_ok:
        return 0.0

    amt_score = 0.62  # big weight for exact cents match

    # Date score (closer is better)
    ref = cl.best_date()
    if not ref:
        date_score = 0.15
    else:
        dd = abs((tx.date - ref).days)
        if dd > DAY_WINDOW:
            return 0.0
        # 0 days => 0.30, 30 days => ~0.05
        date_score = 0.30 * (1.0 - (dd / DAY_WINDOW)) + 0.05

    # Provider text overlap
    prov = cl.provider
    tx_txt = f"{tx.merchant} {tx.description}"
    ov = token_overlap(prov, tx_txt)
    # cap overlap contribution
    text_score = min(0.20, 0.20 * ov)

    return amt_score + date_score + text_score

def pick_best(cl: Claim, candidates: List[PncTx]) -> Tuple[Optional[PncTx], List[Tuple[PncTx,float]]]:
    scored = [(tx, score_match(cl, tx)) for tx in candidates]
    scored = [(tx,s) for (tx,s) in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored:
        return None, []

    best_tx, best_s = scored[0]
    # if close tie, treat as ambiguous
    if len(scored) >= 2 and (best_s - scored[1][1]) < 0.04:
        return None, scored[:MAX_CANDIDATES_LOG]

    if best_s >= SCORE_THRESHOLD:
        return best_tx, scored[:MAX_CANDIDATES_LOG]

    return None, scored[:MAX_CANDIDATES_LOG]

def main():
    if not CLAIMS.exists():
        raise SystemExit(f"missing: {CLAIMS}")
    if not PNC.exists():
        raise SystemExit(f"missing: {PNC}")

    claims = load_claims()
    pnc = load_pnc()

    # Index PNC by date for quick candidate retrieval
    by_date: Dict[datetime, List[PncTx]] = defaultdict(list)
    for tx in pnc:
        by_date[tx.date].append(tx)

    def date_range(center: datetime, days: int) -> List[datetime]:
        return [center + timedelta(days=i) for i in range(-days, days+1)]

    used_pnc_idxs = set()
    matched_rows = []
    unclaimed_rows = []
    ambiguities = []

    for cl in claims:
        ref = cl.best_date()
        # If no date, search all but that’s expensive; instead look at the full year range
        if not ref:
            # fallback: try processed, received, dos_to, else skip
            ref = cl.processed or cl.received or cl.dos_to
        if not ref:
            unclaimed_rows.append(cl.raw)
            continue

        # candidate set: only debit transactions, near date window
        cands: List[PncTx] = []
        for d in date_range(ref, DAY_WINDOW):
            for tx in by_date.get(d, []):
                # If it’s a credit (income/reimbursement), skip for now
                if tx.amount > 0:
                    continue
                cands.append(tx)

        best, scored = pick_best(cl, cands)
        if best and best.idx not in used_pnc_idxs:
            used_pnc_idxs.add(best.idx)
            out = dict(cl.raw)
            # attach PNC fields
            out.update({
                "_pnc_date": best.raw.get("date",""),
                "_pnc_amount": best.raw.get("amount",""),
                "_pnc_category": best.raw.get("category",""),
                "_pnc_merchant": best.raw.get("merchant",""),
                "_pnc_description": best.raw.get("description",""),
            })
            # attach score
            out["_match_score"] = f"{score_match(cl, best):.3f}"
            matched_rows.append(out)
        else:
            if scored:
                # ambiguous or low-score: log candidates
                for rank, (tx, s) in enumerate(scored, start=1):
                    ambiguities.append({
                        "claim_row_id": cl.raw.get("row_id",""),
                        "claim_id": cl.raw.get("claim_id",""),
                        "provider": cl.raw.get("provider",""),
                        "service_date_from": cl.raw.get("service_date_from",""),
                        "patient_responsibility": cl.raw.get("patient_responsibility",""),
                        "billed_amount": cl.raw.get("billed_amount",""),
                        "rank": rank,
                        "score": f"{s:.3f}",
                        "pnc_date": tx.raw.get("date",""),
                        "pnc_amount": tx.raw.get("amount",""),
                        "pnc_category": tx.raw.get("category",""),
                        "pnc_merchant": tx.raw.get("merchant",""),
                        "pnc_description": tx.raw.get("description",""),
                    })
            unclaimed_rows.append(cl.raw)

    # PNC medical-ish debits not used by any claim match
    unmatched_pnc_med = []
    for tx in pnc:
        if tx.amount > 0:
            continue
        if tx.idx in used_pnc_idxs:
            continue
        if tx.category in MEDICALISH_CATS or ("medical" in (tx.category or "").lower()):
            unmatched_pnc_med.append(tx.raw)

    # write outputs
    def write_any(path: Path, rows: List[Dict[str,str]]):
        if not rows:
            # still write headers if we can infer
            path.write_text("", encoding="utf-8")
            return
        # union fieldnames
        fns = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fns.append(k)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fns)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_any(OUT_MATCHED, matched_rows)
    write_any(OUT_UNCLAIMED, unclaimed_rows)
    write_any(OUT_UNPNC, unmatched_pnc_med)
    write_any(OUT_AMBIG, ambiguities)

    # stats
    print("wrote:")
    print(f"  {OUT_MATCHED} rows: {len(matched_rows)}")
    print(f"  {OUT_UNCLAIMED} rows: {len(unclaimed_rows)}")
    print(f"  {OUT_UNPNC} rows: {len(unmatched_pnc_med)}")
    print(f"  {OUT_AMBIG} rows: {len(ambiguities)}")
    if claims:
        print(f"claims total: {len(claims)}  match rate: {len(matched_rows)}/{len(claims)} ({(len(matched_rows)/len(claims))*100:.1f}%)")

if __name__ == "__main__":
    main()
