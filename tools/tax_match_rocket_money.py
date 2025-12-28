from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# -------------------------
# Helpers
# -------------------------

def parse_iso_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    # Rocket uses YYYY-MM-DD
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def to_float(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    # remove commas
    s = s.replace(",", "")
    return float(s)


def cents_key(x: float) -> int:
    # robust cents rounding
    return int(round(x * 100))


_word_re = re.compile(r"[A-Z0-9]+", re.I)


def norm_words(s: str) -> List[str]:
    return _word_re.findall((s or "").upper())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


@dataclass
class PncTx:
    idx: int
    date: date
    amount: float   # signed (debits negative, credits positive)
    merchant: str
    description: str
    category: str
    raw: Dict[str, str]


@dataclass
class RocketTx:
    idx: int
    date: date
    original_date: Optional[date]
    amount: float   # Rocket: expenses positive, income negative (often)
    name: str
    description: str
    category: str
    institution: str
    account_name: str
    raw: Dict[str, str]


def load_pnc(path: Path) -> List[PncTx]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    out: List[PncTx] = []
    for i, r in enumerate(rows):
        d = (r.get("date") or "").strip()
        if not d:
            continue
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        amt = to_float(r.get("amount") or "0")
        out.append(
            PncTx(
                idx=i,
                date=dt,
                amount=amt,
                merchant=(r.get("merchant") or "").strip(),
                description=(r.get("description") or "").strip(),
                category=(r.get("category") or "").strip() or "<?>",
                raw=r,
            )
        )
    return out


def load_rocket(path: Path) -> List[RocketTx]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    out: List[RocketTx] = []
    for i, r in enumerate(rows):
        dt = parse_iso_date(r.get("Date") or "")
        if not dt:
            continue
        odt = parse_iso_date(r.get("Original Date") or "")
        amt = to_float(r.get("Amount") or "0")
        out.append(
            RocketTx(
                idx=i,
                date=dt,
                original_date=odt,
                amount=amt,
                name=(r.get("Name") or "").strip(),
                description=(r.get("Description") or "").strip(),
                category=(r.get("Category") or "").strip() or "<?>",
                institution=(r.get("Institution Name") or "").strip(),
                account_name=(r.get("Account Name") or "").strip(),
                raw=r,
            )
        )
    return out


def rocket_filter_pnc_spend(r: RocketTx) -> bool:
    # Keep only Rocket rows from PNC "Spend" account (your current chosen scope)
    # Adjust here if your export uses different labels.
    return (r.institution or "").upper() == "PNC" and (r.account_name or "").upper() == "SPEND"


def build_pnc_index(pncs: List[PncTx]) -> Dict[Tuple[date, int], List[int]]:
    """
    Index by (date, abs(amount_cents)).
    We match Rocket -> PNC using absolute amount and date-window.
    """
    ix: Dict[Tuple[date, int], List[int]] = defaultdict(list)
    for j, tx in enumerate(pncs):
        key = (tx.date, cents_key(abs(tx.amount)))
        ix[key].append(j)
    return ix


def candidate_dates(rx: RocketTx, date_window_days: int) -> List[date]:
    base: List[date] = []
    if rx.date:
        base.append(rx.date)
    if rx.original_date and rx.original_date != rx.date:
        base.append(rx.original_date)

    # add +/- window
    cand: List[date] = []
    seen = set()
    for b in base:
        for k in range(-date_window_days, date_window_days + 1):
            d = b + timedelta(days=k)
            if d not in seen:
                seen.add(d)
                cand.append(d)
    return cand


def score_match(rx: RocketTx, px: PncTx) -> float:
    # Description similarity (merchant + desc vs name + desc)
    r_words = norm_words(f"{rx.name} {rx.description}")
    p_words = norm_words(f"{px.merchant} {px.description}")
    sim = jaccard(r_words, p_words)

    # Small boost if merchant/name shares prefix-ish tokens
    bonus = 0.0
    if rx.name and px.merchant:
        if rx.name.upper() in px.merchant.upper() or px.merchant.upper() in rx.name.upper():
            bonus += 0.10

    # prefer exact date over shifted
    date_bonus = 0.05 if rx.date == px.date or (rx.original_date and rx.original_date == px.date) else 0.0

    return sim + bonus + date_bonus


def match_rocket_to_pnc(
    rockets: List[RocketTx],
    pncs: List[PncTx],
    date_window_days: int = 1,
    amount_tolerance_cents: int = 0,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Returns:
      matched_rows (merged),
      unmatched_rocket_rows (rocket-only),
      unmatched_pnc_rows (pnc-only)
    """
    pnc_ix = build_pnc_index(pncs)
    used_pnc = set()

    matched: List[Dict[str, str]] = []
    unmatched_rocket: List[Dict[str, str]] = []

    for rx in rockets:
        # Rocket expenses are positive, PNC spend is negative.
        # Match on absolute value; keep sign sanity in output.
        abs_amt_c = cents_key(abs(rx.amount))

        # Search candidates by date-window and +/- amount tolerance
        candidates: List[int] = []
        for d in candidate_dates(rx, date_window_days):
            for dc in range(-amount_tolerance_cents, amount_tolerance_cents + 1):
                key = (d, abs_amt_c + dc)
                candidates.extend(pnc_ix.get(key, []))

        # remove already-used
        candidates = [c for c in candidates if c not in used_pnc]

        if not candidates:
            out = dict(rx.raw)
            out["_match_reason"] = "no_amount_date_candidate"
            unmatched_rocket.append(out)
            continue

        # choose best candidate by similarity score
        best = None
        best_score = -1.0
        for c in candidates:
            sc = score_match(rx, pncs[c])
            if sc > best_score:
                best_score = sc
                best = c

        assert best is not None
        px = pncs[best]
        used_pnc.add(best)

        # merge row: rocket fields first, then pnc_ fields
        out = dict(rx.raw)
        out["_pnc_date"] = px.date.isoformat()
        out["_pnc_amount"] = f"{px.amount:.2f}"
        out["_pnc_merchant"] = px.merchant
        out["_pnc_description"] = px.description
        out["_pnc_category"] = px.category
        out["_pnc_section"] = px.raw.get("section", "")
        out["_pnc_tx_type"] = px.raw.get("tx_type", "")
        out["_pnc_statement_start"] = px.raw.get("statement_start", "")
        out["_pnc_statement_end"] = px.raw.get("statement_end", "")
        out["_match_reason"] = "amount_date_best_desc"
        out["_match_score"] = f"{best_score:.4f}"
        matched.append(out)

    # unmatched PNC = anything not used
    unmatched_pnc: List[Dict[str, str]] = []
    for j, tx in enumerate(pncs):
        if j not in used_pnc:
            out = dict(tx.raw)
            out["_match_reason"] = "unmatched_in_rocket_scope"
            unmatched_pnc.append(out)

    return matched, unmatched_rocket, unmatched_pnc


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # still write a header-only file if we can infer fields
        if fieldnames is None:
            fieldnames = []
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        return

    if fieldnames is None:
        # stable field order: all keys encountered, in first-seen order
        seen = []
        seen_set = set()
        for r in rows:
            for k in r.keys():
                if k not in seen_set:
                    seen_set.add(k)
                    seen.append(k)
        fieldnames = seen

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_category_confusion(matched_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    c = Counter()
    for r in matched_rows:
        rocket_cat = (r.get("Category") or "").strip() or "<?>"
        pnc_cat = (r.get("_pnc_category") or "").strip() or "<?>"
        c[(rocket_cat, pnc_cat)] += 1

    out = []
    for (rc, pc), n in c.most_common():
        out.append({
            "rocket_category": rc,
            "pnc_category": pc,
            "match_count": str(n),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Match Rocket Money export to PNC parsed/categorized CSV (default: PNC Spend only).")
    ap.add_argument("--rocket", default="notes/tax/inbox/rocket_money_export.csv", help="Rocket Money export CSV path")
    ap.add_argument("--pnc", default="notes/tax/work/parsed/2025/pnc_spend_transactions.categorized.csv", help="PNC categorized CSV path")
    ap.add_argument("--out-dir", default="notes/tax/work/parsed/2025", help="Output directory")
    ap.add_argument("--date-window", type=int, default=1, help="Match within +/- N days (uses Date + Original Date)")
    ap.add_argument("--amount-tol-cents", type=int, default=0, help="Amount tolerance in cents for matching (0 = exact cents)")
    ap.add_argument("--scope", default="pnc_spend", choices=["pnc_spend", "all_rocket"], help="Which Rocket rows to attempt")
    args = ap.parse_args()

    rocket_path = Path(args.rocket)
    pnc_path = Path(args.pnc)
    out_dir = Path(args.out_dir)

    pncs = load_pnc(pnc_path)
    rockets_all = load_rocket(rocket_path)

    # coverage
    pnc_dates = [tx.date for tx in pncs]
    pnc_min = min(pnc_dates).isoformat() if pnc_dates else "<?>"
    pnc_max = max(pnc_dates).isoformat() if pnc_dates else "<?>"

    if args.scope == "pnc_spend":
        rockets = [r for r in rockets_all if rocket_filter_pnc_spend(r)]
    else:
        rockets = rockets_all

    matched, unr, unp = match_rocket_to_pnc(
        rockets=rockets,
        pncs=pncs,
        date_window_days=args.date_window,
        amount_tolerance_cents=args.amount_tol_cents,
    )

    confusion = make_category_confusion(matched)

    # outputs (keep your existing filenames)
    out_matched = out_dir / "rocket_x_pnc.matched.csv"
    out_unr = out_dir / "rocket_x_pnc.unmatched_rocket.csv"
    out_unp = out_dir / "rocket_x_pnc.unmatched_pnc.csv"
    out_conf = out_dir / "rocket_x_pnc.category_confusion.csv"

    write_csv(out_matched, matched)
    write_csv(out_unr, unr)
    write_csv(out_unp, unp)
    write_csv(out_conf, confusion, fieldnames=["rocket_category", "pnc_category", "match_count"])

    # summary
    attempted = len(rockets)
    match_rate = (len(matched) / attempted * 100.0) if attempted else 0.0

    print("wrote:")
    print(f"  {out_matched} rows: {len(matched)}")
    print(f"  {out_unr} rows: {len(unr)}")
    print(f"  {out_unp} rows: {len(unp)}")
    print(f"  {out_conf} rows: {len(confusion)}")
    print()
    print(f"PNC statement coverage: {pnc_min} -> {pnc_max}")
    print(f"Rocket rows total: {len(rockets_all)}")
    print(f"Rocket rows attempted: {attempted} ({'PNC Spend only' if args.scope=='pnc_spend' else 'ALL Rocket'})")
    print(f"Match rate: {len(matched)}/{attempted} ({match_rate:.1f}%)")
    print()

    # top pairings (from confusion)
    print("Top category pairings:")
    for row in confusion[:12]:
        print(f"  {int(row['match_count']):4d}  Rocket={row['rocket_category']}  ->  PNC={row['pnc_category']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
