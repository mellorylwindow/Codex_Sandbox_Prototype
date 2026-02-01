#!/usr/bin/env python3
"""
Rocket Money x PNC matcher

Goals:
- Run with ZERO args (so existing wrapper scripts keep working).
- Still allow CLI options (year/outdir/etc).
- Ignore tiny parking "probe/preauth" transactions by default (configurable).
- Avoid pandas regex "match groups" warnings (use non-capturing groups).

Outputs (written to OUTDIR):
- rocket_x_pnc.matched.csv
- rocket_x_pnc.unmatched_rocket.csv
- rocket_x_pnc.unmatched_pnc.csv
- rocket_x_pnc.category_confusion.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List, Set

import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _read_csv_smart(path: Path) -> pd.DataFrame:
    # utf-8-sig handles BOM; fall back to utf-8
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def _pick_first(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _clean_merchant(s: str) -> str:
    """
    Normalize merchant-like strings for comparison:
    - lowercase
    - remove long digit runs / card tail noise
    - replace punctuation with spaces
    - collapse whitespace
    """
    s = (s or "").lower()
    s = re.sub(r"x{6,}\d{2,}", " ", s)            # xxxxx...#### tails
    s = re.sub(r"\d{4,}", " ", s)                 # long digit sequences
    s = re.sub(r"[^a-z0-9]+", " ", s)             # punctuation -> space
    s = _collapse_ws(s)
    return s


_STOP = {
    "the", "and", "of", "to", "in", "for", "on", "at", "by",
    "debit", "card", "purchase", "pos", "ach", "visa", "vis",
    "md", "va", "dc", "online", "payment", "recurring"
}


def _token_set(s: str) -> Set[str]:
    s = _clean_merchant(s)
    toks = [t for t in s.split(" ") if t and t not in _STOP and len(t) > 1]
    return set(toks)


def _token_overlap(a: str, b: str) -> int:
    ta = _token_set(a)
    tb = _token_set(b)
    if not ta or not tb:
        return 0
    return len(ta.intersection(tb))


def _find_input_file(outdir: Path, kind: str, explicit: Optional[str]) -> Path:
    """
    Try hard to find inputs without forcing the user to pass args.
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"{kind} file not found: {p}")
        return p

    # Known common names (customize safely without breaking)
    if kind == "rocket":
        candidates = [
            outdir / "rocket_money.csv",
            outdir / "rocket.csv",
            outdir / "rocket_transactions.csv",
            outdir / "rocket_money.transactions.csv",
            outdir / "rocket_money.export.csv",
            outdir / "rocket_money.spend.csv",
            outdir / "rocket_money.normalized.csv",
        ]
        required_cols = {"date", "amount"}
    else:
        candidates = [
            outdir / "pnc.csv",
            outdir / "pnc_transactions.csv",
            outdir / "pnc_statement.csv",
            outdir / "pnc.normalized.csv",
            outdir / "pnc_transactions.normalized.csv",
        ]
        required_cols = {"date", "amount"}

    for c in candidates:
        if c.exists():
            return c

    # Last resort: scan outdir for something plausible
    globbed = sorted(outdir.glob("*.csv"))
    for p in globbed:
        try:
            df = _norm_cols(_read_csv_smart(p).head(5))
        except Exception:
            continue
        if required_cols.issubset(set(df.columns)):
            # Heuristic: rocket tends to have 'category' and 'name'; pnc tends to have 'merchant'
            if kind == "rocket":
                if ("name" in df.columns) or ("category" in df.columns):
                    return p
            else:
                if ("merchant" in df.columns) or ("section" in df.columns) or ("tx_type" in df.columns):
                    return p

    raise SystemExit(
        f"Could not auto-detect {kind} input CSV in {outdir}.\n"
        f"Pass --{kind} /path/to/file.csv explicitly."
    )


@dataclass(frozen=True)
class Cols:
    date: str
    amount: str
    name: str
    category: Optional[str]
    account: Optional[str]


def _resolve_cols(df: pd.DataFrame, kind: str) -> Cols:
    date = _pick_first(df, ["date", "original_date", "transaction_date", "posted_date"])
    amount = _pick_first(df, ["amount", "signed_amount", "amount_num"])
    name = _pick_first(df, ["merchant", "name", "payee", "description", "memo"])
    category = _pick_first(df, ["category", "category_name"])
    account = _pick_first(df, ["account", "account_name", "wallet", "source_account"])

    if not date or not amount or not name:
        raise SystemExit(
            f"{kind}: missing required columns. "
            f"Need date/amount/name-ish. Found: {list(df.columns)}"
        )
    return Cols(date=date, amount=amount, name=name, category=category, account=account)


def _prep_df(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df[cols.date], errors="coerce")
    df["amount_num"] = pd.to_numeric(df[cols.amount], errors="coerce")
    df["name_norm"] = df[cols.name].fillna("").astype(str)
    if cols.category and cols.category in df.columns:
        df["category_norm"] = df[cols.category].fillna("").astype(str)
    else:
        df["category_norm"] = ""
    if cols.account and cols.account in df.columns:
        df["account_norm"] = df[cols.account].fillna("").astype(str)
    else:
        df["account_norm"] = ""
    return df


def _is_parking_probe(name: str, amt: float) -> bool:
    """
    Parking “probe/preauth” heuristic: tiny amount + parking-ish strings.

    Important: use non-capturing group (?:...) to avoid pandas regex group warning.
    """
    if amt is None or pd.isna(amt):
        return False
    if abs(float(amt)) > 1.00:
        return False

    pat = r"(?:DC\s*PARK\*?METER|MCG\s*DOT|SILSPRG\s*PRK|PRK\s*IP)"
    return bool(re.search(pat, (name or ""), flags=re.IGNORECASE))


def _match_one(
    r_row: pd.Series,
    pnc: pd.DataFrame,
    pnc_by_cents: Dict[int, List[int]],
    used_pnc: Set[int],
    day_window: int,
    amount_tolerance: float,
) -> Tuple[Optional[int], float]:
    """
    Return (pnc_index, score). If no match, (None, 0.0)
    """
    r_date = r_row["date_dt"]
    r_amt = r_row["amount_num"]
    r_name = r_row["name_norm"]

    if pd.isna(r_date) or pd.isna(r_amt):
        return None, 0.0

    r_abs = abs(float(r_amt))
    tol = float(amount_tolerance)
    tol_cents = max(0, int(round(tol * 100)))

    base_cents = int(round(r_abs * 100))
    cand_idx: List[int] = []
    for c in range(base_cents - tol_cents, base_cents + tol_cents + 1):
        cand_idx.extend(pnc_by_cents.get(c, []))

    if not cand_idx:
        return None, 0.0

    best_i = None
    best_score = -1e9

    for i in cand_idx:
        if i in used_pnc:
            continue
        p_row = pnc.iloc[i]
        p_date = p_row["date_dt"]
        p_amt = p_row["amount_num"]
        p_name = p_row["name_norm"]

        if pd.isna(p_date) or pd.isna(p_amt):
            continue

        day_diff = abs((p_date - r_date).days)
        if day_diff > day_window:
            continue

        amt_diff = abs(abs(float(p_amt)) - r_abs)
        if amt_diff > tol:
            continue

        overlap = _token_overlap(r_name, p_name)

        # Score weights:
        # - strong preference for merchant overlap
        # - slight preference for exact date
        # - slight preference for exact amount
        score = (overlap * 10.0) - (day_diff * 1.5) - (amt_diff / max(tol, 0.01))

        if score > best_score:
            best_score = score
            best_i = i

    return best_i, float(best_score if best_i is not None else 0.0)


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    now_year = datetime.now().year

    ap = argparse.ArgumentParser()
    # IMPORTANT: do NOT require --year (wrapper compatibility)
    ap.add_argument("--year", type=int, default=now_year, help="Target tax year (default: current year).")
    ap.add_argument("--outdir", default=None, help="Output dir (default: notes/tax/work/parsed/{year}).")
    ap.add_argument("--rocket", default=None, help="Rocket CSV path (optional; auto-detected if omitted).")
    ap.add_argument("--pnc", default=None, help="PNC CSV path (optional; auto-detected if omitted).")
    ap.add_argument("--day-window", type=int, default=2, help="Allowed date delta in days for matching.")
    ap.add_argument("--amount-tolerance", type=float, default=0.05, help="Allowed absolute amount tolerance.")
    ap.add_argument("--no-ignore-parking-probes", action="store_true",
                    help="Do NOT ignore tiny parking probe/preauth transactions.")
    ap.add_argument("--include-cross-year-unmatched-pnc", action="store_true",
                    help="Keep unmatched PNC rows from years other than --year (default drops them).")

    args = ap.parse_args(argv)

    year = int(args.year)
    outdir = Path(args.outdir) if args.outdir else Path(f"notes/tax/work/parsed/{year}")
    outdir.mkdir(parents=True, exist_ok=True)

    rocket_path = _find_input_file(outdir, "rocket", args.rocket)
    pnc_path = _find_input_file(outdir, "pnc", args.pnc)

    rocket_raw = _norm_cols(_read_csv_smart(rocket_path))
    pnc_raw = _norm_cols(_read_csv_smart(pnc_path))

    rocket_cols = _resolve_cols(rocket_raw, "rocket")
    pnc_cols = _resolve_cols(pnc_raw, "pnc")

    rocket = _prep_df(rocket_raw, rocket_cols)
    pnc = _prep_df(pnc_raw, pnc_cols)

    # Coverage range from PNC input
    p_cov_min = pnc["date_dt"].min()
    p_cov_max = pnc["date_dt"].max()

    # Rocket rows "attempted": prefer account containing 'pnc' if present
    rocket_total = len(rocket)
    rocket_attempt = rocket.copy()

    if rocket_cols.account:
        acct = rocket_attempt["account_norm"].str.lower()
        pnc_mask = acct.str.contains("pnc", na=False)
        if pnc_mask.any():
            rocket_attempt = rocket_attempt[pnc_mask].copy()

    # Parking probes: flag and (by default) exclude from matching attempts but keep in unmatched_rocket output
    rocket_attempt["_is_parking_probe"] = rocket_attempt.apply(
        lambda r: _is_parking_probe(r.get("name_norm", ""), r.get("amount_num", float("nan"))),
        axis=1,
    )

    ignored_probes = pd.DataFrame(columns=rocket_attempt.columns)
    if not args.no_ignore_parking_probes:
        ignored_probes = rocket_attempt[rocket_attempt["_is_parking_probe"]].copy()
        rocket_attempt = rocket_attempt[~rocket_attempt["_is_parking_probe"]].copy()

    attempted_n = len(rocket_attempt)

    # Build PNC index by cents for faster candidate lookup
    pnc = pnc.reset_index(drop=True)
    pnc["abs_cents"] = (pnc["amount_num"].abs() * 100.0).round().astype("Int64")
    pnc_by_cents: Dict[int, List[int]] = {}
    for i, v in enumerate(pnc["abs_cents"].tolist()):
        if pd.isna(v):
            continue
        pnc_by_cents.setdefault(int(v), []).append(i)

    # Matching loop
    rocket_attempt = rocket_attempt.reset_index(drop=False).rename(columns={"index": "_rocket_src_index"})
    rocket_attempt["abs_amount"] = rocket_attempt["amount_num"].abs()

    rocket_attempt = rocket_attempt.sort_values(["date_dt", "abs_amount"], ascending=[True, False])

    used_pnc: Set[int] = set()
    match_rows: List[Tuple[int, int, float]] = []

    for _, r in rocket_attempt.iterrows():
        p_i, score = _match_one(
            r_row=r,
            pnc=pnc,
            pnc_by_cents=pnc_by_cents,
            used_pnc=used_pnc,
            day_window=int(args.day_window),
            amount_tolerance=float(args.amount_tolerance),
        )
        if p_i is not None:
            used_pnc.add(p_i)
            match_rows.append((int(r["_rocket_src_index"]), int(p_i), float(score)))

    # Build matched output
    if match_rows:
        m = pd.DataFrame(match_rows, columns=["_rocket_src_index", "_pnc_index", "_match_score"])
    else:
        m = pd.DataFrame(columns=["_rocket_src_index", "_pnc_index", "_match_score"])

    rocket_attempt_src = rocket_raw.reset_index(drop=False).rename(columns={"index": "_rocket_src_index"})
    pnc_src = pnc_raw.reset_index(drop=True)

    matched = (
        m.merge(rocket_attempt_src, on="_rocket_src_index", how="left", suffixes=("", ""))
         .merge(pnc_src.reset_index(drop=False).rename(columns={"index": "_pnc_index"}), on="_pnc_index", how="left",
                suffixes=("_rocket", "_pnc"))
    )

    # Unmatched Rocket:
    matched_rocket_ids = set(m["_rocket_src_index"].tolist()) if len(m) else set()
    unmatched_rocket = rocket_attempt_src[~rocket_attempt_src["_rocket_src_index"].isin(matched_rocket_ids)].copy()

    # Add ignored probes back as unmatched with reason
    if len(ignored_probes):
        ignored_src = rocket_raw.reset_index(drop=False).rename(columns={"index": "_rocket_src_index"})
        # compute ignored set from rocket_attempt_src indexes based on probe detection
        # probe detection ran on rocket_attempt subset, so recompute on ignored_src directly
        ignored_src["_is_parking_probe"] = ignored_src.apply(
            lambda r: _is_parking_probe(r.get(rocket_cols.name, ""), r.get(rocket_cols.amount, float("nan"))),
            axis=1,
        )
        ignored_src = ignored_src[ignored_src["_is_parking_probe"]].copy()
        ignored_src["_match_reason"] = "parking_meter_probe_ignored"
        unmatched_rocket["_match_reason"] = "no_amount_date_candidate"
        unmatched_rocket = pd.concat([unmatched_rocket, ignored_src], ignore_index=True)

    # Override reason when outside coverage
    if "date" in unmatched_rocket.columns or "original_date" in unmatched_rocket.columns:
        # parse date for reason assignment
        date_col = rocket_cols.date
        tmp_dt = pd.to_datetime(unmatched_rocket[date_col], errors="coerce")
        out_mask = pd.Series(False, index=unmatched_rocket.index)
        if pd.notna(p_cov_min) and pd.notna(p_cov_max):
            out_mask = (tmp_dt < p_cov_min) | (tmp_dt > p_cov_max)
        unmatched_rocket.loc[out_mask, "_match_reason"] = "outside_pnc_coverage"

    # Unmatched PNC:
    unmatched_pnc = pnc_src.copy()
    unmatched_pnc["_pnc_index"] = range(len(unmatched_pnc))
    unmatched_pnc = unmatched_pnc[~unmatched_pnc["_pnc_index"].isin(used_pnc)].copy()

    # Filter unmatched PNC to target year by default (fix #2)
    if not args.include_cross_year_unmatched_pnc:
        p_dt = pd.to_datetime(unmatched_pnc[pnc_cols.date], errors="coerce")
        unmatched_pnc = unmatched_pnc[p_dt.dt.year == year].copy()

    # Category confusion (pair counts)
    # We only compute this if both sides have category-ish columns.
    rocket_cat_col = rocket_cols.category if rocket_cols.category in rocket_raw.columns else None
    pnc_cat_col = pnc_cols.category if pnc_cols.category in pnc_raw.columns else None

    if rocket_cat_col and pnc_cat_col and len(matched):
        # matched has suffixes applied; locate the correct columns
        # after merges, rocket columns should be original names (no suffix) and pnc columns are suffixed "_pnc"
        rcat = rocket_cat_col
        pcat = f"{pnc_cat_col}_pnc" if f"{pnc_cat_col}_pnc" in matched.columns else pnc_cat_col

        if rcat in matched.columns and pcat in matched.columns:
            confusion = (
                matched.groupby([rcat, pcat])
                      .size()
                      .reset_index(name="count")
                      .sort_values("count", ascending=False)
            )
        else:
            confusion = pd.DataFrame(columns=["rocket_category", "pnc_category", "count"])
    else:
        confusion = pd.DataFrame(columns=["rocket_category", "pnc_category", "count"])

    # ----------------------------
    # Write outputs
    # ----------------------------
    out_matched = outdir / "rocket_x_pnc.matched.csv"
    out_unr = outdir / "rocket_x_pnc.unmatched_rocket.csv"
    out_unp = outdir / "rocket_x_pnc.unmatched_pnc.csv"
    out_conf = outdir / "rocket_x_pnc.category_confusion.csv"

    matched.to_csv(out_matched, index=False, encoding="utf-8", lineterminator="\n")
    unmatched_rocket.to_csv(out_unr, index=False, encoding="utf-8", lineterminator="\n")
    unmatched_pnc.to_csv(out_unp, index=False, encoding="utf-8", lineterminator="\n")
    confusion.to_csv(out_conf, index=False, encoding="utf-8", lineterminator="\n")

    # ----------------------------
    # Summary
    # ----------------------------
    print("wrote:")
    print(f"  {out_matched} rows: {len(matched)}")
    print(f"  {out_unr} rows: {len(unmatched_rocket)}")
    print(f"  {out_unp} rows: {len(unmatched_pnc)}")
    print(f"  {out_conf} rows: {len(confusion)}")
    print("")
    if pd.notna(p_cov_min) and pd.notna(p_cov_max):
        print(f"PNC statement coverage: {p_cov_min.date()} -> {p_cov_max.date()}")
    else:
        print("PNC statement coverage: <unknown> (date parse issue)")

    print(f"Rocket rows total: {rocket_total}")
    print(f"Rocket rows attempted: {attempted_n} (PNC account preferred when detectable)")
    if attempted_n:
        print(f"Match rate: {len(m)}/{attempted_n} ({(len(m)/attempted_n)*100:.1f}%)")
    else:
        print("Match rate: 0/0 (0.0%)")

    if "_match_reason" in unmatched_rocket.columns:
        print("\nUnmatched Rocket reasons:")
        print(unmatched_rocket["_match_reason"].fillna("<NA>").value_counts().to_string())

    # Top category pairings
    if len(confusion):
        print("\nTop category pairings:")
        for _, row in confusion.head(12).iterrows():
            print(f"{int(row['count']):6d}  Rocket={row.iloc[0]}  ->  PNC={row.iloc[1]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
