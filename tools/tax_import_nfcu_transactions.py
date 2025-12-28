from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


DEFAULT_IN_DIR = Path("notes/tax/intake/bank/2025/NFCU")
OUT_PATH = Path("notes/tax/work/parsed/2025/nfcu_transactions.normalized.csv")


def _money_to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return None if pd.isna(x) else float(x)
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    neg = False
    # handle parentheses negatives
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").strip()
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return None


def _pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _load_csv(path: Path) -> pd.DataFrame:
    # try a couple encodings/dialects
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python")
        except Exception:
            continue
    # last resort
    return pd.read_csv(path, engine="python")


def _normalize_one(path: Path) -> pd.DataFrame:
    df = _load_csv(path)
    df = df.copy()

    date_col = _pick_col(df.columns, ["Date", "Transaction Date", "Posting Date", "Posted Date"])
    desc_col = _pick_col(df.columns, ["Description", "Transaction Description", "Payee", "Merchant", "Details"])

    amt_col = _pick_col(df.columns, ["Amount", "Transaction Amount"])
    debit_col = _pick_col(df.columns, ["Debit"])
    credit_col = _pick_col(df.columns, ["Credit"])

    if date_col is None:
        raise SystemExit(f"[NFCU import] Can't find a date column in: {path.name} cols={list(df.columns)}")

    if desc_col is None:
        # keep something usable
        desc_col = df.columns[0]

    # compute amount
    if amt_col is not None:
        amt = df[amt_col].map(_money_to_float)
    elif debit_col is not None or credit_col is not None:
        deb = df[debit_col].map(_money_to_float) if debit_col else 0
        cred = df[credit_col].map(_money_to_float) if credit_col else 0
        # debits should be negative in unified spend
        amt = (pd.Series(cred).fillna(0) - pd.Series(deb).fillna(0))
    else:
        raise SystemExit(f"[NFCU import] Can't find amount/debit/credit columns in: {path.name} cols={list(df.columns)}")

    out = pd.DataFrame({
        "source": "NFCU",
        "source_file": path.name,
        "row_index": range(len(df)),
        "date": pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str),
        "amount": pd.to_numeric(amt, errors="coerce"),
        "merchant": df[desc_col].fillna("").astype(str),
        "description": df[desc_col].fillna("").astype(str),
        "section": None,
        "category": None,
    })

    out["section"] = out["amount"].apply(lambda x: "withdrawals" if pd.notna(x) and x < 0 else "deposits")
    out["category"] = "Uncategorized"

    out = out[out["date"].notna() & (out["date"] != "NaT")].copy()
    out = out[out["amount"].notna()].copy()
    return out


def main() -> int:
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        args = [DEFAULT_IN_DIR]

    files = []
    for a in args:
        if a.is_dir():
            files.extend(sorted(a.glob("*.csv")))
        elif a.is_file():
            files.append(a)

    if not files:
        print(f"No NFCU CSV files found under: {DEFAULT_IN_DIR}")
        return 0

    frames = []
    for f in files:
        frames.append(_normalize_one(f))

    out = pd.concat(frames, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Using NFCU files: {len(files)}")
    print(f"wrote:\n  {OUT_PATH} rows: {len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
