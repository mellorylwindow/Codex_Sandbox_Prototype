#!/usr/bin/env python3
"""
tax_parse_lineitems.py — v2 (works with receipt_asset manifest + sha256-keyed text)

Inputs:
- manifest.jsonl rows with:
    sha256, kind=receipt_asset, status=ingested|duplicate, dest_rel (maybe missing), src_rel
- extracted text at:
    tax_intake/30_extracted/text/<sha256>.txt

Outputs:
- tax_intake/30_extracted/lines/<sha256>.json

Design goals:
- schema tolerant
- always emits a per-doc json (even if no line-items found)
- keeps parsing simple + conservative (you can upgrade heuristics later)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class Row:
    sha256: str
    status: str
    kind: str
    batch: str
    original_name: str
    dest_rel: str
    src_rel: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Row":
        return Row(
            sha256=str(d.get("sha256", "")).strip(),
            status=str(d.get("status", "")).strip(),
            kind=str(d.get("kind", "")).strip(),
            batch=str(d.get("batch", "")).strip(),
            original_name=str(d.get("original_name", "")).strip(),
            dest_rel=str(d.get("dest_rel", "")).strip(),
            src_rel=str(d.get("src_rel", "")).strip(),
        )


def read_manifest(path: Path) -> List[Row]:
    rows: List[Row] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(Row.from_dict(json.loads(line)))
    return rows


_money_re = re.compile(r"(?<!\w)\$?\s*([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)\.(\d{2})(?!\w)")
_date_re = re.compile(
    r"\b((?:20)?\d{2})[-/](\d{1,2})[-/](\d{1,2})\b"   # 2025-12-25 or 25-12-25 etc
    r"|\b(\d{1,2})[-/](\d{1,2})[-/](?:20)?(\d{2})\b" # 12/25/25 or 12/25/2025
)

def norm_amount(m: re.Match) -> float:
    whole = m.group(1).replace(",", "")
    cents = m.group(2)
    return float(f"{whole}.{cents}")


def parse_vendor(text: str) -> str:
    # first “real” line that isn’t just noise
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        if len(line) < 2:
            continue
        # skip lines that are mostly digits or just totals
        if sum(ch.isdigit() for ch in line) > max(6, len(line) * 0.6):
            continue
        return line[:80]
    return ""


def parse_dates(text: str) -> List[str]:
    found: List[str] = []
    for m in _date_re.finditer(text):
        if m.group(1):  # yyyy-mm-dd
            y = int(m.group(1))
            mo = int(m.group(2))
            d = int(m.group(3))
        else:           # mm-dd-yy(yy)
            mo = int(m.group(4))
            d = int(m.group(5))
            yy = int(m.group(6))
            y = 2000 + yy if yy < 100 else yy
        try:
            dt = datetime(y, mo, d)
            found.append(dt.date().isoformat())
        except Exception:
            continue
    # de-dupe preserving order
    out: List[str] = []
    for x in found:
        if x not in out:
            out.append(x)
    return out


def parse_amounts(text: str) -> List[float]:
    vals = [norm_amount(m) for m in _money_re.finditer(text)]
    # de-dupe preserving order
    out: List[float] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def guess_total(text: str, amounts: List[float]) -> Optional[float]:
    # conservative: if “total” line exists, pick the last money value near it
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "total" in line.lower():
            # search line + next 2 lines for amounts
            window = "\n".join(lines[i:i+3])
            window_vals = [norm_amount(m) for m in _money_re.finditer(window)]
            if window_vals:
                return window_vals[-1]
    # fallback: biggest amount (often total)
    if amounts:
        return max(amounts)
    return None


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Parse extracted text into line-items per document.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--text-dir", default="tax_intake/30_extracted/text")
    ap.add_argument("--out-dir", default="tax_intake/30_extracted/lines")
    ap.add_argument("--only-canonical", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    repo = repo_root()
    man = Path(args.manifest)
    if not man.is_absolute():
        man = (repo / man).resolve()

    text_dir = (repo / args.text_dir).resolve()
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(man)
    wrote_docs = 0
    skipped = 0
    missing_text = 0

    for r in rows:
        if not r.sha256:
            continue
        if args.only_canonical and r.status.lower() == "duplicate":
            continue

        txt_path = text_dir / f"{r.sha256}.txt"
        out_path = out_dir / f"{r.sha256}.json"

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        if not txt_path.exists():
            missing_text += 1
            write_json(out_path, {
                "ok": False,
                "error": "Missing extracted text file",
                "sha256": r.sha256,
                "status": r.status,
                "kind": r.kind,
                "batch": r.batch,
                "original_name": r.original_name,
                "text_path": str(txt_path),
                "parsed_at": utc_now_iso(),
                "vendor": "",
                "dates": [],
                "amounts": [],
                "total_guess": None,
                "line_items": [],
            })
            wrote_docs += 1
            continue

        text = txt_path.read_text(encoding="utf-8", errors="replace")
        vendor = parse_vendor(text)
        dates = parse_dates(text)
        amounts = parse_amounts(text)
        total_guess = guess_total(text, amounts)

        # For now: we store amounts as “candidates” and leave actual itemization for later.
        write_json(out_path, {
            "ok": True,
            "error": None,
            "sha256": r.sha256,
            "status": r.status,
            "kind": r.kind,
            "batch": r.batch,
            "original_name": r.original_name,
            "text_path": str(txt_path),
            "parsed_at": utc_now_iso(),
            "vendor": vendor,
            "dates": dates,
            "amounts": amounts,
            "total_guess": total_guess,
            "line_items": [],  # later: real item parsing
        })
        wrote_docs += 1

        if args.debug and wrote_docs <= 3:
            print(f"[debug] {r.sha256[:12]} vendor={vendor!r} dates={dates[:2]} total={total_guess}")

    print(f"Wrote line-items for {wrote_docs} documents into: {out_dir}")
    if args.debug:
        print(f"[debug] skipped_existing={skipped} missing_text={missing_text} manifest_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
