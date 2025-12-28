#!/usr/bin/env python3
"""
Patch NFCU statement period into existing _meta/<sha>.json files.

Reads:
- out/tax_text/nfcu/2025/*.txt
- out/tax_text/nfcu/2025/_meta/<sha>.json

Finds in text (OCR tolerant):
- "Statement Period" label if present
- otherwise finds the first mm/dd/yy - mm/dd/yy range with slashes

Writes (if --write):
- Adds:
  statement_period_raw
  statement_period_start (YYYY-MM-DD)
  statement_period_end   (YYYY-MM-DD)
  statement_period_source
  statement_period_text_rel

Also optionally writes an index JSONL (--out-index).

Safe default: DRY-RUN unless --write is provided.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List


# OCR can introduce:
# - en dash / em dash instead of hyphen
# - extra spaces between digits "0 5/2 2/2 5"
# - "to" or other separators occasionally
DASH = r"(?:-|–|—|to|through)"

# Match date with optional whitespace between every digit and separator.
DATE_OCR = r"(\d\s*\d\s*/\s*\d\s*\d\s*/\s*\d\s*\d)"

# Try to match the date range *near* the "Statement Period" label first.
PERIOD_NEAR_LABEL_RE = re.compile(
    rf"Statement\s*Period[\s\S]{{0,120}}?{DATE_OCR}\s*{DASH}\s*{DATE_OCR}",
    re.IGNORECASE,
)

# Fallback: any slash-date-range anywhere (still unlikely to collide with txn rows,
# because txn dates are usually MM-DD-YY without slashes).
PERIOD_ANYWHERE_RE = re.compile(
    rf"{DATE_OCR}\s*{DASH}\s*{DATE_OCR}",
    re.IGNORECASE,
)


def _compact_date(s: str) -> str:
    """Remove all whitespace from something like '0 5 / 2 2 / 2 5' -> '05/22/25'."""
    return re.sub(r"\s+", "", s.strip())


def parse_mmddyy(s: str) -> str:
    dt = datetime.strptime(s.strip(), "%m/%d/%y")
    return dt.date().isoformat()


def extract_statement_period(txt: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Returns (raw_start, raw_end, iso_start, iso_end) or None.
    """
    m = PERIOD_NEAR_LABEL_RE.search(txt)
    if not m:
        m = PERIOD_ANYWHERE_RE.search(txt)
    if not m:
        return None

    raw_start = _compact_date(m.group(1))
    raw_end   = _compact_date(m.group(2))

    try:
        iso_start = parse_mmddyy(raw_start)
        iso_end   = parse_mmddyy(raw_end)
    except Exception:
        return None

    return raw_start, raw_end, iso_start, iso_end


@dataclass
class Result:
    sha: str
    ok: bool
    statement_period_raw: str = ""
    iso_start: str = ""
    iso_end: str = ""
    note: str = ""


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-dir", required=True, help="Folder with extracted .txt files")
    ap.add_argument("--meta-dir", required=True, help="Folder with _meta/<sha>.json")
    ap.add_argument("--out-index", default="", help="Optional JSONL output path")
    ap.add_argument("--write", action="store_true", help="Actually patch meta json files (default is dry-run)")
    args = ap.parse_args()

    text_dir = Path(args.text_dir)
    meta_dir = Path(args.meta_dir)
    out_index = Path(args.out_index) if args.out_index else None

    if not text_dir.exists():
        print(f"ERROR: text dir not found: {text_dir}")
        return 2
    if not meta_dir.exists():
        print(f"ERROR: meta dir not found: {meta_dir}")
        return 2

    txt_files = sorted(text_dir.glob("*.txt"))
    results: List[Result] = []

    matched = 0
    patched = 0
    no_match = 0
    missing_meta = 0
    errors = 0

    for txt_path in txt_files:
        sha = txt_path.stem
        meta_path = meta_dir / f"{sha}.json"
        if not meta_path.exists():
            missing_meta += 1
            results.append(Result(sha=sha, ok=False, note="missing_meta"))
            continue

        try:
            txt = txt_path.read_text(encoding="utf-8", errors="replace")
            found = extract_statement_period(txt)
            if not found:
                no_match += 1
                results.append(Result(sha=sha, ok=False, note="no_match"))
                continue

            raw_start, raw_end, iso_start, iso_end = found
            matched += 1

            statement_period_raw = f"{raw_start} - {raw_end}"
            r = Result(
                sha=sha,
                ok=True,
                statement_period_raw=statement_period_raw,
                iso_start=iso_start,
                iso_end=iso_end,
                note="matched",
            )
            results.append(r)

            if args.write:
                m = read_json(meta_path)
                m["statement_period_raw"] = statement_period_raw
                m["statement_period_start"] = iso_start
                m["statement_period_end"] = iso_end
                m["statement_period_source"] = "extracted_text"
                m["statement_period_text_rel"] = str(txt_path).replace("\\", "/")
                write_json(meta_path, m)
                patched += 1

        except Exception as e:
            errors += 1
            results.append(Result(sha=sha, ok=False, note=f"error: {e}"))

    if out_index:
        out_index.parent.mkdir(parents=True, exist_ok=True)
        with out_index.open("w", encoding="utf-8", newline="\n") as f:
            for r in results:
                f.write(json.dumps({
                    "sha256": r.sha,
                    "ok": r.ok,
                    "statement_period_raw": r.statement_period_raw,
                    "statement_period_start": r.iso_start,
                    "statement_period_end": r.iso_end,
                    "meta_path": str((meta_dir / f"{r.sha}.json")).replace("\\", "/"),
                    "txt_path": str((text_dir / f"{r.sha}.txt")).replace("\\", "/"),
                    "note": r.note,
                }, ensure_ascii=False) + "\n")

    print("---- NFCU Statement Period Patch ----")
    print("TXT files:    ", len(txt_files))
    print("Matched:      ", matched)
    print("Patched:      ", patched, "(WRITE)" if args.write else "(DRY-RUN)")
    print("No match:     ", no_match)
    print("Missing meta: ", missing_meta)
    print("Errors:       ", errors)

    # show a few examples
    examples = [r for r in results if r.ok][:3]
    if examples:
        print("\nExamples:")
        for r in examples:
            print(f"- {r.sha}: {r.iso_start} -> {r.iso_end}")

    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
