#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


def normalize_line(s: str) -> str:
    # Normalize whitespace, keep punctuation.
    s = s.replace("\u200b", "")  # zero-width
    s = re.sub(r"\s+", " ", s).strip()
    return s


def line_is_banned(line: str, banned_patterns: list[re.Pattern]) -> bool:
    return any(p.search(line) for p in banned_patterns)


def collapse_consecutive_duplicates(lines: Iterable[str], max_run: int = 2) -> list[str]:
    """
    Keep at most `max_run` consecutive duplicates.
    This kills infinite loops without deleting *all* repetition.
    """
    out: list[str] = []
    prev = None
    run = 0
    for line in lines:
        if line == prev:
            run += 1
            if run <= max_run:
                out.append(line)
        else:
            prev = line
            run = 1
            out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean transcript text (ban phrases + collapse loops).")
    ap.add_argument("in_path", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ban", action="append", default=[], help="Phrase/regex to remove (case-insensitive).")
    ap.add_argument("--max-run", type=int, default=2, help="Max consecutive duplicates to keep.")
    ap.add_argument("--drop-empty", action="store_true", help="Drop empty lines after cleaning.")
    args = ap.parse_args()

    raw = args.in_path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [normalize_line(x) for x in raw]

    banned_patterns = [re.compile(b, re.IGNORECASE) for b in args.ban]
    if banned_patterns:
        lines = [x for x in lines if not line_is_banned(x, banned_patterns)]

    lines = collapse_consecutive_duplicates(lines, max_run=args.max_run)

    if args.drop_empty:
        lines = [x for x in lines if x]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
