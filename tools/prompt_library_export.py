#!/usr/bin/env python3
"""
tools/prompt_library_export.py

Merge drafts.jsonl files into a single numbered text export + a quick report.

Usage:
  python tools/prompt_library_export.py \
    --drafts out/tiktok_prompts/drafts.jsonl out/tiktok_prompts/chat/drafts.jsonl out/tiktok_prompts/compiled/drafts.jsonl \
    --out-dir out/prompt_library \
    --min-len 60 \
    --ban "patch mode" --ban "make these changes" --ban "create `"

Outputs:
  - <out-dir>/ALL_PROMPTS.txt
  - <out-dir>/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Item:
    id: str
    text: str
    title: str
    source: str


def _iter_draft_files(paths: List[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_file():
            yield p
        elif p.is_dir():
            # common patterns
            for cand in sorted(p.rglob("drafts.jsonl")):
                yield cand


def _read_jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def _pick_text(rec: dict) -> Tuple[str, str]:
    """
    Returns (title, text).
    We prefer prompt_core, fallback to prompt, text, extracted_text.
    """
    title = str(rec.get("title") or "").strip()
    for k in ("prompt_core", "prompt", "text", "extracted_text"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return title, v.strip()
    return title, ""


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", nargs="+", required=True, help="drafts.jsonl file(s) or folder(s).")
    ap.add_argument("--out-dir", default="out/prompt_library", help="Output folder.")
    ap.add_argument("--min-len", type=int, default=60, help="Drop items shorter than this.")
    ap.add_argument("--ban", action="append", default=[], help="Ban phrase (case-insensitive). Repeatable.")
    ap.add_argument("--keep-all", action="store_true", help="Do not drop short items (still flagged in report).")
    args = ap.parse_args(argv)

    draft_paths = [Path(x) for x in args.drafts]
    files = list(_iter_draft_files(draft_paths))
    if not files:
        print("No draft files found.")
        return 2

    ban_phrases = [b.lower().strip() for b in (args.ban or []) if b.strip()]

    items: List[Item] = []
    seen = set()

    for f in files:
        for rec in _read_jsonl(f):
            title, text = _pick_text(rec)
            if not text:
                continue
            nid = str(rec.get("id") or "").strip() or _norm(text)[:24]
            key = (nid, _norm(text))
            if key in seen:
                continue
            seen.add(key)
            items.append(Item(id=nid, text=text, title=title, source=str(f).replace("\\", "/")))

    # filtering
    kept: List[Item] = []
    dropped_short = 0
    dropped_ban = 0

    for it in items:
        tlow = it.text.lower()
        if ban_phrases and any(bp in tlow for bp in ban_phrases):
            dropped_ban += 1
            continue
        if (not args.keep_all) and len(it.text) < args.min_len:
            dropped_short += 1
            continue
        kept.append(it)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_txt = out_dir / "ALL_PROMPTS.txt"
    report_md = out_dir / "REPORT.md"

    # write ALL_PROMPTS.txt
    lines: List[str] = []
    for i, it in enumerate(kept, start=1):
        lines.append(f"{i}. {it.text}")
        lines.append("")  # blank line between prompts
    all_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")

    # report
    lengths = [len(it.text) for it in kept]
    short_flags = [it for it in items if len(it.text) < args.min_len][:20]
    word_counter = Counter(re.findall(r"[A-Za-z']{3,}", " ".join(it.text for it in kept).lower()))

    r: List[str] = []
    r.append("# Prompt Library Report")
    r.append("")
    r.append(f"- sources scanned: {len(files)}")
    r.append(f"- raw items: {len(items)}")
    r.append(f"- kept: {len(kept)}")
    r.append(f"- dropped (ban phrases): {dropped_ban}")
    r.append(f"- dropped (short): {dropped_short} (min_len={args.min_len})")
    r.append("")
    if lengths:
        r.append(f"- kept length: min={min(lengths)}  median={sorted(lengths)[len(lengths)//2]}  max={max(lengths)}")
        r.append("")
    r.append("## Top words (rough signal)")
    r.append("")
    for w, c in word_counter.most_common(25):
        r.append(f"- {w}: {c}")
    r.append("")
    r.append("## Examples of short / low-signal items (from raw, before drops)")
    r.append("")
    for it in short_flags:
        r.append(f"- `{it.text}`")
    r.append("")
    report_md.write_text("\n".join(r), encoding="utf-8", newline="\n")

    print(f"OK: wrote merged export")
    print(f"- {all_txt}")
    print(f"- {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
