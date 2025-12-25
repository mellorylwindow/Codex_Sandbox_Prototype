#!/usr/bin/env python3
"""
Import "compiled prompts" from text/markdown into drafts.jsonl so they can be ranked.

Accepted input shapes (auto-detected):
- Plain text: one prompt per line
- Markdown bullets: "- prompt ..." or "* prompt ..."
- Numbered list: "1. prompt ..."
- Headings + paragraph blocks (we keep paragraphs as a single prompt)

Outputs:
- out/tiktok_prompts/compiled/drafts.jsonl

Notes:
- Deterministic IDs (content-hash) so re-importing won't create duplicates downstream.
- Robust to weird characters; always reads with errors='replace'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Optional


BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)\s*$")
NUM_RE = re.compile(r"^\s*\d+\.\s+(.*)\s*$")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)\s*$")


def stable_id(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return h[:12]


def normalize_prompt(s: str) -> str:
    s = (s or "").strip()
    # collapse internal whitespace but keep intentional newlines minimal
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def split_blocks(text: str) -> List[str]:
    """
    Strategy:
    - If we see lots of bullets/numbered items, parse linewise.
    - Otherwise, treat blank-line separated paragraphs as prompts.
    """
    lines = text.splitlines()
    bulletish = 0
    nonempty = 0
    for ln in lines:
        if ln.strip():
            nonempty += 1
        if BULLET_RE.match(ln) or NUM_RE.match(ln):
            bulletish += 1

    # If at least 25% of nonempty lines look like list items, parse listwise
    if nonempty and (bulletish / nonempty) >= 0.25:
        out: List[str] = []
        for ln in lines:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            m = BULLET_RE.match(ln) or NUM_RE.match(ln)
            if m:
                item = normalize_prompt(m.group(1))
                if item:
                    out.append(item)
            else:
                # allow raw lines too (if people mix formats)
                item = normalize_prompt(ln)
                if item:
                    out.append(item)
        return out

    # Paragraph mode
    paras = re.split(r"\n\s*\n+", text.strip())
    out = []
    for p in paras:
        p = normalize_prompt(p.replace("\n", " ").strip())
        if p:
            out.append(p)
    return out


def load_prompts_from_paths(paths: List[str]) -> List[str]:
    """
    Paths can be files or directories. Directories are scanned for .txt/.md files.
    """
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            continue
        if p.is_dir():
            for ext in ("*.txt", "*.md", "*.markdown"):
                files.extend(sorted(p.rglob(ext)))
        else:
            files.append(p)

    # unique
    seen = set()
    uniq: List[Path] = []
    for f in files:
        key = str(f.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)

    prompts: List[str] = []
    for f in uniq:
        txt = f.read_text(encoding="utf-8", errors="replace")
        blocks = split_blocks(txt)
        prompts.extend(blocks)

    # normalize + de-dupe by normalized text
    out: List[str] = []
    seen_txt = set()
    for p in prompts:
        p2 = normalize_prompt(p)
        if not p2:
            continue
        key = p2.lower()
        if key in seen_txt:
            continue
        seen_txt.add(key)
        out.append(p2)

    return out


def write_drafts_jsonl(prompts: List[str], out_path: Path, source_name: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for p in prompts:
        rid = stable_id(p)
        rows.append(
            {
                "id": rid,
                "title": "",
                "prompt_core": p,
                "extracted_text": "",
                "tags": [],
                "source_name": source_name,
            }
        )

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Import compiled prompts -> drafts.jsonl")
    ap.add_argument(
        "--in",
        dest="inputs",
        nargs="+",
        default=["notes/tiktok/compiled_in/compiled_prompts.txt"],
        help="One or more input files/folders containing compiled prompts (.txt/.md)",
    )
    ap.add_argument(
        "--out",
        default="out/tiktok_prompts/compiled/drafts.jsonl",
        help="Output drafts JSONL path",
    )
    ap.add_argument(
        "--source-name",
        default="compiled_list",
        help="Label stored on each record",
    )
    args = ap.parse_args()

    prompts = load_prompts_from_paths(args.inputs)
    if not prompts:
        print("No prompts found in inputs. Did you paste the list into the file?")
        return 2

    out_path = Path(args.out)
    n = write_drafts_jsonl(prompts, out_path, args.source_name)

    print(f"OK: imported {n} prompts")
    print(f"- {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
