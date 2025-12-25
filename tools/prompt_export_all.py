#!/usr/bin/env python
"""
prompt_export_all.py

Merges prompt "draft" JSONL files into one deduped export:
- out/tiktok_prompts/ALL_PROMPTS.txt  (paste-friendly)
- out/tiktok_prompts/ALL_PROMPTS.jsonl (machine-friendly)

Usage:
  python tools/prompt_export_all.py \
    --drafts out/tiktok_prompts/drafts.jsonl out/tiktok_prompts/chat/drafts.jsonl out/tiktok_prompts/compiled/drafts.jsonl \
    --out out/tiktok_prompts/ALL_PROMPTS.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple


def _iter_draft_paths(drafts: List[str]) -> List[Path]:
    paths: List[Path] = []
    for d in drafts:
        p = Path(d)
        if p.is_dir():
            # Support passing a folder like out/tiktok_prompts
            cand = p / "drafts.jsonl"
            if cand.exists():
                paths.append(cand)
        else:
            if p.exists():
                paths.append(p)
    # de-dupe while preserving order
    seen = set()
    out = []
    for p in paths:
        s = str(p.resolve()).lower()
        if s in seen:
            continue
        seen.add(s)
        out.append(p)
    return out


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # tolerate garbage lines
                continue


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"\s+", " ", t)
    # remove most punctuation for fingerprinting
    t = re.sub(r"[`*_#>\[\]\(\)\{\}:;,\"!?]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _fingerprint(text: str) -> str:
    t = _normalize(text)
    return hashlib.sha1(t.encode("utf-8", errors="replace")).hexdigest()


def _pick_text(row: Dict[str, Any]) -> str:
    # prefer prompt_core if present, else fall back to any common fields
    for k in ("prompt_core", "prompt", "text", "content", "extracted_text"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--drafts",
        nargs="+",
        required=False,
        default=[
            "out/tiktok_prompts/drafts.jsonl",
            "out/tiktok_prompts/chat/drafts.jsonl",
            "out/tiktok_prompts/compiled/drafts.jsonl",
        ],
        help="Draft JSONL files or folders containing drafts.jsonl (multiple allowed).",
    )
    ap.add_argument(
        "--out",
        default="out/tiktok_prompts/ALL_PROMPTS.txt",
        help="Output text file.",
    )
    args = ap.parse_args()

    out_txt = Path(args.out)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_txt.with_suffix(".jsonl")

    draft_paths = _iter_draft_paths(args.drafts)
    if not draft_paths:
        raise SystemExit("No draft files found. Pass --drafts with existing paths.")

    merged: List[Tuple[str, Dict[str, Any]]] = []
    seen_fp = set()

    for p in draft_paths:
        for row in _read_jsonl(p):
            text = _pick_text(row)
            if not text:
                continue
            fp = _fingerprint(text)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)

            # record minimal provenance
            row_out = dict(row)
            row_out["_source_file"] = str(p).replace("\\", "/")
            row_out["_fingerprint"] = fp
            row_out["_export_text"] = text

            merged.append((text, row_out))

    # stable sort: longest first tends to group “real prompts” above fragments
    merged.sort(key=lambda x: (-len(x[0]), x[0].lower()))

    # write txt (paste-friendly)
    with out_txt.open("w", encoding="utf-8", newline="\n") as f:
        for i, (text, _) in enumerate(merged, 1):
            f.write(f"{i}. {text}\n\n")

    # write jsonl
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for _, row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"OK: exported {len(merged)} prompts")
    print(f"- {out_txt}")
    print(f"- {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
