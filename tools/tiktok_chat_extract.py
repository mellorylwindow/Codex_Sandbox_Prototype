#!/usr/bin/env python3
"""
TikTok Prompt Project — Chat Export Extractor (offline-first)

Reads ChatGPT export JSON (or folder of JSON files) and extracts "prompt-like" snippets
from past chats, then writes a JSONL compatible with your compiler.

Outputs records shaped similarly to image-ingest:
- id
- source_name / source_path
- extracted_text

So you can run:
  python tools/tiktok_prompt_compile.py --in out/tiktok_prompts/chat_prompts.jsonl --outdir out/tiktok_prompts/chat

Heuristics:
- Pull blocks around lines containing: "prompt", "tiktok", "hook", "cta", "voiceover", "script"
- Also catch patterns like: Prompt — “...”
- Configurable with include terms file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_TERMS = [
    "prompt",
    "tiktok",
    "hook",
    "cta",
    "voiceover",
    "on-screen",
    "onscreen",
    "script",
    "series",
    "caption",
]


@dataclass(frozen=True)
class Extracted:
    id: str
    source_path: str
    source_name: str
    created_at_utc: str
    extracted_text: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "source_path": self.source_path,
                "source_name": self.source_name,
                "created_at_utc": self.created_at_utc,
                "extracted_text": self.extracted_text,
            },
            ensure_ascii=False,
        )


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def read_terms(path: Optional[Path]) -> list[str]:
    if not path:
        return DEFAULT_TERMS
    if not path.exists():
        return DEFAULT_TERMS
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        terms.append(t)
    return terms or DEFAULT_TERMS


def iter_json_files(p: Path) -> Iterable[Path]:
    if p.is_file() and p.suffix.lower() == ".json":
        yield p
        return
    if p.is_dir():
        for f in sorted(p.rglob("*.json")):
            yield f


def _collect_texts_from_any_json(obj: Any) -> list[str]:
    """
    Best-effort: walk JSON and collect strings that look like message content.
    Works across multiple export schemas.
    """
    out: list[str] = []

    def walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            # Avoid pulling giant base64 blobs etc.
            if len(x) < 20000:
                out.append(x)
            return
        if isinstance(x, dict):
            for k, v in x.items():
                # common keys
                if k in {"content", "text", "message", "body", "snippet"}:
                    walk(v)
                else:
                    walk(v)
            return
        if isinstance(x, list):
            for it in x:
                walk(it)

    walk(obj)
    return out


def extract_blocks(texts: list[str], terms: list[str], window: int = 2) -> list[str]:
    """
    From a list of strings, produce extracted blocks that contain any term.
    window = number of neighboring lines to include around matches.
    """
    term_re = re.compile(r"(" + "|".join(re.escape(t) for t in terms) + r")", re.IGNORECASE)

    blocks: list[str] = []
    for t in texts:
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln.strip() for ln in t.split("\n")]
        lines = [ln for ln in lines if ln]

        hits = [i for i, ln in enumerate(lines) if term_re.search(ln)]
        if not hits:
            continue

        # merge hit windows
        spans: list[tuple[int, int]] = []
        for i in hits:
            a = max(0, i - window)
            b = min(len(lines), i + window + 1)
            spans.append((a, b))
        spans.sort()

        merged: list[tuple[int, int]] = []
        for a, b in spans:
            if not merged or a > merged[-1][1]:
                merged.append((a, b))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))

        for a, b in merged:
            block = "\n".join(lines[a:b]).strip()
            if block:
                blocks.append(block)

    # de-dupe while preserving order
    seen = set()
    uniq: list[str] = []
    for b in blocks:
        key = hashlib.sha256(b.encode("utf-8", errors="replace")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(b)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract TikTok prompts from chat export JSON.")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to conversations.json or a folder of JSON files")
    ap.add_argument("--out", dest="out_path", default="out/tiktok_prompts/chat_prompts.jsonl", help="Output JSONL path")
    ap.add_argument("--terms", dest="terms_path", default="notes/tiktok/chat_extract_terms.txt", help="Terms file (one per line)")
    ap.add_argument("--window", type=int, default=2, help="Lines of context around matches")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    terms_path = Path(args.terms_path) if args.terms_path else None

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        return 2

    terms = read_terms(terms_path)

    records: list[Extracted] = []
    stamp = now_utc()

    files = list(iter_json_files(in_path))
    if not files:
        print(f"ERROR: no .json files found at: {in_path}")
        return 2

    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            print(f"SKIP (bad json): {f} :: {e}")
            continue

        texts = _collect_texts_from_any_json(obj)
        blocks = extract_blocks(texts, terms=terms, window=args.window)

        for idx, block in enumerate(blocks):
            rid = sha_id(str(f), str(idx), block)
            records.append(
                Extracted(
                    id=rid,
                    source_path=str(f),
                    source_name=f.name,
                    created_at_utc=stamp,
                    extracted_text=block,
                )
            )

    # deterministic order
    records_sorted = sorted(records, key=lambda r: (r.source_name, r.id))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fp:
        for r in records_sorted:
            fp.write(r.to_json() + "\n")

    print(f"OK: extracted {len(records_sorted)} blocks")
    print(f"- {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
