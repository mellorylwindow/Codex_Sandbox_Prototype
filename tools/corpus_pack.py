# tools/corpus_pack.py
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if t.strip()]


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chunk_chars <= 0:
        return [text]
    if overlap < 0:
        overlap = 0
    step = max(1, chunk_chars - overlap)

    chunks: List[str] = []
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
    return chunks


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    meta: dict


def _load_chunks_from_my_messages(
    jsonl_path: Path,
    *,
    chunk_chars: int,
    overlap: int,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    base = jsonl_path.name

    for i, row in enumerate(_read_jsonl(jsonl_path)):
        role = (row.get("role") or row.get("speaker") or "").lower()
        if role and role not in ("user", "me", "jimmy"):
            continue

        text = (row.get("text") or row.get("content") or "").strip()
        if not text:
            continue

        meta = {}
        for k in ("title", "conversation_title", "conversation_id", "created_at", "timestamp", "source_path"):
            if k in row and row[k]:
                meta[k] = row[k]

        parts = _chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
        for j, part in enumerate(parts):
            cid = f"{base}:msg{i:06d}:c{j:02d}"
            chunks.append(Chunk(chunk_id=cid, text=part, source=base, meta=meta))

    return chunks


def _build_idf(chunks: List[Chunk]) -> Dict[str, float]:
    df: Dict[str, int] = defaultdict(int)
    N = len(chunks)

    for c in chunks:
        seen = set(_tokenize(c.text))
        for t in seen:
            df[t] += 1

    idf: Dict[str, float] = {}
    for t, d in df.items():
        idf[t] = math.log((N + 1) / (d + 1)) + 1.0
    return idf


def _score(query_tokens: List[str], doc_tf: Counter, idf: Dict[str, float]) -> float:
    if not query_tokens:
        return 0.0

    q_tf = Counter(query_tokens)
    q_w: Dict[str, float] = {t: float(tf) * idf.get(t, 0.0) for t, tf in q_tf.items()}

    d_w: Dict[str, float] = {}
    for t in q_w.keys():
        if t in doc_tf:
            d_w[t] = float(doc_tf[t]) * idf.get(t, 0.0)

    dot = sum(q_w[t] * d_w.get(t, 0.0) for t in q_w.keys())
    q_norm = math.sqrt(sum(v * v for v in q_w.values())) or 1.0
    d_norm = math.sqrt(sum(v * v for v in d_w.values())) or 1.0
    return dot / (q_norm * d_norm)


def _slugify(s: str, max_len: int = 48) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:max_len] or "pack")


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _trim(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="corpuspack",
        description="Create a paste-ready Markdown pack from your extracted 'my messages' corpus (JSONL).",
    )
    ap.add_argument("query", nargs="*", help="Search query (wrap in quotes for multi-word).")

    ap.add_argument(
        "--source",
        default="out/my_corpus/my_messages.jsonl",
        help="Path to my_messages.jsonl (default: out/my_corpus/my_messages.jsonl).",
    )
    ap.add_argument("--k", type=int, default=12, help="How many chunks to include (default: 12).")
    ap.add_argument("--chunk-chars", type=int, default=800, help="Chunk size in characters (default: 800).")
    ap.add_argument("--overlap", type=int, default=120, help="Chunk overlap in characters (default: 120).")
    ap.add_argument("--max-chars", type=int, default=1800, help="Max chars per included chunk (default: 1800).")

    ap.add_argument(
        "--out",
        default="",
        help="Output path. If omitted, writes to out/packs/<query_slug>__YYYYMMDD_HHMMSS.md",
    )
    ap.add_argument(
        "--out-dir",
        default="out/packs",
        help="Output folder when --out is omitted (default: out/packs).",
    )
    ap.add_argument(
        "--include-meta",
        action="store_true",
        help="Include any meta fields found in the JSONL rows (title/timestamp/etc).",
    )
    ap.add_argument(
        "--brief",
        action="store_true",
        help="Make the header shorter (less instructions).",
    )

    args = ap.parse_args()
    q = " ".join(args.query).strip()
    if not q:
        ap.print_help()
        return 2

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"Missing corpus file: {source}\n"
            "Tip: run your extractor first so out/my_corpus/my_messages.jsonl exists."
        )

    chunks = _load_chunks_from_my_messages(
        source,
        chunk_chars=int(args.chunk_chars),
        overlap=int(args.overlap),
    )
    if not chunks:
        print("No chunks loaded. If your source file contains mixed roles, make sure it includes user messages.")
        return 1

    idf = _build_idf(chunks)
    q_tokens = _tokenize(q)

    scored: List[Tuple[float, Chunk]] = []
    for c in chunks:
        tf = Counter(_tokenize(c.text))
        s = _score(q_tokens, tf, idf)
        if s > 0:
            scored.append((s, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, int(args.k))]

    if not top:
        print("No hits; pack not created.")
        return 0

    # Decide output path
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(q)
        out_path = (out_dir / f"{slug}__{_now_stamp()}.md").resolve()

    # Build markdown
    stamp = datetime.now().isoformat(timespec="seconds")
    lines: List[str] = []
    lines.append(f"# Corpus Pack — {q}")
    lines.append("")
    lines.append(f"- created: `{stamp}`")
    lines.append(f"- source: `{source}`")
    lines.append(f"- query: `{q}`")
    lines.append(f"- chunks: `{len(top)}`")
    lines.append("")

    if not args.brief:
        lines.append("## How to use this pack")
        lines.append("")
        lines.append("Paste this entire file into an AI (local or online).")
        lines.append("Ask it to **only** use these excerpts as ground truth, and to:")
        lines.append("- extract themes / facts")
        lines.append("- rewrite / summarize")
        lines.append("- generate outline(s) for a story or essay")
        lines.append("")
        lines.append("If you want *only your voice*, tell it to preserve your tone and avoid adding new facts.")
        lines.append("")

    lines.append("## Excerpts")
    lines.append("")

    for rank, (s, c) in enumerate(top, start=1):
        lines.append(f"### {rank}. {c.chunk_id} (score {s:0.3f})")
        if args.include_meta and c.meta:
            meta_bits = ", ".join(f"{k}={v}" for k, v in c.meta.items())
            lines.append(f"- meta: `{meta_bits}`")
        lines.append("")
        lines.append("```text")
        lines.append(_trim(c.text, int(args.max_chars)))
        lines.append("```")
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"✅ Pack written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
