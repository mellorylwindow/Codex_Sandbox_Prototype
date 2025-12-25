# tools/corpus_search.py
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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
                # tolerate occasional bad lines
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
    source: str  # e.g. file name / conversation id if present
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
        # expected shape: {"role":"user","text":"..."} but be flexible
        role = (row.get("role") or row.get("speaker") or "").lower()
        if role and role not in ("user", "me", "jimmy"):
            # This tool is "my messages" search; skip non-user roles if present.
            # If you point it at mixed logs, this keeps it Jimmy-only.
            continue

        text = (row.get("text") or row.get("content") or "").strip()
        if not text:
            continue

        # attach helpful meta if present
        meta = {}
        for k in ("title", "conversation_title", "conversation_id", "created_at", "timestamp", "source_path"):
            if k in row and row[k]:
                meta[k] = row[k]

        # chunk it
        parts = _chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
        for j, part in enumerate(parts):
            cid = f"{base}:msg{i:06d}:c{j:02d}"
            chunks.append(Chunk(chunk_id=cid, text=part, source=base, meta=meta))

    return chunks


def _build_idf(chunks: List[Chunk]) -> Dict[str, float]:
    """
    Smooth IDF: log((N + 1) / (df + 1)) + 1
    """
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
    """
    Simple TF-IDF cosine-ish score:
    - query weights: idf
    - doc weights: tf * idf
    - normalized by vector lengths
    """
    if not query_tokens:
        return 0.0

    q_tf = Counter(query_tokens)
    # Build query vector
    q_w: Dict[str, float] = {}
    for t, tf in q_tf.items():
        q_w[t] = float(tf) * idf.get(t, 0.0)

    # Build doc vector on query terms only (fast)
    d_w: Dict[str, float] = {}
    for t in q_w.keys():
        if t in doc_tf:
            d_w[t] = float(doc_tf[t]) * idf.get(t, 0.0)

    # dot
    dot = 0.0
    for t in q_w.keys():
        dot += q_w[t] * d_w.get(t, 0.0)

    # norms
    q_norm = math.sqrt(sum(v * v for v in q_w.values())) or 1.0
    d_norm = math.sqrt(sum(v * v for v in d_w.values())) or 1.0

    return dot / (q_norm * d_norm)


def _format_hit(c: Chunk, score: float, *, width: int = 100) -> str:
    snippet = " ".join(c.text.split())
    if len(snippet) > width:
        snippet = snippet[: width - 3] + "..."
    meta_bits = []
    if c.meta.get("title"):
        meta_bits.append(f"title={c.meta['title']}")
    if c.meta.get("created_at"):
        meta_bits.append(f"at={c.meta['created_at']}")
    meta_str = (" | " + ", ".join(meta_bits)) if meta_bits else ""
    return f"[{score:0.3f}] {c.chunk_id}{meta_str}\n  {snippet}\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="corpussearch",
        description="Offline search over your extracted 'my messages' corpus (JSONL).",
    )
    ap.add_argument("query", nargs="*", help="Search query (wrap in quotes for multi-word).")

    ap.add_argument(
        "--source",
        default="out/my_corpus/my_messages.jsonl",
        help="Path to my_messages.jsonl (default: out/my_corpus/my_messages.jsonl).",
    )
    ap.add_argument("--k", type=int, default=10, help="Top K results (default: 10).")
    ap.add_argument("--chunk-chars", type=int, default=800, help="Chunk size in characters (default: 800).")
    ap.add_argument("--overlap", type=int, default=120, help="Chunk overlap in characters (default: 120).")
    ap.add_argument("--show", type=int, default=1000, help="Max chars to display per hit (default: 1000).")

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
        print("No hits.")
        return 0

    for s, c in top:
        # print chunk body up to --show
        body = c.text.strip()
        if len(body) > int(args.show):
            body = body[: int(args.show) - 3] + "..."
        print(f"[{s:0.3f}] {c.chunk_id}")
        if c.meta:
            meta_bits = ", ".join(f"{k}={v}" for k, v in c.meta.items())
            print(f"  meta: {meta_bits}")
        print(body)
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
