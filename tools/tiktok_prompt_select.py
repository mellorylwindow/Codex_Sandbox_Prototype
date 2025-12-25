#!/usr/bin/env python3
"""
TikTok Prompt Project — Selector / Queue Builder (offline-first)

What this does
--------------
Given one or more JSONL files (or folders containing *.jsonl), rank drafts by
alignment to:

A) a freeform task string (your primary mode)
   --task-text "Explain Velvet OS like a hobby to coworkers..."

B) an explicit task profile JSON (optional)
   --task notes/tiktok/tasks/adhd_reset.json

It writes:
- out/tiktok_prompts/queues/<slug>_ranked.jsonl
- out/tiktok_prompts/queues/<slug>_queue.md

Design goals
------------
- Deterministic, offline, no model calls.
- Explainable scoring (reasons list).
- Robust input handling (bad JSONL lines won't crash the run).
- Multiple input paths: files and/or folders.

Usage examples
--------------
# Task-text (preferred)
python tools/tiktok_prompt_select.py \
  --drafts out/tiktok_prompts/drafts.jsonl out/tiktok_prompts/chat/drafts.jsonl \
  --task-text "Explain Velvet OS like a hobby to coworkers. Simple. Non-weird. Benefits for neurodivergent brains." \
  --take 12

# Folder inputs (scans all *.jsonl under each folder)
python tools/tiktok_prompt_select.py \
  --drafts out/tiktok_prompts out/tiktok_prompts/chat \
  --task-text "7 day reset planner routine cleaning productivity" \
  --required-any "reset,planner,routine" \
  --take 10

# Profile mode (optional)
python tools/tiktok_prompt_select.py \
  --drafts out/tiktok_prompts/drafts.jsonl \
  --task notes/tiktok/tasks/adhd_reset.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Stopwords (small & safe)
# ----------------------------

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "so",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
    "we",
    "our",
    "ours",
    "they",
    "their",
    "it",
    "this",
    "that",
    "these",
    "those",
    "do",
    "does",
    "did",
    "doing",
    "not",
    "no",
    "yes",
    "up",
    "down",
    "over",
    "under",
    "again",
    "can",
    "could",
    "should",
    "would",
    "will",
    "just",
    "into",
    "about",
    "than",
    "then",
    "there",
    "here",
}


# ----------------------------
# Data models
# ----------------------------

@dataclass(frozen=True)
class TaskProfile:
    name: str
    include: Dict[str, float]
    exclude: Dict[str, float]
    required_any: List[str]
    tags: Dict[str, float]
    min_score: float
    take: int


# ----------------------------
# Helpers: text + ids
# ----------------------------

TOKEN_RE = re.compile(r"[a-z0-9#][a-z0-9#\-_]{1,}")  # keeps #adhd and hyphenated tokens


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:60] if s else "task"


def stable_hash_id(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="replace"))
    return h.hexdigest()[:12]


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = TOKEN_RE.findall(text)
    out: List[str] = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if len(t) < 3 and not t.startswith("#"):
            continue
        out.append(t)
    return out


def blob_for_draft(d: dict) -> str:
    """
    Build the "document" text for scoring.
    prompt_core is usually the strongest signal, but we include tags + extracted_text too.
    """
    title = str(d.get("title") or "")
    prompt_core = str(d.get("prompt_core") or "")
    extracted = str(d.get("extracted_text") or "")
    tags = " ".join([str(t) for t in (d.get("tags") or [])])
    return "\n".join([title, prompt_core, tags, extracted]).strip()


# ----------------------------
# Helpers: JSONL IO
# ----------------------------

def read_jsonl(path: Path) -> List[dict]:
    """
    Read a JSONL file robustly:
    - skips blank lines
    - skips lines that aren't valid JSON
    """
    rows: List[dict] = []
    txt = path.read_text(encoding="utf-8", errors="replace")
    for idx, line in enumerate(txt.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            # Keep moving; bad line shouldn't take down the run
            continue
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_queue_md(
    path: Path,
    title: str,
    ranked: List[dict],
    take: int,
    min_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# TikTok Queue — {title}")
    lines.append("")
    lines.append(f"Generated (UTC): {utc_now_iso()}")
    lines.append(f"Min score: {min_score}")
    lines.append(f"Showing: {min(take, len(ranked))} of {len(ranked)}")
    lines.append("")

    for i, r in enumerate(ranked[:take], start=1):
        lines.append(f"## {i}. {r.get('id','<no-id>')} — score {r.get('score',0.0):.4f}")
        lines.append(f"- Title: {r.get('title','')}")
        lines.append(f"- Source: `{r.get('_drafts_file','')}`")
        core = (r.get("prompt_core") or "").strip()
        preview = core if core else (r.get("extracted_text") or "").strip()
        preview = preview.replace("\n", " ")
        if len(preview) > 260:
            preview = preview[:260] + "…"
        lines.append(f"- Preview: {preview}")
        tags = r.get("tags") or []
        lines.append(f"- Tags: {', '.join(tags)}")
        lines.append("")
        lines.append("**Reasons**")
        for reason in (r.get("reasons") or [])[:12]:
            lines.append(f"- {reason}")
        lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


# ----------------------------
# Multiple paths support
# ----------------------------

def iter_draft_files(paths: List[str]) -> List[Path]:
    """
    Expand a list of paths into JSONL files:
    - file.jsonl -> include
    - directory  -> include all *.jsonl under it
    Returns unique files (by resolved absolute path), sorted.
    """
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            continue
        if p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
        else:
            files.append(p)

    # unique by resolved path
    uniq: List[Path] = []
    seen: set[str] = set()
    for f in files:
        try:
            key = str(f.resolve())
        except Exception:
            key = str(f)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)

    return sorted(uniq, key=lambda x: str(x))


def read_jsonl_many(paths: List[str]) -> List[dict]:
    """
    Read many JSONL files, annotate each record with origin, and de-dupe by id.
    If a record lacks an id, we generate a stable one from content.
    """
    files = iter_draft_files(paths)
    if not files:
        return []

    rows: List[dict] = []
    for f in files:
        try:
            for r in read_jsonl(f):
                if "_drafts_file" not in r:
                    r["_drafts_file"] = str(f)
                rows.append(r)
        except Exception:
            # Skip problematic files silently; user can tighten inputs if needed
            continue

    deduped: List[dict] = []
    seen_ids: set[str] = set()

    for r in rows:
        rid = str(r.get("id") or "").strip()
        if not rid:
            # content-derived fallback id, deterministic
            blob = blob_for_draft(r)
            rid = stable_hash_id(blob)
            r["id"] = rid

        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        deduped.append(r)

    return deduped


# ----------------------------
# TF-IDF similarity (task-text mode)
# ----------------------------

def tf(tokens: List[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    # log-normalized TF
    out: Dict[str, float] = {}
    for k, c in counts.items():
        out[k] = 1.0 + math.log(c)
    return out


def idf(corpus_tokens: List[List[str]]) -> Dict[str, float]:
    """
    Smooth IDF:
      idf(t) = log((N + 1)/(df + 1)) + 1
    """
    N = len(corpus_tokens)
    df: Dict[str, int] = {}
    for doc in corpus_tokens:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1

    out: Dict[str, float] = {}
    for t, d in df.items():
        out[t] = math.log((N + 1.0) / (d + 1.0)) + 1.0
    return out


def tfidf_vec(tokens: List[str], idf_map: Dict[str, float]) -> Dict[str, float]:
    tmap = tf(tokens)
    return {t: tmap[t] * idf_map.get(t, 0.0) for t in tmap.keys()}


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0

    # dot product; iterate smaller dict for speed
    dot = 0.0
    if len(a) > len(b):
        a, b = b, a
    for k, av in a.items():
        bv = b.get(k)
        if bv:
            dot += av * bv

    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def quality_bonus(d: dict) -> Tuple[float, List[str]]:
    """
    Small deterministic bonuses that favor usable drafts.
    Keep these mild so similarity drives selection.
    """
    score = 0.0
    reasons: List[str] = []

    core = str(d.get("prompt_core") or "").strip()
    if core:
        score += 0.15
        reasons.append("+0.150 has prompt_core")

    if len(core) >= 120:
        score += 0.08
        reasons.append("+0.080 core_len>=120")
    elif len(core) >= 60:
        score += 0.04
        reasons.append("+0.040 core_len>=60")

    tags = d.get("tags") or []
    if tags:
        score += 0.03
        reasons.append("+0.030 has tags")

    return score, reasons


def score_task_text(
    d: dict,
    query_vec: Dict[str, float],
    idf_map: Dict[str, float],
    required_any: Optional[List[str]],
) -> Tuple[float, List[str]]:
    blob = blob_for_draft(d)

    # Optional gate: must match at least one literal term
    if required_any:
        blob_l = blob.lower()
        if not any(term.lower() in blob_l for term in required_any):
            return (-9999.0, ["failed required_any gate"])

    doc_tokens = tokenize(blob)
    doc_vec = tfidf_vec(doc_tokens, idf_map)

    sim = cosine(query_vec, doc_vec)
    bonus, bonus_reasons = quality_bonus(d)

    score = sim + bonus
    reasons = [f"{sim:.4f} cosine_similarity"] + bonus_reasons
    return score, reasons


# ----------------------------
# Profile mode (optional)
# ----------------------------

def load_task_profile(path: Path) -> TaskProfile:
    raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return TaskProfile(
        name=str(raw.get("name") or path.stem),
        include=dict(raw.get("include") or {}),
        exclude=dict(raw.get("exclude") or {}),
        required_any=list(raw.get("required_any") or []),
        tags=dict(raw.get("tags") or {}),
        min_score=float(raw.get("min_score") or 0.0),
        take=int(raw.get("take") or 10),
    )


def score_profile(d: dict, task: TaskProfile) -> Tuple[float, List[str]]:
    blob = blob_for_draft(d)
    blob_l = blob.lower()

    # required gate
    if task.required_any and not any(term.lower() in blob_l for term in task.required_any):
        return (-9999.0, ["failed required_any gate"])

    score = 0.0
    reasons: List[str] = []

    for term, w in task.include.items():
        c = blob_l.count(term.lower())
        if c:
            add = float(w) * c
            score += add
            reasons.append(f"+{add:.2f} include:{term} x{c}")

    for term, w in task.exclude.items():
        c = blob_l.count(term.lower())
        if c:
            sub = float(w) * c
            score -= sub
            reasons.append(f"-{sub:.2f} exclude:{term} x{c}")

    # tag boosts
    tag_blob = " ".join([str(t) for t in (d.get("tags") or [])]).lower()
    for term, w in task.tags.items():
        if term.lower() in tag_blob:
            score += float(w)
            reasons.append(f"+{float(w):.2f} tag:{term}")

    # quality bonus
    bonus, bonus_reasons = quality_bonus(d)
    score += bonus
    reasons.extend(bonus_reasons)

    return score, reasons


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rank TikTok drafts by task alignment (task-text or profile).")

    ap.add_argument(
        "--drafts",
        nargs="+",
        default=["out/tiktok_prompts/drafts.jsonl"],
        help="One or more drafts JSONL paths (files or folders). Folders are scanned for *.jsonl",
    )

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-text", help="Freeform task text to align against (on-the-fly)")
    mode.add_argument("--task", help="Task profile JSON path (notes/tiktok/tasks/<name>.json)")

    ap.add_argument("--name", default=None, help="Override queue title/name")
    ap.add_argument("--outdir", default="out/tiktok_prompts/queues", help="Output directory for ranked + queue files")
    ap.add_argument("--take", type=int, default=10, help="How many to show in the queue markdown")
    ap.add_argument("--min-score", type=float, default=0.10, help="Minimum score to include (task-text mode)")
    ap.add_argument("--required-any", default=None, help="Comma-separated gate terms (must match at least one)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    rows = read_jsonl_many(args.drafts)
    if not rows:
        print("No drafts found. Check your --drafts paths.")
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    required_any: Optional[List[str]] = None
    if args.required_any:
        required_any = [t.strip() for t in args.required_any.split(",") if t.strip()]

    # ----------------------------
    # Profile mode
    # ----------------------------
    if args.task:
        task_path = Path(args.task)
        if not task_path.exists():
            print(f"ERROR: task profile not found: {task_path}")
            return 2

        task = load_task_profile(task_path)
        title = args.name or task.name
        slug = slugify(args.name or task_path.stem)

        ranked: List[dict] = []
        for d in rows:
            score, reasons = score_profile(d, task)
            if score < task.min_score:
                continue
            out = dict(d)
            out["score"] = float(score)
            out["reasons"] = reasons
            ranked.append(out)

        ranked.sort(key=lambda x: x["score"], reverse=True)

        ranked_path = outdir / f"{slug}_ranked.jsonl"
        md_path = outdir / f"{slug}_queue.md"
        write_jsonl(ranked_path, ranked)
        write_queue_md(md_path, title=title, ranked=ranked, take=task.take, min_score=task.min_score)

        print(f"OK: ranked {len(ranked)} drafts for profile '{title}'")
        print(f"- {ranked_path}")
        print(f"- {md_path}")
        return 0

    # ----------------------------
    # Task-text mode (primary)
    # ----------------------------
    task_text = str(args.task_text or "").strip()
    title = args.name or task_text
    slug = slugify(args.name or task_text)

    # Build IDF over all docs (merged inputs)
    corpus_tokens = [tokenize(blob_for_draft(d)) for d in rows]
    idf_map = idf(corpus_tokens)

    # Query vector
    q_tokens = tokenize(task_text)
    query_vec = tfidf_vec(q_tokens, idf_map)

    ranked: List[dict] = []
    for d in rows:
        score, reasons = score_task_text(d, query_vec=query_vec, idf_map=idf_map, required_any=required_any)
        if score < float(args.min_score):
            continue
        out = dict(d)
        out["score"] = float(score)
        out["reasons"] = reasons
        ranked.append(out)

    ranked.sort(key=lambda x: x["score"], reverse=True)

    ranked_path = outdir / f"{slug}_ranked.jsonl"
    md_path = outdir / f"{slug}_queue.md"
    write_jsonl(ranked_path, ranked)
    write_queue_md(md_path, title=title, ranked=ranked, take=int(args.take), min_score=float(args.min_score))

    print(f"OK: ranked {len(ranked)} drafts for task-text '{title}'")
    print(f"- {ranked_path}")
    print(f"- {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
