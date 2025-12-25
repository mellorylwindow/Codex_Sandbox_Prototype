#!/usr/bin/env python3
"""
Extract prompt-like text from a ChatGPT export folder (or any folder),
dedupe it, and output both:

- notes/tiktok/compiled_in/compiled_prompts.txt  (human-editable list)
- out/tiktok_prompts/compiled/drafts.jsonl       (rankable drafts stream)

Key fixes vs v1:
- ChatGPT export conversations.json is typically a LIST of conversations.
  We now support list/dict top-level equally.
- Optional role control: --roles user|assistant|both (default: both)
- --tiktok-only is "content-creation language" (hook/script/caption/voiceover/etc.)
  not a brittle requirement of the literal word "tiktok"
- Debug counts at each stage so you can see why you got 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --------------------------
# Heuristics / vocab
# --------------------------

PROMPT_START_TRIGGERS = (
    "write",
    "create",
    "generate",
    "build",
    "draft",
    "make",
    "outline",
    "explain",
    "summarize",
    "turn this into",
    "act as",
    "pretend you are",
    "you are my",
    "you are an",
    "you are a",
    "give me",
    "help me",
)

STRUCTURE_HINTS = (
    "steps",
    "checklist",
    "template",
    "framework",
    "in the style of",
    "tone",
    "bullets",
    "bullet",
    "outline",
    "sections",
    "constraints",
)

# Content creation / shortform language
CONTENT_CREATION_HINTS = (
    "tiktok",
    "reel",
    "short",
    "shorts",
    "hook",
    "caption",
    "voiceover",
    "script",
    "teleprompter",
    "b-roll",
    "shot list",
    "cta",
    "call to action",
    "on-screen text",
    "onscreen text",
    "beats",
    "scene",
    "open with",
    "cold open",
    "thumbnail",
)

# Noisy “terminal / code / logs / tool” terms that shouldn’t become prompts
NEGATIVE_KEYWORDS = (
    "traceback",
    "exception",
    "stack trace",
    "pip install",
    "apt-get",
    "winget",
    "stderr",
    "stdout",
    "jsonl",
    "pydantic",
    "typer",
    "mkdir",
    "touch",
    "sed -n",
    "grep -r",
    "architect@",
    "ok: wrote",
    "ok: ranked",
    "pipeline complete",
    "file not found",
)

NOISE_LINE_RE = re.compile(
    r"(?i)^\s*(architect@|ok:\s*wrote|ok:\s*ranked|traceback|exception|file not found|\$)\b"
)

RE_INLINE_PROMPT = re.compile(r"(?is)\bprompt\s*[:—-]\s*(.+?)(?:\n{2,}|$)")
RE_LIST_BULLET = re.compile(r"^\s*[-*]\s+(.*)\s*$")
RE_LIST_NUM = re.compile(r"^\s*\d+\.\s+(.*)\s*$")

MIN_LEN_DEFAULT = 18
MAX_LEN_DEFAULT = 900


# --------------------------
# Small data model
# --------------------------

@dataclass(frozen=True)
class ExtractedText:
    text: str
    role: str  # "user" | "assistant" | "unknown"
    source: str  # filename or descriptor


# --------------------------
# Utilities
# --------------------------

def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def norm(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def is_noise_block(s: str) -> bool:
    low = s.lower()
    if any(k in low for k in NEGATIVE_KEYWORDS):
        return True
    # very code-ish density heuristic
    if (low.count("{") + low.count("}") + low.count(";")) > 12:
        return True
    return False


def score_prompt(s: str, tiktok_only: bool) -> float:
    """
    Score 0..1:
    - Instruction-ish start
    - Role prompting
    - Structured hints
    - Content creation hints (required if --tiktok-only)
    """
    s = norm(s)
    low = s.lower()

    if not s:
        return 0.0
    if is_noise_block(s):
        return 0.0

    score = 0.0

    start = low[:140]
    if any(start.startswith(t) for t in PROMPT_START_TRIGGERS):
        score += 0.45

    if "act as" in low or "you are my" in low or low.startswith("you are"):
        score += 0.18

    struct_hits = sum(1 for h in STRUCTURE_HINTS if h in low)
    score += min(0.28, struct_hits * 0.07)

    cc_hits = sum(1 for h in CONTENT_CREATION_HINTS if h in low)

    if tiktok_only:
        # Require at least SOME content-creation smell
        if cc_hits == 0:
            return 0.0
        score += min(0.35, cc_hits * 0.10)
    else:
        score += min(0.18, cc_hits * 0.05)

    # Penalize pure narration (helps keep diaries from scoring)
    if low.startswith(("i ", "we ", "here's ")):
        score -= 0.10

    return max(0.0, min(1.0, score))


def split_candidate_blocks(text: str) -> List[str]:
    """
    Split text into candidate prompt blocks:
    - explicit "Prompt: ..." blocks
    - list items become candidates if listy enough
    - otherwise paragraphs
    """
    text = norm(text)
    if not text:
        return []

    out: List[str] = []

    # explicit Prompt: blocks
    for m in RE_INLINE_PROMPT.finditer(text):
        body = norm(m.group(1))
        if body:
            out.append(body)

    lines = text.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    listish = sum(1 for ln in nonempty if RE_LIST_BULLET.match(ln) or RE_LIST_NUM.match(ln))

    if nonempty and listish >= max(3, int(0.25 * len(nonempty))):
        for ln in nonempty:
            if NOISE_LINE_RE.match(ln):
                continue
            m = RE_LIST_BULLET.match(ln) or RE_LIST_NUM.match(ln)
            if m:
                item = norm(m.group(1))
            else:
                item = norm(ln)
            if item:
                out.append(item)
        return out

    # paragraph mode
    paras = re.split(r"\n\s*\n+", text)
    for p in paras:
        p = norm(p.replace("\n", " "))
        if not p:
            continue
        if NOISE_LINE_RE.match(p):
            continue
        out.append(p)

    return out


# --------------------------
# Export parsing (robust)
# --------------------------

def _role_from_msg(msg: Dict[str, Any]) -> str:
    author = msg.get("author") or {}
    role = (author.get("role") or msg.get("role") or "").lower().strip()
    if role in ("user", "assistant", "system", "tool"):
        return role
    return "unknown"


def iter_texts_from_conversation_obj(obj: Any, source: str) -> Iterable[ExtractedText]:
    """
    Yield ExtractedText from one conversation-like object.
    Supports both:
    - {"mapping": {...}} nodes
    - {"messages": [...]} list
    """
    if not isinstance(obj, dict):
        return

    # mapping-style
    mapping = obj.get("mapping")
    if isinstance(mapping, dict):
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue
            role = _role_from_msg(msg)
            content = msg.get("content") or {}
            parts = content.get("parts")
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, str) and p.strip():
                        yield ExtractedText(text=p, role=role, source=source)
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                yield ExtractedText(text=text, role=role, source=source)

    # messages-style
    messages = obj.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = _role_from_msg(msg)
            content = msg.get("content") or {}
            parts = content.get("parts")
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, str) and p.strip():
                        yield ExtractedText(text=p, role=role, source=source)
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                yield ExtractedText(text=text, role=role, source=source)
            raw_text = msg.get("text")
            if isinstance(raw_text, str) and raw_text.strip():
                yield ExtractedText(text=raw_text, role=role, source=source)


def iter_texts_from_any_json(obj: Any, source: str) -> Iterable[ExtractedText]:
    """
    ChatGPT exports can be:
    - a LIST of conversations
    - a single conversation dict
    - other JSON that still contains conversation-ish dicts
    Strategy:
    - If list: iterate items as conversations
    - If dict: try as conversation; also scan shallow fields for nested lists/dicts
    """
    if isinstance(obj, list):
        for item in obj:
            # each item is often a conversation dict
            yielded_any = False
            for et in iter_texts_from_conversation_obj(item, source):
                yielded_any = True
                yield et
            if not yielded_any:
                # fallback: recursive scan for strings
                yield from iter_texts_recursive(item, source)
        return

    if isinstance(obj, dict):
        yielded_any = False
        for et in iter_texts_from_conversation_obj(obj, source):
            yielded_any = True
            yield et
        if yielded_any:
            return
        # fallback: recursive scan for strings
        yield from iter_texts_recursive(obj, source)
        return

    # fallback
    yield from iter_texts_recursive(obj, source)


def iter_texts_recursive(obj: Any, source: str) -> Iterable[ExtractedText]:
    """
    Last-resort: walk JSON recursively and yield strings.
    Role unknown.
    """
    if isinstance(obj, str):
        if obj.strip():
            yield ExtractedText(text=obj, role="unknown", source=source)
        return
    if isinstance(obj, list):
        for v in obj:
            yield from iter_texts_recursive(v, source)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from iter_texts_recursive(v, source)
        return


# --------------------------
# File walking / reading
# --------------------------

def walk_files(roots: List[Path], max_files: int) -> List[Path]:
    exts = {".json", ".jsonl", ".txt", ".md", ".markdown", ".html", ".htm"}
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in exts:
                files.append(root)
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    files.sort()
    if max_files and max_files > 0:
        files = files[:max_files]
    return files


def read_text_safe(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json_safe(path: Path) -> Optional[Any]:
    try:
        return json.loads(read_text_safe(path))
    except Exception:
        return None


def load_jsonl_rows(path: Path) -> List[Any]:
    rows: List[Any] = []
    for line in read_text_safe(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


# --------------------------
# Writers (single source of truth)
# --------------------------

def write_outputs(final_prompts: List[str], out_txt: Path, out_jsonl: Path, source_name: str) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    out_txt.write_text("\n".join(final_prompts) + "\n", encoding="utf-8", newline="\n")

    with out_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for p in final_prompts:
            row = {
                "id": stable_id(p),
                "title": "",
                "prompt_core": p,
                "tags": [],
                "source_name": source_name,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------
# Main
# --------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Extract prompt text from ChatGPT export(s).")
    ap.add_argument("--export", nargs="+", required=True, help="One or more export folders/files")

    ap.add_argument("--out-txt", default="notes/tiktok/compiled_in/compiled_prompts.txt")
    ap.add_argument("--out-jsonl", default="out/tiktok_prompts/compiled/drafts.jsonl")
    ap.add_argument("--source-name", default="chatgpt_export")

    ap.add_argument(
        "--roles",
        choices=["user", "assistant", "both"],
        default="both",
        help="Which roles to include when roles are available (default: both).",
    )

    ap.add_argument(
        "--tiktok-only",
        action="store_true",
        help="Require content-creation language (hook/script/caption/voiceover/etc.).",
    )

    ap.add_argument("--min-score", type=float, default=0.45, help="Score threshold 0..1")
    ap.add_argument("--min-len", type=int, default=MIN_LEN_DEFAULT)
    ap.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT)
    ap.add_argument("--max-files", type=int, default=0, help="0 = no limit")
    ap.add_argument("--debug", action="store_true", help="Print debug counts / samples.")

    args = ap.parse_args()

    roots = [Path(p) for p in args.export]
    files = walk_files(roots, args.max_files)
    if not files:
        print("ERROR: No files found under provided --export paths.")
        return 2

    # Stage counts
    raw_text_count = 0
    candidate_block_count = 0
    kept_after_len_noise = 0
    kept_after_score = 0
    final_count = 0

    extracted_texts: List[ExtractedText] = []

    for fp in files:
        suf = fp.suffix.lower()
        src = str(fp)

        if suf == ".json":
            obj = load_json_safe(fp)
            if obj is None:
                continue
            for et in iter_texts_from_any_json(obj, src):
                extracted_texts.append(et)
            continue

        if suf == ".jsonl":
            for row in load_jsonl_rows(fp):
                for et in iter_texts_from_any_json(row, src):
                    extracted_texts.append(et)
            continue

        # text-ish files
        txt = read_text_safe(fp)
        if txt.strip():
            extracted_texts.append(ExtractedText(text=txt, role="unknown", source=src))

    # Role filter
    def role_allowed(role: str) -> bool:
        if args.roles == "both":
            return True
        return role == args.roles

    candidates: List[str] = []

    for et in extracted_texts:
        if not role_allowed(et.role):
            continue
        raw_text_count += 1
        for blk in split_candidate_blocks(et.text):
            candidate_block_count += 1
            candidates.append(blk)

    # Filter + dedupe
    final_prompts: List[str] = []
    seen_lower = set()

    for c in candidates:
        c = norm(c)
        if not c:
            continue
        if len(c) < args.min_len:
            continue
        if len(c) > args.max_len:
            c = c[: args.max_len].rstrip() + "…"

        if is_noise_block(c):
            continue

        kept_after_len_noise += 1

        sc = score_prompt(c, tiktok_only=args.tiktok_only)
        if sc < args.min_score:
            continue

        kept_after_score += 1

        one_line = norm(c.replace("\n", " "))
        key = one_line.lower()

        if key in seen_lower:
            continue
        seen_lower.add(key)

        final_prompts.append(one_line)

    final_count = len(final_prompts)

    if args.debug:
        print("DEBUG:")
        print(f"- files scanned:            {len(files)}")
        print(f"- extracted text blobs:     {raw_text_count}")
        print(f"- candidate blocks:         {candidate_block_count}")
        print(f"- after len/noise filter:   {kept_after_len_noise}")
        print(f"- after score filter:       {kept_after_score}")
        print(f"- final unique prompts:     {final_count}")
        if final_prompts[:5]:
            print("\nDEBUG sample prompts:")
            for s in final_prompts[:5]:
                print(f"  - {s[:140]}")

    if not final_prompts:
        print("No prompts extracted after filtering.")
        print("Try one of:")
        print("  - add --debug to see where it drops to 0")
        print("  - lower --min-score (e.g., 0.30)")
        print("  - remove --tiktok-only")
        print("  - use --roles assistant (export prompts are often assistant-authored)")
        return 2

    out_txt = Path(args.out_txt)
    out_jsonl = Path(args.out_jsonl)
    write_outputs(final_prompts, out_txt, out_jsonl, args.source_name)

    print(f"OK: extracted {len(final_prompts)} prompts")
    print(f"- {out_txt}")
    print(f"- {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
