# textlab/pipeline.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Models
# -----------------------------

_TS_RE = re.compile(r"^\[(\d{2}):(\d{2})\]\s+")


@dataclass
class ChunkMeta:
    index: int
    path: Path
    start_s: Optional[float]
    end_s: Optional[float]
    start_ts: Optional[str]
    end_ts: Optional[str]
    line_count: int
    char_count: int


# -----------------------------
# Utilities
# -----------------------------

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _detect_timestamped(text: str) -> bool:
    # If at least a couple lines look timestamped, treat as timestamped
    hits = 0
    for line in text.splitlines()[:80]:
        if _TS_RE.match(line.strip()):
            hits += 1
            if hits >= 2:
                return True
    return False


def _mmss_to_seconds(mm: int, ss: int) -> int:
    return mm * 60 + ss


def _seconds_to_mmss(seconds: int) -> str:
    mm = seconds // 60
    ss = seconds % 60
    return f"{mm:02d}:{ss:02d}"


# -----------------------------
# Corrections (post-fix)
# -----------------------------

def _load_corrections(corrections_path: Optional[Path]) -> List[Tuple[str, str]]:
    """
    Supports multiple formats:
      1) dict: { "A": "B", ... }
      2) { "replacements": [{"from": "A", "to": "B"}, ...] }
      3) { "pairs": [["A","B"], ...] }
    """
    if not corrections_path:
        return []
    p = corrections_path.expanduser().resolve()
    if not p.exists():
        return []

    raw = json.loads(_safe_read_text(p))
    pairs: List[Tuple[str, str]] = []

    if isinstance(raw, dict):
        if "replacements" in raw and isinstance(raw["replacements"], list):
            for item in raw["replacements"]:
                if isinstance(item, dict) and "from" in item and "to" in item:
                    a = str(item["from"])
                    b = str(item["to"])
                    if a:
                        pairs.append((a, b))
            return pairs

        if "pairs" in raw and isinstance(raw["pairs"], list):
            for item in raw["pairs"]:
                if isinstance(item, list) and len(item) == 2:
                    a = str(item[0])
                    b = str(item[1])
                    if a:
                        pairs.append((a, b))
            return pairs

        # plain dict mapping
        # ignore non-string keys
        for k, v in raw.items():
            if k in ("replacements", "pairs"):
                continue
            a = str(k)
            b = str(v)
            if a:
                pairs.append((a, b))
        return pairs

    return pairs


def apply_corrections(text: str, corrections: List[Tuple[str, str]]) -> str:
    # Apply in order (teach order matters sometimes)
    out = text
    for a, b in corrections:
        if not a:
            continue
        out = out.replace(a, b)
    return out


# -----------------------------
# Cleaning (offline-first)
# -----------------------------

def clean_text(text: str, mode: str) -> str:
    """
    mode:
      - light: normalize whitespace safely
      - standard: light + fix common ASR punctuation spacing + collapse repeated whitespace
    """
    if mode not in ("light", "standard"):
        mode = "standard"

    # Normalize line endings + strip trailing spaces
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out = "\n".join(lines)

    # Collapse 3+ newlines to max 2
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"

    if mode == "light":
        return out

    # Standard cleanup:
    # - fix spaces before punctuation
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)

    # - fix punctuation followed by missing space (but don’t break timestamps)
    out = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", out)

    # - collapse multiple spaces (not inside timestamps)
    def _collapse_spaces_line(line: str) -> str:
        if _TS_RE.match(line):
            # keep timestamp prefix stable, collapse after it
            m = _TS_RE.match(line)
            assert m is not None
            prefix = line[: m.end()]
            rest = line[m.end():]
            rest = re.sub(r"[ \t]{2,}", " ", rest).strip()
            return prefix + rest
        return re.sub(r"[ \t]{2,}", " ", line).strip()

    out_lines = [_collapse_spaces_line(ln) for ln in out.splitlines()]
    out = "\n".join(out_lines).strip() + "\n"
    return out


# -----------------------------
# Redaction (offline-first + custom terms)
# -----------------------------

def _load_redaction_terms(terms_path: Optional[Path]) -> Dict[str, Any]:
    if not terms_path:
        return {}
    p = terms_path.expanduser().resolve()
    if not p.exists():
        return {}
    try:
        return json.loads(_safe_read_text(p))
    except Exception:
        return {}


def _compile_custom_patterns(spec: Dict[str, Any]) -> List[Tuple[re.Pattern, str]]:
    """
    Supports:
      - "patterns": [{"regex": "...", "tag": "[X]", "flags": "i"}, ...]
      - "terms": [{"term": "Jimmy", "tag": "[NAME]"}, ...]
    """
    compiled: List[Tuple[re.Pattern, str]] = []

    # patterns
    patterns = spec.get("patterns", [])
    if isinstance(patterns, list):
        for i, item in enumerate(patterns):
            if not isinstance(item, dict):
                continue
            rx = item.get("regex")
            tag = item.get("tag") or item.get("replacement") or "[REDACTED]"
            if not rx:
                # skip invalid patterns quietly (pipeline should not explode)
                continue
            flags = 0
            fl = str(item.get("flags") or "")
            if "i" in fl.lower():
                flags |= re.IGNORECASE
            try:
                compiled.append((re.compile(str(rx), flags), str(tag)))
            except re.error:
                continue

    # terms (escaped literal matches with word boundaries unless requested otherwise)
    terms = spec.get("terms", [])
    if isinstance(terms, list):
        for item in terms:
            if not isinstance(item, dict):
                continue
            term = item.get("term")
            tag = item.get("tag") or "[REDACTED]"
            if not term:
                continue
            case_sensitive = bool(item.get("case_sensitive", False))
            whole_word = item.get("whole_word", True)
            flags = 0 if case_sensitive else re.IGNORECASE
            escaped = re.escape(str(term))
            if whole_word:
                rx = r"\b" + escaped + r"\b"
            else:
                rx = escaped
            try:
                compiled.append((re.compile(rx, flags), str(tag)))
            except re.error:
                continue

    return compiled


def redact_text(text: str, level: Optional[str], terms_spec: Dict[str, Any]) -> str:
    """
    level:
      - None/"none": no redaction
      - light: emails/phones
      - standard: light + dates/ages/weights (basic)
      - heavy: standard + common name phrase ("My name is X") redaction heuristics
    """
    if not level or level == "none":
        # still apply custom patterns if provided (user intent)
        compiled = _compile_custom_patterns(terms_spec)
        out = text
        for rx, tag in compiled:
            out = rx.sub(tag, out)
        return out

    lvl = str(level).lower()
    out = text

    # light: email + phone
    out = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "[PHONE]", out)

    if lvl in ("standard", "heavy"):
        # dates (simple)
        # 10/29/2024, 2024-10-29, "October 29th, 2024" (basic)
        out = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]", out)
        out = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[DATE]", out)
        out = re.sub(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
            "[DATE]",
            out,
            flags=re.IGNORECASE,
        )

        # age patterns
        out = re.sub(r"\bI'?m\s+\d{1,3}\s+years?\s+old\b", "I'm [AGE].", out, flags=re.IGNORECASE)
        out = re.sub(r"\b(?:age\s+is\s+|I'?m\s+)\d{1,3}\b", lambda m: m.group(0).split()[0] + " [AGE]", out, flags=re.IGNORECASE)

        # weight patterns (basic)
        out = re.sub(r"\bweight\s+is\s+about\s+\d{2,4}\b", "weight is about [WEIGHT]", out, flags=re.IGNORECASE)
        out = re.sub(r"\bweight\s+is\s+\d{2,4}\b", "weight is [WEIGHT]", out, flags=re.IGNORECASE)

    if lvl == "heavy":
        # crude “my name is X” redaction
        out = re.sub(r"\bMy name is\s+([A-Za-z][A-Za-z .'-]{1,80})\b", "My name is [NAME]", out)

    # Custom patterns/terms always apply last
    compiled = _compile_custom_patterns(terms_spec)
    for rx, tag in compiled:
        out = rx.sub(tag, out)

    return out


# -----------------------------
# Chunking + TOC
# -----------------------------

def _parse_timestamped_lines(lines: List[str]) -> List[Tuple[int, str]]:
    """
    Returns list of (start_seconds, line_text_without_timestamp_prefix).
    """
    parsed: List[Tuple[int, str]] = []
    for ln in lines:
        m = _TS_RE.match(ln.strip())
        if not m:
            continue
        mm = int(m.group(1))
        ss = int(m.group(2))
        start_s = _mmss_to_seconds(mm, ss)
        body = ln[m.end():].strip()
        if body:
            parsed.append((start_s, body))
    return parsed


def _chunk_timestamped(text: str, chunk_minutes: int, chunks_dir: Path) -> List[ChunkMeta]:
    lines = text.splitlines()
    parsed = _parse_timestamped_lines(lines)
    if not parsed:
        # fallback if timestamp detection lied
        return _chunk_plain(text, chunks_dir)

    chunk_seconds = max(60, int(chunk_minutes) * 60)

    metas: List[ChunkMeta] = []
    buf: List[str] = []
    chunk_start_s: Optional[int] = None
    last_s: Optional[int] = None
    idx = 0

    def _flush() -> None:
        nonlocal idx, buf, chunk_start_s, last_s
        if not buf:
            return
        start_s = float(chunk_start_s) if chunk_start_s is not None else None
        end_s = float(last_s) if last_s is not None else None
        start_ts = _seconds_to_mmss(int(start_s)) if start_s is not None else None
        end_ts = _seconds_to_mmss(int(end_s)) if end_s is not None else None
        out_path = chunks_dir / f"chunk_{idx:03d}.txt"
        _safe_write_text(out_path, "\n".join(buf).strip() + "\n")
        metas.append(
            ChunkMeta(
                index=idx,
                path=out_path,
                start_s=start_s,
                end_s=end_s,
                start_ts=start_ts,
                end_ts=end_ts,
                line_count=len(buf),
                char_count=sum(len(x) for x in buf),
            )
        )
        idx += 1
        buf = []
        chunk_start_s = None
        last_s = None

    for start_s, body in parsed:
        if chunk_start_s is None:
            chunk_start_s = start_s
        last_s = start_s
        # create consistent timestamped lines for chunk files
        buf.append(f"[{_seconds_to_mmss(start_s)}] {body}")

        if (start_s - chunk_start_s) >= chunk_seconds and len(buf) >= 8:
            _flush()

    _flush()
    return metas


def _chunk_plain(text: str, chunks_dir: Path, target_chars: int = 12000) -> List[ChunkMeta]:
    """
    Plain chunker: group paragraphs until approx target chars.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    metas: List[ChunkMeta] = []
    idx = 0
    buf: List[str] = []
    buf_chars = 0

    def _flush() -> None:
        nonlocal idx, buf, buf_chars
        if not buf:
            return
        out_path = chunks_dir / f"chunk_{idx:03d}.txt"
        _safe_write_text(out_path, "\n\n".join(buf).strip() + "\n")
        metas.append(
            ChunkMeta(
                index=idx,
                path=out_path,
                start_s=None,
                end_s=None,
                start_ts=None,
                end_ts=None,
                line_count=sum(x.count("\n") + 1 for x in buf),
                char_count=buf_chars,
            )
        )
        idx += 1
        buf = []
        buf_chars = 0

    for p in paras:
        if buf and (buf_chars + len(p) + 2) > target_chars:
            _flush()
        buf.append(p)
        buf_chars += len(p) + 2

    _flush()
    return metas


def write_toc(run_dir: Path, metas: List[ChunkMeta], timestamped: bool) -> Path:
    toc = run_dir / "TOC.md"
    lines: List[str] = []
    lines.append(f"# TextLab TOC — {run_dir.name}")
    lines.append("")
    lines.append(f"- timestamped: `{timestamped}`")
    lines.append(f"- chunks: `{len(metas)}`")
    lines.append("")

    for m in metas:
        rel = m.path.relative_to(run_dir).as_posix()
        if m.start_ts and m.end_ts:
            lines.append(f"- **chunk_{m.index:03d}** [{m.start_ts} → {m.end_ts}] — `{m.line_count} lines` — [{rel}]({rel})")
        else:
            lines.append(f"- **chunk_{m.index:03d}** — `{m.char_count} chars` — [{rel}]({rel})")

    _safe_write_text(toc, "\n".join(lines).strip() + "\n")
    return toc


def write_manifest(
    run_dir: Path,
    *,
    source_path: Path,
    prepared_path: Path,
    timestamped: bool,
    chunk_minutes: int,
    clean: bool,
    clean_mode: str,
    redact_level: Optional[str],
    corrections_path: Optional[Path],
    terms_path: Optional[Path],
    chunk_metas: List[ChunkMeta],
) -> Path:
    manifest = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_path": str(source_path),
        "prepared_path": str(prepared_path),
        "timestamped": bool(timestamped),
        "chunk_minutes": int(chunk_minutes),
        "clean": bool(clean),
        "clean_mode": str(clean_mode),
        "redact_level": redact_level,
        "corrections_path": str(corrections_path) if corrections_path else None,
        "terms_path": str(terms_path) if terms_path else None,
        "chunks": [
            {
                "index": m.index,
                "path": str(m.path),
                "start_s": m.start_s,
                "end_s": m.end_s,
                "start_ts": m.start_ts,
                "end_ts": m.end_ts,
                "line_count": m.line_count,
                "char_count": m.char_count,
            }
            for m in chunk_metas
        ],
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# -----------------------------
# C: AI Scaffold (offline)
# -----------------------------

def write_ai_scaffold(run_dir: Path, chunk_metas: List[ChunkMeta]) -> Path:
    """
    Offline-only: create a predictable place to paste AI outputs later.
    """
    ai_dir = _ensure_dir(run_dir / "ai")
    tasks_path = ai_dir / "tasks.json"
    readme_path = ai_dir / "README.md"

    tasks = []
    for m in chunk_metas:
        tasks.append(
            {
                "chunk": f"chunk_{m.index:03d}",
                "chunk_path": str(m.path),
                "suggested_ops": [
                    "review_and_format",
                    "grammar_cleanup",
                    "summary_bullets",
                    "rewrite_clean",
                    "rewrite_story_draft",
                ],
            }
        )

    tasks_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = []
    readme.append(f"# TextLab AI Scaffold — {run_dir.name}")
    readme.append("")
    readme.append("This folder is **offline scaffolding** for later AI-assisted work.")
    readme.append("")
    readme.append("## Suggested workflow")
    readme.append("1) Open TOC.md and pick a chunk.")
    readme.append("2) Use the chunk text as input to your AI tool of choice.")
    readme.append("3) Save outputs here, e.g.:")
    readme.append("")
    readme.append("- `ai/chunk_000.summary.md`")
    readme.append("- `ai/chunk_000.rewrite.md`")
    readme.append("- `ai/chunk_000.story.md`")
    readme.append("")
    readme.append("## Task list")
    readme.append(f"- See `tasks.json` for per-chunk suggested operations.")
    readme.append("")
    readme_path.write_text("\n".join(readme).strip() + "\n", encoding="utf-8")

    return ai_dir


# -----------------------------
# Public API
# -----------------------------

def run_textlab(
    source: Path,
    out_root: Path,
    *,
    chunk_minutes: int = 5,
    clean: bool = True,
    clean_mode: str = "standard",
    redact_level: Optional[str] = "standard",
    post_fix: Optional[Path] = None,
    post_terms: Optional[Path] = None,
    report: bool = False,  # reserved for future; manifest already exists
    enable_ai: bool = True,  # scaffold is cheap; keep ON by default
) -> Path:
    """
    Offline-first pipeline:
      1) load transcript
      2) apply corrections (post-fix)
      3) clean
      4) redact (built-in + custom patterns/terms)
      5) chunk
      6) write TOC + manifest (+ AI scaffold)

    Returns: run_dir Path
    """
    source = source.expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Transcript not found: {source}")

    out_root = out_root.expanduser().resolve()
    _ensure_dir(out_root)

    # Run dir naming: use base stem before timestamp if present
    stem = source.stem.split("__")[0] if "__" in source.stem else source.stem
    run_dir = _ensure_dir(out_root / f"{stem}__{_now_stamp()}")

    prepared_dir = _ensure_dir(run_dir / "prepared")
    chunks_dir = _ensure_dir(run_dir / "chunks")

    raw_text = _safe_read_text(source)

    # Defaults if caller passes None
    corrections = _load_corrections(post_fix)
    terms_spec = _load_redaction_terms(post_terms)

    prepared = raw_text
    if corrections:
        prepared = apply_corrections(prepared, corrections)

    if clean:
        prepared = clean_text(prepared, clean_mode)

    prepared = redact_text(prepared, redact_level, terms_spec)

    prepared_path = prepared_dir / f"{stem}.prepared.txt"
    _safe_write_text(prepared_path, prepared)

    timestamped = _detect_timestamped(prepared)
    if timestamped:
        metas = _chunk_timestamped(prepared, chunk_minutes, chunks_dir)
    else:
        metas = _chunk_plain(prepared, chunks_dir)

    write_toc(run_dir, metas, timestamped=timestamped)
    write_manifest(
        run_dir,
        source_path=source,
        prepared_path=prepared_path,
        timestamped=timestamped,
        chunk_minutes=chunk_minutes,
        clean=clean,
        clean_mode=clean_mode,
        redact_level=redact_level,
        corrections_path=post_fix if post_fix else None,
        terms_path=post_terms if post_terms else None,
        chunk_metas=metas,
    )

    if enable_ai:
        write_ai_scaffold(run_dir, metas)

    return run_dir
