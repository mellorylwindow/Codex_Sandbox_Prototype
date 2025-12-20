# textlab/pipeline.py
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from textlab.analyze import extractive_summary, naive_entities, top_keywords
from textlab.chunk import chunk_by_chars, chunk_by_minutes, write_chunks


# -----------------------------------------------------------------------------
# Manifest model
# -----------------------------------------------------------------------------
@dataclass
class TextLabRun:
    started_at: str
    input_path: str
    work_dir: str
    base: str
    options: Dict[str, object]
    outputs: Dict[str, str]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_base(name: str) -> str:
    s = (name or "").strip().replace(" ", "_")
    safe = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))
    return safe or "text"


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _write_text(p: Path, txt: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8", newline="\n")


def _maybe_load_json(p: Optional[Path]) -> Optional[object]:
    if not p:
        return None
    p = Path(p)
    if not p.exists() or not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8", errors="replace"))


# -----------------------------------------------------------------------------
# Timestamp inference helpers (for TOC + per-chunk meta)
# -----------------------------------------------------------------------------
# Supports [MM:SS] or [HH:MM:SS]
_TS_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]")


def _ts_to_seconds(a: int, b: int, c_opt: Optional[int]) -> int:
    # If c_opt is None, interpret as MM:SS (a=MM, b=SS)
    if c_opt is None:
        return a * 60 + b
    # Else interpret as HH:MM:SS (a=HH, b=MM, c=SS)
    return a * 3600 + b * 60 + c_opt


def _seconds_to_ts(total_s: int) -> str:
    if total_s < 0:
        total_s = 0
    hh = total_s // 3600
    rem = total_s % 3600
    mm = rem // 60
    ss = rem % 60
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def _infer_time_window_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Look for [MM:SS] or [HH:MM:SS] markers inside text.
    Returns (start_ts, end_ts) strings or (None, None).
    """
    matches = list(_TS_RE.finditer(text))
    if not matches:
        return None, None

    times_s: List[int] = []
    for m in matches:
        a = int(m.group(1))
        b = int(m.group(2))
        c = int(m.group(3)) if m.group(3) is not None else None
        times_s.append(_ts_to_seconds(a, b, c))

    times_s.sort()
    return _seconds_to_ts(times_s[0]), _seconds_to_ts(times_s[-1])


# -----------------------------------------------------------------------------
# Offline prep steps (no dependency on textops_run.py CLI flags)
# -----------------------------------------------------------------------------
_WS_RE = re.compile(r"[ \t]+")


def clean_text(text: str, mode: str = "standard") -> str:
    """
    Lightweight offline cleanup. Conservative by design:
      - normalize newlines
      - trim trailing whitespace
      - collapse repeated spaces/tabs
      - collapse extreme blank-line runs (standard only)

    mode:
      - "light": keep more original formatting
      - "standard": slightly more normalization
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [(_WS_RE.sub(" ", ln)).rstrip() for ln in text.split("\n")]

    if mode == "light":
        return "\n".join(lines).strip() + "\n"

    out: List[str] = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                out.append("")
        else:
            blank_run = 0
            out.append(ln)
    return "\n".join(out).strip() + "\n"


def _apply_literal_replacements(text: str, mapping: Dict[str, str]) -> str:
    """
    Apply string replacements deterministically (longest keys first).
    """
    if not mapping:
        return text
    items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    for src, dst in items:
        if not src:
            continue
        text = text.replace(src, dst)
    return text


def apply_postfix_corrections(text: str, corrections_path: Optional[Path]) -> str:
    """
    Deterministic post-fix corrections based on notes/corrections.json.

    Expected format:
      {
        "Duran Iran": "Duran Duran",
        "Gismo": "Gizmo"
      }
    """
    data = _maybe_load_json(corrections_path)
    if not isinstance(data, dict):
        return text

    mapping: Dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str) and k:
            mapping[k] = v
    return _apply_literal_replacements(text, mapping)


@dataclass(frozen=True)
class RedactionPattern:
    label: str
    regex: re.Pattern
    replacement: str


def _parse_flags(flag_spec: object) -> int:
    """
    Accepts:
      - "IGNORECASE|MULTILINE"
      - ["IGNORECASE", "MULTILINE"]
      - int bitmask (advanced)
    """
    if flag_spec is None:
        return 0
    if isinstance(flag_spec, int):
        return int(flag_spec)

    flags = 0
    names: List[str] = []
    if isinstance(flag_spec, str):
        names = [p.strip().upper() for p in flag_spec.split("|") if p.strip()]
    elif isinstance(flag_spec, list):
        names = [str(x).strip().upper() for x in flag_spec if str(x).strip()]

    for n in names:
        if n in ("I", "IGNORECASE"):
            flags |= re.IGNORECASE
        elif n in ("M", "MULTILINE"):
            flags |= re.MULTILINE
        elif n in ("S", "DOTALL"):
            flags |= re.DOTALL
        elif n in ("X", "VERBOSE"):
            flags |= re.VERBOSE

    return flags


def _load_terms_patterns(terms_path: Optional[Path]) -> List[RedactionPattern]:
    """
    Flexible loader. Accepts multiple JSON shapes.

    Preferred shape:
      {
        "patterns": [
          {"label": "NAME", "regex": "...", "replacement": "[NAME]", "flags": "IGNORECASE"}
        ],
        "literals": [
          {"text": "Jimmy Swain", "replacement": "[NAME]"}
        ]
      }

    Also accepts a legacy list form:
      [
        {"label": "...", "regex": "...", "replacement": "..."},
        ...
      ]
    """
    data = _maybe_load_json(terms_path)
    patterns: List[RedactionPattern] = []

    def add_pattern(label: str, regex_s: str, replacement: str, flags_spec: object = None) -> None:
        if not regex_s:
            return
        flags = _parse_flags(flags_spec)
        try:
            compiled = re.compile(regex_s, flags)
        except re.error:
            return
        patterns.append(RedactionPattern(label=label or "CUSTOM", regex=compiled, replacement=replacement))

    if isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                continue
            add_pattern(
                label=str(obj.get("label") or obj.get("name") or "CUSTOM"),
                regex_s=str(obj.get("regex") or ""),
                replacement=str(obj.get("replacement") or obj.get("replace") or "[REDACTED]"),
                flags_spec=obj.get("flags"),
            )
        return patterns

    if isinstance(data, dict):
        lits = data.get("literals")
        if isinstance(lits, list):
            for obj in lits:
                if not isinstance(obj, dict):
                    continue
                src = obj.get("text")
                rep = obj.get("replacement", "[REDACTED]")
                if isinstance(src, str) and src:
                    add_pattern(
                        label=str(obj.get("label") or "LITERAL"),
                        regex_s=re.escape(src),
                        replacement=str(rep),
                        flags_spec=obj.get("flags"),
                    )

        pats = data.get("patterns")
        if isinstance(pats, list):
            for obj in pats:
                if not isinstance(obj, dict):
                    continue
                add_pattern(
                    label=str(obj.get("label") or obj.get("name") or "CUSTOM"),
                    regex_s=str(obj.get("regex") or ""),
                    replacement=str(obj.get("replacement") or obj.get("replace") or "[REDACTED]"),
                    flags_spec=obj.get("flags"),
                )

    return patterns


def _builtin_redaction_patterns(level: str) -> List[RedactionPattern]:
    """
    Built-in offline patterns. Intentionally simple.
    Use your custom terms JSON for project-specific patterns.

    levels:
      - light: phone/email only
      - standard: + "my name is", ages, dates, weights
      - heavy: + aggressive number patterns
    """
    level = (level or "standard").strip().lower()
    pats: List[RedactionPattern] = []

    def add(label: str, regex_s: str, repl: str, flags: int = 0) -> None:
        pats.append(RedactionPattern(label=label, regex=re.compile(regex_s, flags), replacement=repl))

    add("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]")
    add("PHONE", r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b", "[PHONE]")

    if level in ("standard", "heavy"):
        add("NAME_IS", r"\bmy name is\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", "my name is [NAME]", re.IGNORECASE)
        add("AGE", r"\b(?:i'?m|i am)\s+\d{1,3}\s+years?\s+old\b", "I'm [AGE]", re.IGNORECASE)
        add(
            "DATE_LONG",
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\,?\s+\d{4}\b",
            "[DATE]",
            re.IGNORECASE,
        )
        add("DATE_SLASH", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]")
        add("YEAR", r"\b(19|20)\d{2}\b", "[YEAR]")
        add("WEIGHT", r"\b(?:my\s+)?weight\s+is\s+\d{2,4}\b", "My weight is [WEIGHT]", re.IGNORECASE)

    if level == "heavy":
        add("NUMBERS", r"\b\d{2,}\b", "[NUMBER]")

    return pats


def redact_text(
    text: str,
    *,
    level: str = "standard",
    terms_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> str:
    """
    Apply built-in patterns + custom patterns loaded from terms JSON.
    Writes an optional report with counts by label.
    """
    patterns: List[RedactionPattern] = []
    patterns.extend(_builtin_redaction_patterns(level))
    patterns.extend(_load_terms_patterns(terms_path))

    counts: Counter[str] = Counter()

    for p in patterns:
        try:
            hits = len(list(p.regex.finditer(text)))
        except Exception:
            hits = 0
        if hits:
            counts[p.label] += hits
            text = p.regex.sub(p.replacement, text)

    if report_path:
        lines = ["# redaction report", ""]
        total = sum(counts.values())
        lines.append(f"total_matches\t{total}")
        lines.append("")
        for label, cnt in counts.most_common():
            lines.append(f"{label}\t{cnt}")
        _write_text(report_path, "\n".join(lines).rstrip() + "\n")

    return text


# -----------------------------------------------------------------------------
# Public pipeline API used by tools/textlab_cli.py
# -----------------------------------------------------------------------------
def run_textlab(
    *,
    input_txt: Path,
    out_root: Path,
    name: Optional[str] = None,
    # prep
    clean: bool = True,
    redact: Optional[str] = "standard",
    terms: Optional[Path] = Path("notes/redaction_terms.json"),
    report: bool = False,
    post_fix: Optional[Path] = Path("notes/corrections.json"),
    clean_mode: str = "standard",
    # chunking
    chunk_minutes: Optional[int] = 5,
    chunk_chars: int = 4000,
    # analysis
    write_keywords: bool = True,
    write_entities: bool = True,
    write_summary: bool = True,
) -> Path:
    input_txt = Path(input_txt).expanduser().resolve()
    if not input_txt.exists():
        raise FileNotFoundError(f"Input not found: {input_txt}")

    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    stamp = _now_stamp()
    base = _safe_base(name or input_txt.stem.split("__")[0])
    work_dir = out_root / f"{base}__{stamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Stage files
    stage_dir = work_dir / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)

    original_p = stage_dir / "01.original.txt"
    cleaned_p = stage_dir / "02.cleaned.txt"
    fixed_p = stage_dir / "03.fixed.txt"
    redacted_p = stage_dir / "04.redacted.txt"
    redaction_report_p = (work_dir / "redaction_report.txt") if report else None

    original_text = _read_text(input_txt)
    _write_text(original_p, original_text)

    current_text = original_text
    current_path = original_p

    # CLEAN
    if clean:
        current_text = clean_text(current_text, mode=clean_mode)
        _write_text(cleaned_p, current_text)
        current_path = cleaned_p

    # POST-FIX
    if post_fix and Path(post_fix).exists():
        current_text = apply_postfix_corrections(current_text, post_fix)
        _write_text(fixed_p, current_text)
        current_path = fixed_p

    # REDACT
    if redact is not None:
        level = (redact or "standard").strip().lower()
        terms_path = terms if (terms and Path(terms).exists()) else None
        current_text = redact_text(
            current_text,
            level=level,
            terms_path=terms_path,
            report_path=redaction_report_p,
        )
        _write_text(redacted_p, current_text)
        current_path = redacted_p

    prepared = current_path

    # CHUNK
    if chunk_minutes is not None and int(chunk_minutes) > 0:
        chunks = chunk_by_minutes(current_text, minutes=int(chunk_minutes))
        chunk_mode = f"minutes:{int(chunk_minutes)}"
    else:
        chunks = chunk_by_chars(current_text, max_chars=int(chunk_chars))
        chunk_mode = f"chars:{int(chunk_chars)}"

    # Safety: chunkers should never return empty unless input is empty.
    if not chunks:
        if current_text.strip():
            chunks = [current_text]
            chunk_mode = f"{chunk_mode} (fallback:single)"
        else:
            raise ValueError("TextLab: prepared text is empty; cannot chunk.")

    chunk_dir = work_dir / "chunks"
    chunk_paths = write_chunks(chunk_dir, base=base, chunks=chunks)

    if not chunk_paths:
        raise RuntimeError("TextLab: write_chunks returned no chunk files; cannot build TOC/meta.")

    # Per-chunk meta + TOC (offline-first)
    chunk_meta_dir = work_dir / "chunk_meta"
    chunk_meta_dir.mkdir(parents=True, exist_ok=True)

    toc_lines: List[str] = []
    toc_lines.append(f"# TextLab TOC — {base}")
    toc_lines.append("")
    toc_lines.append(f"- source: `{input_txt.name}`")
    toc_lines.append(f"- created: `{datetime.now().isoformat(timespec='seconds')}`")
    toc_lines.append(f"- chunking: `{chunk_mode}`")
    toc_lines.append("")

    for cp in chunk_paths:
        ctext = _read_text(cp)
        start_ts, end_ts = _infer_time_window_from_text(ctext)

        kw10 = top_keywords(ctext, k=10)
        ent10 = naive_entities(ctext, k=10)
        summ2 = extractive_summary(ctext, max_sentences=2).strip()

        meta = {
            "chunk_file": cp.name,
            "chunk_path": str(cp),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "chars": len(ctext),
            "lines": len(ctext.splitlines()),
            "keywords_top10": [{"term": w, "count": c} for w, c in kw10],
            "entities_top10": [{"entity": e, "count": c} for e, c in ent10],
            "summary_2sent": summ2,
        }

        meta_path = chunk_meta_dir / f"{cp.stem}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        window = ""
        if start_ts and end_ts:
            window = f" ({start_ts} → {end_ts})"

        # GitHub-friendly clickable link within the run folder
        toc_lines.append(f"- **[{cp.name}](chunks/{cp.name})**{window}")
        if summ2:
            toc_lines.append(f"  - {summ2}")
        toc_lines.append("")

    toc_path = work_dir / "TOC.md"
    _write_text(toc_path, "\n".join(toc_lines).rstrip() + "\n")

    # ANALYSIS
    outputs: Dict[str, str] = {
        "prepared_txt": str(prepared),
        "chunks_dir": str(chunk_dir),
        "chunk_meta_dir": str(chunk_meta_dir),
        "toc": str(toc_path),
        "manifest": str(work_dir / "manifest.json"),
        "stage_dir": str(stage_dir),
    }

    # Outline (kept for backwards compatibility / quick skim)
    outline_p = work_dir / "outline.txt"
    outline_lines: List[str] = []
    for cp in chunk_paths:
        head = _read_text(cp).splitlines()
        if head and head[0].startswith("# chunk"):
            outline_lines.append(head[0].lstrip("# ").strip())
        else:
            outline_lines.append(cp.name)
    _write_text(outline_p, "\n".join(outline_lines).rstrip() + "\n")
    outputs["outline"] = str(outline_p)

    if write_keywords:
        kw = top_keywords(current_text, k=40)
        p = work_dir / "keywords.txt"
        _write_text(p, "\n".join(f"{w}\t{c}" for w, c in kw).rstrip() + "\n")
        outputs["keywords"] = str(p)

    if write_entities:
        ents = naive_entities(current_text, k=40)
        p = work_dir / "entities.txt"
        _write_text(p, "\n".join(f"{e}\t{c}" for e, c in ents).rstrip() + "\n")
        outputs["entities"] = str(p)

    if write_summary:
        summ = extractive_summary(current_text, max_sentences=10)
        p = work_dir / "summary_extractive.txt"
        _write_text(p, summ.strip() + "\n")
        outputs["summary_extractive"] = str(p)

    if report and redaction_report_p:
        outputs["redaction_report"] = str(redaction_report_p)

    run = TextLabRun(
        started_at=datetime.now().isoformat(timespec="seconds"),
        input_path=str(input_txt),
        work_dir=str(work_dir),
        base=base,
        options={
            "clean": clean,
            "clean_mode": clean_mode,
            "post_fix": str(post_fix) if post_fix else None,
            "redact": redact,
            "terms": str(terms) if terms else None,
            "report": report,
            "chunk_mode": chunk_mode,
        },
        outputs=outputs,
    )

    (work_dir / "manifest.json").write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
    return work_dir
