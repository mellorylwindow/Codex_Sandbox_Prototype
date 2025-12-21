# transcribe_run.py
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure repo root on sys.path so `import transcriber...` works when running:
#   python transcribe_run.py ...
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transcriber import TranscribeOptions, run_transcription  # noqa: E402


DEFAULT_MEDIA_IN_DIR = Path("notes/media_in")
DEFAULT_OUT_DIR = Path("out/transcripts")
DEFAULT_POST_OUT_DIR = Path("out/textops")

DEFAULT_HOTWORDS_FILE = Path("notes/hotwords.txt")
DEFAULT_CORRECTIONS_FILE = Path("notes/corrections.json")
DEFAULT_REDACTION_TERMS_FILE = Path("notes/redaction_terms.json")


# ----------------------------
# CLI
# ----------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Offline-first media transcription (VLC-ish inputs -> txt/json/srt + optional post-processing).",
    )

    p.add_argument(
        "input_media",
        help=(
            "Media file path OR a base name to resolve from notes/media_in.\n"
            "Examples:\n"
            "  transcribe notes/media_in/demo_journal.mp4\n"
            "  transcribe demo_journal\n"
        ),
    )

    p.add_argument("--name", default=None, help="Override output base name (default: input filename).")
    p.add_argument("--in-dir", default=str(DEFAULT_MEDIA_IN_DIR), help=f"Where to resolve name-only inputs (default: {DEFAULT_MEDIA_IN_DIR}).")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help=f"Transcript output folder (default: {DEFAULT_OUT_DIR}).")

    p.add_argument("--mode", choices=["plain", "timestamps"], default="plain", help="Text output mode (default: plain).")

    # Backend / model knobs
    p.add_argument("--backend", default="faster-whisper", help="Backend (default: faster-whisper).")
    p.add_argument("--model", default="base", help="Whisper model (default: base).")
    p.add_argument("--device", default="cpu", help="Device (cpu/cuda) (default: cpu).")
    p.add_argument("--compute-type", default="int8", help="Compute type (default: int8).")
    p.add_argument("--language", default=None, help="Language code (e.g. en). Default: auto-detect.")
    p.add_argument("--beam-size", type=int, default=5, help="Beam size (default: 5).")
    p.add_argument("--vad-filter", action="store_true", default=True, help="Enable VAD filter (default: ON).")
    p.add_argument("--no-vad-filter", dest="vad_filter", action="store_false", help="Disable VAD filter.")

    # Prompt / hotwords
    p.add_argument("--prompt", default=None, help="Optional initial prompt (helps with names/terms).")
    p.add_argument("--prompt-file", default=None, help="Path to a prompt text file.")

    p.add_argument("--hotwords", default=None, help="Comma-separated hotwords (e.g. 'Duran Duran,Gizmo').")
    p.add_argument("--hotwords-file", default=None, help="Path to hotwords.txt (one per line).")

    # Defaults behavior toggles
    p.add_argument("--no-default-hotwords", action="store_true", help=f"Do not auto-load {DEFAULT_HOTWORDS_FILE} if it exists.")
    p.add_argument("--no-default-post-fix", action="store_true", help=f"Do not auto-apply {DEFAULT_CORRECTIONS_FILE} if it exists.")

    # Output files
    p.add_argument("--keep-wav", action="store_true", help="Keep extracted wav in out/transcripts/.scratch_transcribe.")
    p.add_argument("--no-json", action="store_true", help="Do not write segments JSON.")
    p.add_argument("--json", action="store_true", help="Force write segments JSON (even if presets disable it).")

    p.add_argument("--srt", action="store_true", help="Write SRT subtitles.")
    p.add_argument("--vtt", action="store_true", help="Write WebVTT subtitles.")

    # Preset-ish convenience flags (safe no-ops if you prefer tools/transcribe.py)
    p.add_argument("--txt", action="store_true", help="TXT-only (disables json/srt/vtt unless re-enabled).")

    # Post-processing (offline)
    p.add_argument("--post-clean", action="store_true", help="Run offline cleaning on transcript text before saving post output.")
    p.add_argument("--post-clean-mode", choices=["light", "standard"], default="standard", help="Cleaning mode (default: standard).")

    p.add_argument(
        "--post-fix",
        nargs="?",
        const=str(DEFAULT_CORRECTIONS_FILE),
        default=None,
        help="Apply corrections.json (string replacements). If used with no value, defaults to notes/corrections.json.",
    )
    p.add_argument("--no-post-fix", action="store_true", help="Disable corrections for this run.")

    p.add_argument(
        "--post-redact",
        nargs="?",
        const="standard",
        default=None,
        help="Apply redaction (levels: light|standard|heavy). If used with no value, defaults to 'standard'.",
    )
    p.add_argument(
        "--post-terms",
        nargs="?",
        const=str(DEFAULT_REDACTION_TERMS_FILE),
        default=None,
        help="Path to redaction_terms.json. If used with no value, defaults to notes/redaction_terms.json.",
    )
    p.add_argument("--no-post-terms", action="store_true", help="Disable custom redaction terms even if provided/defaulted.")
    p.add_argument("--post-report", action="store_true", help="Write a small JSON report for post-processing.")
    p.add_argument("--post-out-dir", default=str(DEFAULT_POST_OUT_DIR), help=f"Post output folder (default: {DEFAULT_POST_OUT_DIR}).")

    # Latest conveniences
    p.add_argument("--print-latest", action="store_true", help="Print the newest file created by this run.")
    p.add_argument("--open-latest", action="store_true", help="Open the newest file created by this run.")
    p.add_argument("--latest-head", type=int, default=60, help="How many lines to print with --print-latest (default: 60).")
    p.add_argument(
        "--open-target",
        choices=["txt", "json", "srt", "vtt", "post", "dir"],
        default="post",
        help="What to open for --open-latest (default: post).",
    )

    return p


# ----------------------------
# Input resolution
# ----------------------------
def _resolve_input_media(user_value: str, in_dir: Path) -> Path:
    p = Path(user_value)

    # 1) Direct path exists
    if p.exists() and p.is_file():
        return p.expanduser().resolve()

    # 2) Relative to in_dir
    if not p.is_absolute():
        candidate = (in_dir / p).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate

    # 3) Name-only: search in_dir for "<name>.*"
    name = p.stem if p.suffix else p.name
    matches = sorted(
        in_dir.glob(f"{name}.*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0].expanduser().resolve()

    tried = [
        str(p.expanduser().resolve()),
        str((in_dir / p).expanduser().resolve()),
        str((in_dir / f"{name}.*").expanduser().resolve()),
    ]
    raise FileNotFoundError(
        "Could not resolve input media. Tried:\n"
        + "\n".join(f"- {t}" for t in tried)
        + "\n\nTip: put files in notes/media_in/ and call: transcribe <basename>"
    )


# ----------------------------
# Hotwords / prompt utilities
# ----------------------------
def _load_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> Optional[str]:
    if prompt_file:
        p = Path(prompt_file).expanduser()
        if p.exists() and p.is_file():
            txt = p.read_text(encoding="utf-8", errors="replace").strip()
            if txt:
                return txt
    if prompt:
        s = str(prompt).strip()
        return s or None
    return None


def _parse_hotwords_csv(s: str) -> List[str]:
    parts = [x.strip() for x in (s or "").split(",")]
    return [p for p in parts if p]


def _load_hotwords(hotwords: Optional[str], hotwords_file: Optional[str], *, allow_default: bool) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    def add_all(items: Iterable[str]) -> None:
        for it in items:
            w = (it or "").strip()
            if not w:
                continue
            if w in seen:
                continue
            out.append(w)
            seen.add(w)

    # CLI inline
    if hotwords:
        add_all(_parse_hotwords_csv(hotwords))

    # CLI file
    if hotwords_file:
        p = Path(hotwords_file).expanduser()
        if p.exists() and p.is_file():
            add_all([ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()])

    # Default file
    if allow_default and DEFAULT_HOTWORDS_FILE.exists() and DEFAULT_HOTWORDS_FILE.is_file():
        add_all([ln.strip() for ln in DEFAULT_HOTWORDS_FILE.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()])

    return out


# ----------------------------
# Post-processing (offline)
# ----------------------------
_WS_RE = re.compile(r"[ \t]+")

def _clean_text(text: str, mode: str) -> str:
    """
    Offline-only cleaning:
    - normalize newlines
    - collapse excessive spaces
    - lightly fix spacing around punctuation
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in t.split("\n")]
    t = "\n".join(lines).strip() + "\n"

    if mode == "light":
        # just collapse tabs/spaces
        t = "\n".join(_WS_RE.sub(" ", ln).strip() for ln in t.splitlines()) + "\n"
        return t

    # standard
    t = "\n".join(_WS_RE.sub(" ", ln).strip() for ln in t.splitlines())
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([(\[{])\s+", r"\1", t)
    t = re.sub(r"\s+([)\]}])", r"\1", t)
    return t.strip() + "\n"


def _load_corrections_json(path: Path) -> Dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # {"bad":"good", ...}
        out: Dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, str) and k:
                out[k] = v
        return out
    if isinstance(data, list):
        # [{"from":"x","to":"y"}]
        out = {}
        for item in data:
            if isinstance(item, dict):
                a = item.get("from")
                b = item.get("to")
                if isinstance(a, str) and isinstance(b, str) and a:
                    out[a] = b
        return out
    return {}


def _apply_corrections(text: str, corrections: Dict[str, str]) -> Tuple[str, Dict[str, int]]:
    """
    Simple string replace (literal), applied in stable order (longest keys first).
    Returns (new_text, counts_by_key).
    """
    counts: Dict[str, int] = {}
    t = text
    for src in sorted(corrections.keys(), key=len, reverse=True):
        dst = corrections[src]
        if not src:
            continue
        n = t.count(src)
        if n:
            t = t.replace(src, dst)
            counts[src] = n
    return t, counts


_MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_DATE_RE = re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b", re.IGNORECASE)
_AGE_RE = re.compile(r"\b\d{1,3}\s*(?:years?\s*old|yo)\b", re.IGNORECASE)
_WEIGHT_RE_1 = re.compile(r"\b\d{2,3}\s*(?:lb|lbs|pounds)\b", re.IGNORECASE)
_WEIGHT_RE_2 = re.compile(r"(\bweight\s+(?:is|was)\s+(?:about\s+)?)\d{2,3}\b", re.IGNORECASE)
_MYNAME_RE = re.compile(r"(\bmy\s+name\s+is\s+)([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,4})", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?1[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?)\d{3}[\s\-\.]?\d{4}\b")
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")


def _parse_regex_flags(flag_str: str | None) -> int:
    if not flag_str:
        return 0
    fs = flag_str.lower()
    flags = 0
    if "i" in fs:
        flags |= re.IGNORECASE
    if "m" in fs:
        flags |= re.MULTILINE
    if "s" in fs:
        flags |= re.DOTALL
    return flags


def _load_redaction_terms(path: Optional[Path]) -> Tuple[List[Tuple[re.Pattern, str]], List[Tuple[str, str]]]:
    """
    Returns:
      - compiled_patterns: [(regex, label)]
      - literal_terms: [(literal, label)]

    Schema tolerance:
      - patterns: [{label, regex|pattern, flags?}, ...]
      - terms: [{label, terms:[...]}] OR [{label, value:"..."}] OR ["literal", ...]
    """
    compiled: List[Tuple[re.Pattern, str]] = []
    literals: List[Tuple[str, str]] = []

    if not path or not path.exists() or not path.is_file():
        return compiled, literals

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        patterns = data.get("patterns", [])
        terms = data.get("terms", [])

        # patterns
        if isinstance(patterns, list):
            for i, item in enumerate(patterns):
                if isinstance(item, str):
                    # bare regex string
                    try:
                        compiled.append((re.compile(item), "CUSTOM"))
                    except re.error:
                        continue
                    continue
                if isinstance(item, dict):
                    label = str(item.get("label") or item.get("name") or "CUSTOM")
                    rx = item.get("regex", None)
                    if rx is None:
                        rx = item.get("pattern", None)  # tolerate "pattern"
                    if not isinstance(rx, str) or not rx.strip():
                        # skip invalid entry (don’t explode)
                        continue
                    flags = _parse_regex_flags(item.get("flags"))
                    try:
                        compiled.append((re.compile(rx, flags), label))
                    except re.error:
                        continue

        # terms
        if isinstance(terms, list):
            for item in terms:
                if isinstance(item, str):
                    literals.append((item, "CUSTOM"))
                    continue
                if isinstance(item, dict):
                    label = str(item.get("label") or item.get("name") or "CUSTOM")
                    if isinstance(item.get("value"), str):
                        literals.append((item["value"], label))
                    elif isinstance(item.get("term"), str):
                        literals.append((item["term"], label))
                    elif isinstance(item.get("terms"), list):
                        for t in item["terms"]:
                            if isinstance(t, str) and t.strip():
                                literals.append((t, label))

        return compiled, literals

    # If they gave a list at top-level, treat strings as literal terms.
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.strip():
                literals.append((item, "CUSTOM"))
            elif isinstance(item, dict):
                label = str(item.get("label") or "CUSTOM")
                rx = item.get("regex") or item.get("pattern")
                if isinstance(rx, str) and rx.strip():
                    try:
                        compiled.append((re.compile(rx, _parse_regex_flags(item.get("flags"))), label))
                    except re.error:
                        pass
        return compiled, literals

    return compiled, literals


def _apply_redaction(
    text: str,
    level: str,
    *,
    terms_file: Optional[Path],
) -> Tuple[str, Dict[str, int]]:
    """
    Applies built-in + custom redactions.
    Returns (new_text, counts_by_label).
    """
    counts: Dict[str, int] = {}
    t = text

    # Built-ins
    def sub_count(rx: re.Pattern, repl: str, label: str) -> None:
        nonlocal t
        t2, n = rx.subn(repl, t)
        if n:
            t = t2
            counts[label] = counts.get(label, 0) + n

    # Always redact obvious PII-ish
    sub_count(_EMAIL_RE, "[EMAIL]", "EMAIL")
    sub_count(_PHONE_RE, "[PHONE]", "PHONE")

    if level in ("light", "standard", "heavy"):
        sub_count(_DATE_RE, "[DATE]", "DATE")

    if level in ("standard", "heavy"):
        # Name line
        t2, n = _MYNAME_RE.subn(r"\1[NAME]", t)
        if n:
            t = t2
            counts["NAME"] = counts.get("NAME", 0) + n

        sub_count(_AGE_RE, "[AGE]", "AGE")
        sub_count(_WEIGHT_RE_1, "[WEIGHT]", "WEIGHT")
        # "weight is about 236" -> keep phrase + replace number
        t2, n = _WEIGHT_RE_2.subn(r"\1[WEIGHT]", t)
        if n:
            t = t2
            counts["WEIGHT"] = counts.get("WEIGHT", 0) + n

    if level == "heavy":
        # Example: redact years (can be noisy, so only heavy)
        year_re = re.compile(r"\b(19|20)\d{2}\b")
        sub_count(year_re, "[YEAR]", "YEAR")

    # Custom terms + patterns
    compiled, literals = _load_redaction_terms(terms_file)

    # Literals: longest first
    for lit, label in sorted(literals, key=lambda x: len(x[0]), reverse=True):
        if not lit:
            continue
        n = t.count(lit)
        if n:
            t = t.replace(lit, f"[{label}]")
            counts[label] = counts.get(label, 0) + n

    # Patterns
    for rx, label in compiled:
        t2, n = rx.subn(f"[{label}]", t)
        if n:
            t = t2
            counts[label] = counts.get(label, 0) + n

    return t, counts


def _run_post_processing(
    *,
    transcript_txt_path: Path,
    base_name: str,
    out_dir: Path,
    do_clean: bool,
    clean_mode: str,
    corrections_path: Optional[Path],
    redact_level: Optional[str],
    terms_path: Optional[Path],
    write_report: bool,
) -> Tuple[Path, Optional[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = transcript_txt_path.read_text(encoding="utf-8", errors="replace")

    report: Dict[str, Any] = {
        "input": str(transcript_txt_path),
        "post_out_dir": str(out_dir),
        "base_name": base_name,
        "steps": [],
    }

    t = raw

    if do_clean:
        t = _clean_text(t, mode=clean_mode)
        report["steps"].append({"clean": {"mode": clean_mode}})

    if corrections_path:
        corrections = _load_corrections_json(corrections_path)
        t, counts = _apply_corrections(t, corrections)
        report["steps"].append({"post_fix": {"path": str(corrections_path), "replacements": counts}})

    if redact_level:
        t, counts = _apply_redaction(t, redact_level, terms_file=terms_path)
        report["steps"].append({"redact": {"level": redact_level, "terms_path": str(terms_path) if terms_path else None, "hits": counts}})

    post_path = out_dir / f"{base_name}.post.txt"
    post_path.write_text(t.rstrip() + "\n", encoding="utf-8")

    report_path: Optional[Path] = None
    if write_report:
        report_path = out_dir / f"{base_name}.post.report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return post_path, report_path


# ----------------------------
# Open / print helpers
# ----------------------------
def _open_path(path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot open missing path: {path}")

    system = platform.system().lower()
    if system.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if system == "darwin":
        subprocess.check_call(["open", str(path)])
        return
    subprocess.check_call(["xdg-open", str(path)])


def _print_head(path: Path, head: int) -> None:
    txt = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in txt[: max(0, head)]:
        print(line)


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    args = _build_parser().parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    post_out_dir = Path(args.post_out_dir).expanduser().resolve()

    input_media = _resolve_input_media(args.input_media, in_dir=in_dir)

    # Preset-ish: --txt disables extras unless user re-enables
    write_segments_json = not args.no_json
    write_srt = bool(args.srt)
    write_vtt = bool(args.vtt)
    if args.txt:
        write_segments_json = False
        write_srt = False
        write_vtt = False
    if args.json:
        write_segments_json = True

    # Build immutable options
    opts = TranscribeOptions(
        backend=args.backend,
        model=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        vad_filter=bool(args.vad_filter),
        beam_size=int(args.beam_size),
    )

    # Prompt
    prompt = _load_prompt(args.prompt, args.prompt_file)
    if prompt:
        opts = opts.with_prompt(prompt)

    # Hotwords (defaults ON unless disabled)
    hotwords = _load_hotwords(
        args.hotwords,
        args.hotwords_file,
        allow_default=not bool(args.no_default_hotwords),
    )
    if hotwords:
        opts = opts.with_hotwords(hotwords)

    # Transcribe
    txt_path = run_transcription(
        input_media=input_media,
        out_dir=out_dir,
        opts=opts,
        keep_wav=bool(args.keep_wav),
        write_segments_json=bool(write_segments_json),
        write_srt=bool(write_srt),
        name=args.name,
        mode=args.mode,
        # transcriber/transcribe_job.py may support vtt in your branch;
        # we write vtt here ourselves only if a .vtt exists is supported by job.
        # If your job supports vtt via write_vtt kw, we pass it; else ignore safely.
        **({"write_vtt": bool(write_vtt)} if "write_vtt" in run_transcription.__code__.co_varnames else {}),
    )

    print(f"✅ Transcript written: {txt_path}")

    created: Dict[str, Path] = {"txt": txt_path}

    # Companion outputs
    base_name = txt_path.stem  # includes __timestamp
    seg_json = out_dir / f"{base_name}.segments.json"
    if write_segments_json and seg_json.exists():
        created["json"] = seg_json

    srt_path = out_dir / f"{base_name}.srt"
    if write_srt and srt_path.exists():
        created["srt"] = srt_path

    vtt_path = out_dir / f"{base_name}.vtt"
    if write_vtt and vtt_path.exists():
        created["vtt"] = vtt_path

    # Decide post defaults
    auto_corrections = (DEFAULT_CORRECTIONS_FILE.exists() and DEFAULT_CORRECTIONS_FILE.is_file() and not args.no_default_post_fix)
    do_post_fix = (not args.no_post_fix) and (args.post_fix is not None or auto_corrections)

    corrections_path: Optional[Path] = None
    if do_post_fix:
        use_path = None
        if args.post_fix is not None:
            # if user used --post-fix without arg, argparse gives const path string
            use_path = str(args.post_fix).strip() if isinstance(args.post_fix, str) else None
        if not use_path and auto_corrections:
            use_path = str(DEFAULT_CORRECTIONS_FILE)
        if use_path:
            cp = Path(use_path).expanduser()
            if cp.exists() and cp.is_file():
                corrections_path = cp.resolve()

    # Redact level
    redact_level: Optional[str] = None
    if args.post_redact is not None:
        lvl = str(args.post_redact).strip().lower()
        if lvl in ("light", "standard", "heavy"):
            redact_level = lvl
        else:
            # tolerate accidental "none"
            redact_level = None

    # Terms file
    terms_path: Optional[Path] = None
    if not args.no_post_terms:
        terms_arg = args.post_terms
        if terms_arg is None and redact_level:
            # if redaction enabled and no explicit terms passed, try default terms file
            if DEFAULT_REDACTION_TERMS_FILE.exists() and DEFAULT_REDACTION_TERMS_FILE.is_file():
                terms_path = DEFAULT_REDACTION_TERMS_FILE.resolve()
        elif isinstance(terms_arg, str) and terms_arg.strip():
            tp = Path(terms_arg).expanduser()
            if tp.exists() and tp.is_file():
                terms_path = tp.resolve()

    do_any_post = bool(args.post_clean) or bool(corrections_path) or bool(redact_level)

    post_path: Optional[Path] = None
    report_path: Optional[Path] = None
    if do_any_post:
        post_path, report_path = _run_post_processing(
            transcript_txt_path=txt_path,
            base_name=base_name,
            out_dir=post_out_dir,
            do_clean=bool(args.post_clean),
            clean_mode=str(args.post_clean_mode),
            corrections_path=corrections_path,
            redact_level=redact_level,
            terms_path=terms_path,
            write_report=bool(args.post_report),
        )
        created["post"] = post_path
        if report_path:
            created["post_report"] = report_path
        print(f"✅ Post output written: {post_path}")

    # Always show what we created (super helpful when scripting)
    # This is intentionally noisy (you asked for “show exact post file every run”)
    created_order = ["txt", "json", "srt", "vtt", "post", "post_report"]
    for k in created_order:
        if k in created:
            print(f"   - {k}: {created[k]}")

    # Latest UX: choose target
    if args.print_latest or args.open_latest:
        target = args.open_target

        chosen: Optional[Path] = None
        if target == "dir":
            chosen = (post_out_dir if post_path else out_dir).resolve()
        else:
            # prefer requested kind
            if target in created:
                chosen = created[target]
            elif target == "post" and "post" in created:
                chosen = created["post"]
            else:
                # fallback priority: post -> txt
                chosen = created.get("post") or created.get("txt")

        if chosen is None:
            # Should never happen, but be defensive.
            return 0

        if args.print_latest:
            print("")
            print(f"📌 Latest ({target}): {chosen}")
            if chosen.is_file():
                _print_head(chosen, head=int(args.latest_head))
            else:
                print(f"(Directory) {chosen}")

        if args.open_latest:
            _open_path(chosen)
            print(f"📂 Opened: {chosen}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
