# transcribe_run.py
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from transcriber import TranscribeOptions, run_transcription

TextMode = Literal["plain", "timestamps"]


# -----------------------------
# Small utils
# -----------------------------
def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def _maybe_open_file(path: Path) -> None:
    """Best-effort open with OS default."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
            return
        subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        # non-fatal
        pass


def _head(path: Path, n_lines: int) -> str:
    if n_lines <= 0:
        return ""
    lines = _read_text(path).splitlines()
    return "\n".join(lines[:n_lines]).rstrip() + ("\n" if lines else "")


# -----------------------------
# Deterministic post-fix (teach corrections)
# -----------------------------
@dataclass(frozen=True)
class FixRule:
    src: str
    dst: str
    whole_word: bool = True
    case_sensitive: bool = False


def _is_simple_word(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_'\-]+", s))


def _load_fix_rules(path: Path) -> List[FixRule]:
    """
    notes/corrections.json
    {
      "version": 1,
      "replacements": [
        {"from": "Gismo", "to": "Gizmo", "whole_word": true, "case_sensitive": false}
      ]
    }
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Corrections file not found: {path}")

    data = json.loads(_read_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"Corrections JSON must be an object: {path}")

    repls = data.get("replacements", [])
    if not isinstance(repls, list):
        raise ValueError(f"'replacements' must be a list in: {path}")

    rules: List[FixRule] = []
    for i, item in enumerate(repls):
        if not isinstance(item, dict):
            raise ValueError(f"Replacement at index {i} must be an object in: {path}")
        src = item.get("from")
        dst = item.get("to")
        if not isinstance(src, str) or not src.strip():
            raise ValueError(f"Replacement at index {i} missing valid 'from' in: {path}")
        if not isinstance(dst, str):
            raise ValueError(f"Replacement at index {i} missing valid 'to' in: {path}")

        rules.append(
            FixRule(
                src=src.strip(),
                dst=dst,
                whole_word=bool(item.get("whole_word", True)),
                case_sensitive=bool(item.get("case_sensitive", False)),
            )
        )
    return rules


def _apply_fix_rules(text: str, rules: Sequence[FixRule]) -> Tuple[str, List[Dict[str, Any]]]:
    out = text
    report: List[Dict[str, Any]] = []

    for r in rules:
        flags = 0 if r.case_sensitive else re.IGNORECASE
        if r.whole_word and _is_simple_word(r.src):
            pattern = r"\b" + re.escape(r.src) + r"\b"
        else:
            pattern = re.escape(r.src)

        out, n = re.subn(pattern, r.dst, out, flags=flags)
        if n:
            report.append({"from": r.src, "to": r.dst, "count": n})

    return out, report


# -----------------------------
# Redaction (offline, file-driven)
# -----------------------------
@dataclass(frozen=True)
class RedactRule:
    label: str
    terms: Tuple[str, ...] = ()
    regex: Optional[str] = None
    replacement: Optional[str] = None
    case_sensitive: bool = False
    whole_word: bool = True


def _load_redaction_terms(path: Path) -> List[RedactRule]:
    """
    Supports:
      {
        "terms": {
          "NAME": ["Jimmy Swain", "Carolynn"],
          "DATE": ["October 29th, 2024"]
        },
        "patterns": [
          {"label": "AGE", "regex": "\\b\\d{1,3}\\s*years\\s*old\\b", "replacement": "[AGE]"},
          {"label": "WEIGHT", "regex": "\\b\\d{2,3}\\s*(lb|lbs|pounds)\\b", "replacement": "[WEIGHT]"}
        ]
      }

    Back-compat: any "patterns" entry without "regex" is ignored (not fatal).
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Redaction terms file not found: {path}")

    data = json.loads(_read_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"Redaction terms JSON must be an object: {path}")

    rules: List[RedactRule] = []

    # Term buckets
    terms_obj = data.get("terms", {})
    if isinstance(terms_obj, dict):
        for label, term_list in terms_obj.items():
            if not isinstance(label, str) or not label.strip():
                continue
            if not isinstance(term_list, list):
                continue
            clean_terms = tuple(str(t).strip() for t in term_list if str(t).strip())
            if clean_terms:
                rules.append(
                    RedactRule(
                        label=label.strip(),
                        terms=clean_terms,
                        regex=None,
                        replacement=None,
                        case_sensitive=False,
                        whole_word=True,
                    )
                )

    # Regex patterns
    patterns_obj = data.get("patterns", [])
    if isinstance(patterns_obj, list):
        for item in patterns_obj:
            if not isinstance(item, dict):
                continue
            if "regex" not in item:
                # back-compat: ignore older pattern entries that aren't regex-based
                continue

            regex = item.get("regex")
            if not isinstance(regex, str) or not regex.strip():
                continue

            label = item.get("label") or item.get("name") or "PATTERN"
            replacement = item.get("replacement")

            if not isinstance(label, str) or not label.strip():
                label = "PATTERN"
            if replacement is not None and not isinstance(replacement, str):
                raise ValueError(f"Pattern replacement for {label} must be a string (or omitted).")

            rules.append(
                RedactRule(
                    label=str(label).strip(),
                    regex=regex,
                    replacement=replacement,
                    case_sensitive=bool(item.get("case_sensitive", False)),
                    whole_word=bool(item.get("whole_word", False)),
                )
            )

    return rules


def _apply_redactions(text: str, rules: Sequence[RedactRule]) -> Tuple[str, List[Dict[str, Any]]]:
    out = text
    report: List[Dict[str, Any]] = []

    for rule in rules:
        flags = 0 if rule.case_sensitive else re.IGNORECASE
        repl = rule.replacement or f"[{rule.label}]"

        if rule.regex:
            out, n = re.subn(rule.regex, repl, out, flags=flags)
            if n:
                report.append({"label": rule.label, "type": "regex", "count": n, "regex": rule.regex})
            continue

        if rule.terms:
            total = 0
            for term in rule.terms:
                if rule.whole_word and _is_simple_word(term):
                    pattern = r"\b" + re.escape(term) + r"\b"
                else:
                    pattern = re.escape(term)
                out, n = re.subn(pattern, repl, out, flags=flags)
                total += n
            if total:
                report.append({"label": rule.label, "type": "terms", "count": total})

    return out, report


# -----------------------------
# Clean
# -----------------------------
def _clean_text_standard(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"


# -----------------------------
# Post-processing orchestration
# -----------------------------
def _post_out_name(
    *,
    transcript_path: Path,
    did_fix: bool,
    did_clean: bool,
    redact_profile: Optional[str],
) -> str:
    stem = transcript_path.stem
    parts: List[str] = []
    if did_fix:
        parts.append("fix")
    if did_clean:
        parts.append("clean-standard")
    if redact_profile:
        parts.append(f"redact-{redact_profile}")
    suffix = "__".join(parts) if parts else "post"
    return f"{stem}.{suffix}.txt"


def _run_post_processing(
    *,
    transcript_path: Path,
    out_dir: Path,
    fix_path: Optional[Path],
    do_clean: bool,
    redact_profile: Optional[str],
    terms_path: Optional[Path],
    write_report: bool,
) -> Tuple[Path, Optional[Path]]:
    text = _read_text(transcript_path)

    did_fix = False
    did_clean = False
    fix_report: List[Dict[str, Any]] = []
    redact_report: List[Dict[str, Any]] = []

    # 1) Fix
    if fix_path is not None:
        rules = _load_fix_rules(fix_path)
        text, fix_report = _apply_fix_rules(text, rules)
        did_fix = bool(fix_report)

    # 2) Clean
    if do_clean:
        text = _clean_text_standard(text)
        did_clean = True

    # 3) Redact
    if redact_profile is not None:
        if redact_profile != "standard":
            raise ValueError(f"Unsupported redaction profile: {redact_profile}")

        if terms_path is None:
            raise ValueError("--post-redact requires --post-terms (or rely on its default).")

        rules = _load_redaction_terms(terms_path)
        text, redact_report = _apply_redactions(text, rules)

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = _post_out_name(
        transcript_path=transcript_path,
        did_fix=did_fix,
        did_clean=did_clean,
        redact_profile=redact_profile,
    )
    out_txt = out_dir / out_name
    _write_text(out_txt, text)

    report_path: Optional[Path] = None
    if write_report:
        payload: Dict[str, Any] = {
            "source": str(transcript_path),
            "output": str(out_txt),
            "fixes": fix_report,
            "redactions": redact_report,
        }
        report_path = out_dir / f"{out_name}.report.json"
        _write_text(report_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    return out_txt, report_path


# -----------------------------
# CLI
# -----------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="transcribe_run.py")

    p.add_argument("input_media", type=Path, help="Path to media file (video/audio).")

    p.add_argument("--name", type=str, default=None, help="Base name for outputs (default: input stem).")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/transcripts"),
        help="Transcript output directory (default: out/transcripts).",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["plain", "timestamps"],
        default="plain",
        help="Transcript rendering mode (default: plain).",
    )

    # Whisper / backend settings
    p.add_argument("--backend", type=str, default="faster-whisper", help="Backend (default: faster-whisper).")
    p.add_argument("--model", type=str, default="base", help="Model name (default: base).")
    p.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda) (default: cpu).")
    p.add_argument("--compute-type", type=str, default="int8", help="Compute type (default: int8).")
    p.add_argument("--language", type=str, default=None, help="Language code (e.g. en). Default: auto.")
    p.add_argument("--beam-size", type=int, default=5, help="Beam size (default: 5).")
    p.add_argument("--vad-filter", action="store_true", help="Enable VAD filter.")

    # Prompt / hotwords (offline biasing)
    p.add_argument("--prompt", type=str, default=None, help="Initial prompt to bias transcription.")
    p.add_argument("--prompt-file", type=Path, default=None, help="Load prompt from file.")
    p.add_argument("--hotwords", type=str, default=None, help="Comma-separated hotwords.")
    p.add_argument("--hotwords-file", type=Path, default=None, help="Load hotwords (one per line).")

    # Outputs
    p.add_argument("--keep-wav", action="store_true", help="Keep extracted WAV in scratch directory.")
    p.add_argument("--no-json", action="store_true", help="Disable writing segments JSON (.segments.json).")
    p.add_argument("--srt", action="store_true", help="Write .srt subtitle file.")

    # Convenience
    p.add_argument("--print-latest", action="store_true", help="Print the file(s) created this run.")
    p.add_argument("--open-latest", action="store_true", help="Open the most relevant output file.")
    p.add_argument("--latest-head", type=int, default=0, help="Print first N lines of the most relevant output.")

    # Post-processing (offline-first)
    p.add_argument("--post-clean", action="store_true", help="Apply standard cleaning after transcription.")
    p.add_argument(
        "--post-fix",
        nargs="?",
        const=Path("notes/corrections.json"),
        default=None,
        type=Path,
        help="Apply deterministic replacements from JSON. If no value, uses notes/corrections.json.",
    )
    p.add_argument(
        "--post-redact",
        nargs="?",
        const="standard",
        default=None,
        help="Apply redaction profile. If provided without value, uses 'standard'.",
    )
    p.add_argument(
        "--post-terms",
        nargs="?",
        const=Path("notes/redaction_terms.json"),
        default=None,
        type=Path,
        help="Redaction terms/patterns JSON. If provided without value, uses notes/redaction_terms.json.",
    )
    p.add_argument("--post-report", action="store_true", help="Write a post-processing report JSON.")
    p.add_argument(
        "--post-out-dir",
        type=Path,
        default=Path("out/textops"),
        help="Post-processed output directory (default: out/textops).",
    )

    return p


def _load_prompt(args: argparse.Namespace) -> Optional[str]:
    if args.prompt_file is not None:
        pf = Path(args.prompt_file).expanduser().resolve()
        if not pf.exists():
            raise FileNotFoundError(f"Prompt file not found: {pf}")
        s = _read_text(pf).strip()
        return s or None
    return (args.prompt or None)


def _load_hotwords(args: argparse.Namespace) -> Optional[str]:
    if args.hotwords_file is not None:
        hf = Path(args.hotwords_file).expanduser().resolve()
        if not hf.exists():
            raise FileNotFoundError(f"Hotwords file not found: {hf}")
        words: List[str] = []
        for ln in _read_text(hf).splitlines():
            s = (ln or "").strip()
            if not s or s.startswith("#"):
                continue
            words.append(s)
        return ", ".join(words) if words else None
    return (args.hotwords or None)


def main() -> int:
    args = _build_parser().parse_args()

    input_media: Path = Path(args.input_media).expanduser().resolve()
    if not input_media.exists():
        raise FileNotFoundError(f"Input not found: {input_media}")
    if not input_media.is_file():
        raise ValueError(f"Input must be a file (not a directory): {input_media}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    name = args.name or input_media.stem

    prompt = _load_prompt(args)
    hotwords = _load_hotwords(args)

    # TranscribeOptions is a frozen dataclass in this repo -> no setattr().
    # Build extra kwargs only if those fields exist.
    extra: Dict[str, object] = {}
    fields = getattr(TranscribeOptions, "__dataclass_fields__", {}) or {}

    if hotwords is not None and "hotwords" in fields:
        extra["hotwords"] = hotwords

    if prompt is not None:
        if "prompt" in fields:
            extra["prompt"] = prompt
        elif "initial_prompt" in fields:
            extra["initial_prompt"] = prompt

    opts = TranscribeOptions(
        backend=str(args.backend),
        model=str(args.model),
        device=str(args.device),
        compute_type=str(args.compute_type),
        language=args.language,
        vad_filter=bool(args.vad_filter),
        beam_size=int(args.beam_size),
        **extra,
    )

    txt_path = run_transcription(
        input_media=input_media,
        out_dir=out_dir,
        opts=opts,
        keep_wav=bool(args.keep_wav),
        write_segments_json=(not bool(args.no_json)),
        write_srt=bool(args.srt),
        name=name,
        mode=str(args.mode),
    )

    created_paths: List[Path] = [txt_path]

    # sidecars (best-effort)
    stem_no_ext = Path(str(txt_path)[:-4])  # remove ".txt" safely enough
    json_sidecar = Path(str(stem_no_ext) + ".segments.json")
    srt_sidecar = Path(str(stem_no_ext) + ".srt")

    if json_sidecar.exists():
        created_paths.append(json_sidecar)
    if srt_sidecar.exists():
        created_paths.append(srt_sidecar)

    print(f"✅ Transcript written: {txt_path}")

    post_out: Optional[Path] = None
    post_report: Optional[Path] = None

    if args.post_clean or args.post_redact is not None or args.post_fix is not None:
        # If user asks to redact but doesn't specify terms, allow default.
        terms_path = args.post_terms
        if args.post_redact is not None and terms_path is None:
            terms_path = Path("notes/redaction_terms.json")

        post_out, post_report = _run_post_processing(
            transcript_path=txt_path,
            out_dir=Path(args.post_out_dir),
            fix_path=args.post_fix,
            do_clean=bool(args.post_clean),
            redact_profile=(str(args.post_redact) if args.post_redact is not None else None),
            terms_path=terms_path,
            write_report=bool(args.post_report),
        )

        created_paths.append(post_out)
        if post_report is not None and post_report.exists():
            created_paths.append(post_report)

        print(f"✅ Post output written: {post_out}")
        if post_report is not None:
            print(f"🧾 Post report: {post_report}")

    # Choose "most relevant" output for open/head:
    # 1) post_out if present
    # 2) .srt if it exists and user asked for it
    # 3) transcript txt
    preferred: Path
    if post_out is not None:
        preferred = post_out
    elif args.srt and srt_sidecar.exists():
        preferred = srt_sidecar
    else:
        preferred = txt_path

    if args.print_latest:
        for pth in created_paths:
            print(f"📄 Created: {pth}")

    if args.latest_head and int(args.latest_head) > 0:
        try:
            print(_head(preferred, int(args.latest_head)), end="")
        except Exception:
            pass

    if args.open_latest:
        _maybe_open_file(preferred)
        print(f"📂 Opened: {preferred}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
