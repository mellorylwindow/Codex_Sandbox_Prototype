# tools/transcribe.py
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIBE_RUN = REPO_ROOT / "transcribe_run.py"

DEFAULT_MEDIA_DIR = REPO_ROOT / "notes" / "media_in"
DEFAULT_OUT_DIR = REPO_ROOT / "out" / "transcripts"

DEFAULT_HOTWORDS_FILE = REPO_ROOT / "notes" / "hotwords.txt"
DEFAULT_CORRECTIONS_FILE = REPO_ROOT / "notes" / "corrections.json"

# Common media extensions we’ll try when user passes a bare name (e.g. "demo_journal")
MEDIA_EXTS = [
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe media by name or path. Wrapper around transcribe_run.py with presets + sane defaults.",
    )

    p.add_argument(
        "target",
        help=(
            "Media path OR a base name to resolve from notes/media_in.\n"
            "Examples:\n"
            "  transcribe demo_journal\n"
            "  transcribe demo_journal --srt\n"
            "  transcribe notes/media_in/demo_journal.mp4 --srt\n"
        ),
    )

    # Presets (simple & explicit)
    p.add_argument(
        "--preset",
        choices=["speech", "music"],
        default="speech",
        help="Preset tuning. speech=default for voice; music=avoid hotword bias (default: speech).",
    )

    # Output selection (wrapper-level sugar)
    # NOTE: transcribe_run always writes .txt; --txt exists mainly for UX symmetry.
    p.add_argument("--txt", action="store_true", help="Explicitly request transcript .txt output (default behavior).")
    p.add_argument("--json", action="store_true", help="Request segments JSON output (segments.json).")
    p.add_argument("--no-json", action="store_true", help="Disable segments JSON output.")
    p.add_argument("--srt", action="store_true", help="Write an .srt subtitles file.")

    # Core knobs (passed through)
    p.add_argument("--name", default=None, help="Explicit name used for output basenames.")
    p.add_argument("--out-dir", default=None, help="Output directory for transcript artifacts (default: out/transcripts).")
    p.add_argument("--mode", choices=["plain", "timestamps"], default=None, help="Transcript text mode.")
    p.add_argument("--backend", default=None, help="Backend (default from transcribe_run).")
    p.add_argument("--model", default=None, help="Model size/name (e.g., base, small, medium).")
    p.add_argument("--device", default=None, help="cpu or cuda.")
    p.add_argument("--compute-type", default=None, help="Compute type (e.g., int8).")
    p.add_argument("--language", default=None, help="Language code (e.g., en).")
    p.add_argument("--beam-size", type=int, default=None, help="Beam size.")
    p.add_argument("--no-vad-filter", action="store_true", help="Disable VAD filter.")
    p.add_argument("--prompt", default=None, help="Prompt text to steer transcription.")
    p.add_argument("--prompt-file", default=None, help="Path to a prompt file.")

    # Hotwords (with defaults)
    p.add_argument("--hotwords", default=None, help="Hotwords string (passed through).")
    p.add_argument("--hotwords-file", default=None, help="Path to hotwords file (passed through).")
    p.add_argument("--no-default-hotwords", action="store_true", help="Do not auto-apply notes/hotwords.txt.")

    # Post-fix corrections (with defaults)
    p.add_argument("--post-fix", default=None, help="Path to corrections.json (string replacements).")
    p.add_argument("--no-default-post-fix", action="store_true", help="Do not auto-apply notes/corrections.json.")

    # Keep wav scratch
    p.add_argument("--keep-wav", action="store_true", help="Keep extracted WAV in scratch dir.")

    # Convenience: show/open latest
    p.add_argument("--print-latest", action="store_true", help="Print the latest output (created by this run).")
    p.add_argument("--open-latest", action="store_true", help="Open the latest output (created by this run).")
    p.add_argument("--latest-head", type=int, default=60, help="How many lines to print for --print-latest (default: 60).")

    # Debug
    p.add_argument("--dry-run", action="store_true", help="Print the underlying command and exit.")

    return p


def _resolve_media(target: str, media_dir: Path) -> Path:
    """
    Resolve `target` into an existing media file.

    Accepts:
      - explicit path to a file
      - bare name (e.g. demo_journal) which resolves to notes/media_in/demo_journal.*
      - bare name w/ extension (e.g. fire_woman.mp3) which resolves inside notes/media_in

    Returns:
      Absolute Path to an existing file.
    """
    p = Path(target)

    # 1) Direct explicit path
    if p.exists() and p.is_file():
        return p.expanduser().resolve()

    # 2) If relative file given, try inside media_dir
    if not p.is_absolute():
        candidate = (media_dir / p).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate

    # 3) Name-only resolution: notes/media_in/<name>.<ext>
    name = p.stem if p.suffix else p.name
    for ext in MEDIA_EXTS:
        candidate = (media_dir / f"{name}{ext}").expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate

    # 4) Glob fallback: notes/media_in/<name>.*
    matches = sorted(media_dir.glob(f"{name}.*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if matches:
        return matches[0].expanduser().resolve()

    tried = [
        f"path: {p}",
        f"{(media_dir / p).resolve()}",
        f"{(media_dir / f'{name}.*').resolve()}",
    ]
    raise FileNotFoundError(
        f"Could not resolve media '{target}'. Tried:\n"
        + "\n".join(f" - {t}" for t in tried)
        + f"\n\nTip: put the file in {media_dir} or pass an explicit path."
    )


def _append_arg(cmd: list[str], flag: str, value: Optional[str]) -> None:
    if value is None:
        return
    if str(value).strip() == "":
        return
    cmd.extend([flag, str(value)])


def _maybe_default_hotwords(args: argparse.Namespace) -> Optional[Path]:
    """
    Default hotwords behavior:
      - Only auto-apply for preset=speech
      - Never auto-apply for preset=music (prevents bias-injection like 'Duran Duran Gizmo')
      - Only if notes/hotwords.txt exists
      - Can be disabled by --no-default-hotwords
      - User-provided --hotwords-file overrides default selection
    """
    if args.no_default_hotwords:
        return None
    if args.hotwords_file or args.hotwords:
        return None
    if args.preset != "speech":
        return None
    if DEFAULT_HOTWORDS_FILE.exists():
        return DEFAULT_HOTWORDS_FILE
    return None


def _maybe_default_post_fix(args: argparse.Namespace) -> Optional[Path]:
    """
    Default post-fix behavior:
      - Auto-apply notes/corrections.json if present
      - Can be disabled by --no-default-post-fix
      - User-provided --post-fix overrides default
    """
    if args.no_default_post_fix:
        return None
    if args.post_fix:
        p = Path(args.post_fix).expanduser()
        return p if p.exists() else None
    if DEFAULT_CORRECTIONS_FILE.exists():
        return DEFAULT_CORRECTIONS_FILE
    return None


def _default_mode_for_preset(args: argparse.Namespace) -> str:
    """
    Your requested behavior:
      - speech preset defaults to timestamps
      - music preset defaults to plain
      - explicit --mode wins
    """
    if args.mode:
        return args.mode
    return "timestamps" if args.preset == "speech" else "plain"


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not TRANSCRIBE_RUN.exists():
        raise FileNotFoundError(f"Missing transcribe_run.py at: {TRANSCRIBE_RUN}")

    media_dir = DEFAULT_MEDIA_DIR
    media_path = _resolve_media(args.target, media_dir)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else DEFAULT_OUT_DIR.expanduser().resolve()

    # Build command for transcribe_run.py
    cmd: list[str] = [sys.executable, str(TRANSCRIBE_RUN), str(media_path)]

    # Name
    if args.name:
        _append_arg(cmd, "--name", args.name)

    # Out dir
    _append_arg(cmd, "--out-dir", str(out_dir))

    # Mode (preset default: speech -> timestamps)
    cmd.extend(["--mode", _default_mode_for_preset(args)])

    # Core knobs pass-through
    _append_arg(cmd, "--backend", args.backend)
    _append_arg(cmd, "--model", args.model)
    _append_arg(cmd, "--device", args.device)
    _append_arg(cmd, "--compute-type", args.compute_type)
    _append_arg(cmd, "--language", args.language)
    if args.beam_size is not None:
        _append_arg(cmd, "--beam-size", str(args.beam_size))
    if args.no_vad_filter:
        cmd.append("--no-vad-filter")

    _append_arg(cmd, "--prompt", args.prompt)
    _append_arg(cmd, "--prompt-file", args.prompt_file)

    # Output selection:
    # - transcribe_run writes .txt always
    # - segments json default ON unless --no-json
    # - if user sets --json, ensure json is ON (i.e., do NOT pass --no-json)
    if args.no_json:
        cmd.append("--no-json")
    # else: do nothing; default is json ON in transcribe_run

    if args.srt:
        cmd.append("--srt")

    # Hotwords (preset default)
    _append_arg(cmd, "--hotwords", args.hotwords)
    if args.hotwords_file:
        _append_arg(cmd, "--hotwords-file", args.hotwords_file)
    else:
        default_hot = _maybe_default_hotwords(args)
        if default_hot:
            _append_arg(cmd, "--hotwords-file", str(default_hot))

    # Keep wav
    if args.keep_wav:
        cmd.append("--keep-wav")

    # Post-fix (corrections) default
    post_fix = _maybe_default_post_fix(args)
    if post_fix:
        _append_arg(cmd, "--post-fix", str(post_fix))

    # Convenience latest flags
    if args.print_latest:
        cmd.append("--print-latest")
        _append_arg(cmd, "--latest-head", str(args.latest_head))
    if args.open_latest:
        cmd.append("--open-latest")

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    # Run
    subprocess.check_call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
