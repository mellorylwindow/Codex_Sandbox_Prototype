# tools/transcribe.py
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


DEFAULT_MEDIA_DIR = Path("notes/media_in")
DEFAULT_HOTWORDS = Path("notes/hotwords.txt")
DEFAULT_FIXES = Path("notes/corrections.json")


@dataclass(frozen=True)
class EffectiveDefaults:
    auto_hotwords: bool
    auto_fixes: bool


def _resolve_media(target: str, media_dir: Path) -> Path:
    """
    Resolve a target into a real media file.

    Accepts:
      - explicit paths (relative/absolute)
      - basename (e.g. "demo_journal")
      - name-with-extension (e.g. "fire_woman.mp3") while still searching media_dir

    Search order:
      1) exact path as given
      2) exact file inside media_dir (media_dir / filename)
      3) glob by stem inside media_dir (stem.*)
    """
    media_dir = media_dir.expanduser().resolve()
    raw = Path(target).expanduser()

    tried: List[str] = []

    # 1) exact as provided
    tried.append(f"path: {raw}")
    if raw.exists() and raw.is_file():
        return raw.resolve()

    # 2) exact within media_dir (supports "fire_woman.mp3" with hidden extensions)
    p2 = media_dir / raw.name
    tried.append(str(p2))
    if p2.exists() and p2.is_file():
        return p2.resolve()

    # 3) glob within media_dir by stem (supports "fire_woman" -> fire_woman.mp3)
    stem = raw.stem if raw.suffix else raw.name
    glob_pat = media_dir / f"{stem}.*"
    tried.append(str(glob_pat))

    matches = sorted([p for p in media_dir.glob(f"{stem}.*") if p.is_file()])
    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        # Prefer common media extensions when ambiguous
        pref = {".mp4", ".mkv", ".mov", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".webm"}
        preferred = [m for m in matches if m.suffix.lower() in pref]
        if len(preferred) == 1:
            return preferred[0].resolve()

        # Otherwise pick newest modified
        newest = max(matches, key=lambda p: p.stat().st_mtime)
        return newest.resolve()

    raise FileNotFoundError(
        f"Could not resolve media '{target}'. Tried:\n"
        + "\n".join(f"- {t}" for t in tried)
        + "\nTip: put the file in notes/media_in or pass an explicit path."
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="transcribe")

    p.add_argument("target", help="Basename in notes/media_in OR an explicit file path.")
    p.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR, help="Media input directory.")

    # Presets (new)
    p.add_argument(
        "--preset",
        choices=["speech", "music"],
        default="speech",
        help="speech (default): auto hotwords+fixes. music: disables auto hotwords+fixes to prevent bleed-in.",
    )

    # Output convenience aliases (map to transcribe_run.py)
    p.add_argument("--txt", action="store_true", help="Alias: transcript text output (default).")
    p.add_argument("--json", action="store_true", help="Ensure segments JSON is written (default).")
    p.add_argument("--no-json", action="store_true", help="Disable segments JSON output.")
    p.add_argument("--srt", action="store_true", help="Write SRT subtitles.")
    p.add_argument("--keep-wav", action="store_true", help="Keep extracted WAV in scratch dir.")

    # Naming / formatting
    p.add_argument("--name", type=str, default=None, help="Output base name (default: resolved file stem).")
    p.add_argument("--mode", choices=["plain", "timestamps"], default=None, help="Transcript mode.")

    # Whisper opts
    p.add_argument("--language", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--compute-type", type=str, default=None)
    p.add_argument("--beam-size", type=int, default=None)
    p.add_argument("--vad-filter", action="store_true")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--hotwords", type=str, default=None)
    p.add_argument("--hotwords-file", type=Path, default=None)

    # Post-processing
    p.add_argument("--post-clean", action="store_true")
    p.add_argument("--post-redact", nargs="?", const="standard", default=None)
    p.add_argument("--post-terms", nargs="?", const=Path("notes/redaction_terms.json"), default=None, type=Path)
    p.add_argument("--post-report", action="store_true")
    p.add_argument("--post-fix", type=Path, default=None)

    # Defaults toggles
    p.add_argument("--no-default-hotwords", action="store_true", help="Disable default hotwords file.")
    p.add_argument("--no-default-fix", action="store_true", help="Disable default corrections file.")

    # UX
    p.add_argument("--print-latest", action="store_true")
    p.add_argument("--open-latest", action="store_true")
    p.add_argument("--latest-head", type=int, default=None)

    return p


def _compute_effective_defaults(args: argparse.Namespace) -> EffectiveDefaults:
    """
    Compute whether we should auto-apply hotwords/fixes.

    Rules:
      - speech preset: auto hotwords + auto fixes (unless user disabled)
      - music preset: auto hotwords OFF + auto fixes OFF (unless user explicitly passes files)
      - explicit --hotwords-file / --post-fix always wins (we pass it)
    """
    auto_hotwords = not args.no_default_hotwords
    auto_fixes = not args.no_default_fix

    if args.preset == "music":
        # Prevent hotword/prompt "bleed" into uncertain singing transcription.
        # Still allow explicit user intent.
        if args.hotwords_file is None and args.hotwords is None:
            auto_hotwords = False
        if args.post_fix is None:
            auto_fixes = False

    return EffectiveDefaults(auto_hotwords=auto_hotwords, auto_fixes=auto_fixes)


def _pick_hotwords_file(args: argparse.Namespace, eff: EffectiveDefaults) -> Optional[Path]:
    if args.hotwords_file is not None:
        return args.hotwords_file
    if eff.auto_hotwords and DEFAULT_HOTWORDS.exists():
        return DEFAULT_HOTWORDS
    return None


def _pick_post_fix(args: argparse.Namespace, eff: EffectiveDefaults) -> Optional[Path]:
    if args.post_fix is not None:
        return args.post_fix
    if eff.auto_fixes and DEFAULT_FIXES.exists():
        return DEFAULT_FIXES
    return None


def _print_plan(*, args: argparse.Namespace, media_path: Path, name: str, hotwords_file: Optional[Path], post_fix: Optional[Path]) -> None:
    outputs: List[str] = ["txt"]
    outputs.append("segments.json" if not args.no_json else "NO segments.json")
    if args.srt:
        outputs.append("srt")
    if args.keep_wav:
        outputs.append("keep-wav")

    print("▶ transcribe")
    print(f"- preset:      {args.preset}")
    print(f"- media:       {media_path}")
    print(f"- name:        {name}")
    print(f"- outputs:     {', '.join(outputs)}")

    if args.hotwords:
        print(f"- hotwords:    (inline) {args.hotwords}")
    elif hotwords_file is not None:
        print(f"- hotwords:    {hotwords_file}")
    else:
        print("- hotwords:    (none)")

    if args.prompt:
        print(f"- prompt:      (inline) {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    elif args.prompt_file:
        print(f"- prompt:      {args.prompt_file}")
    else:
        print("- prompt:      (none)")

    print(f"- post-fix:    {post_fix if post_fix is not None else '(none)'}")
    if args.post_clean:
        print("- post-clean:  ON")
    if args.post_redact is not None:
        print(f"- post-redact: {args.post_redact}")
        if args.post_terms is not None:
            print(f"- post-terms:  {args.post_terms}")
    if args.post_report:
        print("- post-report: ON")
    print("")


def main() -> int:
    args = _build_parser().parse_args()

    # Guardrails for conflicting output toggles
    if args.json and args.no_json:
        raise SystemExit("Error: cannot pass both --json and --no-json.")

    media_path = _resolve_media(args.target, args.media_dir)
    name = args.name or media_path.stem

    eff = _compute_effective_defaults(args)
    hotwords_file = _pick_hotwords_file(args, eff)
    post_fix = _pick_post_fix(args, eff)

    _print_plan(args=args, media_path=media_path, name=name, hotwords_file=hotwords_file, post_fix=post_fix)

    # Build command for transcribe_run.py
    cmd: List[str] = [sys.executable, "transcribe_run.py", str(media_path), "--name", name]

    # Mode (only pass if set, so transcribe_run defaults remain)
    if args.mode:
        cmd += ["--mode", args.mode]

    # Outputs
    if args.srt:
        cmd.append("--srt")
    if args.no_json:
        cmd.append("--no-json")
    if args.keep_wav:
        cmd.append("--keep-wav")

    # Whisper params
    if args.language:
        cmd += ["--language", args.language]
    if args.model:
        cmd += ["--model", args.model]
    if args.device:
        cmd += ["--device", args.device]
    if args.compute_type:
        cmd += ["--compute-type", args.compute_type]
    if args.beam_size is not None:
        cmd += ["--beam-size", str(args.beam_size)]
    if args.vad_filter:
        cmd.append("--vad-filter")

    # Prompt/hotwords (explicit)
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if args.prompt_file:
        cmd += ["--prompt-file", str(args.prompt_file)]
    if args.hotwords:
        cmd += ["--hotwords", args.hotwords]
    if hotwords_file is not None:
        cmd += ["--hotwords-file", str(hotwords_file)]

    # Post-fix (explicit or default)
    if post_fix is not None:
        cmd += ["--post-fix", str(post_fix)]

    # Post-processing toggles
    if args.post_clean:
        cmd.append("--post-clean")
    if args.post_redact is not None:
        cmd += ["--post-redact", str(args.post_redact)]
        if args.post_terms is not None:
            cmd += ["--post-terms", str(args.post_terms)]
    if args.post_report:
        cmd.append("--post-report")

    # UX
    if args.print_latest:
        cmd.append("--print-latest")
    if args.open_latest:
        cmd.append("--open-latest")
    if args.latest_head is not None:
        cmd += ["--latest-head", str(args.latest_head)]

    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
