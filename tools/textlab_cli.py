# tools/textlab_cli.py
from __future__ import annotations

import argparse
import inspect
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


# Ensure repo root is on sys.path so `import textlab...` works when running:
#   python tools/textlab_cli.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUT_ROOT = Path("out/textlab")
DEFAULT_TRANSCRIPTS_DIR = Path("out/transcripts")

DEFAULT_POST_FIX = Path("notes/corrections.json")
DEFAULT_POST_TERMS = Path("notes/redaction_terms.json")


_TS_LINE_RE = re.compile(r"^\s*\[(\d{2}:\d{2}(?::\d{2})?)\]\s+")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="textlab",
        description="Offline-first transcript processing pipeline (chunking + TOC + manifests).",
    )

    p.add_argument(
        "target",
        help=(
            "Transcript file path OR a base name to resolve from out/transcripts.\n"
            "Examples:\n"
            "  textlab out/transcripts/recording_10__20251220_155127.txt\n"
            "  textlab recording_10\n"
        ),
    )

    p.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help=f"Root output folder for textlab runs (default: {DEFAULT_OUT_ROOT}).",
    )

    p.add_argument(
        "--transcripts-dir",
        default=str(DEFAULT_TRANSCRIPTS_DIR),
        help=f"Where to resolve name-only targets (default: {DEFAULT_TRANSCRIPTS_DIR}).",
    )

    p.add_argument(
        "--chunk-minutes",
        type=int,
        default=5,
        help="Chunk size in minutes (default: 5).",
    )

    # Optional textops-ish behavior (only passed if pipeline supports it)
    p.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Enable cleaning (default: ON).",
    )
    p.add_argument(
        "--no-clean",
        action="store_false",
        dest="clean",
        help="Disable cleaning.",
    )

    p.add_argument(
        "--clean-mode",
        choices=["light", "standard"],
        default="standard",
        help="Cleaning aggressiveness (default: standard).",
    )

    p.add_argument(
        "--redact",
        choices=["none", "light", "standard", "heavy"],
        default="standard",
        help="Redaction level (default: standard).",
    )

    p.add_argument(
        "--post-fix",
        default=str(DEFAULT_POST_FIX) if DEFAULT_POST_FIX.exists() else "",
        help="Path to corrections.json (string replacements). Defaults to notes/corrections.json if present.",
    )
    p.add_argument(
        "--no-post-fix",
        action="store_true",
        help="Disable corrections even if a default corrections.json exists.",
    )

    p.add_argument(
        "--post-terms",
        default=str(DEFAULT_POST_TERMS) if DEFAULT_POST_TERMS.exists() else "",
        help="Path to redaction_terms.json (custom patterns/terms). Defaults to notes/redaction_terms.json if present.",
    )
    p.add_argument(
        "--no-post-terms",
        action="store_true",
        help="Disable custom redaction terms even if a default redaction_terms.json exists.",
    )

    p.add_argument(
        "--report",
        action="store_true",
        help="If supported by pipeline, write a processing report.",
    )

    # Convenience UX (match transcribe vibe)
    p.add_argument(
        "--print-latest",
        action="store_true",
        help="Print the TOC for the newest run (or the run created by this command).",
    )
    p.add_argument(
        "--open-latest",
        action="store_true",
        help="Open the TOC for the newest run (or the run created by this command).",
    )
    p.add_argument(
        "--latest-head",
        type=int,
        default=60,
        help="How many lines of TOC to print when using --print-latest (default: 60).",
    )
    p.add_argument(
        "--open-target",
        choices=["toc", "manifest", "dir"],
        default="toc",
        help="What to open for --open-latest (default: toc).",
    )

    # NEW: Inspect latest without creating a new run
    p.add_argument(
        "--latest-only",
        action="store_true",
        help="Do not run the pipeline. Only inspect/open the latest existing run for this target.",
    )

    return p


def _base_name_from_stem(stem: str) -> str:
    """
    Given a transcript stem like:
      - recording_10__20251220_155127
    return:
      - recording_10
    """
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def _resolve_transcript(target: str, transcripts_dir: Path) -> Path:
    """
    Accepts:
      - explicit file path
      - name-only like "recording_10" -> resolves latest out/transcripts/recording_10__*.txt

    Returns a real, existing file path.
    """
    p = Path(target)

    # Explicit path
    if p.exists() and p.is_file():
        return p.expanduser().resolve()

    tried: list[str] = []

    # If user passed "recording_10__....txt" without folder, try relative to transcripts_dir
    if not p.is_absolute():
        candidate = (transcripts_dir / p).expanduser().resolve()
        tried.append(str(candidate))
        if candidate.exists() and candidate.is_file():
            return candidate

    # Name-only resolution: out/transcripts/<name>__*.txt
    name = p.stem if p.suffix else p.name
    glob_pat = f"{name}__*.txt"
    tried.append(str((transcripts_dir / glob_pat).resolve()))

    matches = sorted(transcripts_dir.glob(glob_pat), key=lambda x: x.stat().st_mtime, reverse=True)
    if matches:
        return matches[0].expanduser().resolve()

    raise FileNotFoundError(
        "Could not resolve transcript. Tried:\n"
        + "\n".join(f"- {t}" for t in tried)
        + "\n\nTip: pass a full path or use a base name that exists in out/transcripts/."
    )


def _latest_transcript_for_base(transcripts_dir: Path, base: str) -> Optional[Path]:
    matches = sorted(transcripts_dir.glob(f"{base}__*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
    return matches[0].expanduser().resolve() if matches else None


def _is_timestamped_transcript(path: Path, sample_lines: int = 40) -> bool:
    """
    Heuristic: timestamped transcripts have lines like:
      [00:03] Hello...
    """
    try:
        txt = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return False

    for line in txt[: max(1, sample_lines)]:
        if _TS_LINE_RE.match(line):
            return True
    return False


def _latest_run_dir(out_root: Path, base_name: str) -> Optional[Path]:
    """
    Find newest run dir matching: <base_name>__YYYYMMDD_HHMMSS under out_root.
    """
    if not out_root.exists():
        return None

    candidates = [
        p for p in out_root.iterdir()
        if p.is_dir() and p.name.startswith(f"{base_name}__")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


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


def _print_toc(toc_path: Path, head: int) -> None:
    toc_path = toc_path.expanduser().resolve()
    if not toc_path.exists():
        raise FileNotFoundError(f"TOC not found: {toc_path}")

    lines = toc_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[: max(0, head)]:
        print(line)


def _assign_first_present(kwargs: dict[str, Any], param_names: set[str], candidates: list[str], value: Any) -> bool:
    for name in candidates:
        if name in param_names:
            kwargs[name] = value
            return True
    return False


def _call_run_textlab(*, source: Path, out_root: Path, args: argparse.Namespace) -> Path:
    """
    Call textlab.pipeline.run_textlab defensively:
    - introspect signature
    - only pass params it accepts
    """
    from textlab.pipeline import run_textlab  # local import so textlab is only required at runtime

    sig = inspect.signature(run_textlab)
    param_names = set(sig.parameters.keys())
    kwargs: dict[str, Any] = {}

    # Source
    ok_source = _assign_first_present(
        kwargs,
        param_names,
        ["source", "source_path", "input_path", "input_file", "transcript_path", "transcript"],
        source,
    )
    if not ok_source:
        first = next(iter(sig.parameters.keys()), None)
        if first is None:
            raise RuntimeError("run_textlab() has no parameters; cannot call.")
        kwargs[first] = source

    # Output root
    _assign_first_present(
        kwargs,
        param_names,
        ["out_root", "out_dir", "output_root", "output_dir", "out_base", "out"],
        out_root,
    )

    # Chunking
    _assign_first_present(
        kwargs,
        param_names,
        ["chunk_minutes", "chunking_minutes", "minutes", "chunk_size_minutes"],
        int(args.chunk_minutes),
    )

    # Clean
    _assign_first_present(kwargs, param_names, ["clean", "do_clean", "post_clean", "enable_clean"], bool(args.clean))
    _assign_first_present(kwargs, param_names, ["clean_mode", "cleaning_mode"], str(args.clean_mode))

    # Redact
    redact_level = None if args.redact == "none" else str(args.redact)
    _assign_first_present(kwargs, param_names, ["redact", "redact_level", "redaction_level"], redact_level)

    # Post-fix
    post_fix = None
    if not args.no_post_fix and args.post_fix:
        pf = Path(args.post_fix)
        if pf.exists():
            post_fix = pf
    _assign_first_present(kwargs, param_names, ["post_fix", "postfix", "corrections", "corrections_path"], post_fix)

    # Post-terms
    post_terms = None
    if not args.no_post_terms and args.post_terms:
        pt = Path(args.post_terms)
        if pt.exists():
            post_terms = pt
    _assign_first_present(
        kwargs,
        param_names,
        ["post_terms", "terms", "terms_path", "redaction_terms", "redaction_terms_path"],
        post_terms,
    )

    # Report
    _assign_first_present(kwargs, param_names, ["report", "write_report"], bool(args.report))

    result = run_textlab(**kwargs)  # type: ignore[misc]
    if not isinstance(result, (str, Path)):
        raise RuntimeError(f"run_textlab() returned unexpected type: {type(result)}")

    run_dir = Path(result).expanduser().resolve()
    if not run_dir.exists():
        raise RuntimeError(f"textlab run directory does not exist: {run_dir}")

    return run_dir


def _choose_open_target(run_dir: Path, open_target: str) -> Path:
    toc = run_dir / "TOC.md"
    manifest = run_dir / "manifest.json"

    if open_target == "toc" and toc.exists():
        return toc
    if open_target == "manifest" and manifest.exists():
        return manifest
    return run_dir


def main() -> int:
    args = _build_parser().parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()

    # Resolve transcript (unless latest-only and target is name-only with no file)
    source = _resolve_transcript(args.target, transcripts_dir)
    base_name = _base_name_from_stem(source.stem)

    # If user is about to run TextLab on a non-timestamp transcript while a newer timestamped one exists, warn.
    # (This is the “why did it flip back to 1 chunk?” trap.)
    if not args.latest_only:
        is_ts = _is_timestamped_transcript(source)
        if not is_ts:
            newest = _latest_transcript_for_base(transcripts_dir, base_name)
            if newest and newest != source and _is_timestamped_transcript(newest):
                print("⚠️  Heads up: you are running TextLab on a NON-timestamp transcript.")
                print(f"   - selected: {source}")
                print(f"   - newest timestamped transcript exists: {newest}")
                print("   This will usually produce 1 big chunk and may overwrite your 'latest run'.")
                print("   Recommended:")
                print(f"     textlab {base_name} --print-latest")
                print("   Or inspect without running:")
                print(f"     textlab {base_name} --latest-only --print-latest")
                print("")

    # If latest-only: do not run pipeline; just inspect existing latest run
    if args.latest_only:
        latest = _latest_run_dir(out_root, base_name)
        if not latest:
            raise FileNotFoundError(
                f"No existing textlab run found under {out_root} for base '{base_name}'.\n"
                "Run without --latest-only to create the first run."
            )

        print(f"📌 Latest run: {latest}")

        if args.print_latest:
            toc = latest / "TOC.md"
            if toc.exists():
                print(f"--- {toc} (head {args.latest_head}) ---")
                _print_toc(toc, head=args.latest_head)
            else:
                print(f"(No TOC found at {toc})")

        if args.open_latest:
            to_open = _choose_open_target(latest, args.open_target)
            _open_path(to_open)
            print(f"📂 Opened: {to_open}")

        # If they didn’t ask to print/open, just exit after showing path
        return 0

    # Normal behavior: run pipeline
    run_dir = _call_run_textlab(source=source, out_root=out_root, args=args)

    toc_path = run_dir / "TOC.md"
    manifest_path = run_dir / "manifest.json"

    print(f"✅ textlab run created: {run_dir}")
    if manifest_path.exists():
        print(f"   - manifest: {manifest_path}")
    if (run_dir / "chunks").exists():
        print(f"   - chunks:   {run_dir / 'chunks'}")
    if toc_path.exists():
        print(f"   - toc:      {toc_path}")

    if args.print_latest:
        print("")
        print(f"📌 Latest run: {run_dir}")
        if toc_path.exists():
            print(f"--- {toc_path} (head {args.latest_head}) ---")
            _print_toc(toc_path, head=args.latest_head)
        else:
            print(f"(No TOC found at {toc_path})")

    if args.open_latest:
        to_open = _choose_open_target(run_dir, args.open_target)
        _open_path(to_open)
        print(f"📂 Opened: {to_open}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
