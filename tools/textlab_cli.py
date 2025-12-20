# tools/textlab_cli.py
from __future__ import annotations

import argparse
import inspect
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Import path bootstrap
#
# Allows running:
#   python tools/textlab_cli.py ...
# even though `textlab/` lives at repo root.
# (Without this, sys.path[0] becomes "tools/" and `import textlab` can fail.)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT_ROOT = Path("out/textlab")
DEFAULT_TRANSCRIPTS_DIR = Path("out/transcripts")

DEFAULT_POST_FIX = Path("notes/corrections.json")
DEFAULT_POST_TERMS = Path("notes/redaction_terms.json")


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

    # Defaults: clean ON, redact standard, apply post-fix/terms if present
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
        help="Path to corrections.json. Defaults to notes/corrections.json if present.",
    )
    p.add_argument(
        "--no-post-fix",
        action="store_true",
        help="Disable corrections even if a default corrections.json exists.",
    )

    p.add_argument(
        "--post-terms",
        default=str(DEFAULT_POST_TERMS) if DEFAULT_POST_TERMS.exists() else "",
        help="Path to redaction_terms.json. Defaults to notes/redaction_terms.json if present.",
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

    # Convenience UX
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

    return p


def _resolve_transcript(target: str, transcripts_dir: Path) -> Path:
    """
    Accepts:
      - explicit file path
      - name-only like "recording_10" -> resolves latest out/transcripts/recording_10__*.txt
    """
    p = Path(target)

    if p.exists() and p.is_file():
        return p.expanduser().resolve()

    tried: list[str] = []

    if not p.is_absolute():
        candidate = (transcripts_dir / p).expanduser().resolve()
        tried.append(str(candidate))
        if candidate.exists() and candidate.is_file():
            return candidate

    name = p.stem if p.suffix else p.name
    glob_pat = f"{name}__*.txt"
    tried.append(str((transcripts_dir / glob_pat).resolve()))

    matches = sorted(
        transcripts_dir.glob(glob_pat),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0].expanduser().resolve()

    raise FileNotFoundError(
        "Could not resolve transcript. Tried:\n"
        + "\n".join(f"- {t}" for t in tried)
        + "\n\nTip: pass a full path or use a base name that exists in out/transcripts/."
    )


def _latest_run_dir(out_root: Path, source_stem: str) -> Optional[Path]:
    if not out_root.exists():
        return None

    candidates = [
        p for p in out_root.iterdir()
        if p.is_dir() and p.name.startswith(f"{source_stem}__")
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


def _assign_first_present(
    kwargs: dict[str, Any],
    param_names: set[str],
    candidates: list[str],
    value: Any,
) -> bool:
    for name in candidates:
        if name in param_names:
            kwargs[name] = value
            return True
    return False


def _call_run_textlab(*, source: Path, out_root: Path, args: argparse.Namespace) -> Path:
    from textlab.pipeline import run_textlab  # delayed import

    sig = inspect.signature(run_textlab)
    param_names = set(sig.parameters.keys())

    kwargs: dict[str, Any] = {}

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

    _assign_first_present(
        kwargs,
        param_names,
        ["out_root", "out_dir", "output_root", "output_dir", "out_base", "out"],
        out_root,
    )

    _assign_first_present(
        kwargs,
        param_names,
        ["chunk_minutes", "chunking_minutes", "minutes", "chunk_size_minutes"],
        int(args.chunk_minutes),
    )

    _assign_first_present(
        kwargs,
        param_names,
        ["clean", "do_clean", "post_clean", "enable_clean"],
        bool(args.clean),
    )
    _assign_first_present(
        kwargs,
        param_names,
        ["clean_mode", "cleaning_mode"],
        str(args.clean_mode),
    )

    redact_level = None if args.redact == "none" else str(args.redact)
    _assign_first_present(
        kwargs,
        param_names,
        ["redact", "redact_level", "redaction_level"],
        redact_level,
    )

    post_fix: Optional[Path] = None
    if not args.no_post_fix and args.post_fix:
        p = Path(args.post_fix)
        if p.exists():
            post_fix = p
    _assign_first_present(
        kwargs,
        param_names,
        ["post_fix", "postfix", "corrections", "corrections_path"],
        post_fix,
    )

    post_terms: Optional[Path] = None
    if not args.no_post_terms and args.post_terms:
        p = Path(args.post_terms)
        if p.exists():
            post_terms = p
    _assign_first_present(
        kwargs,
        param_names,
        ["post_terms", "terms", "terms_path", "redaction_terms", "redaction_terms_path"],
        post_terms,
    )

    _assign_first_present(
        kwargs,
        param_names,
        ["report", "write_report"],
        bool(args.report),
    )

    result = run_textlab(**kwargs)  # type: ignore[misc]
    if not isinstance(result, (str, Path)):
        raise RuntimeError(f"run_textlab() returned unexpected type: {type(result)}")

    run_dir = Path(result).expanduser().resolve()
    if not run_dir.exists():
        raise RuntimeError(f"textlab run directory does not exist: {run_dir}")

    return run_dir


def main() -> int:
    args = _build_parser().parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()

    source = _resolve_transcript(args.target, transcripts_dir)
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

    latest_dir = run_dir if run_dir.exists() else (_latest_run_dir(out_root, source.stem) or run_dir)
    latest_toc = latest_dir / "TOC.md"
    latest_manifest = latest_dir / "manifest.json"

    if args.print_latest:
        print("")
        print(f"📌 Latest run: {latest_dir}")
        if latest_toc.exists():
            print(f"--- {latest_toc} (head {args.latest_head}) ---")
            _print_toc(latest_toc, head=args.latest_head)
        else:
            print(f"(No TOC found at {latest_toc})")

    if args.open_latest:
        if args.open_target == "toc" and latest_toc.exists():
            to_open = latest_toc
        elif args.open_target == "manifest" and latest_manifest.exists():
            to_open = latest_manifest
        else:
            to_open = latest_dir

        _open_path(to_open)
        print(f"📂 Opened: {to_open}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
