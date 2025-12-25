# tools/run_ai.py
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# Ensure repo root is importable when running:
#   python tools/run_ai.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_TEXTLAB_OUT = Path("out/textlab")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_ai",
        description="Materialize offline prompt files from a textlab run (no API calls).",
    )
    p.add_argument(
        "target",
        help=(
            "A textlab run dir OR a transcript stem (e.g. recording_10) OR a transcript file path.\n"
            "Examples:\n"
            "  run_ai recording_10\n"
            "  run_ai out/textlab/recording_10__20251221_144222\n"
            "  run_ai out/transcripts/recording_10__20251221_143318.txt\n"
        ),
    )
    p.add_argument(
        "--out-root",
        default=str(DEFAULT_TEXTLAB_OUT),
        help=f"Textlab output root (default: {DEFAULT_TEXTLAB_OUT})",
    )
    p.add_argument(
        "--tasks",
        default="",
        help="Comma-separated tasks to materialize (e.g. review,summary,rewrite). Default: all known tasks.",
    )
    p.add_argument(
        "--out-dir-name",
        default="ai_materialized",
        help="Folder name under the run dir to write prompts into (default: ai_materialized).",
    )
    p.add_argument(
        "--print",
        action="store_true",
        help="Print the first prompt path generated and its first ~80 lines.",
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the output directory after generation.",
    )
    return p


def _open_path(path: Path) -> None:
    path = path.expanduser().resolve()
    system = platform.system().lower()
    if system.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif system == "darwin":
        subprocess.check_call(["open", str(path)])
    else:
        subprocess.check_call(["xdg-open", str(path)])


def _resolve_run_dir(target: str, out_root: Path) -> Path:
    p = Path(target).expanduser()

    # Case 1: user passed a run directory
    if p.exists() and p.is_dir() and (p / "TOC.md").exists():
        return p.resolve()

    # Case 2: user passed a transcript file: recording_10__YYYYMMDD_HHMMSS.txt
    if p.exists() and p.is_file():
        stem = p.name
        # derive "recording_10" from "recording_10__2025....txt"
        base = stem.split("__", 1)[0]
        return _latest_run_dir_for_base(base, out_root)

    # Case 3: user passed a base name "recording_10"
    base = p.stem if p.suffix else p.name
    return _latest_run_dir_for_base(base, out_root)


def _latest_run_dir_for_base(base: str, out_root: Path) -> Path:
    out_root = out_root.expanduser().resolve()
    if not out_root.exists():
        raise FileNotFoundError(f"textlab out_root does not exist: {out_root}")

    candidates = [d for d in out_root.iterdir() if d.is_dir() and d.name.startswith(f"{base}__")]
    if not candidates:
        raise FileNotFoundError(f"No textlab runs found for '{base}' under {out_root}")

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def _maybe_parse_tasks(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts or None


def _print_prompt_preview(prompt_path: Path, head_lines: int = 80) -> None:
    txt = prompt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"\n--- {prompt_path} (head {head_lines}) ---")
    for line in txt[:head_lines]:
        print(line)


def main() -> int:
    args = _build_parser().parse_args()

    out_root = Path(args.out_root)
    run_dir = _resolve_run_dir(args.target, out_root=out_root)
    tasks = _maybe_parse_tasks(args.tasks)

    from textlab.run_ai import materialize_prompts  # local import

    result = materialize_prompts(
        run_dir=run_dir,
        tasks=tasks,
        out_dir_name=str(args.out_dir_name),
    )

    print(f"✅ Materialized prompts: {len(result.prompt_files)}")
    print(f"   - out:      {result.out_dir}")
    print(f"   - manifest: {result.manifest_path}")

    if args.print and result.prompt_files:
        _print_prompt_preview(result.prompt_files[0])

    if args.open:
        _open_path(result.out_dir)
        print(f"📂 Opened: {result.out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
