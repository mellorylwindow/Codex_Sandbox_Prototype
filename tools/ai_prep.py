# tools/ai_prep.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root on sys.path so `import textlab...` works
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT_ROOT = Path("out/textlab")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai_prep",
        description="Offline-only: generate AI-ready prompt packs from a textlab run dir.",
    )
    p.add_argument(
        "run",
        help=(
            "TextLab run dir OR a source stem (e.g. recording_10) to resolve latest run under out/textlab.\n"
            "Examples:\n"
            "  python tools/ai_prep.py out/textlab/recording_10__20251221_144222\n"
            "  python tools/ai_prep.py recording_10\n"
        ),
    )
    p.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help=f"Where to look for latest runs when passing a stem (default: {DEFAULT_OUT_ROOT}).",
    )
    p.add_argument(
        "--tasks",
        default="review,grammar,summary",
        help="Comma list: review,grammar,summary,rewrite,story (default: review,grammar,summary).",
    )
    p.add_argument(
        "--subdir",
        default="ai_prep",
        help="Subfolder name created inside the run dir (default: ai_prep).",
    )
    return p


def _latest_run_dir(out_root: Path, stem: str) -> Path:
    if not out_root.exists():
        raise FileNotFoundError(f"out root does not exist: {out_root}")
    candidates = [p for p in out_root.iterdir() if p.is_dir() and p.name.startswith(f"{stem}__")]
    if not candidates:
        raise FileNotFoundError(f"No runs found for stem '{stem}' under {out_root}")
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> int:
    args = _build_parser().parse_args()
    out_root = Path(args.out_root).expanduser().resolve()

    run_arg = Path(args.run)
    if run_arg.exists() and run_arg.is_dir():
        run_dir = run_arg.expanduser().resolve()
    else:
        stem = args.run.strip()
        run_dir = _latest_run_dir(out_root, stem)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    from textlab.ai_prep import prepare_ai_pack  # local import

    ai_dir = prepare_ai_pack(run_dir=run_dir, tasks=tasks, out_subdir=args.subdir)

    print(f"✅ AI prep pack created: {ai_dir}")
    print(f"   - README:   {ai_dir / 'README.md'}")
    print(f"   - requests: {ai_dir / 'requests.jsonl'}")
    print(f"   - tasks:    {', '.join(tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
