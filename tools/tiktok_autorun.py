#!/usr/bin/env python3
"""
tiktok_autorun.py

One-command runner for your prompt library workflow.

Goal:
- Treat "tiktok" as just the prompt library bucket name (source irrelevant).
- Run selection automatically based on a task JSON (alignment-first).
- Optionally refresh the library from:
  - ChatGPT export (extractor)
  - TikTok images pipeline (OCR pipeline)
- Then:
  - select best prompts for the task
  - copy queue to notes/tiktok/recording_picks/<date>_<task>.md
  - generate teleprompter markdown from that pick file

This script intentionally shells out to your existing tools so it doesn't depend
on internal Python APIs that may shift as you keep iterating.

Assumptions (based on your current repo):
- tools/tiktok_prompt_extract_from_export.py exists
- tools/tiktok_pipeline.py exists
- tools/tiktok_prompt_select.py exists
- tools/tiktok_prompt_teleprompter.py exists
- tasks live in: notes/tiktok/tasks/*.json

Task JSON minimal schema (example):
{
  "name": "Velvet OS — explain to coworkers",
  "task_text": "Explain Velvet OS like a hobby to coworkers...",
  "take": 12,
  "min_score": 0.25,
  "required_any": ["simple","work-safe"],         # optional
  "teleprompter_top": 3                           # optional
}

"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "task"


def _run(cmd: List[str]) -> None:
    # Print command in a friendly way, run, raise on failure.
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_task(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Task file must be a JSON object: {path}")
    if not data.get("task_text"):
        raise ValueError(f"Task JSON missing required field 'task_text': {path}")
    return data


def _find_latest_queue_md(queue_dir: Path, slug_hint: str) -> Optional[Path]:
    """
    Selector writes to out/tiktok_prompts/queues/*.md
    We'll look for the newest file containing the slug_hint, else newest .md.
    """
    if not queue_dir.exists():
        return None
    mds = sorted(queue_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mds:
        return None
    for p in mds:
        if slug_hint in p.name:
            return p
    return mds[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="One-command prompt pipeline runner.")
    ap.add_argument("--task", required=True, help="Path to task json (e.g. notes/tiktok/tasks/velvet_os.json)")
    ap.add_argument("--refresh-export", action="store_true", help="Re-extract prompts from a ChatGPT export folder")
    ap.add_argument("--export", default="", help='ChatGPT export path, e.g. "/c/Users/naked/Downloads/chatgpt-export"')
    ap.add_argument("--export-min-score", type=float, default=0.30, help="Extractor min-score for export parsing")
    ap.add_argument("--export-roles", choices=["user", "assistant", "both"], default="both", help="Roles to extract from export")

    ap.add_argument("--refresh-images", action="store_true", help="Run OCR pipeline on notes/tiktok/images_in")
    ap.add_argument("--images-in", default="notes/tiktok/images_in", help="Where new images live")
    ap.add_argument("--images-out", default="out/tiktok_prompts", help="Base output folder for prompt library")
    ap.add_argument("--copy-images", action="store_true", help="Pass --copy-images to tiktok_pipeline.py if supported")

    ap.add_argument("--select-min-score", type=float, default=None, help="Override task min_score")
    ap.add_argument("--take", type=int, default=None, help="Override task take")
    ap.add_argument("--teleprompter-top", type=int, default=None, help="Override task teleprompter_top")

    args = ap.parse_args()

    repo_root = Path(".").resolve()
    task_path = Path(args.task)
    task = _read_task(task_path)

    name = str(task.get("name") or task_path.stem)
    slug = _slug(str(task.get("slug") or name))
    task_text = str(task["task_text"]).strip()

    required_any = task.get("required_any")
    if isinstance(required_any, str):
        required_any_list = [x.strip() for x in required_any.split(",") if x.strip()]
    elif isinstance(required_any, list):
        required_any_list = [str(x).strip() for x in required_any if str(x).strip()]
    else:
        required_any_list = []

    take = int(args.take if args.take is not None else task.get("take", 12))
    min_score = float(args.select_min_score if args.select_min_score is not None else task.get("min_score", 0.25))
    tele_top = int(args.teleprompter_top if args.teleprompter_top is not None else task.get("teleprompter_top", 3))

    # 1) Optional: refresh from export
    if args.refresh_export:
        if not args.export:
            raise SystemExit("ERROR: --refresh-export requires --export <path>")
        _run([
            sys.executable, "tools/tiktok_prompt_extract_from_export.py",
            "--export", args.export,
            "--roles", args.export_roles,
            "--min-score", str(args.export_min_score),
        ])

    # 2) Optional: refresh from images (OCR pipeline)
    if args.refresh_images:
        cmd = [
            sys.executable, "tools/tiktok_pipeline.py",
            "--in", args.images_in,
            "--out", args.images_out,
        ]
        if args.copy_images:
            cmd.append("--copy-images")
        _run(cmd)

    # 3) Select best prompts for this task across all known draft sources
    # We keep it simple and explicit. If a file doesn't exist, selector should handle or error.
    draft_sources = [
        "out/tiktok_prompts/drafts.jsonl",
        "out/tiktok_prompts/chat/drafts.jsonl",
        "out/tiktok_prompts/compiled/drafts.jsonl",
    ]

    sel_cmd = [
        sys.executable, "tools/tiktok_prompt_select.py",
        "--drafts", *draft_sources,
        "--task-text", task_text,
        "--min-score", str(min_score),
        "--take", str(take),
    ]
    if required_any_list:
        sel_cmd += ["--required-any", ",".join(required_any_list)]

    _run(sel_cmd)

    # 4) Copy newest queue into recording_picks with date stamp
    queue_dir = Path("out/tiktok_prompts/queues")
    queue_md = _find_latest_queue_md(queue_dir, slug_hint=slug)
    if not queue_md:
        raise SystemExit("ERROR: Could not find any queue.md output in out/tiktok_prompts/queues")

    stamp = datetime.now().strftime("%Y-%m-%d")
    picks_dir = Path("notes/tiktok/recording_picks")
    picks_dir.mkdir(parents=True, exist_ok=True)

    pick_md = picks_dir / f"{stamp}_{slug}_queue.md"
    pick_md.write_text(queue_md.read_text(encoding="utf-8", errors="replace"), encoding="utf-8", newline="\n")
    print(f"\nOK: copied queue -> {pick_md}")

    # 5) Teleprompter from picks file
    _run([
        sys.executable, "tools/tiktok_prompt_teleprompter.py",
        "--in-md", str(pick_md),
        "--top", str(tele_top),
    ])

    print("\nOK: autorun complete")
    print(f"- task: {name}")
    print(f"- pick: {pick_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
