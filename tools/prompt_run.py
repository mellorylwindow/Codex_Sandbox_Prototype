#!/usr/bin/env python
"""
prompt_run.py

Single-command wrapper:
- writes a task JSON
- runs tools/tiktok_autorun.py for that task

Usage:
  python tools/prompt_run.py "Explain Velvet OS like a hobby..." --take 12 --min-score 0.25 --tele-top 3
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] or "task"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_text", help="Task text to rank prompts against.")
    ap.add_argument("--name", default=None, help="Optional task name.")
    ap.add_argument("--take", type=int, default=12)
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--tele-top", type=int, default=3)
    args = ap.parse_args()

    tasks_dir = Path("notes/tiktok/tasks")
    tasks_dir.mkdir(parents=True, exist_ok=True)

    name = args.name or f"Task — {args.task_text[:48].strip()}"
    slug = _slugify(name)

    task_path = tasks_dir / f"{slug}.json"
    payload = {
        "name": name,
        "slug": slug,
        "task_text": args.task_text,
        "take": args.take,
        "min_score": args.min_score,
        "teleprompter_top": args.tele_top,
    }
    task_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cmd = ["python", "tools/tiktok_autorun.py", "--task", str(task_path)]
    print("\n$ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        return r.returncode

    print(f"\nOK: task written -> {task_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
