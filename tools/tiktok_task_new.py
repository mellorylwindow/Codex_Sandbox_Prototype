#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "task"


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a new notes/tiktok/tasks/*.json task file.")
    ap.add_argument("name", help="Human name for the task (used for file + slug)")
    ap.add_argument("--task-text", default="", help="Task text used for ranking prompts")
    ap.add_argument("--take", type=int, default=12)
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--teleprompter-top", type=int, default=3)
    args = ap.parse_args()

    tasks_dir = Path("notes/tiktok/tasks")
    tasks_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify(args.name)
    path = tasks_dir / f"{slug}.json"

    payload = {
        "name": args.name,
        "slug": slug,
        "task_text": args.task_text or args.name,
        "take": args.take,
        "min_score": args.min_score,
        "teleprompter_top": args.teleprompter_top,
    }

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"OK: wrote task -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
