#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Task:
    name: str
    slug: str
    task_text: str
    take: int = 12
    min_score: float = 0.25
    teleprompter_top: int = 3
    required_any: str | None = None
    # refresh knobs
    refresh_images: bool = True
    refresh_export: bool = False
    export_dir: str | None = None
    export_min_score: float = 0.30
    export_user_only: bool = False
    export_roles: str = "both"


def _run(cmd: list[str]) -> None:
    # Stream output live, fail loudly.
    p = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "task"


def _read_task_file(path: Path) -> Task:
    data = json.loads(path.read_text(encoding="utf-8"))

    def g(key: str, default: Any = None) -> Any:
        return data.get(key, default)

    name = str(g("name", path.stem))
    slug = str(g("slug", _slugify(name)))
    task_text = str(g("task_text", name))

    return Task(
        name=name,
        slug=slug,
        task_text=task_text,
        take=int(g("take", 12)),
        min_score=float(g("min_score", 0.25)),
        teleprompter_top=int(g("teleprompter_top", 3)),
        required_any=g("required_any", None),
        refresh_images=bool(g("refresh_images", True)),
        refresh_export=bool(g("refresh_export", False)),
        export_dir=g("export_dir", None),
        export_min_score=float(g("export_min_score", 0.30)),
        export_user_only=bool(g("export_user_only", False)),
        export_roles=str(g("export_roles", "both")),
    )


def _default_paths() -> dict[str, Path]:
    return {
        "images_in": REPO_ROOT / "notes" / "tiktok" / "images_in",
        "tasks_dir": REPO_ROOT / "notes" / "tiktok" / "tasks",
        "picks_dir": REPO_ROOT / "notes" / "tiktok" / "recording_picks",
        "out_root": REPO_ROOT / "out" / "tiktok_prompts",
        "compiled_dir": REPO_ROOT / "out" / "tiktok_prompts" / "compiled",
        "drafts_main": REPO_ROOT / "out" / "tiktok_prompts" / "drafts.jsonl",
        "drafts_chat": REPO_ROOT / "out" / "tiktok_prompts" / "chat" / "drafts.jsonl",
        "drafts_compiled": REPO_ROOT / "out" / "tiktok_prompts" / "compiled" / "drafts.jsonl",
    }


def _maybe_refresh_images(images_in: Path, out_root: Path, copy_images: bool) -> None:
    if not images_in.exists():
        return
    files = [p for p in images_in.iterdir() if p.is_file()]
    if not files:
        return

    cmd = [
        sys.executable,
        "tools/tiktok_pipeline.py",
        "--in",
        str(images_in),
        "--out",
        str(out_root),
    ]
    if copy_images:
        cmd.append("--copy-images")

    _run(cmd)


def _maybe_refresh_export(task: Task, export_dir: str) -> None:
    cmd = [
        sys.executable,
        "tools/tiktok_prompt_extract_from_export.py",
        "--export",
        export_dir,
        "--min-score",
        str(task.export_min_score),
    ]
    if task.export_user_only:
        cmd.append("--user-only")
    if task.export_roles:
        cmd.extend(["--roles", task.export_roles])

    _run(cmd)


def _select_for_task(task: Task, drafts: list[Path]) -> Path:
    cmd = [
        sys.executable,
        "tools/tiktok_prompt_select.py",
        "--task-text",
        task.task_text,
        "--min-score",
        str(task.min_score),
        "--take",
        str(task.take),
    ]

    # drafts can be files or dirs; selector already supports both in your workflow
    cmd.append("--drafts")
    cmd.extend([str(p) for p in drafts])

    if task.required_any:
        cmd.extend(["--required-any", task.required_any])

    _run(cmd)

    # The selector writes the queue to out/tiktok_prompts/queues/<slug>_queue.md,
    # but the filename is computed by the selector, not here.
    # So we find the newest queue.md.
    queues_dir = REPO_ROOT / "out" / "tiktok_prompts" / "queues"
    newest = max(queues_dir.glob("*_queue.md"), key=lambda p: p.stat().st_mtime)
    return newest


def _copy_pick(task: Task, queue_md: Path, picks_dir: Path) -> Path:
    picks_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    pick_path = picks_dir / f"{stamp}_{task.slug}_queue.md"
    pick_path.write_text(queue_md.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    print(f"OK: copied queue -> {pick_path}")
    return pick_path


def _teleprompter(pick_md: Path, top: int) -> Path:
    cmd = [
        sys.executable,
        "tools/tiktok_prompt_teleprompter.py",
        "--in-md",
        str(pick_md),
        "--top",
        str(top),
    ]
    _run(cmd)
    return pick_md.with_name(pick_md.stem + "_teleprompter.md")


def _open_in_code(path: Path) -> None:
    code = shutil.which("code")
    if not code:
        return
    subprocess.run([code, str(path)], cwd=str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="One-command: refresh inputs (optional) + select prompts for a task + write pick + teleprompter."
    )
    ap.add_argument(
        "task",
        help="Task slug OR a path to notes/tiktok/tasks/*.json (e.g., velvet_os_explain or notes/tiktok/tasks/x.json)",
    )
    ap.add_argument("--no-refresh-images", action="store_true", help="Skip image pipeline refresh")
    ap.add_argument("--refresh-export", action="store_true", help="Refresh compiled prompts from ChatGPT export")
    ap.add_argument("--export", default=os.environ.get("CHATGPT_EXPORT_DIR", ""), help="Path to chatgpt-export folder")
    ap.add_argument("--copy-images", action="store_true", help="Copy images into out/tiktok_prompts/images (if pipeline runs)")
    ap.add_argument("--open", action="store_true", help="Open pick + teleprompter in VS Code")
    args = ap.parse_args()

    paths = _default_paths()
    tasks_dir = paths["tasks_dir"]
    picks_dir = paths["picks_dir"]
    out_root = paths["out_root"]

    # Resolve task path
    task_arg = args.task.strip()
    task_path: Path
    if task_arg.lower().endswith(".json") or "/" in task_arg or "\\" in task_arg:
        task_path = (REPO_ROOT / task_arg).resolve()
    else:
        task_path = (tasks_dir / f"{task_arg}.json").resolve()

    if not task_path.exists():
        raise SystemExit(f"Task not found: {task_path}")

    task = _read_task_file(task_path)

    # Refresh sources
    if not args.no_refresh_images and task.refresh_images:
        _maybe_refresh_images(paths["images_in"], out_root, copy_images=args.copy_images)

    if args.refresh_export or task.refresh_export:
        export_dir = args.export or task.export_dir or ""
        if not export_dir:
            raise SystemExit("Missing export dir. Provide --export or set CHATGPT_EXPORT_DIR.")
        _maybe_refresh_export(task, export_dir)

    # Draft inputs (only include the ones that exist)
    drafts: list[Path] = []
    for p in (paths["drafts_main"], paths["drafts_chat"], paths["drafts_compiled"]):
        if p.exists():
            drafts.append(p)

    if not drafts:
        raise SystemExit("No drafts found. Run image pipeline and/or export extractor first.")

    # Select → pick → teleprompter
    queue_md = _select_for_task(task, drafts)
    pick_md = _copy_pick(task, queue_md, picks_dir)
    tele_md = _teleprompter(pick_md, top=task.teleprompter_top)

    print("OK: tiktokpick complete")
    print(f"- task: {task.name}")
    print(f"- pick: {pick_md}")
    print(f"- teleprompter: {tele_md}")

    if args.open:
        _open_in_code(pick_md)
        _open_in_code(tele_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
