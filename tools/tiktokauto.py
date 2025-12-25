#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AutoConfig:
    images_in: Path
    images_ingested_root: Path
    out_root: Path
    state_dir: Path
    export_dir: Optional[str]


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")


def _dir_mtime_recursive(d: Path) -> float:
    if not d.exists():
        return 0.0
    newest = d.stat().st_mtime
    for p in d.rglob("*"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except FileNotFoundError:
            continue
    return newest


def _has_files(d: Path) -> bool:
    if not d.exists():
        return False
    return any(p.is_file() for p in d.iterdir())


def _ingest_images(cfg: AutoConfig, copy_images: bool) -> bool:
    if not _has_files(cfg.images_in):
        return False

    cmd = [
        sys.executable,
        "tools/tiktok_pipeline.py",
        "--in",
        str(cfg.images_in),
        "--out",
        str(cfg.out_root),
    ]
    if copy_images:
        cmd.append("--copy-images")

    _run(cmd)

    stamp = _now_stamp()
    dest = cfg.images_ingested_root / stamp
    dest.mkdir(parents=True, exist_ok=True)

    # Move only files (leave folders alone)
    for p in list(cfg.images_in.iterdir()):
        if p.is_file():
            shutil.move(str(p), str(dest / p.name))

    return True


def _maybe_refresh_export(cfg: AutoConfig, force: bool, min_score: float, user_only: bool, roles: str) -> bool:
    if not cfg.export_dir:
        return False

    export_path = Path(cfg.export_dir)
    if not export_path.exists():
        return False

    state_path = cfg.state_dir / "export_state.json"
    state = _load_state(state_path)
    last_seen = float(state.get("last_seen_mtime", 0.0))

    current = _dir_mtime_recursive(export_path)

    if not force and current <= last_seen:
        return False

    cmd = [
        sys.executable,
        "tools/tiktok_prompt_extract_from_export.py",
        "--export",
        str(export_path),
        "--min-score",
        str(min_score),
        "--roles",
        roles,
    ]
    if user_only:
        cmd.append("--user-only")

    _run(cmd)

    _save_state(state_path, {"last_seen_mtime": current, "export_dir": str(export_path)})
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Automatic intake: ingest images (if any), archive originals, refresh export if newer, then run tiktokpick."
    )
    ap.add_argument("task", help="Task slug or task json path (same as tiktokpick)")
    ap.add_argument("--export", default=os.environ.get("CHATGPT_EXPORT_DIR", ""), help="Path to chatgpt-export folder")
    ap.add_argument("--refresh-export", action="store_true", help="Force export refresh this run")
    ap.add_argument("--export-min-score", type=float, default=0.30)
    ap.add_argument("--export-user-only", action="store_true")
    ap.add_argument("--export-roles", default="both")
    ap.add_argument("--no-images", action="store_true", help="Skip images ingest even if images_in has files")
    ap.add_argument("--copy-images", action="store_true", help="Copy images into out/tiktok_prompts/images when ingesting")
    ap.add_argument("--open", action="store_true", help="Open results in VS Code")
    args = ap.parse_args()

    cfg = AutoConfig(
        images_in=REPO_ROOT / "notes" / "tiktok" / "images_in",
        images_ingested_root=REPO_ROOT / "notes" / "tiktok" / "images_ingested",
        out_root=REPO_ROOT / "out" / "tiktok_prompts",
        state_dir=REPO_ROOT / "out" / "tiktok_prompts" / ".state",
        export_dir=args.export.strip() or None,
    )

    did_images = False
    if not args.no_images:
        did_images = _ingest_images(cfg, copy_images=args.copy_images)

    did_export = _maybe_refresh_export(
        cfg,
        force=args.refresh_export,
        min_score=args.export_min_score,
        user_only=args.export_user_only,
        roles=args.export_roles,
    )

    pick_cmd = [sys.executable, "tools/tiktokpick.py", args.task]
    if args.open:
        pick_cmd.append("--open")

    _run(pick_cmd)

    summary = {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "task": args.task,
        "images_ingested": did_images,
        "export_refreshed": did_export,
        "export_dir": cfg.export_dir,
    }
    _save_state(cfg.state_dir / "last_run.json", summary)
    print("OK: tiktokauto complete")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
