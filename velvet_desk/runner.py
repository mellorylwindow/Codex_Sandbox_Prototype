from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import ModuleSpec


def app_data_dir() -> Path:
    # Local-first, outside the repo by default (good for tax images).
    base = Path.home() / ".swainlabs" / "velvet_desk"
    base.mkdir(parents=True, exist_ok=True)
    (base / "runs").mkdir(parents=True, exist_ok=True)
    return base


def render_command(template: List[str], values: Dict[str, object]) -> List[str]:
    def sub_one(s: str) -> str:
        out = s
        for k, v in values.items():
            out = out.replace("{" + k + "}", str(v))
        return out

    return [sub_one(t) for t in template]


class JobHandle:
    def __init__(self, proc: subprocess.Popen[str], q: "queue.Queue[Tuple[str, str]]"):
        self.proc = proc
        self.q = q

    def terminate(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass


def _stream(pipe, q: "queue.Queue[Tuple[str, str]]", stream_name: str) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            q.put((stream_name, line.rstrip("\n")))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def run_module(module: ModuleSpec, values: Dict[str, object], cwd: Optional[Path] = None) -> Tuple[JobHandle, Path]:
    cmd = render_command(module.command, values)

    q: "queue.Queue[Tuple[str, str]]" = queue.Queue()

    start = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{start}_{module.id}"
    run_dir = app_data_dir() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "module": asdict(module),
        "values": {k: str(v) for k, v in values.items()},
        "command": cmd,
        "cwd": str(cwd or Path.cwd()),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Windows note: use text mode + universal newlines
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd or Path.cwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        universal_newlines=True,
        env=os.environ.copy(),
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    t1 = threading.Thread(target=_stream, args=(proc.stdout, q, "stdout"), daemon=True)
    t2 = threading.Thread(target=_stream, args=(proc.stderr, q, "stderr"), daemon=True)
    t1.start()
    t2.start()

    return JobHandle(proc=proc, q=q), manifest_path
