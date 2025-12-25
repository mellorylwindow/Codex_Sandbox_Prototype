#!/usr/bin/env python3
"""
TikTok Prompt Project — Pipeline Runner (offline-first)

Runs:
  1) tools/tiktok_prompt_ingest.py
  2) tools/tiktok_prompt_compile.py

Outputs:
  - <out>/prompts.jsonl, <out>/index.md
  - <out>/drafts.jsonl, <out>/drafts.md

Why a pipeline script?
- One command for the whole flow
- Keeps everything local + deterministic
- No hidden state
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _repo_root() -> Path:
    # tools/tiktok_pipeline.py -> tools -> repo root
    return Path(__file__).resolve().parents[1]


def _run(cmd: List[str]) -> int:
    # Keep output visible; avoid Windows cp1252 decode blowups
    proc = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Run TikTok image->OCR->drafts pipeline.")
    ap.add_argument("--in", dest="in_dir", required=True, help="Input folder of images")
    ap.add_argument("--out", dest="out_dir", default="out/tiktok_prompts", help="Output folder")
    ap.add_argument("--copy-images", action="store_true", help="Copy images into <out>/images/")

    # pass-through knobs for ingest
    ap.add_argument("--lang", default="eng", help="Tesseract language code (default: eng)")
    ap.add_argument("--psm", type=int, default=None, help="Tesseract --psm (optional)")
    ap.add_argument("--oem", type=int, default=None, help="Tesseract --oem (optional)")

    ap.add_argument("--skip-ingest", action="store_true", help="Skip ingest step")
    ap.add_argument("--skip-compile", action="store_true", help="Skip compile step")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")

    args = ap.parse_args()

    root = _repo_root()
    py = sys.executable

    ingest = root / "tools" / "tiktok_prompt_ingest.py"
    compile_ = root / "tools" / "tiktok_prompt_compile.py"

    if not ingest.exists():
        print(f"ERROR: missing ingest script: {ingest}")
        return 2
    if not compile_.exists():
        print(f"ERROR: missing compile script: {compile_}")
        return 2

    # Step 1: ingest
    ingest_cmd = [
        py,
        str(ingest),
        "--in",
        str(args.in_dir),
        "--out",
        str(args.out_dir),
        "--lang",
        str(args.lang),
    ]
    if args.copy_images:
        ingest_cmd.append("--copy-images")
    if args.psm is not None:
        ingest_cmd += ["--psm", str(args.psm)]
    if args.oem is not None:
        ingest_cmd += ["--oem", str(args.oem)]

    # Step 2: compile
    compile_cmd = [
        py,
        str(compile_),
        "--in",
        str(Path(args.out_dir) / "prompts.jsonl"),
        "--outdir",
        str(args.out_dir),
    ]

    if args.dry_run:
        print("DRY RUN:")
        if not args.skip_ingest:
            print("  ", " ".join(ingest_cmd))
        if not args.skip_compile:
            print("  ", " ".join(compile_cmd))
        return 0

    if not args.skip_ingest:
        rc = _run(ingest_cmd)
        if rc != 0:
            print(f"ERROR: ingest failed with code {rc}")
            return rc

    if not args.skip_compile:
        rc = _run(compile_cmd)
        if rc != 0:
            print(f"ERROR: compile failed with code {rc}")
            return rc

    print("OK: pipeline complete")
    print(f"- {Path(args.out_dir) / 'prompts.jsonl'}")
    print(f"- {Path(args.out_dir) / 'index.md'}")
    print(f"- {Path(args.out_dir) / 'drafts.jsonl'}")
    print(f"- {Path(args.out_dir) / 'drafts.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
