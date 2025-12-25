#!/usr/bin/env python3
"""
tools/tiktok_prompt_ingest.py

OCR a folder of images into JSONL + an index markdown.
- Robust subprocess decoding (forces UTF-8, avoids cp1252 UnicodeDecodeError)
- Optional corrections map (string replacements) via JSON
- Optional image copying into out_dir/images with stable hashed names

Output:
- <out>/prompts.jsonl  (one JSON object per image)
- <out>/index.md       (human-friendly list)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class OcrResult:
    id: str
    source_path: str
    source_name: str
    copied_image: Optional[str]
    extracted_text: str
    raw_text: str
    engine: str
    created_utc: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iter_images(inputs: List[Path]) -> Iterable[Path]:
    for p in inputs:
        if p.is_file():
            if p.suffix.lower() in IMAGE_EXTS:
                yield p
            continue
        if p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS:
                    yield fp


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_slug(s: str, max_len: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s[:max_len] or "item"


def _read_corrections(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(data, dict):
        # {"bad":"good", ...}
        return {str(k): str(v) for k, v in data.items()}
    raise SystemExit(f"corrections file must be a JSON object/dict: {path}")


def _apply_corrections(text: str, corrections: Dict[str, str]) -> str:
    out = text
    for bad, good in corrections.items():
        if bad:
            out = out.replace(bad, good)
    return out


def _run_tesseract(
    tesseract_exe: Path,
    image_path: Path,
    lang: str,
    psm: int,
    oem: int,
    extra_args: List[str],
    tessdata_prefix: Optional[Path],
) -> Tuple[str, str]:
    """
    Returns (stdout_text, stderr_text) decoded as UTF-8 with errors replaced.
    """
    cmd = [
        str(tesseract_exe),
        str(image_path),
        "stdout",
        "-l",
        lang,
        "--psm",
        str(psm),
        "--oem",
        str(oem),
        *extra_args,
    ]

    env = os.environ.copy()
    if tessdata_prefix:
        env["TESSDATA_PREFIX"] = str(tessdata_prefix)

    # IMPORTANT: capture bytes and decode ourselves as UTF-8 (tesseract outputs UTF-8)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    return out, err


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_index_md(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# TikTok prompt OCR index")
    lines.append("")
    lines.append(f"- generated: {_utc_now_iso()}")
    lines.append(f"- records: {len(rows)}")
    lines.append("")
    for i, r in enumerate(rows, start=1):
        title = r.get("source_name") or r.get("source_path") or f"item_{i}"
        copied = r.get("copied_image")
        text = (r.get("extracted_text") or "").strip()
        preview = text.replace("\r", "").split("\n")[0][:180] if text else "(no text)"
        lines.append(f"## {i}. {title}")
        if copied:
            lines.append(f"- image: `{copied}`")
        lines.append(f"- id: `{r.get('id','')}`")
        lines.append("")
        lines.append("```text")
        lines.append(preview)
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="inputs",
        nargs="+",
        required=True,
        help="Input image file(s) and/or folder(s).",
    )
    ap.add_argument("--out", default="out/tiktok_prompts", help="Output folder.")
    ap.add_argument(
        "--tesseract",
        default="tesseract",
        help="Path to tesseract executable (or rely on PATH).",
    )
    ap.add_argument("--lang", default="eng", help="Tesseract language (default: eng).")
    ap.add_argument("--psm", type=int, default=6, help="Tesseract --psm (default: 6).")
    ap.add_argument("--oem", type=int, default=1, help="Tesseract --oem (default: 1).")
    ap.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra tesseract args (e.g. -c preserve_interword_spaces=1).",
    )
    ap.add_argument(
        "--tessdata-prefix",
        default=None,
        help="Optional TESSDATA_PREFIX directory (contains tessdata).",
    )
    ap.add_argument(
        "--corrections",
        default="notes/tiktok/ocr_corrections.json",
        help="JSON dict of string replacements for OCR cleanup.",
    )
    ap.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into <out>/images with hashed filename prefix.",
    )

    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    tesseract_exe = Path(args.tesseract)
    tessdata_prefix = Path(args.tessdata_prefix) if args.tessdata_prefix else None

    corrections_path = Path(args.corrections) if args.corrections else None
    corrections = _read_corrections(corrections_path)

    input_paths = [Path(x) for x in args.inputs]
    images = list(_iter_images(input_paths))

    if not images:
        print("No images found.", file=sys.stderr)
        return 2

    images_out_dir = out_dir / "images"
    if args.copy_images:
        images_out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    created = _utc_now_iso()

    for img in images:
        file_hash = _sha256_file(img)[:12]
        stable_id = file_hash

        copied_rel: Optional[str] = None
        if args.copy_images:
            dest_name = f"{file_hash}__{img.name}"
            dest = images_out_dir / dest_name
            if not dest.exists():
                shutil.copy2(img, dest)
            copied_rel = str(Path("images") / dest_name).replace("\\", "/")

        raw_text, err_text = _run_tesseract(
            tesseract_exe=tesseract_exe,
            image_path=img,
            lang=args.lang,
            psm=args.psm,
            oem=args.oem,
            extra_args=args.extra,
            tessdata_prefix=tessdata_prefix,
        )

        # Some tesseract versions emit progress/diagnostics to stderr;
        # we keep stderr out of extracted_text, but raw_text is just stdout OCR.
        cleaned = raw_text
        cleaned = cleaned.replace("\x00", "")
        cleaned = cleaned.strip()

        corrected = _apply_corrections(cleaned, corrections).strip()

        row = OcrResult(
            id=stable_id,
            source_path=str(img).replace("\\", "/"),
            source_name=img.name,
            copied_image=copied_rel,
            extracted_text=corrected,  # downstream expects extracted_text
            raw_text=cleaned,
            engine="tesseract",
            created_utc=created,
        )
        rows.append(row.__dict__)

        # Keep stderr visible but non-fatal for debugging if needed
        if err_text.strip():
            # Quiet by default; comment this in if you want per-file diagnostics
            pass

    prompts_jsonl = out_dir / "prompts.jsonl"
    index_md = out_dir / "index.md"
    _write_jsonl(prompts_jsonl, rows)
    _write_index_md(index_md, rows)

    print(f"OK: wrote {len(rows)} records")
    print(f"- {prompts_jsonl.resolve()}")
    print(f"- {index_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
