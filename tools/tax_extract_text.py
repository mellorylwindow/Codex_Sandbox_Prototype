#!/usr/bin/env python3
"""
Extract text from ingested tax assets.

- PDFs: try embedded text via pypdf; if empty and --pdf-ocr is set, render pages via pypdfium2 and OCR.
- Images: OCR via Tesseract.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Bad JSON on line {i} of {path}: {e}")
    return rows


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")


def guess_tesseract_cmd() -> str | None:
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        return env_cmd

    which = shutil.which("tesseract")
    if which and Path(which).exists():
        return which

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


@dataclass
class OCRSettings:
    lang: str = "eng"
    psm: int = 6
    oem: int = 1
    dpi: int = 250
    resize: float = 1.5
    sharpen: bool = True
    autocontrast: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lang": self.lang,
            "psm": self.psm,
            "oem": self.oem,
            "dpi": self.dpi,
            "resize": self.resize,
            "sharpen": self.sharpen,
            "autocontrast": self.autocontrast,
        }


def extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")  # type: ignore[attr-defined]
        except Exception:
            pass

    parts: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()


def render_pdf_pages(pdf_path: Path, dpi: int, max_pages: int | None = None) -> List[Image.Image]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)

    scale = dpi / 72.0
    images: List[Image.Image] = []
    for i in range(n_pages):
        page = pdf[i]
        images.append(page.render(scale=scale).to_pil())
    return images


def preprocess_image(img: Image.Image, settings: OCRSettings) -> Image.Image:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img = img.convert("L")

    if settings.resize and settings.resize != 1.0:
        w, h = img.size
        img = img.resize((max(1, int(w * settings.resize)), max(1, int(h * settings.resize))), resample=Image.LANCZOS)

    if settings.autocontrast:
        img = ImageOps.autocontrast(img)

    img = ImageEnhance.Contrast(img).enhance(1.2)

    if settings.sharpen:
        img = img.filter(ImageFilter.SHARPEN)

    return img


def ocr_image(img: Image.Image, settings: OCRSettings) -> str:
    import pytesseract

    cmd = guess_tesseract_cmd()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

    config = f"--oem {settings.oem} --psm {settings.psm}"
    img2 = preprocess_image(img, settings)
    return (pytesseract.image_to_string(img2, lang=settings.lang, config=config) or "").strip()


def extract_one(
    row: Dict[str, Any],
    out_text_dir: Path,
    out_meta_dir: Path,
    settings: OCRSettings,
    force: bool,
    only_canonical: bool,
    pdf_ocr: bool,
    max_pdf_pages: int | None,
) -> Tuple[str, str]:
    sha = (row.get("sha256") or "").strip()
    abs_path = (row.get("abs_path") or "").strip()
    status = (row.get("status") or "").strip()

    if only_canonical and status == "duplicate":
        return "skipped", "duplicate"

    if not sha or not abs_path:
        return "error", "missing_fields"

    src = Path(abs_path)
    out_txt = out_text_dir / f"{sha}.txt"
    out_meta = out_meta_dir / f"{sha}.json"

    meta: Dict[str, Any] = dict(row)
    meta["extracted_at"] = utc_iso()
    meta["ocr_settings"] = settings.as_dict()

    if not src.exists():
        meta["ok"] = False
        meta["mode"] = "missing"
        meta["error"] = f"Missing file on disk: {abs_path}"
        ensure_dir(out_text_dir); write_text(out_txt, "")
        ensure_dir(out_meta_dir); write_json(out_meta, meta)
        return "missing", "not_found"

    if not force and out_txt.exists() and out_meta.exists():
        return "ok", "cached"

    ext = src.suffix.lower()

    try:
        if ext == ".pdf":
            text = extract_pdf_text(src)
            if len(text.strip()) >= 50:
                meta["ok"] = True
                meta["mode"] = "pdf_text"
                ensure_dir(out_text_dir); write_text(out_txt, text)
                ensure_dir(out_meta_dir); write_json(out_meta, meta)
                return "ok", "pdf_text"

            if not pdf_ocr:
                meta["ok"] = False
                meta["mode"] = "pdf_text_empty"
                meta["error"] = "PDF has little/no embedded text and pdf_ocr is disabled."
                ensure_dir(out_text_dir); write_text(out_txt, "")
                ensure_dir(out_meta_dir); write_json(out_meta, meta)
                return "error", "pdf_text_empty"

            imgs = render_pdf_pages(src, dpi=settings.dpi, max_pages=max_pdf_pages)
            parts = [ocr_image(im, settings) for im in imgs]
            joined = "\n\n".join(p for p in parts if p.strip()).strip()

            meta["mode"] = "pdf_ocr"
            meta["ok"] = bool(joined)
            if not joined:
                meta["error"] = "PDF OCR produced empty output."

            ensure_dir(out_text_dir); write_text(out_txt, joined)
            ensure_dir(out_meta_dir); write_json(out_meta, meta)
            return ("ok" if joined else "error"), "pdf_ocr"

        # image
        img = Image.open(src)
        text = ocr_image(img, settings)
        meta["ok"] = True
        meta["mode"] = "image_ocr"
        ensure_dir(out_text_dir); write_text(out_txt, text)
        ensure_dir(out_meta_dir); write_json(out_meta, meta)
        return "ok", "image_ocr"

    except Exception as e:
        meta["ok"] = False
        meta["mode"] = "exception"
        meta["error"] = f"{type(e).__name__}: {e}"
        ensure_dir(out_text_dir); write_text(out_txt, "")
        ensure_dir(out_meta_dir); write_json(out_meta, meta)
        return "error", "exception"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract text from ingested tax assets.")
    p.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    p.add_argument("--out-text", default=None, help="Output text directory")
    p.add_argument("--out-meta", default=None, help="Output meta directory")
    p.add_argument("--force", action="store_true", help="Re-extract even if outputs exist")
    p.add_argument("--only-canonical", action="store_true", help="Skip rows with status == duplicate")

    p.add_argument("--lang", default="eng")
    p.add_argument("--psm", type=int, default=6)
    p.add_argument("--oem", type=int, default=1)
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--resize", type=float, default=1.5)
    p.add_argument("--no-sharpen", action="store_true")
    p.add_argument("--no-autocontrast", action="store_true")

    p.add_argument("--pdf-ocr", action="store_true", help="OCR PDFs that have no embedded text")
    p.add_argument("--max-pdf-pages", type=int, default=None, help="Limit pages OCR'd per PDF")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    manifest_path = Path(args.manifest)
    out_text_dir = Path(args.out_text) if args.out_text else Path("notes/tax/work/extracted_text")
    out_meta_dir = Path(args.out_meta) if args.out_meta else out_text_dir / "_meta"

    settings = OCRSettings(
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        dpi=args.dpi,
        resize=args.resize,
        sharpen=not args.no_sharpen,
        autocontrast=not args.no_autocontrast,
    )

    rows = read_jsonl(manifest_path)
    total = len(rows)
    candidates = sum(1 for r in rows if not (args.only_canonical and (r.get("status") == "duplicate")))

    counts = collections.Counter()
    for r in rows:
        status, _detail = extract_one(
            r,
            out_text_dir=out_text_dir,
            out_meta_dir=out_meta_dir,
            settings=settings,
            force=args.force,
            only_canonical=args.only_canonical,
            pdf_ocr=args.pdf_ocr,
            max_pdf_pages=args.max_pdf_pages,
        )
        counts[status] += 1

    print(f"Manifest rows: {total}")
    print(f"Candidates:    {candidates}")
    print(f"Processed: {sum(counts.values())}")
    print(f"OK:        {counts['ok']}")
    print(f"Skipped:   {counts['skipped']}")
    print(f"Errors:    {counts['error']}")
    print(f"Missing:   {counts['missing']}")
    print(f"Out text:  {out_text_dir.resolve()}")
    print(f"Out meta:  {out_meta_dir.resolve()}")

    return 0 if (counts["error"] == 0 and counts["missing"] == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
