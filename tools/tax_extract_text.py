from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# PDF text extraction (pure python)
def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        # Fallback to PyPDF2 name if older env
        from PyPDF2 import PdfReader  # type: ignore

    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(f"\n\n===== PAGE {i+1} =====\n{txt}")
    return "\n".join(parts).strip()


def ocr_image_text(path: Path) -> str:
    """
    OCR for images. Requires:
      - pip package: pytesseract
      - system binary: tesseract
    If not available, we raise a clear error.
    """
    try:
        import pytesseract  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pytesseract (pip install pytesseract)") from e

    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: Pillow (pip install pillow)") from e

    img = Image.open(path)
    # Basic normalization helps OCR a bit without getting fancy.
    # Convert to RGB to avoid mode issues.
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return (pytesseract.image_to_string(img) or "").strip()


@dataclass
class ExtractMeta:
    doc_id: str
    canonical_path: str
    method: str                 # pdf_text | ocr_image | skipped | error
    extracted_at_iso: str
    bytes_out: int
    error: Optional[str] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_manifest_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_text(out_path: Path, txt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt, encoding="utf-8", newline="\n")


def write_meta(out_path: Path, meta: ExtractMeta) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract readable text from canonical tax documents (PDF + images).")
    ap.add_argument("--manifest", default="tax_intake/index/manifest.jsonl")
    ap.add_argument("--out-text", default="tax_intake/30_extracted/text")
    ap.add_argument("--out-meta", default="tax_intake/30_extracted/meta")
    ap.add_argument("--only-canonical", action="store_true", help="Skip duplicate rows; process only canonical records.")
    ap.add_argument("--force", action="store_true", help="Re-extract even if output exists.")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N docs (0 = all).")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_text_dir = Path(args.out_text)
    out_meta_dir = Path(args.out_meta)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        return 2

    rows = read_manifest_jsonl(manifest_path)
    count = 0
    ok = 0
    err = 0
    skip = 0

    for r in rows:
        if args.only_canonical and r.get("duplicate_of"):
            continue

        doc_id = r.get("doc_id") or ""
        canonical_path = Path(r.get("canonical_path") or "")
        if not doc_id or not canonical_path.exists():
            continue

        ext = canonical_path.suffix.lower()
        out_txt = out_text_dir / f"{doc_id}.txt"
        out_meta = out_meta_dir / f"{doc_id}.meta.json"

        if not args.force and out_txt.exists() and out_meta.exists():
            skip += 1
            continue

        extracted_at = utc_now_iso()

        try:
            if ext == ".pdf":
                txt = extract_pdf_text(canonical_path)
                method = "pdf_text"
            elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
                txt = ocr_image_text(canonical_path)
                method = "ocr_image"
            else:
                # Unknown type; keep metadata so you can see it was skipped.
                txt = ""
                method = "skipped"

            write_text(out_txt, txt)
            meta = ExtractMeta(
                doc_id=doc_id,
                canonical_path=str(canonical_path),
                method=method,
                extracted_at_iso=extracted_at,
                bytes_out=len(txt.encode("utf-8")),
                error=None,
            )
            write_meta(out_meta, meta)
            ok += 1

        except Exception as e:
            # Still write meta so you can see failures and rerun later
            meta = ExtractMeta(
                doc_id=doc_id,
                canonical_path=str(canonical_path),
                method="error",
                extracted_at_iso=extracted_at,
                bytes_out=0,
                error=str(e),
            )
            write_meta(out_meta, meta)
            err += 1

        count += 1
        if args.limit and count >= args.limit:
            break

    print(f"Processed: {count}")
    print(f"OK:        {ok}")
    print(f"Skipped:   {skip}")
    print(f"Errors:    {err}")
    print(f"Out text:  {out_text_dir}")
    print(f"Out meta:  {out_meta_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
