#!/usr/bin/env python3
"""
tax_extract_text.py — robust OCR extractor for Tax Intake

Reads a manifest.jsonl where rows may look like:
  kind: receipt_asset
  status: ingested | duplicate
  sha256: ...
  dest_rel: out/tax/ingested/YYYY/MM/<sha>__name.jpg   (may be missing)
  src_rel:  notes/tax/images_in/<batch>/<name>.jpg     (often present)

Behavior:
- Chooses source path in this order: dest_rel -> src_rel -> abs_path (if present)
- Processes ingested + duplicate by default
- If --only-canonical: skip rows where status == "duplicate"
- Writes:
    tax_intake/30_extracted/text/<sha256>.txt
    tax_intake/30_extracted/meta/<sha256>.json
- Keeps metadata for traceability and debugging.

Requires:
  pip install pillow pytesseract
  and a working Tesseract install (Windows: UB-Mannheim recommended).

Tip (Windows):
  Tesseract installed at: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
  tessdata folder at:     C:\\Program Files\\Tesseract-OCR\\tessdata
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Pillow + pytesseract are optional at import-time (we handle missing gracefully)
try:
    from PIL import Image, ImageOps, ImageFilter  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore


# -----------------------------
# time / paths
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def repo_root() -> Path:
    # tools/tax_extract_text.py -> repo root
    return Path(__file__).resolve().parents[1]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            # keep going; bad row gets ignored
            continue
    return rows

def infer_batch_id(manifest_path: Path, rows: List[Dict[str, Any]]) -> str:
    # 1) explicit batch in rows
    for r in rows:
        b = r.get("batch")
        if isinstance(b, str) and b.strip():
            return b.strip()

    # 2) folder name out/tax/batches/<BATCH>/manifest.jsonl
    try:
        return manifest_path.parent.name
    except Exception:
        return "unknown_batch"


# -----------------------------
# tesseract discovery / wiring
# -----------------------------

def _candidate_tesseract_paths() -> List[Path]:
    candidates: List[Path] = []

    # 1) PATH
    w = shutil.which("tesseract")
    if w:
        candidates.append(Path(w))

    # 2) common Windows installs
    candidates += [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.EXE"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.EXE"),
    ]
    # de-dupe while preserving order
    seen = set()
    out: List[Path] = []
    for p in candidates:
        ps = str(p).lower()
        if ps in seen:
            continue
        seen.add(ps)
        out.append(p)
    return out

def _candidate_tessdata_dirs(tesseract_exe: Path) -> List[Path]:
    # TESSDATA_PREFIX must point to *tessdata directory*.
    base = tesseract_exe.parent
    candidates = [
        base / "tessdata",
        base,
    ]
    # if user already set something, consider it too
    cur = os.environ.get("TESSDATA_PREFIX")
    if cur:
        candidates.insert(0, Path(cur))
        # if they set it to parent, try <parent>/tessdata
        candidates.insert(1, Path(cur) / "tessdata")
    # de-dupe
    seen = set()
    out: List[Path] = []
    for p in candidates:
        ps = str(p).lower()
        if ps in seen:
            continue
        seen.add(ps)
        out.append(p)
    return out

def configure_tesseract(debug: bool = False) -> Dict[str, Any]:
    """
    Returns diagnostic info:
      { "ok": bool, "tesseract_cmd": str|None, "tessdata_prefix": str|None, "error": str|None }
    """
    info: Dict[str, Any] = {"ok": False, "tesseract_cmd": None, "tessdata_prefix": None, "error": None}

    if pytesseract is None:
        info["error"] = "Missing pytesseract. Install: python -m pip install pytesseract"
        return info
    if Image is None:
        info["error"] = "Missing Pillow. Install: python -m pip install pillow"
        return info

    # Find tesseract.exe
    texe: Optional[Path] = None
    for c in _candidate_tesseract_paths():
        if c.exists():
            texe = c
            break
    if texe is None:
        info["error"] = "tesseract.exe not found. Install Tesseract OCR and/or add it to PATH."
        return info

    # Wire pytesseract to full path
    pytesseract.pytesseract.tesseract_cmd = str(texe)
    info["tesseract_cmd"] = str(texe)

    # Set TESSDATA_PREFIX to tessdata folder (not the parent)
    tessdata_dir: Optional[Path] = None
    for d in _candidate_tessdata_dirs(texe):
        # real tessdata dir has eng.traineddata inside
        if d.is_dir() and (d / "eng.traineddata").exists():
            tessdata_dir = d
            break
    # If not found, try a softer check: exists and has *.traineddata
    if tessdata_dir is None:
        for d in _candidate_tessdata_dirs(texe):
            if d.is_dir() and any(d.glob("*.traineddata")):
                tessdata_dir = d
                break

    if tessdata_dir is None:
        info["error"] = (
            "Could not locate tessdata directory (eng.traineddata). "
            "Set TESSDATA_PREFIX to your tessdata folder, e.g. "
            r'C:\Program Files\Tesseract-OCR\tessdata'
        )
        return info

    os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
    info["tessdata_prefix"] = str(tessdata_dir)

    # Sanity check: can we query version?
    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as e:
        info["error"] = f"pytesseract cannot run tesseract: {e!r}"
        return info

    info["ok"] = True
    if debug:
        print(f"[debug] pytesseract.tesseract_cmd={info['tesseract_cmd']}")
        print(f"[debug] Set TESSDATA_PREFIX={info['tessdata_prefix']}")
    return info


# -----------------------------
# OCR / preprocessing
# -----------------------------

@dataclass
class OcrSettings:
    lang: str = "eng"
    psm: int = 6
    oem: int = 1
    dpi: int = 300
    resize: float = 1.5
    sharpen: bool = True
    autocontrast: bool = True

def preprocess_image(img: "Image.Image", s: OcrSettings) -> "Image.Image":
    # Convert to RGB then grayscale (stable for receipts)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")

    # Upscale (helps thin fonts)
    if s.resize and s.resize != 1.0:
        w, h = img.size
        img = img.resize((max(1, int(w * s.resize)), max(1, int(h * s.resize))))

    if s.autocontrast:
        img = ImageOps.autocontrast(img)

    # Mild sharpen
    if s.sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    return img

def ocr_image(path: Path, s: OcrSettings) -> Tuple[bool, str, Optional[str]]:
    """
    Returns (ok, text, error)
    """
    if pytesseract is None or Image is None:
        return (False, "", "Missing dependencies: pillow and/or pytesseract")

    try:
        img = Image.open(path)  # type: ignore
    except Exception as e:
        return (False, "", f"Cannot open image: {e!r}")

    try:
        img2 = preprocess_image(img, s)
        config = f"--oem {s.oem} --psm {s.psm} -c user_defined_dpi={s.dpi}"
        text = pytesseract.image_to_string(img2, lang=s.lang, config=config)  # type: ignore
        # normalize
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return (True, text, None)
    except Exception as e:
        return (False, "", repr(e))


# -----------------------------
# manifest row -> file resolution
# -----------------------------

def resolve_source_path(repo: Path, row: Dict[str, Any]) -> Tuple[Optional[Path], str]:
    """
    Returns (abs_path, reason_if_missing)
    """
    for key in ("dest_rel", "src_rel"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            p = repo / v
            if p.exists():
                return (p, "")
            # If manifest used Windows separators accidentally
            p2 = repo / v.replace("\\", "/")
            if p2.exists():
                return (p2, "")

    # Some rows may already contain absolute path
    v = row.get("abs_path")
    if isinstance(v, str) and v.strip():
        p = Path(v)
        if p.exists():
            return (p, "")

    return (None, "missing_source_path")


# -----------------------------
# main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    ap.add_argument("--out-text", default="tax_intake/30_extracted/text", help="Output text directory")
    ap.add_argument("--out-meta", default="tax_intake/30_extracted/meta", help="Output meta directory")
    ap.add_argument("--force", action="store_true", help="Re-OCR even if outputs exist")
    ap.add_argument("--only-canonical", action="store_true", help="Skip rows with status == duplicate")
    ap.add_argument("--lang", default="eng", help="Tesseract language (default: eng)")
    ap.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode (default: 6)")
    ap.add_argument("--oem", type=int, default=1, help="Tesseract OCR engine mode (default: 1)")
    ap.add_argument("--dpi", type=int, default=300, help="DPI hint passed to tesseract (default: 300)")
    ap.add_argument("--resize", type=float, default=1.5, help="Preprocess resize scale (default: 1.5)")
    ap.add_argument("--no-sharpen", action="store_true", help="Disable sharpen")
    ap.add_argument("--no-autocontrast", action="store_true", help="Disable autocontrast")
    ap.add_argument("--debug", action="store_true", help="Verbose diagnostics")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    repo = repo_root()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo / manifest_path).resolve()
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        return 2

    rows = read_jsonl(manifest_path)
    batch_id = infer_batch_id(manifest_path, rows)

    out_text_dir = (repo / args.out_text).resolve()
    out_meta_dir = (repo / args.out_meta).resolve()
    ensure_dir(out_text_dir)
    ensure_dir(out_meta_dir)

    # Configure tesseract
    tinfo = configure_tesseract(debug=args.debug)
    if not tinfo["ok"]:
        print(f"ERROR: {tinfo['error']}")
        return 2

    settings = OcrSettings(
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        dpi=args.dpi,
        resize=args.resize,
        sharpen=not args.no_sharpen,
        autocontrast=not args.no_autocontrast,
    )

    # Filter rows
    candidates: List[Dict[str, Any]] = []
    skipped_counts: Dict[str, int] = {}

    for r in rows:
        status = (r.get("status") or "").strip() if isinstance(r.get("status"), str) else ""
        if args.only_canonical and status == "duplicate":
            skipped_counts["duplicate_skipped"] = skipped_counts.get("duplicate_skipped", 0) + 1
            continue
        candidates.append(r)

    print(f"Manifest rows: {len(rows)}")
    print(f"Candidates:    {len(candidates)}")
    if args.debug:
        print(f"[debug] batch={batch_id}")

    processed = ok = err = missing = skipped = 0
    missing_reasons: Dict[str, int] = {}

    for r in candidates:
        sha = r.get("sha256")
        if not isinstance(sha, str) or not sha.strip():
            missing += 1
            missing_reasons["missing_sha256"] = missing_reasons.get("missing_sha256", 0) + 1
            continue
        sha = sha.strip()

        out_txt = out_text_dir / f"{sha}.txt"
        out_meta = out_meta_dir / f"{sha}.json"

        if (not args.force) and out_txt.exists() and out_meta.exists():
            skipped += 1
            continue

        src_path, reason = resolve_source_path(repo, r)
        if src_path is None:
            missing += 1
            missing_reasons[reason] = missing_reasons.get(reason, 0) + 1
            # still write meta so downstream can see why it failed
            meta = {
                "ok": False,
                "error": f"Missing source file ({reason})",
                "sha256": sha,
                "batch": batch_id,
                "status": r.get("status"),
                "kind": r.get("kind"),
                "dest_rel": r.get("dest_rel"),
                "src_rel": r.get("src_rel"),
                "abs_path": r.get("abs_path"),
                "extracted_at": utc_now_iso(),
                "tesseract_cmd": tinfo.get("tesseract_cmd"),
                "tessdata_prefix": tinfo.get("tessdata_prefix"),
            }
            out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            continue

        processed += 1
        ok_flag, text, oerr = ocr_image(src_path, settings)

        meta = {
            "ok": bool(ok_flag),
            "error": oerr,
            "sha256": sha,
            "batch": batch_id,
            "status": r.get("status"),
            "kind": r.get("kind"),
            "original_name": r.get("original_name"),
            "dest_rel": r.get("dest_rel"),
            "src_rel": r.get("src_rel"),
            "abs_path": str(src_path),
            "extracted_at": utc_now_iso(),
            "tesseract_cmd": tinfo.get("tesseract_cmd"),
            "tessdata_prefix": tinfo.get("tessdata_prefix"),
            "ocr_settings": {
                "lang": settings.lang,
                "psm": settings.psm,
                "oem": settings.oem,
                "dpi": settings.dpi,
                "resize": settings.resize,
                "sharpen": settings.sharpen,
                "autocontrast": settings.autocontrast,
            },
            "text_chars": len(text or ""),
        }

        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if ok_flag:
            out_txt.write_text(text, encoding="utf-8", errors="replace")
            ok += 1
        else:
            # still write whatever text we got (often empty), for debugging
            out_txt.write_text(text or "", encoding="utf-8", errors="replace")
            err += 1

    if skipped_counts:
        print("Skipped filters:")
        for k, v in sorted(skipped_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {k}: {v}")

    if missing_reasons:
        print("Missing reasons:")
        for k, v in sorted(missing_reasons.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {k}: {v}")

    print(f"Processed: {processed}")
    print(f"OK:        {ok}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {err}")
    print(f"Missing:   {missing}")
    print(f"Out text:  {out_text_dir}")
    print(f"Out meta:  {out_meta_dir}")

    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
