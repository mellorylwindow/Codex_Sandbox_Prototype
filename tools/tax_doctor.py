from __future__ import annotations

import shutil
import sys

def check_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False

def main() -> int:
    print("🧾 tax_doctor")
    ok = True

    have_pypdf = check_import("pypdf") or check_import("PyPDF2")
    print(f"- PDF text:      {'OK' if have_pypdf else 'MISSING'}  (pypdf or PyPDF2)")
    ok = ok and have_pypdf

    have_pillow = check_import("PIL")
    print(f"- Image open:    {'OK' if have_pillow else 'MISSING'}  (Pillow)")
    ok = ok and have_pillow

    have_pytess = check_import("pytesseract")
    print(f"- OCR binding:   {'OK' if have_pytess else 'MISSING'}  (pytesseract)")
    # OCR is optional (PDFs still work), so don't fail overall just for this.
    if not have_pytess:
        print("  note: OCR for images will not run until pytesseract is installed.")

    tess_bin = shutil.which("tesseract")
    print(f"- tesseract exe: {'OK' if tess_bin else 'MISSING'}")
    if not tess_bin:
        print("  note: OCR requires the system 'tesseract' binary on PATH.")

    print("")
    print("If OCR is missing, PDFs can still extract text; images will log method=error.")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
