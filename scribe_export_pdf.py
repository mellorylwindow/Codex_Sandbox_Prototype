from __future__ import annotations

import argparse
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Scribe Export</title>
  <style>
    @page { margin: 18mm 14mm; }
    body { font-family: Arial, Helvetica, sans-serif; font-size: 12.5px; line-height: 1.45; color: #111; }
    h1 { font-size: 20px; margin: 0 0 10px; }
    h2 { font-size: 16px; margin: 18px 0 8px; }
    h3 { font-size: 14px; margin: 14px 0 6px; }
    code { font-family: Consolas, "Courier New", monospace; font-size: 11.5px; }
    img { max-width: 100%; height: auto; border: 1px solid #e5e5e5; border-radius: 6px; }
    ul { margin-top: 6px; }
    li { margin: 4px 0; }
    .pagebreak { page-break-before: always; }
    .meta { color: #555; margin-bottom: 10px; }
  </style>
</head>
<body>
__CONTENT__
</body>
</html>
"""


def build_html(md_text: str) -> str:
    html_body = markdown.markdown(md_text, extensions=["extra"])
    return HTML_TEMPLATE.replace("__CONTENT__", html_body)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Scribe steps.md (+ screenshots) to PDF via Playwright.")
    ap.add_argument("--in", dest="in_dir", required=True, help="Scribe output folder (contains steps.md)")
    ap.add_argument("--out", dest="out_pdf", required=True, help="Output PDF path")
    ap.add_argument("--html", dest="out_html", default="", help="Optional output HTML path (default: <in>/steps.html)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    md_path = in_dir / "steps.md"
    if not md_path.exists():
        raise SystemExit(f"Missing steps.md at: {md_path}")

    out_pdf = Path(args.out_pdf).resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    out_html = Path(args.out_html).resolve() if args.out_html else (in_dir / "steps.html")

    md_text = md_path.read_text(encoding="utf-8")
    html = build_html(md_text)
    out_html.write_text(html, encoding="utf-8")

    # Render HTML -> PDF using local file URL so relative image paths work.
    file_url = out_html.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(file_url, wait_until="networkidle")
        page.emulate_media(media="screen")
        page.pdf(
            path=str(out_pdf),
            format="Letter",
            print_background=True,
        )
        browser.close()

    print(f"OK: wrote HTML -> {out_html}")
    print(f"OK: wrote PDF  -> {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
