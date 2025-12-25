from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from openpyxl import Workbook
from openpyxl.utils import get_column_letter


def read_manifest_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def tax_year_from_date(iso_date: Optional[str]) -> Optional[int]:
    if not iso_date:
        return None
    try:
        return int(iso_date.split("-")[0])
    except Exception:
        return None


def excerpt(text: str, max_chars: int = 240) -> str:
    t = " ".join(text.split())
    return (t[:max_chars] + "…") if len(t) > max_chars else t


def load_text(text_dir: Path, doc_id: str) -> str:
    p = text_dir / f"{doc_id}.txt"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def load_lineitems(lines_dir: Path, doc_id: str) -> list[dict[str, Any]]:
    p = lines_dir / f"{doc_id}.lines.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def autosize(ws) -> None:
    for col in range(1, ws.max_column + 1):
        max_len = 0
        col_letter = get_column_letter(col)
        for row in range(1, ws.max_row + 1):
            v = ws.cell(row=row, column=col).value
            if v is None:
                continue
            s = str(v)
            if len(s) > max_len:
                max_len = len(s)
        ws.column_dimensions[col_letter].width = min(max(10, max_len + 2), 60)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export tax review spreadsheet (line-item aware).")
    ap.add_argument("--manifest", default="tax_intake/index/manifest.jsonl")
    ap.add_argument("--text-dir", default="tax_intake/30_extracted/text")
    ap.add_argument("--lines-dir", default="tax_intake/30_extracted/lines")
    ap.add_argument("--out", default="tax_intake/40_reports/tax_lines.xlsx")
    ap.add_argument("--only-canonical", action="store_true", help="Skip duplicates; include only canonical docs.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    text_dir = Path(args.text_dir)
    lines_dir = Path(args.lines_dir)
    out_path = Path(args.out)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        return 2

    rows = read_manifest_jsonl(manifest_path)

    wb = Workbook()
    ws = wb.active
    ws.title = "tax_lines"

    headers = [
        "line_id",
        "doc_id",
        "page",
        "canonical_path",
        "file_type",
        "provider",
        "service_date",
        "amount",
        "service_desc",
        "tax_year",
        "duplicate_of",
        "confidence",
        "excerpt",
        "notes",
    ]
    ws.append(headers)

    out_rows = 0

    for r in rows:
        doc_id = (r.get("doc_id") or "").strip()
        canonical_path = (r.get("canonical_path") or "").strip()
        duplicate_of = r.get("duplicate_of")

        if args.only_canonical and duplicate_of:
            continue
        if not doc_id or not canonical_path:
            continue

        ext = Path(canonical_path).suffix.lower().lstrip(".")
        doc_text = load_text(text_dir, doc_id)
        items = load_lineitems(lines_dir, doc_id)

        if items:
            for it in items:
                service_date = (it.get("service_date_guess") or "").strip()
                amount = it.get("amount_guess")
                provider = (it.get("provider_guess") or "").strip()
                service_desc = (it.get("service_desc_guess") or "").strip()
                page = it.get("page")
                confidence = (it.get("confidence") or "").strip()
                tax_year = tax_year_from_date(service_date)

                ws.append([
                    it.get("line_id") or "",
                    doc_id,
                    page if page is not None else "",
                    canonical_path,
                    ext,
                    provider,
                    service_date,
                    amount if amount is not None else "",
                    service_desc,
                    tax_year if tax_year is not None else "",
                    duplicate_of or "",
                    confidence,
                    excerpt(doc_text),
                    "",
                ])
                out_rows += 1
        else:
            # fallback: one row per doc (still useful if parsing didn’t find items)
            ws.append([
                f"{doc_id}:000",
                doc_id,
                "",
                canonical_path,
                ext,
                "",
                "",
                "",
                "",
                "",
                duplicate_of or "",
                "none",
                excerpt(doc_text),
                "",
            ])
            out_rows += 1

    autosize(ws)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)

    print(f"Wrote: {out_path}")
    print(f"Rows:  {out_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
