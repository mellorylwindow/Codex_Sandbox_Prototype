from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=REPO_ROOT)
    return p.returncode

def main() -> int:
    ap = argparse.ArgumentParser(description="Run full tax pipeline: ingest -> extract -> parse line-items -> export.")
    ap.add_argument("--dry-run", action="store_true", help="Dry-run ingest (no file writes).")
    ap.add_argument("--mode", choices=["copy", "move", "hardlink"], default="copy",
                    help="Ingest mode for canonical storage.")
    ap.add_argument("--force-extract", action="store_true", help="Re-extract text even if outputs exist.")
    ap.add_argument("--force-lines", action="store_true", help="Rebuild line-items even if outputs exist.")
    ap.add_argument("--keep-duplicates", action="store_true", help="Copy/move duplicates into 20_duplicates.")
    args = ap.parse_args()

    py = sys.executable

    ingest = [py, "tools/tax_ingest.py", "--mode", args.mode]
    if args.dry_run:
        ingest.append("--dry-run")
    if args.keep_duplicates:
        ingest.append("--keep-duplicates")

    extract = [py, "tools/tax_extract_text.py", "--only-canonical"]
    if args.force_extract:
        extract.append("--force")

    parse_lines = [py, "tools/tax_parse_lineitems.py", "--only-canonical"]
    if args.force_lines:
        parse_lines.append("--force")

    export = [py, "tools/tax_export_xlsx.py", "--only-canonical"]

    for step in (ingest, extract, parse_lines, export):
        rc = run(step)
        if rc != 0:
            print(f"\nERROR: step failed (exit {rc})")
            return rc

    print("\n✅ Done.")
    print("- Spreadsheet: tax_intake/40_reports/tax_lines.xlsx")
    print("- Manifest:    tax_intake/index/manifest.jsonl")
    print("- Dupe map:    tax_intake/index/duplicates_map.csv")
    print("- Line-items:  tax_intake/30_extracted/lines/<doc_id>.lines.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
