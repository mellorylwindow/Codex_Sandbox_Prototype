from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Ensure repo root is importable when running: python tools/textlab_cli.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from textlab.pipeline import run_textlab


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="textlab", description="Offline-first transcript processing pipeline.")
    p.add_argument("input_txt", help="Path to a transcript .txt (usually from out/transcripts/).")
    p.add_argument("--out", dest="out_root", type=Path, default=Path("out/textlab"), help="Output root directory.")
    p.add_argument("--name", type=str, default=None, help="Base name for the run folder.")

    # Prep
    p.add_argument("--no-clean", action="store_true", help="Disable clean.")
    p.add_argument("--redact", nargs="?", const="standard", default="standard", help="Redaction preset (default: standard). Use --redact '' to disable.")
    p.add_argument("--terms", type=Path, default=Path("notes/redaction_terms.json"), help="Redaction terms JSON.")
    p.add_argument("--no-fix", action="store_true", help="Disable post-fix corrections.")
    p.add_argument("--fix-file", type=Path, default=Path("notes/corrections.json"), help="Corrections JSON file.")
    p.add_argument("--report", action="store_true", help="Write redaction report (if supported by textops).")

    # Chunking
    p.add_argument("--chunk-minutes", type=int, default=5, help="Chunk by N-minute windows when timestamps exist.")
    p.add_argument("--chunk-chars", type=int, default=4000, help="Fallback chunk size by characters.")

    # Analysis
    p.add_argument("--no-keywords", action="store_true")
    p.add_argument("--no-entities", action="store_true")
    p.add_argument("--no-summary", action="store_true")

    return p


def main() -> int:
    args = _build_parser().parse_args()

    input_txt = Path(args.input_txt)

    redact = args.redact
    if redact is not None and redact.strip() == "":
        redact = None

    work_dir = run_textlab(
        input_txt=input_txt,
        out_root=args.out_root,
        name=args.name,
        clean=not args.no_clean,
        redact=redact,
        terms=args.terms if args.terms and args.terms.exists() else None,
        report=args.report,
        post_fix=(None if args.no_fix else args.fix_file),
        chunk_minutes=args.chunk_minutes if args.chunk_minutes > 0 else None,
        chunk_chars=args.chunk_chars,
        write_keywords=not args.no_keywords,
        write_entities=not args.no_entities,
        write_summary=not args.no_summary,
    )

    print(f"✅ textlab run created: {work_dir}")
    print(f"   - manifest: {work_dir / 'manifest.json'}")
    print(f"   - chunks:   {work_dir / 'chunks'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
