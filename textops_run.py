from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path
from typing import List, Optional

from textops import clean_text, redact_text


def _expand_inputs(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        # Support glob patterns like out/transcripts/demo__*.txt
        matches = glob.glob(item)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(item))
    # De-dupe while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        rp = str(p)
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq


def _write_out(out_dir: Path, src: Path, suffix: str, text: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}.{suffix}{src.suffix}"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="Offline transcript transforms: clean, redact (no AI).")
    p.add_argument("inputs", nargs="+", help="Input .txt path(s) or glob(s)")
    p.add_argument("--out", type=Path, default=Path("out/textops"), help="Output directory")
    p.add_argument("--clean", action="store_true", help="Apply offline cleanup (linewrap repair + spacing)")
    p.add_argument("--clean-mode", choices=["light", "standard"], default="standard")

    p.add_argument("--redact", choices=["light", "standard", "heavy"], default=None, help="Apply offline redaction")
    p.add_argument("--redact-name", action="append", default=[], help="Exact name to redact (repeatable)")
    p.add_argument("--redact-org", action="append", default=[], help="Exact org to redact (repeatable)")
    p.add_argument("--redact-location", action="append", default=[], help="Exact location to redact (repeatable)")

    p.add_argument("--report", action="store_true", help="Write .redactions.json report next to output")
    args = p.parse_args()

    in_paths = _expand_inputs(args.inputs)

    for src in in_paths:
        src = src.expanduser()
        if not src.exists():
            raise FileNotFoundError(f"Input not found: {src}")
        if not src.is_file():
            raise ValueError(f"Input must be a file: {src}")

        text = src.read_text(encoding="utf-8", errors="replace")

        applied_suffixes: List[str] = []
        redaction_hits = None

        if args.clean:
            text = clean_text(text, mode=args.clean_mode)
            applied_suffixes.append(f"clean-{args.clean_mode}")

        if args.redact:
            text, hits = redact_text(
                text,
                level=args.redact,
                redact_names=args.redact_name,
                redact_orgs=args.redact_org,
                redact_locations=args.redact_location,
            )
            redaction_hits = hits
            applied_suffixes.append(f"redact-{args.redact}")

        if not applied_suffixes:
            # Default: do nothing but write a copy (keeps CLI predictable)
            applied_suffixes.append("copy")

        suffix = "__".join(applied_suffixes)
        out_path = _write_out(args.out, src, suffix, text)
        print(f"✅ Wrote: {out_path}")

        if args.report and redaction_hits is not None:
            report_path = out_path.with_suffix(out_path.suffix + ".redactions.json")
            payload = [
                {"label": h.label, "match": h.match, "start": h.start, "end": h.end}
                for h in redaction_hits
            ]
            report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"🧾 Report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
