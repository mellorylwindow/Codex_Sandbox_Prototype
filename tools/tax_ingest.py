from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DocRecord:
    doc_id: str                 # sha256 hex
    original_path: str          # where it was found
    canonical_path: str         # where canonical lives (or would live)
    size_bytes: int
    mtime_iso: str
    ingested_at_iso: str
    duplicate_of: str | None    # doc_id of canonical if this is a dup
    mode: str                   # copy/move/hardlink
    status: str                 # canonical/duplicate/error/dry_run


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_mtime_iso(p: Path) -> str:
    ts = p.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def sha256_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_name(name: str) -> str:
    # Keep it filesystem-safe and stable.
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            keep.append(ch)
        else:
            keep.append("_")
    s = "".join(keep).strip().replace("  ", " ")
    return s[:180] if len(s) > 180 else s


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def do_transfer(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    elif mode == "hardlink":
        # Hardlink keeps one physical copy; may fail across volumes.
        ensure_dir(dst.parent)
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def unique_path(base: Path) -> Path:
    """If base exists, append __NNN before suffix."""
    if not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    for i in range(1, 10_000):
        cand = base.with_name(f"{stem}__{i:03d}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find unique filename for {base}")


def write_jsonl(path: Path, rows: list[DocRecord]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_dupes_csv(path: Path, dup_rows: list[DocRecord]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "duplicate_path", "canonical_path", "duplicate_of"])
        for r in dup_rows:
            w.writerow([r.doc_id, r.original_path, r.canonical_path, r.duplicate_of or ""])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tax intake ingest: dedupe by sha256, write manifest.jsonl, keep one canonical copy."
    )
    ap.add_argument("--inbox", default="tax_intake/00_inbox", help="Drop files here.")
    ap.add_argument("--canonical", default="tax_intake/10_raw_canonical", help="One-true-copy store.")
    ap.add_argument("--duplicates", default="tax_intake/20_duplicates", help="Where duplicate files go (optional).")
    ap.add_argument("--index", default="tax_intake/index", help="Where manifest/dupe maps go.")
    ap.add_argument("--mode", choices=["copy", "move", "hardlink"], default="copy",
                    help="How to place files into canonical/duplicates.")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="If set, duplicates are also copied/moved into duplicates folder. Otherwise we only record them.")
    ap.add_argument("--dry-run", action="store_true", help="No filesystem writes; still computes hashes.")
    args = ap.parse_args()

    inbox = Path(args.inbox)
    canonical_root = Path(args.canonical)
    dup_root = Path(args.duplicates)
    index_root = Path(args.index)

    if not inbox.exists():
        print(f"ERROR: inbox not found: {inbox}", file=sys.stderr)
        return 2

    ensure_dir(canonical_root)
    ensure_dir(index_root)

    ingested_at = utc_now_iso()

    seen: dict[str, Path] = {}  # sha -> canonical path
    records: list[DocRecord] = []
    dup_records: list[DocRecord] = []

    files = list(iter_files(inbox))
    if not files:
        print(f"No files found under: {inbox}")
        return 0

    for src in files:
        try:
            digest = sha256_file(src)
            ext = src.suffix.lower() or ""
            # Canonical is content-addressed.
            canonical_path = canonical_root / f"{digest}{ext}"

            size = src.stat().st_size
            mtime = file_mtime_iso(src)

            if digest not in seen and not canonical_path.exists():
                # This is the canonical copy.
                do_transfer(src, canonical_path, args.mode, args.dry_run)
                status = "dry_run" if args.dry_run else "canonical"
                rec = DocRecord(
                    doc_id=digest,
                    original_path=str(src),
                    canonical_path=str(canonical_path),
                    size_bytes=size,
                    mtime_iso=mtime,
                    ingested_at_iso=ingested_at,
                    duplicate_of=None,
                    mode=args.mode,
                    status=status,
                )
                seen[digest] = canonical_path
                records.append(rec)

                # If mode=move, src path no longer exists; that's OK.
            else:
                # Duplicate of existing canonical (either seen in this run, or already on disk).
                dup_of = digest
                status = "dry_run" if args.dry_run else "duplicate"
                rec = DocRecord(
                    doc_id=digest,
                    original_path=str(src),
                    canonical_path=str(canonical_path),
                    size_bytes=size,
                    mtime_iso=mtime,
                    ingested_at_iso=ingested_at,
                    duplicate_of=dup_of,
                    mode=args.mode,
                    status=status,
                )
                records.append(rec)
                dup_records.append(rec)

                if args.keep_duplicates:
                    # Store a copy of the duplicate file (optional).
                    safe_name = sanitize_name(src.name)
                    dup_path = dup_root / f"{digest}__{safe_name}"
                    dup_path = unique_path(dup_path)
                    do_transfer(src, dup_path, args.mode, args.dry_run)

        except Exception as e:
            rec = DocRecord(
                doc_id="",
                original_path=str(src),
                canonical_path="",
                size_bytes=0,
                mtime_iso="",
                ingested_at_iso=ingested_at,
                duplicate_of=None,
                mode=args.mode,
                status=f"error: {e}",
            )
            records.append(rec)

    manifest_path = index_root / "manifest.jsonl"
    dupes_path = index_root / "duplicates_map.csv"

    write_jsonl(manifest_path, records)
    write_dupes_csv(dupes_path, dup_records)

    canon_count = sum(1 for r in records if r.status in ("canonical", "dry_run") and r.duplicate_of is None)
    dup_count = len(dup_records)

    print(f"Processed: {len(records)} files")
    print(f"Canonicals: {canon_count}")
    print(f"Duplicates: {dup_count}")
    print(f"Manifest:   {manifest_path}")
    print(f"Dupe map:   {dupes_path}")
    if args.dry_run:
        print("DRY RUN: no files were copied/moved/linked.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
