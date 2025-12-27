# tools/tax_ingest.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def file_mtime_dt(path: Path) -> datetime:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


_slug_bad = re.compile(r"[^A-Za-z0-9._-]+")


def slugify_filename(name: str, max_len: int = 60) -> str:
    """
    Keep it filesystem-safe and human-legible.
    We preserve dots/underscores/hyphens, collapse other junk to underscore.
    """
    name = name.strip()
    name = _slug_bad.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    if not name:
        name = "receipt"
    if len(name) > max_len:
        name = name[:max_len].rstrip("._-")
    return name


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_image_like(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic", ".pdf"}


def iter_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("*"):
            if p.is_file():
                yield p
    else:
        for p in root.glob("*"):
            if p.is_file():
                yield p


def read_jsonl_hashes(path: Path) -> set[str]:
    """
    Read existing master index and build a set of sha256 hashes.
    Only counts records that have sha256 + status == "ingested".
    """
    hashes: set[str] = set()
    if not path.exists():
        return hashes
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("sha256") and obj.get("status") == "ingested":
                hashes.add(obj["sha256"])
    return hashes


def append_jsonl(path: Path, records: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8", newline="\n")


def rel_to_repo(repo_root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def open_in_explorer(path: Path) -> None:
    """
    Best-effort open folder on Windows/macOS/Linux.
    """
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except Exception:
        pass


# -----------------------------
# Core
# -----------------------------
@dataclass
class IngestPaths:
    repo_root: Path
    inbox_root: Path              # notes/tax/images_in
    current_batch_file: Path      # out/tax/.current_batch
    out_root: Path                # out/tax
    ingested_root: Path           # out/tax/ingested
    batches_root: Path            # out/tax/batches
    index_root: Path              # out/tax/index
    master_index_jsonl: Path      # out/tax/index/receipts.jsonl


def detect_repo_root(start: Path) -> Path:
    """
    Walk upward to find a .git folder.
    """
    cur = start.resolve()
    for _ in range(10):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def load_current_batch(paths: IngestPaths) -> Optional[str]:
    if paths.current_batch_file.exists():
        txt = paths.current_batch_file.read_text(encoding="utf-8", errors="replace").strip()
        if txt:
            return txt
    return None


def pick_latest_batch(inbox_root: Path) -> Optional[str]:
    if not inbox_root.exists():
        return None
    candidates = [p for p in inbox_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name


def make_dest_name(sha: str, src: Path) -> str:
    base = slugify_filename(src.stem)
    ext = src.suffix.lower() if src.suffix else ""
    return f"{sha[:12]}__{base}{ext}"


def safe_copy_or_move(src: Path, dest: Path, move: bool) -> None:
    ensure_dir(dest.parent)
    if move:
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(str(src), str(dest))


def run_ingest(
    paths: IngestPaths,
    batch: str,
    recursive: bool,
    move: bool,
    dry_run: bool,
    include_non_images: bool,
    verbose: bool,
) -> int:
    batch_inbox = paths.inbox_root / batch
    if not batch_inbox.exists():
        print(f"[{now_iso()}] ❌ Batch inbox not found: {batch_inbox}")
        return 2

    batch_out = paths.batches_root / batch
    ensure_dir(batch_out)
    ensure_dir(paths.ingested_root)
    ensure_dir(paths.index_root)

    master_hashes = read_jsonl_hashes(paths.master_index_jsonl)

    # Collect files
    all_files = list(iter_files(batch_inbox, recursive=recursive))
    if not include_non_images:
        all_files = [p for p in all_files if is_image_like(p)]

    total = len(all_files)
    if total == 0:
        write_text(batch_out / "REPORT.md", f"# Tax Batch {batch}\n\nNo files found.\n")
        print(f"[{now_iso()}] ℹ️ No files to ingest in: {batch_inbox}")
        return 0

    ingested_records: list[dict] = []
    batch_manifest_records: list[dict] = []

    ingested_count = 0
    dup_count = 0
    err_count = 0

    print(f"[{now_iso()}] ▶ Tax ingest")
    print(f"[{now_iso()}]    repo:   {paths.repo_root}")
    print(f"[{now_iso()}]    batch:  {batch}")
    print(f"[{now_iso()}]    inbox:  {batch_inbox}")
    print(f"[{now_iso()}]    out:    {paths.ingested_root}")
    print(f"[{now_iso()}]    index:  {paths.master_index_jsonl}")
    print(f"[{now_iso()}]    mode:   {'MOVE' if move else 'COPY'}{' (dry-run)' if dry_run else ''}")
    print(f"[{now_iso()}]    files:  {total}")

    for i, src in enumerate(all_files, start=1):
        try:
            mdt = file_mtime_dt(src)
            yyyy = mdt.strftime("%Y")
            mm = mdt.strftime("%m")
            dest_dir = paths.ingested_root / yyyy / mm

            sha = sha256_file(src)
            original_name = src.name

            # Duplicate?
            if sha in master_hashes:
                dup_count += 1
                rec = {
                    "kind": "receipt_asset",
                    "status": "duplicate",
                    "batch": batch,
                    "sha256": sha,
                    "original_name": original_name,
                    "src_rel": rel_to_repo(paths.repo_root, src),
                    "src_bytes": src.stat().st_size,
                    "src_mtime": to_iso(mdt),
                    "ingested_at": now_iso(),
                }
                batch_manifest_records.append(rec)
                if verbose:
                    print(f"[{now_iso()}]    ({i}/{total}) ↩ duplicate: {src}")
                continue

            dest_name = make_dest_name(sha, src)
            dest = dest_dir / dest_name

            # Avoid ultra-rare collision on name (same 12 prefix + same base)
            if dest.exists():
                # If file exists but hash differs (unlikely), add counter
                counter = 2
                while dest.exists():
                    dest = dest_dir / f"{dest.stem}__{counter}{dest.suffix}"
                    counter += 1

            if verbose:
                print(f"[{now_iso()}]    ({i}/{total}) → {dest}")

            if not dry_run:
                safe_copy_or_move(src, dest, move=move)

            master_hashes.add(sha)
            ingested_count += 1

            rec_master = {
                "kind": "receipt_asset",
                "status": "ingested",
                "batch": batch,
                "sha256": sha,
                "original_name": original_name,
                "src_rel": rel_to_repo(paths.repo_root, src),
                "dest_rel": rel_to_repo(paths.repo_root, dest),
                "bytes": dest.stat().st_size if dest.exists() else src.stat().st_size,
                "mtime": to_iso(mdt),
                "ingested_at": now_iso(),
            }
            rec_batch = dict(rec_master)

            ingested_records.append(rec_master)
            batch_manifest_records.append(rec_batch)

        except Exception as e:
            err_count += 1
            rec = {
                "kind": "receipt_asset",
                "status": "error",
                "batch": batch,
                "original_name": getattr(src, "name", "unknown"),
                "src_rel": rel_to_repo(paths.repo_root, src) if isinstance(src, Path) else str(src),
                "error": repr(e),
                "ingested_at": now_iso(),
            }
            batch_manifest_records.append(rec)
            print(f"[{now_iso()}] ❌ Error ingesting {src}: {e}")

    # Write outputs
    batch_manifest_path = batch_out / "manifest.jsonl"
    report_path = batch_out / "REPORT.md"

    if not dry_run:
        if ingested_records:
            append_jsonl(paths.master_index_jsonl, ingested_records)
        append_jsonl(batch_manifest_path, batch_manifest_records)
    else:
        # still write report in dry-run so you can see summary
        append_jsonl(batch_manifest_path, batch_manifest_records)

    # Human report
    lines = []
    lines.append(f"# Tax Batch {batch}")
    lines.append("")
    lines.append(f"- Started: {now_iso()}")
    lines.append(f"- Inbox: `{rel_to_repo(paths.repo_root, batch_inbox)}`")
    lines.append(f"- Output root: `{rel_to_repo(paths.repo_root, paths.ingested_root)}`")
    lines.append(f"- Master index: `{rel_to_repo(paths.repo_root, paths.master_index_jsonl)}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Files scanned: **{total}**")
    lines.append(f"- Ingested: **{ingested_count}**")
    lines.append(f"- Duplicates: **{dup_count}**")
    lines.append(f"- Errors: **{err_count}**")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Output filenames are hash-based so it doesn’t matter if your screenshots have junk names.")
    lines.append("- Master index only stores unique ingests; batch manifest logs everything (including duplicates).")
    lines.append("")

    write_text(report_path, "\n".join(lines) + "\n")

    print(f"[{now_iso()}] ✅ Done")
    print(f"[{now_iso()}]    ingested={ingested_count} duplicates={dup_count} errors={err_count}")
    print(f"[{now_iso()}]    batch manifest: {batch_manifest_path}")
    print(f"[{now_iso()}]    report:         {report_path}")

    return 0 if err_count == 0 else 1


def build_default_paths() -> IngestPaths:
    repo_root = detect_repo_root(Path(__file__).parent)
    inbox_root = repo_root / "notes" / "tax" / "images_in"
    out_root = repo_root / "out" / "tax"
    current_batch_file = out_root / ".current_batch"
    ingested_root = out_root / "ingested"
    batches_root = out_root / "batches"
    index_root = out_root / "index"
    master_index_jsonl = index_root / "receipts.jsonl"
    return IngestPaths(
        repo_root=repo_root,
        inbox_root=inbox_root,
        current_batch_file=current_batch_file,
        out_root=out_root,
        ingested_root=ingested_root,
        batches_root=batches_root,
        index_root=index_root,
        master_index_jsonl=master_index_jsonl,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Tax receipt image ingest: batch inbox → hashed archive + indexes",
    )
    p.add_argument("--batch", help="Batch id (defaults to out/tax/.current_batch; else newest inbox folder)")
    p.add_argument("--recursive", action="store_true", help="Scan inbox recursively (default: on)")
    p.add_argument("--no-recursive", dest="recursive", action="store_false", help="Scan only top-level files")
    p.set_defaults(recursive=True)

    p.add_argument("--move", action="store_true", help="Move files from inbox (default: copy)")
    p.add_argument("--dry-run", action="store_true", help="Do everything except file copy/move")
    p.add_argument("--include-non-images", action="store_true", help="Also ingest non-image files")

    p.add_argument("--verbose", action="store_true", help="Verbose per-file output")
    p.add_argument("--open", action="store_true", help="Open batch output folder when finished")

    args = p.parse_args(argv)

    paths = build_default_paths()

    batch = args.batch or load_current_batch(paths) or pick_latest_batch(paths.inbox_root)
    if not batch:
        print(f"[{now_iso()}] ❌ No batch found.")
        print(f"[{now_iso()}]    Create one under: {paths.inbox_root}")
        print(f"[{now_iso()}]    Or set: {paths.current_batch_file}")
        return 2

    code = run_ingest(
        paths=paths,
        batch=batch,
        recursive=args.recursive,
        move=args.move,
        dry_run=args.dry_run,
        include_non_images=args.include_non_images,
        verbose=args.verbose,
    )

    if args.open:
        open_in_explorer(paths.batches_root / batch)

    return code


if __name__ == "__main__":
    raise SystemExit(main())
