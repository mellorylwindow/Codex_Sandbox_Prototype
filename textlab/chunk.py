from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


_TS_RE = re.compile(r"^\[(\d{2}):(\d{2})\]\s*(.*)$")


@dataclass(frozen=True)
class Chunk:
    index: int
    start_ts: Optional[str]
    end_ts: Optional[str]
    text: str


def _parse_ts_line(line: str) -> Optional[Tuple[int, str, str]]:
    """
    Parse a transcript line like:
      [MM:SS] blah blah
    Returns:
      (seconds, "MM:SS", content)
    """
    m = _TS_RE.match(line.strip())
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    content = m.group(3).strip()
    ts = f"{mm:02d}:{ss:02d}"
    return (mm * 60 + ss, ts, content)


def detect_timestamped(lines: Iterable[str]) -> bool:
    for line in lines:
        if _parse_ts_line(line) is not None:
            return True
    return False


def chunk_by_minutes(text: str, minutes: int) -> List[Chunk]:
    """
    Chunk timestamped transcripts into N-minute windows.
    If no timestamp lines exist, falls back to chunk_by_chars.
    """
    lines = text.splitlines()
    if not detect_timestamped(lines):
        return chunk_by_chars(text, max_chars=4000)

    window_s = max(1, minutes) * 60

    chunks: List[Chunk] = []
    buf: List[str] = []
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None
    window_start_s: Optional[int] = None

    idx = 0
    for raw in lines:
        parsed = _parse_ts_line(raw)
        if parsed is None:
            # keep non-timestamp lines as-is
            if buf:
                buf.append(raw.rstrip())
            else:
                # ignore leading stray lines (common in mixed outputs)
                if raw.strip():
                    buf.append(raw.rstrip())
            continue

        sec, ts, content = parsed
        if window_start_s is None:
            window_start_s = sec
            start_ts = ts

        # If we crossed into the next window, flush.
        if (sec - window_start_s) >= window_s and buf:
            chunks.append(
                Chunk(
                    index=idx,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    text="\n".join(buf).strip(),
                )
            )
            idx += 1
            buf = []
            window_start_s = sec
            start_ts = ts
            end_ts = None

        end_ts = ts
        if content:
            buf.append(f"[{ts}] {content}")
        else:
            buf.append(f"[{ts}]")

    if buf:
        chunks.append(
            Chunk(
                index=idx,
                start_ts=start_ts,
                end_ts=end_ts,
                text="\n".join(buf).strip(),
            )
        )

    return chunks


def chunk_by_chars(text: str, max_chars: int = 4000) -> List[Chunk]:
    """
    Generic chunking when timestamps are absent.
    Splits on paragraph boundaries; ensures chunks stay <= max_chars.
    """
    max_chars = max(500, int(max_chars))
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[Chunk] = []
    buf: List[str] = []
    size = 0
    idx = 0

    for p in paras:
        # +2 for paragraph break spacing
        add_len = len(p) + (2 if buf else 0)
        if buf and (size + add_len) > max_chars:
            chunks.append(Chunk(index=idx, start_ts=None, end_ts=None, text="\n\n".join(buf).strip()))
            idx += 1
            buf = []
            size = 0

        buf.append(p)
        size += add_len

    if buf:
        chunks.append(Chunk(index=idx, start_ts=None, end_ts=None, text="\n\n".join(buf).strip()))

    return chunks


def write_chunks(out_dir: Path, base: str, chunks: List[Chunk]) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for c in chunks:
        tag = f"{c.index:03d}"
        p = out_dir / f"{base}.chunk_{tag}.txt"
        header = []
        if c.start_ts or c.end_ts:
            header.append(f"# chunk {c.index}  [{c.start_ts or '--:--'} → {c.end_ts or '--:--'}]")
            header.append("")
        p.write_text("\n".join(header) + (c.text.strip() + "\n"), encoding="utf-8")
        paths.append(p)
    return paths
