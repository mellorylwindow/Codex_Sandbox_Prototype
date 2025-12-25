#!/usr/bin/env python3
"""
Build a recording-ready teleprompter script from a queue markdown.

Modes:
- PICK mode: headings like "## PICK 1. ..." (optional)
- TOP mode: take the first N items "## 1. ..." automatically (default)

Outputs:
- <queue>_teleprompter.md (or --out)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

RE_PICK = re.compile(r"^##\s+PICK\s+(\d+)\.\s*(.*)$")
RE_NUM = re.compile(r"^##\s+(\d+)\.\s*(.*)$")

RE_TITLE = re.compile(r"^- Title:\s+(.*)$")
RE_PREVIEW = re.compile(r"^- Preview:\s+(.*)$")


def parse_queue(md_text: str) -> List[Dict[str, str]]:
    """
    Parse queue.md sections like:
      ## 1. <id> — score ...
      - Title: ...
      - Preview: ...
    """
    items: List[Dict[str, str]] = []
    cur: Optional[Dict[str, str]] = None

    for raw in md_text.splitlines():
        line = raw.strip()

        m_pick = RE_PICK.match(line)
        m_num = RE_NUM.match(line)

        if m_pick:
            if cur:
                items.append(cur)
            cur = {"pick": m_pick.group(1), "hdr": m_pick.group(2).strip(), "title": "", "preview": ""}
            continue

        if m_num:
            if cur:
                items.append(cur)
            cur = {"pick": "", "hdr": m_num.group(2).strip(), "title": "", "preview": ""}
            continue

        if not cur:
            continue

        mt = RE_TITLE.match(line)
        if mt and not cur["title"]:
            cur["title"] = mt.group(1).strip()
            continue

        mp = RE_PREVIEW.match(line)
        if mp and not cur["preview"]:
            cur["preview"] = mp.group(1).strip()
            continue

    if cur:
        items.append(cur)

    return items


def render(items: List[Dict[str, str]], title: str) -> str:
    out: List[str] = []
    out.append("# TikTok Teleprompter — Recording Draft")
    out.append("")
    out.append(f"**Queue:** {title}")
    out.append("")
    out.append("## Format")
    out.append("- Hook (1 sentence)")
    out.append("- Core (3 bullets)")
    out.append("- Example (1 line)")
    out.append("- Close + CTA (1 line)")
    out.append("")

    for i, it in enumerate(items, start=1):
        label = f"PICK {it['pick']}" if it.get("pick") else f"TOP {i}"
        headline = it.get("title") or it.get("hdr") or "(untitled)"
        preview = it.get("preview") or "(add your one-sentence core here)"

        out.append("---")
        out.append("")
        out.append(f"## {label}: {headline}")
        out.append("")
        out.append("**Hook:**")
        out.append("> ")
        out.append("")
        out.append("**Core:**")
        out.append(f"- {preview}")
        out.append("- Why it matters (benefit #1).")
        out.append("- Why it’s easy (benefit #2).")
        out.append("")
        out.append("**Example:**")
        out.append("> ")
        out.append("")
        out.append("**Close + CTA:**")
        out.append("> Comment ‘VELVET’ and I’ll drop the template.")
        out.append("")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-md", required=True, help="Queue markdown path")
    ap.add_argument("--out", default=None, help="Output markdown path (default: <in>_teleprompter.md)")
    ap.add_argument("--top", type=int, default=3, help="If no PICKs exist, take first N numbered items")
    args = ap.parse_args()

    in_path = Path(args.in_md)
    if not in_path.exists():
        print(f"ERROR: not found: {in_path}")
        return 2

    text = in_path.read_text(encoding="utf-8", errors="replace")
    items = parse_queue(text)
    if not items:
        print("No items found in queue.md")
        return 2

    # Prefer PICKs if present
    picks = [it for it in items if it.get("pick")]
    chosen = picks if picks else items[: max(1, int(args.top))]

    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_teleprompter.md")
    out_path.write_text(render(chosen, title=in_path.name), encoding="utf-8", newline="\n")

    print(f"OK: wrote teleprompter ({len(chosen)} items)")
    print(f"- {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
