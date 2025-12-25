#!/usr/bin/env python3
"""
TikTok Prompt Project — Prompt Compiler (offline-first)

Reads:
- out/tiktok_prompts/prompts.jsonl

Writes:
- out/tiktok_prompts/drafts.jsonl
- out/tiktok_prompts/drafts.md

Goal:
Turn messy OCR text (notes/screenshot text) into structured TikTok drafts.
No AI calls. Deterministic, reviewable, and easy to iterate.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


# ----------------------------
# Models
# ----------------------------

@dataclass(frozen=True)
class Draft:
    id: str
    source_name: str
    source_path: str
    extracted_text: str

    # derived
    prompt_core: str
    title: str
    hook: str
    on_screen: str
    voiceover: str
    beats: list[str]
    cta: str
    tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "extracted_text": self.extracted_text,
            "prompt_core": self.prompt_core,
            "title": self.title,
            "hook": self.hook,
            "on_screen": self.on_screen,
            "voiceover": self.voiceover,
            "beats": self.beats,
            "cta": self.cta,
            "tags": self.tags,
        }


# ----------------------------
# IO
# ----------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_md(path: Path, drafts: list[Draft]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# TikTok Prompt Drafts")
    lines.append("")
    lines.append(f"Total: {len(drafts)}")
    lines.append("")

    for d in drafts:
        lines.append(f"## {d.id} — {d.title}")
        lines.append(f"- Source: `{d.source_name}`")
        lines.append(f"- Path: `{d.source_path}`")
        lines.append("")
        lines.append("**Prompt core**")
        lines.append(f"> {d.prompt_core if d.prompt_core else '(none detected)'}")
        lines.append("")
        lines.append("**Hook**")
        lines.append(f"> {d.hook}")
        lines.append("")
        lines.append("**On-screen text**")
        lines.append(f"> {d.on_screen}")
        lines.append("")
        lines.append("**Voiceover**")
        lines.append(d.voiceover)
        lines.append("")
        lines.append("**Beats**")
        for b in d.beats:
            lines.append(f"- {b}")
        lines.append("")
        lines.append("**CTA**")
        lines.append(f"> {d.cta}")
        lines.append("")
        lines.append("**Tags**")
        lines.append(", ".join(d.tags))
        lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


# ----------------------------
# Text cleanup + extraction
# ----------------------------

_JUNK_LINES = {
    "ee ae",
    "ee",
    "ae",
}

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse repeated spaces, keep newlines
    s = re.sub(r"[ \t]+", " ", s)
    # Trim each line; drop empty noise lines
    out_lines: list[str] = []
    for line in s.split("\n"):
        t = line.strip()
        if not t:
            continue
        low = t.lower()
        if low in _JUNK_LINES:
            continue
        # strip bullet-ish OCR artifacts
        t = re.sub(r"^[\-\*\u2022]+[ ]*", "", t)
        out_lines.append(t)
    return "\n".join(out_lines).strip()


def extract_prompt_core(text: str) -> str:
    """
    Try to extract the user's intended "Prompt — ..." payload.
    Falls back to the most 'prompt-like' chunk.
    """
    t = text

    # Strong pattern: Prompt — “...”
    m = re.search(r"(?i)\bprompt\b\s*[—:\-]\s*[\"“](.+?)[\"”]\s*$", t, flags=re.DOTALL)
    if m:
        core = m.group(1).strip()
        return re.sub(r"\s+", " ", core)

    # Another pattern: Prompt — You are my ...
    m2 = re.search(r"(?i)\bprompt\b\s*[—:\-]\s*(.+)$", t, flags=re.DOTALL)
    if m2:
        core = m2.group(1).strip()
        core = core.strip("“”\"' ")
        return re.sub(r"\s+", " ", core)

    # Fallback: use the longest line or paragraph
    parts = re.split(r"\n{2,}", t)
    if parts:
        best = max(parts, key=lambda x: len(x))
        best = best.strip().strip("“”\"' ")
        return re.sub(r"\s+", " ", best)

    return ""


def title_from_core(core: str, fallback: str) -> str:
    if core:
        # First 7–10 words
        words = core.split()
        return " ".join(words[:9]) + ("…" if len(words) > 9 else "")
    # fallback to filename-ish
    return fallback


def make_draft_from_core(core: str) -> tuple[str, str, str, list[str], str, list[str]]:
    """
    Deterministic templating: same core => same format.

    Returns: hook, on_screen, voiceover, beats, cta, tags
    """
    if not core:
        hook = "Found a note. Turning it into a usable TikTok prompt."
        on_screen = "Screenshot → Prompt → Post"
        voiceover = (
            "I found this in my notes and I’m converting it into a usable prompt.\n"
            "If you want me to turn your screenshots into clean prompts, drop one in the comments."
        )
        beats = [
            "Show the screenshot for 1 second (blur anything personal).",
            "Highlight the key line with a quick zoom.",
            "Cut to the cleaned prompt on screen.",
            "End with a simple call-to-action.",
        ]
        cta = "Comment “PROMPT” and I’ll format the next one."
        tags = ["#productivity", "#notes", "#contentcreator", "#promptengineering"]
        return hook, on_screen, voiceover, beats, cta, tags

    hook = "If you feel behind, use this one prompt to rebuild your week."
    on_screen = "One prompt → a 7-day reset plan"
    voiceover = (
        "Here’s a prompt I use when my life feels messy but I still want to look effortless.\n\n"
        f"Prompt:\n“{core}”\n\n"
        "Paste it into your planner or AI, then tweak the schedule to match your real life."
    )
    beats = [
        "Open with the hook + your face or text-only.",
        "Flash the original screenshot for authenticity.",
        "Show the cleaned prompt on screen (big font).",
        "Show a 7-day layout (Mon–Sun) with 3 lanes: work / home / rest.",
        "End on the CTA.",
    ]
    cta = "Want me to turn your messy note into a clean prompt? Comment “CLEAN”."
    tags = ["#productivity", "#adhd", "#planner", "#lifeadmin", "#tiktokprompts"]
    return hook, on_screen, voiceover, beats, cta, tags


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compile OCR’d image notes into TikTok prompt drafts.")
    ap.add_argument("--in", dest="in_path", default="out/tiktok_prompts/prompts.jsonl", help="Input prompts.jsonl")
    ap.add_argument("--outdir", dest="out_dir", default="out/tiktok_prompts", help="Output directory")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        return 2

    rows = read_jsonl(in_path)

    drafts: list[Draft] = []
    for r in rows:
        rid = str(r.get("id", "")).strip()
        extracted = normalize_text(str(r.get("extracted_text", "") or ""))
        core = extract_prompt_core(extracted)

        title = title_from_core(core, fallback=str(r.get("source_name", rid)))
        hook, on_screen, voiceover, beats, cta, tags = make_draft_from_core(core)

        drafts.append(
            Draft(
                id=rid,
                source_name=str(r.get("source_name", "")),
                source_path=str(r.get("source_path", "")),
                extracted_text=extracted,
                prompt_core=core,
                title=title,
                hook=hook,
                on_screen=on_screen,
                voiceover=voiceover,
                beats=beats,
                cta=cta,
                tags=tags,
            )
        )

    # Deterministic order
    drafts_sorted = sorted(drafts, key=lambda d: d.id)

    drafts_jsonl = out_dir / "drafts.jsonl"
    drafts_md = out_dir / "drafts.md"

    write_jsonl(drafts_jsonl, (d.to_dict() for d in drafts_sorted))
    write_md(drafts_md, drafts_sorted)

    print(f"OK: wrote {len(drafts_sorted)} drafts")
    print(f"- {drafts_jsonl}")
    print(f"- {drafts_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
