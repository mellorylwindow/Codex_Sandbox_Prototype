from __future__ import annotations

import re
from typing import Literal

CleanMode = Literal["light", "standard"]


def _normalize_whitespace(text: str) -> str:
    # Normalize line endings and strip trailing whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text


def _repair_wrapped_lines(text: str) -> str:
    """
    Whisper outputs often have hard-wrapped lines.
    This merges lines that look like they are the same paragraph/sentence.

    Heuristic:
      - if a line does NOT end with sentence punctuation and next line starts lowercase,
        join them with a space.
      - if a line ends with a comma and next line starts lowercase, join them.
    """
    lines = [ln.strip() for ln in text.split("\n")]
    out = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line:
            out.append("")
            i += 1
            continue

        # Try to merge with next lines while it looks like wrapping
        while i + 1 < len(lines):
            nxt = lines[i + 1]
            if not nxt:
                break

            ends = line[-1]
            nxt_starts_lower = bool(re.match(r"^[a-z]", nxt))
            nxt_starts_quote_lower = bool(re.match(r"^[\"'“”‘’]?[a-z]", nxt))

            should_join = (
                (ends not in ".!?:" and nxt_starts_quote_lower)
                or (ends == "," and nxt_starts_quote_lower)
            )

            if not should_join:
                break

            line = f"{line} {nxt}"
            i += 1

        out.append(line)
        i += 1

    # Collapse multiple blank lines to at most 2
    merged = "\n".join(out)
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged.strip() + "\n"


def _fix_spacing(text: str, mode: CleanMode) -> str:
    # Fix common spacing around punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)         # space before punctuation
    text = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", text)  # missing space after punctuation
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    # Light quote normalization (optional)
    if mode == "standard":
        # Replace smart quotes with plain quotes
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Collapse repeated spaces (but preserve newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text


def clean_text(text: str, *, mode: CleanMode = "standard") -> str:
    text = _normalize_whitespace(text)
    text = _repair_wrapped_lines(text)
    text = _fix_spacing(text, mode=mode)
    return text
