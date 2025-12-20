from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class ReplaceRule:
    src: str
    dst: str
    whole_word: bool = True
    case_sensitive: bool = False


def load_replacements(path: Path) -> List[ReplaceRule]:
    """
    Load deterministic text replacements.

    Expected JSON:
    {
      "version": 1,
      "replacements": [
        {"from": "Gismo", "to": "Gizmo", "whole_word": true, "case_sensitive": false},
        {"from": "Duran Iran", "to": "Duran Duran", "whole_word": false, "case_sensitive": false}
      ]
    }
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Corrections file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Corrections JSON must be an object: {path}")

    repls = data.get("replacements", [])
    if not isinstance(repls, list):
        raise ValueError(f"'replacements' must be a list in: {path}")

    rules: List[ReplaceRule] = []
    for i, item in enumerate(repls):
        if not isinstance(item, dict):
            raise ValueError(f"Replacement at index {i} must be an object in: {path}")

        src = item.get("from")
        dst = item.get("to")
        if not isinstance(src, str) or not src.strip():
            raise ValueError(f"Replacement at index {i} missing valid 'from' in: {path}")
        if not isinstance(dst, str):
            raise ValueError(f"Replacement at index {i} missing valid 'to' in: {path}")

        whole_word = bool(item.get("whole_word", True))
        case_sensitive = bool(item.get("case_sensitive", False))

        rules.append(ReplaceRule(src=src, dst=dst, whole_word=whole_word, case_sensitive=case_sensitive))

    return rules


def apply_replacements(text: str, rules: Iterable[ReplaceRule]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Apply replacements in order. Returns (new_text, report).

    Report entries:
      {"from": ..., "to": ..., "count": N}
    """
    out = text
    report: List[Dict[str, Any]] = []

    for rule in rules:
        if not rule.src:
            continue

        flags = 0 if rule.case_sensitive else re.IGNORECASE

        if rule.whole_word and _is_simple_word(rule.src):
            pattern = r"\b" + re.escape(rule.src) + r"\b"
        else:
            pattern = re.escape(rule.src)

        out, n = re.subn(pattern, rule.dst, out, flags=flags)
        if n:
            report.append({"from": rule.src, "to": rule.dst, "count": n})

    return out, report


def _is_simple_word(s: str) -> bool:
    """
    True if s looks like a single token (letters/numbers/underscore/hyphen/apostrophe),
    so word-boundary replacement is safe-ish.
    """
    return bool(re.fullmatch(r"[A-Za-z0-9_'\-]+", s))
