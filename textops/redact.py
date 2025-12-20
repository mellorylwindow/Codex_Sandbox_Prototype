# textops/redact.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

RedactLevel = Literal["light", "standard", "heavy"]


@dataclass(frozen=True)
class RedactionHit:
    label: str
    match: str
    start: int
    end: int


@dataclass(frozen=True)
class CustomPattern:
    """
    User-supplied regex redaction rule (offline).

    Fields:
      - label (required): used for audit + default replacement token
      - regex (required): regex pattern string
      - replace (optional): replacement text. If omitted, defaults to f"[{label}]"
      - min_level (optional): only apply when requested level >= min_level
      - flags (optional): compiled re flags (IGNORECASE/MULTILINE/DOTALL)
    """
    label: str
    regex: str
    replace: Optional[str] = None
    min_level: Optional[RedactLevel] = None
    flags: int = 0


# -----------------------------
# Built-in patterns (offline)
# -----------------------------

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# US-ish phone patterns: (555) 555-5555, 555-555-5555, 555.555.5555, +1 555 555 5555
PHONE_RE = re.compile(
    r"""
    (?<!\w)
    (?:\+?1[\s.-]?)?
    (?:\(\s*\d{3}\s*\)|\d{3})
    [\s.-]?
    \d{3}
    [\s.-]?
    \d{4}
    (?!\w)
    """,
    re.VERBOSE,
)

SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Credit card-ish (very rough): 13-19 digits with spaces/dashes
CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")

ADDRESS_RE = re.compile(
    r"""
    \b
    \d{1,6}\s+
    [A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,4}\s+
    (?:St|Street|Rd|Road|Ave|Avenue|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Pl|Place|Way|Pkwy|Parkway)
    \.?
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Dates: 10/29/2024, 2024-10-29 (basic)
DATE_NUM_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2})\b")

# Dates: October 29th, 2024 / Oct 29, 2024
DATE_WORD_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)

# "My name is X" heuristic (standard+)
MY_NAME_IS_RE = re.compile(r"\bmy name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", re.IGNORECASE)

# Conservative "First Last" matcher used only in heavy
FULLNAME_RE = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b")


# -----------------------------
# Helpers
# -----------------------------


def _level_rank(level: RedactLevel) -> int:
    return {"light": 1, "standard": 2, "heavy": 3}[level]


def _apply_regex(text: str, pattern: re.Pattern, label: str, replace_with: str) -> Tuple[str, List[RedactionHit]]:
    """
    Apply a regex replacement while collecting hit metadata.

    Note: indices refer to the pre-replacement text. For auditing this is fine.
    """
    hits: List[RedactionHit] = []
    out_parts: List[str] = []
    last = 0

    for m in pattern.finditer(text):
        start, end = m.span()
        hits.append(RedactionHit(label=label, match=m.group(0), start=start, end=end))
        out_parts.append(text[last:start])
        out_parts.append(replace_with)
        last = end

    out_parts.append(text[last:])
    return "".join(out_parts), hits


def _redact_exact_terms(text: str, terms: Iterable[str], label: str) -> Tuple[str, List[RedactionHit]]:
    """
    Redact caller-provided terms (names, orgs, etc) case-insensitively.

    This is high precision (only exact matches), and works well when you maintain
    a curated terms file.
    """
    hits: List[RedactionHit] = []
    terms_clean = [t.strip() for t in terms if t and t.strip()]
    if not terms_clean:
        return text, hits

    # Sort longer first to avoid partial masking issues
    terms_clean.sort(key=len, reverse=True)

    out = text
    for term in terms_clean:
        escaped = re.escape(term)
        pat = re.compile(rf"(?i)\b{escaped}\b")
        out, new_hits = _apply_regex(out, pat, label=label, replace_with=f"[{label}]")
        hits.extend(new_hits)

    return out, hits


def _parse_flags(flag_names: Iterable[str]) -> int:
    mapping = {
        "IGNORECASE": re.IGNORECASE,
        "I": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "M": re.MULTILINE,
        "DOTALL": re.DOTALL,
        "S": re.DOTALL,
    }
    flags = 0
    for name in flag_names:
        key = (name or "").strip().upper()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unsupported regex flag: {name!r}. Allowed: IGNORECASE, MULTILINE, DOTALL")
        flags |= mapping[key]
    return flags


def compile_custom_patterns(raw_patterns: Iterable[Dict[str, Any]]) -> List[CustomPattern]:
    """
    Convert JSON-friendly dict rules into CustomPattern objects.

    Each pattern dict supports:
      - label (required)
      - regex (required)
      - replace (optional) -> literal replacement string
      - min_level (optional) -> light|standard|heavy
      - flags (optional) -> list[str]: IGNORECASE/MULTILINE/DOTALL (or I/M/S)
    """
    compiled: List[CustomPattern] = []
    for i, p in enumerate(raw_patterns):
        if not isinstance(p, dict):
            raise ValueError(f"Pattern at index {i} must be an object/dict.")

        label = (p.get("label") or "").strip()
        regex_s = (p.get("regex") or "").strip()
        replace = p.get("replace", None)
        min_level = p.get("min_level", None)
        flags_raw = p.get("flags", [])

        if not label:
            raise ValueError(f"Pattern at index {i} missing required 'label'.")
        if not regex_s:
            raise ValueError(f"Pattern at index {i} missing required 'regex'.")

        if replace is not None and not isinstance(replace, str):
            raise ValueError(f"Pattern at index {i} 'replace' must be a string if provided.")

        if min_level is not None:
            if min_level not in ("light", "standard", "heavy"):
                raise ValueError(f"Pattern at index {i} 'min_level' must be one of: light|standard|heavy.")

        if flags_raw is None:
            flags_raw = []
        if not isinstance(flags_raw, list) or any(not isinstance(x, str) for x in flags_raw):
            raise ValueError(f"Pattern at index {i} 'flags' must be a list of strings.")

        flags = _parse_flags(flags_raw)

        # Validate regex compiles
        try:
            re.compile(regex_s, flags)
        except re.error as e:
            raise ValueError(f"Pattern at index {i} has invalid regex: {e}") from e

        compiled.append(
            CustomPattern(
                label=label,
                regex=regex_s,
                replace=replace,
                min_level=min_level,
                flags=flags,
            )
        )

    return compiled


def _apply_custom_patterns(
    text: str,
    patterns: Iterable[CustomPattern],
    level: RedactLevel,
) -> Tuple[str, List[RedactionHit]]:
    hits: List[RedactionHit] = []
    out = text

    for pat in patterns:
        if pat.min_level is not None and _level_rank(level) < _level_rank(pat.min_level):
            continue

        replace_with = pat.replace if pat.replace is not None else f"[{pat.label}]"
        rx = re.compile(pat.regex, pat.flags)
        out, h = _apply_regex(out, rx, pat.label, replace_with)
        hits.extend(h)

    return out, hits


# -----------------------------
# Public API
# -----------------------------


def redact_text(
    text: str,
    *,
    level: RedactLevel = "light",
    redact_names: Iterable[str] = (),
    redact_orgs: Iterable[str] = (),
    redact_locations: Iterable[str] = (),
    patterns: Iterable[CustomPattern] = (),
) -> Tuple[str, List[RedactionHit]]:
    """
    Offline redaction (deterministic):

      - light:
          EMAIL, PHONE, SSN, CARD, ADDRESS

      - standard:
          + ZIP, DATE (numeric + word), "my name is X" heuristic,
          + caller-provided NAME/ORG/LOCATION exact terms,
          + custom patterns

      - heavy:
          + standard
          + mask long digit runs (4+)
          + conservative full-name masking (can false-positive)
          + custom patterns (also applied)

    Returns:
      (redacted_text, hits)
    """
    hits_all: List[RedactionHit] = []
    out = text

    # Always-on basics (high confidence)
    out, h = _apply_regex(out, EMAIL_RE, "EMAIL", "[EMAIL]")
    hits_all.extend(h)

    out, h = _apply_regex(out, PHONE_RE, "PHONE", "[PHONE]")
    hits_all.extend(h)

    out, h = _apply_regex(out, SSN_RE, "SSN", "[SSN]")
    hits_all.extend(h)

    out, h = _apply_regex(out, CC_RE, "CARD", "[CARD]")
    hits_all.extend(h)

    out, h = _apply_regex(out, ADDRESS_RE, "ADDRESS", "[ADDRESS]")
    hits_all.extend(h)

    if level in ("standard", "heavy"):
        out, h = _apply_regex(out, ZIP_RE, "ZIP", "[ZIP]")
        hits_all.extend(h)

        out, h = _apply_regex(out, DATE_NUM_RE, "DATE", "[DATE]")
        hits_all.extend(h)

        out, h = _apply_regex(out, DATE_WORD_RE, "DATE", "[DATE]")
        hits_all.extend(h)

        # "My name is X" heuristic
        for m in MY_NAME_IS_RE.finditer(out):
            start, end = m.span(1)
            hits_all.append(RedactionHit(label="NAME", match=m.group(1), start=start, end=end))

        out = MY_NAME_IS_RE.sub(lambda m: re.sub(re.escape(m.group(1)), "[NAME]", m.group(0)), out)

        # Caller-provided exact matches
        out, h = _redact_exact_terms(out, redact_names, "NAME")
        hits_all.extend(h)

        out, h = _redact_exact_terms(out, redact_orgs, "ORG")
        hits_all.extend(h)

        out, h = _redact_exact_terms(out, redact_locations, "LOCATION")
        hits_all.extend(h)

    # Apply custom patterns after baseline redactions, before heavy-mode aggressiveness.
    out, h = _apply_custom_patterns(out, patterns, level)
    hits_all.extend(h)

    if level == "heavy":
        # Mask long digit sequences (account numbers, ids, etc.)
        long_digits = re.compile(r"\b\d{4,}\b")
        out, h = _apply_regex(out, long_digits, "NUM", "[NUM]")
        hits_all.extend(h)

        # Conservative full-name masking (false positives possible)
        def _full_name_sub(m: re.Match) -> str:
            full = m.group(0)
            hits_all.append(RedactionHit(label="NAME", match=full, start=m.start(), end=m.end()))
            return "[NAME]"

        out = FULLNAME_RE.sub(_full_name_sub, out)

    return out, hits_all
