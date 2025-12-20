from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple

RedactLevel = Literal["light", "standard", "heavy"]


@dataclass(frozen=True)
class RedactionHit:
    label: str
    match: str
    start: int
    end: int


# --- Core patterns (offline, deterministic) ---

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

# SSN
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Credit card-ish (very rough): 13-19 digits with spaces/dashes
CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# US ZIP / ZIP+4
ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")

# Simple street address heuristic: "123 Main St", "45 Elm Road", etc.
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

# Very conservative "full name" heuristic: Two Capitalized Words (can false-positive)
FULLNAME_RE = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b")


def _apply_regex(text: str, pattern: re.Pattern, label: str, replace_with: str) -> Tuple[str, List[RedactionHit]]:
    hits: List[RedactionHit] = []
    out = []
    last = 0

    for m in pattern.finditer(text):
        start, end = m.span()
        hits.append(RedactionHit(label=label, match=m.group(0), start=start, end=end))
        out.append(text[last:start])
        out.append(replace_with)
        last = end

    out.append(text[last:])
    return "".join(out), hits


def _redact_exact_terms(text: str, terms: Iterable[str], label: str) -> Tuple[str, List[RedactionHit]]:
    """
    Redact caller-provided terms (names, orgs, etc) case-insensitively.
    """
    hits: List[RedactionHit] = []
    terms_clean = [t.strip() for t in terms if t and t.strip()]
    if not terms_clean:
        return text, hits

    # Sort longer first to avoid partial masking issues
    terms_clean.sort(key=len, reverse=True)

    for term in terms_clean:
        # word boundary where possible; allow spaces inside term
        escaped = re.escape(term)
        pat = re.compile(rf"(?i)\b{escaped}\b")
        text, new_hits = _apply_regex(text, pat, label=label, replace_with=f"[{label}]")
        hits.extend(new_hits)

    return text, hits


def redact_text(
    text: str,
    *,
    level: RedactLevel = "light",
    redact_names: Iterable[str] = (),
    redact_orgs: Iterable[str] = (),
    redact_locations: Iterable[str] = (),
) -> Tuple[str, List[RedactionHit]]:
    """
    Offline redaction:
      - light: email/phone/ssn/cc/address/zip
      - standard: + dates + "my name is X" + caller-provided names/orgs/locations
      - heavy: + aggressive digit masking + conservative full-name masking (can false-positive)

    Returns (redacted_text, hits).
    """
    hits_all: List[RedactionHit] = []
    out = text

    # Always-on basics
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

    # ZIP alone is weakly identifying; redact only standard+
    if level in ("standard", "heavy"):
        out, h = _apply_regex(out, ZIP_RE, "ZIP", "[ZIP]")
        hits_all.extend(h)

        out, h = _apply_regex(out, DATE_NUM_RE, "DATE", "[DATE]")
        hits_all.extend(h)

        out, h = _apply_regex(out, DATE_WORD_RE, "DATE", "[DATE]")
        hits_all.extend(h)

        # "My name is X" heuristic
        def _name_replacer(m: re.Match) -> str:
            # keep "my name is" phrase, replace name
            return re.sub(m.group(1), "[NAME]", m.group(0), flags=re.IGNORECASE)

        # Use sub with manual hit capture
        for m in MY_NAME_IS_RE.finditer(out):
            start, end = m.span(1)
            hits_all.append(RedactionHit(label="NAME", match=m.group(1), start=start, end=end))
        out = MY_NAME_IS_RE.sub(lambda m: re.sub(re.escape(m.group(1)), "[NAME]", m.group(0)), out)

        # Caller-provided lists (high precision)
        out, h = _redact_exact_terms(out, redact_names, "NAME")
        hits_all.extend(h)

        out, h = _redact_exact_terms(out, redact_orgs, "ORG")
        hits_all.extend(h)

        out, h = _redact_exact_terms(out, redact_locations, "LOCATION")
        hits_all.extend(h)

    if level == "heavy":
        # Mask long digit sequences (account numbers, ids, etc.)
        # Replace any run of 4+ digits with [NUM]
        long_digits = re.compile(r"\b\d{4,}\b")
        out, h = _apply_regex(out, long_digits, "NUM", "[NUM]")
        hits_all.extend(h)

        # Conservative full-name masking (false positives possible)
        # Mask "First Last" where both are capitalized.
        # Only do this in heavy mode.
        def _full_name_sub(m: re.Match) -> str:
            full = m.group(0)
            hits_all.append(RedactionHit(label="NAME", match=full, start=m.start(), end=m.end()))
            return "[NAME]"

        out = FULLNAME_RE.sub(_full_name_sub, out)

    return out, hits_all
