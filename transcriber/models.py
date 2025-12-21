# transcriber/models.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """
    text: convenience full text (may be derived from segments)
    segments: time-aligned segments (preferred for timestamps/subtitles)
    """
    text: str
    segments: Tuple[Segment, ...] = ()


@dataclass(frozen=True)
class TranscribeOptions:
    """
    Immutable options object (frozen=True) to prevent accidental mutation.
    Use helper methods (.with_prompt / .with_hotwords) to derive updated copies.
    """
    backend: Literal["faster-whisper"] = "faster-whisper"
    model: str = "base"
    language: Optional[str] = None  # e.g. "en" (None = auto-detect)
    device: str = "cpu"             # "cpu" or "cuda"
    compute_type: str = "int8"      # good CPU default
    vad_filter: bool = True         # trims silence / improves robustness
    beam_size: int = 5

    # “assist” knobs (offline-first, but passed into backend if supported)
    prompt: Optional[str] = None
    hotwords: Tuple[str, ...] = ()

    def with_prompt(self, prompt: Optional[str]) -> "TranscribeOptions":
        p = (prompt or "").strip()
        return replace(self, prompt=p or None)

    def with_hotwords(self, hotwords: Sequence[str] | None) -> "TranscribeOptions":
        if not hotwords:
            return replace(self, hotwords=())
        cleaned = []
        seen = set()
        for w in hotwords:
            if not isinstance(w, str):
                continue
            ww = w.strip()
            if not ww:
                continue
            if ww in seen:
                continue
            cleaned.append(ww)
            seen.add(ww)
        return replace(self, hotwords=tuple(cleaned))
