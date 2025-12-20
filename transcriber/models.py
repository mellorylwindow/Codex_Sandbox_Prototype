# transcriber/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class Segment:
    """
    A single recognized speech segment with timestamps (seconds).
    """
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """
    Full transcript result.

    - text: convenience full-text (often joined segments)
    - segments: timestamped segments
    """
    text: str
    segments: list[Segment]


@dataclass(frozen=True)
class TranscribeOptions:
    """
    Configuration for transcription backends.

    Keep this frozen so calls are deterministic + safe to share across components.
    Any tuning should be done by constructing a new instance (not mutating).
    """

    # Backend selection (explicit for now)
    backend: Literal["faster-whisper"] = "faster-whisper"

    # Model selection / runtime
    model: str = "base"
    language: Optional[str] = None  # e.g. "en" (None = auto-detect)
    device: str = "cpu"  # "cpu" or "cuda"
    compute_type: str = "int8"  # good CPU default
    vad_filter: bool = True  # trims silence / improves robustness
    beam_size: int = 5

    # Offline biasing knobs (optional)
    initial_prompt: Optional[str] = None
    hotwords: Optional[str] = None
