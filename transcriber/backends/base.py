# transcriber/backends/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from transcriber.models import Transcript, TranscribeOptions


class TranscriptionBackend(ABC):
    """
    Base interface for transcription backends.

    Notes:
    - `audio_wav_16k_mono` should be a path to a 16kHz mono WAV file.
    - `opts` contains common backend-agnostic options.
    - `**kwargs` allows backend-specific options (e.g. initial_prompt, hotwords, word_timestamps)
      without polluting the shared options model. Backends MUST ignore unknown kwargs safely.
    """

    @abstractmethod
    def transcribe(
        self,
        audio_wav_16k_mono: Path,
        opts: TranscribeOptions,
        **kwargs: Any,
    ) -> Transcript:
        raise NotImplementedError
