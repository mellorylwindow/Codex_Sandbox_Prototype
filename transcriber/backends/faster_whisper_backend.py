# transcriber/backends/faster_whisper_backend.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from transcriber.backends.base import TranscriptionBackend
from transcriber.models import Segment, Transcript, TranscribeOptions


class FasterWhisperBackend(TranscriptionBackend):
    """
    Faster-Whisper backend wrapper.

    Notes:
    - We try to pass optional knobs (prompt/hotwords) when available.
    - If the installed faster-whisper version doesn’t support a kwarg, we retry
      without it (so the CLI doesn’t explode on version drift).
    """

    def transcribe(self, audio_wav_16k_mono: Path, opts: TranscribeOptions) -> Transcript:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            opts.model,
            device=opts.device,
            compute_type=opts.compute_type,
        )

        # Base kwargs (stable across versions)
        kwargs: Dict[str, Any] = dict(
            language=opts.language,
            vad_filter=opts.vad_filter,
            beam_size=opts.beam_size,
        )

        # Optional assist knobs (may vary by faster-whisper version)
        if opts.prompt:
            # faster-whisper commonly uses `initial_prompt`
            kwargs["initial_prompt"] = opts.prompt

        if opts.hotwords:
            # faster-whisper "hotwords" is typically a string (comma-separated works well)
            kwargs["hotwords"] = ", ".join(opts.hotwords)

        segments_iter = self._transcribe_with_fallback(model, audio_wav_16k_mono, kwargs)

        segs: List[Segment] = []
        full_lines: List[str] = []

        for s in segments_iter:
            txt = (getattr(s, "text", "") or "").strip()
            if not txt:
                continue

            start = float(getattr(s, "start", 0.0))
            end = float(getattr(s, "end", 0.0))

            segs.append(Segment(start_s=start, end_s=end, text=txt))
            full_lines.append(txt)

        return Transcript(text="\n".join(full_lines).strip(), segments=tuple(segs))

    @staticmethod
    def _transcribe_with_fallback(model: Any, audio_path: Path, kwargs: Dict[str, Any]):
        """
        Try model.transcribe() with kwargs. If the installed faster-whisper rejects
        optional kwargs, retry without them (prompt/hotwords).
        """
        # First attempt: everything
        try:
            segments_iter, _info = model.transcribe(str(audio_path), **kwargs)
            return segments_iter
        except TypeError as e:
            msg = str(e)

            # Retry ladder: remove optional kwargs that are most likely unsupported.
            retry_kwargs = dict(kwargs)

            removed_any = False
            for maybe in ("hotwords", "initial_prompt"):
                if maybe in retry_kwargs and ("unexpected keyword" in msg or maybe in msg):
                    retry_kwargs.pop(maybe, None)
                    removed_any = True
                    try:
                        segments_iter, _info = model.transcribe(str(audio_path), **retry_kwargs)
                        return segments_iter
                    except TypeError:
                        # keep falling through
                        msg = str(e)

            # If we didn’t remove anything (or still failing), re-raise the original.
            if not removed_any:
                raise

            # Final fallback: strip all optional bits and try once more.
            retry_kwargs.pop("hotwords", None)
            retry_kwargs.pop("initial_prompt", None)
            segments_iter, _info = model.transcribe(str(audio_path), **retry_kwargs)
            return segments_iter
