# transcriber/backends/faster_whisper_backend.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from transcriber.backends.base import TranscriptionBackend
from transcriber.models import Segment, Transcript, TranscribeOptions


class FasterWhisperBackend(TranscriptionBackend):
    """
    faster-whisper backend.

    Notes:
    - We pass optional biasing args (initial_prompt, hotwords) when available.
    - Some faster-whisper versions may not support one or both args; we retry without them.
    """

    def transcribe(self, audio_wav_16k_mono: Path, opts: TranscribeOptions) -> Transcript:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            opts.model,
            device=opts.device,
            compute_type=opts.compute_type,
        )

        base_kwargs = dict(
            language=opts.language,
            vad_filter=opts.vad_filter,
            beam_size=opts.beam_size,
        )

        # Optional offline biasing knobs (supported by newer faster-whisper builds).
        # We add them conditionally and retry without if the installed version rejects them.
        bias_kwargs = dict(base_kwargs)

        if opts.initial_prompt:
            bias_kwargs["initial_prompt"] = opts.initial_prompt
        if opts.hotwords:
            bias_kwargs["hotwords"] = opts.hotwords

        segments_iter, _info = self._transcribe_with_fallback(model, audio_wav_16k_mono, bias_kwargs, base_kwargs)

        segs: List[Segment] = []
        full_lines: List[str] = []

        for s in segments_iter:
            txt = (getattr(s, "text", "") or "").strip()
            if not txt:
                continue
            segs.append(Segment(start_s=float(s.start), end_s=float(s.end), text=txt))
            full_lines.append(txt)

        return Transcript(text="\n".join(full_lines).strip(), segments=segs)

    @staticmethod
    def _transcribe_with_fallback(
        model: "WhisperModel",
        audio_wav_16k_mono: Path,
        bias_kwargs: dict,
        base_kwargs: dict,
    ) -> Tuple[object, object]:
        """
        Try transcription with bias kwargs first; if the installed faster-whisper version
        rejects unknown kwargs, retry with base kwargs only.
        """
        try:
            return model.transcribe(str(audio_wav_16k_mono), **bias_kwargs)
        except TypeError:
            # Unknown parameter(s) such as hotwords/initial_prompt on older versions.
            return model.transcribe(str(audio_wav_16k_mono), **base_kwargs)
