# transcriber/transcribe_job.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Sequence

from transcriber.backends.faster_whisper_backend import FasterWhisperBackend
from transcriber.ffmpeg_audio import extract_audio_wav_16k_mono
from transcriber.models import Segment, Transcript, TranscribeOptions

TextMode = Literal["plain", "timestamps"]


def _safe_basename(name: str) -> str:
    """
    Convert any human-ish name into a filesystem-friendly basename.
    Keeps only [A-Za-z0-9 _ - .], converts whitespace to underscore, trims.
    """
    s = (name or "").strip().replace(" ", "_")
    safe = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))
    return safe or "media"


def _fmt_mmss(seconds: float) -> str:
    s = int(seconds)
    mm = s // 60
    ss = s % 60
    return f"{mm:02d}:{ss:02d}"


def _render_text(segments: Sequence[Segment], mode: TextMode) -> str:
    """
    Render transcript segments into a text-only representation.

    - plain: each segment text on its own line
    - timestamps: "[MM:SS] segment text"
    """
    if mode == "plain":
        return "\n".join(seg.text for seg in segments).strip()

    if mode == "timestamps":
        return "\n".join(f"[{_fmt_mmss(seg.start_s)}] {seg.text}" for seg in segments).strip()

    raise ValueError(f"Unsupported mode: {mode}")


def run_transcription(
    *,
    input_media: Path,
    out_dir: Path,
    opts: Optional[TranscribeOptions] = None,
    keep_wav: bool = False,
    write_segments_json: bool = True,
    write_srt: bool = False,
    write_vtt: bool = False,
    name: Optional[str] = None,
    mode: TextMode = "plain",
    subs_shift_ms: int = 0,
    subs_scale: float = 1.0,
    backend_kwargs: Optional[dict] = None,
) -> Path:
    """
    Transcribe a media file and write outputs to out_dir.

    Outputs:
      - <base>__YYYYMMDD_HHMMSS.txt
      - <base>__YYYYMMDD_HHMMSS.segments.json  (optional)
      - <base>__YYYYMMDD_HHMMSS.srt            (optional)
      - <base>__YYYYMMDD_HHMMSS.vtt            (optional)
      - out_dir/.scratch_transcribe/<base>__YYYYMMDD_HHMMSS.wav  (optional keep)

    Subtitle timing:
      - subs_shift_ms: constant offset (+ late / - early)
      - subs_scale: correct drift by scaling time (e.g. 0.9995, 1.0008)

    backend_kwargs:
      - backend-specific options passed through (e.g. initial_prompt, hotwords, word_timestamps)

    Returns:
      Path to the generated .txt transcript.
    """
    opts = opts or TranscribeOptions()

    input_media = input_media.expanduser().resolve()
    if not input_media.exists():
        raise FileNotFoundError(f"Input not found: {input_media}")
    if not input_media.is_file():
        raise ValueError(f"Input must be a file (not a directory): {input_media}")

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scratch_dir = out_dir / ".scratch_transcribe"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _safe_basename(name) if name else _safe_basename(input_media.stem)
    base_name = f"{base}__{stamp}"

    wav_path = scratch_dir / f"{base_name}.wav"
    extract_audio_wav_16k_mono(input_media, wav_path)

    # Backend selection stays explicit + simple for now
    if opts.backend != "faster-whisper":
        raise ValueError(f"Unsupported backend: {opts.backend}")

    backend = FasterWhisperBackend()
    transcript: Transcript = backend.transcribe(wav_path, opts, **(backend_kwargs or {}))

    # Prefer segments for output modes; transcript.text is retained as a convenience
    txt_out = _render_text(transcript.segments, mode=mode) if transcript.segments else (transcript.text or "").strip()

    txt_path = out_dir / f"{base_name}.txt"
    txt_path.write_text(txt_out + "\n", encoding="utf-8")

    if write_segments_json:
        json_path = out_dir / f"{base_name}.segments.json"
        payload = [{"start_s": seg.start_s, "end_s": seg.end_s, "text": seg.text} for seg in transcript.segments]
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if write_srt:
        srt_path = out_dir / f"{base_name}.srt"
        srt_path.write_text(
            _to_srt(transcript.segments, shift_ms=subs_shift_ms, scale=subs_scale) + "\n",
            encoding="utf-8",
        )

    if write_vtt:
        vtt_path = out_dir / f"{base_name}.vtt"
        vtt_path.write_text(
            _to_vtt(transcript.segments, shift_ms=subs_shift_ms, scale=subs_scale) + "\n",
            encoding="utf-8",
        )

    if not keep_wav:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            # Non-fatal: leaving scratch files is acceptable.
            pass

    return txt_path


def _apply_time_transform(seconds: float, *, shift_ms: int, scale: float) -> float:
    """
    Apply subtitle timing transform:
      t' = max(0, t * scale + shift_ms/1000)

    - shift_ms fixes constant offset (late/early)
    - scale fixes drift (subtitles slowly slide)
    """
    if scale <= 0:
        raise ValueError(f"subs_scale must be > 0, got: {scale}")

    shifted = (seconds * float(scale)) + (float(shift_ms) / 1000.0)
    return max(0.0, shifted)


def _fmt_srt_time(seconds: float) -> str:
    # seconds -> HH:MM:SS,mmm
    ms_total = int(round(seconds * 1000))
    hh = ms_total // 3_600_000
    ms_total -= hh * 3_600_000
    mm = ms_total // 60_000
    ms_total -= mm * 60_000
    ss = ms_total // 1000
    ms_total -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms_total:03d}"


def _fmt_vtt_time(seconds: float) -> str:
    # seconds -> HH:MM:SS.mmm (WebVTT)
    ms_total = int(round(seconds * 1000))
    hh = ms_total // 3_600_000
    ms_total -= hh * 3_600_000
    mm = ms_total // 60_000
    ms_total -= mm * 60_000
    ss = ms_total // 1000
    ms_total -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms_total:03d}"


def _to_srt(segments: Sequence[Segment], *, shift_ms: int = 0, scale: float = 1.0) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _apply_time_transform(seg.start_s, shift_ms=shift_ms, scale=scale)
        end = _apply_time_transform(seg.end_s, shift_ms=shift_ms, scale=scale)

        # Safety: avoid zero/negative duration cues after transforms
        if end <= start:
            end = start + 0.25  # 250ms minimum cue length

        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines).strip()


def _to_vtt(segments: Sequence[Segment], *, shift_ms: int = 0, scale: float = 1.0) -> str:
    """
    Minimal WebVTT writer (no styling). Many players accept this format readily.
    """
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _apply_time_transform(seg.start_s, shift_ms=shift_ms, scale=scale)
        end = _apply_time_transform(seg.end_s, shift_ms=shift_ms, scale=scale)

        if end <= start:
            end = start + 0.25

        lines.append(f"{_fmt_vtt_time(start)} --> {_fmt_vtt_time(end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines).strip()
