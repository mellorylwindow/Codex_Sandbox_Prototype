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


def _to_srt(segments: Sequence[Segment]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(seg.start_s)} --> {_fmt_srt_time(seg.end_s)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines).strip()


def _fmt_vtt_time(seconds: float) -> str:
    # seconds -> HH:MM:SS.mmm  (WebVTT)
    ms_total = int(round(seconds * 1000))
    hh = ms_total // 3_600_000
    ms_total -= hh * 3_600_000
    mm = ms_total // 60_000
    ms_total -= mm * 60_000
    ss = ms_total // 1000
    ms_total -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms_total:03d}"


def _to_vtt(segments: Sequence[Segment]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_fmt_vtt_time(seg.start_s)} --> {_fmt_vtt_time(seg.end_s)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines).strip()


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
) -> Path:
    """
    Transcribe a media file and write outputs to out_dir.

    Outputs:
      - <base>__YYYYMMDD_HHMMSS.txt
      - <base>__YYYYMMDD_HHMMSS.segments.json  (optional)
      - <base>__YYYYMMDD_HHMMSS.srt            (optional)
      - <base>__YYYYMMDD_HHMMSS.vtt            (optional)
      - out_dir/.scratch_transcribe/<base>__YYYYMMDD_HHMMSS.wav  (optional keep)

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
    if getattr(opts, "backend", "faster-whisper") != "faster-whisper":
        raise ValueError(f"Unsupported backend: {getattr(opts, 'backend', None)}")

    backend = FasterWhisperBackend()
    transcript: Transcript = backend.transcribe(wav_path, opts)

    segments = transcript.segments or []
    txt_out = _render_text(segments, mode=mode) if segments else (transcript.text or "").strip()

    txt_path = out_dir / f"{base_name}.txt"
    txt_path.write_text(txt_out + "\n", encoding="utf-8")

    if write_segments_json:
        json_path = out_dir / f"{base_name}.segments.json"
        payload = [{"start_s": seg.start_s, "end_s": seg.end_s, "text": seg.text} for seg in segments]
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if write_srt:
        srt_path = out_dir / f"{base_name}.srt"
        srt_path.write_text((_to_srt(segments) if segments else "") + "\n", encoding="utf-8")

    if write_vtt:
        vtt_path = out_dir / f"{base_name}.vtt"
        vtt_path.write_text((_to_vtt(segments) if segments else "") + "\n", encoding="utf-8")

    if not keep_wav:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            # Non-fatal: leaving scratch files is acceptable.
            pass

    return txt_path
