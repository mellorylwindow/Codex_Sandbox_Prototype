from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FfmpegNotInstalledError(RuntimeError):
    pass


class AudioExtractionError(RuntimeError):
    pass


def assert_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise FfmpegNotInstalledError("ffmpeg not found on PATH. Install ffmpeg and try again.")


def extract_audio_wav_16k_mono(input_media: Path, output_wav: Path) -> Path:
    """
    Extract audio from arbitrary media into a 16kHz mono WAV suitable for Whisper.
    """
    assert_ffmpeg_available()

    input_media = input_media.expanduser().resolve()
    output_wav = output_wav.expanduser().resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_media),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise AudioExtractionError(
            "ffmpeg failed extracting audio.\n"
            f"Input: {input_media}\n"
            f"Output: {output_wav}\n"
            f"ffmpeg stderr:\n{stderr}"
        )

    if not output_wav.exists() or output_wav.stat().st_size == 0:
        raise AudioExtractionError(
            f"ffmpeg reported success, but output WAV is missing/empty: {output_wav}"
        )

    return output_wav
