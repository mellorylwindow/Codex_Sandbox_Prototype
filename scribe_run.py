from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Models
# ----------------------------

@dataclass(frozen=True)
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class VisualStep:
    step: int
    ts_s: float
    ts: str
    image_relpath: str  # empty string => transcript-only step


# ----------------------------
# Shell helpers
# ----------------------------

def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(cmd)}\n\n"
            f"STDOUT:\n{proc.stdout}\n\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit(
            "ffmpeg/ffprobe not found on PATH.\n"
            "Install with:\n"
            "  winget install -e --id Gyan.FFmpeg\n"
            "Then close/reopen Git Bash."
        )


# ----------------------------
# Formatting + cleaning
# ----------------------------

def _format_ts(seconds: float) -> str:
    """mm:ss.t (tenths)"""
    if seconds < 0:
        seconds = 0.0
    s_int = int(seconds)
    tenths = int((seconds - s_int) * 10 + 1e-9)
    m = (s_int % 3600) // 60
    s = s_int % 60
    return f"{m:02d}:{s:02d}.{tenths}"


def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------
# Video probing
# ----------------------------

def _has_audio_stream(video_path: Path) -> bool:
    proc = subprocess.run(
        ["ffprobe", "-hide_banner", "-i", str(video_path)],
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return "Audio:" in combined


# ----------------------------
# Storyboard extraction
# ----------------------------

def _extract_start_frame(video_path: Path, screenshots_dir: Path) -> Tuple[float, Path]:
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    out_path = screenshots_dir / "start_00000.png"
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss", "0",
            "-i", str(video_path),
            "-frames:v", "1",
            str(out_path),
        ]
    )
    return (0.0, out_path)


def _extract_scene_frames(
    video_path: Path,
    screenshots_dir: Path,
    *,
    scene_threshold: float,
    max_frames: int,
) -> List[Tuple[float, Path]]:
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(screenshots_dir / "scene_%05d.png")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
        "-vsync", "vfr",
        "-frames:v", str(max_frames),
        pattern,
    ]
    proc = _run(cmd)

    files = sorted(screenshots_dir.glob("scene_*.png"))
    if not files:
        return []

    times: List[float] = []
    for line in (proc.stderr or "").splitlines():
        if "showinfo" not in line:
            continue
        m = re.search(r"pts_time:(\d+(\.\d+)?)", line)
        if m:
            times.append(float(m.group(1)))

    if times:
        n = min(len(files), len(times))
        return [(times[i], files[i]) for i in range(n)]

    # fallback: sequential numbering as time
    return [(float(i), files[i]) for i in range(len(files))]


def _extract_interval_frames(
    video_path: Path,
    screenshots_dir: Path,
    *,
    every_s: float,
    max_frames: int,
) -> List[Tuple[float, Path]]:
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(screenshots_dir / "step_%05d.png")

    _run(
        [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vf", f"fps=1/{every_s}",
            "-frames:v", str(max_frames),
            pattern,
        ]
    )

    files = sorted(screenshots_dir.glob("step_*.png"))
    return [(i * every_s, files[i]) for i in range(len(files))]


def _dedupe_by_time(frames: List[Tuple[float, Path]], eps: float = 0.05) -> List[Tuple[float, Path]]:
    out: List[Tuple[float, Path]] = []
    last_t: Optional[float] = None
    for t, p in sorted(frames, key=lambda x: x[0]):
        if last_t is None or abs(t - last_t) > eps:
            out.append((t, p))
            last_t = t
    return out


def _apply_min_gap(frames: List[Tuple[float, Path]], min_gap: float) -> List[Tuple[float, Path]]:
    if min_gap <= 0:
        return sorted(frames, key=lambda x: x[0])
    kept: List[Tuple[float, Path]] = []
    last_t: Optional[float] = None
    for t, p in sorted(frames, key=lambda x: x[0]):
        if last_t is None or (t - last_t) >= min_gap:
            kept.append((t, p))
            last_t = t
    return kept


def _build_visual_steps(out_dir: Path, frames: List[Tuple[float, Path]]) -> List[VisualStep]:
    steps: List[VisualStep] = []
    for idx, (t, p) in enumerate(sorted(frames, key=lambda x: x[0]), start=1):
        rel = str(p.relative_to(out_dir)).replace("\\", "/")
        steps.append(VisualStep(step=idx, ts_s=float(t), ts=_format_ts(t), image_relpath=rel))
    return steps


# ----------------------------
# Audio transcription (secondary)
# ----------------------------

def _extract_audio(video_path: Path, out_wav: Path, *, volume: float = 1.0) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
    ]
    if volume and volume != 1.0:
        cmd += ["-af", f"volume={volume}"]
    cmd += [str(out_wav)]
    _run(cmd)


def _transcribe(audio_wav: Path) -> List[TranscriptSegment]:
    # faster-whisper first
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel("base", device="cpu", compute_type="int8")

        segs, _ = model.transcribe(str(audio_wav), vad_filter=True)
        out: List[TranscriptSegment] = []
        for s in segs:
            txt = _clean_text(getattr(s, "text", "") or "")
            if txt:
                out.append(TranscriptSegment(float(s.start), float(s.end), txt))
        if out:
            return out

        segs, _ = model.transcribe(str(audio_wav), vad_filter=False)
        out = []
        for s in segs:
            txt = _clean_text(getattr(s, "text", "") or "")
            if txt:
                out.append(TranscriptSegment(float(s.start), float(s.end), txt))
        return out
    except Exception:
        pass

    # openai-whisper fallback
    try:
        import whisper  # type: ignore

        model = whisper.load_model("base")
        res = model.transcribe(str(audio_wav), fp16=False)
        out: List[TranscriptSegment] = []
        for s in res.get("segments", []):
            txt = _clean_text((s.get("text") or ""))
            if txt:
                out.append(
                    TranscriptSegment(
                        float(s.get("start", 0.0)),
                        float(s.get("end", 0.0)),
                        txt,
                    )
                )
        return out
    except Exception:
        return []


def _merge_segments(segs: List[TranscriptSegment]) -> List[TranscriptSegment]:
    def wc(x: str) -> int:
        return len([w for w in x.split(" ") if w])

    merged: List[TranscriptSegment] = []
    buf: Optional[TranscriptSegment] = None

    for s in segs:
        s = TranscriptSegment(s.start_s, s.end_s, _clean_text(s.text))
        if not s.text:
            continue

        if buf is None:
            buf = s
            continue

        if wc(buf.text) < 8 or (buf.end_s - buf.start_s) < 2.5:
            buf = TranscriptSegment(buf.start_s, s.end_s, f"{buf.text} {s.text}".strip())
        else:
            merged.append(buf)
            buf = s

    if buf is not None:
        merged.append(buf)
    return merged


# ----------------------------
# Transcript under screenshots
# ----------------------------

def _assign_transcript_to_steps(
    steps: List[VisualStep],
    segs: List[TranscriptSegment],
    *,
    mode: str = "previous",     # "previous" | "nearest"
    tail_step: str = "auto",    # "auto" | "last"
    bucket_s: float = 30.0,     # larger = fewer transcript-only steps
) -> Tuple[Dict[int, List[TranscriptSegment]], List[VisualStep]]:
    """
    Assign transcript segments to visual steps.

    If a segment occurs AFTER the final screenshot:
      - tail_step="last": attach to last screenshot
      - tail_step="auto": create transcript-only steps (time buckets)
    """
    by_step: Dict[int, List[TranscriptSegment]] = {s.step: [] for s in steps}
    extra_steps: List[VisualStep] = []

    if not steps or not segs:
        return by_step, extra_steps

    step_times = [(s.step, s.ts_s) for s in steps]
    step_times.sort(key=lambda x: x[1])

    last_step_num = steps[-1].step
    last_ts = step_times[-1][1]

    def nearest_step(t: float) -> int:
        best_step = step_times[0][0]
        best_dist = abs(t - step_times[0][1])
        for st, ts in step_times[1:]:
            d = abs(t - ts)
            if d < best_dist:
                best_dist = d
                best_step = st
        return best_step

    def previous_step(t: float) -> int:
        prev = step_times[0][0]
        for st, ts in step_times:
            if ts <= t:
                prev = st
            else:
                break
        return prev

    tail: List[TranscriptSegment] = []

    for seg in segs:
        mid = (seg.start_s + seg.end_s) / 2.0

        if mid > last_ts and tail_step == "auto":
            tail.append(seg)
            continue

        st = previous_step(mid) if mode == "previous" else nearest_step(mid)
        by_step.setdefault(st, []).append(seg)

    if tail_step == "auto" and tail:
        buckets: Dict[int, List[TranscriptSegment]] = {}
        for seg in tail:
            mid = (seg.start_s + seg.end_s) / 2.0
            k = int(mid // bucket_s)
            buckets.setdefault(k, []).append(seg)

        next_step = last_step_num + 1
        for k in sorted(buckets.keys()):
            bucket = buckets[k]
            t0 = min(s.start_s for s in bucket)
            extra_steps.append(VisualStep(step=next_step, ts_s=t0, ts=_format_ts(t0), image_relpath=""))
            by_step[next_step] = bucket
            next_step += 1

    return by_step, extra_steps


# ----------------------------
# Writers
# ----------------------------

def _write_outputs(
    *,
    video_path: Path,
    out_dir: Path,
    visual_steps: List[VisualStep],
    transcript_segments: List[TranscriptSegment],
    transcript_mode: str,
    tail_step: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # transcript.txt (raw)
    if transcript_segments:
        transcript_txt = "\n".join(f"[{_format_ts(s.start_s)}] {s.text}" for s in transcript_segments) + "\n"
    else:
        transcript_txt = "(no transcript)\n"
    (out_dir / "transcript.txt").write_text(transcript_txt, encoding="utf-8")

    # steps.json
    payload: Dict[str, Any] = {
        "source_video": str(video_path),
        "visual_steps": [
            {"step": s.step, "ts_s": s.ts_s, "ts": s.ts, "image": s.image_relpath}
            for s in visual_steps
        ],
        "transcript_steps": [
            {
                "step": i + 1,
                "start_s": s.start_s,
                "end_s": s.end_s,
                "start_ts": _format_ts(s.start_s),
                "end_ts": _format_ts(s.end_s),
                "text": s.text,
            }
            for i, s in enumerate(transcript_segments)
        ],
    }
    (out_dir / "steps.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    assigned, extra_steps = _assign_transcript_to_steps(
        visual_steps,
        transcript_segments,
        mode=transcript_mode,
        tail_step=tail_step,
    )
    all_steps = sorted((visual_steps + extra_steps), key=lambda s: s.ts_s)

    # steps.md (merged)
    md: List[str] = []
    md.append("# Scribe Output (Sandbox)")
    md.append("")
    md.append(f"**Source:** `{video_path.name}`")
    md.append("")
    md.append("## Steps")
    md.append("")

    for s in all_steps:
        md.append(f"{s.step}. **{s.ts}**")
        md.append("")

        if s.image_relpath:
            md.append(f"   ![]({s.image_relpath})")
        else:
            md.append("   _(Transcript-only step)_")
        md.append("")

        segs = assigned.get(s.step, [])
        if segs:
            md.append("   **Transcript:**")
            for seg in segs:
                md.append(f"   - `{_format_ts(seg.start_s)} → {_format_ts(seg.end_s)}` — {seg.text}")
            md.append("")

    if not transcript_segments:
        md.append("_No transcript available._")
        md.append("")

    (out_dir / "steps.md").write_text("\n".join(md), encoding="utf-8")


# ----------------------------
# PDF export (optional)
# ----------------------------

def _export_steps_pdf(out_dir: Path, pdf_path: Path) -> None:
    """
    Export out_dir/steps.md (+ relative screenshots) to a PDF using Playwright.

    One-time install:
      python -m pip install playwright markdown
      python -m playwright install chromium
    """
    try:
        import markdown  # type: ignore
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:
        raise SystemExit(
            "PDF export dependencies missing.\n"
            "Install with:\n"
            "  python -m pip install playwright markdown\n"
            "  python -m playwright install chromium\n"
            f"\nDetails: {e}"
        )

    md_path = out_dir / "steps.md"
    if not md_path.exists():
        raise SystemExit(f"Missing steps.md at: {md_path}")

    html_path = out_dir / "steps.html"

    template = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Scribe Export</title>
  <style>
    @page { margin: 18mm 14mm; }
    body { font-family: Arial, Helvetica, sans-serif; font-size: 12.5px; line-height: 1.45; color: #111; }
    h1 { font-size: 20px; margin: 0 0 10px; }
    h2 { font-size: 16px; margin: 18px 0 8px; }
    h3 { font-size: 14px; margin: 14px 0 6px; }
    code { font-family: Consolas, "Courier New", monospace; font-size: 11.5px; }
    img { max-width: 100%; height: auto; border: 1px solid #e5e5e5; border-radius: 6px; }
    ul { margin-top: 6px; }
    li { margin: 4px 0; }
    .pagebreak { page-break-before: always; }
    .meta { color: #555; margin-bottom: 10px; }
  </style>
</head>
<body>
__CONTENT__
</body>
</html>
"""

    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["extra"])
    html = template.replace("__CONTENT__", html_body)
    html_path.write_text(html, encoding="utf-8")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    file_url = html_path.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(file_url, wait_until="networkidle")
        page.emulate_media(media="screen")
        page.pdf(path=str(pdf_path), format="Letter", print_background=True)
        browser.close()


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Scribe Clone (Sandbox) — v0.7 (merged transcript + PDF)")
    parser.add_argument("video", help="Path to .mp4 screen recording")
    parser.add_argument("--out", required=True, help="Output folder")

    # Visual capture
    parser.add_argument(
        "--strategy",
        choices=["scene", "interval", "hybrid"],
        default="hybrid",
        help="scene=scene change only, interval=every N seconds, hybrid=start + scene (+ interval if needed)",
    )
    parser.add_argument("--scene", type=float, default=0.45, help="Scene-change threshold (higher = fewer frames)")
    parser.add_argument("--every", type=float, default=2.0, help="Interval seconds (used by interval/hybrid fallback)")
    parser.add_argument("--max-frames", type=int, default=160, help="Max frames to extract per extractor")
    parser.add_argument("--min-gap", type=float, default=0.9, help="Minimum seconds between kept frames")
    parser.add_argument("--min-frames", type=int, default=6, help="Hybrid: if scene yields fewer than this, add interval frames")

    # Audio
    parser.add_argument("--volume", type=float, default=1.0, help="Audio volume multiplier (e.g. 8, 16)")

    # Transcript merging
    parser.add_argument("--transcript-mode", choices=["nearest", "previous"], default="previous",
                        help="How transcript segments attach to steps")
    parser.add_argument("--tail-step", choices=["auto", "last"], default="auto",
                        help="If transcript occurs after final screenshot: create transcript-only steps or attach to last")

    # PDF export (optional)
    parser.add_argument("--pdf", action="store_true", help="Also export steps.pdf into the output folder")
    parser.add_argument("--pdf-name", default="steps.pdf", help="PDF filename (default: steps.pdf)")

    args = parser.parse_args()
    _ensure_ffmpeg()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    out_dir = Path(args.out).resolve()
    screenshots_dir = out_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Visual storyboard
    print("[1/3] Extracting storyboard frames…")
    frames: List[Tuple[float, Path]] = []

    if args.strategy in ("hybrid", "interval"):
        frames.append(_extract_start_frame(video_path, screenshots_dir))

    if args.strategy in ("scene", "hybrid"):
        frames += _extract_scene_frames(
            video_path,
            screenshots_dir,
            scene_threshold=float(args.scene),
            max_frames=int(args.max_frames),
        )

    if args.strategy == "interval":
        frames = _extract_interval_frames(
            video_path,
            screenshots_dir,
            every_s=max(0.25, float(args.every)),
            max_frames=int(args.max_frames),
        )
        frames = [(_extract_start_frame(video_path, screenshots_dir))] + frames

    if args.strategy == "hybrid":
        if len(frames) < int(args.min_frames):
            frames += _extract_interval_frames(
                video_path,
                screenshots_dir,
                every_s=max(0.25, float(args.every)),
                max_frames=int(args.max_frames),
            )

    frames = _dedupe_by_time(frames)
    frames = _apply_min_gap(frames, float(args.min_gap))
    visual_steps = _build_visual_steps(out_dir, frames)

    # 2) Audio transcript (best-effort)
    transcript_segments: List[TranscriptSegment] = []
    if _has_audio_stream(video_path):
        tmp_dir = Path("scribe/tmp").resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        audio_wav = tmp_dir / f"{video_path.stem}.wav"

        print("[2/3] Extracting audio…")
        _extract_audio(video_path, audio_wav, volume=float(args.volume))

        print("[3/3] Transcribing (best-effort)…")
        transcript_segments = _merge_segments(_transcribe(audio_wav))
    else:
        print("[2/3] No audio stream detected (skipping transcription).")
        print("[3/3] Building outputs…")

    _write_outputs(
        video_path=video_path,
        out_dir=out_dir,
        visual_steps=visual_steps,
        transcript_segments=transcript_segments,
        transcript_mode=str(args.transcript_mode),
        tail_step=str(args.tail_step),
    )

    if args.pdf:
        pdf_path = out_dir / str(args.pdf_name)
        _export_steps_pdf(out_dir, pdf_path)
        print(f"OK: wrote PDF  -> {pdf_path}")

    print(f"Done. Output in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
