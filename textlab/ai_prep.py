# textlab/ai_prep.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

TaskName = Literal["review", "grammar", "summary", "rewrite", "story"]


@dataclass(frozen=True)
class AITask:
    name: TaskName
    title: str
    instructions: str


DEFAULT_TASKS: dict[TaskName, AITask] = {
    "review": AITask(
        name="review",
        title="Review + structure",
        instructions=(
            "You are helping clean up a raw transcript.\n"
            "Goals:\n"
            "1) Fix obvious transcription errors (names, repeated fragments, garbled words).\n"
            "2) Improve readability with paragraphs and light punctuation.\n"
            "3) Do NOT add new facts.\n"
            "4) Keep timestamps if present.\n"
            "Output: cleaned transcript text only."
        ),
    ),
    "grammar": AITask(
        name="grammar",
        title="Grammar + punctuation",
        instructions=(
            "Fix grammar, punctuation, and line breaks while preserving meaning.\n"
            "Do NOT rewrite style heavily. Do NOT add new content.\n"
            "Output: corrected transcript text only."
        ),
    ),
    "summary": AITask(
        name="summary",
        title="Summary + bullets",
        instructions=(
            "Summarize the transcript.\n"
            "Output format:\n"
            "1) 5–10 bullet key points\n"
            "2) 1 short paragraph summary\n"
            "3) (Optional) action items if any are implied"
        ),
    ),
    "rewrite": AITask(
        name="rewrite",
        title="Rewrite for clarity",
        instructions=(
            "Rewrite the transcript into clear, readable prose.\n"
            "Preserve meaning. Remove filler and repetitions.\n"
            "Do NOT add facts.\n"
            "Output: rewritten prose only."
        ),
    ),
    "story": AITask(
        name="story",
        title="Novelize / short story draft",
        instructions=(
            "Turn the transcript into a short narrative scene.\n"
            "You may adjust pacing and sensory detail, but do NOT introduce new factual claims.\n"
            "Keep names and events consistent with the transcript.\n"
            "Output: story draft only."
        ),
    ),
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_manifest(run_dir: Path) -> dict:
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest.json in run dir: {run_dir}")
    return _read_json(manifest)


def _resolve_chunks(run_dir: Path, manifest: dict) -> list[Path]:
    # Prefer manifest chunks, fallback to filesystem
    chunk_entries = manifest.get("chunks") or []
    chunks_dir = run_dir / "chunks"

    if chunk_entries:
        paths = []
        for entry in chunk_entries:
            rel = entry.get("path") or entry.get("rel_path") or ""
            if not rel:
                continue
            p = (run_dir / rel).resolve()
            if p.exists():
                paths.append(p)
        if paths:
            return paths

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Missing chunks dir: {chunks_dir}")

    return sorted(chunks_dir.glob("chunk_*.txt"))


def _prompt_header(*, task: AITask, source_name: str, chunk_name: str, timestamped: bool) -> str:
    ts_note = "Timestamps are present — preserve them." if timestamped else "No timestamps — do not invent them."
    return (
        f"# TextLab AI Prep\n\n"
        f"**Task:** {task.title}\n\n"
        f"**Source:** {source_name}\n"
        f"**Chunk:** {chunk_name}\n"
        f"**Note:** {ts_note}\n\n"
        f"## Instructions\n{task.instructions}\n\n"
        f"## Input\n"
        f"(Paste the chunk content below this line when using an AI tool, if needed.)\n\n"
        f"---\n\n"
    )


def prepare_ai_pack(
    *,
    run_dir: Path,
    tasks: Iterable[TaskName],
    out_subdir: str = "ai_prep",
) -> Path:
    """
    Offline-only: generates prompt files + requests.jsonl inside the run directory.

    Creates:
      run_dir/ai_prep/<task>/chunk_000.prompt.md
      run_dir/ai_prep/requests.jsonl
      run_dir/ai_prep/README.md

    Returns ai_prep_dir.
    """
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    manifest = _load_manifest(run_dir)
    source_name = manifest.get("source_stem") or run_dir.name.split("__")[0]
    timestamped = bool(manifest.get("timestamped", False))

    chunk_paths = _resolve_chunks(run_dir, manifest)
    if not chunk_paths:
        raise RuntimeError("No chunks found to prepare.")

    ai_dir = _safe_mkdir(run_dir / out_subdir)

    selected: list[AITask] = []
    for t in tasks:
        if t not in DEFAULT_TASKS:
            raise ValueError(f"Unknown task: {t}")
        selected.append(DEFAULT_TASKS[t])

    # Write prompts + requests.jsonl
    requests_path = ai_dir / "requests.jsonl"
    req_lines: list[str] = []

    for task in selected:
        task_dir = _safe_mkdir(ai_dir / task.name)
        for chunk_path in chunk_paths:
            chunk_name = chunk_path.name
            prompt_path = task_dir / f"{chunk_name}.prompt.md"

            header = _prompt_header(
                task=task,
                source_name=str(source_name),
                chunk_name=chunk_name,
                timestamped=timestamped,
            )
            prompt_path.write_text(header, encoding="utf-8")

            req = {
                "custom_id": f"{source_name}:{task.name}:{chunk_name}",
                "task": task.name,
                "source": source_name,
                "chunk_path": str(chunk_path.relative_to(run_dir)),
                "prompt_path": str(prompt_path.relative_to(run_dir)),
                "timestamped": timestamped,
            }
            req_lines.append(json.dumps(req, ensure_ascii=False))

    requests_path.write_text("\n".join(req_lines) + "\n", encoding="utf-8")

    readme = (
        "# TextLab AI Prep Pack\n\n"
        "This folder is generated offline. Nothing here calls an AI.\n\n"
        "## What you get\n"
        "- Per-task prompt stubs in subfolders (review/grammar/summary/rewrite/story)\n"
        "- `requests.jsonl` mapping tasks ↔ chunks ↔ prompt files\n\n"
        "## Typical flow\n"
        "1) Choose a task folder (ex: `review/`)\n"
        "2) Open the prompt for a chunk\n"
        "3) Paste the chunk content below the `---` line\n"
        "4) Run through an AI tool (later) and save output next to the chunk if desired\n\n"
        "Tip: run textlab on *redacted* transcripts when you want share-safe packs.\n"
    )
    (ai_dir / "README.md").write_text(readme, encoding="utf-8")

    return ai_dir
