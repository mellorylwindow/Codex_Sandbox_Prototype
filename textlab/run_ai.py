# textlab/run_ai.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -------------------------
# Models / Helpers
# -------------------------

@dataclass(frozen=True)
class MaterializeResult:
    out_dir: Path
    manifest_path: Path
    prompt_files: List[Path]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL at {path} line {i}: {e}") from e
    return items


def _safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "untitled"


def _infer_chunk_id(obj: Dict[str, Any]) -> Optional[str]:
    """
    Be tolerant to different schemas:
      - chunk: "chunk_000"
      - chunk_id: "chunk_000"
      - chunk_index: 0  -> chunk_000
      - chunk_path: ".../chunk_000.txt"
    """
    for k in ("chunk", "chunk_id"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    idx = obj.get("chunk_index")
    if isinstance(idx, int) and idx >= 0:
        return f"chunk_{idx:03d}"

    p = obj.get("chunk_path") or obj.get("path")
    if isinstance(p, str) and "chunk_" in p:
        m = re.search(r"(chunk_\d{3})", p)
        if m:
            return m.group(1)

    return None


def _load_chunk_text(run_dir: Path, chunk_id: str, obj: Dict[str, Any]) -> str:
    """
    Prefer explicit chunk_path if provided, else run_dir/chunks/<chunk_id>.txt
    """
    # explicit path in request
    p = obj.get("chunk_path") or obj.get("path")
    if isinstance(p, str) and p.strip():
        pp = Path(p)
        if not pp.is_absolute():
            pp = (run_dir / pp).resolve()
        if pp.exists() and pp.is_file():
            return pp.read_text(encoding="utf-8", errors="replace")

    # standard chunk path
    candidate = run_dir / "chunks" / f"{chunk_id}.txt"
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8", errors="replace")

    raise FileNotFoundError(f"Could not load chunk text for {chunk_id}. Tried: {candidate}")


def _load_optional_text(path: Path) -> Optional[str]:
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace").strip()
    return None


# -------------------------
# Default task prompts
# -------------------------

DEFAULT_TASK_ORDER = ["review", "grammar", "summary", "rewrite", "story"]

DEFAULT_TASK_INSTRUCTIONS: Dict[str, str] = {
    "review": (
        "You are reviewing a transcript chunk.\n"
        "Return:\n"
        "1) Key points (bullets)\n"
        "2) Notable quotes (verbatim, short)\n"
        "3) Any unclear spots / likely mishears (bullets)\n"
        "Keep it concise and faithful to the source.\n"
    ),
    "grammar": (
        "Clean up the text for readability.\n"
        "- Fix obvious grammar/punctuation.\n"
        "- Keep the speaker's voice.\n"
        "- Do NOT add facts.\n"
        "- Preserve proper nouns and the meaning.\n"
        "Output ONLY the corrected text.\n"
    ),
    "summary": (
        "Summarize the chunk.\n"
        "- 5-10 bullet points.\n"
        "- Include any dates/names mentioned.\n"
        "- Be faithful (no invention).\n"
    ),
    "rewrite": (
        "Rewrite the chunk as a clearer, more readable paragraph(s).\n"
        "- Keep meaning and tone.\n"
        "- Remove filler and repeated phrases.\n"
        "- Do NOT add facts.\n"
    ),
    "story": (
        "Turn the chunk into a short narrative draft.\n"
        "- Keep the events faithful.\n"
        "- You may stylize, but do NOT invent new events.\n"
        "- Use vivid but grounded language.\n"
    ),
}


def _build_prompt(
    *,
    task: str,
    chunk_id: Optional[str],
    chunk_text: str,
    source_label: str,
    task_override: Optional[str] = None,
) -> str:
    header = [
        "You are helping process a transcript.",
        f"Task: {task}",
        f"Source: {source_label}",
    ]
    if chunk_id:
        header.append(f"Chunk: {chunk_id}")

    instructions = (task_override or DEFAULT_TASK_INSTRUCTIONS.get(task) or DEFAULT_TASK_INSTRUCTIONS["review"]).strip()

    return (
        "\n".join(header)
        + "\n\n"
        + "Instructions:\n"
        + instructions
        + "\n\n"
        + "Transcript chunk:\n"
        + "-----\n"
        + chunk_text.strip()
        + "\n-----\n"
    )


# -------------------------
# Public API
# -------------------------

def materialize_prompts(
    *,
    run_dir: Path,
    tasks: Optional[Sequence[str]] = None,
    out_dir_name: str = "ai_materialized",
    ai_prep_dir_name: str = "ai_prep",
    requests_filename: str = "requests.jsonl",
    prompt_overrides_dir_name: str = "prompt_overrides",
) -> MaterializeResult:
    """
    Creates offline, ready-to-paste prompt files for each request in ai_prep/requests.jsonl.

    Outputs:
      <run_dir>/<out_dir_name>/
        manifest.json
        prompts/<task>/<task>__<chunk_id>.prompt.txt
        prompts/<task>/<task>__ALL.prompt.txt   (if chunkless)
    """
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    ai_prep_dir = run_dir / ai_prep_dir_name
    requests_path = ai_prep_dir / requests_filename
    if not requests_path.exists():
        raise FileNotFoundError(f"Missing requests file: {requests_path}")

    items = _read_jsonl(requests_path)

    if tasks:
        allowed = {_safe_slug(t) for t in tasks}
    else:
        allowed = set(DEFAULT_TASK_ORDER)

    out_dir = run_dir / out_dir_name
    prompts_root = out_dir / "prompts"
    prompts_root.mkdir(parents=True, exist_ok=True)

    # optional per-task override files:
    # <run_dir>/ai_prep/prompt_overrides/summary.txt etc
    overrides_dir = ai_prep_dir / prompt_overrides_dir_name
    override_cache: Dict[str, Optional[str]] = {}

    def get_override(task: str) -> Optional[str]:
        task_key = _safe_slug(task)
        if task_key in override_cache:
            return override_cache[task_key]
        override = _load_optional_text(overrides_dir / f"{task_key}.txt")
        override_cache[task_key] = override
        return override

    prompt_files: List[Path] = []
    manifest: Dict[str, Any] = {
        "version": 1,
        "run_dir": str(run_dir),
        "ai_prep": str(ai_prep_dir),
        "requests": str(requests_path),
        "out_dir": str(out_dir),
        "generated": [],
    }

    for obj in items:
        task = obj.get("task") or obj.get("type") or obj.get("name") or "review"
        task_key = _safe_slug(str(task))
        if task_key not in allowed:
            continue

        chunk_id = _infer_chunk_id(obj)
        chunk_text = ""
        if chunk_id:
            chunk_text = _load_chunk_text(run_dir, chunk_id, obj)
        else:
            # If the request isn't chunk-specific, allow 'text' or 'content' field
            for k in ("text", "content", "input"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    chunk_text = v
                    break

        if not chunk_text.strip():
            # Skip empty requests safely
            continue

        override = get_override(task_key)

        source_label = obj.get("source") or obj.get("source_label") or run_dir.name

        prompt = _build_prompt(
            task=task_key,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            source_label=str(source_label),
            task_override=override,
        )

        task_dir = prompts_root / task_key
        task_dir.mkdir(parents=True, exist_ok=True)

        if chunk_id:
            out_name = f"{task_key}__{chunk_id}.prompt.txt"
        else:
            out_name = f"{task_key}__ALL.prompt.txt"

        out_path = (task_dir / out_name).resolve()
        out_path.write_text(prompt, encoding="utf-8", newline="\n")
        prompt_files.append(out_path)

        manifest["generated"].append(
            {
                "task": task_key,
                "chunk_id": chunk_id,
                "prompt_path": str(out_path),
            }
        )

    manifest_path = (out_dir / "manifest.json").resolve()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")

    return MaterializeResult(out_dir=out_dir, manifest_path=manifest_path, prompt_files=prompt_files)
