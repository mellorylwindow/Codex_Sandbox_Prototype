from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


def _find_first(parent: Path, names: list[str]) -> Path | None:
    for n in names:
        p = parent / n
        if p.exists():
            return p
    return None


def _iter_user_text_from_conversations_json(p: Path) -> Iterable[str]:
    data = json.loads(p.read_text(encoding="utf-8", errors="replace"))

    # Export formats vary. Commonly: list of conversations, each with "mapping" of nodes.
    if isinstance(data, dict) and "conversations" in data and isinstance(data["conversations"], list):
        conversations = data["conversations"]
    elif isinstance(data, list):
        conversations = data
    else:
        conversations = [data]

    for conv in conversations:
        mapping = conv.get("mapping") if isinstance(conv, dict) else None
        if not isinstance(mapping, dict):
            continue

        for _node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue

            author = msg.get("author") or {}
            role = author.get("role") if isinstance(author, dict) else None
            if role != "user":
                continue

            content = msg.get("content") or {}
            parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(parts, list):
                text = "\n".join(str(x) for x in parts if x is not None).strip()
                if text:
                    yield text


def _extract_user_text_from_chat_html(p: Path) -> list[str]:
    """
    Fallback: export includes chat.html per OpenAI doc. :contentReference[oaicite:2]{index=2}
    HTML structure can change; this is a best-effort heuristic.
    """
    html = p.read_text(encoding="utf-8", errors="replace")

    # Heuristic: many exports include markers for user turns.
    # We grab blocks that contain "You:" labels or role=user-ish hints.
    # If this misses content, use conversations.json when available.
    chunks: list[str] = []

    # Try common "You" label blocks
    for m in re.finditer(r"(?:^|\n)\s*You:\s*(.+?)(?=\n\S+:\s|\Z)", html, flags=re.DOTALL):
        txt = re.sub(r"<[^>]+>", "", m.group(1))
        txt = re.sub(r"\s+\n", "\n", txt).strip()
        if txt:
            chunks.append(txt)

    return chunks


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/extract_my_messages.py <export_folder>")
        return 2

    export_dir = Path(sys.argv[1]).expanduser().resolve()
    if not export_dir.exists():
        print(f"Not found: {export_dir}")
        return 2

    out_dir = Path("out/my_corpus").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conversations = None
    # Common export filenames
    for cand in ["conversations.json", "chatgpt-conversations.json", "conversations.jsonl"]:
        p = export_dir / cand
        if p.exists():
            conversations = p
            break

    chat_html = _find_first(export_dir, ["chat.html"])

    texts: list[str] = []
    if conversations and conversations.suffix == ".json":
        texts = list(_iter_user_text_from_conversations_json(conversations))
        print(f"✅ Extracted from {conversations.name}: {len(texts)} user messages")
    elif chat_html:
        texts = _extract_user_text_from_chat_html(chat_html)
        print(f"✅ Extracted (heuristic) from chat.html: {len(texts)} user messages")
    else:
        print("Could not find conversations.json or chat.html in that folder.")
        return 2

    # Write outputs
    (out_dir / "my_messages.txt").write_text("\n\n---\n\n".join(texts), encoding="utf-8")
    with (out_dir / "my_messages.jsonl").open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"role": "user", "text": t}, ensure_ascii=False) + "\n")

    print(f"📝 Wrote: {out_dir / 'my_messages.txt'}")
    print(f"📝 Wrote: {out_dir / 'my_messages.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
