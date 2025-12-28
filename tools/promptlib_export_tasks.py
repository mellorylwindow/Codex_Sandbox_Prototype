from __future__ import annotations

import json
from pathlib import Path

LIB_PATH = Path("notes/prompt_library/velvet_prompts.tax_moneymaximizer.json")
OUT_DIR_DEFAULT = Path("notes/prompt_library/exported_tasks")

def main(out_dir: str | None = None) -> None:
    lib = json.loads(LIB_PATH.read_text(encoding="utf-8"))
    out_base = Path(out_dir) if out_dir else OUT_DIR_DEFAULT
    out_base.mkdir(parents=True, exist_ok=True)

    written = 0
    for group in lib.get("groups", []):
        gname = group.get("group", "misc")
        gdir = out_base / gname
        gdir.mkdir(parents=True, exist_ok=True)

        for item in group.get("items", []):
            slug = item["slug"]
            out_path = gdir / f"{slug}.json"
            out_path.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written += 1

    print(f"Wrote {written} task files to: {out_base}")

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else None
    main(out)
