from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_hotwords(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: List[str] = []
    for ln in lines:
        s = (ln or "").strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _write_hotwords(path: Path, hotwords: List[str]) -> None:
    _ensure_parent(path)
    # Preserve order, dedupe case-insensitively
    seen = set()
    cleaned: List[str] = []
    for w in hotwords:
        s = (w or "").strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)

    path.write_text("\n".join(cleaned) + ("\n" if cleaned else ""), encoding="utf-8")


def _load_corrections(path: Path) -> Dict:
    if not path.exists():
        return {"version": 1, "replacements": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Corrections JSON must be an object: {path}")
    data.setdefault("version", 1)
    data.setdefault("replacements", [])
    if not isinstance(data["replacements"], list):
        raise ValueError(f"'replacements' must be a list in: {path}")
    return data


def _default_whole_word(src: str) -> bool:
    # Whole-word by default if it's a single token (no spaces)
    s = (src or "").strip()
    return (" " not in s) and ("\t" not in s)


def _upsert_replacement(
    data: Dict,
    src: str,
    dst: str,
    *,
    whole_word: bool | None,
    case_sensitive: bool,
) -> Tuple[bool, Dict]:
    """
    Upsert by matching 'from' case-insensitively.
    Returns (created_new, replacement_obj).
    """
    src_norm = src.strip().casefold()
    repls: List[Dict] = data["replacements"]

    for item in repls:
        if not isinstance(item, dict):
            continue
        f = item.get("from")
        if isinstance(f, str) and f.strip().casefold() == src_norm:
            item["from"] = src.strip()
            item["to"] = dst
            item["whole_word"] = _default_whole_word(src) if whole_word is None else bool(whole_word)
            item["case_sensitive"] = bool(case_sensitive)
            return (False, item)

    new_item = {
        "from": src.strip(),
        "to": dst,
        "whole_word": _default_whole_word(src) if whole_word is None else bool(whole_word),
        "case_sensitive": bool(case_sensitive),
    }
    repls.append(new_item)
    return (True, new_item)


def main() -> int:
    root = _repo_root()

    ap = argparse.ArgumentParser(
        prog="teach",
        description="Teach the system a correction: adds hotwords + a deterministic replacement rule (offline).",
    )
    ap.add_argument("wrong", help="Wrong phrase observed in transcript (e.g. 'Duran Iran')")
    ap.add_argument("right", help="Correct phrase to use (e.g. 'Duran Duran')")

    ap.add_argument(
        "--hotwords",
        type=Path,
        default=root / "notes" / "hotwords.txt",
        help="Hotwords file to update (default: notes/hotwords.txt)",
    )
    ap.add_argument(
        "--corrections",
        type=Path,
        default=root / "notes" / "corrections.json",
        help="Corrections JSON to update (default: notes/corrections.json)",
    )

    ap.add_argument(
        "--no-hotword",
        action="store_true",
        help="Do not add the RIGHT phrase to hotwords",
    )
    ap.add_argument(
        "--whole-word",
        action="store_true",
        help="Force whole-word replacement (best for single words like Gizmo)",
    )
    ap.add_argument(
        "--phrase",
        action="store_true",
        help="Force phrase replacement (no word-boundary anchors; best for multi-word phrases)",
    )
    ap.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Make replacement case-sensitive (default: case-insensitive)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change but do not write files",
    )

    args = ap.parse_args()

    wrong = (args.wrong or "").strip()
    right = (args.right or "").strip()
    if not wrong:
        raise ValueError("wrong must be non-empty")
    if not right:
        raise ValueError("right must be non-empty")

    # Determine whole_word override (None = auto)
    whole_word = None
    if args.whole_word and args.phrase:
        raise ValueError("Choose only one: --whole-word OR --phrase")
    if args.whole_word:
        whole_word = True
    if args.phrase:
        whole_word = False

    hotwords_path: Path = args.hotwords.expanduser().resolve()
    corrections_path: Path = args.corrections.expanduser().resolve()

    # Load existing
    hotwords = _load_hotwords(hotwords_path)
    corr = _load_corrections(corrections_path)

    # Update hotwords with the CORRECT spelling (right)
    hotwords_added = False
    if not args.no_hotword:
        if right and right.casefold() not in {h.casefold() for h in hotwords}:
            hotwords.append(right)
            hotwords_added = True

    created, repl_obj = _upsert_replacement(
        corr,
        wrong,
        right,
        whole_word=whole_word,
        case_sensitive=args.case_sensitive,
    )

    if args.dry_run:
        print("🧪 DRY RUN")
        if hotwords_added:
            print(f"Would add hotword: {right}")
        else:
            print("Hotwords unchanged.")
        print("Would upsert replacement:")
        print(json.dumps(repl_obj, indent=2))
        return 0

    # Write changes
    if hotwords_added:
        _write_hotwords(hotwords_path, hotwords)
        print(f"✅ Hotwords updated: {hotwords_path}")
        print(f"   + {right}")
    else:
        # still ensure file exists (optional)
        if not hotwords_path.exists():
            _write_hotwords(hotwords_path, hotwords)
            print(f"✅ Hotwords created: {hotwords_path}")

    _ensure_parent(corrections_path)
    corrections_path.write_text(json.dumps(corr, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✅ Corrections updated: {corrections_path}")
    print(f"   {'+ ' if created else '~ '} {wrong}  ->  {right}")

    # Next run suggestion (works with your wrapper)
    print("\nNext run suggestion:")
    print(
        "  transcribe <name_or_path> "
        "--hotwords-file notes/hotwords.txt "
        "--post-fix notes/corrections.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
