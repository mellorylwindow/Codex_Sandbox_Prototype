# tools/help.py
from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMANDS_MD = REPO_ROOT / "notes" / "COMMANDS.md"


def main() -> int:
    if not COMMANDS_MD.exists():
        print(f"ERROR: {COMMANDS_MD} not found.")
        print("Tip: create it first (see notes/COMMANDS.md workflow).")
        return 2

    txt = COMMANDS_MD.read_text(encoding="utf-8", errors="replace")

    # If user pipes to less, keep it clean (no extra framing).
    # Otherwise give a tiny header.
    if sys.stdout.isatty():
        print("=== Codex Sandbox: COMMANDS.md ===")
        print(f"Path: {COMMANDS_MD}")
        print("")

    print(txt.rstrip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
