from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agents.contracts import AgentContext, AgentResult
from agents.hello_agent import HelloAgent

# ---------------------------------------------------------------------
# Registry (sandbox-simple)
# ---------------------------------------------------------------------

AGENT_REGISTRY = {
    "hello": HelloAgent(),
}


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------


def _repo_root() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def _log_path() -> Path:
    return Path(_repo_root()) / "notes" / "agent_runs.ndjson"


def run_agent(agent_name: str, payload: Dict[str, Any]) -> AgentResult:
    agent = AGENT_REGISTRY.get(agent_name)
    if agent is None:
        available = ", ".join(AGENT_REGISTRY.keys())
        raise SystemExit(f"Unknown agent '{agent_name}'. Available: {available}")

    ctx = AgentContext(repo_root=_repo_root(), sandbox=True)
    return agent.run(input=payload, ctx=ctx)


def _parse_input_json(input_json: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(input_json)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON for --input: {e}", file=sys.stderr)
        raise SystemExit(2)

    if not isinstance(parsed, dict):
        print(
            "Invalid --input: must be a JSON object (e.g. '{\"name\":\"Jimmy\"}')",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return parsed


def _append_ndjson_log(*, agent: str, payload: Dict[str, Any], result: AgentResult) -> None:
    """
    Append a single NDJSON event line to notes/agent_runs.ndjson.
    """
    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "payload": payload,
        "ok": result.ok,
        "summary": result.summary,
        "data": result.data,
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _tail_lines(path: Path, n: int) -> List[str]:
    """
    Tail the last n lines of a text file without loading the whole file.
    Simple + robust for small/medium logs.
    """
    if n <= 0:
        return []

    if not path.exists():
        return []

    # Read in blocks from the end until we have enough lines.
    # NDJSON lines are UTF-8; we decode per block with replacement to be safe.
    block_size = 8192
    data = bytearray()
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        pos = file_size

        while pos > 0 and data.count(b"\n") <= n:
            read_size = block_size if pos >= block_size else pos
            pos -= read_size
            f.seek(pos)
            data[:0] = f.read(read_size)

    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()

    return lines[-n:]


def _cmd_log_tail(tail: int) -> int:
    """
    Print the last N NDJSON entries (as-is, one per line).
    """
    tail = max(1, tail)
    path = _log_path()
    lines = _tail_lines(path, tail)

    if not lines:
        print(f"No log entries found at: {path}")
        return 0

    for line in lines:
        print(line)

    return 0


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Codex Sandbox Agent Runner")

    subparsers = parser.add_subparsers(dest="cmd")

    # log command
    log_parser = subparsers.add_parser("log", help="Inspect agent run logs")
    log_parser.add_argument("--tail", type=int, default=20, help="Print last N log lines (default: 20)")

    # default: run agent
    parser.add_argument("agent", nargs="?", help=f"Agent name. Options: {', '.join(AGENT_REGISTRY)}")
    parser.add_argument(
        "--name",
        default="friend",
        help="Convenience input (used if payload has no name)",
    )
    parser.add_argument(
        "--input",
        dest="input_json",
        help="JSON object payload (e.g. '{\"name\":\"Jimmy\"}')",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable NDJSON run logging to notes/agent_runs.ndjson",
    )

    args = parser.parse_args()

    # Handle subcommand: log
    if args.cmd == "log":
        return _cmd_log_tail(args.tail)

    # Otherwise: run an agent
    if not args.agent:
        parser.print_help()
        return 2

    payload: Dict[str, Any] = {}
    if args.input_json:
        payload.update(_parse_input_json(args.input_json))

    payload.setdefault("name", args.name)

    result = run_agent(args.agent, payload)

    if not args.no_log:
        _append_ndjson_log(agent=args.agent, payload=payload, result=result)

    if args.json:
        print(json.dumps(result.__dict__, indent=2))
    else:
        print(result.summary)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
