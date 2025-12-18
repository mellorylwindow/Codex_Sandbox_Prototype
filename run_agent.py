from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from agents.contracts import AgentContext, AgentResult
from agents.hello_agent import HelloAgent


AGENT_REGISTRY = {
    "hello": HelloAgent(),
}


def _repo_root() -> str:
    return os.path.abspath(os.path.dirname(__file__))


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Codex Sandbox Agent Runner")
    parser.add_argument("agent", help=f"Agent name. Options: {', '.join(AGENT_REGISTRY)}")
    parser.add_argument("--name", default="friend", help="Convenience input (used if payload has no name)")
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    parser.add_argument(
        "--input",
        dest="input_json",
        help="JSON object payload (e.g. '{\"name\":\"Jimmy\"}')",
    )

    args = parser.parse_args()

    payload: Dict[str, Any] = {}

    if args.input_json:
        payload.update(_parse_input_json(args.input_json))

    # Default name if not provided in JSON payload
    payload.setdefault("name", args.name)

    result = run_agent(args.agent, payload)

    if args.json:
        print(json.dumps(result.__dict__, indent=2))
    else:
        print(result.summary)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
