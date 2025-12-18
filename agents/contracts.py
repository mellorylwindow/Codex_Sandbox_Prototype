from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class AgentContext:
    repo_root: str
    sandbox: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    ok: bool
    agent_id: str
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    agent_id: str

    def run(self, *, input: Dict[str, Any], ctx: AgentContext) -> AgentResult:
        ...
