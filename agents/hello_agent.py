from __future__ import annotations

from typing import Dict

from .contracts import AgentContext, AgentResult


class HelloAgent:
    agent_id = "hello"

    def run(self, *, input: Dict[str, object], ctx: AgentContext) -> AgentResult:
        name = str(input.get("name", "friend"))
        mode = "SANDBOX" if ctx.sandbox else "PROD"
        message = f"Hello, {name}. Agent online in {mode}. Repo: {ctx.repo_root}"

        return AgentResult(
            ok=True,
            agent_id=self.agent_id,
            summary=message,
            data={"name": name, "mode": mode, "repo_root": ctx.repo_root},
        )
