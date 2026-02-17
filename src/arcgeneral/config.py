from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcgeneral.host_functions import HostFunctionRegistry


@dataclass
class AgentConfig:
    model: str = "openai/gpt-4o"
    api_key_env_var: str = "OPENROUTER_API_KEY"
    system_prompt: str | None = None
    sandbox_image: str | None = None
    code_timeout: float = 3600.0
    max_tool_rounds: int = 50
    max_sub_agent_depth: int = 10
    output_limit_lines: int = 2000
    output_limit_bytes: int = 50_000
    temperature: float | None = None
    sandbox_binds: dict[str, str] = field(default_factory=dict)
    workspace_path: str | None = None  # resolved at runtime; shown in system prompt
    spool_path: str | None = None        # resolved at runtime; shown in system prompt
