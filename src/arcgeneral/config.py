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
    sandbox_image: str = "arcgeneral:sandbox"
    code_timeout: float = 120.0
    max_tool_rounds: int = 50
    temperature: float | None = None
    sandbox_binds: dict[str, str] = field(default_factory=dict)
    host_functions: HostFunctionRegistry | None = None
