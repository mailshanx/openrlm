"""Agent loop event types emitted by _run_turn for consumer observation.

Events signal state transitions in the agent loop. They are informational —
they do not alter control flow, interrupt logic, or the agent's return value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoundStart:
    """Emitted at the start of each LLM→tool round within a turn."""
    agent_id: str
    round_num: int
    max_rounds: int


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Emitted before calling the LLM."""
    agent_id: str


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Emitted after the LLM responds."""
    agent_id: str
    model: str
    finish_reason: str
    has_tool_calls: bool
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True, slots=True)
class ToolExecStart:
    """Emitted before executing a tool call."""
    agent_id: str
    tool_name: str
    code: str


@dataclass(frozen=True, slots=True)
class ToolExecEnd:
    """Emitted after tool execution completes."""
    agent_id: str
    tool_name: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class TurnEnd:
    """Emitted when _run_turn completes (final response or max rounds)."""
    agent_id: str
    rounds: int
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int


AgentEvent = RoundStart | ModelRequest | ModelResponse | ToolExecStart | ToolExecEnd | TurnEnd
