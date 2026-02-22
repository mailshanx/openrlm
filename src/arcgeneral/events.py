"""Event types emitted by Session for consumer observation.

Events signal state transitions in the agent loop and sub-agent lifecycle.
They are informational — they do not alter control flow, interrupt logic,
or the agent's return value.
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

# ── Agent tree lifecycle events ──

@dataclass(frozen=True, slots=True)
class AgentCreated:
    """Emitted when a sub-agent is registered in the session's agent tree."""
    agent_id: str
    parent_id: str
    depth: int


@dataclass(frozen=True, slots=True)
class TaskStarted:
    """Emitted when a task begins on a sub-agent."""
    agent_id: str
    task_id: str
    task: str


@dataclass(frozen=True, slots=True)
class TaskCompleted:
    """Emitted when a sub-agent task finishes."""
    agent_id: str
    task_id: str

AgentEvent = (RoundStart | ModelRequest | ModelResponse | ToolExecStart | ToolExecEnd | TurnEnd
            | AgentCreated | TaskStarted | TaskCompleted)
