"""Event types and dispatch utilities for consumer observation.

Events signal state transitions in the agent loop and sub-agent lifecycle.
They are informational — they do not alter control flow, interrupt logic,
or the agent's return value.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Callable

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

EventCallback = Callable[[AgentEvent], None]
"""Sync callback signature for event consumers."""


logger = logging.getLogger(__name__)


class EventBus:
    """Fan-out dispatcher for agent events.

    Presents a single sync callback to the engine via :meth:`callback`.
    Distributes events to an arbitrary number of sync listeners and
    async :class:`EventStream` consumers.
    """

    def __init__(self) -> None:
        self._listeners: list[EventCallback] = []
        self._streams: list[asyncio.Queue[AgentEvent | None]] = []

    def callback(self, event: AgentEvent) -> None:
        """Sync callback passed as *on_event* to :class:`Session`.

        Dispatches *event* to every registered listener and stream.
        Errors in individual listeners are swallowed with a debug log.
        Full queues cause the event to be dropped for that stream.
        """
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                logger.debug("EventBus listener error", exc_info=True)
        for q in self._streams:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("EventStream queue full, dropping event")

    def add_listener(self, fn: EventCallback) -> None:
        """Register a sync callback invoked for every event."""
        self._listeners.append(fn)

    def remove_listener(self, fn: EventCallback) -> None:
        """Remove a previously registered sync callback.

        Raises ValueError if *fn* was never added.
        """
        self._listeners.remove(fn)

    def stream(self, maxsize: int = 0) -> EventStream:
        """Create an async iterator over events.

        Each stream gets its own queue with independent backpressure
        and consumption rate.  The engine never blocks.
        """
        q: asyncio.Queue[AgentEvent | None] = asyncio.Queue(maxsize=maxsize)
        self._streams.append(q)
        return EventStream(q, self)

    def close(self) -> None:
        """Signal all streams to stop iterating.  Idempotent."""
        for q in self._streams:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass


class EventStream:
    """Async iterator over events from an :class:`EventBus`.

    Created by :meth:`EventBus.stream`.  Supports ``async for event in stream``.
    """

    def __init__(self, queue: asyncio.Queue[AgentEvent | None], bus: EventBus) -> None:
        self._queue = queue
        self._bus = bus

    def __aiter__(self) -> EventStream:
        return self

    async def __anext__(self) -> AgentEvent:
        event = await self._queue.get()
        if event is None:
            raise StopAsyncIteration
        return event

    def close(self) -> None:
        """Detach this stream from its bus and stop iteration."""
        try:
            self._bus._streams.remove(self._queue)
        except ValueError:
            pass
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
