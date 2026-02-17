"""LLM client protocol and response types.

Defines the contract between AgentRuntime and any LLM provider.
AgentRuntime only depends on these types — never on provider SDK types directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ToolCallFunction:
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    function: ToolCallFunction


@dataclass(frozen=True, slots=True)
class CompletionMessage:
    content: str | None
    tool_calls: list[ToolCall] | None


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class CompletionChoice:
    message: CompletionMessage
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    model: str
    choices: list[CompletionChoice]
    usage: TokenUsage | None


class LLMClient(Protocol):
    async def complete(self, messages: list[dict], **kwargs) -> CompletionResponse: ...
    async def close(self) -> None: ...


class OpenRouterClient:
    """LLMClient backed by the OpenRouter SDK."""

    def __init__(self, api_key: str):
        import httpx
        from openrouter import OpenRouter

        self._http_client = httpx.AsyncClient(http2=True, follow_redirects=True)
        self._sdk = OpenRouter(api_key=api_key, async_client=self._http_client)

    async def complete(self, messages: list[dict], **kwargs) -> CompletionResponse:
        raw = await self._sdk.chat.send_async(messages=messages, **kwargs)
        choice = raw.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    function=ToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments or "{}",
                    ),
                )
                for tc in msg.tool_calls
            ]

        usage = None
        if raw.usage:
            usage = TokenUsage(
                prompt_tokens=int(raw.usage.prompt_tokens),
                completion_tokens=int(raw.usage.completion_tokens),
            )

        # content can be str, list of content items, or None/UNSET
        content = msg.content if isinstance(msg.content, str) else None

        return CompletionResponse(
            model=raw.model,
            choices=[CompletionChoice(
                message=CompletionMessage(content=content, tool_calls=tool_calls),
                finish_reason=choice.finish_reason,
            )],
            usage=usage,
        )

    async def close(self) -> None:
        await self._http_client.aclose()
