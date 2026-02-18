"""LLM client protocol and response types.

Defines the contract between AgentRuntime and any LLM provider.
AgentRuntime only depends on these types — never on provider SDK types directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol


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
    async def complete(self, messages: list[dict], *, api_key: str, **kwargs) -> CompletionResponse: ...
    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Provider → environment variable mapping
# ---------------------------------------------------------------------------

PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic":    "ANTHROPIC_API_KEY",
    "openai":       "OPENAI_API_KEY",
    "google":       "GEMINI_API_KEY",
    "openrouter":   "OPENROUTER_API_KEY",
    "groq":         "GROQ_API_KEY",
    "xai":          "XAI_API_KEY",
    "mistral":      "MISTRAL_API_KEY",
}


def default_api_key_resolver() -> Callable[[str], Awaitable[str]]:
    """Build a resolver that reads API keys from environment variables.

    Returns an async callable: (provider) -> api_key.
    The provider argument is used to look up the correct env var.
    """
    async def resolve(provider: str) -> str:
        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var is None:
            # Unknown provider — try PROVIDER_API_KEY convention
            env_var = f"{provider.upper().replace('-', '_')}_API_KEY"
        key = os.environ.get(env_var, "")
        if not key:
            raise ValueError(
                f"No API key for provider {provider!r}. "
                f"Set the {env_var} environment variable."
            )
        return key
    return resolve


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

class OpenRouterClient:
    """LLMClient backed by the OpenRouter SDK. Stateless w.r.t. auth — key is passed per-call."""

    def __init__(self):
        import httpx
        self._http_client = httpx.AsyncClient(http2=True, follow_redirects=True)

    async def complete(self, messages: list[dict], *, api_key: str, **kwargs) -> CompletionResponse:
        from openrouter import OpenRouter

        sdk = OpenRouter(api_key=api_key, async_client=self._http_client)
        raw = await sdk.chat.send_async(messages=messages, **kwargs)
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
