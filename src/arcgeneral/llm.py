"""LLM client protocol and response types.
Defines the contract between AgentRuntime and any LLM provider.
AgentRuntime only depends on these types — never on provider SDK types directly.
"""

from __future__ import annotations

import json as _json
import os
from pathlib import Path
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
        # Token file takes highest priority (written by pi extension, refreshed on interval)
        if provider == "anthropic":
            token_file = os.environ.get("ARCGENERAL_TOKEN_FILE")
            if token_file:
                try:
                    key = Path(token_file).read_text().strip()
                    if key:
                        return key
                except OSError:
                    pass  # Fall through to env vars
            key = os.environ.get("ANTHROPIC_OAUTH_TOKEN") or os.environ.get(env_var, "")
        else:
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

# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class AnthropicClient:
    """LLMClient backed by the Anthropic SDK.

    Translates between OpenAI-shaped messages (used internally by _run_turn)
    and the Anthropic Messages API format.
    """

    DEFAULT_MAX_TOKENS = 16384

    def __init__(self):
        self._client = None  # lazily created
        self._is_oauth = False
        self._current_key = None
        self._stale_client = None

    # Stealth headers matching Pi's Anthropic OAuth implementation.
    # Required for OAuth tokens (sk-ant-oat-*) to be accepted by Anthropic's API.
    _OAUTH_HEADERS = {
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
        "user-agent": "claude-cli/2.1.2 (external, cli)",
        "x-app": "cli",
    }
    _OAUTH_SYSTEM_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
    def _ensure_client(self, api_key: str):
        import anthropic
        if self._client is not None and self._current_key == api_key:
            return self._client
        # Key changed or first call — (re)create client
        if self._client is not None:
            self._stale_client = self._client  # defer close to async close()
        self._current_key = api_key
        if "sk-ant-oat" in api_key:
            self._client = anthropic.AsyncAnthropic(
                api_key=None,
                auth_token=api_key,
                default_headers=self._OAUTH_HEADERS,
            )
            self._is_oauth = True
        else:
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            self._is_oauth = False
        return self._client

    async def complete(self, messages: list[dict], *, api_key: str, **kwargs) -> CompletionResponse:
        client = self._ensure_client(api_key)

        # Extract system message from the list — Anthropic takes it as a separate param
        system_content = None
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg["content"]
            else:
                api_messages.append(msg)

        # Translate messages from OpenAI format to Anthropic format
        anthropic_messages = self._translate_messages(api_messages)

        # Translate tools
        anthropic_tools = None
        if "tools" in kwargs:
            anthropic_tools = [self._translate_tool(t) for t in kwargs.pop("tools")]

        # Build request
        model = kwargs.pop("model", "claude-sonnet-4-5-20250514")
        max_tokens = kwargs.pop("max_tokens", self.DEFAULT_MAX_TOKENS)
        temperature = kwargs.pop("temperature", None)

        create_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if self._is_oauth:
            # OAuth requires Claude Code identity as first system block
            blocks = [{"type": "text", "text": self._OAUTH_SYSTEM_IDENTITY}]
            if system_content:
                blocks.append({"type": "text", "text": system_content})
            create_kwargs["system"] = blocks
        elif system_content is not None:
            create_kwargs["system"] = system_content
        if anthropic_tools is not None:
            create_kwargs["tools"] = anthropic_tools
        if temperature is not None:
            create_kwargs["temperature"] = temperature

        raw = await client.messages.create(**create_kwargs)

        # Translate response back to our CompletionResponse
        return self._translate_response(raw)

    @staticmethod
    def _translate_tool(openai_tool: dict) -> dict:
        """Convert OpenAI tool format to Anthropic tool format."""
        fn = openai_tool["function"]
        return {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn["parameters"],
        }

    @staticmethod
    def _translate_messages(messages: list[dict]) -> list[dict]:
        """Convert OpenAI-shaped messages to Anthropic-shaped messages.

        Key differences:
        - Assistant tool_calls become content blocks (text + tool_use)
        - Tool result messages (role=tool) become user messages with tool_result blocks
        - Consecutive same-role messages are merged (Anthropic rejects them)
        """
        result: list[dict] = []

        for msg in messages:
            role = msg.get("role")

            if role == "assistant":
                content_blocks = []
                # Add text if present
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                # Convert tool_calls to tool_use blocks
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        fn = tc["function"]
                        try:
                            input_obj = _json.loads(fn["arguments"])
                        except (ValueError, TypeError):
                            input_obj = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": fn["name"],
                            "input": input_obj,
                        })
                anthropic_msg = {"role": "assistant", "content": content_blocks}

            elif role == "tool":
                # Tool results become a user message with tool_result content block
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg.get("content", ""),
                }
                anthropic_msg = {"role": "user", "content": [block]}

            elif role == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    anthropic_msg = {"role": "user", "content": content}
                else:
                    anthropic_msg = {"role": "user", "content": content}

            else:
                # Pass through unknown roles
                anthropic_msg = msg

            # Merge consecutive same-role messages
            if result and result[-1]["role"] == anthropic_msg["role"]:
                prev = result[-1]
                # Normalize both to lists of content blocks for merging
                prev_content = prev["content"]
                new_content = anthropic_msg["content"]
                if isinstance(prev_content, str):
                    prev_content = [{"type": "text", "text": prev_content}]
                if isinstance(new_content, str):
                    new_content = [{"type": "text", "text": new_content}]
                prev["content"] = prev_content + new_content
            else:
                result.append(anthropic_msg)

        return result

    @staticmethod
    def _translate_response(raw) -> CompletionResponse:
        """Convert an Anthropic Message to our CompletionResponse."""
        text_parts = []
        tool_calls = []

        for block in raw.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    function=ToolCallFunction(
                        name=block.name,
                        arguments=_json.dumps(block.input),
                    ),
                ))

        content = "\n".join(text_parts) if text_parts else None
        tc_list = tool_calls if tool_calls else None

        usage = TokenUsage(
            prompt_tokens=raw.usage.input_tokens,
            completion_tokens=raw.usage.output_tokens,
        )

        return CompletionResponse(
            model=raw.model,
            choices=[CompletionChoice(
                message=CompletionMessage(content=content, tool_calls=tc_list),
                finish_reason=raw.stop_reason,
            )],
            usage=usage,
        )

    async def close(self) -> None:
        if self._stale_client:
            await self._stale_client.close()
            self._stale_client = None
        if self._client is not None:
            await self._client.close()
            self._client = None