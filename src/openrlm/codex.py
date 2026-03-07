"""Codex Responses API client.

Implements LLMClient for OpenAI Codex models (gpt-5.1, gpt-5.2, gpt-5.3 and
their codex-mini/max/spark variants) accessed via the ChatGPT backend API.

Translates between openrlm's internal Chat Completions message format and
the Responses API wire format.  Captures reasoning blocks, message IDs, and
tool call ID pairs as provider_metadata for verbatim replay on subsequent
rounds (required for prompt caching and conversation consistency).

Imports only from llm.py (protocol types) and the standard library.
No coupling to the engine.
"""

from __future__ import annotations

import base64
import json
import logging
import platform
import re
from dataclasses import dataclass

import httpx

from openrlm.llm import (
    CompletionChoice,
    CompletionMessage,
    CompletionResponse,
    TokenUsage,
    ToolCall,
    ToolCallFunction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
MAX_RETRIES = 2  # within a single complete() call; engine retries on top
BASE_DELAY_S = 1.0

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_TEXT_RE = re.compile(
    r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Provider metadata item types (stored on assistant messages, opaque to engine)
# ---------------------------------------------------------------------------

_META_REASONING = "reasoning"       # full reasoning item dict (with encrypted_content)
_META_MESSAGE_ID = "message_id"     # {"type": "message_id", "id": str}
_META_FC_IDS = "function_call_ids"  # {"type": "function_call_ids", "call_id": str, "id": str}


# ---------------------------------------------------------------------------
# CodexClient
# ---------------------------------------------------------------------------

class CodexClient:
    """LLMClient backed by the OpenAI Codex Responses API.

    Translates between OpenAI Chat Completions-shaped messages (used
    internally by the engine) and the Responses API format.

    Provider-specific settings are constructor arguments, not engine config:
        reasoning_effort:  none | minimal | low | medium | high | xhigh
        reasoning_summary: auto | concise | detailed | off | on
        text_verbosity:    low | medium | high
        session_id:        optional prompt cache key
    """

    def __init__(
        self,
        *,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "auto",
        text_verbosity: str = "medium",
        session_id: str | None = None,
    ):
        self._reasoning_effort = reasoning_effort
        self._reasoning_summary = reasoning_summary
        self._text_verbosity = text_verbosity
        self._session_id = session_id
        self._http: httpx.AsyncClient | None = None

    def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(http2=True, follow_redirects=True, timeout=300.0)
        return self._http

    # ── LLMClient protocol ──

    async def complete(
        self,
        messages: list[dict],
        *,
        api_key: str,
        **kwargs,
    ) -> CompletionResponse:
        model = kwargs.pop("model", "gpt-5.2")
        tools = kwargs.pop("tools", None)
        temperature = kwargs.pop("temperature", None)

        account_id = self._extract_account_id(api_key)
        headers = self._build_headers(account_id, api_key)
        system_prompt, input_items = self._translate_messages(messages)
        codex_tools = self._translate_tools(tools) if tools else None

        body = self._build_request_body(
            model=model,
            instructions=system_prompt,
            input_items=input_items,
            tools=codex_tools,
            temperature=temperature,
        )

        url = self._resolve_url()
        http = self._ensure_http()

        # Retry loop for transient HTTP errors
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await http.post(url, headers=headers, json=body, timeout=300.0)
                if resp.status_code == 200:
                    return self._parse_sse_response(resp, model)

                error_text = resp.text
                if attempt < MAX_RETRIES and self._is_retryable(resp.status_code, error_text):
                    delay = BASE_DELAY_S * (2 ** attempt)
                    logger.warning(
                        "Codex API returned %d (attempt %d/%d), retrying in %.1fs",
                        resp.status_code, attempt + 1, MAX_RETRIES + 1, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
                    continue

                # Non-retryable or final attempt — parse error for friendly message
                raise self._build_error(resp.status_code, error_text)

            except httpx.HTTPError as e:
                last_exc = e
                if attempt < MAX_RETRIES:
                    delay = BASE_DELAY_S * (2 ** attempt)
                    logger.warning(
                        "Codex HTTP error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, MAX_RETRIES + 1, delay, e,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(f"Codex API request failed after {MAX_RETRIES + 1} attempts: {e}") from e

        # Should not reach here, but satisfy the type checker
        raise last_exc or RuntimeError("Codex API request failed")  # pragma: no cover

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ── JWT / Auth ──

    @staticmethod
    def _extract_account_id(token: str) -> str:
        """Extract chatgpt_account_id from JWT claims."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT: expected 3 dot-separated parts")
        # JWT base64url → standard base64
        payload_b64 = parts[1]
        # Add padding
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_bytes)
        auth_claim = payload.get(JWT_CLAIM_PATH)
        if not isinstance(auth_claim, dict):
            raise ValueError(f"JWT missing '{JWT_CLAIM_PATH}' claim")
        account_id = auth_claim.get("chatgpt_account_id")
        if not account_id:
            raise ValueError("JWT missing chatgpt_account_id in auth claim")
        return account_id

    @staticmethod
    def _build_headers(account_id: str, token: str) -> dict[str, str]:
        """Build request headers for the Codex API."""
        system = platform.system().lower()
        release = platform.release()
        arch = platform.machine()
        return {
            "Authorization": f"Bearer {token}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "openrlm",
            "User-Agent": f"openrlm ({system} {release}; {arch})",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }

    # ── Request building ──

    def _build_request_body(
        self,
        *,
        model: str,
        instructions: str | None,
        input_items: list,
        tools: list | None,
        temperature: float | None,
    ) -> dict:
        body: dict = {
            "model": model,
            "store": False,
            "stream": True,
            "input": input_items,
            "text": {"verbosity": self._text_verbosity},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        if instructions is not None:
            body["instructions"] = instructions
        if tools:
            body["tools"] = tools
        if temperature is not None:
            body["temperature"] = temperature
        if self._session_id is not None:
            body["prompt_cache_key"] = self._session_id
        if self._reasoning_effort is not None:
            body["reasoning"] = {
                "effort": self._reasoning_effort,
                "summary": self._reasoning_summary,
            }
        return body

    @staticmethod
    def _resolve_url(base_url: str | None = None) -> str:
        raw = base_url or DEFAULT_CODEX_BASE_URL
        normalized = raw.rstrip("/")
        if normalized.endswith("/codex/responses"):
            return normalized
        if normalized.endswith("/codex"):
            return f"{normalized}/responses"
        return f"{normalized}/codex/responses"

    # ── Message translation (internal → Responses API) ──

    @staticmethod
    def _translate_messages(messages: list[dict]) -> tuple[str | None, list]:
        """Convert internal Chat Completions messages to Responses API input.

        Returns (system_prompt, input_items).  System prompt goes to the
        ``instructions`` field on the request body, not into the input array.
        """
        system_prompt: str | None = None
        input_items: list = []
        msg_counter = 0

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                system_prompt = msg.get("content")

            elif role == "user":
                input_items.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": msg.get("content", "")}],
                })

            elif role == "assistant":
                metadata = msg.get("_provider_metadata")
                _translate_assistant(msg, metadata, input_items, msg_counter)
                msg_counter += 1

            elif role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        return system_prompt, input_items

    @staticmethod
    def _translate_tools(tools: list[dict]) -> list[dict]:
        """Convert OpenAI Chat Completions tool schemas to Responses API format.

        Input:  {"type": "function", "function": {"name": ..., "parameters": ...}}
        Output: {"type": "function", "name": ..., "parameters": ..., "strict": null}
        """
        result = []
        for tool in tools:
            fn = tool.get("function", tool)
            result.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
                "strict": None,
            })
        return result

    # ── SSE parsing and response materialization ──

    def _parse_sse_response(self, resp: httpx.Response, model: str) -> CompletionResponse:
        """Parse SSE response body synchronously (response already received)."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        provider_metadata: list[dict] = []
        usage: TokenUsage | None = None
        finish_reason: str | None = None

        # Track in-progress items for delta accumulation
        current_fc_args: dict[str, str] = {}  # call_id → accumulated arguments JSON

        for event in self._iter_sse_events(resp.text):
            event_type = event.get("type")
            if not event_type:
                continue

            # ── Error events ──
            if event_type == "error":
                code = event.get("code", "")
                message = event.get("message", "")
                raise RuntimeError(f"Codex stream error: {message or code or json.dumps(event)}")

            if event_type == "response.failed":
                err_msg = _nested_get(event, "response", "error", "message")
                raise RuntimeError(err_msg or "Codex response failed")

            # ── Text deltas (accumulate) ──
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    text_parts.append(delta)

            # ── Function call argument deltas ──
            if event_type == "response.function_call_arguments.delta":
                call_id = event.get("call_id", "")
                delta = event.get("delta", "")
                current_fc_args[call_id] = current_fc_args.get(call_id, "") + delta

            if event_type == "response.function_call_arguments.done":
                call_id = event.get("call_id", "")
                current_fc_args[call_id] = event.get("arguments", current_fc_args.get(call_id, "{}"))

            # ── Finalized output items ──
            if event_type == "response.output_item.done":
                item = event.get("item", {})
                item_type = item.get("type")

                if item_type == "reasoning":
                    # Store the entire reasoning item for verbatim replay
                    provider_metadata.append(item)

                elif item_type == "message":
                    # Extract text content and store message ID
                    item_id = item.get("id")
                    if item_id:
                        provider_metadata.append({"type": _META_MESSAGE_ID, "id": item_id})
                    for content_block in item.get("content", []):
                        if content_block.get("type") == "output_text":
                            # Use the finalized text, replacing any accumulated deltas
                            final_text = content_block.get("text", "")
                            if final_text:
                                # Replace delta-accumulated text with the finalized version
                                text_parts.clear()
                                text_parts.append(final_text)
                        elif content_block.get("type") == "refusal":
                            refusal = content_block.get("refusal", "")
                            if refusal:
                                text_parts.clear()
                                text_parts.append(refusal)

                elif item_type == "function_call":
                    call_id = item.get("call_id", "")
                    item_id = item.get("id", "")
                    name = item.get("name", "")
                    # Use finalized arguments from the item, falling back to accumulated deltas
                    arguments = item.get("arguments", current_fc_args.get(call_id, "{}"))

                    tool_calls.append(ToolCall(
                        id=call_id,
                        function=ToolCallFunction(name=name, arguments=arguments),
                    ))
                    # Store the ID pair for replay
                    provider_metadata.append({
                        "type": _META_FC_IDS,
                        "call_id": call_id,
                        "id": item_id,
                    })

            # ── Completion (usage + status) ──
            if event_type in ("response.completed", "response.done"):
                response_obj = event.get("response", {})
                raw_usage = response_obj.get("usage", {})
                if raw_usage:
                    cached = 0
                    details = raw_usage.get("input_tokens_details")
                    if isinstance(details, dict):
                        cached = details.get("cached_tokens", 0)
                    prompt_tokens = raw_usage.get("input_tokens", 0)
                    completion_tokens = raw_usage.get("output_tokens", 0)
                    usage = TokenUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                status = response_obj.get("status")
                finish_reason = _map_stop_reason(status)

        content = "".join(text_parts) if text_parts else None
        tc_list = tool_calls if tool_calls else None
        meta = provider_metadata if provider_metadata else None

        return CompletionResponse(
            model=model,
            choices=[CompletionChoice(
                message=CompletionMessage(
                    content=content,
                    tool_calls=tc_list,
                    provider_metadata=meta,
                ),
                finish_reason=finish_reason,
            )],
            usage=usage,
        )

    @staticmethod
    def _iter_sse_events(body: str):
        """Parse SSE text into event dicts.

        Splits on double newlines, extracts ``data:`` lines, yields parsed JSON.
        """
        for chunk in body.split("\n\n"):
            data_lines = []
            for line in chunk.split("\n"):
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if not data_lines:
                continue
            data = "\n".join(data_lines).strip()
            if not data or data == "[DONE]":
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                logger.debug("Failed to parse SSE data: %s", data[:200])

    # ── Error handling ──

    @staticmethod
    def _is_retryable(status: int, error_text: str) -> bool:
        if status in _RETRYABLE_STATUS:
            return True
        return bool(_RETRYABLE_TEXT_RE.search(error_text))

    @staticmethod
    def _build_error(status: int, error_text: str) -> RuntimeError:
        """Parse error response and build a RuntimeError with a friendly message."""
        message = error_text or f"Codex API error (HTTP {status})"
        try:
            parsed = json.loads(error_text)
            err = parsed.get("error", {})
            if isinstance(err, dict):
                code = err.get("code", "") or err.get("type", "")
                if re.search(r"usage_limit_reached|usage_not_included|rate_limit_exceeded", code, re.I) or status == 429:
                    plan = err.get("plan_type", "")
                    plan_str = f" ({plan.lower()} plan)" if plan else ""
                    resets_at = err.get("resets_at")
                    when = ""
                    if resets_at is not None:
                        import time
                        mins = max(0, round((resets_at * 1000 - time.time() * 1000) / 60000))
                        when = f" Try again in ~{mins} min."
                    message = f"You have hit your ChatGPT usage limit{plan_str}.{when}".strip()
                elif err.get("message"):
                    message = err["message"]
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return RuntimeError(message)


# ---------------------------------------------------------------------------
# Helpers (module-level, not methods — no self needed)
# ---------------------------------------------------------------------------

def _translate_assistant(
    msg: dict,
    metadata: list[dict] | None,
    output: list,
    msg_counter: int,
) -> None:
    """Translate an assistant message dict to Responses API input items.

    If ``metadata`` (from ``_provider_metadata``) is present, reasoning items
    are replayed verbatim, message IDs and tool call ID pairs are restored.
    Otherwise, synthetic IDs are generated (for context messages or history
    from before Codex was the provider).
    """
    # Index metadata by type for O(1) lookups
    reasoning_items: list[dict] = []
    message_id: str | None = None
    fc_id_map: dict[str, str] = {}  # call_id → item_id

    if metadata:
        for entry in metadata:
            entry_type = entry.get("type")
            if entry_type == _META_MESSAGE_ID:
                message_id = entry.get("id")
            elif entry_type == _META_FC_IDS:
                fc_id_map[entry.get("call_id", "")] = entry.get("id", "")
            elif entry_type == _META_REASONING:
                # Explicit reasoning metadata marker
                reasoning_items.append(entry)
            elif entry.get("type") == "reasoning":
                # Raw reasoning item (stored as-is from response)
                reasoning_items.append(entry)

    # 1. Replay reasoning items first (they precede the message in the API)
    for item in reasoning_items:
        output.append(item)

    # 2. Reconstruct the text message if there is text content
    text = msg.get("content")
    if text:
        mid = message_id or f"msg_{msg_counter}"
        # Ensure ID is max 64 chars (Codex API requirement)
        if len(mid) > 64:
            mid = f"msg_{msg_counter}"
        output.append({
            "type": "message",
            "role": "assistant",
            "id": mid,
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })

    # 3. Reconstruct tool calls
    tool_calls = msg.get("tool_calls", [])
    for tc in tool_calls:
        call_id = tc.get("id", "")
        fn = tc.get("function", {})
        item_id = fc_id_map.get(call_id)
        fc_item: dict = {
            "type": "function_call",
            "call_id": call_id,
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        }
        if item_id is not None:
            fc_item["id"] = item_id
        output.append(fc_item)


def _map_stop_reason(status: str | None) -> str:
    """Map Responses API status to a finish_reason string."""
    if status is None:
        return "stop"
    mapping = {
        "completed": "stop",
        "incomplete": "length",
        "failed": "error",
        "cancelled": "error",
        "in_progress": "stop",
        "queued": "stop",
    }
    return mapping.get(status, "stop")


def _nested_get(d: dict, *keys: str):
    """Safely traverse nested dicts."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d
