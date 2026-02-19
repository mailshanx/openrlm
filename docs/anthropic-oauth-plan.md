# Anthropic OAuth Support Plan

## Goal

Allow arcgeneral to use Anthropic OAuth tokens (`sk-ant-oat-*`) from users who logged in via Pi's `/login` command.

## Changes

### 1. `AnthropicClient._ensure_client` — Bearer auth + stealth headers

Detect OAuth tokens by prefix. Use `auth_token` (Bearer) instead of `api_key` (x-api-key), plus required stealth headers.

```python
def _ensure_client(self, api_key: str):
    import anthropic
    if self._client is None:
        if "sk-ant-oat" in api_key:
            self._client = anthropic.AsyncAnthropic(
                api_key=None,
                auth_token=api_key,
                default_headers={
                    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
                    "user-agent": "claude-cli/2.1.2 (external, cli)",
                    "x-app": "cli",
                },
            )
            self._is_oauth = True
        else:
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            self._is_oauth = False
    return self._client
```

### 2. `AnthropicClient.complete` — prepend Claude Code identity to system prompt

When OAuth, prepend `"You are Claude Code, Anthropic's official CLI for Claude."` as a separate text block before the user's system prompt. Pass as an array of content blocks (not a plain string).

```python
if self._is_oauth:
    blocks = [{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}]
    if system_content:
        blocks.append({"type": "text", "text": system_content})
    create_kwargs["system"] = blocks
elif system_content is not None:
    create_kwargs["system"] = system_content
```

### 3. `arcgeneral-pi` extension — bridge the API key

In `runArcgeneral`, resolve the key from Pi's auth system and inject it into the subprocess environment. Pass `--provider anthropic` and `--model`.

```typescript
const apiKey = await ctx.modelRegistry.getApiKeyForProvider("anthropic");
const proc = spawn(bin, args, {
    env: { ...process.env, ANTHROPIC_API_KEY: apiKey },
});
args.push("--provider", "anthropic");
args.push("--model", ctx.model.id);
```

### 4. Tests

- Mock `_ensure_client` with an OAuth token (`sk-ant-oat-test`): verify `auth_token` is used, `api_key` is `None`, stealth headers are set.
- Mock `complete` with an OAuth token: verify system prompt is an array with Claude Code identity prepended.
- Mock `complete` with a regular API key: verify system prompt is a plain string, no identity prepend.

## Not needed

- **Tool name remapping** — arcgeneral's only tool is `python`, not in the Claude Code tool list. Passes through unchanged.
- **Tool description changes** — Pi doesn't change descriptions for OAuth mode.
- **Message/response translation changes** — already correct.

## Token refresh concern

OAuth tokens expire. Pi refreshes them with file-locked reads of `~/.pi/agent/auth.json`. The arcgeneral subprocess does not have access to this mechanism. For runs shorter than the token's remaining lifetime (5-minute buffer), this is fine. For long runs, `messages.create()` will get a 401. Options:

1. Accept the failure (simplest).
2. On 401, re-read `~/.pi/agent/auth.json`, create a new SDK client with the refreshed token, retry.
3. Implement the full file-lock refresh protocol from Python.
