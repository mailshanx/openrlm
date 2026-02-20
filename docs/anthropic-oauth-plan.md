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

## Token refresh for long-running tasks

OAuth tokens expire. The extension now continuously refreshes the token via a file-based IPC channel.

### Architecture

**Extension side** (arcgeneral-pi):
- Creates a temp file, writes the initial token atomically (write tmp + rename)
- Passes the file path to the subprocess as `ARCGENERAL_TOKEN_FILE` env var
- Runs a `setInterval` (4 minutes) calling `ctx.modelRegistry.getApiKeyForProvider("anthropic")` and writing the refreshed token to the file
- Cleans up the interval and file on process exit

**arcgeneral side** (llm.py):
- `default_api_key_resolver` checks `ARCGENERAL_TOKEN_FILE` first for the `anthropic` provider. Reads the file on every `_llm_call`.
- `AnthropicClient._ensure_client` tracks the current key. When the key changes (refreshed token), it reconstructs the SDK client and defers the old one to `_stale_client` for cleanup in `close()`.

### Resolution priority (anthropic provider)

1. `ARCGENERAL_TOKEN_FILE` (read from file, refreshed by extension)
2. `ANTHROPIC_OAUTH_TOKEN` (env var)
3. `ANTHROPIC_API_KEY` (env var)

## Implementation Status

All changes implemented and tested.

### Commits
- `6292503` (arcgeneral) — AnthropicClient with OAuth stealth mode + tests (15 assertions)
- `c7d200b` (arcgeneral) — ANTHROPIC_OAUTH_TOKEN env var precedence (3 assertions)
- `38bc4d4` (arcgeneral-pi) — Bridge API key from Pi's model registry to subprocess
- Token file refresh + key-change client reconstruction (pending commit)

### Verified against Pi's implementation
| Aspect | Pi's implementation | arcgeneral | Match |
|--------|-------------------|------------|-------|
| Token detection | `apiKey.includes("sk-ant-oat")` | `"sk-ant-oat" in api_key` | ✅ |
| Bearer auth | `apiKey: null, authToken: apiKey` | `api_key=None, auth_token=api_key` | ✅ |
| `anthropic-beta` | `claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14` | Same | ✅ |
| `user-agent` | `claude-cli/2.1.2 (external, cli)` | Same | ✅ |
| `x-app` | `cli` | Same | ✅ |
| System prompt | Array of content blocks, identity first | Same | ✅ |
| Identity string | `"You are Claude Code, Anthropic's official CLI for Claude."` | Same | ✅ |
| Tool renaming | `python` not in CC list, passthrough | No renaming | ✅ |
| Env var precedence | `ANTHROPIC_OAUTH_TOKEN` > `ANTHROPIC_API_KEY` | Same | ✅ |
| Token refresh at resolve time | `getApiKeyForProvider()` auto-refreshes | Handled by Pi before bridging | ✅ |
| Live token refresh | Per-request in `streamAnthropic` | Token file + key-change detection | ✅ |