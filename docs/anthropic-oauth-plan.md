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

### 3. `arcgeneral-pi` extension — bridge the API key via auth file

The extension writes the provider's API key to `~/.arcgeneral/auth.json` before spawning the subprocess. A `setInterval` (4 minutes) refreshes the token for long-running tasks. On exit, the entry is removed.

```typescript
// Write initial token
writeAuthFile("anthropic", await tokenRefresher());

// Refresh every 4 minutes
refreshInterval = setInterval(async () => {
    const newToken = await tokenRefresher();
    if (newToken) writeAuthFile("anthropic", newToken);
}, 240_000);

// Cleanup on exit
removeAuthEntry("anthropic");
```

Auth file format (`~/.arcgeneral/auth.json`):
```json
{"anthropic": "sk-ant-oat-..."}
```

File permissions: directory `0o700`, file `0o600`. Writes are atomic (tmp + rename).

### 4. `default_api_key_resolver` — auth file with 3-tier priority

The resolver in `llm.py` reads the auth file on every `_llm_call`, so external refreshers (the Pi extension) can update it mid-run.

Resolution priority for any provider:
1. Auth file (`~/.arcgeneral/auth.json` or `ARCGENERAL_AUTH_FILE` env var override)
2. `ANTHROPIC_OAUTH_TOKEN` (Anthropic only, legacy compat)
3. Provider-specific env var (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, etc.)

### 5. Key-change client reconstruction

`AnthropicClient._ensure_client` tracks `_current_key`. When the key changes (refreshed token written to auth file → re-read by resolver), the SDK client is reconstructed. The old client is deferred to `_stale_client` for cleanup in `close()`.

### 6. Tests

- `test_auth_file_resolver()` — covers auth file wins, fallthrough to env var, non-anthropic providers, refresh (auth file changes between calls), deleted file, invalid JSON
- OAuth token detection: verify `auth_token` is used, `api_key` is `None`, stealth headers are set
- OAuth system prompt: verify array with Claude Code identity prepended
- Regular API key: verify no identity prepend, plain string system prompt

## Not needed

- **Tool name remapping** — arcgeneral's only tool is `python`, not in the Claude Code tool list. Passes through unchanged.
- **Tool description changes** — Pi doesn't change descriptions for OAuth mode.
- **Message/response translation changes** — already correct.

## Implementation Status

All changes implemented and tested. 237 tests pass.

### Commits

- `6292503` (arcgeneral) — AnthropicClient with OAuth stealth mode + tests (15 assertions)
- `e965b5c` (arcgeneral) — Auth file system (`_read_auth_file`, `default_api_key_resolver` 3-tier priority, key-change client reconstruction)
- `21fce12` (arcgeneral-pi) — Bridge API key + auth file refresh (`writeAuthFile`, `removeAuthEntry`, `resolveLLMConfig`, `tokenRefresher`)

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
| Live token refresh | Per-request in `streamAnthropic` | Auth file re-read per `_llm_call` + key-change client reconstruction | ✅ |
