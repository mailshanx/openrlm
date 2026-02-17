"""Host-side function bridge for the sandbox kernel.

Mechanism:
    1. On sandbox startup, we start a lightweight HTTP server on the host.
    2. A preamble cell is injected into the kernel that defines thin stub
       functions. Each stub does a synchronous HTTP POST to the host server
       with the function name and JSON-serialized arguments.
    3. The host server dispatches to the registered async Python callable,
       runs it, and returns the JSON result.
    4. The stub deserializes the response and returns it — execution in the
       kernel continues as if the function were local.

Usage:
    registry = HostFunctionRegistry()
    registry.register("internet_extract", execute_internet_extract)

    async with HostFunctionServer(registry) as server:
        # server.port is the listening port
        # server.preamble_code() returns Python source to inject into the kernel
        ...
"""

import asyncio
import inspect
import json as _json
import string
import logging
import traceback
from typing import Any, get_type_hints

from aiohttp import web
from pydantic import create_model
from pydantic.fields import FieldInfo


logger = logging.getLogger(__name__)

PREAMBLE_HEADER = """\
import json as _json
import httpx as _httpx
_HOST_FUNCTION_URL = "$BASE_URL"
"""
STUB_TEMPLATE = """\
async def $name($sig):
    _kwargs = {}
$kwargs_lines
    async with _httpx.AsyncClient(timeout=_httpx.Timeout($timeout)) as _client:
        _resp = await _client.post(f"{_HOST_FUNCTION_URL}/$name", json={"kwargs": _kwargs})
    _body = _resp.json()
    if "error" in _body:
        raise RuntimeError(_body["error"])
    return _body["result"]
"""


class HostFunctionRegistry:
    """Registry of async callables that the sandbox kernel can invoke."""

    def __init__(self):
        self._functions: dict[str, tuple[callable, list[str], str, float]] = {}

    def register(self, name: str, fn: callable, description: str = "", timeout: float = 300.0) -> None:
        """Register an async callable under the given name.
        Args:
            name: Name the kernel stub will be callable as.
            fn: Async callable to execute on the host when the stub is invoked.
            description: Human-readable description for the system prompt.
            timeout: HTTP timeout in seconds for the kernel-side stub (default 300).
        """
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(f"{name}: host functions must be async")
        sig = inspect.signature(fn)
        param_names = [p.name for p in sig.parameters.values()]
        self._functions[name] = (fn, param_names, description, timeout)

    @property
    def names(self) -> list[str]:
        return list(self._functions.keys())

    def get(self, name: str) -> tuple[callable, list[str], str, float] | None:
        return self._functions.get(name)

    @staticmethod
    def _build_function_schema(fn: callable, name: str, description: str) -> dict:
        """Build a JSON schema dict for a single host function using Pydantic.

        Inspects the function's type hints and signature to produce a proper
        JSON Schema.  Parameters whose names start with '_' are treated as
        internal and excluded from the schema.
        """
        type_hints = get_type_hints(fn, include_extras=True)
        sig = inspect.signature(fn)
        field_definitions: dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            if pname.startswith("_"):
                continue
            param_type = type_hints.get(pname, str)
            if param.default is not inspect.Parameter.empty:
                field_definitions[pname] = (param_type, FieldInfo(default=param.default, annotation=param_type))
            else:
                field_definitions[pname] = (param_type, FieldInfo(annotation=param_type))

        model = create_model(f"{name}_Schema", **field_definitions)
        schema = model.model_json_schema()

        # Strip Pydantic's auto-generated titles
        HostFunctionRegistry._strip_titles(schema)

        return {
            "name": name,
            "description": description,
            "parameters": schema,
        }

    @staticmethod
    def _strip_titles(schema: dict | list) -> None:
        """Remove 'title' keys from a JSON schema in-place, recursively."""
        if isinstance(schema, dict):
            schema.pop("title", None)
            for v in schema.values():
                HostFunctionRegistry._strip_titles(v)
        elif isinstance(schema, list):
            for item in schema:
                HostFunctionRegistry._strip_titles(item)

    def build_schemas_json(self) -> str:
        """Return JSON array of function schemas for embedding in system prompt."""
        if not self._functions:
            return "(none)"
        schemas = []
        for name, (fn, _, description, _) in self._functions.items():
            schemas.append(self._build_function_schema(fn, name, description))
        return _json.dumps(schemas, indent=2)


class HostFunctionServer:
    """HTTP server that bridges kernel stub calls to host-side async functions.

    Use as an async context manager. The server binds to a random available port
    on localhost.
    """

    def __init__(self, registry: HostFunctionRegistry):
        self._registry = registry
        self._app = web.Application()
        self._app.router.add_post("/call/{name}", self._handle_call)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._port: int | None = None

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("Server not started")
        return self._port

    async def __aenter__(self):
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", 0)
        await self._site.start()
        # Extract the actual port from the listening socket
        sockets = self._site._server.sockets
        self._port = sockets[0].getsockname()[1]
        logger.info("Host function server listening on port %d", self._port)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._runner:
            await self._runner.cleanup()

    async def _handle_call(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        entry = self._registry.get(name)
        if entry is None:
            return web.json_response({"error": f"Unknown function: {name}"}, status=404)
        fn, _, _, _ = entry
        try:
            body = await request.json()
            kwargs = body.get("kwargs", {})
            logger.info("Host function call: %s(%s)", name, ", ".join(f"{k}={v!r}" for k, v in kwargs.items()))
            fn_task = asyncio.create_task(fn(**kwargs))

            while not fn_task.done():
                proto = request.protocol
                transport = getattr(proto, 'transport', None) if proto else None
                if transport is None or transport.is_closing():
                    fn_task.cancel()
                    await asyncio.gather(fn_task, return_exceptions=True)
                    logger.warning("Client disconnected, cancelled %s", name)
                    return web.Response(status=499)
                try:
                    await asyncio.wait_for(asyncio.shield(fn_task), timeout=5.0)
                except asyncio.TimeoutError:
                    continue

            return web.json_response({"result": fn_task.result()})
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Host function %s failed:\n%s", name, tb)
            return web.json_response({"error": str(e), "traceback": tb}, status=500)

    def preamble_code(self) -> str:
        """Generate Python source to inject into the kernel.
        Defines one stub function per registered host function. Each stub
        makes a synchronous HTTP POST to the host server and returns the
        deserialized result.
        """
        base_url = f"http://host.docker.internal:{self.port}/call"
        stubs = []
        for name, (fn, param_names, _, timeout) in self._registry._functions.items():
            sig = inspect.signature(fn)
            params = []
            for pname, param in sig.parameters.items():
                if param.default is inspect.Parameter.empty:
                    params.append(pname)
                else:
                    params.append(f"{pname}={param.default!r}")
            sig_str = ", ".join(params)
            kwargs_lines = "\n".join(f"    _kwargs[{pname!r}] = {pname}" for pname in param_names)
            t = string.Template(STUB_TEMPLATE)
            stubs.append(t.substitute(name=name, sig=sig_str, kwargs_lines=kwargs_lines, timeout=timeout))

        preamble = PREAMBLE_HEADER.replace("$BASE_URL", base_url)
        return preamble + "\n" + "\n".join(stubs)
