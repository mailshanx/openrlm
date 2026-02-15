import asyncio
import logging

import aiohttp
from ipybox import ExecutionClient, ExecutionContainer, ExecutionError
from arcgeneral.host_functions import HostFunctionRegistry, HostFunctionServer
from typing import Self


logger = logging.getLogger(__name__)


# Code cells executed in every new kernel before the agent loop starts.
# Add setup code here (imports, helpers, data loading, etc.).
KERNEL_PREAMBLE: list[str] = []


class _Container(ExecutionContainer):
    """ExecutionContainer with AutoRemove disabled for post-mortem inspection."""

    async def _run(self):
        executor_host_port = {"HostPort": str(self._executor_port)} if self._executor_port else {}
        resource_host_port = {"HostPort": str(self._resource_port)} if self._resource_port else {}

        executor_port_key = f"{8888}/tcp"
        resource_port_key = f"{8900}/tcp"

        config = {
            "Image": self.tag,
            "HostConfig": {
                "CapAdd": ["NET_ADMIN", "NET_RAW"],
                "PortBindings": {
                    executor_port_key: [executor_host_port],
                    resource_port_key: [resource_host_port],
                },
                "AutoRemove": False,
                "Binds": await self._container_binds(),
            },
            "Env": self._container_env() + ["KG_WS_PING_INTERVAL_SECS=0"],
            "ExposedPorts": {
                executor_port_key: {},
                resource_port_key: {},
            },
        }

        if not await self._local_image():
            await self._pull_image()

        container = await self._docker.containers.create(config=config)
        await container.start()

        self._container = container
        self._executor_port = await self._host_port(container, executor_port_key)
        self._resource_port = await self._host_port(container, resource_port_key)

        return container

    async def kill(self):
        """Kill the container and explicitly remove it (since AutoRemove is off)."""
        if self._container:
            try:
                await self._container.kill()
            except Exception:
                pass
            try:
                await self._container.delete(force=True)
            except Exception:
                pass
        if self._docker:
            await self._docker.close()

    async def dump_logs(self) -> str:
        """Return container stdout+stderr logs. Safe to call even if container is dead."""
        if not self._container:
            return "(no container)"
        try:
            lines = await self._container.log(stdout=True, stderr=True)
            return "".join(lines)
        except Exception as e:
            return f"(failed to get logs: {e})"


class Sandbox:
    """Manages Docker container + IPython kernel lifecycle.

    Automatically reconnects when the kernel dies mid-session. Kernel state
    is lost on reconnect but the session continues.
    """

    def __init__(self, tag: str = "arcgeneral:sandbox", binds: dict[str, str] | None = None, host_functions: HostFunctionRegistry | None = None):
        self._tag = tag
        self._binds = binds or {}
        self._container: _Container | None = None
        self._client: ExecutionClient | None = None
        self._host_functions = host_functions
        self._host_server: HostFunctionServer | None = None

    async def __aenter__(self) -> Self:
        if self._host_functions and self._host_functions.names:
            self._host_server = HostFunctionServer(self._host_functions)
            await self._host_server.__aenter__()
        self._container = _Container(tag=self._tag, binds=self._binds)
        await self._container.__aenter__()
        self._client = ExecutionClient(port=self._container.executor_port)
        await self._client.__aenter__()
        await self._run_preamble()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client is not None:
            try:
                await self._client.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                logger.warning("Failed to disconnect ExecutionClient", exc_info=True)
            finally:
                self._client = None
        if self._container is not None:
            try:
                await self._container.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                logger.warning("Failed to kill ExecutionContainer", exc_info=True)
            finally:
                self._container = None
        if self._host_server is not None:
            try:
                await self._host_server.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                logger.warning("Failed to stop HostFunctionServer", exc_info=True)
            finally:
                self._host_server = None

    async def _run_preamble(self) -> None:
        """Execute host function stubs + KERNEL_PREAMBLE cells in the kernel."""
        if self._host_server is not None:
            code = self._host_server.preamble_code()
            logger.info("Injecting host function stubs into kernel")
            await self._client.execute(code=code, timeout=30.0)
        for cell in KERNEL_PREAMBLE:
            logger.info("Running preamble cell")
            await self._client.execute(code=cell, timeout=30.0)

    async def _probe_kernel_gateway(self) -> None:
        """Check that the kernel gateway is reachable. Raises RuntimeError if not."""
        url = f"http://localhost:{self._container.executor_port}/api"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"Kernel gateway probe: HTTP {resp.status}\n{body}")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Kernel gateway probe: {e}") from e

    async def _dump_logs_and_die(self, original_error: Exception) -> None:
        """Dump container logs for diagnostics, then re-raise the original error."""
        logger.error("Kernel died: %s", original_error)
        logs = await self._container.dump_logs()
        logger.error("Container logs at time of crash:\n%s", logs)
        raise original_error

    async def execute(self, code: str, timeout: float = 120.0) -> str:
        """Execute code in the IPython kernel. Returns output or error as a string.
        On kernel death, dumps container logs for diagnostics and crashes.
        """
        assert self._client is not None, "Sandbox not initialized. Use 'async with'."
        try:
            await self._probe_kernel_gateway()
            result = await self._client.execute(code=code, timeout=timeout)
            if result.text is not None:
                return result.text
            return "Code executed successfully. No output was returned."
        except ExecutionError as e:
            return e.trace if isinstance(e.trace, str) else "Code execution failed."
        except asyncio.TimeoutError:
            return f"Code execution timed out after {timeout} seconds."
        except Exception as e:
            await self._dump_logs_and_die(e)
