import asyncio
import logging
import uuid

import aiodocker
from arcgeneral.ipybox import ForkServerClient, ExecutionContainer, ExecutionError
from arcgeneral.host_functions import HostFunctionServer
from typing import Self


logger = logging.getLogger(__name__)


# Code cells executed in every new agent before the agent loop starts.
KERNEL_PREAMBLE: list[str] = ["import asyncio"]


class _Container(ExecutionContainer):
    """ExecutionContainer with AutoRemove disabled for post-mortem inspection."""

    async def _run(self):
        data_host_port = {"HostPort": str(self._executor_port)} if self._executor_port else {}
        ctrl_host_port = {"HostPort": str(self._control_port)} if self._control_port else {}

        data_port_key = "8888/tcp"
        ctrl_port_key = "8889/tcp"
        config = {
            "Image": self.tag,
            "Labels": {"arcgeneral": "true"},
            "HostConfig": {
                "CapAdd": ["NET_ADMIN", "NET_RAW"],
                "PortBindings": {
                    data_port_key: [data_host_port],
                    ctrl_port_key: [ctrl_host_port],
                },
                "AutoRemove": False,
                "Binds": await self._container_binds(),
            },
            "Env": self._container_env(),
            "ExposedPorts": {
                data_port_key: {},
                ctrl_port_key: {},
            },
        }

        container = await self._docker.containers.create(config=config)
        await container.start()

        self._container = container
        self._executor_port = await self._host_port(container, data_port_key)
        self._control_port = await self._host_port(container, ctrl_port_key)
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


class ForkServer:
    """Owns a single Docker container running the fork server.

    One ForkServer per AgentRuntime. Call create_sandbox() to get
    a Sandbox for each agent (root or sub-agent at any depth).
    """

    def __init__(
        self,
        tag: str = "arcgeneral:sandbox",
        binds: dict[str, str] | None = None,
        host_function_server: HostFunctionServer | None = None,
    ):
        self._tag = tag
        self._binds = binds or {}
        self._host_function_server = host_function_server
        self._container: _Container | None = None
        self._client: ForkServerClient | None = None

    async def __aenter__(self) -> Self:
        self._container = _Container(tag=self._tag, binds=self._binds)
        await self._container.__aenter__()
        self._client = ForkServerClient(
            port=self._container.executor_port,
            control_port=self._container.control_port,
        )
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client is not None:
            try:
                await self._client.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                logger.warning("Failed to disconnect ForkServerClient", exc_info=True)
            finally:
                self._client = None
        if self._container is not None:
            try:
                await self._container.kill()
            except Exception:
                logger.warning("Failed to kill container", exc_info=True)
            finally:
                self._container = None

    async def create_sandbox(self) -> "Sandbox":
        """Spawn a new agent process and return a Sandbox handle for it."""
        assert self._client is not None, "ForkServer not initialized. Use 'async with'."
        agent_id = uuid.uuid4().hex[:12]
        await self._client.spawn(agent_id)

        sandbox = Sandbox(
            agent_id=agent_id,
            client=self._client,
            container=self._container,
            host_function_server=self._host_function_server,
        )
        await sandbox._run_preamble()
        return sandbox

    async def dump_logs(self) -> str:
        if self._container is not None:
            return await self._container.dump_logs()
        return "(no container)"


class Sandbox:
    """One agent's execution context within a fork server.

    The interface is the same as before: sandbox.execute(code, timeout) -> str.
    Agents don't know or care that they share a container.
    """

    def __init__(
        self,
        agent_id: str,
        client: ForkServerClient,
        container: _Container,
        host_function_server: HostFunctionServer | None = None,
    ):
        self._agent_id = agent_id
        self._client = client
        self._container = container
        self._host_function_server = host_function_server
        self._alive = True

    async def close(self):
        """Destroy this agent's process in the fork server."""
        if self._alive:
            try:
                await self._client.destroy(self._agent_id)
            except Exception:
                logger.warning("Failed to destroy agent %s", self._agent_id, exc_info=True)
            finally:
                self._alive = False

    async def _run_preamble(self) -> None:
        """Execute host function stubs + KERNEL_PREAMBLE cells."""
        if self._host_function_server is not None:
            code = self._host_function_server.preamble_code()
            logger.info("Injecting host function stubs into agent %s", self._agent_id)
            await self.execute(code, timeout=30.0)
        for cell in KERNEL_PREAMBLE:
            logger.info("Running preamble cell in agent %s", self._agent_id)
            await self.execute(cell, timeout=30.0)

    async def _dump_logs_and_die(self, original_error: Exception) -> None:
        """Dump container logs for diagnostics, then re-raise the original error."""
        logger.error("Agent %s kernel died: %s", self._agent_id, original_error)
        logs = await self._container.dump_logs()
        logger.error("Container logs at time of crash:\n%s", logs)
        raise original_error

    async def execute(self, code: str, timeout: float = 120.0) -> str:
        """Execute code in this agent's process. Returns output or error as a string.
        On kernel death, dumps container logs for diagnostics and crashes.
        """
        assert self._alive, f"Agent {self._agent_id} has been destroyed."
        try:
            result = await self._client.execute(
                agent_id=self._agent_id, code=code, timeout=timeout,
            )
            if result.text is not None:
                return result.text
            return "Code executed successfully. No output was returned."
        except ExecutionError as e:
            return e.trace if isinstance(e.trace, str) else "Code execution failed."
        except asyncio.TimeoutError:
            return f"Code execution timed out after {timeout} seconds."
        except Exception as e:
            await self._dump_logs_and_die(e)


async def cleanup_orphaned_containers():
    """Kill and remove any containers from previous arcgeneral runs."""
    try:
        docker = aiodocker.Docker()
        containers = await docker.containers.list(
            all=True,
            filters={"label": ["arcgeneral=true"]},
        )
        for c in containers:
            info = await c.show()
            name = info.get("Name", c.id[:12]).lstrip("/")
            try:
                await c.kill()
            except Exception:
                pass
            try:
                await c.delete(force=True)
            except Exception:
                pass
            logger.info("Cleaned up orphaned container %s", name)
        await docker.close()
    except Exception:
        logger.debug("Failed to clean up orphaned containers", exc_info=True)
