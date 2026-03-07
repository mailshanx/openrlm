import asyncio
import logging
from functools import partial
from pathlib import Path

from aiodocker import Docker
from aiodocker.containers import DockerContainer

logger = logging.getLogger(__name__)


class ExecutionContainer:
    """Context manager for the lifecycle of a code execution Docker container."""

    def __init__(
        self,
        tag: str,
        binds: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        executor_port: int | None = None,
        control_port: int | None = None,
        port_allocation_timeout: float = 10,
    ):
        self.tag = tag
        self.binds = binds or {}
        self.env = env or {}

        self._docker = None
        self._container = None
        self._executor_port = executor_port
        self._control_port = control_port
        self._port_allocation_timeout = port_allocation_timeout

    async def __aenter__(self):
        try:
            await self.run()
        except Exception as e:
            await self.kill()
            raise e
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.kill()

    @property
    def executor_port(self) -> int:
        if self._executor_port is None:
            raise RuntimeError("Container not running")
        return self._executor_port

    @property
    def control_port(self) -> int:
        if self._control_port is None:
            raise RuntimeError("Container not running")
        return self._control_port

    async def kill(self):
        """Kills and removes the container."""
        if self._container:
            await self._container.kill()
        if self._docker:
            await self._docker.close()

    async def run(self):
        """Creates and starts the container."""
        self._docker = Docker()
        await self._run()

    async def _run(self):
        data_host_port = {"HostPort": str(self._executor_port)} if self._executor_port else {}
        ctrl_host_port = {"HostPort": str(self._control_port)} if self._control_port else {}
        data_port_key = "8888/tcp"
        ctrl_port_key = "8889/tcp"

        config = {
            "Image": self.tag,
            "HostConfig": {
                "PortBindings": {
                    data_port_key: [data_host_port],
                    ctrl_port_key: [ctrl_host_port],
                },
                "AutoRemove": True,
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

    async def _host_port(self, container: DockerContainer, port_key: str) -> int:
        try:
            async with asyncio.timeout(self._port_allocation_timeout):
                while True:
                    container_info = await container.show()
                    host_ports = container_info["NetworkSettings"]["Ports"].get(port_key)
                    if host_ports and host_ports[0].get("HostPort"):
                        return int(host_ports[0]["HostPort"])
                    await asyncio.sleep(0.1)
        except TimeoutError:
            raise TimeoutError(
                f"Timed out waiting for host port allocation after {self._port_allocation_timeout} seconds"
            )

    async def _container_binds(self) -> list[str]:
        container_binds = []
        loop = asyncio.get_running_loop()
        for host_path, container_path in self.binds.items():
            resolved = await loop.run_in_executor(None, partial(self._prepare_host_path, host_path))
            container_binds.append(f"{resolved}:/app/{container_path}")
        return container_binds

    def _prepare_host_path(self, host_path: str) -> Path:
        resolved = Path(host_path).resolve()
        if not resolved.exists():
            resolved.mkdir(parents=True)
        return resolved

    def _container_env(self) -> list[str]:
        return [f"{k}={v}" for k, v in self.env.items()]
