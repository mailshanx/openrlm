import asyncio
import json
import logging
import struct
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when code execution in the sandbox raises an error."""

    def __init__(self, message: str, trace: str | None = None):
        super().__init__(message)
        self.trace = trace


@dataclass
class ExecutionResult:
    """The result of a successful code execution."""
    text: str | None


# Sentinel placed on per-agent queues when the TCP connection dies.
_DISCONNECT = object()


class ForkServerClient:
    """Client for the openrlm fork server.

    Speaks length-prefixed JSON over TCP. Multiplexes multiple agents
    over a single persistent data connection.

    A single reader task owns the StreamReader and dispatches incoming
    messages to per-agent asyncio.Queues keyed by agent_id. All sends
    go through a write lock to prevent interleaved bytes.

    Data channel (port) for spawn/execute/destroy requests.
    Control channel (control_port) for interrupt signals.
    """

    def __init__(self, port: int, control_port: int, host: str = "localhost"):
        self.port = port
        self.control_port = control_port
        self.host = host
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._write_lock = asyncio.Lock()
        self._queues: dict[str, asyncio.Queue] = {}
        self._reader_task: asyncio.Task | None = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self, retries: int = 20, retry_interval: float = 0.5):
        """Connect to the fork server's data channel with retries."""
        for attempt in range(retries):
            writer = None
            try:
                reader, writer = await asyncio.open_connection(
                    self.host, self.port
                )
                # Probe: spawn a throwaway agent, execute empty code, destroy it.
                # This verifies the fork server is fully ready (Docker port
                # forwarding can accept TCP before the process is listening).
                # Probe runs before the reader task starts — no concurrent reads.
                probe_id = "__probe__"
                probe_spawn = json.dumps({"type": "spawn", "agent_id": probe_id}).encode()
                writer.write(struct.pack(">I", len(probe_spawn)) + probe_spawn)
                await writer.drain()
                header = await asyncio.wait_for(reader.readexactly(4), timeout=5.0)
                length = struct.unpack(">I", header)[0]
                await asyncio.wait_for(reader.readexactly(length), timeout=5.0)

                probe_exec = json.dumps({"type": "execute", "agent_id": probe_id, "code": ""}).encode()
                writer.write(struct.pack(">I", len(probe_exec)) + probe_exec)
                await writer.drain()
                header = await asyncio.wait_for(reader.readexactly(4), timeout=5.0)
                length = struct.unpack(">I", header)[0]
                await asyncio.wait_for(reader.readexactly(length), timeout=5.0)

                probe_destroy = json.dumps({"type": "destroy", "agent_id": probe_id}).encode()
                writer.write(struct.pack(">I", len(probe_destroy)) + probe_destroy)
                await writer.drain()
                header = await asyncio.wait_for(reader.readexactly(4), timeout=5.0)
                length = struct.unpack(">I", header)[0]
                await asyncio.wait_for(reader.readexactly(length), timeout=5.0)

                self._reader = reader
                self._writer = writer
                self._reader_task = asyncio.create_task(self._read_loop())
                logger.info("Connected to fork server data channel")
                return
            except (ConnectionRefusedError, OSError, asyncio.IncompleteReadError,
                    asyncio.TimeoutError, ConnectionResetError):
                if writer is not None:
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass
                if attempt < retries - 1:
                    await asyncio.sleep(retry_interval)
        raise ConnectionError(
            f"Failed to connect to fork server at {self.host}:{self.port} "
            f"after {retries} attempts"
        )

    async def disconnect(self):
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
        self._queues.clear()

    # ── Reader task ──────────────────────────────────────────────────

    async def _read_loop(self):
        """Single reader coroutine: owns self._reader, dispatches by agent_id."""
        try:
            while True:
                header = await self._reader.readexactly(4)
                length = struct.unpack(">I", header)[0]
                data = await self._reader.readexactly(length)
                msg = json.loads(data)

                agent_id = msg.get("agent_id")
                queue = self._queues.get(agent_id)
                if queue is not None:
                    queue.put_nowait(msg)
                else:
                    logger.warning("Message for unknown agent %s: %s", agent_id, msg.get("type"))
        except asyncio.CancelledError:
            raise
        except (asyncio.IncompleteReadError, ConnectionResetError, OSError):
            logger.warning("Fork server connection lost")
        except Exception:
            logger.error("Reader loop error", exc_info=True)
        finally:
            # Poison all queues so waiting coroutines wake up with an error
            for queue in self._queues.values():
                queue.put_nowait(_DISCONNECT)

    # ── Send helper ──────────────────────────────────────────────────

    async def _send(self, msg: dict):
        async with self._write_lock:
            payload = json.dumps(msg).encode()
            self._writer.write(struct.pack(">I", len(payload)) + payload)
            await self._writer.drain()

    # ── Queue helper ─────────────────────────────────────────────────

    async def _queue_get(self, queue: asyncio.Queue) -> dict:
        """Get next message from queue, raising ConnectionError on disconnect."""
        msg = await queue.get()
        if msg is _DISCONNECT:
            raise ConnectionError("Fork server connection lost")
        return msg

    # ── Public API ───────────────────────────────────────────────────

    async def spawn(self, agent_id: str) -> None:
        """Spawn a new agent process in the fork server."""
        queue = asyncio.Queue()
        self._queues[agent_id] = queue
        await self._send({"type": "spawn", "agent_id": agent_id})
        resp = await self._queue_get(queue)
        if resp.get("type") != "spawned" or resp.get("agent_id") != agent_id:
            self._queues.pop(agent_id, None)
            raise RuntimeError(f"Unexpected spawn response: {resp}")

    async def destroy(self, agent_id: str) -> None:
        """Destroy an agent process in the fork server."""
        queue = self._queues.get(agent_id)
        if queue is None:
            queue = asyncio.Queue()
            self._queues[agent_id] = queue
        await self._send({"type": "destroy", "agent_id": agent_id})
        resp = await self._queue_get(queue)
        self._queues.pop(agent_id, None)
        if resp.get("type") != "destroyed" or resp.get("agent_id") != agent_id:
            raise RuntimeError(f"Unexpected destroy response: {resp}")

    async def execute(self, agent_id: str, code: str, timeout: float = 120) -> ExecutionResult:
        """Execute code in the specified agent. Interrupts on timeout or cancellation."""
        if self._writer is None:
            raise ConnectionError("Not connected to fork server")
        queue = self._queues.get(agent_id)
        if queue is None:
            raise RuntimeError(f"No queue for agent {agent_id!r} — was it spawned?")
        await self._send({"type": "execute", "agent_id": agent_id, "code": code})

        chunks: list[str] = []
        try:
            async with asyncio.timeout(timeout):
                while True:
                    msg = await self._queue_get(queue)
                    if msg["type"] == "stream":
                        chunks.append(msg["text"])
                    elif msg["type"] == "result":
                        if msg["error"] is not None:
                            trace = msg["error"]
                            lines = trace.strip().splitlines()
                            short = lines[-1] if lines else "Execution error"
                            raise ExecutionError(short, trace)
                        text = "".join(chunks).strip() if chunks else None
                        return ExecutionResult(text=text)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            await self._interrupt_and_drain(agent_id, queue)
            raise

    async def _interrupt_and_drain(self, agent_id: str, queue: asyncio.Queue) -> None:
        """Interrupt a running execution and drain messages until the result arrives.

        Called on both timeout and cancellation to leave the queue clean
        so the next execute call on this agent won't read stale messages.
        """
        await self._interrupt(agent_id)
        try:
            async with asyncio.timeout(3.0):
                while True:
                    msg = await self._queue_get(queue)
                    if msg["type"] == "result":
                        break
        except (asyncio.TimeoutError, ConnectionError):
            pass

    async def _interrupt(self, agent_id: str):
        """Send interrupt via the control channel (ephemeral connection)."""
        try:
            reader, writer = await asyncio.open_connection(
                self.host, self.control_port
            )
            payload = json.dumps({"type": "interrupt", "agent_id": agent_id}).encode()
            writer.write(struct.pack(">I", len(payload)) + payload)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            logger.info("Sent interrupt for agent %s", agent_id)
        except Exception:
            logger.warning("Failed to send interrupt for agent %s", agent_id, exc_info=True)
