"""Fork server for arcgeneral sandboxes.

Single Python file, zero non-stdlib dependencies. Runs as PID 1 in the container.

Architecture:
  - Supervisor process imports all expensive packages once, calls gc.freeze(),
    then runs asyncio TCP servers.
  - On "spawn" request, supervisor forks a child. The child inherits all
    imported modules via copy-on-write pages. Each child gets a fresh
    namespace dict and communicates with the supervisor over a socketpair.
  - Code executes in each child's main thread so os.kill(pid, SIGINT)
    reliably interrupts even C-level blocking calls.

Data channel (port 8888): persistent TCP connection, multiplexed by agent_id.
  Client -> Server:
    {"type": "spawn",   "agent_id": "..."}
    {"type": "execute", "agent_id": "...", "code": "..."}
    {"type": "destroy", "agent_id": "..."}
  Server -> Client:
    {"type": "spawned",   "agent_id": "..."}
    {"type": "stream",    "agent_id": "...", "text": "..."}  (zero or more)
    {"type": "result",    "agent_id": "...", "error": null|"..."}  (exactly one)
    {"type": "destroyed", "agent_id": "..."}

  Multiple agents' operations can be in flight concurrently. The server
  dispatches spawn/destroy inline and starts a background relay task for
  each execute. A per-agent asyncio.Lock serializes execute calls to the
  same child (the child is single-threaded). A write lock on the TCP writer
  prevents interleaved bytes from concurrent relay tasks.

Control channel (port 8889): ephemeral connections, fire-and-forget.
  Client -> Server:
    {"type": "interrupt", "agent_id": "..."}

Supervisor <-> Child (over socketpair, same 4-byte length-prefix + JSON):
  Supervisor -> Child: {"type": "execute", "code": "..."}
  Child -> Supervisor: {"type": "stream", "text": "..."}  (zero or more)
                       {"type": "result", "error": null|"..."}  (exactly one)
"""

import ast
import asyncio
import gc
import inspect
import json
import os
import signal
import socket
import struct
import sys
import traceback

DATA_PORT = int(os.environ.get("ARCGENERAL_DATA_PORT", 8888))
CONTROL_PORT = int(os.environ.get("ARCGENERAL_CONTROL_PORT", 8889))

# ── Pre-import expensive packages (before any fork) ──────────────────

if os.path.isdir("/app"):
    sys.path.append("/app")

def _preload():
    """Import packages that agents commonly use. Called once in supervisor."""
    mods = ["numpy", "pandas", "requests"]
    for name in mods:
        try:
            __import__(name)
        except ImportError:
            pass

# ── Code execution (runs in child processes) ─────────────────────────

def _run_code(code_obj, ns, loop):
    """Execute a code object, handling top-level async transparently."""
    if inspect.CO_COROUTINE & code_obj.co_flags:
        coro = eval(code_obj, ns)
        loop.run_until_complete(coro)
    else:
        exec(code_obj, ns)


def _run_cell(code, ns, loop):
    """Parse, compile, and execute a code string in the persistent namespace.

    Last expression is compiled with ast.Interactive + mode='single'
    so CPython emits PRINT_EXPR -> sys.displayhook(value).
    """
    tree = ast.parse(code)
    if not tree.body:
        return

    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

    if isinstance(tree.body[-1], ast.Expr):
        if tree.body[:-1]:
            mod = ast.Module(body=tree.body[:-1], type_ignores=[])
            _run_code(compile(mod, "<cell>", "exec", flags=flags), ns, loop)
        interactive = ast.Interactive(body=[tree.body[-1]])
        _run_code(compile(interactive, "<cell>", "single", flags=flags), ns, loop)
    else:
        _run_code(compile(tree, "<cell>", "exec", flags=flags), ns, loop)


# ── Socket helpers (blocking, for child process) ─────────────────────

def _sock_send(sock, msg):
    payload = json.dumps(msg).encode()
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _sock_recv(sock):
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise EOFError("connection closed")
        header += chunk
    length = struct.unpack(">I", header)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise EOFError("connection closed")
        data += chunk
    return json.loads(data)


# ── Child process ────────────────────────────────────────────────────

class _SocketStreamWriter:
    """Patches sys.stdout/stderr in the child to send stream messages."""

    def __init__(self, sock):
        self._sock = sock

    def write(self, text):
        if text:
            _sock_send(self._sock, {"type": "stream", "text": text})
        return len(text) if text else 0

    def flush(self):
        pass

    @property
    def encoding(self):
        return "utf-8"


_child_executing = False


def _child_sigint_handler(signum, frame):
    if _child_executing:
        raise KeyboardInterrupt


def _child_main(sock):
    """Entry point for a forked child. Runs code in the main thread.

    SIGINT reliably interrupts even C-level blocking calls because
    this runs in the main thread of its own process.
    """
    signal.signal(signal.SIGINT, _child_sigint_handler)
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    namespace = {"__builtins__": __builtins__}
    exec_loop = asyncio.new_event_loop()

    real_stdout, real_stderr = sys.stdout, sys.stderr

    while True:
        try:
            msg = _sock_recv(sock)
        except (EOFError, OSError):
            break

        if msg.get("type") != "execute":
            continue

        writer = _SocketStreamWriter(sock)
        sys.stdout = sys.stderr = writer

        global _child_executing
        _child_executing = True
        error = None
        try:
            _run_cell(msg.get("code", ""), namespace, exec_loop)
        except KeyboardInterrupt:
            error = "KeyboardInterrupt"
        except Exception:
            error = traceback.format_exc()
        finally:
            _child_executing = False
            sys.stdout, sys.stderr = real_stdout, real_stderr

        _sock_send(sock, {"type": "result", "error": error})

    sock.close()
    os._exit(0)


# ── Supervisor ───────────────────────────────────────────────────────

# agent_id -> (pid, parent_socket)
_agents: dict[str, tuple[int, socket.socket]] = {}


def _spawn_agent(agent_id):
    """Fork a child process for the given agent_id."""
    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    pid = os.fork()
    if pid == 0:
        # ── Child ──
        parent_sock.close()
        # Close all agent sockets inherited from supervisor
        for _, (_, s) in _agents.items():
            try:
                s.close()
            except Exception:
                pass
        _child_main(child_sock)
        # _child_main calls os._exit; this is a safety net
        os._exit(0)
    else:
        # ── Supervisor ──
        child_sock.close()
        parent_sock.setblocking(False)
        _agents[agent_id] = (pid, parent_sock)


def _destroy_agent(agent_id):
    """Kill and reap a child process."""
    entry = _agents.pop(agent_id, None)
    if entry is None:
        return
    pid, sock = entry
    sock.close()
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _interrupt_agent(agent_id):
    """Send SIGINT to a child process."""
    entry = _agents.get(agent_id)
    if entry is None:
        return
    pid, _ = entry
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        pass


# ── Async socketpair I/O (non-blocking, for supervisor) ──────────────

async def _async_recv_msg(sock):
    """Read one length-prefixed JSON message from a non-blocking socketpair."""
    loop = asyncio.get_running_loop()
    header = b""
    while len(header) < 4:
        chunk = await loop.sock_recv(sock, 4 - len(header))
        if not chunk:
            raise EOFError("connection closed")
        header += chunk
    length = struct.unpack(">I", header)[0]
    data = b""
    while len(data) < length:
        chunk = await loop.sock_recv(sock, length - len(data))
        if not chunk:
            raise EOFError("connection closed")
        data += chunk
    return json.loads(data)


async def _async_send_msg(sock, msg):
    """Send one length-prefixed JSON message to a non-blocking socketpair."""
    loop = asyncio.get_running_loop()
    payload = json.dumps(msg).encode()
    await loop.sock_sendall(sock, struct.pack(">I", len(payload)) + payload)


# ── Asyncio TCP protocol helpers ─────────────────────────────────────

async def _send_msg(writer, msg):
    payload = json.dumps(msg).encode()
    writer.write(struct.pack(">I", len(payload)) + payload)
    await writer.drain()


async def _recv_msg(reader):
    header = await reader.readexactly(4)
    length = struct.unpack(">I", header)[0]
    data = await reader.readexactly(length)
    return json.loads(data)


# ── Data channel handler ─────────────────────────────────────────────

async def _handle_data(reader, writer):
    """Handles the persistent TCP data connection from the host.

    Spawn/destroy are handled inline (fast). Execute starts a background
    relay task. A per-agent lock serializes execute calls to the same child.
    A write lock on the TCP writer prevents interleaved bytes.
    """
    write_lock = asyncio.Lock()
    agent_locks: dict[str, asyncio.Lock] = {}
    relay_tasks: set[asyncio.Task] = set()

    async def _locked_send(msg):
        async with write_lock:
            await _send_msg(writer, msg)

    async def _relay_child(agent_id, sock, code):
        """Execute code in a child and relay messages back to TCP.

        Holds the per-agent lock for the duration, ensuring only one
        execute at a time per child (the child is single-threaded).
        """
        lock = agent_locks.setdefault(agent_id, asyncio.Lock())
        async with lock:
            try:
                await _async_send_msg(sock, {"type": "execute", "code": code})
            except (EOFError, OSError):
                _destroy_agent(agent_id)
                await _locked_send({
                    "type": "result",
                    "agent_id": agent_id,
                    "error": "Agent process died unexpectedly",
                })
                return

            while True:
                try:
                    child_msg = await _async_recv_msg(sock)
                except (EOFError, OSError):
                    _destroy_agent(agent_id)
                    await _locked_send({
                        "type": "result",
                        "agent_id": agent_id,
                        "error": "Agent process died unexpectedly",
                    })
                    return

                child_msg["agent_id"] = agent_id
                await _locked_send(child_msg)
                if child_msg["type"] == "result":
                    return

    try:
        while True:
            try:
                msg = await _recv_msg(reader)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                break

            msg_type = msg.get("type")
            agent_id = msg.get("agent_id")

            if msg_type == "spawn":
                _spawn_agent(agent_id)
                await _locked_send({"type": "spawned", "agent_id": agent_id})

            elif msg_type == "destroy":
                _destroy_agent(agent_id)
                agent_locks.pop(agent_id, None)
                await _locked_send({"type": "destroyed", "agent_id": agent_id})

            elif msg_type == "execute":
                entry = _agents.get(agent_id)
                if entry is None:
                    await _locked_send({
                        "type": "result",
                        "agent_id": agent_id,
                        "error": f"No agent with id {agent_id!r}",
                    })
                    continue

                _pid, sock = entry
                task = asyncio.create_task(_relay_child(agent_id, sock, msg.get("code", "")))
                relay_tasks.add(task)
                task.add_done_callback(relay_tasks.discard)
    finally:
        # TCP connection dropped — cancel all active relay tasks
        for task in relay_tasks:
            task.cancel()
        await asyncio.gather(*relay_tasks, return_exceptions=True)


# ── Control channel handler ──────────────────────────────────────────

async def _handle_control(reader, writer):
    try:
        msg = await _recv_msg(reader)
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return
    finally:
        writer.close()

    if msg.get("type") == "interrupt":
        _interrupt_agent(msg.get("agent_id"))


# ── SIGCHLD handler ──────────────────────────────────────────────────

def _sigchld_handler(signum, frame):
    """Reap zombie children that exit unexpectedly."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
        except ChildProcessError:
            break


# ── Main ─────────────────────────────────────────────────────────────

async def _serve():
    bind_host = os.environ.get("ARCGENERAL_BIND_HOST", "0.0.0.0")
    data_server = await asyncio.start_server(_handle_data, bind_host, DATA_PORT)
    control_server = await asyncio.start_server(_handle_control, bind_host, CONTROL_PORT)
    print(f"fork server ready on data={DATA_PORT} control={CONTROL_PORT}", flush=True)
    async with data_server, control_server:
        await asyncio.gather(
            data_server.serve_forever(),
            control_server.serve_forever(),
        )


def main():
    # Pre-import expensive packages
    _preload()

    # Freeze GC: all objects from imports go to permanent generation.
    # GC will never scan them, avoiding COW page dirtying in children.
    gc.freeze()

    # Reap zombies
    signal.signal(signal.SIGCHLD, _sigchld_handler)

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
