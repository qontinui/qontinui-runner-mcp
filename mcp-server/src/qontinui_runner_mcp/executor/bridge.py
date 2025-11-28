"""Python executor bridge - spawns and communicates with the executor."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from qontinui_runner_mcp.executor.protocol import (
    ExecutionResult,
    ExecutorCommand,
    ExecutorEvent,
    ExecutorResponse,
)

logger = logging.getLogger(__name__)


class ExecutorState(Enum):
    """Executor state."""

    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    FAILED = "failed"


@dataclass
class PendingCommand:
    """A command waiting for a response."""

    command: ExecutorCommand
    future: asyncio.Future[ExecutorResponse]


class ExecutorBridge:
    """Bridge to the Python executor process."""

    def __init__(
        self,
        python_bridge_path: str | None = None,
        executor_type: str = "simple",
    ) -> None:
        self.python_bridge_path = python_bridge_path or self._find_python_bridge()
        self.executor_type = executor_type
        self.state = ExecutorState.STOPPED
        self.process: subprocess.Popen[bytes] | None = None
        self.pending_commands: dict[str, PendingCommand] = {}
        self.event_handlers: list[Callable[[ExecutorEvent], None]] = []
        self.events: list[ExecutorEvent] = []
        self._reader_task: asyncio.Task[None] | None = None
        self._ready_event: asyncio.Event = asyncio.Event()

    def _find_python_bridge(self) -> str:
        """Find the python-bridge directory."""
        possible_paths = [
            Path(__file__).parent.parent.parent.parent.parent.parent.parent
            / "qontinui-runner"
            / "python-bridge",
            Path.home() / "qontinui" / "qontinui-runner" / "python-bridge",
            Path("/mnt/c/qontinui/qontinui-runner/python-bridge"),
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        raise FileNotFoundError(
            "Could not find python-bridge directory. "
            "Please set QONTINUI_PYTHON_BRIDGE_PATH environment variable."
        )

    def _get_script_name(self) -> str:
        """Get the script name based on executor type."""
        if self.executor_type == "minimal":
            return "minimal_bridge.py"
        elif self.executor_type == "real":
            return "qontinui_executor.py"
        else:
            return "qontinui_bridge.py"

    def _find_python_executable(self) -> str:
        """Find the Python executable to use."""
        bridge_path = Path(self.python_bridge_path)

        if sys.platform == "win32":
            venv_python = bridge_path / ".venv" / "Scripts" / "python.exe"
        else:
            venv_python = bridge_path / ".venv" / "bin" / "python"

        if venv_python.exists():
            return str(venv_python)

        return sys.executable

    async def start(self) -> None:
        """Start the executor process."""
        if self.state != ExecutorState.STOPPED:
            raise RuntimeError(f"Executor is already in state: {self.state}")

        self.state = ExecutorState.STARTING
        self._ready_event.clear()

        script_name = self._get_script_name()
        script_path = Path(self.python_bridge_path) / script_name

        if not script_path.exists():
            raise FileNotFoundError(f"Executor script not found: {script_path}")

        python_exe = self._find_python_executable()
        logger.info(f"Starting executor: {python_exe} {script_path}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["QONTINUI_DISABLE_CONSOLE_LOGGING"] = "1"

        cmd = [python_exe, "-u", str(script_path)]
        if self.executor_type != "real":
            cmd.append("--mock")
        cmd.append("--disable-console-logging")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=self.python_bridge_path,
        )

        self._reader_task = asyncio.create_task(self._read_output())

        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=30.0)
            self.state = ExecutorState.READY
            logger.info("Executor is ready")
        except asyncio.TimeoutError:
            self.state = ExecutorState.FAILED
            await self.stop()
            raise RuntimeError("Executor failed to become ready within 30 seconds")

    async def stop(self) -> None:
        """Stop the executor process."""
        if self.process is None:
            return

        logger.info("Stopping executor")

        try:
            await self.send_command("stop")
        except Exception:
            pass

        await asyncio.sleep(0.5)

        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        self.process = None
        self.state = ExecutorState.STOPPED
        logger.info("Executor stopped")

    async def send_command(
        self,
        command: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> ExecutorResponse:
        """Send a command and wait for response."""
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Executor is not running")

        cmd = ExecutorCommand(
            command=command,
            id=str(uuid.uuid4()),
            params=params,
        )

        loop = asyncio.get_event_loop()
        future: asyncio.Future[ExecutorResponse] = loop.create_future()
        self.pending_commands[cmd.id] = PendingCommand(command=cmd, future=future)

        cmd_json = json.dumps(cmd.to_json()) + "\n"
        self.process.stdin.write(cmd_json.encode())
        self.process.stdin.flush()

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            del self.pending_commands[cmd.id]
            raise RuntimeError(f"Command '{command}' timed out after {timeout}s")

    async def execute_workflow(
        self,
        config_path: str,
        timeout: float = 300.0,
    ) -> ExecutionResult:
        """Execute a workflow and return results."""
        execution_id = str(uuid.uuid4())
        self.events = []
        start_time = asyncio.get_event_loop().time()

        try:
            if self.state != ExecutorState.READY:
                await self.start()

            load_response = await self.send_command(
                "load",
                {"config_path": config_path},
            )

            if not load_response.success:
                return ExecutionResult(
                    execution_id=execution_id,
                    success=False,
                    error=load_response.error or "Failed to load configuration",
                )

            self.state = ExecutorState.RUNNING

            start_response = await self.send_command(
                "start",
                {"mode": "execute"},
                timeout=timeout,
            )

            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return ExecutionResult(
                execution_id=execution_id,
                success=start_response.success,
                duration_ms=duration_ms,
                events=list(self.events),
                error=start_response.error,
            )

        except Exception as e:
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                duration_ms=duration_ms,
                events=list(self.events),
                error=str(e),
            )
        finally:
            self.state = ExecutorState.READY

    async def _read_output(self) -> None:
        """Read output from the executor process."""
        if self.process is None or self.process.stdout is None:
            return

        loop = asyncio.get_event_loop()

        while True:
            try:
                line = await loop.run_in_executor(
                    None, self.process.stdout.readline
                )

                if not line:
                    break

                line_str = line.decode().strip()
                if not line_str:
                    continue

                try:
                    data = json.loads(line_str)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON output from executor: {line_str}")

            except Exception as e:
                logger.error(f"Error reading executor output: {e}")
                break

        logger.info("Executor output reader finished")

    async def _handle_message(self, data: dict[str, Any]) -> None:
        """Handle a message from the executor."""
        msg_type = data.get("type", "")

        if msg_type == "ready":
            logger.info("Received READY signal from executor")
            self._ready_event.set()

        elif msg_type == "response":
            response = ExecutorResponse.from_json(data)
            pending = self.pending_commands.pop(response.id, None)
            if pending:
                pending.future.set_result(response)

        elif msg_type == "event" or msg_type == "tree_event":
            event = ExecutorEvent.from_json(data)
            self.events.append(event)
            for handler in self.event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")

        elif msg_type == "pong":
            pass

        elif msg_type == "error":
            logger.error(f"Executor error: {data.get('message', 'Unknown error')}")

        else:
            logger.debug(f"Unknown message type: {msg_type}")

    def add_event_handler(self, handler: Callable[[ExecutorEvent], None]) -> None:
        """Add an event handler."""
        self.event_handlers.append(handler)

    def remove_event_handler(self, handler: Callable[[ExecutorEvent], None]) -> None:
        """Remove an event handler."""
        self.event_handlers.remove(handler)
