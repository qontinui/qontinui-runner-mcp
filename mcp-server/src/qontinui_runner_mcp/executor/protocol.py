"""Protocol definitions for executor communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutorCommand:
    """Command to send to the Python executor."""

    command: str
    id: str
    params: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "type": "command",
            "id": self.id,
            "command": self.command,
        }
        if self.params:
            result["params"] = self.params
        return result


@dataclass
class ExecutorResponse:
    """Response from the Python executor."""

    id: str
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ExecutorResponse:
        """Parse from JSON dict."""
        return cls(
            id=data.get("id", ""),
            success=data.get("success", False),
            data=data.get("data"),
            error=data.get("error"),
        )


@dataclass
class ExecutorEvent:
    """Event from the Python executor."""

    event_type: str
    event: str
    timestamp: float
    sequence: int
    data: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ExecutorEvent:
        """Parse from JSON dict."""
        return cls(
            event_type=data.get("type", "event"),
            event=data.get("event", ""),
            timestamp=data.get("timestamp", 0.0),
            sequence=data.get("sequence", 0),
            data=data.get("data"),
        )


@dataclass
class ExecutionResult:
    """Result of an automation execution."""

    execution_id: str
    success: bool
    duration_ms: int = 0
    events: list[ExecutorEvent] = field(default_factory=list)
    error: str | None = None
    screenshot_base64: str | None = None
