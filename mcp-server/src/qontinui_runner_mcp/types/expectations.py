"""Expectations and success criteria type definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from qontinui_runner_mcp.types.models import Region


class SuccessCriteriaType(str, Enum):
    """Types of success criteria for workflow evaluation."""

    ALL_ACTIONS_PASS = "all_actions_pass"
    """All actions must succeed (default behavior)."""

    MIN_MATCHES = "min_matches"
    """At least N matches must be found across all actions."""

    MAX_FAILURES = "max_failures"
    """Allow up to N action failures."""

    CHECKPOINT_PASSED = "checkpoint_passed"
    """Specific named checkpoint action must pass."""

    REQUIRED_STATES = "required_states"
    """Must reach all specified states."""

    CUSTOM = "custom"
    """Custom Python expression evaluated against execution state."""


class SuccessCriterion(BaseModel):
    """A single success criterion definition."""

    criteria_type: SuccessCriteriaType = Field(alias="type")
    description: str | None = None

    # Type-specific parameters
    min_matches: int | None = None
    max_failures: int | None = None
    checkpoint_name: str | None = None
    required_states: list[str] | None = None
    custom_condition: str | None = None


class OCRAssertionType(str, Enum):
    """Types of OCR assertions."""

    TEXT_PRESENT = "text_present"
    TEXT_ABSENT = "text_absent"
    NO_DUPLICATE_MATCHES = "no_duplicate_matches"
    TEXT_COUNT = "text_count"
    TEXT_IN_REGION = "text_in_region"


class OCRAssertion(BaseModel):
    """OCR-based assertion for checkpoint validation."""

    assertion_type: OCRAssertionType = Field(alias="type")
    text: str
    case_sensitive: bool = False
    expected_count: int | None = None
    region: Region | None = None


class CheckpointDefinition(BaseModel):
    """Definition of a checkpoint for validation."""

    description: str | None = None
    screenshot_required: bool = True
    ocr_assertions: list[OCRAssertion] | None = None
    claude_review: list[str] | None = None
    max_wait_ms: int = Field(default=5000, gt=0)
    retry_interval_ms: int = Field(default=500, gt=0)


class GlobalExpectations(BaseModel):
    """Global workflow-level expectations."""

    max_workflow_duration_ms: int | None = None
    max_action_duration_ms: int | None = None
    screenshot_on_failure: bool = True
    fail_fast: bool = False


class WorkflowExpectations(BaseModel):
    """Complete expectations definition for a workflow."""

    global_settings: GlobalExpectations | None = Field(default=None, alias="global")
    success_criteria: list[SuccessCriterion] | None = None
    checkpoints: dict[str, CheckpointDefinition] | None = None


class ActionExpectations(BaseModel):
    """Per-action expectations configuration."""

    is_terminal_on_failure: bool = True
    capture_checkpoint_on_failure: bool = False
    capture_checkpoint_after: bool = False
    checkpoint_name: str | None = None
    max_retries: int | None = None
    retry_delay_ms: int | None = None
    max_duration_ms: int | None = None
    expected_state_after: str | None = None


class CheckpointResult(BaseModel):
    """Result of checkpoint capture and validation."""

    checkpoint_name: str
    timestamp: str
    screenshot_path: str | None = None
    ocr_text: str | None = None
    ocr_assertions_passed: bool = True
    ocr_assertion_results: list[dict[str, Any]] | None = None
    claude_review_results: list[dict[str, Any]] | None = None
    validation_errors: list[str] | None = None


class ExpectationsEvaluationResult(BaseModel):
    """Result of expectations evaluation for a workflow."""

    workflow_id: str
    success: bool
    criteria_results: list[dict[str, Any]]
    checkpoint_results: list[CheckpointResult] | None = None
    evaluation_summary: str
    duration_ms: int
