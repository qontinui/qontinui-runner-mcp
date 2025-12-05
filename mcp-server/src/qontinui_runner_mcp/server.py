"""Qontinui Runner MCP Server.

Provides AI-powered workflow generation, node discovery, and automation execution
with integrated Python executor for real GUI automation.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from qontinui_runner_mcp.database.loader import get_node, initialize_database
from qontinui_runner_mcp.database.search import (
    get_all_categories,
    search_nodes,
    search_nodes_by_action_type,
    search_nodes_by_category,
    search_workflows,
)
from qontinui_runner_mcp.executor.bridge import ExecutorBridge, ExecutorState
from qontinui_runner_mcp.tools.expectations import (
    evaluate_checkpoint,
    evaluate_workflow_expectations,
    validate_expectations_config,
)
from qontinui_runner_mcp.tools.ocr import extract_ocr_text, is_ocr_available
from qontinui_runner_mcp.tools.workflow import WorkflowGenerator
from qontinui_runner_mcp.types.models import CheckpointDefinition
from qontinui_runner_mcp.utils.validation import validate_workflow_structure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_DIR = Path.home() / ".qontinui" / "runner-mcp"
DB_PATH = DB_DIR / "qontinui.db"

# Automation results directory (for QA feedback loop)
AUTOMATION_RESULTS_DIR = Path("/mnt/c/Users/Joshua/Documents/qontinui_parent_directory/.automation-results")
DEV_LOGS_DIR = Path("/mnt/c/Users/Joshua/Documents/qontinui_parent_directory/.dev-logs")
MAX_HISTORY_RUNS = 10

server = Server("qontinui-runner-mcp")
executor: ExecutorBridge | None = None


def get_db_connection() -> sqlite3.Connection:
    """Get database connection, initializing if needed."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return initialize_database(str(DB_PATH))


def get_executor() -> ExecutorBridge:
    """Get or create the executor bridge."""
    global executor
    if executor is None:
        executor = ExecutorBridge()
    return executor


def get_workflow_expectations(workflow_config: dict[str, Any]) -> dict[str, Any] | None:
    """Extract expectations from workflow config if present.

    Args:
        workflow_config: Full workflow configuration dict

    Returns:
        Expectations dict or None if not present
    """
    return workflow_config.get("expectations")


def _build_execution_stats(events: list[Any]) -> dict[str, Any]:
    """Build execution statistics from event list.

    Args:
        events: List of execution events

    Returns:
        Dictionary with execution statistics:
        - total_actions: int
        - successful_actions: int
        - failed_actions: int
        - skipped_actions: int
        - match_count: int
        - states_reached: set[str]
        - checkpoints_passed: set[str]
        - checkpoints_failed: set[str]
    """
    stats = {
        "total_actions": 0,
        "successful_actions": 0,
        "failed_actions": 0,
        "skipped_actions": 0,
        "match_count": 0,
        "states_reached": set(),
        "checkpoints_passed": set(),
        "checkpoints_failed": set(),
    }

    for event in events:
        event_name = getattr(event, 'event', '')
        event_data = getattr(event, 'data', {}) or {}

        # Count actions
        if event_name in ['action_started', 'action_completed']:
            if event_name == 'action_started':
                stats["total_actions"] += 1
            elif event_name == 'action_completed':
                if event_data.get('success'):
                    stats["successful_actions"] += 1
                else:
                    stats["failed_actions"] += 1

        # Count matches
        if event_name == 'match_found':
            stats["match_count"] += 1

        # Track states
        if event_name == 'state_changed':
            new_state = event_data.get('state')
            if new_state:
                stats["states_reached"].add(new_state)

        # Track checkpoints
        if event_name == 'checkpoint_passed':
            checkpoint = event_data.get('checkpoint_name')
            if checkpoint:
                stats["checkpoints_passed"].add(checkpoint)
        elif event_name == 'checkpoint_failed':
            checkpoint = event_data.get('checkpoint_name')
            if checkpoint:
                stats["checkpoints_failed"].add(checkpoint)

        # Track skipped actions
        if event_name == 'action_skipped':
            stats["skipped_actions"] += 1

    return stats


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    tools = [
        # Knowledge/Search Tools
        Tool(
            name="search_nodes",
            description="Search for Qontinui action nodes using natural language.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'click button', 'find image')",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum results (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_workflows",
            description="Search for workflow templates using natural language.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'login workflow')",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum results (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_nodes_by_category",
            description="Get all nodes in a category (mouse, keyboard, vision).",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Category name"},
                },
                "required": ["category"],
            },
        ),
        Tool(
            name="get_nodes_by_action_type",
            description="Get nodes by action type (CLICK, FIND, TYPE).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "description": "Action type"},
                },
                "required": ["action_type"],
            },
        ),
        Tool(
            name="list_categories",
            description="List all available categories.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_action_details",
            description="Get detailed info about a specific action node.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_id": {"type": "string", "description": "Action node ID"},
                },
                "required": ["action_id"],
            },
        ),
        # Workflow Tools
        Tool(
            name="validate_workflow",
            description="Validate a workflow JSON structure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow": {"type": "object", "description": "Workflow JSON"},
                },
                "required": ["workflow"],
            },
        ),
        Tool(
            name="create_workflow",
            description="Create a workflow from action steps.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Workflow name"},
                    "description": {"type": "string", "description": "Description"},
                    "steps": {
                        "type": "array",
                        "description": "Workflow steps",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "description": {"type": "string"},
                                "options": {"type": "object"},
                            },
                            "required": ["action"],
                        },
                    },
                },
                "required": ["name", "description", "steps"],
            },
        ),
        Tool(
            name="generate_workflow",
            description="Generate workflow from natural language description.",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description",
                    },
                },
                "required": ["description"],
            },
        ),
        # Execution Tools
        Tool(
            name="run_automation",
            description="Execute a Qontinui automation workflow.",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_path": {
                        "type": "string",
                        "description": "Path to workflow configuration file",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Maximum execution time (default: 300)",
                        "default": 300,
                    },
                },
                "required": ["config_path"],
            },
        ),
        Tool(
            name="get_executor_status",
            description="Get the current status of the automation executor.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="start_executor",
            description="Start the automation executor.",
            inputSchema={
                "type": "object",
                "properties": {
                    "executor_type": {
                        "type": "string",
                        "description": "Executor type: 'simple', 'minimal', or 'real'",
                        "default": "simple",
                    },
                },
            },
        ),
        Tool(
            name="stop_executor",
            description="Stop the automation executor.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_execution_events",
            description="Get events from the last execution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum events to return",
                        "default": 100,
                    },
                },
            },
        ),
        # Database Tools
        Tool(
            name="list_executions",
            description="List recent automation executions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum results (default: 20)",
                        "default": 20,
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by status (pending, running, completed, failed)",
                    },
                },
            },
        ),
        Tool(
            name="get_execution",
            description="Get details of a specific execution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "execution_id": {"type": "string", "description": "Execution ID"},
                },
                "required": ["execution_id"],
            },
        ),
        # Config Management Tools
        Tool(
            name="load_config",
            description="Load a JSON configuration file into the executor. Use this before running workflows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_path": {
                        "type": "string",
                        "description": "Absolute path to the JSON configuration file",
                    },
                },
                "required": ["config_path"],
            },
        ),
        Tool(
            name="get_loaded_config",
            description="Get information about the currently loaded configuration, including available workflows.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="ensure_config_loaded",
            description="Ensure a specific config file is loaded. Loads it if not already loaded, skips if already loaded.",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_path": {
                        "type": "string",
                        "description": "Absolute path to the JSON configuration file",
                    },
                },
                "required": ["config_path"],
            },
        ),
        Tool(
            name="run_workflow",
            description="Run a specific workflow by name from the currently loaded configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_name": {
                        "type": "string",
                        "description": "Name of the workflow to run",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Maximum execution time in seconds (default: 300)",
                        "default": 300,
                    },
                },
                "required": ["workflow_name"],
            },
        ),
        # Expectations Tools
        Tool(
            name="validate_expectations",
            description="Validate the structure of an expectations configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expectations": {
                        "type": "object",
                        "description": "Expectations configuration to validate",
                    },
                },
                "required": ["expectations"],
            },
        ),
        Tool(
            name="evaluate_expectations",
            description="Evaluate workflow expectations against execution statistics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow identifier",
                    },
                    "expectations": {
                        "type": "object",
                        "description": "Expectations configuration",
                    },
                    "execution_stats": {
                        "type": "object",
                        "description": "Execution statistics (total_actions, failed_actions, etc.)",
                    },
                },
                "required": ["workflow_id", "execution_stats"],
            },
        ),
        Tool(
            name="capture_and_evaluate_checkpoint",
            description="Capture a screenshot, extract OCR text, and evaluate checkpoint assertions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "checkpoint_name": {
                        "type": "string",
                        "description": "Name of the checkpoint",
                    },
                    "checkpoint_definition": {
                        "type": "object",
                        "description": "Checkpoint configuration with assertions",
                    },
                    "screenshot_path": {
                        "type": "string",
                        "description": "Path to screenshot file or base64 image data",
                    },
                },
                "required": ["checkpoint_name", "checkpoint_definition", "screenshot_path"],
            },
        ),
    ]
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    conn = get_db_connection()
    generator = WorkflowGenerator(conn)

    try:
        result: Any = None

        # Knowledge/Search Tools
        if name == "search_nodes":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            if not query:
                raise ValueError("Query parameter is required")
            result = search_nodes(conn, query, limit)

        elif name == "search_workflows":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            if not query:
                raise ValueError("Query parameter is required")
            result = search_workflows(conn, query, limit)

        elif name == "get_nodes_by_category":
            category = arguments.get("category", "")
            if not category:
                raise ValueError("Category parameter is required")
            result = search_nodes_by_category(conn, category)

        elif name == "get_nodes_by_action_type":
            action_type = arguments.get("action_type", "")
            if not action_type:
                raise ValueError("Action type parameter is required")
            result = search_nodes_by_action_type(conn, action_type)

        elif name == "list_categories":
            result = get_all_categories(conn)

        elif name == "get_action_details":
            action_id = arguments.get("action_id", "")
            if not action_id:
                raise ValueError("action_id parameter is required")
            node = get_node(conn, action_id)
            result = node if node else {"error": f"Action not found: {action_id}"}

        # Workflow Tools
        elif name == "validate_workflow":
            workflow = arguments.get("workflow")
            if not workflow:
                raise ValueError("workflow parameter is required")
            validation = validate_workflow_structure(workflow)
            result = validation.model_dump()

        elif name == "create_workflow":
            wf_name = arguments.get("name", "")
            description = arguments.get("description", "")
            steps = arguments.get("steps", [])
            if not wf_name or not description or not steps:
                raise ValueError("name, description, and steps are required")
            workflow = generator.create_workflow(wf_name, description, steps)
            validation = validate_workflow_structure(workflow)
            result = {"workflow": workflow, "validation": validation.model_dump()}

        elif name == "generate_workflow":
            description = arguments.get("description", "")
            if not description:
                raise ValueError("description parameter is required")
            gen_result = generator.generate_from_description(description)
            result = {
                "success": gen_result.success,
                "workflow": gen_result.workflow,
                "error": gen_result.error,
                "suggestions": gen_result.suggestions,
            }

        # Execution Tools
        elif name == "run_automation":
            config_path = arguments.get("config_path", "")
            timeout = arguments.get("timeout_seconds", 300)
            if not config_path:
                raise ValueError("config_path parameter is required")

            exec_bridge = get_executor()
            exec_result = await exec_bridge.execute_workflow(config_path, timeout)

            # Save to database
            _save_execution(
                conn,
                exec_result.execution_id,
                config_path,
                exec_result.success,
                exec_result.duration_ms,
                exec_result.error,
                exec_result.events,
            )

            # Save to filesystem for QA feedback loop
            results_file = _save_automation_results(
                exec_result.execution_id,
                config_path,
                exec_result.success,
                exec_result.duration_ms,
                exec_result.error,
                exec_result.events,
            )

            result = {
                "execution_id": exec_result.execution_id,
                "success": exec_result.success,
                "duration_ms": exec_result.duration_ms,
                "error": exec_result.error,
                "event_count": len(exec_result.events),
                "results_file": str(results_file),
            }

        elif name == "get_executor_status":
            exec_bridge = get_executor()
            result = {
                "state": exec_bridge.state.value,
                "is_running": exec_bridge.state in [ExecutorState.READY, ExecutorState.RUNNING],
            }

        elif name == "start_executor":
            executor_type = arguments.get("executor_type", "simple")
            global executor
            executor = ExecutorBridge(executor_type=executor_type)
            await executor.start()
            result = {"success": True, "state": executor.state.value}

        elif name == "stop_executor":
            exec_bridge = get_executor()
            await exec_bridge.stop()
            result = {"success": True, "state": exec_bridge.state.value}

        elif name == "get_execution_events":
            limit = arguments.get("limit", 100)
            exec_bridge = get_executor()
            events = exec_bridge.events[-limit:]
            result = {
                "events": [
                    {
                        "event_type": e.event_type,
                        "event": e.event,
                        "timestamp": e.timestamp,
                        "data": e.data,
                    }
                    for e in events
                ],
                "total": len(exec_bridge.events),
            }

        # Database Tools
        elif name == "list_executions":
            limit = arguments.get("limit", 20)
            status = arguments.get("status")
            result = _list_executions(conn, limit, status)

        elif name == "get_execution":
            execution_id = arguments.get("execution_id", "")
            if not execution_id:
                raise ValueError("execution_id parameter is required")
            result = _get_execution(conn, execution_id)

        # Config Management Tools
        elif name == "load_config":
            config_path = arguments.get("config_path", "")
            if not config_path:
                raise ValueError("config_path parameter is required")

            exec_bridge = get_executor()
            load_response = await exec_bridge.load_config(config_path)

            result = {
                "success": load_response.success,
                "config_path": config_path,
                "error": load_response.error,
                "config_info": exec_bridge.get_loaded_config_info() if load_response.success else None,
            }

        elif name == "get_loaded_config":
            exec_bridge = get_executor()
            result = exec_bridge.get_loaded_config_info()

        elif name == "ensure_config_loaded":
            config_path = arguments.get("config_path", "")
            if not config_path:
                raise ValueError("config_path parameter is required")

            exec_bridge = get_executor()
            load_response = await exec_bridge.ensure_config_loaded(config_path)

            already_loaded = load_response.data.get("already_loaded", False) if load_response.data else False
            result = {
                "success": load_response.success,
                "config_path": config_path,
                "already_loaded": already_loaded,
                "error": load_response.error,
                "config_info": exec_bridge.get_loaded_config_info(),
            }

        elif name == "run_workflow":
            workflow_name = arguments.get("workflow_name", "")
            timeout = arguments.get("timeout_seconds", 300)
            if not workflow_name:
                raise ValueError("workflow_name parameter is required")

            exec_bridge = get_executor()

            # Check if a config is loaded
            if exec_bridge.loaded_config is None:
                result = {
                    "success": False,
                    "error": "No configuration loaded. Use load_config or ensure_config_loaded first.",
                }
            else:
                exec_result = await exec_bridge.run_workflow(workflow_name, timeout)
                config_path = exec_bridge.loaded_config_path or "unknown"

                # Save execution to database
                _save_execution(
                    conn,
                    exec_result.execution_id,
                    config_path,
                    exec_result.success,
                    exec_result.duration_ms,
                    exec_result.error,
                    exec_result.events,
                )

                # Save to filesystem for QA feedback loop
                results_file = _save_automation_results(
                    exec_result.execution_id,
                    config_path,
                    exec_result.success,
                    exec_result.duration_ms,
                    exec_result.error,
                    exec_result.events,
                    workflow_name=workflow_name,
                )

                # Extract and evaluate expectations if present
                expectations_result = None
                if exec_bridge.loaded_config:
                    expectations = get_workflow_expectations(exec_bridge.loaded_config)
                    if expectations:
                        # Build execution stats from events
                        execution_stats = _build_execution_stats(exec_result.events)

                        # Evaluate expectations
                        eval_result = evaluate_workflow_expectations(
                            workflow_id=workflow_name,
                            expectations=expectations,
                            execution_stats=execution_stats,
                        )
                        expectations_result = eval_result.model_dump()

                result = {
                    "execution_id": exec_result.execution_id,
                    "workflow_name": workflow_name,
                    "success": exec_result.success,
                    "duration_ms": exec_result.duration_ms,
                    "error": exec_result.error,
                    "event_count": len(exec_result.events),
                    "results_file": str(results_file),
                    "expectations_result": expectations_result,
                }

        # Expectations Tools
        elif name == "validate_expectations":
            expectations = arguments.get("expectations")
            if not expectations:
                raise ValueError("expectations parameter is required")

            is_valid, errors = validate_expectations_config(expectations)
            result = {
                "valid": is_valid,
                "errors": errors,
            }

        elif name == "evaluate_expectations":
            workflow_id = arguments.get("workflow_id", "")
            expectations = arguments.get("expectations")
            execution_stats = arguments.get("execution_stats", {})

            if not workflow_id:
                raise ValueError("workflow_id parameter is required")
            if not execution_stats:
                raise ValueError("execution_stats parameter is required")

            eval_result = evaluate_workflow_expectations(
                workflow_id=workflow_id,
                expectations=expectations,
                execution_stats=execution_stats,
            )
            result = eval_result.model_dump()

        elif name == "capture_and_evaluate_checkpoint":
            checkpoint_name = arguments.get("checkpoint_name", "")
            checkpoint_definition = arguments.get("checkpoint_definition")
            screenshot_path = arguments.get("screenshot_path", "")

            if not checkpoint_name:
                raise ValueError("checkpoint_name parameter is required")
            if not checkpoint_definition:
                raise ValueError("checkpoint_definition parameter is required")
            if not screenshot_path:
                raise ValueError("screenshot_path parameter is required")

            # Parse checkpoint definition
            try:
                checkpoint_def = CheckpointDefinition.model_validate(checkpoint_definition)
            except Exception as e:
                result = {
                    "success": False,
                    "error": f"Invalid checkpoint definition: {e}",
                }
            else:
                # Extract OCR text if OCR is available and assertions are present
                ocr_text = None
                if checkpoint_def.ocr_assertions and is_ocr_available():
                    ocr_text = extract_ocr_text(screenshot_path)
                    if ocr_text is None:
                        logger.warning("OCR extraction failed for checkpoint")

                # Evaluate checkpoint
                checkpoint_result = evaluate_checkpoint(
                    checkpoint_name=checkpoint_name,
                    checkpoint_def=checkpoint_def,
                    screenshot_path=screenshot_path,
                    ocr_text=ocr_text,
                )
                result = checkpoint_result.model_dump()

        else:
            raise ValueError(f"Unknown tool: {name}")

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        raise

    finally:
        conn.close()


def _save_execution(
    conn: sqlite3.Connection,
    execution_id: str,
    config_path: str,
    success: bool,
    duration_ms: int,
    error: str | None,
    events: list[Any],
) -> None:
    """Save execution to database."""
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO executions
        (id, workflow_name, status, started_at, completed_at, duration_ms, error)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            execution_id,
            Path(config_path).stem,
            "completed" if success else "failed",
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            duration_ms,
            error,
        ),
    )

    for event in events:
        cursor.execute(
            """
            INSERT INTO execution_events (execution_id, event_type, event_data)
            VALUES (?, ?, ?)
            """,
            (
                execution_id,
                event.event,
                json.dumps(event.data) if event.data else None,
            ),
        )

    conn.commit()


def _list_executions(
    conn: sqlite3.Connection,
    limit: int,
    status: str | None,
) -> list[dict[str, Any]]:
    """List executions from database."""
    cursor = conn.cursor()

    if status:
        cursor.execute(
            """
            SELECT * FROM executions
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (status, limit),
        )
    else:
        cursor.execute(
            """
            SELECT * FROM executions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

    rows = cursor.fetchall()
    return [
        {
            "id": row["id"],
            "workflow_name": row["workflow_name"],
            "status": row["status"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "duration_ms": row["duration_ms"],
            "error": row["error"],
        }
        for row in rows
    ]


def _get_execution(
    conn: sqlite3.Connection,
    execution_id: str,
) -> dict[str, Any] | None:
    """Get execution details from database."""
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM executions WHERE id = ?", (execution_id,))
    row = cursor.fetchone()

    if not row:
        return None

    cursor.execute(
        "SELECT * FROM execution_events WHERE execution_id = ? ORDER BY timestamp",
        (execution_id,),
    )
    event_rows = cursor.fetchall()

    return {
        "id": row["id"],
        "workflow_name": row["workflow_name"],
        "status": row["status"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "duration_ms": row["duration_ms"],
        "error": row["error"],
        "events": [
            {
                "event_type": e["event_type"],
                "event_data": json.loads(e["event_data"]) if e["event_data"] else None,
                "timestamp": e["timestamp"],
            }
            for e in event_rows
        ],
    }


def _save_automation_results(
    execution_id: str,
    config_path: str,
    success: bool,
    duration_ms: int,
    error: str | None,
    events: list[Any],
    workflow_name: str | None = None,
) -> Path:
    """Save automation results to filesystem for QA feedback loop.

    This saves results to .automation-results/latest/ and archives to history/.
    Log snapshots are captured from .dev-logs/ at execution time.
    """
    # Ensure directories exist
    latest_dir = AUTOMATION_RESULTS_DIR / "latest"
    history_dir = AUTOMATION_RESULTS_DIR / "history"
    latest_logs_dir = latest_dir / "logs"
    latest_screenshots_dir = latest_dir / "screenshots"

    for d in [latest_dir, history_dir, latest_logs_dir, latest_screenshots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Archive previous latest to history (if exists)
    existing_execution_file = latest_dir / "execution.json"
    if existing_execution_file.exists():
        try:
            with open(existing_execution_file) as f:
                prev_data = json.load(f)
                prev_id = prev_data.get("execution_id", "unknown")
                prev_timestamp = prev_data.get("timestamp", "unknown").replace(":", "-").replace(".", "-")

            # Create history entry
            history_entry_name = f"{prev_timestamp}_{prev_id[:8]}"
            history_entry_dir = history_dir / history_entry_name

            # Move latest to history
            if not history_entry_dir.exists():
                shutil.copytree(latest_dir, history_entry_dir)
                logger.info(f"Archived previous run to history: {history_entry_name}")

            # Clean up old history entries
            _cleanup_history(history_dir)
        except Exception as e:
            logger.warning(f"Failed to archive previous results: {e}")

    # Clear latest directory
    for item in latest_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    # Recreate subdirectories
    latest_logs_dir.mkdir(exist_ok=True)
    latest_screenshots_dir.mkdir(exist_ok=True)

    # Capture log snapshots from .dev-logs
    log_files = ["backend.log", "frontend.log", "qontinui-api.log", "runner.log"]
    for log_file in log_files:
        src_log = DEV_LOGS_DIR / log_file
        if src_log.exists():
            try:
                # Copy last 500 lines of each log (truncate for efficiency)
                with open(src_log, "r", errors="ignore") as f:
                    lines = f.readlines()
                    last_lines = lines[-500:] if len(lines) > 500 else lines

                dst_log = latest_logs_dir / log_file
                with open(dst_log, "w") as f:
                    f.writelines(last_lines)

                logger.info(f"Captured log snapshot: {log_file} ({len(last_lines)} lines)")
            except Exception as e:
                logger.warning(f"Failed to capture log {log_file}: {e}")

    # Build execution results JSON
    timestamp = datetime.now().isoformat()

    # Extract test results from events
    test_results = []
    console_errors = []
    network_failures = []
    screenshots = []

    for event in events:
        event_name = getattr(event, 'event', '')
        event_data = getattr(event, 'data', {}) or {}

        # Capture test-related events
        if event_name in ['test_passed', 'test_failed', 'assertion_passed', 'assertion_failed']:
            test_results.append({
                "event": event_name,
                "data": event_data,
                "timestamp": getattr(event, 'timestamp', ''),
            })

        # Capture console errors
        if event_name == 'console_error' or (event_name == 'browser_event' and event_data.get('type') == 'error'):
            console_errors.append(event_data)

        # Capture network failures
        if event_name == 'network_failure' or (event_name == 'network_event' and event_data.get('status', 200) >= 400):
            network_failures.append(event_data)

        # Capture screenshots
        if event_name == 'screenshot' and event_data.get('path'):
            screenshots.append(event_data.get('path'))

    execution_result = {
        "execution_id": execution_id,
        "config_path": config_path,
        "workflow_name": workflow_name or Path(config_path).stem,
        "success": success,
        "duration_ms": duration_ms,
        "timestamp": timestamp,
        "error": error,
        "summary": {
            "total_events": len(events),
            "test_results_count": len(test_results),
            "console_errors_count": len(console_errors),
            "network_failures_count": len(network_failures),
        },
        "test_results": test_results,
        "console_errors": console_errors,
        "network_failures": network_failures,
        "screenshots": screenshots,
        "log_snapshots": {
            "backend": str(latest_logs_dir / "backend.log") if (latest_logs_dir / "backend.log").exists() else None,
            "frontend": str(latest_logs_dir / "frontend.log") if (latest_logs_dir / "frontend.log").exists() else None,
            "api": str(latest_logs_dir / "qontinui-api.log") if (latest_logs_dir / "qontinui-api.log").exists() else None,
            "runner": str(latest_logs_dir / "runner.log") if (latest_logs_dir / "runner.log").exists() else None,
        },
    }

    # Write execution.json
    execution_file = latest_dir / "execution.json"
    with open(execution_file, "w") as f:
        json.dump(execution_result, f, indent=2)

    logger.info(f"Saved automation results to {execution_file}")

    return execution_file


def _cleanup_history(history_dir: Path) -> None:
    """Keep only the most recent MAX_HISTORY_RUNS in history."""
    entries = sorted(
        [d for d in history_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    # Remove old entries
    for old_entry in entries[MAX_HISTORY_RUNS:]:
        try:
            shutil.rmtree(old_entry)
            logger.info(f"Removed old history entry: {old_entry.name}")
        except Exception as e:
            logger.warning(f"Failed to remove old history entry {old_entry.name}: {e}")


async def run_server() -> None:
    """Run the MCP server."""
    logger.info("Starting Qontinui Runner MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Entry point."""
    import asyncio
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
