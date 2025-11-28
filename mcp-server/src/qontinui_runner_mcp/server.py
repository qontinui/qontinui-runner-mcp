"""Qontinui Runner MCP Server.

Provides AI-powered workflow generation, node discovery, and automation execution
with integrated Python executor for real GUI automation.
"""

from __future__ import annotations

import json
import logging
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
from qontinui_runner_mcp.tools.workflow import WorkflowGenerator
from qontinui_runner_mcp.utils.validation import validate_workflow_structure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_DIR = Path.home() / ".qontinui" / "runner-mcp"
DB_PATH = DB_DIR / "qontinui.db"

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

            _save_execution(
                conn,
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
