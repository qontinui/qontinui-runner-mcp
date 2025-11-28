# Qontinui Runner MCP Server

MCP (Model Context Protocol) server for Qontinui visual automation with integrated execution.

## Overview

This MCP server enables AI agents (like Claude) to:
- **Discover** automation actions and workflow templates via full-text search
- **Generate** workflows from natural language descriptions
- **Validate** workflow structures with cycle detection
- **Execute** GUI automations using the Qontinui Python executor
- **Track** execution history and retrieve results

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Claude Code                             │
│                            ↓                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           qontinui-runner-mcp (Python MCP Server)       │  │
│  │                                                         │  │
│  │  MCP Tools:                                             │  │
│  │  • search_nodes, search_workflows (knowledge)           │  │
│  │  • generate_workflow, validate_workflow                 │  │
│  │  • run_automation, get_execution (execution)            │  │
│  │                                                         │  │
│  │  Executor Bridge:                                       │  │
│  │  • Spawns Python executor process                       │  │
│  │  • JSON communication via stdin/stdout                  │  │
│  │  • Event collection and result tracking                 │  │
│  │                                                         │  │
│  │  Database:                                              │  │
│  │  • SQLite with FTS5 for search                          │  │
│  │  • Execution history and events                         │  │
│  └────────────────────────────────────────────────────────┘  │
│                            ↓                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │        Python Executor (from qontinui-runner)           │  │
│  │        • GUI automation (mouse, keyboard, vision)       │  │
│  │        • Screenshot capture                             │  │
│  │        • State detection                                │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.12+
- Poetry
- qontinui-runner (for the Python executor)

### Setup

```bash
cd mcp-server

# Install dependencies
poetry install

# Run the server
poetry run qontinui-runner-mcp
```

## MCP Tools

### Knowledge Tools
| Tool | Description |
|------|-------------|
| `search_nodes` | Search action nodes by natural language |
| `search_workflows` | Search workflow templates |
| `get_nodes_by_category` | Filter nodes by category |
| `get_nodes_by_action_type` | Filter by action type |
| `list_categories` | List all categories |
| `get_action_details` | Get detailed node info |

### Workflow Tools
| Tool | Description |
|------|-------------|
| `generate_workflow` | Generate from natural language |
| `create_workflow` | Create from structured steps |
| `validate_workflow` | Validate structure and connections |

### Execution Tools
| Tool | Description |
|------|-------------|
| `run_automation` | Execute a workflow |
| `start_executor` | Start the executor process |
| `stop_executor` | Stop the executor |
| `get_executor_status` | Check executor state |
| `get_execution_events` | Get events from last run |

### History Tools
| Tool | Description |
|------|-------------|
| `list_executions` | List recent executions |
| `get_execution` | Get execution details |

## Configuration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "qontinui": {
      "command": "poetry",
      "args": ["run", "qontinui-runner-mcp"],
      "cwd": "/path/to/qontinui-runner-mcp/mcp-server"
    }
  }
}
```

### Claude Code

Add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "qontinui": {
      "command": "poetry",
      "args": ["run", "qontinui-runner-mcp"],
      "cwd": "/path/to/qontinui-runner-mcp/mcp-server"
    }
  }
}
```

## Example Usage

```
Claude: Use the search_nodes tool to find click actions.

Claude: Generate a workflow to "click the login button and type username"

Claude: Execute the workflow at /path/to/workflow.json

Claude: Get the execution events from the last run.
```

## Development

```bash
cd mcp-server

# Install dev dependencies
poetry install --with dev

# Run linting
poetry run black src/
poetry run isort src/
poetry run ruff src/
poetry run mypy src/
```

## Database

The server uses SQLite with FTS5 for full-text search. Database is stored at:
- `~/.qontinui/runner-mcp/qontinui.db`

### Tables
- `nodes` - Action node definitions
- `workflows` - Workflow templates
- `executions` - Execution history
- `execution_events` - Detailed event logs

## License

MIT License
