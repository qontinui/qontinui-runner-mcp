# Qontinui Desktop Runner & MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Tauri-based desktop runner and Model Context Protocol (MCP) server for Qontinui automation.

## Overview

The Qontinui Runner provides:
- Desktop application for running automation scripts
- MCP server for AI agent integration
- Real-time execution monitoring
- Cross-platform support (Windows, macOS, Linux)

## Architecture

```
qontinui-runner-mcp/
├── src/              # Svelte frontend
│   ├── lib/         # Components and utilities
│   ├── routes/      # SvelteKit routes
│   └── app.html     # Main HTML template
├── src-tauri/       # Rust backend
│   ├── src/         # Rust source code
│   ├── Cargo.toml   # Rust dependencies
│   └── tauri.conf.json
└── mcp-server/      # MCP server implementation
    ├── server.py    # FastMCP server
    └── tools.py     # MCP tools
```

## Features

### Desktop Runner
- Execute Qontinui automation scripts
- Visual feedback during execution
- State visualization
- Error handling and recovery
- Screenshot capture and analysis

### MCP Server
- Expose Qontinui capabilities to AI agents
- Tool definitions for automation actions
- State management interface
- Async execution support

## Installation

### Prerequisites
- Rust 1.70+
- Node.js 20+
- Python 3.12+ (for MCP server)

### Setup

```bash
# Install dependencies
npm install

# Install Tauri CLI
npm install -g @tauri-apps/cli

# Install Rust dependencies
cd src-tauri
cargo build

# Install Python dependencies for MCP
cd ../mcp-server
pip install -r requirements.txt
```

## Development

### Run in development mode

```bash
# Start the desktop app
npm run tauri dev

# Start the MCP server (in another terminal)
cd mcp-server
python server.py
```

### Build for production

```bash
npm run tauri build
```

## MCP Integration

The MCP server exposes Qontinui tools for AI agents:

```python
# Example MCP tool usage
tools = {
    "capture_screenshot": capture_screenshot_tool,
    "detect_state": detect_state_tool,
    "execute_action": execute_action_tool,
    "run_automation": run_automation_tool,
}
```

### Connecting with Claude Desktop

Add to Claude Desktop config:

```json
{
  "mcpServers": {
    "qontinui": {
      "command": "python",
      "args": ["/path/to/qontinui-runner-mcp/mcp-server/server.py"],
      "env": {}
    }
  }
}
```

## API

### Desktop Runner API

```typescript
// Execute automation script
await invoke('execute_script', { script: dslScript });

// Capture screenshot
const screenshot = await invoke('capture_screenshot');

// Get current state
const state = await invoke('get_current_state');
```

### MCP Tools

- `capture_screenshot`: Capture and analyze screen
- `detect_state`: Detect current application state
- `execute_action`: Execute a single automation action
- `run_automation`: Run complete automation script
- `get_elements`: Get UI elements from current screen

## Configuration

### tauri.conf.json

Configure app settings, permissions, and build options.

### MCP Server Config

```python
# mcp_config.py
MCP_CONFIG = {
    "server_name": "qontinui",
    "version": "0.1.0",
    "capabilities": ["screenshot", "automation", "state_detection"],
}
```

## Testing

```bash
# Run frontend tests
npm test

# Run Rust tests
cd src-tauri
cargo test

# Run MCP server tests
cd mcp-server
pytest
```

## Building

### Windows
```bash
npm run tauri build -- --target x86_64-pc-windows-msvc
```

### macOS
```bash
npm run tauri build -- --target universal-apple-darwin
```

### Linux
```bash
npm run tauri build -- --target x86_64-unknown-linux-gnu
```

## License

MIT License - see [LICENSE](LICENSE) file for details.