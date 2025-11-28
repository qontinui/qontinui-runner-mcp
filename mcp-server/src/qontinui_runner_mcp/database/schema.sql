-- SQLite schema for Qontinui Runner MCP server
-- Stores node metadata, workflows, and execution history

-- Create nodes table
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    action_type TEXT NOT NULL,
    parameters TEXT NOT NULL,
    examples TEXT,
    tags TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    tags TEXT,
    complexity TEXT CHECK(complexity IN ('simple', 'medium', 'complex')),
    template TEXT NOT NULL,
    use_cases TEXT,
    customization_points TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create executions table for tracking automation runs
CREATE TABLE IF NOT EXISTS executions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT,
    workflow_name TEXT,
    status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    started_at DATETIME,
    completed_at DATETIME,
    duration_ms INTEGER,
    result TEXT,
    error TEXT,
    screenshots TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create execution_events table for detailed event logs
CREATE TABLE IF NOT EXISTS execution_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES executions(id)
);

-- Create FTS5 virtual table for node search
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    id UNINDEXED,
    name,
    category,
    description,
    action_type,
    tags,
    examples,
    content='nodes',
    content_rowid='rowid'
);

-- Create FTS5 virtual table for workflow search
CREATE VIRTUAL TABLE IF NOT EXISTS workflows_fts USING fts5(
    id UNINDEXED,
    name,
    category,
    description,
    tags,
    use_cases,
    content='workflows',
    content_rowid='rowid'
);

-- Triggers to keep FTS5 tables in sync

CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, id, name, category, description, action_type, tags, examples)
    VALUES (new.rowid, new.id, new.name, new.category, new.description, new.action_type, new.tags, new.examples);
END;

CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    UPDATE nodes_fts SET
        name = new.name,
        category = new.category,
        description = new.description,
        action_type = new.action_type,
        tags = new.tags,
        examples = new.examples
    WHERE rowid = new.rowid;
END;

CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    DELETE FROM nodes_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS workflows_ai AFTER INSERT ON workflows BEGIN
    INSERT INTO workflows_fts(rowid, id, name, category, description, tags, use_cases)
    VALUES (new.rowid, new.id, new.name, new.category, new.description, new.tags, new.use_cases);
END;

CREATE TRIGGER IF NOT EXISTS workflows_au AFTER UPDATE ON workflows BEGIN
    UPDATE workflows_fts SET
        name = new.name,
        category = new.category,
        description = new.description,
        tags = new.tags,
        use_cases = new.use_cases
    WHERE rowid = new.rowid;
END;

CREATE TRIGGER IF NOT EXISTS workflows_ad AFTER DELETE ON workflows BEGIN
    DELETE FROM workflows_fts WHERE rowid = old.rowid;
END;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_nodes_category ON nodes(category);
CREATE INDEX IF NOT EXISTS idx_nodes_action_type ON nodes(action_type);
CREATE INDEX IF NOT EXISTS idx_workflows_category ON workflows(category);
CREATE INDEX IF NOT EXISTS idx_workflows_complexity ON workflows(complexity);
CREATE INDEX IF NOT EXISTS idx_executions_status ON executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_workflow_id ON executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_execution_events_execution_id ON execution_events(execution_id);
