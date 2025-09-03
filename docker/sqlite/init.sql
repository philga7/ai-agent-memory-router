-- SQLite initialization script for AI Agent Memory Router
-- This script creates the initial database schema

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Create agents table
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    capabilities TEXT, -- JSON string of agent capabilities
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create memories table
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    tags TEXT, -- JSON string of tags
    metadata TEXT, -- JSON string of metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

-- Create memory_routes table
CREATE TABLE IF NOT EXISTS memory_routes (
    id TEXT PRIMARY KEY,
    source_agent_id TEXT NOT NULL,
    target_agent_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    route_status TEXT DEFAULT 'pending',
    priority TEXT DEFAULT 'normal',
    context TEXT, -- JSON string of context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_agent_id) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (target_agent_id) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

-- Create contexts table
CREATE TABLE IF NOT EXISTS contexts (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    conversation_id TEXT,
    context_data TEXT NOT NULL, -- JSON string of context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_routes_source ON memory_routes(source_agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_routes_target ON memory_routes(target_agent_id);
CREATE INDEX IF NOT EXISTS idx_contexts_agent_id ON contexts(agent_id);
CREATE INDEX IF NOT EXISTS idx_contexts_conversation_id ON contexts(conversation_id);

-- Insert some sample data for testing
INSERT OR IGNORE INTO agents (id, name, description, capabilities) VALUES 
    ('agent-001', 'Memory Router', 'Central memory routing agent', '["routing", "memory_management"]'),
    ('agent-002', 'Context Manager', 'Manages conversation context', '["context_management", "state_tracking"]');

-- Create a trigger to update the updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_agents_timestamp 
    AFTER UPDATE ON agents
    FOR EACH ROW
    BEGIN
        UPDATE agents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_memories_timestamp 
    AFTER UPDATE ON memories
    FOR EACH ROW
    BEGIN
        UPDATE memories SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
