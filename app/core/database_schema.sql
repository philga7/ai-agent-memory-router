-- SQLite Database Schema for AI Agent Memory Router
-- This schema supports the storage abstraction layer and can be easily
-- migrated to PostgreSQL in the future.

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Enable WAL mode for better concurrency
PRAGMA journal_mode = WAL;

-- Create tables with proper indexing for performance

-- Agents table - stores information about AI agents
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    agent_type TEXT NOT NULL, -- assistant, tool, router, monitor, gateway, specialist
    version TEXT NOT NULL, -- agent version string
    capabilities TEXT, -- JSON string of agent capabilities
    status TEXT NOT NULL DEFAULT 'active', -- active, inactive, error
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT -- JSON string of additional metadata
);

-- Create index on agent status for quick filtering
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_created_at ON agents(created_at);

-- Memory items table - stores the actual memory content
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL, -- knowledge, experience, context
    priority INTEGER DEFAULT 1, -- 1=low, 2=normal, 3=high, 4=critical
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP, -- optional expiration
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memory_items_agent_id ON memory_items(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_items_type ON memory_items(memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_items_priority ON memory_items(priority);
CREATE INDEX IF NOT EXISTS idx_memory_items_created_at ON memory_items(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_items_expires_at ON memory_items(expires_at);

-- Memory metadata table - stores additional information about memories
CREATE TABLE IF NOT EXISTS memory_metadata (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    tags TEXT, -- JSON array of tags
    source TEXT, -- where the memory came from (weaviate, sqlite, cipher, etc.)
    confidence REAL DEFAULT 1.0, -- confidence score 0.0-1.0
    embedding_vector TEXT, -- base64 encoded vector for semantic search (optional with Weaviate)
    vector_dimension INTEGER, -- dimension of the embedding vector
    weaviate_object_id TEXT, -- Weaviate object ID for cross-reference
    project_id TEXT, -- project ID for cross-project knowledge sharing
    similarity_threshold REAL DEFAULT 0.7, -- similarity threshold used for this memory
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memory_items(id) ON DELETE CASCADE
);

-- Create indexes for metadata queries
CREATE INDEX IF NOT EXISTS idx_memory_metadata_memory_id ON memory_metadata(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_metadata_tags ON memory_metadata(tags);
CREATE INDEX IF NOT EXISTS idx_memory_metadata_confidence ON memory_metadata(confidence);
CREATE INDEX IF NOT EXISTS idx_memory_metadata_source ON memory_metadata(source);
CREATE INDEX IF NOT EXISTS idx_memory_metadata_weaviate_object_id ON memory_metadata(weaviate_object_id);
CREATE INDEX IF NOT EXISTS idx_memory_metadata_project_id ON memory_metadata(project_id);

-- Memory routes table - stores routing decisions between agents
CREATE TABLE IF NOT EXISTS memory_routes (
    id TEXT PRIMARY KEY,
    source_agent_id TEXT NOT NULL,
    target_agent_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    route_type TEXT NOT NULL, -- direct, broadcast, conditional
    priority INTEGER DEFAULT 2, -- 1=low, 2=normal, 3=high, 4=critical
    status TEXT DEFAULT 'pending', -- pending, delivered, failed, acknowledged
    routing_reason TEXT, -- why this route was chosen
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMP,
    acknowledged_at TIMESTAMP,
    FOREIGN KEY (source_agent_id) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (target_agent_id) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (memory_id) REFERENCES memory_items(id) ON DELETE CASCADE
);

-- Create indexes for routing queries
CREATE INDEX IF NOT EXISTS idx_memory_routes_source_agent ON memory_routes(source_agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_routes_target_agent ON memory_routes(target_agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_routes_memory_id ON memory_routes(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_routes_status ON memory_routes(status);
CREATE INDEX IF NOT EXISTS idx_memory_routes_priority ON memory_routes(priority);
CREATE INDEX IF NOT EXISTS idx_memory_routes_created_at ON memory_routes(created_at);

-- Conversation contexts table - stores conversation state
CREATE TABLE IF NOT EXISTS conversation_contexts (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    context_data TEXT NOT NULL, -- JSON string of context information
    context_type TEXT NOT NULL, -- session, project, user
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP, -- when context expires
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

-- Create indexes for context queries
CREATE INDEX IF NOT EXISTS idx_conversation_contexts_conversation_id ON conversation_contexts(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_contexts_agent_id ON conversation_contexts(agent_id);
CREATE INDEX IF NOT EXISTS idx_conversation_contexts_type ON conversation_contexts(context_type);
CREATE INDEX IF NOT EXISTS idx_conversation_contexts_expires_at ON conversation_contexts(expires_at);

-- Performance metrics table - stores system performance data
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit TEXT, -- seconds, bytes, count, etc.
    tags TEXT, -- JSON string of additional tags
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for metrics queries
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded_at ON performance_metrics(recorded_at);

-- System events table - stores important system events
CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL, -- info, warning, error, critical
    event_message TEXT NOT NULL,
    agent_id TEXT, -- optional, if event is agent-specific
    metadata TEXT, -- JSON string of additional event data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE SET NULL
);

-- Create indexes for event queries
CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_system_events_agent_id ON system_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_system_events_created_at ON system_events(created_at);

-- Create a view for memory search that combines memory items with metadata
CREATE VIEW IF NOT EXISTS memory_search_view AS
SELECT 
    mi.id,
    mi.agent_id,
    mi.content,
    mi.memory_type,
    mi.priority,
    mi.created_at,
    mi.updated_at,
    mi.expires_at,
    mm.tags,
    mm.source,
    mm.confidence,
    mm.embedding_vector,
    mm.vector_dimension,
    mm.weaviate_object_id,
    mm.project_id,
    mm.similarity_threshold
FROM memory_items mi
LEFT JOIN memory_metadata mm ON mi.id = mm.memory_id;

-- Create a view for agent activity that shows recent memory and route activity
CREATE VIEW IF NOT EXISTS agent_activity_view AS
SELECT 
    a.id as agent_id,
    a.name as agent_name,
    a.status as agent_status,
    COUNT(DISTINCT mi.id) as memory_count,
    COUNT(DISTINCT mr.id) as route_count,
    MAX(mi.created_at) as last_memory_created,
    MAX(mr.created_at) as last_route_created
FROM agents a
LEFT JOIN memory_items mi ON a.id = mi.agent_id
LEFT JOIN memory_routes mr ON a.id = mr.source_agent_id OR a.id = mr.target_agent_id
GROUP BY a.id, a.name, a.status;

-- Insert default system agent for internal operations
INSERT OR IGNORE INTO agents (id, name, description, capabilities, status) 
VALUES (
    'system',
    'System Agent',
    'Internal system agent for administrative operations',
    '["admin", "system", "internal"]',
    'active'
);

-- Create trigger to update updated_at timestamp on memory_items
CREATE TRIGGER IF NOT EXISTS update_memory_items_timestamp 
    AFTER UPDATE ON memory_items
BEGIN
    UPDATE memory_items SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create trigger to update updated_at timestamp on memory_metadata
CREATE TRIGGER IF NOT EXISTS update_memory_metadata_timestamp 
    AFTER UPDATE ON memory_metadata
BEGIN
    UPDATE memory_metadata SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create trigger to update updated_at timestamp on agents
CREATE TRIGGER IF NOT EXISTS update_agents_timestamp 
    AFTER UPDATE ON agents
BEGIN
    UPDATE agents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create trigger to update updated_at timestamp on conversation_contexts
CREATE TRIGGER IF NOT EXISTS update_conversation_contexts_timestamp 
    AFTER UPDATE ON conversation_contexts
BEGIN
    UPDATE conversation_contexts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create trigger to update updated_at timestamp on memory_routes
CREATE TRIGGER IF NOT EXISTS update_memory_routes_timestamp 
    AFTER UPDATE ON memory_routes
BEGIN
    UPDATE memory_routes SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
