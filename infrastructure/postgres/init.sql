-- Genesis PostgreSQL Database Initialization
-- Includes pgvector for face embeddings and TimescaleDB for time-series data

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search
CREATE EXTENSION IF NOT EXISTS btree_gist;  -- For advanced indexing

-- =============================================================================
-- IDENTITIES TABLE - Face embeddings with pgvector
-- =============================================================================

CREATE TABLE IF NOT EXISTS identities (
    person_id VARCHAR(255) PRIMARY KEY,
    embedding vector(512),  -- Facenet512 produces 512-dimensional embeddings
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    total_appearances INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector similarity index using IVFFlat (Inverted File with Flat compression)
-- Lists parameter: sqrt(total_rows) is a good starting point
-- We'll use 100 for up to 10,000 identities
CREATE INDEX IF NOT EXISTS identities_embedding_idx
ON identities
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Regular indexes
CREATE INDEX IF NOT EXISTS idx_identities_last_seen ON identities(last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_identities_first_seen ON identities(first_seen DESC);

-- Metadata GIN index for flexible querying
CREATE INDEX IF NOT EXISTS idx_identities_metadata ON identities USING GIN (metadata);

COMMENT ON TABLE identities IS 'Known identities with face embeddings for recognition';
COMMENT ON COLUMN identities.embedding IS 'Facenet512 face embedding (512-dimensional vector)';
COMMENT ON COLUMN identities.metadata IS 'Flexible JSON metadata: name, attributes, tags, etc.';

-- =============================================================================
-- SESSIONS TABLE - Person visit sessions
-- =============================================================================

CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id VARCHAR(255) REFERENCES identities(person_id) ON DELETE SET NULL,
    camera_id VARCHAR(255) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    duration_seconds INTEGER,
    zones_visited TEXT[],
    entry_zone VARCHAR(255),
    exit_zone VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sessions_person ON sessions(person_id);
CREATE INDEX IF NOT EXISTS idx_sessions_camera ON sessions(camera_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_duration ON sessions(duration_seconds DESC NULLS LAST);

-- Composite index for person timeline queries
CREATE INDEX IF NOT EXISTS idx_sessions_person_time ON sessions(person_id, start_time DESC);

COMMENT ON TABLE sessions IS 'Person visit sessions tracked across time';

-- =============================================================================
-- IDENTITY EVENTS TABLE - Timeline of identity activities
-- =============================================================================

CREATE TABLE IF NOT EXISTS identity_events (
    event_id BIGSERIAL PRIMARY KEY,
    person_id VARCHAR(255) REFERENCES identities(person_id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,  -- zone_entered, zone_exited, face_recognized, etc.
    zone_name VARCHAR(255),
    camera_id VARCHAR(255),
    confidence REAL,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for timeline queries
CREATE INDEX IF NOT EXISTS idx_events_person ON identity_events(person_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_session ON identity_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON identity_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON identity_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_camera ON identity_events(camera_id, timestamp DESC);

COMMENT ON TABLE identity_events IS 'Timeline of all identity-related events';

-- =============================================================================
-- METRICS TABLE - TimescaleDB hypertable for time-series data
-- =============================================================================

CREATE TABLE IF NOT EXISTS metrics (
    time TIMESTAMPTZ NOT NULL,
    camera_id VARCHAR(255) NOT NULL,
    people_total INTEGER DEFAULT 0,
    people_by_zone JSONB DEFAULT '{}',
    queue_len INTEGER DEFAULT 0,
    avg_wait_sec REAL DEFAULT 0,
    new_faces INTEGER DEFAULT 0,
    recognized_faces INTEGER DEFAULT 0,
    alerts_triggered INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable
-- Partition by time with 1-day chunks
SELECT create_hypertable(
    'metrics',
    'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Create continuous aggregate for hourly rollups
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    camera_id,
    AVG(people_total) AS avg_people,
    MAX(people_total) AS max_people,
    MIN(people_total) AS min_people,
    AVG(queue_len) AS avg_queue,
    MAX(queue_len) AS max_queue,
    SUM(new_faces) AS total_new_faces,
    SUM(recognized_faces) AS total_recognized_faces,
    SUM(alerts_triggered) AS total_alerts
FROM metrics
GROUP BY bucket, camera_id
WITH NO DATA;

-- Refresh policy: update every hour
SELECT add_continuous_aggregate_policy(
    'metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Retention policy: keep raw metrics for 30 days
SELECT add_retention_policy(
    'metrics',
    INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_metrics_camera_time ON metrics(camera_id, time DESC);

COMMENT ON TABLE metrics IS 'Time-series metrics data (TimescaleDB hypertable)';

-- =============================================================================
-- ALERTS TABLE - Alert history
-- =============================================================================

CREATE TABLE IF NOT EXISTS alerts (
    alert_id BIGSERIAL PRIMARY KEY,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,  -- info, warning, critical
    message TEXT NOT NULL,
    camera_id VARCHAR(255),
    person_id VARCHAR(255) REFERENCES identities(person_id) ON DELETE SET NULL,
    context JSONB DEFAULT '{}',
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(255),
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged) WHERE NOT acknowledged;

COMMENT ON TABLE alerts IS 'Alert history with acknowledgment tracking';

-- =============================================================================
-- FACE EMBEDDINGS TABLE - All face sightings for training/analysis
-- =============================================================================

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id BIGSERIAL PRIMARY KEY,
    person_id VARCHAR(255) REFERENCES identities(person_id) ON DELETE SET NULL,
    embedding vector(512) NOT NULL,
    confidence REAL,
    camera_id VARCHAR(255),
    zone_name VARCHAR(255),
    timestamp TIMESTAMPTZ NOT NULL,
    is_new_face BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector index for similarity search
CREATE INDEX IF NOT EXISTS face_embeddings_vector_idx
ON face_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Regular indexes
CREATE INDEX IF NOT EXISTS idx_face_embeddings_person ON face_embeddings(person_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_timestamp ON face_embeddings(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_camera ON face_embeddings(camera_id);

COMMENT ON TABLE face_embeddings IS 'Historical face embeddings for analytics and retraining';

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to search for similar faces using cosine similarity
CREATE OR REPLACE FUNCTION search_similar_faces(
    query_embedding vector(512),
    similarity_threshold REAL DEFAULT 0.6,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    person_id VARCHAR(255),
    distance REAL,
    last_seen TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.person_id,
        (1 - (i.embedding <=> query_embedding)) AS distance,  -- Cosine similarity
        i.last_seen
    FROM identities i
    WHERE (1 - (i.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY i.embedding <=> query_embedding ASC  -- Order by cosine distance
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_similar_faces IS 'Search for similar faces using cosine similarity';

-- Function to update last_seen timestamp
CREATE OR REPLACE FUNCTION update_identity_last_seen()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE identities
    SET last_seen = NEW.timestamp,
        total_appearances = total_appearances + 1,
        updated_at = NOW()
    WHERE person_id = NEW.person_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update last_seen on new identity events
CREATE TRIGGER trigger_update_identity_last_seen
    AFTER INSERT ON identity_events
    FOR EACH ROW
    WHEN (NEW.person_id IS NOT NULL)
    EXECUTE FUNCTION update_identity_last_seen();

-- =============================================================================
-- SAMPLE DATA (for testing)
-- =============================================================================

-- Insert sample identity (commented out for production)
-- INSERT INTO identities (person_id, embedding, metadata) VALUES
-- ('person_sample_001', array_fill(0.0, ARRAY[512])::vector(512),
--  '{"name": "Sample Person", "role": "test"}');

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================

-- Grant permissions to genesis user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO genesis;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO genesis;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO genesis;

-- =============================================================================
-- DATABASE STATISTICS
-- =============================================================================

-- View to show database statistics
CREATE OR REPLACE VIEW database_stats AS
SELECT
    (SELECT COUNT(*) FROM identities) AS total_identities,
    (SELECT COUNT(*) FROM sessions) AS total_sessions,
    (SELECT COUNT(*) FROM identity_events) AS total_events,
    (SELECT COUNT(*) FROM alerts) AS total_alerts,
    (SELECT COUNT(*) FROM face_embeddings) AS total_face_embeddings,
    (SELECT COUNT(*) FROM metrics) AS total_metrics,
    pg_size_pretty(pg_database_size(current_database())) AS database_size;

COMMENT ON VIEW database_stats IS 'Quick overview of database statistics';

-- Initialization complete
SELECT 'Genesis database initialized successfully!' AS status;
