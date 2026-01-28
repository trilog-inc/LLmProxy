# LLM Proxy Database Logging Architecture

## Overview
This document outlines the architecture for migrating from file-based logging to a PostgreSQL database with a web frontend for log analysis and visualization.

## Current State Analysis

### Existing Logging Implementation
- **File-based logging** using Loguru with multiple log files
- **Request correlation** via UUID-based request IDs
- **Structured logging** with JSON payloads
- **Streaming support** with individual chunk logging
- **Tool call transformation** logging at chunk level
- **Queue management** for request throttling

### Key Components
1. [`app/logger.py`](app/logger.py:1) - Centralized logging interface
2. [`app/proxy.py`](app/proxy.py:1) - Request/response processing and logging
3. [`app/streaming_tool_transformer.py`](app/streaming_tool_transformer.py:1) - Chunk transformation and logging
4. [`app/main.py`](app/main.py:1) - FastAPI endpoints and middleware

## Database Schema Design

### Core Tables

#### 1. `requests` Table
Stores incoming request metadata and payload.

```sql
CREATE TABLE requests (
    id UUID PRIMARY KEY,
    request_id VARCHAR(64) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    method VARCHAR(10) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    model_name VARCHAR(100),
    stream BOOLEAN NOT NULL DEFAULT false,
    request_headers JSONB,
    request_body JSONB,
    client_ip INET,
    user_agent TEXT,
    queue_wait_time_ms INTEGER,
    processing_time_ms INTEGER
);

CREATE INDEX idx_requests_timestamp ON requests (timestamp DESC);
CREATE INDEX idx_requests_request_id ON requests (request_id);
CREATE INDEX idx_requests_model ON requests (model_name);
```

#### 2. `responses` Table  
Stores aggregated responses for both streaming and non-streaming requests.

```sql
CREATE TABLE responses (
    id UUID PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    response_data JSONB NOT NULL,
    content_length INTEGER,
    reasoning_content_length INTEGER,
    tool_calls_count INTEGER DEFAULT 0,
    finish_reason VARCHAR(50),
    model_name VARCHAR(100),
    system_fingerprint VARCHAR(100),
    usage JSONB
);

CREATE INDEX idx_responses_request_id ON responses (request_id);
CREATE INDEX idx_responses_timestamp ON responses (timestamp DESC);
CREATE INDEX idx_responses_tool_calls ON responses (tool_calls_count);
```

#### 3. `stream_chunks` Table
Stores individual streaming chunks for detailed analysis.

```sql
CREATE TABLE stream_chunks (
    id UUID PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    chunk_data JSONB NOT NULL,
    chunk_source VARCHAR(20) NOT NULL, -- 'upstream', 'transformed', 'client'
    chunk_type VARCHAR(50), -- 'content', 'reasoning_content', 'tool_call', 'done'
    tool_call_id VARCHAR(100)
);

CREATE INDEX idx_chunks_request_id ON stream_chunks (request_id, chunk_index);
CREATE INDEX idx_chunks_timestamp ON stream_chunks (timestamp DESC);
CREATE INDEX idx_chunks_source ON stream_chunks (chunk_source, chunk_type);
```

#### 4. `tool_calls` Table  
Denormalized table for efficient tool call analysis.

```sql
CREATE TABLE tool_calls (
    id UUID PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    response_id UUID REFERENCES responses(id) ON DELETE CASCADE,
    tool_call_id VARCHAR(100) NOT NULL,
    tool_name VARCHAR(100) NOT NULL,
    tool_arguments JSONB,
    chunk_index_start INTEGER,
    chunk_index_end INTEGER
);

CREATE INDEX idx_tool_calls_request_id ON tool_calls (request_id);
CREATE INDEX idx_tool_calls_tool_name ON tool_calls (tool_name);
CREATE INDEX idx_tool_calls_timestamp ON tool_calls (request_id, tool_call_id);
```

#### 5. `errors` Table
Stores error information for failed requests.

```sql
CREATE TABLE errors (
    id UUID PRIMARY KEY,
    request_id UUID REFERENCES requests(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    error_details JSONB,
    status_code INTEGER
);

CREATE INDEX idx_errors_request_id ON errors (request_id);
CREATE INDEX idx_errors_timestamp ON errors (timestamp DESC);
CREATE INDEX idx_errors_type ON errors (error_type);
```

## System Architecture

### Component Overview

```mermaid
graph TB
    Client[Client Application]
    Proxy[LLM Proxy Server]
    DB[(PostgreSQL Database)]
    Logger[Database Logger]
    API[Log Query API]
    Frontend[Web Frontend]
    
    Client --> Proxy
    Proxy --> Logger
    Logger --> DB
    API --> DB
    Frontend --> API
    
    subgraph "Proxy Server"
        Proxy
        Logger
    end
    
    subgraph "Logging & Analysis"
        DB
        API
        Frontend
    end
```

### Data Flow

```mermaid
sequenceDiagram
    participant Client
    participant Proxy
    participant Logger
    participant Database
    participant Backend
    
    Client->>Proxy: HTTP Request
    Proxy->>Logger: log_request()
    Logger->>Database: INSERT INTO requests
    
    Proxy->>Backend: Forward Request
    Backend-->>Proxy: Stream Response
    
    loop For each chunk
        Proxy->>Logger: log_stream_chunk()
        Logger->>Database: INSERT INTO stream_chunks
        Proxy->>Client: Forward chunk
    end
    
    Proxy->>Logger: log_aggregated_response()
    Logger->>Database: INSERT INTO responses
    Logger->>Database: INSERT INTO tool_calls (if applicable)
```

## Implementation Strategy

### Phase 1: Database Layer
1. **Add PostgreSQL dependencies** to requirements.txt
2. **Create database models** using SQLAlchemy async ORM
3. **Implement async database connection** pool
4. **Create database migration** system (Alembic)

### Phase 2: Logging Integration
1. **Extend ProxyLogger** to support database logging
2. **Implement async batch inserts** for performance
3. **Maintain backward compatibility** with file logging
4. **Add configuration options** for database settings

### Phase 3: REST API
1. **Create log query endpoints** for the web frontend
2. **Implement pagination** and filtering
3. **Add search capabilities** across requests/responses
4. **Create streaming endpoints** for real-time updates

### Phase 4: Web Frontend
1. **Choose frontend framework** (React + TypeScript recommended)
2. **Design responsive UI** for log browsing
3. **Implement request list view** with pagination
4. **Create detailed request view** with chunk inspection
5. **Add tool call visualization**

### Phase 5: Deployment & Operations
1. **Create Docker Compose** setup
2. **Add database backup** and maintenance procedures
3. **Implement log retention** policies
4. **Add monitoring** and alerting

## Technology Stack

### Backend
- **Database**: PostgreSQL 14+ with JSONB support
- **ORM**: SQLAlchemy 2.0+ with async support
- **Migrations**: Alembic
- **API**: FastAPI (existing)

### Frontend
- **Framework**: React 18+ with TypeScript
- **UI Library**: Material-UI or Ant Design
- **State Management**: React Query + Zustand
- **Charts**: Recharts for analytics

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Database**: Official PostgreSQL image
- **Web Server**: Nginx for static files

## Database Configuration

### Environment Variables
```env
# Database Settings
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/llm_proxy_logs
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
DB_POOL_RECYCLE=3600

# Logging Configuration
ENABLE_DATABASE_LOGGING=true
ENABLE_FILE_LOGGING=false  # Can disable file logging when DB is ready
DB_BATCH_SIZE=100  # Batch inserts for performance
DB_FLUSH_INTERVAL=5  # Seconds between batch flushes
```

### Performance Optimization
- **Connection pooling** with asyncpg
- **Batch inserts** for high-volume chunk logging
- **Materialized views** for common analytics queries
- **Partitioning** for large tables by timestamp
- **Indexes** on frequently queried columns

## API Endpoints

### Log Query API
```python
# List requests with pagination and filtering
GET /api/logs/requests?page=1&limit=50&model=default&start_date=2024-01-01

# Get single request with full details
GET /api/logs/requests/{request_id}

# Get stream chunks for a request
GET /api/logs/requests/{request_id}/chunks?page=1&limit=100

# Get tool calls for a request
GET /api/logs/requests/{request_id}/tool-calls

# Search across requests/responses
GET /api/logs/search?q=error&model=default&start_date=2024-01-01

# Analytics endpoints
GET /api/analytics/summary?date_range=7d
GET /api/analytics/tool-usage?date_range=7d
GET /api/analytics/error-rates?date_range=7d
```

## Web Frontend Features

### Dashboard
- **Request volume charts** over time
- **Tool call frequency** by function
- **Error rate monitoring**
- **Average response times**

### Request Browser
- **Filterable request list** with pagination
- **Search by request_id, content, or model**
- **Date range filtering**
- **Export capabilities** (CSV, JSON)

### Request Detail View
- **Request metadata** (timestamps, model, etc.)
- **Request payload** with syntax highlighting
- **Aggregated response** with tool call visualization
- **Stream chunk viewer** with timeline
- **Tool call argument explorer**
- **Error information** if applicable

### Real-time Monitoring (Optional)
- **Live request feed** with WebSocket updates
- **Streaming response visualization**
- **Active tool calls** panel

## Migration Strategy

### Phase 1: Dual Logging (Recommended)
- Keep existing file logging enabled
- Add database logging in parallel
- Compare data integrity between both systems
- Performance testing under load

### Phase 2: File Logging Deprecation
- Once database logging is stable
- Disable file logging for new deployments
- Keep file logging as optional fallback

## Security Considerations

### Data Protection
- **Sanitize sensitive data** before database storage
- **Encrypt database connections** (SSL/TLS)
- **Implement row-level security** for multi-tenant setups
- **Regular backups** with encryption

### Access Control
- **API authentication** for log query endpoints
- **Role-based access** for different user types
- **Rate limiting** on API endpoints
- **Audit logging** for data access

## Monitoring & Maintenance

### Database Health
- **Connection pool monitoring**
- **Query performance analysis**
- **Index usage statistics**
- **Disk space management**

### Log Retention
- **Automated cleanup** of old data
- **Archival strategy** for long-term storage
- **Size-based partitioning** for large deployments

## Future Enhancements

### Analytics Features
- **Token usage tracking** by model and user
- **Tool call success rates**
- **Prompt/response similarity analysis**
- **Cost estimation** based on token usage

### Advanced Features
- **Full-text search** across all logged data
- **Custom dashboards** with saved filters
- **Alerting system** for anomalies
- **Integration** with external monitoring tools

---

This architecture provides a scalable, efficient solution for logging and analyzing LLM proxy data while maintaining the existing functionality and enabling powerful web-based analysis tools.