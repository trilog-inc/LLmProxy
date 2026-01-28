# Database Schema Implementation

This document provides the detailed implementation for the PostgreSQL database schema using SQLAlchemy ORM.

## Dependencies to Add

```
# Database
sqlalchemy>=2.0.0
asyncpg>=0.29.0
alembic>=1.12.0

# For JSON handling and validation
psycopg2-binary>=2.9.9
```

## Database Models

### 1. Database Configuration

File: `app/database.py`

```python
"""Database configuration and session management."""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool
from .config import settings

# SQLAlchemy models base
Base = declarative_base()

# Database configuration
def get_database_url() -> str:
    """Get database URL from environment or settings."""
    return os.getenv(
        "DATABASE_URL",
        f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}@"
        f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )

# Create async engine with connection pooling
engine = create_async_engine(
    get_database_url(),
    poolclass=AsyncAdaptedQueuePool,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=settings.DB_POOL_RECYCLE,
    echo=settings.DB_ECHO,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Dependency for FastAPI
async def get_db_session() -> AsyncSession:
    """Get database session for dependency injection."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

### 2. Request Model

File: `app/models/request.py`

```python
"""Request model for logging incoming API requests."""

import uuid
from datetime import datetime
from sqlalchemy import Column, BigInteger, Boolean, DateTime, Integer, String, Text, JSON, Index
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.orm import relationship
from ..database import Base


class Request(Base):
    """Store incoming request metadata and payload."""
    
    __tablename__ = "requests"
    __table_args__ = (
        Index("idx_requests_timestamp", "timestamp"),
        Index("idx_requests_request_id", "request_id"),
        Index("idx_requests_model", "model_name"),
        {"schema": "llm_proxy"}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(64), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Request metadata
    method = Column(String(10), nullable=False)
    endpoint = Column(String(255), nullable=False)
    model_name = Column(String(100))
    stream = Column(Boolean, nullable=False, default=False)
    
    # Headers and payload
    request_headers = Column(JSON)
    request_body = Column(JSON)
    
    # Client information
    client_ip = Column(INET)
    user_agent = Column(Text)
    
    # Performance metrics
    queue_wait_time_ms = Column(Integer)  # Time spent waiting in queue
    processing_time_ms = Column(Integer)  # Total processing time
    
    # Relationships
    responses = relationship("Response", back_populates="request", cascade="all, delete-orphan")
    stream_chunks = relationship("StreamChunk", back_populates="request", cascade="all, delete-orphan")
    tool_calls = relationship("ToolCall", back_populates="request", cascade="all, delete-orphan")
    errors = relationship("Error", back_populates="request", cascade="all, delete-orphan")

    @property
    def status(self) -> str:
        """Get request status based on relationships."""
        if self.errors:
            return "error"
        if self.responses:
            return "completed"
        return "processing"

    @property
    def total_tokens(self) -> int:
        """Get total token usage if available."""
        if self.responses and self.responses[0].usage:
            usage = self.responses[0].usage
            return usage.get("total_tokens", 0)
        return 0

    def __repr__(self) -> str:
        return f"<Request(id={self.id}, request_id={self.request_id}, model={self.model_name})>"
```

### 3. Response Model

File: `app/models/response.py`

```python
"""Response model for storing aggregated responses."""

import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String, Text, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from ..database import Base


class Response(Base):
    """Store aggregated responses for both streaming and non-streaming requests."""
    
    __tablename__ = "responses"
    __table_args__ = (
        Index("idx_responses_request_id", "request_id"),
        Index("idx_responses_timestamp", "timestamp"),
        Index("idx_responses_tool_calls", "tool_calls_count"),
        {"schema": "llm_proxy"}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("llm_proxy.requests.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Complete response data
    response_data = Column(JSON, nullable=False)
    
    # Content metrics
    content_length = Column(Integer, default=0)
    reasoning_content_length = Column(Integer, default=0)
    tool_calls_count = Column(Integer, default=0)
    
    # Response metadata
    finish_reason = Column(String(50))
    model_name = Column(String(100))
    system_fingerprint = Column(String(100))
    usage = Column(JSON)  # Token usage, etc.
    
    # Relationships
    request = relationship("Request", back_populates="responses")
    tool_calls = relationship("ToolCall", back_populates="response", cascade="all, delete-orphan")

    @property
    def content_preview(self) -> str:
        """Get a preview of the response content."""
        if self.response_data and "choices" in self.response_data:
            choices = self.response_data["choices"]
            if choices and "message" in choices[0]:
                content = choices[0]["message"].get("content", "")
                return content[:200] + "..." if len(content) > 200 else content
        return ""

    @property
    def tool_calls_list(self) -> list:
        """Get list of tool calls from response data."""
        if self.response_data and "choices" in self.response_data:
            choices = self.response_data["choices"]
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("tool_calls", [])
        return []

    def __repr__(self) -> str:
        return f"<Response(id={self.id}, request_id={self.request_id}, tool_calls={self.tool_calls_count})>"
```

### 4. Stream Chunk Model

File: `app/models/stream_chunk.py`

```python
"""Stream chunk model for storing individual streaming response chunks."""

import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String, JSON, ForeignKey, Text, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from ..database import Base


class StreamChunk(Base):
    """Store individual streaming response chunks for detailed analysis."""
    
    __tablename__ = "stream_chunks"
    __table_args__ = (
        Index("idx_chunks_request_id", "request_id", "chunk_index"),
        Index("idx_chunks_timestamp", "timestamp"),
        Index("idx_chunks_source_type", "chunk_source", "chunk_type"),
        CheckConstraint(
            "chunk_source IN ('upstream', 'transformed', 'client')",
            name="valid_chunk_source"
        ),
        {"schema": "llm_proxy"}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("llm_proxy.requests.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Chunk data and metadata
    chunk_data = Column(JSON, nullable=False)
    chunk_source = Column(String(20), nullable=False)  # 'upstream', 'transformed', 'client'
    chunk_type = Column(String(50))  # 'content', 'reasoning_content', 'tool_call', 'done'
    tool_call_id = Column(String(100))
    
    # Relationships
    request = relationship("Request", back_populates="stream_chunks")

    @property
    def delta_content(self) -> str:
        """Extract delta content from chunk data."""
        if self.chunk_data and "choices" in self.chunk_data:
            choices = self.chunk_data["choices"]
            if choices and "delta" in choices[0]:
                delta = choices[0]["delta"]
                return delta.get("content", "") or delta.get("reasoning_content", "")
        return ""

    @property
    def is_tool_call_chunk(self) -> bool:
        """Check if this chunk contains tool call data."""
        if self.chunk_data and "choices" in self.chunk_data:
            choices = self.chunk_data["choices"]
            if choices and "delta" in choices[0]:
                delta = choices[0]["delta"]
                return "tool_calls" in delta
        return False

    def __repr__(self) -> str:
        return f"<StreamChunk(id={self.id}, request_id={self.request_id}, index={self.chunk_index})>"
```

### 5. Tool Call Model

File: `app/models/tool_call.py`

```python
"""Tool call model for denormalized tool call analysis."""

import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from ..database import Base


class ToolCall(Base):
    """Denormalized table for efficient tool call analysis."""
    
    __tablename__ = "tool_calls"
    __table_args__ = (
        Index("idx_tool_calls_request_id", "request_id"),
        Index("idx_tool_calls_tool_name", "tool_name"),
        Index("idx_tool_calls_timestamp", "request_id", "tool_call_id"),
        {"schema": "llm_proxy"}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("llm_proxy.requests.id"), nullable=False, index=True)
    response_id = Column(UUID(as_uuid=True), ForeignKey("llm_proxy.responses.id"))
    
    # Tool call identification
    tool_call_id = Column(String(100), nullable=False, index=True)
    tool_name = Column(String(100), nullable=False, index=True)
    
    # Tool arguments and execution context
    tool_arguments = Column(JSON)
    
    # Chunk tracking for debugging
    chunk_index_start = Column(Integer)
    chunk_index_end = Column(Integer)
    
    # Relationships
    request = relationship("Request", back_populates="tool_calls")
    response = relationship("Response", back_populates="tool_calls")

    @hybrid_property
    def is_successful(self) -> bool:
        """Check if tool call has valid arguments."""
        return self.tool_arguments is not None and len(self.tool_arguments) > 0

    @hybrid_property
    def argument_preview(self) -> str:
        """Get preview of tool arguments."""
        if self.tool_arguments:
            import json
            args_str = json.dumps(self.tool_arguments, indent=2)
            return args_str[:200] + "..." if len(args_str) > 200 else args_str
        return ""

    def __repr__(self) -> str:
        return f"<ToolCall(id={self.id}, request_id={self.request_id}, tool={self.tool_name})>"
```

### 6. Error Model

File: `app/models/error.py`

```python
"""Error model for storing failed request information."""

import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from ..database import Base


class Error(Base):
    """Store error information for failed requests."""
    
    __tablename__ = "errors"
    __table_args__ = (
        Index("idx_errors_request_id", "request_id"),
        Index("idx_errors_timestamp", "timestamp"),
        Index("idx_errors_type", "error_type"),
        {"schema": "llm_proxy"}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("llm_proxy.requests.id"), index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Error information
    error_type = Column(String(100), nullable=False, index=True)
    error_message = Column(Text, nullable=False)
    error_details = Column(JSON)
    status_code = Column(Integer)
    
    # Relationships
    request = relationship("Request", back_populates="errors")

    @property
    def is_timeout(self) -> bool:
        """Check if error is a timeout."""
        return self.status_code == 504 or "timeout" in self.error_type.lower()

    @property
    def is_connection_error(self) -> bool:
        """Check if error is a connection error."""
        return self.status_code == 502 or "connection" in self.error_type.lower()

    def __repr__(self) -> str:
        return f"<Error(id={self.id}, request_id={self.request_id}, type={self.error_type})>"
```

### 7. Model Imports (init file)

File: `app/models/__init__.py`

```python
"""SQLAlchemy models for database logging."""

from .request import Request
from .response import Response
from .stream_chunk import StreamChunk
from .tool_call import ToolCall
from .error import Error

__all__ = ["Request", "Response", "StreamChunk", "ToolCall", "Error"]
```

## Database Migration Setup

### Alembic Configuration

File: `alembic.ini`

```ini
[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql+asyncpg://user:password@localhost:5432/llm_proxy_logs

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

### Alembic Environment

File: `migrations/env.py`

```python
"""Alembic migration environment."""

import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.database import Base
from app.config import settings

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata
target_metadata = Base.metadata

# Override sqlalchemy.url with environment variable
def get_url():
    return os.getenv("DATABASE_URL", settings.DATABASE_URL)

config.set_main_option("sqlalchemy.url", get_url())


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Initial Migration Script

After setting up Alembic, run:

```bash
# Initialize Alembic
alembic init migrations

# Create initial migration
alembic revision --autogenerate -m "Initial database schema"

# Apply migration
alembic upgrade head
```

This will generate the migration script that creates all tables with their indexes and constraints.

## Database Configuration

Add these to `.env.example`:

```env
# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/llm_proxy_logs
DB_HOST=localhost
DB_PORT=5432
DB_NAME=llm_proxy_logs
DB_USER=user
DB_PASSWORD=password

# Connection Pooling
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
DB_POOL_RECYCLE=3600
DB_ECHO=false

# Logging Configuration
ENABLE_DATABASE_LOGGING=true
ENABLE_FILE_LOGGING=true  # Keep file logging as backup initially
DB_BATCH_SIZE=100
DB_FLUSH_INTERVAL=5

# Data Retention (days)
DB_LOG_RETENTION_DAYS=30
```

## Repository Layer

The next step is to implement the repository layer for database operations, which will handle:
- Async batch inserts for performance
- Connection management
- Error handling
- Data validation

This schema provides a solid foundation for efficient logging and analysis of LLM proxy data.