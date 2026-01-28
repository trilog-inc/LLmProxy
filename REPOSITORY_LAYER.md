# Database Repository Layer Design

This document outlines the repository pattern implementation for efficient database operations.

## Repository Pattern Architecture

The repository layer provides a clean abstraction over database operations and handles:
- Async batch operations for performance
- Connection management
- Data validation and transformation
- Error handling and retry logic
- Performance optimization

## Repository Classes

### 1. Base Repository

File: `app/repositories/base.py`

```python
"""Base repository with common database operations."""

from typing import TypeVar, Generic, Type, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, func
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations."""
    
    def __init__(self, session: AsyncSession, model: Type[T]):
        self.session = session
        self.model = model
    
    async def get_by_id(self, id: Any) -> Optional[T]:
        """Get entity by ID."""
        try:
            result = await self.session.get(self.model, id)
            return result
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} by id {id}: {e}")
            raise
    
    async def create(self, **kwargs) -> T:
        """Create a new entity."""
        try:
            instance = self.model(**kwargs)
            self.session.add(instance)
            await self.session.flush()
            return instance
        except SQLAlchemyError as e:
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise
    
    async def bulk_create(self, items: list[dict]) -> list[T]:
        """Bulk create multiple entities."""
        try:
            result = await self.session.execute(
                insert(self.model).returning(self.model),
                items
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Error bulk creating {self.model.__name__}: {e}")
            raise
    
    async def update(self, id: Any, **kwargs) -> Optional[T]:
        """Update entity by ID."""
        try:
            instance = await self.get_by_id(id)
            if instance:
                for key, value in kwargs.items():
                    setattr(instance, key, value)
                await self.session.flush()
            return instance
        except SQLAlchemyError as e:
            logger.error(f"Error updating {self.model.__name__} {id}: {e}")
            raise
    
    async def delete(self, id: Any) -> bool:
        """Delete entity by ID."""
        try:
            instance = await self.get_by_id(id)
            if instance:
                await self.session.delete(instance)
                await self.session.flush()
                return True
            return False
        except SQLAlchemyError as e:
            logger.error(f"Error deleting {self.model.__name__} {id}: {e}")
            raise
    
    async def count(self, **filters) -> int:
        """Count entities matching filters."""
        try:
            query = select(func.count()).select_from(self.model)
            for key, value in filters.items():
                query = query.where(getattr(self.model, key) == value)
            result = await self.session.execute(query)
            return result.scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error counting {self.model.__name__}: {e}")
            raise
```

### 2. Request Repository

File: `app/repositories/request_repository.py`

```python
"""Repository for request-related operations."""

from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, and_
from sqlalchemy.orm import selectinload
from ..models.request import Request
from .base import BaseRepository

class RequestRepository(BaseRepository[Request]):
    """Repository for request operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Request)
    
    async def get_by_request_id(self, request_id: str) -> Optional[Request]:
        """Get request by request_id with related data."""
        query = (
            select(Request)
            .options(
                selectinload(Request.responses),
                selectinload(Request.tool_calls),
                selectinload(Request.errors)
            )
            .where(Request.request_id == request_id)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_list(
        self,
        page: int = 1,
        limit: int = 50,
        model_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        has_errors: Optional[bool] = None,
        has_tool_calls: Optional[bool] = None,
        search: Optional[str] = None
    ) -> Tuple[List[Request], int]:
        """Get paginated list of requests with filters."""
        query = select(Request)
        
        # Apply filters
        if model_name:
            query = query.where(Request.model_name == model_name)
        
        if start_date:
            query = query.where(Request.timestamp >= start_date)
        
        if end_date:
            query = query.where(Request.timestamp <= end_date)
        
        if has_errors is not None:
            if has_errors:
                query = query.where(Request.errors.isnot(None))
        
        if has_tool_calls is not None:
            if has_tool_calls:
                query = query.where(Request.tool_calls_count > 0)
        
        if search:
            # Simple search in request body (can be enhanced)
            query = query.where(Request.request_body.contains(search))
        
        # Count total for pagination
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.session.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        query = query.order_by(desc(Request.timestamp)).offset((page - 1) * limit).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all(), total
    
    async def get_stats(
        self,
        hours: int = 24
    ) -> dict:
        """Get request statistics for the last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Total requests
        total_query = select(func.count()).where(Request.timestamp >= since)
        total_result = await self.session.execute(total_query)
        total_requests = total_result.scalar()
        
        # Streaming requests
        streaming_query = select(func.count()).where(
            and_(Request.timestamp >= since, Request.stream == True)
        )
        streaming_result = await self.session.execute(streaming_query)
        streaming_requests = streaming_result.scalar()
        
        # Average processing time
        avg_time_query = select(func.avg(Request.processing_time_ms)).where(
            and_(Request.timestamp >= since, Request.processing_time_ms.isnot(None))
        )
        avg_time_result = await self.session.execute(avg_time_query)
        avg_processing_time = avg_time_result.scalar()
        
        return {
            "total_requests": total_requests,
            "streaming_requests": streaming_requests,
            "regular_requests": total_requests - streaming_requests,
            "avg_processing_time_ms": avg_processing_time or 0,
            "period_hours": hours
        }
```

### 3. Stream Chunk Repository

File: `app/repositories/stream_chunk_repository.py`

```python
"""Repository for stream chunk operations with batch support."""

from typing import List, Tuple
from collections import deque
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from sqlalchemy.orm import joinedload
from ..models.stream_chunk import StreamChunk
from .base import BaseRepository

class StreamChunkRepository(BaseRepository[StreamChunk]):
    """Repository for stream chunk operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, StreamChunk)
        self._batch_queue: deque = deque()
        self._batch_size = 100
    
    async def add_to_batch(self, chunk_data: dict) -> None:
        """Add chunk to batch queue for bulk insert."""
        self._batch_queue.append(chunk_data)
        
        if len(self._batch_queue) >= self._batch_size:
            await self.flush_batch()
    
    async def flush_batch(self) -> None:
        """Flush all pending chunks to database."""
        if not self._batch_queue:
            return
        
        try:
            await self.bulk_create(list(self._batch_queue))
            self._batch_queue.clear()
        except Exception as e:
            # Log error but don't raise to avoid breaking the main flow
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error flushing chunk batch: {e}")
    
    async def get_by_request_id(
        self,
        request_id: str,
        page: int = 1,
        limit: int = 100
    ) -> Tuple[List[StreamChunk], int]:
        """Get chunks for a request with pagination."""
        from ..models.request import Request
        
        # Get chunks with request join
        query = (
            select(StreamChunk)
            .join(Request)
            .where(Request.request_id == request_id)
            .order_by(StreamChunk.chunk_index)
        )
        
        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        count_result = await self.session.execute(count_query)
        total = count_result.scalar()
        
        # Paginate
        query = query.offset((page - 1) * limit).limit(limit)
        result = await self.session.execute(query)
        return result.scalars().all(), total
    
    async def get_tool_call_chunks(
        self,
        request_id: str,
        tool_call_id: str
    ) -> List[StreamChunk]:
        """Get all chunks for a specific tool call."""
        from ..models.request import Request
        
        query = (
            select(StreamChunk)
            .join(Request)
            .where(
                Request.request_id == request_id,
                StreamChunk.tool_call_id == tool_call_id
            )
            .order_by(StreamChunk.chunk_index)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
```

### 4. Response Repository

File: `app/repositories/response_repository.py`

```python
"""Repository for response operations."""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from ..models.response import Response
from .base import BaseRepository

class ResponseRepository(BaseRepository[Response]):
    """Repository for response operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Response)
    
    async def get_by_request_id(self, request_id: str) -> Optional[Response]:
        """Get response by request_id."""
        from ..models.request import Request
        
        query = (
            select(Response)
            .join(Request)
            .options(selectinload(Response.tool_calls))
            .where(Request.request_id == request_id)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_tool_usage_stats(self, hours: int = 24) -> List[dict]:
        """Get tool usage statistics."""
        from ..models.request import Request
        from sqlalchemy import func, desc as desc_clause
        
        since = datetime.utcnow() - timedelta(hours=hours)
        
        query = (
            select(
                Response.model_name,
                func.count().label("total_calls"),
                func.sum(Response.tool_calls_count).label("total_tool_calls")
            )
            .join(Request)
            .where(Request.timestamp >= since)
            .where(Response.tool_calls_count > 0)
            .group_by(Response.model_name)
            .order_by(desc_clause("total_tool_calls"))
        )
        
        result = await self.session.execute(query)
        return [
            {
                "model": row.model_name,
                "total_requests": row.total_calls,
                "total_tool_calls": row.total_tool_calls
            }
            for row in result.all()
        ]
```

### 5. Unit of Work Pattern

File: `app/repositories/unit_of_work.py`

```python
"""Unit of Work pattern for managing database transactions."""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from .request_repository import RequestRepository
from .response_repository import ResponseRepository
from .stream_chunk_repository import StreamChunkRepository
from .tool_call_repository import ToolCallRepository
from .error_repository import ErrorRepository

class UnitOfWork:
    """Manages database session and repositories for a business transaction."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self._request_repo: Optional[RequestRepository] = None
        self._response_repo: Optional[ResponseRepository] = None
        self._chunk_repo: Optional[StreamChunkRepository] = None
        self._tool_call_repo: Optional[ToolCallRepository] = None
        self._error_repo: Optional[ErrorRepository] = None
    
    @property
    def requests(self) -> RequestRepository:
        """Request repository."""
        if not self._request_repo:
            self._request_repo = RequestRepository(self.session)
        return self._request_repo
    
    @property
    def responses(self) -> ResponseRepository:
        """Response repository."""
        if not self._response_repo:
            self._response_repo = ResponseRepository(self.session)
        return self._response_repo
    
    @property
    def stream_chunks(self) -> StreamChunkRepository:
        """Stream chunk repository."""
        if not self._chunk_repo:
            self._chunk_repo = StreamChunkRepository(self.session)
        return self._chunk_repo
    
    @property
    def tool_calls(self) -> ToolCallRepository:
        """Tool call repository."""
        if not self._tool_call_repo:
            self._tool_call_repo = ToolCallRepository(self.session)
        return self._tool_call_repo
    
    @property
    def errors(self) -> ErrorRepository:
        """Error repository."""
        if not self._error_repo:
            self._error_repo = ErrorRepository(self.session)
        return self._error_repo
    
    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()
    
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.session.rollback()
    
    async def flush_chunks(self) -> None:
        """Flush any pending stream chunks."""
        if self._chunk_repo:
            await self._chunk_repo.flush_batch()
```

## Database Logger Integration

### Enhanced Logger with Database Support

File: `app/logger/database_logger.py`

```python
"""Database logger that extends the existing file logger."""

import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from .proxy_logger import proxy_logger
from ..repositories.unit_of_work import UnitOfWork

class DatabaseLogger:
    """Extends existing logger to also log to database."""
    
    def __init__(self, session_factory: callable):
        self.session_factory = session_factory
        self._enabled = True
    
    async def log_request(self, request_id: str, request_data: Dict[str, Any], 
                         headers: Dict[str, str], **kwargs) -> Optional[str]:
        """Log request to database and call existing logger."""
        # Call existing file logger
        proxy_logger.log_request(request_id, request_data, headers)
        
        if not self._enabled:
            return None
        
        try:
            async with self.session_factory() as session:
                uow = UnitOfWork(session)
                
                # Extract model name from request
                model_name = request_data.get("model")
                is_stream = request_data.get("stream", False)
                
                # Create request record
                request = await uow.requests.create(
                    request_id=request_id,
                    method=kwargs.get("method", "POST"),
                    endpoint=kwargs.get("endpoint", "/v1/chat/completions"),
                    model_name=model_name,
                    stream=is_stream,
                    request_headers=headers,
                    request_body=request_data,
                    client_ip=kwargs.get("client_ip"),
                    user_agent=kwargs.get("user_agent"),
                    queue_wait_time_ms=kwargs.get("queue_wait_time_ms")
                )
                
                await uow.commit()
                return str(request.id)
                
        except Exception as e:
            # Don't let database errors break the main flow
            proxy_logger.log_error(request_id, f"Database logging error: {e}")
            return None
    
    async def log_stream_chunk(self, request_id: str, chunk: Dict[str, Any], 
                              chunk_index: int, source: str = "upstream") -> None:
        """Log stream chunk to database and call existing logger."""
        # Call existing file logger
        proxy_logger.log_stream_chunk(request_id, chunk)
        
        if not self._enabled:
            return
        
        try:
            async with self.session_factory() as session:
                uow = UnitOfWork(session)
                
                # Get request UUID from request_id
                request = await uow.requests.get_by_request_id(request_id)
                if not request:
                    return
                
                # Determine chunk type
                chunk_type = self._determine_chunk_type(chunk)
                tool_call_id = self._extract_tool_call_id(chunk)
                
                # Add to batch queue
                await uow.stream_chunks.add_to_batch({
                    "request_id": request.id,
                    "chunk_index": chunk_index,
                    "chunk_data": chunk,
                    "chunk_source": source,
                    "chunk_type": chunk_type,
                    "tool_call_id": tool_call_id
                })
                
        except Exception as e:
            proxy_logger.log_error(request_id, f"Database chunk logging error: {e}")
    
    async def log_aggregated_response(self, request_id: str, 
                                    aggregated_response: Dict[str, Any]) -> None:
        """Log aggregated response to database and call existing logger."""
        # Call existing file logger
        proxy_logger.log_aggregated_response(request_id, aggregated_response)
        
        if not self._enabled:
            return
        
        try:
            async with self.session_factory() as session:
                uow = UnitOfWork(session)
                
                # Get request
                request = await uow.requests.get_by_request_id(request_id)
                if not request:
                    return
                
                # Extract response data
                choices = aggregated_response.get('choices', [])
                message = choices[0].get('message', {}) if choices else {}
                
                content = message.get('content', '')
                reasoning_content = message.get('reasoning_content', '')
                tool_calls = message.get('tool_calls', [])
                
                # Create response record
                response = await uow.responses.create(
                    request_id=request.id,
                    response_data=aggregated_response,
                    content_length=len(content or []),
                    reasoning_content_length=len(reasoning_content or []),
                    tool_calls_count=len(tool_calls),
                    finish_reason=choices[0].get('finish_reason') if choices else None,
                    model_name=aggregated_response.get('model'),
                    system_fingerprint=aggregated_response.get('system_fingerprint'),
                    usage=aggregated_response.get('usage')
                )
                
                # Create tool call records
                for tool_call in tool_calls:
                    function = tool_call.get('function', {})
                    await uow.tool_calls.create(
                        request_id=request.id,
                        response_id=response.id,
                        tool_call_id=tool_call.get('id', ''),
                        tool_name=function.get('name', ''),
                        tool_arguments=function.get('arguments', ''),
                        chunk_index_start=None,  # Can be updated from chunks
                        chunk_index_end=None
                    )
                
                # Flush any pending chunks
                await uow.flush_chunks()
                await uow.commit()
                
        except Exception as e:
            proxy_logger.log_error(request_id, f"Database response logging error: {e}")
    
    async def log_error(self, request_id: str, error: str, 
                       details: Optional[Dict[str, Any]] = None) -> None:
        """Log error to database and call existing logger."""
        # Call existing file logger
        proxy_logger.log_error(request_id, error, details)
        
        if not self._enabled:
            return
        
        try:
            async with self.session_factory() as session:
                uow = UnitOfWork(session)
                
                # Get request
                request = await uow.requests.get_by_request_id(request_id)
                if not request:
                    return
                
                # Create error record
                await uow.errors.create(
                    request_id=request.id,
                    error_type=details.get("error_type", "Unknown") if details else "Unknown",
                    error_message=error,
                    error_details=details,
                    status_code=details.get("status_code") if details else None
                )
                
                await uow.commit()
                
        except Exception as e:
            # Don't let database errors break error logging
            pass
    
    def _determine_chunk_type(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Determine chunk type from chunk data."""
        if not chunk or "choices" not in chunk:
            return None
        
        choices = chunk.get("choices", [])
        if not choices:
            return None
        
        delta = choices[0].get("delta", {})
        
        if "tool_calls" in delta:
            return "tool_call"
        elif "reasoning_content" in delta:
            return "reasoning_content"
        elif "content" in delta:
            return "content"
        elif chunk.get("finish_reason"):
            return "finish"
        
        return None
    
    def _extract_tool_call_id(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Extract tool call ID from chunk if present."""
        if not chunk or "choices" not in chunk:
            return None
        
        choices = chunk.get("choices", [])
        if not choices:
            return None
        
        delta = choices[0].get("delta", {})
        tool_calls = delta.get("tool_calls", [])
        
        if tool_calls and "id" in tool_calls[0]:
            return tool_calls[0]["id"]
        
        return None
```

## Performance Optimizations

### 1. Batch Operations
- Stream chunks are batched (default 100 chunks)
- Batch flushed automatically when full or on request completion
- Reduces database round trips significantly

### 2. Connection Pooling
- Async connection pool with optimal settings
- Pre-ping connections to avoid stale connections
- Connection recycling to prevent leaks

### 3. Index Strategy
- Composite indexes for common query patterns
- Covering indexes for faster lookups
- Partial indexes for frequently filtered data

### 4. Query Optimization
- Select loading for related data (N+1 prevention)
- Efficient count queries for pagination
- Denormalized data where beneficial

## Error Handling Strategy

1. **Graceful Degradation**: Database errors don't break the main proxy flow
2. **Logging**: All database errors are logged to existing file logger
3. **Circuit Breaker**: Can disable database logging if persistent failures
4. **Retry Logic**: Built into connection pool for transient failures

## Usage Example

```python
from app.database import get_db_session
from app.logger.database_logger import DatabaseLogger

# Create database logger
session_factory = get_db_session
db_logger = DatabaseLogger(session_factory)

# Log request
request_id = generate_request_id()
await db_logger.log_request(request_id, request_data, headers)

# Log stream chunks
chunk_index = 0
async for chunk in stream_response:
    await db_logger.log_stream_chunk(request_id, chunk, chunk_index)
    chunk_index += 1

# Log final response
await db_logger.log_aggregated_response(request_id, aggregated_response)
```

This repository layer provides a solid foundation for efficient database logging while maintaining the existing file logging functionality.