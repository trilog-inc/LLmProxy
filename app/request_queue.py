"""
Request queue manager for handling single-backend server concurrency.

This module implements a proper request queueing system to ensure that
only one request is forwarded to the backend SGLang server at a time,
preventing request dropping and ensuring fair processing order.
"""

import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import HTTPException
import httpx

from .config import settings
from .logger import proxy_logger


class RequestQueueManager:
    """
    Manages a queue of requests to ensure only one request is processed
    by the backend server at a time. This solves the issue where the
    SGLang backend can only handle 1 concurrent request.
    """
    
    def __init__(self):
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        self._queue = asyncio.Queue(maxsize=settings.MAX_QUEUE_SIZE)
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._request_counter = 0
        
    async def acquire_slot(self, request_id: str) -> bool:
        """
        Acquire a slot to forward a request to the backend.
        
        Args:
            request_id: Unique ID for the request
            
        Returns:
            True if slot acquired, False if queue is full
            
        Raises:
            HTTPException: If queue is full or timeout occurs
        """
        if self._queue.full():
            proxy_logger.log_error(
                request_id,
                "Request queue is full",
                {
                    "queue_size": self._queue.qsize(),
                    "max_queue_size": settings.MAX_QUEUE_SIZE,
                    "active_requests": len(self._active_requests)
                }
            )
            raise HTTPException(
                status_code=503,
                detail=f"Server is busy. Queue is full ({settings.MAX_QUEUE_SIZE} requests). Please try again later.",
                headers={"Retry-After": "5"}
            )
        
        # Add request to queue first
        queue_entry = {
            "request_id": request_id,
            "timestamp": asyncio.get_event_loop().time(),
            "event": asyncio.Event()
        }
        
        try:
            # Wait for available slot with timeout
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=settings.QUEUE_TIMEOUT
            )
            
            # Mark request as active
            self._active_requests[request_id] = {
                "start_time": asyncio.get_event_loop().time(),
                "queue_time": asyncio.get_event_loop().time() - queue_entry["timestamp"]
            }
            
            proxy_logger.log_info(
                f"[{request_id}] Backend slot acquired. "
                f"Queue wait time: {self._active_requests[request_id]['queue_time']:.2f}s, "
                f"Active requests: {len(self._active_requests)}"
            )
            
            return True
            
        except asyncio.TimeoutError:
            proxy_logger.log_error(
                request_id,
                "Timeout waiting for backend slot",
                {
                    "queue_timeout": settings.QUEUE_TIMEOUT,
                    "queue_size": self._queue.qsize(),
                    "active_requests": len(self._active_requests)
                }
            )
            raise HTTPException(
                status_code=503,
                detail=f"Server timeout waiting for available slot. Please try again later.",
                headers={"Retry-After": "10"}
            )
    
    def release_slot(self, request_id: str):
        """
        Release a backend slot after request completion.
        
        Args:
            request_id: Unique ID for the completed request
        """
        if request_id in self._active_requests:
            request_info = self._active_requests[request_id]
            total_time = asyncio.get_event_loop().time() - request_info["start_time"]
            
            proxy_logger.log_info(
                f"[{request_id}] Backend slot released. "
                f"Processing time: {total_time:.2f}s, "
                f"Queue time: {request_info['queue_time']:.2f}s"
            )
            
            del self._active_requests[request_id]
        
        # Release the semaphore to allow next request
        try:
            self._semaphore.release()
        except ValueError:
            # Semaphore was already released (shouldn't happen, but be defensive)
            proxy_logger.log_error(request_id, "Attempted to release semaphore that was not acquired")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get current queue statistics for monitoring.
        
        Returns:
            Dictionary with queue metrics
        """
        return {
            "queue_size": self._queue.qsize(),
            "max_queue_size": settings.MAX_QUEUE_SIZE,
            "active_requests": len(self._active_requests),
            "max_concurrent": settings.MAX_CONCURRENT_REQUESTS,
            "queue_timeout": settings.QUEUE_TIMEOUT
        }


# Global queue manager instance
queue_manager = RequestQueueManager()


@asynccontextmanager
async def backend_request_slot(request_id: str):
    """
    Context manager for acquiring and releasing backend slots.
    
    Usage:
        async with backend_request_slot(request_id):
            # Make request to backend server
            # Only one request can be in this block at a time
    """
    try:
        await queue_manager.acquire_slot(request_id)
        yield
    finally:
        queue_manager.release_slot(request_id)