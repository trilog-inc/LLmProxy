import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from loguru import logger
from .config import settings


class ProxyLogger:
    def __init__(self):
        # Configure loguru logger
        logger.remove()  # Remove default handler
        
        # Add console handler
        logger.add(
            lambda msg: print(msg, flush=True),
            level=settings.LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
        
        # Add main file handler if file logging is enabled and LOG_FILE is specified
        if settings.ENABLE_FILE_LOGGING and settings.LOG_FILE:
            logger.add(
                settings.LOG_FILE,
                rotation="5 MB",
                retention="10 days",
                level=settings.LOG_LEVEL,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                encoding="utf-8"
            )
        
        if settings.ENABLE_FILE_LOGGING:
            # Dedicated chunk log file for detailed streaming inspection
            # Only messages with extra.stream_chunk == True will be written here.
            logger.add(
                "llm_proxy_chunks.log",
                rotation="5 MB",
                retention="10 days",
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                encoding="utf-8",
                filter=lambda record: record["extra"].get("stream_chunk") is True,
            )

            # Dedicated server-chunk log file for raw upstream LLM chunks
            # Only messages with extra.server_chunk == True will be written here.
            logger.add(
                "llm_proxy_server_chunks.log",
                rotation="5 MB",
                retention="10 days",
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                encoding="utf-8",
                filter=lambda record: record["extra"].get("server_chunk") is True,
            )
    
    def log_request(self, request_id: str, request_data: Dict[str, Any], headers: Dict[str, str]):
        """Log incoming request details"""
        safe_headers = {k: v for k, v in headers.items() 
                       if k.lower() not in ['authorization', 'cookie', 'x-api-key']}
        
        logger.info(f"[{request_id}] Incoming request to /api/chat/completions")
        logger.debug(f"[{request_id}] Request headers: {json.dumps(safe_headers, indent=2)}")
        logger.debug(f"[{request_id}] Request body: {json.dumps(request_data, indent=2, default=str)}")
    
    def log_stream_chunk(self, request_id: str, chunk: Dict[str, Any]):
        """Log individual streaming response chunk (parsed JSON)"""
        logger.bind(stream_chunk=True).debug(
            f"[{request_id}] Stream chunk: {json.dumps(chunk, indent=2, default=str)}"
        )
    
    def log_stream_raw_line(self, request_id: str, line: str, source: str = "upstream"):
        """Log raw SSE data line for debugging the streaming transformer."""
        logger.bind(stream_chunk=True).debug(
            f"[{request_id}] [{source}] SSE line: {line}"
        )

    def log_server_chunk(self, request_id: str, data: str):
        """Log raw upstream LLM chunks before any SSE/transform processing."""
        logger.bind(server_chunk=True).debug(
            f"[{request_id}] RAW server chunk: {data!r}"
        )
    
    def log_aggregated_response(self, request_id: str, aggregated_response: Dict[str, Any]):
        """Log the final aggregated response from streaming"""
        logger.info(f"[{request_id}] Final aggregated response:")
        
        # Extract key information
        choices = aggregated_response.get('choices', [])
        if choices:
            message = choices[0].get('message', {})
            content = message.get('content', '')
            reasoning_content = message.get('reasoning_content', '')
            tool_calls = message.get('tool_calls', [])
            
            log_data = {
                'request_id': request_id,
                'content_length': len(content) if content else 0,
                'reasoning_content_length': len(reasoning_content) if reasoning_content else 0,
                'tool_calls_count': len(tool_calls),
                'full_message': message
            }
            
            logger.info(f"Aggregated response info: {json.dumps(log_data, indent=2, default=str)}")
    
    def log_error(self, request_id: str, error: str, details: Optional[Dict[str, Any]] = None):
        """Log errors"""
        error_data = {
            'request_id': request_id,
            'error': error,
            'details': details or {}
        }
        logger.error(f"Error occurred: {json.dumps(error_data, indent=2, default=str)}")


# Global logger instance
proxy_logger = ProxyLogger()


def generate_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())
