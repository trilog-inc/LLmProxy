import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
import httpx
from fastapi import HTTPException
from .config import settings
from .logger import proxy_logger, generate_request_id
from .streaming_tool_transformer import StreamingToolCallTransformer


class SGLangProxy:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.FORWARD_TIMEOUT),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
    
    async def forward_chat_completion(
        self, 
        request_data: Dict[str, Any], 
        headers: Dict[str, str]
    ) -> tuple[AsyncGenerator[bytes, None], bool]:
        """
        Forward chat completion request to SGLang server
        
        Returns:
            Tuple of (response_generator, is_streaming)
        """
        request_id = generate_request_id()
        
        # Log the incoming request
        proxy_logger.log_request(request_id, request_data, headers)
        
        # Determine if streaming is requested
        is_streaming = request_data.get('stream', False)
        
        try:
            # Prepare headers for forwarding
            forward_headers = {
                k: v for k, v in headers.items()
                if k.lower() not in ['host', 'content-length', 'content-type']
            }
            forward_headers['content-type'] = 'application/json'
            
            # Make request to SGLang server
            target_url = f"{settings.SGLANG_API_BASE}/chat/completions"
            
            if is_streaming:
                return await self._handle_streaming_request(
                    request_id, target_url, request_data, forward_headers
                )
            else:
                return await self._handle_regular_request(
                    request_id, target_url, request_data, forward_headers
                )
                
        except httpx.TimeoutException:
            proxy_logger.log_error(request_id, "Request timeout", {
                "target_url": target_url,
                "timeout": settings.FORWARD_TIMEOUT
            })
            raise HTTPException(status_code=504, detail="Gateway timeout")
        except httpx.ConnectError as e:
            proxy_logger.log_error(request_id, "Connection error", {
                "target_url": target_url,
                "error": str(e)
            })
            raise HTTPException(status_code=502, detail="Bad gateway - connection failed")
        except Exception as e:
            proxy_logger.log_error(request_id, "Unexpected error", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _handle_streaming_request(
        self,
        request_id: str,
        target_url: str,
        request_data: Dict[str, Any],
        headers: Dict[str, str]
    ) -> tuple[AsyncGenerator[bytes, None], bool]:
        """Handle streaming chat completion requests"""
        
        async def stream_generator() -> AsyncGenerator[bytes, None]:
            async with httpx.AsyncClient(timeout=httpx.Timeout(settings.FORWARD_TIMEOUT)) as client:
                async with client.stream(
                    "POST",
                    target_url,
                    json=request_data,
                    headers=headers
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Upstream error: {error_body.decode()}"
                        )
                    
                    # Initialize response aggregator
                    aggregator = StreamingResponseAggregator()
                    transformer: Optional[StreamingToolCallTransformer] = None
                    done_sent = False

                    if settings.ENABLE_STREAMING_TOOL_PARSER:
                        transformer = StreamingToolCallTransformer()
                    
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue

                        chunk_str = chunk.decode("utf-8")
                        proxy_logger.log_server_chunk(request_id, chunk_str)
                        
                        # If transformer is disabled, pass through raw chunks as before
                        if not settings.ENABLE_STREAMING_TOOL_PARSER or transformer is None:
                            # Log the raw SSE lines derived from this chunk
                            for line in chunk_str.strip().split("\n"):
                                if line.startswith("data: "):
                                    json_data = line[6:]  # Remove "data: " prefix
                                    if json_data != "[DONE]":
                                        try:
                                            parsed_chunk = json.loads(json_data)
                                            proxy_logger.log_stream_chunk(request_id, parsed_chunk)
                                            aggregator.process_chunk(parsed_chunk)
                                        except json.JSONDecodeError:
                                            proxy_logger.log_error(
                                                request_id,
                                                "Invalid JSON in stream chunk",
                                                {"chunk": json_data[:100]},
                                            )
                            # Forward original bytes unchanged
                            yield chunk
                            continue

                        # Transformer enabled: rebuild tool_calls before forwarding
                        for line in chunk_str.strip().split("\n"):
                            if not line.startswith("data: "):
                                # For now, ignore non-data lines (or forward as-is if needed)
                                continue

                            json_data = line[6:]
                            if json_data == "[DONE]":
                                # Flush any pending reconstructed chunks
                                for pending in transformer.flush_pending():
                                    proxy_logger.log_stream_chunk(request_id, pending)
                                    aggregator.process_chunk(pending)
                                    out_data = json.dumps(pending, separators=(",", ":"))
                                    yield f"data: {out_data}\n\n".encode("utf-8")
                                
                                # Forward the done sentinel and mark as sent
                                yield b"data: [DONE]\n\n"
                                done_sent = True
                                continue

                            try:
                                parsed_chunk = json.loads(json_data)
                            except json.JSONDecodeError:
                                proxy_logger.log_error(
                                    request_id,
                                    "Invalid JSON in stream chunk (transformer path)",
                                    {"chunk": json_data[:100]},
                                )
                                # Fallback: forward original line if parsing failed
                                yield (line + "\n").encode("utf-8")
                                continue

                            # Run through transformer, then log + aggregate + forward
                            for out_chunk in transformer.process_chunk(parsed_chunk):
                                proxy_logger.log_stream_chunk(request_id, out_chunk)
                                aggregator.process_chunk(out_chunk)
                                out_data = json.dumps(out_chunk, separators=(",", ":"))
                                yield f"data: {out_data}\n\n".encode("utf-8")

                    # End of stream: if transformer enabled and [DONE] never seen,
                    # flush any pending reconstructed chunks.
                    if settings.ENABLE_STREAMING_TOOL_PARSER and transformer is not None and not done_sent:
                        for pending in transformer.flush_pending():
                            proxy_logger.log_stream_chunk(request_id, pending)
                            aggregator.process_chunk(pending)
                            out_data = json.dumps(pending, separators=(",", ":"))
                            yield f"data: {out_data}\n\n".encode("utf-8")
                    
                    # Log the final aggregated response
                    aggregated = aggregator.get_final_response()
                    if aggregated:
                        proxy_logger.log_aggregated_response(request_id, aggregated)
        
        return stream_generator(), True
    
    async def _handle_regular_request(
        self,
        request_id: str,
        target_url: str,
        request_data: Dict[str, Any],
        headers: Dict[str, str]
    ) -> tuple[Dict[str, Any], bool]:
        """Handle non-streaming chat completion requests"""
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.FORWARD_TIMEOUT)) as client:
            response = await client.post(
                target_url,
                json=request_data,
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Upstream error: {response.text}"
                )
            
            response_data = response.json()
            
            # Log the response
            proxy_logger.log_aggregated_response(request_id, response_data)
            
            return response_data, False


class StreamingResponseAggregator:
    """Aggregates streaming response chunks into a complete message"""
    
    def __init__(self):
        self.choices = []
        self.usage = None
        self.system_fingerprint = None
        self.model = None
    
    def process_chunk(self, chunk: Dict[str, Any]):
        """Process a single streaming chunk"""
        if 'choices' not in chunk:
            return
        
        # Initialize choices if this is the first chunk
        if not self.choices:
            self.choices = [
                {
                    'index': choice.get('index', 0),
                    'delta': {},
                    'finish_reason': None,
                    'logprobs': choice.get('logprobs', None)
                }
                for choice in chunk['choices']
            ]
        
        # Process each choice in the chunk
        for i, choice in enumerate(chunk['choices']):
            if i >= len(self.choices):
                continue
            
            delta = choice.get('delta', {})
            
            # Accumulate content
            if 'content' in delta:
                current_content = self.choices[i]['delta'].get('content', '') or ''
                new_content = delta.get('content') or ''
                self.choices[i]['delta']['content'] = current_content + new_content
            
            # Accumulate reasoning_content
            if 'reasoning_content' in delta:
                current_reasoning = self.choices[i]['delta'].get('reasoning_content', '') or ''
                new_reasoning = delta.get('reasoning_content') or ''
                self.choices[i]['delta']['reasoning_content'] = current_reasoning + new_reasoning
            
            # Accumulate tool_calls
            tool_calls_delta = delta.get('tool_calls') or []
            if tool_calls_delta:
                if 'tool_calls' not in self.choices[i]['delta']:
                    self.choices[i]['delta']['tool_calls'] = []
                
                for tool_call in tool_calls_delta:
                    # Find existing tool call by index or append new one
                    existing_idx = None
                    for idx, existing in enumerate(self.choices[i]['delta']['tool_calls']):
                        if existing.get('index') == tool_call.get('index'):
                            existing_idx = idx
                            break
                    
                    if existing_idx is not None:
                        # Append to existing tool call
                        existing = self.choices[i]['delta']['tool_calls'][existing_idx]
                        if 'function' in tool_call:
                            if 'function' not in existing:
                                existing['function'] = {'name': '', 'arguments': ''}
                            if 'name' in tool_call['function']:
                                existing_name = existing['function'].get('name') or ''
                                new_name = tool_call['function'].get('name') or ''
                                existing['function']['name'] = existing_name + new_name
                            if 'arguments' in tool_call['function']:
                                existing_args = existing['function'].get('arguments') or ''
                                new_args = tool_call['function'].get('arguments') or ''
                                existing['function']['arguments'] = existing_args + new_args
                    else:
                        # Add new tool call
                        self.choices[i]['delta']['tool_calls'].append(tool_call.copy())
            
            # Update finish_reason if present
            if 'finish_reason' in choice:
                self.choices[i]['finish_reason'] = choice['finish_reason']
            
            # Update logprobs if present
            if 'logprobs' in choice:
                self.choices[i]['logprobs'] = choice['logprobs']
        
        # Store other metadata from chunk
        if 'usage' in chunk:
            self.usage = chunk['usage']
        if 'system_fingerprint' in chunk:
            self.system_fingerprint = chunk['system_fingerprint']
        if 'model' in chunk:
            self.model = chunk['model']
    
    def get_final_response(self) -> Optional[Dict[str, Any]]:
        """Get the final aggregated response"""
        if not self.choices:
            return None
        
        # Convert deltas to messages
        choices_data = []
        for choice in self.choices:
            delta = choice['delta']
            message = {
                'role': delta.get('role', 'assistant'),
                'content': delta.get('content', ''),
                'reasoning_content': delta.get('reasoning_content', ''),
                'tool_calls': delta.get('tool_calls', [])
            }
            
            choices_data.append({
                'index': choice['index'],
                'message': message,
                'finish_reason': choice['finish_reason'],
                'logprobs': choice['logprobs']
            })
        
        response = {
            'id': 'chatcmpl-' + generate_request_id()[:8],  # Generate a simple ID
            'object': 'chat.completion',
            'created': None,  # Will be set by the system
            'model': self.model or 'unknown',
            'choices': choices_data,
            'usage': self.usage,
            'system_fingerprint': self.system_fingerprint
        }
        
        return response


# Global proxy instance
sglang_proxy = SGLangProxy()
