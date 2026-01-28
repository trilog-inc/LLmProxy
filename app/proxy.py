import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
import httpx
from fastapi import HTTPException
from .config import settings
from .logger import proxy_logger, generate_request_id
from .streaming_tool_transformer import StreamingToolCallTransformer
from .utils import clean_tool_call_arguments
from .request_queue import backend_request_slot


class SGLangProxy:
    """
    Proxy for handling requests to SGLang backend server.
    Uses per-request HTTP clients to avoid shared state issues.
    """
    
    async def forward_chat_completion(
        self, 
        request_data: Dict[str, Any], 
        headers: Dict[str, str]
    ) -> tuple[AsyncGenerator[bytes, None], bool]:
        """
        Forward chat completion request to SGLang server with queue management.
        
        Returns:
            Tuple of (response_generator, is_streaming)
        """
        request_id = generate_request_id()
        
        # Check if model name contains "kimi" and add thinking parameters if not present
        model = request_data.get('model', '')
        proxy_logger.log_info(f"[Thinking Debug] Model detected: '{model}', checking if it contains 'kimi'...")
        
        if 'kimi' in model.lower():
            proxy_logger.log_info(f"[Thinking Debug] Kimi model detected: '{model}'")
            if 'enable_thinking' not in request_data:
                proxy_logger.log_info(f"[Thinking Debug] Adding enable_thinking=true to request_data")
                request_data['enable_thinking'] = True
            if 'thinking' not in request_data:
                proxy_logger.log_info(f"[Thinking Debug] Adding thinking=true to request_data")
                request_data['thinking'] = True
            if 'chat_template_kwargs' not in request_data:
                proxy_logger.log_info(f"[Thinking Debug] Adding chat_template_kwargs with thinking=true")
                request_data['chat_template_kwargs'] = {'thinking': True}
            proxy_logger.log_info(f"[Thinking Debug] Final request_data: enable_thinking={request_data.get('enable_thinking')}, thinking={request_data.get('thinking')}, chat_template_kwargs={request_data.get('chat_template_kwargs')}")
        else:
            proxy_logger.log_info(f"[Thinking Debug] Not a kimi model, skipping thinking parameters. Model: '{model}'")

        
        # Log the incoming request
        proxy_logger.log_request(request_id, request_data, headers)
        
        # Determine if streaming is requested
        is_streaming = request_data.get('stream', False)
        target_url = f"{settings.SGLANG_API_BASE}/chat/completions"
        
        try:
            if is_streaming:
                # For streaming, the queue management is handled inside the generator
                return self._create_streaming_generator(
                    request_id, target_url, request_data, headers
                )(), True
            else:
                # Non-streaming: acquire slot, make request, return result, release slot
                async with backend_request_slot(request_id):
                    forward_headers = {
                        k: v for k, v in headers.items()
                        if k.lower() not in ['host', 'content-length', 'content-type']
                    }
                    forward_headers['content-type'] = 'application/json'
                    
                    result = await self._handle_regular_request(
                        request_id, target_url, request_data, forward_headers
                    )
                    return result
                
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
                "error_type": type(e).name
            })
            raise HTTPException(status_code=500, detail="Internal server error")
    
    def _create_streaming_generator(self, request_id: str, target_url: str, request_data: Dict[str, Any], headers: Dict[str, str]) -> AsyncGenerator[bytes, None]:
        """
        Create the streaming response generator function.
        This generator acquires the backend slot and keeps it during the entire stream.
        """
        async def stream_generator() -> AsyncGenerator[bytes, None]:
            proxy_logger.log_info(f"[{request_id}] Streaming generator started")
            
            # Acquire backend slot at streaming time, not before
            try:
                async with backend_request_slot(request_id):
                    proxy_logger.log_info(f"[{request_id}] Backend slot acquired INSIDE generator")
                    
                    # Prepare headers for forwarding
                    forward_headers = {
                        k: v for k, v in headers.items()
                        if k.lower() not in ['host', 'content-length', 'content-type']
                    }
                    forward_headers['content-type'] = 'application/json'
                    
                    # Use per-request client with single connection limit
                    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(settings.FORWARD_TIMEOUT),
                        limits=limits
                    ) as client:
                        proxy_logger.log_info(f"[{request_id}] Making request to {target_url}")
                        async with client.stream(
                            "POST",
                            target_url,
                            json=request_data,
                            headers=forward_headers
                        ) as response:
                            if response.status_code != 200:
                                error_body = await response.aread()
                                raise HTTPException(
                                    status_code=response.status_code,
                                    detail=f"Upstream error: {error_body.decode()}"
                                )
                            
                            # Initialize response aggregator
                            aggregator = StreamingResponseAggregator(request_id=request_id)
                            transformer: Optional[StreamingToolCallTransformer] = None
                            done_sent = False

                            if settings.ENABLE_STREAMING_TOOL_PARSER:
                                transformer = StreamingToolCallTransformer()
                            
                            # Process streaming chunks
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

                                    # Run through transformer, then aggregate, normalize finish_reason, log + forward
                                    for out_chunk in transformer.process_chunk(parsed_chunk):
                                        # First update aggregator state with this chunk
                                        aggregator.process_chunk(out_chunk)

                                        # If this choice has used tool_calls overall, force finish_reason="tool_calls"
                                        if 'choices' in out_chunk and aggregator.choices:
                                            for idx, ch in enumerate(out_chunk['choices']):
                                                fr = ch.get('finish_reason')
                                                if fr == 'stop' and idx < len(aggregator.choices or []):
                                                    agg_choice = aggregator.choices[idx]
                                                    tool_calls_accum = agg_choice['delta'].get('tool_calls') or []
                                                    if tool_calls_accum:
                                                        ch['finish_reason'] = 'tool_calls'
                                                        agg_choice['finish_reason'] = 'tool_calls'

                                        proxy_logger.log_stream_chunk(request_id, out_chunk)
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
            except Exception as e:
                proxy_logger.log_error(request_id, f"Error in stream_generator: {str(e)}", {
                    "error_type": type(e).__name__,
                    "error": str(e)
                })
                raise
        
        return stream_generator

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
    
    def __init__(self, request_id: Optional[str] = None):
        self.choices = []
        self.usage = None
        self.system_fingerprint = None
        self.model = None
        self.request_id = request_id
    
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
            if i >= len(self.choices or []):
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
                                new_args_input = tool_call['function'].get('arguments') or ''
                                
                                # Clean new arguments to remove problematic markers
                                cleaned_new_args, was_cleaned = clean_tool_call_arguments(new_args_input, self.request_id)
                                existing['function']['arguments'] = existing_args + cleaned_new_args
                    else:
                        # Add new tool call
                        # Clean arguments in new tool calls too
                        tool_call_copy = tool_call.copy()
                        if 'function' in tool_call_copy and 'arguments' in tool_call_copy.get('function', {}):
                            original_args = tool_call_copy['function']['arguments'] or ''
                            if original_args:
                                cleaned_args, was_cleaned = clean_tool_call_arguments(original_args, self.request_id)
                                tool_call_copy['function']['arguments'] = cleaned_args
                        self.choices[i]['delta']['tool_calls'].append(tool_call_copy)
            
            # Update finish_reason if present
            if 'finish_reason' in choice:
                fr = choice['finish_reason']
                self.choices[i]['finish_reason'] = fr
                # If this choice used tool calls, normalize "stop" to "tool_calls"
                if fr == 'stop':
                    tool_calls_accum = self.choices[i]['delta'].get('tool_calls') or []
                    if tool_calls_accum:
                        self.choices[i]['finish_reason'] = 'tool_calls'
            
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


# Global proxy instance - removed to use per-request proxy instances
# sglang_proxy = SGLangProxy()
