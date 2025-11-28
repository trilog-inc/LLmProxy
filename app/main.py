from typing import Dict, Any
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import httpx
from .config import settings
from .proxy import sglang_proxy
from .logger import proxy_logger


app = FastAPI(
    title="LLM Proxy",
    description="A transparent proxy for LLM APIs with request/response logging",
    version="1.0.0"
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests"""
    from .logger import generate_request_id
    request_id = generate_request_id()
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.post("/api/chat/completions")
@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """
    Proxy endpoint for chat completions that forwards requests to SGLang server
    """
    try:
        # Parse request body
        request_data = await request.json()
        
        # Get request headers
        headers = dict(request.headers)
        
        # Forward to SGLang server
        response_data, is_streaming = await sglang_proxy.forward_chat_completion(
            request_data, headers
        )
        
        if is_streaming:
            # Return streaming response
            return StreamingResponse(
                response_data,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Return JSON response for non-streaming requests
            return JSONResponse(
                content=response_data,
                headers={"X-Request-ID": request.state.request_id}
            )
    
    except Exception as e:
        proxy_logger.log_error(
            getattr(request.state, 'request_id', 'unknown'),
            f"Error in proxy_chat_completions: {str(e)}",
            {"error_type": type(e).__name__}
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_generic_v1(path: str, request: Request):
    """
    Generic proxy for all other /v1/* endpoints.
    Forwards the request to the configured SGLang server without special handling.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Read incoming request details
    body = await request.body()
    incoming_headers = dict(request.headers)

    # Prepare headers for forwarding (strip hop-by-hop headers)
    hop_by_hop_headers = {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    forward_headers = {
        k: v for k, v in incoming_headers.items() if k.lower() not in hop_by_hop_headers
    }

    target_url = f"{settings.SGLANG_API_BASE.rstrip('/')}/{path}"

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.FORWARD_TIMEOUT)
        ) as client:
            upstream_response = await client.request(
                method=request.method,
                url=target_url,
                headers=forward_headers,
                content=body,
                params=request.query_params,
            )
    except httpx.TimeoutException:
        proxy_logger.log_error(
            request_id,
            "Request timeout in generic /v1 proxy",
            {"target_url": target_url, "timeout": settings.FORWARD_TIMEOUT},
        )
        raise HTTPException(status_code=504, detail="Gateway timeout")
    except httpx.ConnectError as e:
        proxy_logger.log_error(
            request_id,
            "Connection error in generic /v1 proxy",
            {"target_url": target_url, "error": str(e)},
        )
        raise HTTPException(
            status_code=502, detail="Bad gateway - connection failed"
        )
    except Exception as e:
        proxy_logger.log_error(
            request_id,
            "Unexpected error in generic /v1 proxy",
            {"error": str(e), "error_type": type(e).__name__},
        )
        raise HTTPException(status_code=500, detail="Internal server error")

    # Build response back to client
    excluded_response_headers = {
        "content-encoding",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
    }
    response_headers = {
        k: v
        for k, v in upstream_response.headers.items()
        if k.lower() not in excluded_response_headers
    }

    # Preserve request ID header
    if request_id and "x-request-id" not in {k.lower() for k in response_headers}:
        response_headers["X-Request-ID"] = request_id

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=upstream_response.headers.get("content-type"),
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "llm-proxy"}


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "LLM Proxy is running",
        "endpoints": {
            "chat_completions": "/api/chat/completions",
            "health": "/health"
        },
        "target_server": settings.SGLANG_API_BASE
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="LLM Proxy")
    parser.add_argument(
        "--enable-streaming-tool-parser",
        action="store_true",
        help="Enable streaming tool-call transformer for /chat/completions",
    )
    args = parser.parse_args()

    # Toggle runtime flag for streaming tool-call parsing
    if args.enable_streaming_tool_parser:
        settings.ENABLE_STREAMING_TOOL_PARSER = True

    uvicorn.run(
        app,
        host=settings.PROXY_HOST,
        port=settings.PROXY_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
