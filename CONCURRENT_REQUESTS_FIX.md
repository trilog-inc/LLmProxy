# Concurrent Request Handling Fix

## Problem Solved

The LLM proxy was experiencing request drops when multiple clients sent requests while the backend SGLang server was processing a previous request. The backend can only handle **1 concurrent request**, but the proxy's architecture allowed request interference through shared state, causing:

- Requests to be silently dropped
- Response chunks to disappear from logs
- Tool calls appearing in `reasoning_content` instead of being properly parsed
- Unpredictable timeout behavior

## Root Causes Identified

1. **Global Singleton Pattern**: The `SGLangProxy` class was instantiated as a global singleton, creating shared state across all requests.

2. **Shared HTTP Client**: Used a single `httpx.AsyncClient` with `max_connections=100`, but the backend only supports 1 concurrent connection.

3. **No Request Queueing**: Requests arrived while the backend was busy but weren't properly queued or isolated.

4. **Race Conditions**: The streaming aggregation and tool parsing logic created timing issues for interleaved requests.

## Solution Implemented

### 1. Request Queueing System (`app/request_queue.py`)

- **`RequestQueueManager`**: Manages a semaphore and tracks active requests
- **`backend_request_slot`**: Context manager that ensures only one request accesses the backend at a time
- **Queue Statistics**: Monitor queue depth, wait times, and active request count
- **Configurable Limits**: MAX_QUEUE_SIZE prevents memory exhaustion

```python
async with backend_request_slot(request_id):
    # Only one request can be in this block at a time
    # Other requests wait in queue with proper timeout handling
```

### 2. Per-Request Isolation (`app/proxy.py`)

- **Removed Global Singleton**: No more `sglang_proxy = SGLangProxy()`
- **Per-Request Proxy Instances**: Each request creates its own `SGLangProxy()` instance
- **Dedicated HTTP Clients**: Each request uses `httpx.AsyncClient` with `max_connections=1`
- **Thread-Safe Streaming**: Stream generators are properly isolated per request

### 3. Configuration Options (`app/config.py`, `.env.example`)

```env
# Queue settings for single-backend server handling
MAX_CONCURRENT_REQUESTS=1          # Set to 1 for SGLang backend
MAX_QUEUE_SIZE=100                 # Prevent memory exhaustion  
QUEUE_TIMEOUT=600                  # 10 minute queue wait limit
```

### 4. Enhanced Logging and Monitoring

- **Queue Wait Times**: Logged for each request to track queuing performance
- **Request State Tracking**: Active requests are tracked with start times and queue duration
- **Queue Status Endpoint**: `GET /queue/status` provides real-time queue metrics
- **Better Error Messages**: Clear HTTP 503 responses with Retry-After headers when queue is full

### 5. Updated Main Application (`app/main.py`)

- **Per-Request Proxy Creation**: `proxy = SGLangProxy()` for each incoming request
- **Proper Error Handling**: HTTP exceptions are re-raised to preserve status codes
- **Queue Status Endpoint**: Monitor queue health

## Files Modified

1. **`app/config.py`**: Added queue configuration settings
2. **`app/request_queue.py`**: New file implementing queue management
3. **`app/proxy.py`**: Removed singleton, added per-request isolation, integrated queueing
4. **`app/main.py`**: Updated to use per-request proxy instances
5. **`.env.example`**: Added queue configuration documentation
6. **`test_concurrent.py`**: New comprehensive test script

## Behavior Changes

### Before
- Requests arriving while backend busy → **Silently dropped or corrupted**
- No visibility into queue state → **Impossible to debug**
- Shared HTTP client → **Race conditions and chunk loss**
- Tool calls in reasoning_content → **Parser failures**

### After
- Requests arriving while backend busy → **Properly queued with timeout**
- Queue status endpoint → **Full visibility and monitoring**
- Per-request clients → **Complete isolation, zero interference**
- Complete chunk logs → **Proper tool call parsing**

## Testing

Use the included test script to verify concurrent handling:

```bash
# Install test dependencies
pip install aiohttp

# Run concurrent request test
python test_concurrent.py --requests 10 --url http://localhost:8000
```

The test will:
1. Send multiple concurrent requests
2. Verify all requests complete successfully
3. Analyze timing to confirm sequential processing
4. Check queue statistics before and after

## Monitoring

### Queue Status
```bash
curl http://localhost:8000/queue/status
```

Response:
```json
{
  "status": "ok",
  "queue_stats": {
    "queue_size": 3,
    "max_queue_size": 100,
    "active_requests": 1,
    "max_concurrent": 1,
    "queue_timeout": 600
  }
}
```

### Log Output
Each request now logs:
```
INFO | [request-id] Backend slot acquired. Queue wait time: 2.34s, Active requests: 1
INFO | [request-id] Backend slot released. Processing time: 5.67s, Queue time: 2.34s
```

## Configuration Tuning

For different backend capabilities:

```env
# For faster backends (reduce queue timeout)
QUEUE_TIMEOUT=300

# For more concurrent backends (if supported)
MAX_CONCURRENT_REQUESTS=2

# For high-traffic scenarios (increase queue depth)
MAX_QUEUE_SIZE=500
```

## Migration Notes

No breaking changes for client applications. The proxy API remains identical, but now:
- Returns HTTP 503 with Retry-After instead of silently dropping requests
- Provides consistent latency rather than unpredictable failures
- Logs all requests properly with complete chunk traces

## Future Enhancements

- **Priority Queue**: Support for prioritizing certain request types
- **Dynamic Scaling**: Auto-adjust queue settings based on backend performance
- **Circuit Breaker**: Detect backend failures and temporarily reject requests
- **Metrics Export**: Prometheus/StatsD integration for monitoring systems