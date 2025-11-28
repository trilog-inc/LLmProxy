I quickly vibe coded this proxy server to clean up tool calls from Kimi-K2-Thinking to Kilo Code when using [SGLang](https://github.com/sgl-project/sglang)( Release Gateway-v0.2.3 ) with the [Ktransformers](https://github.com/kvcache-ai/ktransformers) backend. The SGLang kimi_k2 parsers sometimes leave tool call content in the reasoning payloads, thus ruining the agentic benefit of this model. With this proxy server, I can have 100+ function calls in a task ( with 1 or 2 failures).


# LLM Proxy

A transparent proxy server for OpenAI-compatible LLM APIs, built with FastAPI.  
This proxy sits between your client and a backend such as an SGLang server, forwarding requests, aggregating streaming responses, and providing detailed logging for debugging and analysis.

---

## Table of Contents

- [Features](#features)
- [Architecture & Project Structure](#architecture--project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [From Source](#from-source)
- [Configuration](#configuration)
- [Running the Proxy](#running-the-proxy)
- [Usage](#usage)
  - [Pointing a Client at the Proxy](#pointing-a-client-at-the-proxy)
  - [curl Examples](#curl-examples)
- [Testing](#testing)
- [API Endpoints](#api-endpoints)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **OpenAI-compatible proxy**
  - Drop-in replacement for the OpenAI API base URL, targeting an SGLang (or compatible) backend.
- **Request logging**
  - Captures and logs all incoming requests to `/v1/chat/completions` (and its `/api/chat/completions` alias).
- **Response logging**
  - Logs both streaming and non-streaming responses.
- **Streaming aggregation**
  - Aggregates streaming responses into complete messages for easier analysis and debugging.
- **Transparent streaming support**
  - Handles both streaming (`stream=true`) and non-streaming (`stream=false`) responses.
- **Flexible configuration**
  - Environment-based configuration with sensible defaults via `.env`.
- **Error handling**
  - Comprehensive error handling with detailed logging and configurable timeouts.
- **Request IDs**
  - Each request gets a unique ID for correlation across logs and tooling.
- **Optional streaming tool-call transformer**
  - Optional server-side repair of tool calls embedded in `reasoning_content` for streaming responses.

---

## Architecture & Project Structure

```text
llm-proxy/
├── app/
│   ├── __init__.py                # Package initialization
│   ├── main.py                    # FastAPI application and endpoints
│   ├── config.py                  # Configuration management
│   ├── proxy.py                   # Core proxy logic
│   ├── logger.py                  # Logging utilities
│   └── streaming_tool_transformer.py  # Optional tool-call repair for streaming
├── test_proxy.py                  # Test script for the proxy
├── requirements.txt               # Python dependencies
├── .env.example                   # Example environment configuration
├── README.md                      # Project documentation
└── Docs/                          # Additional design docs and analysis
```

- The FastAPI app lives in `app/main.py`.
- Proxy behavior (forwarding, aggregation, logging) is implemented in `app/proxy.py`.
- Configuration is read from environment variables (see [Configuration](#configuration)) via `app/config.py`.

---

## Installation

### Prerequisites

- **Python**: 3.8 or higher (python 3.11 tested)
- **pip**: Python package manager
- **Virtual environment** (recommended): `venv`, `pyenv`, `conda`, etc.
- **Backend LLM server**:
  - An SGLang server or another OpenAI-compatible backend.
  - Default example: `http://10.0.0.5:60000/v1` (you will likely want to change this).

### From Source

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-org-or-user>/llm-proxy.git
   cd llm-proxy
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create your environment file**

   ```bash
   cp .env.example .env
   # Then edit .env to point to your SGLang / backend server and adjust options
   ```

---

## Configuration

The proxy is configured entirely via environment variables.  
Copy `.env.example` to `.env` and adjust as needed:

```env
# Target SGLang (or OpenAI-compatible) server URL
SGLANG_API_BASE=http://10.0.0.5:60000/v1

# Proxy server settings
PROXY_HOST=0.0.0.0
PROXY_PORT=8000

# Logging settings
LOG_LEVEL=INFO
LOG_FILE=llm_proxy.log
# Enable/disable writing logs to files; when false, logs go to console only
ENABLE_FILE_LOGGING=true

# Timeout settings (seconds)
FORWARD_TIMEOUT=300

# Streaming tool-call transformer (optional)
# Set to true to enable server-side repair of tool calls
# embedded in reasoning_content for streaming responses
ENABLE_STREAMING_TOOL_PARSER=false
```

### Configuration Options

| Variable                     | Description                                                                                  | Default                        |
|-----------------------------|----------------------------------------------------------------------------------------------|--------------------------------|
| `SGLANG_API_BASE`           | Base URL of the target SGLang/OpenAI-compatible server (e.g., `http://host:port/v1`).       | `http://10.0.0.5:60000/v1`    |
| `PROXY_HOST`                | Host interface for the proxy to bind to.                                                    | `0.0.0.0`                      |
| `PROXY_PORT`                | Port for the proxy server.                                                                  | `8000`                         |
| `LOG_LEVEL`                 | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).                           | `INFO`                         |
| `LOG_FILE`                  | Path to the log file for persistent logs (used only when `ENABLE_FILE_LOGGING=true`).      | `llm_proxy.log`               |
| `ENABLE_FILE_LOGGING`       | Toggle for file-based logging; when `false`, logging is console-only.                      | `true`                         |
| `FORWARD_TIMEOUT`           | Timeout (seconds) for forwarding requests to the backend.                                   | `300`                          |
| `ENABLE_STREAMING_TOOL_PARSER` | Enables streaming tool-call repair in responses when set to `true`.                     | `false`                        |

---

## Running the Proxy

You can run the proxy either via the Python module entrypoint or directly with `uvicorn`.

### Option 1: Using Python

```bash
# Default behavior (no streaming tool-call transformation)
python -m app.main

# Enable streaming tool-call transformer for /v1/chat/completions streaming responses
python -m app.main --enable-streaming-tool-parser
```

### Option 2: Using uvicorn directly

```bash
# Default behavior (no streaming tool-call transformation)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Enable streaming tool-call transformer via environment variable
ENABLE_STREAMING_TOOL_PARSER=true \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, the proxy is available at:

```text
http://localhost:8000
```

The OpenAI-compatible API base is:

```text
http://localhost:8000/v1
```

---

## Usage

### Pointing a Client at the Proxy

The proxy is designed to be a drop-in replacement for the OpenAI API base URL.

#### Example: OpenAI Python client (v1.x)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key",  # The proxy does not validate this
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Hello! Tell me a short story."},
    ],
    stream=False,
)

print(response.choices[0].message.content)
```

#### Example: Using `requests` directly

```python
import requests

url = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "default",
    "messages": [
        {"role": "user", "content": "Hello! Tell me a short story."},
    ],
    "stream": False,
}

resp = requests.post(url, json=payload)
resp.raise_for_status()
print(resp.json())
```

- **API Base URL**: `http://localhost:8000`
- **API Key**: Not required (you can leave it blank or use any dummy value).

### curl Examples

#### Non-streaming Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": "Hello! Tell me a short story."
    }],
    "stream": false
  }'

# (Legacy alias, still supported)
# curl -X POST http://localhost:8000/api/chat/completions ...
```

#### Streaming Request

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": "Hello! Tell me a short story."
    }],
    "stream": true
  }'

# (Legacy alias, still supported)
# curl -N -X POST http://localhost:8000/api/chat/completions ...
```

> Note: `-N` disables curl's output buffering so you can see tokens as they stream.

---

## Testing

A small test script is included to validate the proxy behavior:

```bash
# Test regular (non-streaming) requests
python test_proxy.py

# Test streaming requests
python test_proxy.py --stream

# Test health endpoint
python test_proxy.py --health
```

The test script uses `aiohttp`. If you don't already have it installed:

```bash
pip install aiohttp
```

---

## API Endpoints

- **POST `/v1/chat/completions`**
  - Primary proxy endpoint for chat completions.
  - OpenAI-compatible request/response schema.
  - Enhanced logging and optional streaming aggregation.

- **POST `/api/chat/completions`**
  - Legacy/alias route for chat completions.
  - Same behavior as `/v1/chat/completions`.

- **ANY `/v1/{path}`**
  - Transparent pass-through to the SGLang/OpenAI-compatible backend for all other endpoints.
  - Used to forward requests such as `/v1/models`, `/v1/embeddings`, etc.

- **GET `/health`**
  - Simple health-check endpoint for monitoring / readiness probes.

- **GET `/`**
  - Basic service information.

---

## Logging

The proxy logs all requests and responses to help with debugging and analysis.

- **Console logs**
  - Colored, human-readable logs to stdout/stderr.
- **File logs**
-  - Persistent logs written to the file specified by `LOG_FILE` (default: `llm_proxy.log`), when `ENABLE_FILE_LOGGING=true`. When `ENABLE_FILE_LOGGING=false`, no log files are created and logging is console-only.
- **Request IDs**
  - Every request is tagged with a unique ID for traceability.
- **Streaming aggregation logs**
  - Final aggregated responses (for streaming) are logged with content, `reasoning_content`, and `tool_calls`.

Logs typically include:

- Request method, URL, and headers (sanitized).
- Request body (JSON payload).
- Individual stream chunks (at `DEBUG` level).
- Aggregated final response for streaming requests.
- Errors, timeouts, and exceptions with full context.

---

## Troubleshooting

### Connection Errors

If requests fail with connection errors, ensure your SGLang/backend server is reachable from where the proxy is running.

```bash
# Check proxy pass-through (if proxy is running)
curl http://localhost:8000/v1/models

# Check backend SGLang (directly, bypassing proxy)
curl http://10.0.0.5:60000/v1/models
```

Update `SGLANG_API_BASE` in `.env` if necessary.

### Timeout Issues

For long-running operations, increase the forward timeout:

```env
FORWARD_TIMEOUT=600
```

### Port Already in Use

If `8000` is already in use on your machine, change the proxy port:

```env
PROXY_PORT=8001
```

Then start the proxy again and point your clients to `http://localhost:8001`.

### No Logs Appearing

If you do not see the expected logs:

```env
# In .env
LOG_LEVEL=DEBUG  # Enable verbose logging
```

Ensure the process has write permissions to the directory containing `LOG_FILE`.

---

## Dependencies

Core dependencies (see `requirements.txt` for exact versions):

- **FastAPI** — Web framework for building the proxy.
- **Uvicorn** — ASGI server to run the FastAPI app.
- **HTTPX** — Async HTTP client used to forward requests to the backend.
- **Loguru** — Structured and convenient logging.
- **Pydantic** — Data validation and settings management.
- **python-dotenv** — Loads configuration from `.env` files.

Testing/development may additionally use:

- **aiohttp** — For the included test script.
- Other utilities as defined in `requirements.txt`.

---

## Limitations

- Special handling (logging, aggregation, optional tool-call repair) is implemented only for:
  - `/v1/chat/completions`
  - `/api/chat/completions`
- Other `/v1/*` endpoints are proxied transparently without extra aggregation or inspection.
- No built-in authentication or authorization.
- No rate limiting or request quota enforcement.
- No request/response caching.
- Basic error handling (no retry logic or backoff).

---

## Future Enhancements

Planned or potential improvements:

- Support for additional OpenAI-style endpoints with enhanced logging/aggregation.
- Authentication / API key middleware.
- Rate limiting and quota management.
- Request/response caching.
- Request modification hooks (e.g., prompt rewriting, policy injection).
- Metrics and monitoring (Prometheus, OpenTelemetry, etc.).
- Live configuration reloading.
- Multiple backend support and routing rules.

---

## Contributing

Contributions, bug reports, and feature requests are welcome.

1. Fork this repository on GitHub.
2. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Implement your changes.
4. Add tests where appropriate.
5. Run tests and linting locally.
6. Open a pull request with a clear description and context.

---

## License

This project is provided as-is for development and testing purposes.  
MIT
