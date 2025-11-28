import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Target SGLANG server URL
    SGLANG_API_BASE: str = "http://10.0.0.5:60000/v1"
    
    # Proxy server settings
    PROXY_HOST: str = "0.0.0.0"
    PROXY_PORT: int = 8000

    # Streaming tool-call transformer
    ENABLE_STREAMING_TOOL_PARSER: bool = False
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "llm_proxy.log"
    ENABLE_FILE_LOGGING: bool = True
    
    # Timeout settings
    FORWARD_TIMEOUT: int = 300  # 5 minutes for long-running LLM requests
    
    # CORS settings
    ALLOWED_ORIGINS: list[str] = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
