"""
Configuration management for the FastAPI server.
Handles environment-based settings for development vs production.
"""

import os
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """API configuration settings with environment variable support."""
    
    # Server Configuration
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=True, description="Debug mode")
    reload: bool = Field(default=True, description="Auto-reload on changes")
    
    # Security Configuration
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="JWT secret key"
    )
    access_token_expire_minutes: int = Field(
        default=30, 
        description="JWT token expiration in minutes"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="API key header name"
    )
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5000", "http://localhost:5001", "http://127.0.0.1:5000", "http://127.0.0.1:5001", "http://192.168.1.58:5001", "http://192.168.1.58:5000"],
        description="Allowed CORS origins"
    )
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    cors_allow_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed CORS methods"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        description="Allowed CORS headers"
    )
    
    # Rate Limiting Configuration
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Database Configuration
    redis_url: str = Field(default="redis://redis:6379", description="Redis connection URL")
    
    # API Configuration
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 prefix")
    docs_url: Optional[str] = Field(default="/docs", description="OpenAPI docs URL")
    redoc_url: Optional[str] = Field(default="/redoc", description="ReDoc URL")
    openapi_url: Optional[str] = Field(default="/openapi.json", description="OpenAPI schema URL")
    
    # Application Configuration
    app_name: str = Field(default="BTC Research Paper Trading API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    app_description: str = Field(
        default="RESTful API for managing paper trading strategies and monitoring performance",
        description="Application description"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Data Paths
    config_dir: str = Field(
        default="btc_research/config",
        description="Strategy configuration directory"
    )
    data_dir: str = Field(
        default="data",
        description="Data storage directory"
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "API_"
        case_sensitive = False
        extra = "ignore"  # Allow extra environment variables


# Global settings instance
settings = APISettings()


def get_settings() -> APISettings:
    """Get the current API settings."""
    return settings


def is_production() -> bool:
    """Check if running in production mode."""
    return not settings.debug


def get_cors_settings() -> dict:
    """Get CORS configuration for FastAPI."""
    return {
        "allow_origins": settings.cors_origins,
        "allow_credentials": settings.cors_allow_credentials,
        "allow_methods": settings.cors_allow_methods,
        "allow_headers": settings.cors_allow_headers,
    }


def get_docs_config() -> dict:
    """Get documentation configuration."""
    if is_production():
        # Disable docs in production for security
        return {
            "docs_url": None,
            "redoc_url": None,
            "openapi_url": None,
        }
    
    return {
        "docs_url": settings.docs_url,
        "redoc_url": settings.redoc_url,
        "openapi_url": settings.openapi_url,
    }