"""
Custom middleware for the FastAPI application.
Includes CORS, rate limiting, request logging, and error handling.
"""

import time
import logging
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis

from .config import get_settings

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {get_remote_address(request)}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} "
                f"in {process_time:.4f}s"
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"in {process_time:.4f}s - {str(e)}"
            )
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add CSP header for production
        if not get_settings().debug:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self'"
            )
        
        return response


def get_rate_limiter() -> Limiter:
    """Create and configure the rate limiter."""
    settings = get_settings()
    
    # Use Redis for rate limiting if available, otherwise use memory
    try:
        redis_client = redis.from_url(settings.redis_url)
        storage_uri = settings.redis_url
    except Exception as e:
        logger.warning(f"Redis not available for rate limiting: {e}")
        storage_uri = "memory://"
    
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=storage_uri,
        default_limits=[f"{settings.rate_limit_requests}/minute"]
    )
    
    return limiter


def setup_cors_middleware(app) -> None:
    """Setup CORS middleware for the FastAPI app."""
    from .config import get_cors_settings
    
    cors_settings = get_cors_settings()
    
    app.add_middleware(
        CORSMiddleware,
        **cors_settings
    )


def setup_rate_limiting(app) -> Limiter:
    """Setup rate limiting middleware for the FastAPI app."""
    limiter = get_rate_limiter()
    
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    
    return limiter


def setup_custom_middleware(app) -> None:
    """Setup custom middleware for the FastAPI app."""
    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)


def setup_all_middleware(app) -> Limiter:
    """Setup all middleware for the FastAPI app."""
    # Order matters: last added middleware runs first
    
    # 1. Custom middleware (runs first)
    setup_custom_middleware(app)
    
    # 2. Rate limiting middleware
    limiter = setup_rate_limiting(app)
    
    # 3. CORS middleware (runs last, closest to the app)
    setup_cors_middleware(app)
    
    return limiter