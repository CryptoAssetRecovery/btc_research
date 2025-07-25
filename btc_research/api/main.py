"""
Main FastAPI application for the BTC Research Paper Trading API.
Handles strategy management, real-time data, and performance monitoring.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import get_settings, get_docs_config
from .middleware import setup_all_middleware
from .models import HealthCheck, ErrorResponse, HealthStatus
from .routers import strategies, statistics, config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def serialize_error_response(error_response: ErrorResponse) -> Dict[str, Any]:
    """Serialize ErrorResponse with proper datetime handling."""
    data = error_response.model_dump()
    # Convert datetime to ISO string
    if 'timestamp' in data and isinstance(data['timestamp'], datetime):
        data['timestamp'] = data['timestamp'].isoformat()
    return data

# Application state
app_state = {
    "start_time": time.time(),
    "strategy_manager": None,
    "paper_trader": None,
    "stream_manager": None,
    "redis_client": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    settings = get_settings()
    
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    try:
        # Initialize Redis connection
        import redis.asyncio as redis
        app_state["redis_client"] = redis.from_url(settings.redis_url)
        await app_state["redis_client"].ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        app_state["redis_client"] = None
    
    # Initialize live trading components
    try:
        # Import the live trading components
        from btc_research.live.stream_manager import StreamManager
        from btc_research.live.paper_trader import PaperTrader
        from btc_research.api.strategy_manager import StrategyManager
        
        # Initialize StreamManager for real-time data
        symbols = ["BTC/USDT"]  # Default symbols to stream
        timeframes = ["1m", "5m", "15m", "1h"]  # Default timeframes
        app_state["stream_manager"] = StreamManager(symbols=symbols, timeframes=timeframes)
        
        # Initialize base PaperTrader (individual strategies will create their own instances)
        app_state["paper_trader"] = PaperTrader(initial_balance=0)  # Base instance for API queries
        
        # Initialize StrategyManager with required components
        if app_state["redis_client"]:
            app_state["strategy_manager"] = StrategyManager(
                paper_trader=app_state["paper_trader"],
                stream_manager=app_state["stream_manager"],
                redis_client=app_state["redis_client"]
            )
            
            # Start the strategy manager
            await app_state["strategy_manager"].initialize()
            logger.info("StrategyManager initialized successfully")
        else:
            logger.warning("StrategyManager not initialized: Redis connection required")
        
        logger.info("Live trading components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize live trading components: {e}")
        # Don't fail startup, just log the error
        app_state["stream_manager"] = None
        app_state["paper_trader"] = None
        app_state["strategy_manager"] = None
    
    # Inject app state for dependency access
    inject_app_state(app)
    
    logger.info("FastAPI server startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI server")
    
    # Cleanup Redis connection
    if app_state["redis_client"]:
        await app_state["redis_client"].close()
    
    # Cleanup live trading components
    if app_state["strategy_manager"]:
        try:
            await app_state["strategy_manager"].shutdown()
            logger.info("StrategyManager shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down StrategyManager: {e}")
    
    if app_state["stream_manager"]:
        try:
            # Stream manager cleanup would go here
            # await app_state["stream_manager"].stop()
            pass
        except Exception as e:
            logger.error(f"Error shutting down StreamManager: {e}")
    
    logger.info("FastAPI server shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    docs_config = get_docs_config()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        lifespan=lifespan,
        **docs_config
    )
    
    # Setup middleware (order matters)
    limiter = setup_all_middleware(app)
    
    # Add routers
    app.include_router(
        strategies.router,
        prefix=settings.api_v1_prefix
    )
    app.include_router(
        statistics.router,
        prefix=settings.api_v1_prefix
    )
    app.include_router(
        config.router,
        prefix=settings.api_v1_prefix
    )
    
    # Error handlers
    setup_error_handlers(app)
    
    # Health check endpoint
    setup_health_check(app)
    
    # Store limiter in app state for dependency injection
    app.state.limiter = limiter
    
    return app


def setup_error_handlers(app: FastAPI) -> None:
    """Setup custom error handlers."""
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=serialize_error_response(ErrorResponse(
                error=exc.detail,
                detail=f"HTTP {exc.status_code} error occurred"
            ))
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=serialize_error_response(ErrorResponse(
                error="Validation Error",
                detail=str(exc)
            ))
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=serialize_error_response(ErrorResponse(
                error="Internal Server Error",
                detail="An unexpected error occurred"
            ))
        )


def setup_health_check(app: FastAPI) -> None:
    """Setup health check endpoint."""
    
    @app.get("/health", response_model=HealthCheck, tags=["health"])
    async def health_check():
        """
        Health check endpoint.
        
        Returns the current health status of the API and its components.
        """
        settings = get_settings()
        current_time = time.time()
        uptime = current_time - app_state["start_time"]
        
        # Check component health
        components = {}
        overall_status = HealthStatus.HEALTHY
        
        # Check Redis
        if app_state["redis_client"]:
            try:
                await app_state["redis_client"].ping()
                components["redis"] = HealthStatus.HEALTHY
            except Exception:
                components["redis"] = HealthStatus.UNHEALTHY
                overall_status = HealthStatus.DEGRADED
        else:
            components["redis"] = HealthStatus.UNHEALTHY
            overall_status = HealthStatus.DEGRADED
        
        # Check live trading components
        components["stream_manager"] = (
            HealthStatus.HEALTHY if app_state["stream_manager"] else HealthStatus.UNHEALTHY
        )
        components["paper_trader"] = (
            HealthStatus.HEALTHY if app_state["paper_trader"] else HealthStatus.UNHEALTHY
        )
        components["strategy_manager"] = (
            HealthStatus.HEALTHY if app_state["strategy_manager"] else HealthStatus.UNHEALTHY
        )
        
        # If any critical component is down, mark as degraded
        if any(status == HealthStatus.UNHEALTHY for status in components.values()):
            if overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        return HealthCheck(
            status=overall_status,
            version=settings.app_version,
            uptime=uptime,
            components=components,
            details={
                "timestamp": current_time,
                "debug_mode": settings.debug,
                "api_prefix": settings.api_v1_prefix
            }
        )
    
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information."""
        settings = get_settings()
        
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "docs_url": settings.docs_url if settings.debug else None,
            "health_url": "/health",
            "api_prefix": settings.api_v1_prefix
        }


# Create the FastAPI app instance
app = create_app()


# Dependency injection helpers for the routers
def get_app_state() -> Dict[str, Any]:
    """Get the current application state."""
    return app_state


def get_strategy_manager():
    """Get the strategy manager instance."""
    return app_state.get("strategy_manager")


def get_paper_trader():
    """Get the paper trader instance."""
    return app_state.get("paper_trader")


def get_stream_manager():
    """Get the stream manager instance."""
    return app_state.get("stream_manager")


# Update app state with instances to be accessible via dependencies
def inject_app_state(app: FastAPI):
    """Inject app state into the FastAPI app for dependency access."""
    app.state.strategy_manager = app_state.get("strategy_manager")
    app.state.paper_trader = app_state.get("paper_trader")
    app.state.stream_manager = app_state.get("stream_manager")
    app.state.redis_client = app_state.get("redis_client")


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "btc_research.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )