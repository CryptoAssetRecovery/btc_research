"""
Dependency injection for FastAPI endpoints.
Handles authentication, rate limiting, and database connections.
"""

import logging
from typing import Optional, Annotated
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from slowapi import Limiter
import redis.asyncio as redis

from .config import get_settings, APISettings

logger = logging.getLogger(__name__)

# Security schemes
security = HTTPBearer(auto_error=False)


class AuthenticationError(HTTPException):
    """Custom authentication error."""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """Custom authorization error."""
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=message,
        )


def get_limiter(request: Request) -> Limiter:
    """Get the rate limiter from the app state."""
    return request.app.state.limiter


async def get_redis_client() -> redis.Redis:
    """Get Redis client for caching and data storage."""
    settings = get_settings()
    try:
        client = redis.from_url(settings.redis_url)
        # Test connection
        await client.ping()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache service unavailable"
        )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm="HS256")
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify and decode a JWT token."""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        return payload
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise AuthenticationError("Invalid token")


async def verify_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None
) -> Optional[str]:
    """Verify API key for development mode."""
    settings = get_settings()
    
    if not settings.debug:
        # API keys only for development
        return None
    
    if not x_api_key:
        return None
    
    # Simple API key validation for development
    # In production, this would check against a database
    valid_keys = {"dev-key-123", "test-key-456"}
    
    if x_api_key not in valid_keys:
        raise AuthenticationError("Invalid API key")
    
    return x_api_key


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)] = None,
    api_key: Annotated[Optional[str], Depends(verify_api_key)] = None
) -> Optional[dict]:
    """Get the current authenticated user."""
    settings = get_settings()
    
    # Development mode: allow API key authentication
    if settings.debug and api_key:
        return {"user_id": "dev-user", "auth_method": "api_key"}
    
    # Production mode: require JWT token
    if not credentials:
        if not settings.debug:
            raise AuthenticationError("Authentication required")
        return None
    
    token = credentials.credentials
    payload = verify_token(token)
    
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid token payload")
    
    return {"user_id": user_id, "auth_method": "jwt"}


async def require_auth(
    current_user: Annotated[Optional[dict], Depends(get_current_user)]
) -> dict:
    """Require authentication for protected endpoints."""
    if not current_user:
        raise AuthenticationError("Authentication required")
    return current_user


async def get_strategy_manager(request: Request):
    """Get the strategy manager instance from app state."""
    strategy_manager = getattr(request.app.state, 'strategy_manager', None)
    if strategy_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Strategy management service unavailable"
        )
    return strategy_manager


async def get_strategy_runner():
    """Get the strategy runner instance (placeholder for backward compatibility)."""
    # This dependency is kept for backward compatibility but isn't used
    # All strategy operations go through the StrategyManager now
    return None


async def get_paper_trader(request: Request):
    """Get the paper trader instance from app state."""
    paper_trader = getattr(request.app.state, 'paper_trader', None)
    if paper_trader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Paper trading service unavailable"
        )
    return paper_trader


async def get_stream_manager(request: Request):
    """Get the stream manager instance from app state."""
    stream_manager = getattr(request.app.state, 'stream_manager', None)
    if stream_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data streaming service unavailable"
        )
    return stream_manager


def validate_symbol(symbol: str) -> str:
    """Validate and normalize a trading symbol."""
    symbol = symbol.upper().strip()
    
    # Basic validation - extend as needed
    valid_symbols = ["BTC/USDT", "BTC/USDC", "ETH/USDT", "ETH/USDC"]
    
    if symbol not in valid_symbols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid symbol: {symbol}. Valid symbols: {valid_symbols}"
        )
    
    return symbol


def validate_timeframe(timeframe: str) -> str:
    """Validate and normalize a timeframe."""
    timeframe = timeframe.lower().strip()
    
    # Basic validation - extend as needed
    valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    
    if timeframe not in valid_timeframes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid timeframe: {timeframe}. Valid timeframes: {valid_timeframes}"
        )
    
    return timeframe


def validate_pagination(skip: int = 0, limit: int = 100) -> tuple[int, int]:
    """Validate pagination parameters."""
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip parameter must be non-negative"
        )
    
    if limit <= 0 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit parameter must be between 1 and 1000"
        )
    
    return skip, limit


# Common dependency combinations
CommonDeps = Annotated[dict, Depends(get_current_user)]
RequiredAuth = Annotated[dict, Depends(require_auth)]
RedisClient = Annotated[redis.Redis, Depends(get_redis_client)]
Settings = Annotated[APISettings, Depends(get_settings)]