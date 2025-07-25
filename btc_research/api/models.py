"""
Pydantic models for API request/response validation.
Defines all data structures used in the FastAPI endpoints.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict
from decimal import Decimal


class HealthStatus(str, Enum):
    """Health check status enum."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class StrategyStatus(str, Enum):
    """Strategy execution status enum."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class OrderSide(str, Enum):
    """Order side enum."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Order status enum."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# Base Models

class BaseResponse(BaseModel):
    """Base response model."""
    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})
    
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})
    
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Health Check Models

class HealthCheck(BaseModel):
    """Health check response model."""
    status: HealthStatus
    version: str
    uptime: float
    components: Dict[str, HealthStatus]
    details: Optional[Dict[str, Any]] = None


# Authentication Models

class TokenRequest(BaseModel):
    """Token request model."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# Strategy Models

class StrategyConfig(BaseModel):
    """Strategy configuration model."""
    name: str
    symbol: str
    timeframes: List[str]
    parameters: Dict[str, Any]
    risk_management: Optional[Dict[str, Any]] = None
    
    @field_validator('timeframes')
    @classmethod
    def validate_timeframes(cls, v):
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        for tf in v:
            if tf not in valid_timeframes:
                raise ValueError(f'Invalid timeframe: {tf}')
        return v


class StrategyStartRequest(BaseModel):
    """Strategy start request model."""
    config_path: str
    initial_balance: Optional[float] = 10000.0
    risk_per_trade: Optional[float] = 0.02
    max_positions: Optional[int] = 1


class StrategyInfo(BaseModel):
    """Strategy information model."""
    id: str
    name: str
    config_path: str
    status: StrategyStatus
    symbol: str
    timeframes: List[str]
    created_at: datetime
    updated_at: datetime
    initial_balance: float
    current_balance: float
    total_trades: int
    active_positions: int


class StrategyStats(BaseModel):
    """Strategy statistics model."""
    strategy_id: str
    performance: Dict[str, Union[float, int]]
    trading: Dict[str, Union[float, int]]
    risk: Dict[str, Union[float, int]]
    updated_at: datetime


class StrategyListResponse(BaseResponse):
    """Strategy list response model."""
    strategies: List[StrategyInfo]
    total: int


class StrategyResponse(BaseResponse):
    """Single strategy response model."""
    strategy: StrategyInfo


class StrategyStatsResponse(BaseResponse):
    """Strategy statistics response model."""
    stats: StrategyStats


# Trading Models

class Position(BaseModel):
    """Position model."""
    symbol: str
    side: OrderSide
    size: Decimal
    average_price: Decimal
    market_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    unrealized_pnl_pct: Optional[float] = None
    opened_at: datetime
    updated_at: datetime


class Trade(BaseModel):
    """Trade model."""
    id: str
    strategy_id: str
    symbol: str
    side: OrderSide
    size: Decimal
    price: Decimal
    commission: Decimal
    realized_pnl: Optional[Decimal] = None
    timestamp: datetime


class Order(BaseModel):
    """Order model."""
    id: str
    strategy_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    size: Decimal
    price: Optional[Decimal] = None
    status: OrderStatus
    filled_size: Decimal = Decimal('0')
    average_fill_price: Optional[Decimal] = None
    commission: Decimal = Decimal('0')
    created_at: datetime
    updated_at: datetime


class PositionListResponse(BaseResponse):
    """Position list response model."""
    positions: List[Position]
    total: int


class TradeListResponse(BaseResponse):
    """Trade list response model."""
    trades: List[Trade]
    total: int
    skip: int
    limit: int


class OrderListResponse(BaseResponse):
    """Order list response model."""
    orders: List[Order]
    total: int
    skip: int
    limit: int


# Market Data Models

class OHLCV(BaseModel):
    """OHLCV candle model."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class MarketData(BaseModel):
    """Market data model."""
    symbol: str
    timeframe: str
    current_price: Decimal
    change_24h: Optional[float] = None
    volume_24h: Optional[Decimal] = None
    last_update: datetime
    candles: Optional[List[OHLCV]] = None


class MarketDataResponse(BaseResponse):
    """Market data response model."""
    data: MarketData


# Configuration Models

class ConfigInfo(BaseModel):
    """Configuration file information."""
    name: str
    path: str
    description: Optional[str] = None
    strategy_type: Optional[str] = None
    symbols: List[str]
    timeframes: List[str]
    parameters: Dict[str, Any]
    last_modified: datetime


class ConfigListResponse(BaseResponse):
    """Configuration list response model."""
    configs: List[ConfigInfo]
    total: int


class ConfigValidationRequest(BaseModel):
    """Configuration validation request."""
    config: Dict[str, Any]


class ConfigValidationResponse(BaseResponse):
    """Configuration validation response model."""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []


# WebSocket Models

class WSMessage(BaseModel):
    """WebSocket message model."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSSubscription(BaseModel):
    """WebSocket subscription request."""
    channels: List[str]
    symbols: Optional[List[str]] = None
    strategy_ids: Optional[List[str]] = None


# Pagination Models

class PaginationParams(BaseModel):
    """Pagination parameters."""
    skip: int = Field(0, ge=0, description="Number of items to skip")
    limit: int = Field(100, ge=1, le=1000, description="Number of items to return")


# Statistics Models

class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    total_return: float
    total_return_pct: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: Optional[float] = None
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int


class RiskMetrics(BaseModel):
    """Risk metrics model."""
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    expected_shortfall: Optional[float] = None
    beta: Optional[float] = None
    volatility: float
    downside_deviation: Optional[float] = None


class TradingMetrics(BaseModel):
    """Trading metrics model."""
    total_equity: float
    available_balance: float
    margin_used: float
    margin_available: float
    total_positions: int
    long_positions: int
    short_positions: int
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float