"""
Statistics and data endpoints.
Handles trade history, positions, and market data.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from slowapi import Limiter

from ..models import (
    TradeListResponse, PositionListResponse, OrderListResponse,
    MarketDataResponse, BaseResponse, PerformanceMetrics
)
from ..dependencies import (
    RequiredAuth, get_limiter, get_strategy_manager, get_stream_manager,
    validate_symbol, validate_timeframe, validate_pagination, get_redis_client
)
from ..data_access import DataAccess
from ..performance import PerformanceCalculator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/statistics", tags=["statistics"])


@router.get("/strategies/{strategy_id}/stats", response_model=BaseResponse)
async def get_strategy_statistics(
    request: Request,
    strategy_id: str,
    current_user: RequiredAuth,
    strategy_manager=Depends(get_strategy_manager),
    redis_client=Depends(get_redis_client)
):
    """
    Get real-time performance metrics for a strategy.
    
    Returns comprehensive performance statistics including returns, 
    risk metrics, drawdown analysis, and trading statistics.
    """
    # Apply rate limiting
    
    try:
        # Initialize data access layer
        data_access = DataAccess(strategy_manager, redis_client)
        
        # Get strategy statistics
        stats = await data_access.get_strategy_statistics(strategy_id)
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy {strategy_id} not found"
            )
        
        logger.info(f"Retrieved statistics for strategy {strategy_id}")
        
        return BaseResponse(
            message="Strategy statistics retrieved successfully",
            data=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get statistics for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve strategy statistics: {str(e)}"
        )


@router.get("/strategies/{strategy_id}/trades", response_model=TradeListResponse)
async def get_strategy_trades(
    request: Request,
    strategy_id: str,
    current_user: RequiredAuth,
    skip: int = Query(0, ge=0, description="Number of trades to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of trades to return"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy_manager=Depends(get_strategy_manager),
    redis_client=Depends(get_redis_client)
):
    """
    Get trade history for a strategy with pagination.
    
    Returns a paginated list of executed trades for the specified strategy.
    """
    # Apply rate limiting
    
    try:
        # Validate pagination parameters
        skip, limit = validate_pagination(skip, limit)
        
        # Validate symbol if provided
        if symbol:
            symbol = validate_symbol(symbol)
        
        # Initialize data access layer
        data_access = DataAccess(strategy_manager, redis_client)
        
        # Get trade history
        trades_result = await data_access.get_strategy_trades(strategy_id, skip, limit, symbol)
        
        logger.info(f"Retrieved {len(trades_result.items)} trades for strategy {strategy_id}")
        
        return TradeListResponse(
            message="Trade history retrieved successfully",
            trades=trades_result.items,
            total=trades_result.total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Failed to get trades for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve trade history: {str(e)}"
        )


@router.get("/strategies/{strategy_id}/positions", response_model=PositionListResponse)
async def get_strategy_positions(
    request: Request,
    strategy_id: str,
    current_user: RequiredAuth,
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy_manager=Depends(get_strategy_manager),
    redis_client=Depends(get_redis_client)
):
    """
    Get current positions for a strategy.
    
    Returns all active positions for the specified strategy.
    """
    # Apply rate limiting
    
    try:
        # Validate symbol if provided
        if symbol:
            symbol = validate_symbol(symbol)
        
        # Initialize data access layer
        data_access = DataAccess(strategy_manager, redis_client)
        
        # Get current positions
        positions = await data_access.get_strategy_positions(strategy_id, symbol)
        
        logger.info(f"Retrieved {len(positions)} positions for strategy {strategy_id}")
        
        return PositionListResponse(
            message="Positions retrieved successfully",
            positions=positions,
            total=len(positions)
        )
        
    except Exception as e:
        logger.error(f"Failed to get positions for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve positions: {str(e)}"
        )


@router.get("/strategies/{strategy_id}/orders", response_model=OrderListResponse)
async def get_strategy_orders(
    request: Request,
    strategy_id: str,
    current_user: RequiredAuth,
    skip: int = Query(0, ge=0, description="Number of orders to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of orders to return"),
    status_filter: Optional[str] = Query(None, description="Filter by order status"),
    strategy_manager=Depends(get_strategy_manager),
    redis_client=Depends(get_redis_client)
):
    """
    Get order history for a strategy with pagination.
    
    Returns a paginated list of orders (filled, pending, cancelled) for the strategy.
    """
    # Apply rate limiting
    
    try:
        # Validate pagination parameters
        skip, limit = validate_pagination(skip, limit)
        
        # Initialize data access layer
        data_access = DataAccess(strategy_manager, redis_client)
        
        # Get order history
        orders_result = await data_access.get_strategy_orders(strategy_id, skip, limit, status_filter)
        
        logger.info(f"Retrieved {len(orders_result.items)} orders for strategy {strategy_id}")
        
        return OrderListResponse(
            message="Order history retrieved successfully",
            orders=orders_result.items,
            total=orders_result.total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Failed to get orders for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve order history: {str(e)}"
        )


@router.get("/market/data/{symbol}", response_model=MarketDataResponse)
async def get_market_data(
    request: Request,
    symbol: str,
    current_user: RequiredAuth,
    timeframe: str = Query("1m", description="Timeframe for candle data"),
    limit: int = Query(100, ge=1, le=1000, description="Number of candles to return"),
    strategy_manager=Depends(get_strategy_manager),
    redis_client=Depends(get_redis_client)
):
    """
    Get current market data for a symbol.
    
    Returns current price, 24h statistics, and recent candle data.
    """
    # Apply rate limiting
    
    try:
        # Validate parameters
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(timeframe)
        
        # Initialize data access layer
        data_access = DataAccess(strategy_manager, redis_client)
        
        # Get market data
        market_data = await data_access.get_market_data(symbol, timeframe, limit)
        
        if not market_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Market data not available for {symbol}"
            )
        
        logger.info(f"Retrieved market data for {symbol} ({timeframe})")
        
        return MarketDataResponse(
            message="Market data retrieved successfully",
            data=market_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get market data for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve market data: {str(e)}"
        )


@router.get("/strategies/{strategy_id}/performance", response_model=BaseResponse)
async def get_strategy_performance(
    request: Request,
    strategy_id: str,
    current_user: RequiredAuth,
    period: str = Query("1d", description="Period for chart data (1h, 1d, 1w, 1m)"),
    strategy_manager=Depends(get_strategy_manager),
    redis_client=Depends(get_redis_client)
):
    """
    Get historical performance chart data for a strategy.
    
    Returns time-series performance data suitable for charting including
    P&L over time, balance history, and key performance metrics.
    """
    # Apply rate limiting
    
    try:
        # Initialize data access layer
        data_access = DataAccess(strategy_manager, redis_client)
        
        # Get performance data
        performance_data = await data_access.get_strategy_performance(strategy_id, period)
        
        if not performance_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Performance data not available for strategy {strategy_id}"
            )
        
        logger.info(f"Retrieved performance data for strategy {strategy_id} (period: {period})")
        
        return BaseResponse(
            message="Performance data retrieved successfully",
            data=performance_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance data for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance data: {str(e)}"
        )