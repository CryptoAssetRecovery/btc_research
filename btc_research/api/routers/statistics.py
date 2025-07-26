"""
Statistics and data endpoints.
Handles trade history, positions, and market data.
"""

import logging
from typing import Any, Dict, List, Optional
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
        
        # Get condition metrics summary for enhanced response
        condition_summary = await data_access.get_strategy_condition_summary(strategy_id)
        
        # Add condition metrics summary to the response data
        if condition_summary:
            # Extract key metrics for the summary
            condition_summary_data = condition_summary.get('condition_summary', {})
            performance_data = condition_summary.get('performance', {})
            
            # Calculate overall pass rate
            total_evaluations = performance_data.get('total_evaluations', 0)
            total_passed = sum([
                condition_summary_data.get('entry_long', {}).get('passed', 0),
                condition_summary_data.get('entry_short', {}).get('passed', 0),
                condition_summary_data.get('exit_long', {}).get('passed', 0),
                condition_summary_data.get('exit_short', {}).get('passed', 0)
            ])
            overall_pass_rate = (total_passed / total_evaluations) if total_evaluations > 0 else 0.0
            
            stats['condition_metrics_summary'] = {
                'total_evaluations': total_evaluations,
                'overall_pass_rate': overall_pass_rate,
                'entry_long_pass_rate': condition_summary_data.get('entry_long', {}).get('pass_rate', 0.0),
                'entry_short_pass_rate': condition_summary_data.get('entry_short', {}).get('pass_rate', 0.0),
                'exit_long_pass_rate': condition_summary_data.get('exit_long', {}).get('pass_rate', 0.0),
                'exit_short_pass_rate': condition_summary_data.get('exit_short', {}).get('pass_rate', 0.0),
                'top_performing_rules': condition_summary.get('top_performing_rules', []),
                'underperforming_rules': condition_summary.get('underperforming_rules', []),
                'recent_activity_count': len(condition_summary.get('recent_activity', []))
            }
        else:
            # Provide empty structure if no condition metrics available
            stats['condition_metrics_summary'] = {
                'total_evaluations': 0,
                'overall_pass_rate': 0.0,
                'entry_long_pass_rate': 0.0,
                'entry_short_pass_rate': 0.0,
                'exit_long_pass_rate': 0.0,
                'exit_short_pass_rate': 0.0,
                'top_performing_rules': [],
                'underperforming_rules': [],
                'recent_activity_count': 0
            }
        
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


@router.get("/strategies/{strategy_id}/conditions", response_model=BaseResponse)
async def get_strategy_conditions(
    request: Request,
    strategy_id: str,
    current_user: RequiredAuth,
    strategy_manager=Depends(get_strategy_manager),
    redis_client=Depends(get_redis_client)
):
    """
    Get detailed condition metrics and analysis for a strategy.
    
    Returns comprehensive condition evaluation metrics including individual rule
    performance, pass rates, recent activity, and detailed analysis of rule
    correlations and performance trends.
    """
    # Apply rate limiting
    
    try:
        # Initialize data access layer
        data_access = DataAccess(strategy_manager, redis_client)
        
        # Get detailed condition metrics
        condition_metrics = await data_access.get_strategy_condition_metrics(strategy_id)
        
        if not condition_metrics:
            # Return empty structure instead of 404 when no metrics are available
            condition_metrics = {
                'strategy_id': strategy_id,
                'rule_statistics': {},
                'recent_evaluations': [],
                'entry_long_evaluations': 0,
                'entry_long_passed': 0,
                'entry_short_evaluations': 0,
                'entry_short_passed': 0,
                'exit_long_evaluations': 0,
                'exit_long_passed': 0,
                'exit_short_evaluations': 0,
                'exit_short_passed': 0,
                'last_update': None
            }
        
        # Get condition summary
        condition_summary = await data_access.get_strategy_condition_summary(strategy_id)
        
        # Perform additional analysis
        analysis = _analyze_condition_metrics(condition_metrics)
        
        logger.info(f"Retrieved condition metrics for strategy {strategy_id}")
        
        return BaseResponse(
            message="Strategy condition metrics retrieved successfully",
            data={
                'strategy_id': strategy_id,
                'condition_metrics': condition_metrics,
                'summary': condition_summary or {},
                'analysis': analysis
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get condition metrics for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve condition metrics: {str(e)}"
        )


def _analyze_condition_metrics(condition_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze condition metrics to provide insights and trends.
    
    Args:
        condition_metrics: Full condition metrics dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        analysis = {
            'most_frequent_failures': [],
            'performance_trends': {},
            'rule_correlation_analysis': {}
        }
        
        # Get rule statistics
        rule_stats = condition_metrics.get('rule_statistics', {})
        
        # Find most frequent failures (rules with lowest pass rates)
        if rule_stats:
            sorted_by_failure = sorted(
                rule_stats.items(),
                key=lambda x: (x[1]['pass_rate'], -x[1]['total'])
            )
            
            # Get top 5 most frequent failures
            analysis['most_frequent_failures'] = [
                {
                    'rule': rule,
                    'pass_rate': stats['pass_rate'],
                    'total_evaluations': stats['total'],
                    'failure_rate': 1.0 - stats['pass_rate']
                }
                for rule, stats in sorted_by_failure[:5]
                if stats['total'] > 0  # Only include rules that have been evaluated
            ]
        
        # Analyze performance trends
        recent_evaluations = condition_metrics.get('recent_evaluations', [])
        if recent_evaluations:
            # Calculate trend over recent evaluations
            recent_count = len(recent_evaluations)
            recent_passed = sum(1 for eval_data in recent_evaluations if eval_data.get('result', False))
            recent_pass_rate = recent_passed / recent_count if recent_count > 0 else 0.0
            
            # Compare with overall rates
            condition_types = ['entry_long', 'entry_short', 'exit_long', 'exit_short']
            overall_evaluations = sum(condition_metrics.get(f"{ct}_evaluations", 0) for ct in condition_types)
            overall_passed = sum(condition_metrics.get(f"{ct}_passed", 0) for ct in condition_types)
            overall_pass_rate = overall_passed / overall_evaluations if overall_evaluations > 0 else 0.0
            
            analysis['performance_trends'] = {
                'recent_pass_rate': recent_pass_rate,
                'overall_pass_rate': overall_pass_rate,
                'trend_direction': 'improving' if recent_pass_rate > overall_pass_rate else (
                    'declining' if recent_pass_rate < overall_pass_rate else 'stable'
                ),
                'recent_evaluation_count': recent_count
            }
        
        # Basic rule correlation analysis (simplified)
        if rule_stats and len(rule_stats) > 1:
            # Group rules by performance tiers
            high_performers = [rule for rule, stats in rule_stats.items() if stats['pass_rate'] > 0.8]
            low_performers = [rule for rule, stats in rule_stats.items() if stats['pass_rate'] < 0.3]
            
            analysis['rule_correlation_analysis'] = {
                'high_performing_rules': high_performers,
                'low_performing_rules': low_performers,
                'performance_distribution': {
                    'high_performers_count': len(high_performers),
                    'medium_performers_count': len(rule_stats) - len(high_performers) - len(low_performers),
                    'low_performers_count': len(low_performers)
                }
            }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze condition metrics: {e}")
        return {
            'most_frequent_failures': [],
            'performance_trends': {},
            'rule_correlation_analysis': {},
            'analysis_error': str(e)
        }