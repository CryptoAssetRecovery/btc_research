"""
Strategy management endpoints.
Handles strategy lifecycle: start, stop, list, status monitoring.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from slowapi import Limiter

from ..models import (
    StrategyStartRequest, StrategyResponse, StrategyListResponse,
    StrategyStatsResponse, BaseResponse, ErrorResponse
)
from ..dependencies import (
    RequiredAuth, get_limiter, get_strategy_manager, get_paper_trader,
    get_stream_manager
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.post("/start", response_model=StrategyResponse)
async def start_strategy(
    request: Request,
    strategy_request: StrategyStartRequest,
    current_user: RequiredAuth,
    strategy_manager=Depends(get_strategy_manager)
):
    """
    Start a new strategy from configuration.
    
    This endpoint creates and starts a new strategy instance based on the
    provided configuration file and parameters.
    """
    # Apply rate limiting
    
    try:
        # Start the strategy using StrategyManager
        strategy_id = await strategy_manager.start_strategy(
            config_path=strategy_request.config_path,
            initial_balance=strategy_request.initial_balance,
            user_id=current_user['user_id']
        )
        
        # Get strategy information
        strategy_info = await strategy_manager.get_strategy_status(strategy_id)
        if not strategy_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve strategy information after creation"
            )
        
        # Format response
        formatted_info = {
            "id": strategy_info["id"],
            "name": strategy_info.get("strategy_name", "Unknown Strategy"),
            "config_path": strategy_info["config_path"],
            "status": strategy_info["status"],
            "symbol": strategy_info.get("symbol", "Unknown"),
            "timeframes": [],  # Will be populated from config if available
            "created_at": strategy_info["created_at"],
            "updated_at": strategy_info["updated_at"],
            "initial_balance": strategy_info["initial_balance"],
            "current_balance": strategy_info.get("trading", {}).get("total_equity", strategy_info["initial_balance"]),
            "total_trades": strategy_info.get("trading", {}).get("total_trades", 0),
            "active_positions": strategy_info.get("trading", {}).get("total_positions", 0)
        }
        
        logger.info(f"Started strategy {strategy_id[:8]} for user {current_user['user_id']}")
        
        return StrategyResponse(
            message="Strategy started successfully",
            strategy=formatted_info
        )
        
    except Exception as e:
        logger.error(f"Failed to start strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start strategy: {str(e)}"
        )


@router.post("/{strategy_id}/stop", response_model=BaseResponse)
async def stop_strategy(
    request: Request,
    strategy_id: str,
    current_user: RequiredAuth,
    strategy_manager=Depends(get_strategy_manager)
):
    """
    Stop a running strategy.
    
    This endpoint gracefully stops a running strategy and cleans up resources.
    """
    # Apply rate limiting
    
    try:
        # Verify strategy exists and belongs to user
        strategy_info = await strategy_manager.get_strategy_status(strategy_id)
        if not strategy_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy not found: {strategy_id}"
            )
        
        if strategy_info.get("user_id") != current_user['user_id']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to stop this strategy"
            )
        
        # Stop the strategy
        await strategy_manager.stop_strategy(strategy_id)
        
        logger.info(f"Stopped strategy {strategy_id[:8]} for user {current_user['user_id']}")
        
        return BaseResponse(
            message=f"Strategy {strategy_id} stopped successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop strategy: {str(e)}"
        )


@router.get("/active", response_model=StrategyListResponse)
async def list_active_strategies(
    request: Request,
    current_user: RequiredAuth,
    strategy_manager=Depends(get_strategy_manager)
):
    """
    List all active strategies.
    
    Returns a list of all currently running strategies for the authenticated user.
    """
    
    try:
        # Get active strategies for the user
        strategies_data = await strategy_manager.get_active_strategies(user_id=current_user['user_id'])
        
        # Format strategies for response
        formatted_strategies = []
        for strategy_info in strategies_data:
            formatted_info = {
                "id": strategy_info["id"],
                "name": strategy_info.get("strategy_name", "Unknown Strategy"),
                "config_path": strategy_info["config_path"],
                "status": strategy_info["status"],
                "symbol": strategy_info.get("symbol", "Unknown"),
                "timeframes": [],  # Will be populated from config if available
                "created_at": strategy_info["created_at"],
                "updated_at": strategy_info["updated_at"],
                "initial_balance": strategy_info["initial_balance"],
                "current_balance": strategy_info.get("trading", {}).get("total_equity", strategy_info["initial_balance"]),
                "total_trades": strategy_info.get("trading", {}).get("total_trades", 0),
                "active_positions": strategy_info.get("trading", {}).get("total_positions", 0)
            }
            formatted_strategies.append(formatted_info)
        
        logger.info(f"Listed {len(formatted_strategies)} active strategies for user {current_user['user_id']}")
        
        return StrategyListResponse(
            message="Active strategies retrieved successfully",
            strategies=formatted_strategies,
            total=len(formatted_strategies)
        )
        
    except Exception as e:
        logger.error(f"Failed to list strategies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list strategies: {str(e)}"
        )


@router.get("/{strategy_id}/status", response_model=StrategyResponse)
async def get_strategy_status(
    request: Request,
    strategy_id: str,
    current_user: RequiredAuth,
    strategy_manager=Depends(get_strategy_manager)
):
    """
    Get detailed status of a specific strategy.
    
    Returns health status, performance metrics, and current state of the strategy.
    """
    # Apply rate limiting
    
    try:
        # Get strategy status from manager
        strategy_info = await strategy_manager.get_strategy_status(strategy_id)
        if not strategy_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy not found: {strategy_id}"
            )
        
        # Verify user has access to this strategy
        if strategy_info.get("user_id") != current_user['user_id']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to view this strategy"
            )
        
        # Format response
        formatted_info = {
            "id": strategy_info["id"],
            "name": strategy_info.get("strategy_name", "Unknown Strategy"),
            "config_path": strategy_info["config_path"],
            "status": strategy_info["status"],
            "symbol": strategy_info.get("symbol", "Unknown"),
            "timeframes": [],  # Will be populated from config if available
            "created_at": strategy_info["created_at"],
            "updated_at": strategy_info["updated_at"],
            "initial_balance": strategy_info["initial_balance"],
            "current_balance": strategy_info.get("trading", {}).get("total_equity", strategy_info["initial_balance"]),
            "total_trades": strategy_info.get("trading", {}).get("total_trades", 0),
            "active_positions": strategy_info.get("trading", {}).get("total_positions", 0)
        }
        
        logger.info(f"Retrieved status for strategy {strategy_id[:8]}")
        
        return StrategyResponse(
            message="Strategy status retrieved successfully",
            strategy=formatted_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy status {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve strategy status: {str(e)}"
        )


@router.get("/{strategy_id}/stats", response_model=StrategyStatsResponse)
async def get_strategy_statistics(
    request: Request,
    strategy_id: str,
    current_user: RequiredAuth,
    strategy_manager=Depends(get_strategy_manager)
):
    """
    Get real-time performance metrics for a strategy.
    
    Returns detailed performance, trading, and risk statistics.
    """
    # Apply rate limiting
    
    try:
        # Get strategy statistics from manager
        strategy_stats = await strategy_manager.get_strategy_statistics(strategy_id)
        if not strategy_stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy not found: {strategy_id}"
            )
        
        # Verify user has access to this strategy
        strategy_info = await strategy_manager.get_strategy_status(strategy_id)
        if not strategy_info or strategy_info.get("user_id") != current_user['user_id']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to view this strategy"
            )
        
        # Extract and format statistics
        trading_stats = strategy_stats.get("trading", {})
        performance_stats = strategy_stats.get("performance", {})
        
        # Calculate additional metrics
        total_return = trading_stats.get("total_pnl", 0.0)
        initial_balance = strategy_info.get("initial_balance", 10000.0)
        total_return_pct = (total_return / initial_balance) * 100 if initial_balance > 0 else 0.0
        
        win_rate = 0.0
        if "performance" in strategy_stats and "signals_generated" in strategy_stats["performance"]:
            signals = strategy_stats["performance"]["signals_generated"]
            successful_orders = strategy_stats["performance"].get("successful_orders", 0)
            win_rate = (successful_orders / signals) if signals > 0 else 0.0
        
        # Format response
        formatted_stats = {
            "strategy_id": strategy_id,
            "performance": {
                "total_return": total_return,
                "total_return_pct": total_return_pct,
                "sharpe_ratio": performance_stats.get("sharpe_ratio", 0.0),
                "max_drawdown": performance_stats.get("max_drawdown", 0.0),
                "max_drawdown_pct": performance_stats.get("max_drawdown_pct", 0.0),
                "win_rate": win_rate,
                "signals_generated": strategy_stats.get("performance", {}).get("signals_generated", 0),
                "trades_executed": trading_stats.get("total_trades", 0)
            },
            "trading": {
                "total_equity": trading_stats.get("total_equity", initial_balance),
                "available_balance": trading_stats.get("available_balance", initial_balance),
                "total_pnl": trading_stats.get("total_pnl", 0.0),
                "realized_pnl": trading_stats.get("realized_pnl", 0.0),
                "unrealized_pnl": trading_stats.get("unrealized_pnl", 0.0),
                "total_positions": trading_stats.get("total_positions", 0),
                "total_trades": trading_stats.get("total_trades", 0)
            },
            "risk": {
                "position_size_pct": 10.0,  # Default, would be calculated from config
                "max_position_size": trading_stats.get("max_position_size", 0.0),
                "risk_per_trade": 2.0,  # Default, would be from config
                "current_drawdown": performance_stats.get("current_drawdown", 0.0),
                "volatility": performance_stats.get("volatility", 0.0)
            },
            "updated_at": strategy_stats.get("last_update_time", strategy_info.get("updated_at"))
        }
        
        logger.info(f"Retrieved statistics for strategy {strategy_id[:8]}")
        
        return StrategyStatsResponse(
            message="Strategy statistics retrieved successfully",
            stats=formatted_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy statistics {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve strategy statistics: {str(e)}"
        )


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy_details(
    request: Request,
    strategy_id: str,
    current_user: RequiredAuth,
    strategy_manager=Depends(get_strategy_manager)
):
    """
    Get detailed information about a specific strategy.
    
    Returns comprehensive strategy details including configuration,
    current status, and recent activity.
    """
    # Apply rate limiting
    
    try:
        # Get strategy status from manager
        strategy_info = await strategy_manager.get_strategy_status(strategy_id)
        if not strategy_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy not found: {strategy_id}"
            )
        
        # Verify user has access to this strategy
        if strategy_info.get("user_id") != current_user['user_id']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to view this strategy"
            )
        
        # Format response with additional details
        formatted_info = {
            "id": strategy_info["id"],
            "name": strategy_info.get("strategy_name", "Unknown Strategy"),
            "config_path": strategy_info["config_path"],
            "status": strategy_info["status"],
            "symbol": strategy_info.get("symbol", "Unknown"),
            "timeframes": [],  # Will be populated from config if available
            "created_at": strategy_info["created_at"],
            "updated_at": strategy_info["updated_at"],
            "initial_balance": strategy_info["initial_balance"],
            "current_balance": strategy_info.get("trading", {}).get("total_equity", strategy_info["initial_balance"]),
            "total_trades": strategy_info.get("trading", {}).get("total_trades", 0),
            "active_positions": strategy_info.get("trading", {}).get("total_positions", 0)
        }
        
        logger.info(f"Retrieved details for strategy {strategy_id[:8]}")
        
        return StrategyResponse(
            message="Strategy details retrieved successfully",
            strategy=formatted_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy details {strategy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve strategy details: {str(e)}"
        )