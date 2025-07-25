"""
Data access layer for retrieving strategy statistics and trading data.

This module provides a centralized interface for accessing trading data from
both active (running) strategies via StrategyManager and historical data
from persistent storage. It handles data retrieval, pagination, and provides
a consistent interface for the API endpoints.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import pandas as pd
import redis.asyncio as redis

from ..live.paper_trader import Order, Position, Trade, OrderSide, OrderStatus, OrderType
from .strategy_manager import StrategyManager
from .cache import CacheManager, cache_result

__all__ = ["DataAccess", "DataAccessError", "PaginationResult"]

logger = logging.getLogger(__name__)


class DataAccessError(Exception):
    """Base exception for data access related errors."""
    pass


class PaginationResult:
    """
    Result container for paginated data queries.
    
    Provides consistent structure for paginated responses with
    total count information for client-side pagination.
    """
    
    def __init__(self, items: List[Any], total: int, skip: int, limit: int):
        """
        Initialize pagination result.
        
        Args:
            items: List of data items for this page
            total: Total number of items available
            skip: Number of items skipped
            limit: Maximum number of items per page
        """
        self.items = items
        self.total = total
        self.skip = skip
        self.limit = limit
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'items': self.items,
            'total': self.total,
            'skip': self.skip,
            'limit': self.limit,
            'has_more': (self.skip + len(self.items)) < self.total
        }


class DataAccess:
    """
    Data access layer for trading statistics and data retrieval.
    
    This class provides a unified interface for accessing trading data from
    multiple sources:
    - Active strategies via StrategyManager and StrategyRunner
    - PaperTrader instances for current positions and trade history
    - StreamManager for real-time market data
    - Redis cache for performance optimization
    - Historical data from persistent storage
    
    Key Features:
    - Real-time data for active strategies
    - Historical data persistence for stopped strategies
    - Efficient pagination for large datasets
    - Performance metric calculations
    - Redis caching for expensive operations
    - Thread-safe access to shared resources
    """
    
    def __init__(
        self,
        strategy_manager: StrategyManager,
        redis_client: redis.Redis,
        data_dir: Path = None
    ):
        """
        Initialize data access layer.
        
        Args:
            strategy_manager: StrategyManager instance for active strategies
            redis_client: Redis client for caching
            data_dir: Directory for historical data files
        """
        self.strategy_manager = strategy_manager
        self.redis = redis_client
        self.data_dir = data_dir or Path("data/strategies")
        
        # Initialize cache manager
        self.cache_manager = CacheManager(redis_client)
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DataAccess layer initialized")
    
    @cache_result('strategy_stats', ttl=60)
    async def get_strategy_statistics(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive strategy statistics.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Dictionary with strategy statistics or None if not found
        """
        try:
            # Get from strategy manager (active strategies)
            stats = await self.strategy_manager.get_strategy_statistics(strategy_id)
            
            if stats:
                return stats
            
            # Try to get historical data
            return await self._get_historical_statistics(strategy_id)
            
        except Exception as e:
            logger.error(f"Failed to get strategy statistics for {strategy_id}: {e}")
            return None
    
    async def get_strategy_trades(
        self,
        strategy_id: str,
        skip: int = 0,
        limit: int = 100,
        symbol: Optional[str] = None
    ) -> PaginationResult:
        """
        Get paginated trade history for a strategy.
        
        Args:
            strategy_id: Strategy ID
            skip: Number of trades to skip
            limit: Maximum number of trades to return
            symbol: Optional symbol filter
            
        Returns:
            PaginationResult with trade data
        """
        try:
            # Try to get from active strategy first
            async with self.strategy_manager.get_strategy_runner(strategy_id) as runner:
                paper_trader = runner.paper_trader
                
                # Get all trades for the symbol filter
                all_trades = paper_trader.get_trade_history(symbol=symbol)
                
                # Convert to API format
                trade_data = []
                for trade in all_trades:
                    trade_dict = trade.to_dict()
                    trade_dict['strategy_id'] = strategy_id
                    
                    # Calculate realized P&L if possible
                    # This is a simplified calculation - real implementation would track positions
                    trade_dict['realized_pnl'] = None  # TODO: Implement proper P&L calculation
                    
                    trade_data.append(trade_dict)
                
                # Apply pagination
                total = len(trade_data)
                paginated_trades = trade_data[skip:skip + limit]
                
                return PaginationResult(paginated_trades, total, skip, limit)
                
        except Exception as e:
            logger.warning(f"Could not get active trades for {strategy_id}: {e}")
            
            # Fall back to historical data
            return await self._get_historical_trades(strategy_id, skip, limit, symbol)
    
    async def get_strategy_positions(
        self,
        strategy_id: str,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get current positions for a strategy.
        
        Args:
            strategy_id: Strategy ID
            symbol: Optional symbol filter
            
        Returns:
            List of position dictionaries
        """
        try:
            # Try to get from active strategy first
            async with self.strategy_manager.get_strategy_runner(strategy_id) as runner:
                paper_trader = runner.paper_trader
                
                # Get current positions
                positions = paper_trader.get_positions()
                
                # Filter by symbol if specified
                if symbol:
                    positions = {s: p for s, p in positions.items() if s == symbol}
                
                # Convert to API format
                position_data = []
                for position in positions.values():
                    pos_dict = position.to_dict()
                    
                    # Determine side (long/short)
                    pos_dict['side'] = 'buy' if position.size > 0 else 'sell'
                    pos_dict['size'] = str(abs(position.size))
                    pos_dict['average_price'] = str(position.average_price)
                    pos_dict['unrealized_pnl'] = str(position.unrealized_pnl)
                    
                    # Calculate percentage P&L
                    if position.average_price > 0:
                        current_price = paper_trader.get_current_price(position.symbol)
                        if current_price:
                            pnl_pct = ((current_price - position.average_price) / position.average_price) * 100
                            if position.size < 0:  # Short position
                                pnl_pct = -pnl_pct
                            pos_dict['unrealized_pnl_pct'] = pnl_pct
                        else:
                            pos_dict['unrealized_pnl_pct'] = 0.0
                    else:
                        pos_dict['unrealized_pnl_pct'] = 0.0
                    
                    # Add timestamps
                    pos_dict['opened_at'] = position.created_at.isoformat()
                    pos_dict['updated_at'] = position.updated_at.isoformat()
                    
                    # Get current market price
                    pos_dict['market_price'] = str(paper_trader.get_current_price(position.symbol) or 0.0)
                    
                    position_data.append(pos_dict)
                
                return position_data
                
        except Exception as e:
            logger.warning(f"Could not get active positions for {strategy_id}: {e}")
            
            # For stopped strategies, positions would be empty or historical
            return []
    
    async def get_strategy_orders(
        self,
        strategy_id: str,
        skip: int = 0,
        limit: int = 100,
        status_filter: Optional[str] = None
    ) -> PaginationResult:
        """
        Get paginated order history for a strategy.
        
        Args:
            strategy_id: Strategy ID
            skip: Number of orders to skip
            limit: Maximum number of orders to return
            status_filter: Optional status filter
            
        Returns:
            PaginationResult with order data
        """
        try:
            # Try to get from active strategy first
            async with self.strategy_manager.get_strategy_runner(strategy_id) as runner:
                paper_trader = runner.paper_trader
                
                # Get all orders
                all_orders = list(paper_trader.get_orders().values())
                
                # Filter by status if specified
                if status_filter:
                    all_orders = [o for o in all_orders if o.status.value == status_filter]
                
                # Sort by creation time (newest first)
                all_orders.sort(key=lambda o: o.created_at, reverse=True)
                
                # Convert to API format
                order_data = []
                for order in all_orders:
                    order_dict = order.to_dict()
                    order_dict['strategy_id'] = strategy_id
                    order_dict['type'] = order_dict.pop('order_type')  # Rename for API consistency
                    
                    order_data.append(order_dict)
                
                # Apply pagination
                total = len(order_data)
                paginated_orders = order_data[skip:skip + limit]
                
                return PaginationResult(paginated_orders, total, skip, limit)
                
        except Exception as e:
            logger.warning(f"Could not get active orders for {strategy_id}: {e}")
            
            # Fall back to historical data
            return await self._get_historical_orders(strategy_id, skip, limit, status_filter)
    
    @cache_result('market_data', ttl=30)  # Shorter cache for market data
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Get current market data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for candle data
            limit: Number of candles to return
            
        Returns:
            Market data dictionary or None if not available
        """
        try:
            # Get StreamManager from the strategy manager
            stream_manager = self.strategy_manager.stream_manager
            
            # Get current price
            current_price = stream_manager.get_latest_price(symbol)
            if current_price is None:
                logger.warning(f"No current price available for {symbol}")
                return None
            
            # Get candle data
            df = stream_manager.get_data(symbol, timeframe, limit=limit)
            
            # Convert to API format
            candles = []
            for _, row in df.iterrows():
                candle = {
                    'timestamp': row.name.isoformat(),
                    'open': str(row['open']),
                    'high': str(row['high']),
                    'low': str(row['low']),
                    'close': str(row['close']),
                    'volume': str(row['volume'])
                }
                candles.append(candle)
            
            # Calculate 24h change (simplified - using first and last candle)
            change_24h = 0.0
            volume_24h = "0.0"
            
            if len(candles) >= 2:
                first_price = float(candles[0]['open'])
                last_price = float(candles[-1]['close'])
                if first_price > 0:
                    change_24h = ((last_price - first_price) / first_price) * 100
                
                # Sum volume (simplified)
                volume_24h = str(sum(float(c['volume']) for c in candles))
            
            market_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': str(current_price),
                'change_24h': change_24h,
                'volume_24h': volume_24h,
                'last_update': datetime.now(UTC).isoformat(),
                'candles': candles
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    async def get_strategy_performance(
        self,
        strategy_id: str,
        period: str = "1d"
    ) -> Optional[Dict[str, Any]]:
        """
        Get historical performance chart data for a strategy.
        
        Args:
            strategy_id: Strategy ID
            period: Period for chart data (1h, 1d, 1w, 1m)
            
        Returns:
            Performance chart data or None if not available
        """
        try:
            # Get strategy statistics
            stats = await self.get_strategy_statistics(strategy_id)
            if not stats:
                return None
            
            # Get trade history for P&L calculation
            trades_result = await self.get_strategy_trades(strategy_id, limit=1000)
            trades = trades_result.items
            
            # Calculate P&L over time
            pnl_history = []
            cumulative_pnl = 0.0
            
            # Group trades by time period
            if trades:
                for trade in sorted(trades, key=lambda t: t['timestamp']):
                    # Simplified P&L calculation
                    # In real implementation, this would track actual position P&L
                    trade_pnl = -float(trade.get('commission', 0))  # At minimum, lose commission
                    cumulative_pnl += trade_pnl
                    
                    pnl_history.append({
                        'timestamp': trade['timestamp'],
                        'pnl': cumulative_pnl,
                        'balance': stats.get('total_balance', 10000) + cumulative_pnl
                    })
            
            # Add current balance point
            current_balance = stats.get('total_balance', 10000)
            current_unrealized = stats.get('total_unrealized_pnl', 0)
            
            pnl_history.append({
                'timestamp': datetime.now(UTC).isoformat(),
                'pnl': cumulative_pnl + current_unrealized,
                'balance': current_balance + current_unrealized
            })
            
            performance_data = {
                'strategy_id': strategy_id,
                'period': period,
                'pnl_history': pnl_history,
                'key_metrics': {
                    'total_return': current_balance + current_unrealized - stats.get('initial_balance', 10000),
                    'total_return_pct': ((current_balance + current_unrealized) / stats.get('initial_balance', 10000) - 1) * 100,
                    'max_drawdown': 0.0,  # TODO: Calculate actual drawdown
                    'sharpe_ratio': None,  # TODO: Calculate Sharpe ratio
                    'total_trades': len(trades),
                    'win_rate': 0.0  # TODO: Calculate win rate
                }
            }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get performance data for {strategy_id}: {e}")
            return None
    
    async def _get_historical_statistics(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get historical statistics from persistent storage."""
        try:
            stats_file = self.data_dir / f"{strategy_id}_stats.json"
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get historical statistics for {strategy_id}: {e}")
            return None
    
    async def _get_historical_trades(
        self,
        strategy_id: str,
        skip: int,
        limit: int,
        symbol: Optional[str] = None
    ) -> PaginationResult:
        """Get historical trades from persistent storage."""
        try:
            trades_file = self.data_dir / f"{strategy_id}_trades.jsonl"
            
            if not trades_file.exists():
                return PaginationResult([], 0, skip, limit)
            
            # Read trades from JSONL file
            trades = []
            with open(trades_file, 'r') as f:
                for line in f:
                    try:
                        trade = json.loads(line.strip())
                        if symbol is None or trade.get('symbol') == symbol:
                            trades.append(trade)
                    except json.JSONDecodeError:
                        continue
            
            # Sort by timestamp (newest first)
            trades.sort(key=lambda t: t.get('timestamp', ''), reverse=True)
            
            # Apply pagination
            total = len(trades)
            paginated_trades = trades[skip:skip + limit]
            
            return PaginationResult(paginated_trades, total, skip, limit)
            
        except Exception as e:
            logger.error(f"Failed to get historical trades for {strategy_id}: {e}")
            return PaginationResult([], 0, skip, limit)
    
    async def _get_historical_orders(
        self,
        strategy_id: str,
        skip: int,
        limit: int,
        status_filter: Optional[str] = None
    ) -> PaginationResult:
        """Get historical orders from persistent storage."""
        try:
            orders_file = self.data_dir / f"{strategy_id}_orders.jsonl"
            
            if not orders_file.exists():
                return PaginationResult([], 0, skip, limit)
            
            # Read orders from JSONL file
            orders = []
            with open(orders_file, 'r') as f:
                for line in f:
                    try:
                        order = json.loads(line.strip())
                        if status_filter is None or order.get('status') == status_filter:
                            orders.append(order)
                    except json.JSONDecodeError:
                        continue
            
            # Sort by creation time (newest first)
            orders.sort(key=lambda o: o.get('created_at', ''), reverse=True)
            
            # Apply pagination
            total = len(orders)
            paginated_orders = orders[skip:skip + limit]
            
            return PaginationResult(paginated_orders, total, skip, limit)
            
        except Exception as e:
            logger.error(f"Failed to get historical orders for {strategy_id}: {e}")
            return PaginationResult([], 0, skip, limit)
    
    async def save_strategy_data(self, strategy_id: str) -> None:
        """
        Save current strategy data to persistent storage.
        
        This method should be called periodically and when strategies are stopped
        to ensure data persistence across system restarts.
        
        Args:
            strategy_id: Strategy ID to save data for
        """
        try:
            # Get current data from active strategy
            async with self.strategy_manager.get_strategy_runner(strategy_id) as runner:
                paper_trader = runner.paper_trader
                
                # Save statistics
                stats = paper_trader.get_performance_stats()
                stats_file = self.data_dir / f"{strategy_id}_stats.json"
                
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
                
                # Save trades (append to JSONL)
                trades_file = self.data_dir / f"{strategy_id}_trades.jsonl"
                recent_trades = paper_trader.get_trade_history(limit=100)
                
                with open(trades_file, 'a') as f:
                    for trade in recent_trades:
                        trade_dict = trade.to_dict()
                        trade_dict['strategy_id'] = strategy_id
                        f.write(json.dumps(trade_dict, default=str) + '\n')
                
                # Save orders (append to JSONL)
                orders_file = self.data_dir / f"{strategy_id}_orders.jsonl"
                all_orders = list(paper_trader.get_orders().values())
                
                with open(orders_file, 'a') as f:
                    for order in all_orders:
                        order_dict = order.to_dict()
                        order_dict['strategy_id'] = strategy_id
                        f.write(json.dumps(order_dict, default=str) + '\n')
                
                logger.info(f"Saved strategy data for {strategy_id}")
                
        except Exception as e:
            logger.error(f"Failed to save strategy data for {strategy_id}: {e}")