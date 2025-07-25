"""
StrategyManager for coordinating multiple trading strategies in the paper trading API.

This module implements the StrategyManager class that manages the lifecycle of
multiple StrategyRunner instances, provides Redis-based coordination, and serves
as the main interface for the FastAPI endpoints.

The StrategyManager handles:
- Multiple concurrent strategy execution
- Strategy state persistence via Redis
- Strategy lifecycle management (start/stop/monitor)
- Resource cleanup and error isolation
- Real-time status monitoring and statistics
"""

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from contextlib import asynccontextmanager

import redis.asyncio as redis
from btc_research.live.strategy_runner import StrategyRunner, StrategyRunnerError
from btc_research.live.paper_trader import PaperTrader
from btc_research.live.stream_manager import StreamManager

__all__ = ["StrategyManager", "StrategyManagerError", "StrategyRegistry"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyManagerError(Exception):
    """Base exception for StrategyManager related errors."""
    pass


class StrategyRegistry:
    """
    Redis-based registry for strategy coordination and state persistence.
    
    Maintains a persistent registry of all active strategies across API server
    restarts and provides coordination between multiple StrategyManager instances.
    """
    
    def __init__(self, redis_client: redis.Redis, key_prefix: str = "paper_trading:strategies"):
        """
        Initialize strategy registry.
        
        Args:
            redis_client: Redis client for data persistence
            key_prefix: Prefix for Redis keys
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.strategies_key = f"{key_prefix}:active"
        self.stats_key_prefix = f"{key_prefix}:stats"
        self.config_key_prefix = f"{key_prefix}:config"
    
    async def register_strategy(
        self, 
        strategy_id: str, 
        config_path: str, 
        initial_balance: float,
        user_id: str
    ) -> None:
        """Register a new strategy in the registry."""
        try:
            strategy_info = {
                "id": strategy_id,
                "config_path": config_path,
                "initial_balance": initial_balance,
                "user_id": user_id,
                "status": "starting",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "error_count": 0,
                "last_error": None
            }
            
            # Store strategy info
            await self.redis.hset(self.strategies_key, strategy_id, json.dumps(strategy_info))
            
            # Store config path separately for easy access
            await self.redis.hset(self.config_key_prefix, strategy_id, config_path)
            
            # Set expiration for cleanup (24 hours)
            await self.redis.expire(f"{self.strategies_key}:{strategy_id}", 86400)
            
            logger.info(f"Registered strategy {strategy_id[:8]} in Redis registry")
            
        except Exception as e:
            logger.error(f"Failed to register strategy {strategy_id}: {e}")
            raise StrategyManagerError(f"Failed to register strategy: {e}") from e
    
    async def unregister_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from the registry."""
        try:
            await self.redis.hdel(self.strategies_key, strategy_id)
            await self.redis.hdel(self.config_key_prefix, strategy_id)
            await self.redis.delete(f"{self.stats_key_prefix}:{strategy_id}")
            
            logger.info(f"Unregistered strategy {strategy_id[:8]} from Redis registry")
            
        except Exception as e:
            logger.error(f"Failed to unregister strategy {strategy_id}: {e}")
    
    async def update_strategy_status(self, strategy_id: str, status: str, error: str = None) -> None:
        """Update strategy status in the registry."""
        try:
            # Get current strategy info
            strategy_data = await self.redis.hget(self.strategies_key, strategy_id)
            if not strategy_data:
                logger.warning(f"Strategy {strategy_id} not found in registry")
                return
            
            strategy_info = json.loads(strategy_data)
            strategy_info["status"] = status
            strategy_info["updated_at"] = datetime.now(UTC).isoformat()
            
            if error:
                strategy_info["error_count"] = strategy_info.get("error_count", 0) + 1
                strategy_info["last_error"] = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "error": str(error)
                }
            
            # Update in Redis
            await self.redis.hset(self.strategies_key, strategy_id, json.dumps(strategy_info))
            
        except Exception as e:
            logger.error(f"Failed to update strategy status {strategy_id}: {e}")
    
    async def update_strategy_stats(self, strategy_id: str, stats: Dict[str, Any]) -> None:
        """Update strategy statistics in the registry."""
        try:
            stats_data = {
                "strategy_id": strategy_id,
                "updated_at": datetime.now(UTC).isoformat(),
                "stats": stats
            }
            
            await self.redis.setex(
                f"{self.stats_key_prefix}:{strategy_id}",
                3600,  # 1 hour expiration
                json.dumps(stats_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to update strategy stats {strategy_id}: {e}")
    
    async def get_strategy_info(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy information from the registry."""
        try:
            strategy_data = await self.redis.hget(self.strategies_key, strategy_id)
            if strategy_data:
                return json.loads(strategy_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get strategy info {strategy_id}: {e}")
            return None
    
    async def get_strategy_stats(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy statistics from the registry."""
        try:
            stats_data = await self.redis.get(f"{self.stats_key_prefix}:{strategy_id}")
            if stats_data:
                return json.loads(stats_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get strategy stats {strategy_id}: {e}")
            return None
    
    async def list_active_strategies(self, user_id: str = None) -> List[Dict[str, Any]]:
        """List all active strategies, optionally filtered by user."""
        try:
            all_strategies = await self.redis.hgetall(self.strategies_key)
            active_strategies = []
            
            for strategy_id, strategy_data in all_strategies.items():
                if isinstance(strategy_id, bytes):
                    strategy_id = strategy_id.decode('utf-8')
                if isinstance(strategy_data, bytes):
                    strategy_data = strategy_data.decode('utf-8')
                
                strategy_info = json.loads(strategy_data)
                
                # Filter by user if specified
                if user_id and strategy_info.get("user_id") != user_id:
                    continue
                
                # Only include active strategies
                if strategy_info.get("status") in ["starting", "running"]:
                    active_strategies.append(strategy_info)
            
            return active_strategies
            
        except Exception as e:
            logger.error(f"Failed to list active strategies: {e}")
            return []
    
    async def cleanup_expired_strategies(self) -> None:
        """Clean up expired or orphaned strategy records."""
        try:
            all_strategies = await self.redis.hgetall(self.strategies_key)
            current_time = datetime.now(UTC)
            
            for strategy_id, strategy_data in all_strategies.items():
                if isinstance(strategy_id, bytes):
                    strategy_id = strategy_id.decode('utf-8')
                if isinstance(strategy_data, bytes):
                    strategy_data = strategy_data.decode('utf-8')
                
                try:
                    strategy_info = json.loads(strategy_data)
                    created_at = datetime.fromisoformat(strategy_info["created_at"].replace('Z', '+00:00'))
                    
                    # Remove strategies older than 24 hours
                    if current_time - created_at > timedelta(hours=24):
                        await self.unregister_strategy(strategy_id)
                        logger.info(f"Cleaned up expired strategy {strategy_id[:8]}")
                        
                except Exception as e:
                    logger.error(f"Error cleaning up strategy {strategy_id}: {e}")
                    await self.unregister_strategy(strategy_id)
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired strategies: {e}")


class StrategyManager:
    """
    Manages multiple trading strategies for the paper trading API.
    
    The StrategyManager is the central coordination point for all strategy execution.
    It provides:
    
    - Multiple concurrent strategy management
    - Strategy lifecycle control (start/stop/monitor)
    - Redis-based state persistence and coordination
    - Resource cleanup and error isolation
    - Real-time monitoring and statistics
    - Integration with FastAPI endpoints
    
    Key Features:
    1. Multi-Strategy Coordination: Run multiple strategies concurrently
    2. State Persistence: Strategy state survives API server restarts
    3. Error Isolation: One strategy failure doesn't affect others
    4. Resource Management: Proper cleanup when strategies are stopped
    5. Real-time Monitoring: Live statistics and health monitoring
    6. Thread Safety: Safe for concurrent access from multiple endpoints
    
    Example:
        >>> manager = StrategyManager(paper_trader, stream_manager, redis_client)
        >>> await manager.initialize()
        >>> strategy_id = await manager.start_strategy("config/volume-profile-breakout.yaml", 10000, "user123")
        >>> status = await manager.get_strategy_status(strategy_id)
        >>> await manager.stop_strategy(strategy_id)
    """
    
    def __init__(
        self,
        paper_trader: PaperTrader,
        stream_manager: StreamManager,
        redis_client: redis.Redis,
        max_concurrent_strategies: int = 10,
        stats_update_interval: int = 30  # seconds
    ):
        """
        Initialize the StrategyManager.
        
        Args:
            paper_trader: PaperTrader instance for order execution
            stream_manager: StreamManager instance for real-time data
            redis_client: Redis client for state persistence
            max_concurrent_strategies: Maximum number of concurrent strategies
            stats_update_interval: Interval for updating strategy statistics
        """
        self.paper_trader = paper_trader
        self.stream_manager = stream_manager
        self.redis = redis_client
        self.max_concurrent_strategies = max_concurrent_strategies
        self.stats_update_interval = stats_update_interval
        
        # Strategy management
        self.active_strategies: Dict[str, StrategyRunner] = {}
        self.strategy_tasks: Dict[str, asyncio.Task] = {}
        
        # Registry for coordination
        self.registry = StrategyRegistry(redis_client)
        
        # Cleanup and monitoring tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("StrategyManager initialized")
    
    async def initialize(self) -> None:
        """Initialize the StrategyManager and start background tasks."""
        if self._running:
            logger.warning("StrategyManager is already running")
            return
        
        try:
            logger.info("Starting StrategyManager...")
            
            self._running = True
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._stats_task = asyncio.create_task(self._stats_update_loop())
            
            # Restore strategies from Redis if any
            await self._restore_strategies()
            
            logger.info("StrategyManager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize StrategyManager: {e}")
            await self.shutdown()
            raise StrategyManagerError(f"Failed to initialize: {e}") from e
    
    async def shutdown(self) -> None:
        """Shutdown the StrategyManager and cleanup resources."""
        if not self._running:
            return
        
        logger.info("Shutting down StrategyManager...")
        
        self._running = False
        
        # Stop all active strategies
        async with self._lock:
            strategy_ids = list(self.active_strategies.keys())
            for strategy_id in strategy_ids:
                try:
                    await self._stop_strategy_internal(strategy_id)
                except Exception as e:
                    logger.error(f"Error stopping strategy {strategy_id}: {e}")
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._stats_task:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass
        
        logger.info("StrategyManager shutdown complete")
    
    async def start_strategy(
        self, 
        config_path: str, 
        initial_balance: float, 
        user_id: str
    ) -> str:
        """
        Start a new trading strategy.
        
        Args:
            config_path: Path to the YAML strategy configuration
            initial_balance: Initial trading balance for the strategy
            user_id: User ID for the strategy owner
            
        Returns:
            Strategy ID for the newly started strategy
            
        Raises:
            StrategyManagerError: If strategy cannot be started
        """
        async with self._lock:
            # Check concurrent strategy limit
            if len(self.active_strategies) >= self.max_concurrent_strategies:
                raise StrategyManagerError(
                    f"Maximum concurrent strategies ({self.max_concurrent_strategies}) reached"
                )
            
            # Validate config file exists
            config_file = Path(config_path)
            if not config_file.exists():
                raise StrategyManagerError(f"Configuration file not found: {config_path}")
            
            # Generate unique strategy ID
            strategy_id = str(uuid.uuid4())
            
            try:
                logger.info(f"Starting strategy {strategy_id[:8]} with config {config_path}")
                
                # Register strategy in Redis
                await self.registry.register_strategy(
                    strategy_id, config_path, initial_balance, user_id
                )
                
                # Create paper trader instance for this strategy
                strategy_paper_trader = PaperTrader(
                    initial_balance=initial_balance,
                    commission_rate=0.001,  # 0.1% commission
                    slippage_rate=0.0005  # 0.05% slippage
                )
                
                # Create strategy runner
                strategy_runner = StrategyRunner(
                    config_path=config_path,
                    paper_trader=strategy_paper_trader,
                    stream_manager=self.stream_manager
                )
                
                # Start the strategy
                await strategy_runner.start()
                
                # Store references
                self.active_strategies[strategy_id] = strategy_runner
                
                # Create monitoring task
                self.strategy_tasks[strategy_id] = asyncio.create_task(
                    self._monitor_strategy(strategy_id, strategy_runner)
                )
                
                # Update registry status
                await self.registry.update_strategy_status(strategy_id, "running")
                
                logger.info(f"Strategy {strategy_id[:8]} started successfully")
                return strategy_id
                
            except Exception as e:
                logger.error(f"Failed to start strategy {strategy_id[:8]}: {e}")
                
                # Cleanup on failure
                await self.registry.update_strategy_status(strategy_id, "failed", str(e))
                if strategy_id in self.active_strategies:
                    try:
                        await self.active_strategies[strategy_id].stop()
                        del self.active_strategies[strategy_id]
                    except Exception:
                        pass
                
                raise StrategyManagerError(f"Failed to start strategy: {e}") from e
    
    async def stop_strategy(self, strategy_id: str) -> None:
        """
        Stop a running strategy.
        
        Args:
            strategy_id: ID of the strategy to stop
            
        Raises:
            StrategyManagerError: If strategy cannot be stopped
        """
        async with self._lock:
            await self._stop_strategy_internal(strategy_id)
    
    async def _stop_strategy_internal(self, strategy_id: str) -> None:
        """Internal method to stop a strategy (assumes lock is held)."""
        if strategy_id not in self.active_strategies:
            raise StrategyManagerError(f"Strategy not found: {strategy_id}")
        
        try:
            logger.info(f"Stopping strategy {strategy_id[:8]}")
            
            # Update registry status
            await self.registry.update_strategy_status(strategy_id, "stopping")
            
            # Stop the strategy runner
            strategy_runner = self.active_strategies[strategy_id]
            await strategy_runner.stop()
            
            # Cancel monitoring task
            if strategy_id in self.strategy_tasks:
                self.strategy_tasks[strategy_id].cancel()
                try:
                    await self.strategy_tasks[strategy_id]
                except asyncio.CancelledError:
                    pass
                del self.strategy_tasks[strategy_id]
            
            # Remove from active strategies
            del self.active_strategies[strategy_id]
            
            # Update registry status
            await self.registry.update_strategy_status(strategy_id, "stopped")
            
            # Unregister from Redis after a delay (keep for audit trail)
            asyncio.create_task(self._delayed_unregister(strategy_id))
            
            logger.info(f"Strategy {strategy_id[:8]} stopped successfully")
            
        except Exception as e:
            logger.error(f"Failed to stop strategy {strategy_id[:8]}: {e}")
            await self.registry.update_strategy_status(strategy_id, "error", str(e))
            raise StrategyManagerError(f"Failed to stop strategy: {e}") from e
    
    async def get_active_strategies(self, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Get list of active strategies.
        
        Args:
            user_id: Optional user filter
            
        Returns:
            List of strategy information dictionaries
        """
        try:
            # Get strategies from registry
            registry_strategies = await self.registry.list_active_strategies(user_id)
            
            # Enhance with runtime information
            enhanced_strategies = []
            for strategy_info in registry_strategies:
                strategy_id = strategy_info["id"]
                
                # Add runtime status if available
                if strategy_id in self.active_strategies:
                    runner = self.active_strategies[strategy_id]
                    runtime_stats = runner.get_statistics()
                    
                    # Merge registry info with runtime stats
                    enhanced_info = {**strategy_info, **runtime_stats}
                    enhanced_strategies.append(enhanced_info)
                else:
                    enhanced_strategies.append(strategy_info)
            
            return enhanced_strategies
            
        except Exception as e:
            logger.error(f"Failed to get active strategies: {e}")
            return []
    
    async def get_strategy_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a specific strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Strategy status dictionary or None if not found
        """
        try:
            # Get registry information
            registry_info = await self.registry.get_strategy_info(strategy_id)
            if not registry_info:
                return None
            
            # Get runtime information if available
            if strategy_id in self.active_strategies:
                runner = self.active_strategies[strategy_id]
                runtime_stats = runner.get_statistics()
                
                # Get trading information
                position = runner.get_current_position()
                recent_trades = runner.get_recent_trades(limit=5)
                recent_orders = runner.get_recent_orders(limit=5)
                
                # Combine all information
                status_info = {
                    **registry_info,
                    **runtime_stats,
                    "current_position": position,
                    "recent_trades": recent_trades,
                    "recent_orders": recent_orders
                }
            else:
                # Strategy not running, return registry info only
                status_info = registry_info
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get strategy status {strategy_id}: {e}")
            return None
    
    async def get_strategy_statistics(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed performance statistics for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Strategy statistics dictionary or None if not found
        """
        try:
            # Get from Redis cache first
            cached_stats = await self.registry.get_strategy_stats(strategy_id)
            if cached_stats:
                return cached_stats["stats"]
            
            # Get from active strategy if running
            if strategy_id in self.active_strategies:
                runner = self.active_strategies[strategy_id]
                stats = runner.get_statistics()
                
                # Cache the stats
                await self.registry.update_strategy_stats(strategy_id, stats)
                
                return stats
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get strategy statistics {strategy_id}: {e}")
            return None
    
    async def _monitor_strategy(self, strategy_id: str, strategy_runner: StrategyRunner) -> None:
        """Monitor a strategy and handle errors."""
        try:
            while self._running and strategy_runner.is_running:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Update statistics
                try:
                    stats = strategy_runner.get_statistics()
                    await self.registry.update_strategy_stats(strategy_id, stats)
                except Exception as e:
                    logger.error(f"Failed to update stats for strategy {strategy_id}: {e}")
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Strategy monitor error for {strategy_id}: {e}")
            await self.registry.update_strategy_status(strategy_id, "error", str(e))
    
    async def _restore_strategies(self) -> None:
        """Restore strategies from Redis registry after restart."""
        try:
            active_strategies = await self.registry.list_active_strategies()
            
            for strategy_info in active_strategies:
                strategy_id = strategy_info["id"]
                
                # Mark as orphaned (will be cleaned up if not restored)
                await self.registry.update_strategy_status(
                    strategy_id, "orphaned", "API server restarted"
                )
            
            logger.info(f"Found {len(active_strategies)} strategies in registry after restart")
            
        except Exception as e:
            logger.error(f"Failed to restore strategies: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background task for cleanup and maintenance."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Cleanup expired strategies
                await self.registry.cleanup_expired_strategies()
                
                # Check for orphaned strategies
                async with self._lock:
                    active_strategy_ids = set(self.active_strategies.keys())
                    registry_strategies = await self.registry.list_active_strategies()
                    registry_ids = {s["id"] for s in registry_strategies}
                    
                    # Mark orphaned strategies
                    orphaned_ids = registry_ids - active_strategy_ids
                    for strategy_id in orphaned_ids:
                        await self.registry.update_strategy_status(
                            strategy_id, "orphaned", "Strategy runner not found"
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _stats_update_loop(self) -> None:
        """Background task for updating strategy statistics."""
        while self._running:
            try:
                await asyncio.sleep(self.stats_update_interval)
                
                # Update statistics for all active strategies
                async with self._lock:
                    for strategy_id, runner in self.active_strategies.items():
                        try:
                            stats = runner.get_statistics()
                            await self.registry.update_strategy_stats(strategy_id, stats)
                        except Exception as e:
                            logger.error(f"Failed to update stats for {strategy_id}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stats update loop error: {e}")
    
    async def _delayed_unregister(self, strategy_id: str) -> None:
        """Unregister a strategy after a delay for audit trail."""
        try:
            await asyncio.sleep(3600)  # Wait 1 hour
            await self.registry.unregister_strategy(strategy_id)
        except Exception as e:
            logger.error(f"Failed to unregister strategy {strategy_id}: {e}")
    
    @asynccontextmanager
    async def get_strategy_runner(self, strategy_id: str):
        """Context manager to safely access a strategy runner."""
        async with self._lock:
            if strategy_id not in self.active_strategies:
                raise StrategyManagerError(f"Strategy not found: {strategy_id}")
            
            yield self.active_strategies[strategy_id]