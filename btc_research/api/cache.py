"""
Redis caching utilities for API performance optimization.

This module provides caching decorators and utilities for expensive calculations
such as performance metrics, trade analytics, and market data processing.
It implements intelligent cache invalidation and supports both sync and async
operations.
"""

import asyncio
import functools
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import redis.asyncio as redis

__all__ = ["CacheManager", "cache_result", "invalidate_cache", "get_cache_key"]

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Centralized cache management for API operations.
    
    Provides intelligent caching with TTL, cache invalidation,
    and performance monitoring for Redis-backed caching.
    """
    
    def __init__(self, redis_client: redis.Redis, key_prefix: str = "api_cache"):
        """
        Initialize cache manager.
        
        Args:
            redis_client: Redis client instance
            key_prefix: Prefix for all cache keys
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.errors = 0
        
        logger.info("CacheManager initialized")
    
    def get_cache_key(self, namespace: str, *args, **kwargs) -> str:
        """
        Generate consistent cache key from arguments.
        
        Args:
            namespace: Cache namespace (e.g., 'strategy_stats', 'market_data')
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Generated cache key
        """
        # Create deterministic hash from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())  # Sort for consistency
        }
        
        key_json = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_json.encode()).hexdigest()
        
        return f"{self.key_prefix}:{namespace}:{key_hash}"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        try:
            cached_data = await self.redis.get(key)
            
            if cached_data:
                self.hits += 1
                return json.loads(cached_data)
            else:
                self.misses += 1
                return default
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.errors += 1
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 300,
        serialize: bool = True
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize: Whether to JSON serialize the value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if serialize:
                cache_data = json.dumps(value, default=str)
            else:
                cache_data = value
            
            await self.redis.setex(key, ttl, cache_data)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.redis.delete(key)
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.errors += 1
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.
        
        Args:
            pattern: Redis pattern (e.g., "strategy_stats:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            full_pattern = f"{self.key_prefix}:{pattern}"
            keys = await self.redis.keys(full_pattern)
            
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(f"Invalidated {deleted} cache keys matching {full_pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache invalidation error for pattern {pattern}: {e}")
            self.errors += 1
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        # Get Redis info
        redis_info = {}
        try:
            info = await self.redis.info()
            redis_info = {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.warning(f"Failed to get Redis info: {e}")
        
        return {
            'cache_hits': self.hits,
            'cache_misses': self.misses,
            'cache_errors': self.errors,
            'hit_rate_pct': hit_rate,
            'redis_info': redis_info
        }
    
    async def clear_all(self) -> bool:
        """
        Clear all cache entries with the configured prefix.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            pattern = f"{self.key_prefix}:*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.errors += 1
            return False


def cache_result(
    namespace: str,
    ttl: int = 300,
    key_func: Optional[Callable] = None,
    invalidate_on_error: bool = False
):
    """
    Decorator for caching function results.
    
    Args:
        namespace: Cache namespace
        ttl: Time to live in seconds
        key_func: Custom function to generate cache key
        invalidate_on_error: Whether to invalidate cache on function error
        
    Example:
        @cache_result('strategy_stats', ttl=60)
        async def get_strategy_statistics(strategy_id: str):
            # Expensive calculation
            return stats
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache manager from first argument (assumed to be self with cache_manager)
            cache_manager = None
            
            if args and hasattr(args[0], 'cache_manager'):
                cache_manager = args[0].cache_manager
            elif 'cache_manager' in kwargs:
                cache_manager = kwargs.pop('cache_manager')
            
            if not cache_manager:
                # No cache manager available, execute function directly
                return await func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager.get_cache_key(namespace, *args, **kwargs)
            
            # Try to get from cache first
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            try:
                result = await func(*args, **kwargs)
                
                # Cache the result
                await cache_manager.set(cache_key, result, ttl)
                
                return result
                
            except Exception as e:
                if invalidate_on_error:
                    await cache_manager.delete(cache_key)
                raise
        
        return wrapper
    return decorator


async def invalidate_cache(
    cache_manager: CacheManager,
    namespace: str,
    *args,
    **kwargs
) -> bool:
    """
    Invalidate specific cache entry.
    
    Args:
        cache_manager: CacheManager instance
        namespace: Cache namespace
        *args: Function arguments used to generate key
        **kwargs: Function keyword arguments used to generate key
        
    Returns:
        True if invalidation was successful
    """
    cache_key = cache_manager.get_cache_key(namespace, *args, **kwargs)
    return await cache_manager.delete(cache_key)


def get_cache_key(namespace: str, *args, **kwargs) -> str:
    """
    Generate cache key without CacheManager instance.
    
    Args:
        namespace: Cache namespace
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Generated cache key
    """
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    
    key_json = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_json.encode()).hexdigest()
    
    return f"api_cache:{namespace}:{key_hash}"


class CacheInvalidationManager:
    """
    Manages cache invalidation triggers for different data types.
    
    Automatically invalidates related caches when data changes occur.
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize cache invalidation manager.
        
        Args:
            cache_manager: CacheManager instance
        """
        self.cache_manager = cache_manager
        
        # Define invalidation rules
        self.invalidation_rules = {
            'strategy_update': [
                'strategy_stats:*',
                'strategy_performance:*',
                'strategy_trades:*',
                'strategy_positions:*',
                'strategy_orders:*'
            ],
            'trade_execution': [
                'strategy_stats:*',
                'strategy_performance:*',
                'strategy_trades:*',
                'strategy_positions:*'
            ],
            'position_update': [
                'strategy_stats:*',
                'strategy_positions:*'
            ],
            'market_data_update': [
                'market_data:*',
                'strategy_positions:*'  # For unrealized P&L updates
            ]
        }
    
    async def invalidate_for_event(self, event_type: str, strategy_id: str = None) -> int:
        """
        Invalidate caches based on event type.
        
        Args:
            event_type: Type of event that occurred
            strategy_id: Optional strategy ID for targeted invalidation
            
        Returns:
            Number of cache entries invalidated
        """
        if event_type not in self.invalidation_rules:
            logger.warning(f"Unknown cache invalidation event: {event_type}")
            return 0
        
        total_invalidated = 0
        patterns = self.invalidation_rules[event_type]
        
        for pattern in patterns:
            # If strategy_id is provided, make invalidation more specific
            if strategy_id and ':*' in pattern:
                specific_pattern = pattern.replace(':*', f':*{strategy_id}*')
                invalidated = await self.cache_manager.invalidate_pattern(specific_pattern)
            else:
                invalidated = await self.cache_manager.invalidate_pattern(pattern)
            
            total_invalidated += invalidated
        
        if total_invalidated > 0:
            logger.info(f"Invalidated {total_invalidated} cache entries for event: {event_type}")
        
        return total_invalidated
    
    async def schedule_periodic_cleanup(self, interval_minutes: int = 60):
        """
        Schedule periodic cache cleanup task.
        
        Args:
            interval_minutes: Cleanup interval in minutes
        """
        while True:
            try:
                await asyncio.sleep(interval_minutes * 60)
                
                # Get cache stats
                stats = await self.cache_manager.get_stats()
                logger.info(f"Cache stats: {stats}")
                
                # Clean up expired entries (Redis handles this automatically)
                # but we can log statistics
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")