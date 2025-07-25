"""
Comprehensive tests for statistics and data endpoints.

This module tests all the statistics endpoints including trade history,
positions, orders, market data, and performance metrics. It uses mocked
dependencies to ensure isolated testing.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import redis.asyncio as redis
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import the components to test
from btc_research.api.routers.statistics import router
from btc_research.api.data_access import DataAccess, PaginationResult
from btc_research.api.performance import PerformanceCalculator, PerformanceMetrics
from btc_research.api.cache import CacheManager
from btc_research.live.paper_trader import PaperTrader, Trade, Position, Order, OrderSide, OrderStatus, OrderType


@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    
    # Mock app state
    app.state.strategy_manager = AsyncMock()
    app.state.stream_manager = AsyncMock()
    app.state.limiter = MagicMock()
    
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis_mock = AsyncMock(spec=redis.Redis)
    
    # Use async coroutine functions for async methods
    async def mock_ping():
        return True
    
    async def mock_get(key):
        return None
    
    async def mock_setex(key, ttl, value):
        return True
    
    async def mock_delete(key):
        return 1
    
    async def mock_keys(pattern):
        return []
    
    async def mock_info():
        return {
            'connected_clients': 1,
            'used_memory_human': '1M',
            'keyspace_hits': 100,
            'keyspace_misses': 50
        }
    
    redis_mock.ping.side_effect = mock_ping
    redis_mock.get.side_effect = mock_get
    redis_mock.setex.side_effect = mock_setex
    redis_mock.delete.side_effect = mock_delete
    redis_mock.keys.side_effect = mock_keys
    redis_mock.info.side_effect = mock_info
    
    return redis_mock


@pytest.fixture
def mock_strategy_manager():
    """Create mock StrategyManager."""
    manager = MagicMock()
    
    # Mock strategy statistics - needs to be awaitable for async calls
    async def mock_get_strategy_statistics(*args, **kwargs):
        return {
        'strategy_id': 'test-strategy-123',
        'total_return': 500.0,
        'total_return_pct': 5.0,
        'total_trades': 10,
        'winning_trades': 6,
        'losing_trades': 4,
        'win_rate': 0.6,
        'sharpe_ratio': 1.2,
        'max_drawdown': -150.0,
        'max_drawdown_pct': -1.5,
        'total_balance': 10500.0,
        'initial_balance': 10000.0,
        'total_unrealized_pnl': 0.0
    }
    
    manager.get_strategy_statistics.side_effect = mock_get_strategy_statistics
    
    # Mock strategy runner context manager
    mock_runner = AsyncMock()
    mock_paper_trader = create_mock_paper_trader()
    mock_runner.paper_trader = mock_paper_trader
    
    # Create async context manager that actually works
    class MockAsyncContextManager:
        def __init__(self, return_value):
            self.return_value = return_value
        
        async def __aenter__(self):
            return self.return_value
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None
    
    def mock_get_strategy_runner(*args, **kwargs):
        return MockAsyncContextManager(mock_runner)
    
    manager.get_strategy_runner.side_effect = mock_get_strategy_runner
    
    # Mock stream manager
    manager.stream_manager = create_mock_stream_manager()
    
    return manager


def create_mock_paper_trader():
    """Create mock PaperTrader with sample data."""
    paper_trader = MagicMock(spec=PaperTrader)
    
    # Sample trades
    sample_trades = [
        Trade(
            id="trade-1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            size=0.1,
            price=42000.0,
            commission=4.2,
            order_id="order-1",
            timestamp=datetime.utcnow() - timedelta(hours=2)
        ),
        Trade(
            id="trade-2",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            size=0.05,
            price=43000.0,
            commission=2.15,
            order_id="order-2",
            timestamp=datetime.utcnow() - timedelta(hours=1)
        )
    ]
    
    paper_trader.get_trade_history.return_value = sample_trades
    
    # Sample positions
    sample_position = Position(
        symbol="BTC/USDT",
        size=0.05,
        average_price=42500.0,
        unrealized_pnl=25.0
    )
    
    paper_trader.get_positions.return_value = {"BTC/USDT": sample_position}
    
    # Sample orders
    sample_orders = {
        "order-1": Order(
            id="order-1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.1,
            status=OrderStatus.FILLED,
            filled_size=0.1,
            average_fill_price=42000.0,
            commission=4.2
        ),
        "order-2": Order(
            id="order-2",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            size=0.05,
            status=OrderStatus.FILLED,
            filled_size=0.05,
            average_fill_price=43000.0,
            commission=2.15
        )
    }
    
    paper_trader.get_orders.return_value = sample_orders
    paper_trader.get_current_price.return_value = 43000.0
    
    # Performance stats
    paper_trader.get_performance_stats.return_value = {
        'total_balance': 10500.0,
        'available_balance': 10450.0,
        'total_equity': 10525.0,
        'total_return': 525.0,
        'total_trades': 2,
        'winning_trades': 1,
        'losing_trades': 1
    }
    
    return paper_trader


def create_mock_stream_manager():
    """Create mock StreamManager."""
    stream_manager = MagicMock()
    
    stream_manager.get_latest_price.return_value = 43000.0
    
    # Mock OHLCV data
    import pandas as pd
    sample_data = pd.DataFrame({
        'open': [42000.0, 42100.0, 42200.0],
        'high': [42200.0, 42300.0, 42400.0],
        'low': [41900.0, 42000.0, 42100.0],
        'close': [42100.0, 42200.0, 43000.0],
        'volume': [100.0, 150.0, 200.0]
    }, index=pd.DatetimeIndex([
        datetime.utcnow() - timedelta(minutes=3),
        datetime.utcnow() - timedelta(minutes=2),
        datetime.utcnow() - timedelta(minutes=1)
    ], name='timestamp'))
    
    stream_manager.get_data.return_value = sample_data
    
    return stream_manager


@pytest.fixture
def data_access(mock_strategy_manager, mock_redis):
    """Create DataAccess instance with mocked dependencies."""
    return DataAccess(mock_strategy_manager, mock_redis)


class TestStatisticsEndpoints:
    """Test class for statistics endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_strategy_statistics_success(self, data_access):
        """Test successful strategy statistics retrieval."""
        # Test the data access method
        stats = await data_access.get_strategy_statistics("test-strategy-123")
        
        assert stats is not None
        assert stats['strategy_id'] == 'test-strategy-123'
        assert stats['total_return'] == 500.0
        assert stats['win_rate'] == 0.6
        assert stats['sharpe_ratio'] == 1.2
    
    @pytest.mark.asyncio
    async def test_get_strategy_statistics_not_found(self, mock_strategy_manager, mock_redis):
        """Test strategy statistics when strategy not found."""
        # Override the side_effect to return None for this test
        async def mock_get_none(*args, **kwargs):
            return None
        
        mock_strategy_manager.get_strategy_statistics.side_effect = mock_get_none
        
        data_access = DataAccess(mock_strategy_manager, mock_redis)
        stats = await data_access.get_strategy_statistics("nonexistent-strategy")
        
        assert stats is None
    
    @pytest.mark.asyncio
    async def test_get_strategy_trades_success(self, data_access):
        """Test successful trade history retrieval."""
        result = await data_access.get_strategy_trades("test-strategy-123", skip=0, limit=10)
        
        assert isinstance(result, PaginationResult)
        assert len(result.items) >= 1
        assert result.total >= 1
        assert result.skip == 0
        assert result.limit == 10
        
        # Check trade data structure
        trade = result.items[0]
        assert 'id' in trade
        assert 'symbol' in trade
        assert 'side' in trade
        assert 'size' in trade
        assert 'price' in trade
        assert 'commission' in trade
        assert 'timestamp' in trade
    
    @pytest.mark.asyncio
    async def test_get_strategy_trades_with_symbol_filter(self, data_access):
        """Test trade history with symbol filter."""
        result = await data_access.get_strategy_trades(
            "test-strategy-123", 
            skip=0, 
            limit=10, 
            symbol="BTC/USDT"
        )
        
        assert isinstance(result, PaginationResult)
        
        # All returned trades should match the filter
        for trade in result.items:
            assert trade['symbol'] == "BTC/USDT"
    
    @pytest.mark.asyncio
    async def test_get_strategy_positions_success(self, data_access):
        """Test successful positions retrieval."""
        positions = await data_access.get_strategy_positions("test-strategy-123")
        
        assert isinstance(positions, list)
        assert len(positions) >= 1
        
        # Check position data structure
        position = positions[0]
        assert 'symbol' in position
        assert 'side' in position
        assert 'size' in position
        assert 'average_price' in position
        assert 'unrealized_pnl' in position
        assert 'unrealized_pnl_pct' in position
    
    @pytest.mark.asyncio
    async def test_get_strategy_positions_with_symbol_filter(self, data_access):
        """Test positions with symbol filter."""
        positions = await data_access.get_strategy_positions(
            "test-strategy-123", 
            symbol="BTC/USDT"
        )
        
        assert isinstance(positions, list)
        
        # All returned positions should match the filter
        for position in positions:
            assert position['symbol'] == "BTC/USDT"
    
    @pytest.mark.asyncio
    async def test_get_strategy_orders_success(self, data_access):
        """Test successful order history retrieval."""
        result = await data_access.get_strategy_orders("test-strategy-123", skip=0, limit=10)
        
        assert isinstance(result, PaginationResult)
        assert len(result.items) >= 1
        assert result.total >= 1
        
        # Check order data structure
        order = result.items[0]
        assert 'id' in order
        assert 'symbol' in order
        assert 'side' in order
        assert 'type' in order
        assert 'size' in order
        assert 'status' in order
        assert 'commission' in order
    
    @pytest.mark.asyncio
    async def test_get_strategy_orders_with_status_filter(self, data_access):
        """Test order history with status filter."""
        result = await data_access.get_strategy_orders(
            "test-strategy-123", 
            skip=0, 
            limit=10, 
            status_filter="filled"
        )
        
        assert isinstance(result, PaginationResult)
        
        # All returned orders should match the filter
        for order in result.items:
            assert order['status'] == "filled"
    
    @pytest.mark.asyncio
    async def test_get_market_data_success(self, data_access):
        """Test successful market data retrieval."""
        market_data = await data_access.get_market_data("BTC/USDT", "1m", 100)
        
        assert market_data is not None
        assert market_data['symbol'] == "BTC/USDT"
        assert market_data['timeframe'] == "1m"
        assert 'current_price' in market_data
        assert 'candles' in market_data
        assert isinstance(market_data['candles'], list)
        
        # Check candle data structure
        if market_data['candles']:
            candle = market_data['candles'][0]
            assert 'timestamp' in candle
            assert 'open' in candle
            assert 'high' in candle
            assert 'low' in candle
            assert 'close' in candle
            assert 'volume' in candle
    
    @pytest.mark.asyncio
    async def test_get_strategy_performance_success(self, data_access):
        """Test successful performance data retrieval."""
        performance_data = await data_access.get_strategy_performance("test-strategy-123", "1d")
        
        assert performance_data is not None
        assert performance_data['strategy_id'] == "test-strategy-123"
        assert performance_data['period'] == "1d"
        assert 'pnl_history' in performance_data
        assert 'key_metrics' in performance_data
        assert isinstance(performance_data['pnl_history'], list)
        assert isinstance(performance_data['key_metrics'], dict)
        
        # Check key metrics structure
        metrics = performance_data['key_metrics']
        assert 'total_return' in metrics
        assert 'total_return_pct' in metrics
        assert 'total_trades' in metrics


class TestPerformanceCalculator:
    """Test class for performance calculation utilities."""
    
    def test_performance_calculator_initialization(self):
        """Test PerformanceCalculator initialization."""
        calculator = PerformanceCalculator(risk_free_rate=0.02)
        assert calculator.risk_free_rate == 0.02
    
    def test_calculate_metrics_empty_trades(self):
        """Test performance calculation with empty trade list."""
        calculator = PerformanceCalculator()
        metrics = calculator.calculate_metrics([], initial_balance=10000.0)
        
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_return == 0.0
    
    def test_calculate_metrics_with_trades(self):
        """Test performance calculation with sample trades."""
        calculator = PerformanceCalculator()
        
        sample_trades = [
            {
                'id': 'trade-1',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'size': '0.1',
                'price': '42000.0',
                'commission': '4.2',
                'timestamp': datetime.utcnow().isoformat()
            },
            {
                'id': 'trade-2',
                'symbol': 'BTC/USDT',
                'side': 'sell',
                'size': '0.1',
                'price': '43000.0',
                'commission': '4.3',
                'timestamp': datetime.utcnow().isoformat()
            }
        ]
        
        metrics = calculator.calculate_metrics(
            trades=sample_trades,
            initial_balance=10000.0,
            current_balance=10100.0
        )
        
        assert metrics.total_trades == 2
        assert metrics.total_return == 100.0
        assert abs(metrics.total_return_pct - 1.0) < 0.001  # Handle floating point precision
        assert isinstance(metrics.to_dict(), dict)
    
    def test_calculate_rolling_metrics(self):
        """Test rolling metrics calculation."""
        calculator = PerformanceCalculator()
        
        # Sample balance history
        balance_history = []
        base_time = datetime.utcnow() - timedelta(days=60)
        
        for i in range(60):
            balance_history.append({
                'timestamp': (base_time + timedelta(days=i)).isoformat(),
                'balance': 10000 + (i * 10)  # Gradually increasing balance
            })
        
        rolling_metrics = calculator.calculate_rolling_metrics(balance_history, window_days=30)
        
        assert isinstance(rolling_metrics, list)
        assert len(rolling_metrics) > 0
        
        if rolling_metrics:
            metric = rolling_metrics[0]
            assert 'timestamp' in metric
            assert 'balance' in metric
            assert 'rolling_return' in metric
            assert 'rolling_volatility' in metric
            assert 'rolling_sharpe' in metric


class TestCacheManager:
    """Test class for cache management."""
    
    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, mock_redis):
        """Test CacheManager initialization."""
        cache_manager = CacheManager(mock_redis)
        assert cache_manager.redis == mock_redis
        assert cache_manager.key_prefix == "api_cache"
        assert cache_manager.hits == 0
        assert cache_manager.misses == 0
    
    def test_get_cache_key_generation(self, mock_redis):
        """Test cache key generation."""
        cache_manager = CacheManager(mock_redis)
        
        key1 = cache_manager.get_cache_key("test_namespace", "arg1", "arg2", kwarg1="value1")
        key2 = cache_manager.get_cache_key("test_namespace", "arg1", "arg2", kwarg1="value1")
        key3 = cache_manager.get_cache_key("test_namespace", "arg1", "arg2", kwarg1="value2")
        
        # Same arguments should generate same key
        assert key1 == key2
        
        # Different arguments should generate different keys
        assert key1 != key3
        
        # Key should contain namespace
        assert "test_namespace" in key1
    
    @pytest.mark.asyncio
    async def test_cache_get_miss(self, mock_redis):
        """Test cache miss."""
        mock_redis.get.return_value = None
        
        cache_manager = CacheManager(mock_redis)
        result = await cache_manager.get("test_key", default="default_value")
        
        assert result == "default_value"
        assert cache_manager.misses == 1
        assert cache_manager.hits == 0
    
    @pytest.mark.asyncio
    async def test_cache_get_hit(self, mock_redis):
        """Test cache hit."""
        # Override the mock to return data for this test
        async def mock_get_with_data(key):
            return json.dumps({"cached": "data"})
        
        mock_redis.get.side_effect = mock_get_with_data
        
        cache_manager = CacheManager(mock_redis)
        result = await cache_manager.get("test_key")
        
        assert result == {"cached": "data"}
        assert cache_manager.hits == 1
        assert cache_manager.misses == 0
    
    @pytest.mark.asyncio
    async def test_cache_set(self, mock_redis):
        """Test cache set operation."""
        cache_manager = CacheManager(mock_redis)
        
        success = await cache_manager.set("test_key", {"data": "value"}, ttl=300)
        
        assert success is True
        mock_redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, mock_redis):
        """Test cache statistics."""
        cache_manager = CacheManager(mock_redis)
        cache_manager.hits = 80
        cache_manager.misses = 20
        cache_manager.errors = 5
        
        stats = await cache_manager.get_stats()
        
        assert stats['cache_hits'] == 80
        assert stats['cache_misses'] == 20
        assert stats['cache_errors'] == 5
        assert stats['hit_rate_pct'] == 80.0  # 80/(80+20) * 100
        assert 'redis_info' in stats


@pytest.mark.asyncio
async def test_integration_data_access_with_cache(mock_strategy_manager, mock_redis):
    """Test DataAccess integration with caching."""
    data_access = DataAccess(mock_strategy_manager, mock_redis)
    
    # First call should miss cache and fetch from strategy manager
    stats1 = await data_access.get_strategy_statistics("test-strategy-123")
    assert stats1 is not None
    
    # Second call should potentially hit cache (depending on cache decorator implementation)
    stats2 = await data_access.get_strategy_statistics("test-strategy-123")
    assert stats2 is not None
    
    # Results should be the same
    assert stats1['strategy_id'] == stats2['strategy_id']


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])