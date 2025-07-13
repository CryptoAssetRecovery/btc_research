"""
Comprehensive unit tests for the DataFeed service.

This module tests all aspects of the DataFeed including:
- Data fetching and caching
- Timeframe conversion and resampling
- Error handling for network failures and invalid symbols
- Cache performance and persistence
- Mock CCXT responses for reliable testing
"""

import shutil
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import ccxt
import numpy as np
import pandas as pd
import pytest

from btc_research.core.datafeed import (
    DataFeed,
    DataFeedError,
    ValidationError,
)


class TestDataFeedValidation:
    """Test input validation and error handling."""

    def test_invalid_symbol(self):
        """Test validation of invalid symbols."""
        feed = DataFeed(cache_enabled=False)

        with pytest.raises(DataFeedError, match="Symbol must be a non-empty string"):
            feed.get("", "1h", "2024-01-01", "2024-01-02")

        with pytest.raises(DataFeedError, match="Symbol must be a non-empty string"):
            feed.get(None, "1h", "2024-01-01", "2024-01-02")

    def test_invalid_timeframe(self):
        """Test validation of invalid timeframes."""
        feed = DataFeed(cache_enabled=False)

        with pytest.raises(DataFeedError, match="Unsupported timeframe"):
            feed.get("BTC/USD", "3m", "2024-01-01", "2024-01-02")

        with pytest.raises(DataFeedError, match="Unsupported timeframe"):
            feed.get("BTC/USD", "invalid", "2024-01-01", "2024-01-02")

    def test_invalid_date_range(self):
        """Test validation of invalid date ranges."""
        feed = DataFeed(cache_enabled=False)

        # End before start
        with pytest.raises(
            DataFeedError, match="Start datetime must be before end datetime"
        ):
            feed.get("BTC/USD", "1h", "2024-01-02", "2024-01-01")

        # Same datetime
        with pytest.raises(
            DataFeedError, match="Start datetime must be before end datetime"
        ):
            feed.get("BTC/USD", "1h", "2024-01-01", "2024-01-01")

    def test_invalid_datetime_format(self):
        """Test validation of invalid datetime formats."""
        feed = DataFeed(cache_enabled=False)

        with pytest.raises(DataFeedError, match="Invalid datetime format"):
            feed.get("BTC/USD", "1h", "invalid-date", "2024-01-02")

    def test_future_date_validation(self):
        """Test validation of dates too far in the future."""
        feed = DataFeed(cache_enabled=False)

        future_start = (datetime.now(UTC) + timedelta(days=30)).strftime(
            "%Y-%m-%d"
        )
        future_end = (datetime.now(UTC) + timedelta(days=31)).strftime(
            "%Y-%m-%d"
        )

        with pytest.raises(
            DataFeedError, match="Start date cannot be more than 7 days in the future"
        ):
            feed.get("BTC/USD", "1h", future_start, future_end)


class TestDataFeedCaching:
    """Test caching functionality and performance."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2024-01-01 00:00:00", periods=100, freq="1min", tz="UTC")

        # Generate realistic OHLCV data
        np.random.seed(42)  # For reproducible tests
        base_price = 42000.0
        returns = np.random.normal(0, 0.001, len(dates))  # 0.1% std dev
        prices = base_price * np.exp(np.cumsum(returns))

        # Create OHLCV from price series
        data = []
        for i, (date, price) in enumerate(zip(dates, prices, strict=False)):
            high = price * (1 + abs(np.random.normal(0, 0.0005)))
            low = price * (1 - abs(np.random.normal(0, 0.0005)))
            open_price = prices[i - 1] if i > 0 else price
            close_price = price
            volume = np.random.uniform(10, 1000)

            data.append(
                {
                    "timestamp": int(date.timestamp() * 1000),
                    "open": open_price,
                    "high": max(high, open_price, close_price),
                    "low": min(low, open_price, close_price),
                    "close": close_price,
                    "volume": volume,
                }
            )

        return data

    @patch("ccxt.binanceus")
    def test_cache_miss_and_save(
        self, mock_exchange_class, temp_cache_dir, sample_ohlcv_data
    ):
        """Test cache miss, data fetching, and cache saving."""
        # Setup mock exchange
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_data
        mock_exchange.rateLimit = 1000
        mock_exchange_class.return_value = mock_exchange

        feed = DataFeed(cache_dir=temp_cache_dir)

        # First call should be cache miss
        df = feed.get("BTC/USD", "1m", "2024-01-01 00:00", "2024-01-01 01:39")

        assert len(df) == 100
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"

        # Verify cache file was created
        cache_file = Path(temp_cache_dir) / "binanceus" / "BTC-USD_1m.parquet"
        assert cache_file.exists()

        # Check cache stats
        stats = feed.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    @patch("ccxt.binanceus")
    def test_cache_hit_performance(
        self, mock_exchange_class, temp_cache_dir, sample_ohlcv_data
    ):
        """Test cache hit and performance requirements (<200ms)."""
        # Setup mock exchange
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_data
        mock_exchange.rateLimit = 1000
        mock_exchange_class.return_value = mock_exchange

        feed = DataFeed(cache_dir=temp_cache_dir)

        # First call to populate cache
        df1 = feed.get("BTC/USD", "1m", "2024-01-01 00:00", "2024-01-01 01:39")

        # Second call should hit cache
        start_time = time.time()
        df2 = feed.get("BTC/USD", "1m", "2024-01-01 00:00", "2024-01-01 01:39")
        load_time_ms = (time.time() - start_time) * 1000

        # Verify data is identical
        pd.testing.assert_frame_equal(df1, df2)

        # Verify performance requirement (<200ms)
        assert (
            load_time_ms < 200
        ), f"Cache load took {load_time_ms:.2f}ms, exceeds 200ms requirement"

        # Check cache stats
        stats = feed.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["avg_load_time_ms"] < 200

    @patch("ccxt.binanceus")
    def test_cache_persistence(
        self, mock_exchange_class, temp_cache_dir, sample_ohlcv_data
    ):
        """Test that cache persists across DataFeed instances."""
        # Setup mock exchange
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_data
        mock_exchange.rateLimit = 1000
        mock_exchange_class.return_value = mock_exchange

        # First instance - populate cache
        feed1 = DataFeed(cache_dir=temp_cache_dir)
        df1 = feed1.get("BTC/USD", "1m", "2024-01-01 00:00", "2024-01-01 01:39")

        # Second instance - should use cached data
        feed2 = DataFeed(cache_dir=temp_cache_dir)
        df2 = feed2.get("BTC/USD", "1m", "2024-01-01 00:00", "2024-01-01 01:39")

        # Data should be identical
        pd.testing.assert_frame_equal(df1, df2)

        # Second instance should have cache hit
        stats2 = feed2.get_cache_stats()
        assert stats2["hits"] == 1
        assert stats2["misses"] == 0

    def test_cache_disabled(self, sample_ohlcv_data):
        """Test DataFeed behavior with caching disabled."""
        with patch("ccxt.binanceus") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_data
            mock_exchange.rateLimit = 1000
            mock_exchange_class.return_value = mock_exchange

            feed = DataFeed(cache_enabled=False)

            # Multiple calls should all result in cache misses
            df1 = feed.get("BTC/USD", "1m", "2024-01-01 00:00", "2024-01-01 01:39")
            df2 = feed.get("BTC/USD", "1m", "2024-01-01 00:00", "2024-01-01 01:39")

            stats = feed.get_cache_stats()
            assert stats["hits"] == 0
            assert stats["misses"] == 2

    def test_cache_clear(self, temp_cache_dir, sample_ohlcv_data):
        """Test cache clearing functionality."""
        with patch("ccxt.binanceus") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_data
            mock_exchange.rateLimit = 1000
            mock_exchange_class.return_value = mock_exchange

            feed = DataFeed(cache_dir=temp_cache_dir)

            # Populate cache
            feed.get("BTC/USD", "1m", "2024-01-01 00:00", "2024-01-01 01:39")
            feed.get("ETH/USD", "1m", "2024-01-01 00:00", "2024-01-01 01:39")

            # Verify cache files exist
            btc_cache = Path(temp_cache_dir) / "binanceus" / "BTC-USD_1m.parquet"
            eth_cache = Path(temp_cache_dir) / "binanceus" / "ETH-USD_1m.parquet"
            assert btc_cache.exists()
            assert eth_cache.exists()

            # Clear specific symbol
            feed.clear_cache("BTC/USD", "binanceus")
            assert not btc_cache.exists()
            assert eth_cache.exists()

            # Clear all cache
            feed.clear_cache()
            assert not eth_cache.exists()


class TestDataFeedResampling:
    """Test timeframe resampling functionality."""

    @pytest.fixture
    def minute_data(self):
        """Create minute-level test data."""
        dates = pd.date_range("2024-01-01 00:00:00", periods=300, freq="1min", tz="UTC")

        # Create realistic minute data
        np.random.seed(42)
        base_price = 42000.0

        data = []
        current_price = base_price

        for i, date in enumerate(dates):
            # Small random walk
            change = np.random.normal(0, 10)  # $10 average change per minute
            current_price += change

            high = current_price + abs(np.random.normal(0, 5))
            low = current_price - abs(np.random.normal(0, 5))
            volume = np.random.uniform(10, 100)

            data.append(
                {
                    "open": current_price - change,
                    "high": high,
                    "low": low,
                    "close": current_price,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(data, index=dates)
        return df

    def test_5m_resampling(self, minute_data):
        """Test resampling from 1m to 5m data."""
        feed = DataFeed(cache_enabled=False)
        resampled = feed._resample_data(minute_data, "5m")

        # Should have 60 5-minute bars (300 minutes / 5)
        assert len(resampled) == 60

        # Check OHLCV aggregation is correct
        first_5m = resampled.iloc[0]
        first_5_1m = minute_data.iloc[0:5]

        assert first_5m["open"] == first_5_1m["open"].iloc[0]
        assert first_5m["high"] == first_5_1m["high"].max()
        assert first_5m["low"] == first_5_1m["low"].min()
        assert first_5m["close"] == first_5_1m["close"].iloc[-1]
        assert first_5m["volume"] == first_5_1m["volume"].sum()

    def test_1h_resampling(self, minute_data):
        """Test resampling from 1m to 1h data."""
        feed = DataFeed(cache_enabled=False)
        resampled = feed._resample_data(minute_data, "1h")

        # Should have 5 1-hour bars (300 minutes / 60)
        assert len(resampled) == 5

        # Verify OHLCV aggregation
        first_1h = resampled.iloc[0]
        first_60_1m = minute_data.iloc[0:60]

        assert first_1h["open"] == first_60_1m["open"].iloc[0]
        assert first_1h["high"] == first_60_1m["high"].max()
        assert first_1h["low"] == first_60_1m["low"].min()
        assert first_1h["close"] == first_60_1m["close"].iloc[-1]
        assert first_1h["volume"] == first_60_1m["volume"].sum()

    def test_1d_resampling(self, minute_data):
        """Test resampling from 1m to 1d data."""
        # Create a full day of minute data
        dates = pd.date_range(
            "2024-01-01 00:00:00", periods=1440, freq="1min", tz="UTC"
        )

        np.random.seed(42)
        data = []
        current_price = 42000.0

        for date in dates:
            change = np.random.normal(0, 1)
            current_price += change
            data.append(
                {
                    "open": current_price - change,
                    "high": current_price + abs(np.random.normal(0, 5)),
                    "low": current_price - abs(np.random.normal(0, 5)),
                    "close": current_price,
                    "volume": np.random.uniform(10, 100),
                }
            )

        df = pd.DataFrame(data, index=dates)

        feed = DataFeed(cache_enabled=False)
        resampled = feed._resample_data(df, "1d")

        # Should have 1 daily bar
        assert len(resampled) == 1

        daily_bar = resampled.iloc[0]
        assert daily_bar["open"] == df["open"].iloc[0]
        assert daily_bar["high"] == df["high"].max()
        assert daily_bar["low"] == df["low"].min()
        assert daily_bar["close"] == df["close"].iloc[-1]
        assert (
            abs(daily_bar["volume"] - df["volume"].sum()) < 1e-10
        )  # Handle floating point precision

    def test_invalid_timeframe_resampling(self, minute_data):
        """Test resampling to invalid timeframe."""
        feed = DataFeed(cache_enabled=False)

        with pytest.raises(
            DataFeedError, match="Cannot resample to unsupported timeframe"
        ):
            feed._resample_data(minute_data, "invalid")


class TestDataFeedValidationAndCleaning:
    """Test data validation and cleaning functionality."""

    def test_invalid_ohlc_cleaning(self):
        """Test cleaning of invalid OHLC data."""
        # Create data with invalid OHLC relationships
        dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")

        data = pd.DataFrame(
            {
                "open": [100, 100, 100, 100, 100],
                "high": [110, 90, 110, 110, 110],  # high < low for index 1
                "low": [90, 110, 90, 90, 90],  # low > high for index 1
                "close": [105, 105, 105, 105, 105],
                "volume": [1000, 1000, 1000, 1000, 1000],
            },
            index=dates,
        )

        feed = DataFeed(cache_enabled=False)

        with pytest.warns(UserWarning, match="Removed .* invalid OHLC rows"):
            cleaned = feed._validate_and_clean(data, "BTC/USD", "1h")

        # Should have removed the invalid row
        assert len(cleaned) == 4
        assert dates[1] not in cleaned.index

    def test_negative_price_cleaning(self):
        """Test cleaning of negative or zero prices."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")

        data = pd.DataFrame(
            {
                "open": [100, -100, 100, 0, 100],  # Negative and zero prices
                "high": [110, 110, 110, 110, 110],
                "low": [90, 90, 90, 90, 90],
                "close": [105, 105, 105, 105, 105],
                "volume": [1000, 1000, 1000, 1000, 1000],
            },
            index=dates,
        )

        feed = DataFeed(cache_enabled=False)

        with pytest.warns(UserWarning, match="Removed .* invalid OHLC rows"):
            cleaned = feed._validate_and_clean(data, "BTC/USD", "1h")

        # Should have removed rows with invalid prices
        assert len(cleaned) == 3
        assert dates[1] not in cleaned.index
        assert dates[3] not in cleaned.index

    def test_missing_columns_validation(self):
        """Test validation of missing required columns."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")

        # Missing 'close' column
        data = pd.DataFrame(
            {
                "open": [100, 100, 100, 100, 100],
                "high": [110, 110, 110, 110, 110],
                "low": [90, 90, 90, 90, 90],
                "volume": [1000, 1000, 1000, 1000, 1000],
            },
            index=dates,
        )

        feed = DataFeed(cache_enabled=False)

        with pytest.raises(ValidationError, match="Missing required columns"):
            feed._validate_and_clean(data, "BTC/USD", "1h")

    def test_empty_dataframe_validation(self):
        """Test validation of empty DataFrame."""
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty_df.index = pd.DatetimeIndex([], name="timestamp", tz="UTC")

        feed = DataFeed(cache_enabled=False)

        with pytest.warns(UserWarning, match="No data returned"):
            result = feed._validate_and_clean(empty_df, "BTC/USD", "1h")

        assert len(result) == 0
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]


class TestDataFeedErrorHandling:
    """Test error handling for various failure scenarios."""

    @patch("ccxt.binanceus")
    def test_exchange_network_error(self, mock_exchange_class):
        """Test handling of network errors."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("Connection failed")
        mock_exchange.rateLimit = 1000
        mock_exchange_class.return_value = mock_exchange

        feed = DataFeed(cache_enabled=False)

        with pytest.raises(DataFeedError, match="Failed to fetch data from binanceus"):
            feed.get("BTC/USD", "1h", "2024-01-01", "2024-01-02")

    @patch("ccxt.binanceus")
    def test_invalid_symbol_error(self, mock_exchange_class):
        """Test handling of invalid symbol errors."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = ccxt.BadSymbol("Invalid symbol")
        mock_exchange.rateLimit = 1000
        mock_exchange_class.return_value = mock_exchange

        feed = DataFeed(cache_enabled=False)

        with pytest.raises(
            DataFeedError, match="Invalid symbol 'BTC/USD' for exchange binanceus"
        ):
            feed.get("BTC/USD", "1h", "2024-01-01", "2024-01-02")

    @patch("ccxt.binanceus")
    def test_rate_limit_handling(self, mock_exchange_class):
        """Test handling of rate limit errors."""
        mock_exchange = MagicMock()
        # First call hits rate limit, second succeeds
        mock_exchange.fetch_ohlcv.side_effect = [
            ccxt.RateLimitExceeded("Rate limit exceeded"),
            [],  # Empty response after rate limit
        ]
        mock_exchange.rateLimit = 100  # Fast rate limit for testing
        mock_exchange_class.return_value = mock_exchange

        feed = DataFeed(cache_enabled=False)

        # Should handle rate limit and return empty DataFrame
        df = feed.get("BTC/USD", "1h", "2024-01-01", "2024-01-02")
        assert len(df) == 0
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_unsupported_exchange(self):
        """Test handling of unsupported exchange."""
        feed = DataFeed(cache_enabled=False)

        with pytest.raises(DataFeedError, match="Unsupported exchange"):
            feed.get("BTC/USD", "1h", "2024-01-01", "2024-01-02", source="nonexistent")

    def test_context_manager(self):
        """Test DataFeed as context manager."""
        with patch("ccxt.binanceus") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.return_value = []
            mock_exchange.rateLimit = 1000
            mock_exchange_class.return_value = mock_exchange

            # Test context manager usage
            with DataFeed(cache_enabled=False) as feed:
                df = feed.get("BTC/USD", "1h", "2024-01-01", "2024-01-02")
                assert len(df) == 0

            # Exchange should be cleaned up
            if hasattr(mock_exchange, "close"):
                mock_exchange.close.assert_called()


class TestDataFeedIntegration:
    """Integration tests combining multiple features."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for integration testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("ccxt.binanceus")
    def test_minute_to_hourly_workflow(self, mock_exchange_class, temp_cache_dir):
        """Test complete workflow: fetch minutes, cache, then resample to hourly."""
        # Create realistic minute data for 2 hours (120 minutes exactly)
        dates = pd.date_range("2024-01-01 10:00:00", periods=120, freq="1min", tz="UTC")

        ohlcv_data = []
        base_price = 42000.0

        for i, date in enumerate(dates):
            price = base_price + i * 10  # Trending up
            ohlcv_data.append(
                {
                    "timestamp": int(date.timestamp() * 1000),
                    "open": price,
                    "high": price + 50,
                    "low": price - 50,
                    "close": price + 25,
                    "volume": 100 + i,
                }
            )

        # Setup mock
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = ohlcv_data
        mock_exchange.rateLimit = 1000
        mock_exchange_class.return_value = mock_exchange

        feed = DataFeed(cache_dir=temp_cache_dir)

        # First, get minute data (should cache it)
        minute_df = feed.get("BTC/USD", "1m", "2024-01-01 10:00", "2024-01-01 11:59")
        assert len(minute_df) == 120

        # Now get hourly data (should resample from cached minute data)
        hourly_df = feed.get("BTC/USD", "1h", "2024-01-01 10:00", "2024-01-01 11:59")
        assert len(hourly_df) == 2  # 2 hourly periods

        # Verify first hour aggregation
        first_hour = hourly_df.iloc[0]
        first_60_minutes = minute_df.iloc[0:60]

        assert first_hour["open"] == first_60_minutes["open"].iloc[0]
        assert first_hour["high"] == first_60_minutes["high"].max()
        assert first_hour["low"] == first_60_minutes["low"].min()
        assert first_hour["close"] == first_60_minutes["close"].iloc[-1]
        assert first_hour["volume"] == first_60_minutes["volume"].sum()

        # Check that both minute and hourly data are cached
        minute_cache = Path(temp_cache_dir) / "binanceus" / "BTC-USD_1m.parquet"
        hourly_cache = Path(temp_cache_dir) / "binanceus" / "BTC-USD_1h.parquet"

        assert minute_cache.exists()
        assert hourly_cache.exists()

        # Verify cache performance - both requests were cache misses (hourly data wasn't cached)
        # but minute data was resampled from cache, so we should have fast cache loads
        stats = feed.get_cache_stats()
        assert (
            stats["misses"] == 2
        )  # Both minute and hourly requests were misses initially
        if stats["load_times"]:  # If any cache loading happened
            assert stats["avg_load_time_ms"] < 200  # Performance requirement

    @patch("ccxt.binanceus")
    def test_multiple_symbols_and_timeframes(self, mock_exchange_class, temp_cache_dir):
        """Test handling multiple symbols and timeframes simultaneously."""
        # Create different data for different symbols
        symbols_data = {
            "BTC/USD": self._create_sample_data("2024-01-01 10:00:00", 60, 42000),
            "ETH/USD": self._create_sample_data("2024-01-01 10:00:00", 60, 2500),
        }

        def mock_fetch_ohlcv(symbol, timeframe, since, limit):
            return symbols_data.get(symbol, [])

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = mock_fetch_ohlcv
        mock_exchange.rateLimit = 1000
        mock_exchange_class.return_value = mock_exchange

        feed = DataFeed(cache_dir=temp_cache_dir)

        # Fetch data for multiple symbols and timeframes
        btc_1m = feed.get("BTC/USD", "1m", "2024-01-01 10:00", "2024-01-01 11:00")
        btc_5m = feed.get("BTC/USD", "5m", "2024-01-01 10:00", "2024-01-01 11:00")
        eth_1m = feed.get("ETH/USD", "1m", "2024-01-01 10:00", "2024-01-01 11:00")

        assert len(btc_1m) == 60
        assert len(btc_5m) == 12  # 60 minutes / 5
        assert len(eth_1m) == 60

        # Verify different price levels
        assert btc_1m["close"].mean() > 40000  # BTC around 42k
        assert eth_1m["close"].mean() > 2000  # ETH around 2.5k

        # Check cache files for all combinations
        cache_files = ["BTC-USD_1m.parquet", "BTC-USD_5m.parquet", "ETH-USD_1m.parquet"]

        for cache_file in cache_files:
            cache_path = Path(temp_cache_dir) / "binanceus" / cache_file
            assert cache_path.exists()

    def _create_sample_data(self, start_time: str, periods: int, base_price: float):
        """Helper to create sample OHLCV data."""
        dates = pd.date_range(start_time, periods=periods, freq="1min", tz="UTC")

        data = []
        for i, date in enumerate(dates):
            price = base_price + i * 0.1  # Small trend
            data.append(
                {
                    "timestamp": int(date.timestamp() * 1000),
                    "open": price,
                    "high": price + abs(np.random.normal(0, 10)),
                    "low": price - abs(np.random.normal(0, 10)),
                    "close": price + np.random.normal(0, 5),
                    "volume": np.random.uniform(50, 200),
                }
            )

        return data


class TestDataFeedPerformance:
    """Performance-focused tests for the DataFeed."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for performance testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("ccxt.binanceus")
    def test_large_dataset_performance(self, mock_exchange_class, temp_cache_dir):
        """Test performance with large datasets (simulating days of minute data)."""
        # Create 7 days of minute data (10,080 data points)
        dates = pd.date_range("2024-01-01", periods=10080, freq="1min", tz="UTC")

        ohlcv_data = []
        base_price = 42000.0

        # Generate realistic price movement
        np.random.seed(42)
        price_changes = np.random.normal(0, 10, len(dates))
        cumulative_changes = np.cumsum(price_changes)

        for i, date in enumerate(dates):
            price = base_price + cumulative_changes[i]

            # Generate valid OHLC data
            high_offset = abs(np.random.normal(0, 20))
            low_offset = abs(np.random.normal(0, 20))
            close_offset = np.random.normal(0, 10)

            open_price = price
            high_price = price + high_offset
            low_price = price - low_offset
            close_price = price + close_offset

            # Ensure OHLC constraints: low <= open,close <= high
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            # CCXT format: [timestamp, open, high, low, close, volume]
            ohlcv_data.append(
                [
                    int(date.timestamp() * 1000),  # timestamp
                    open_price,  # open
                    high_price,  # high
                    low_price,  # low
                    close_price,  # close
                    np.random.uniform(100, 1000),  # volume
                ]
            )

        # Setup mock to return data in chunks (simulating API pagination)
        def mock_fetch_ohlcv(symbol, timeframe, since, limit):
            # Convert since from ms to index
            start_idx = max(0, int((since / 1000 - dates[0].timestamp()) / 60))
            end_idx = min(len(ohlcv_data), start_idx + limit)
            return ohlcv_data[start_idx:end_idx]

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = mock_fetch_ohlcv
        mock_exchange.rateLimit = 1000
        mock_exchange_class.return_value = mock_exchange

        feed = DataFeed(cache_dir=temp_cache_dir)

        # Test initial fetch and cache save performance
        start_time = time.time()
        df = feed.get("BTC/USD", "1m", "2024-01-01", "2024-01-08")
        fetch_time = (time.time() - start_time) * 1000

        assert len(df) == 10080
        print(f"Large dataset fetch took: {fetch_time:.2f}ms")

        # Test cache load performance (critical requirement)
        start_time = time.time()
        df_cached = feed.get("BTC/USD", "1m", "2024-01-01", "2024-01-08")
        cache_load_time = (time.time() - start_time) * 1000

        assert (
            cache_load_time < 200
        ), f"Cache load took {cache_load_time:.2f}ms, exceeds 200ms requirement"
        print(f"Large dataset cache load took: {cache_load_time:.2f}ms")

        # Verify data integrity
        pd.testing.assert_frame_equal(df, df_cached)

    @patch("ccxt.binanceus")
    def test_concurrent_access_performance(self, mock_exchange_class, temp_cache_dir):
        """Test performance under concurrent access (thread safety)."""
        import concurrent.futures

        # Create sample data
        dates = pd.date_range("2024-01-01", periods=1440, freq="1min", tz="UTC")
        ohlcv_data = []

        for i, date in enumerate(dates):
            # CCXT format: [timestamp, open, high, low, close, volume]
            ohlcv_data.append(
                [
                    int(date.timestamp() * 1000),  # timestamp
                    42000 + i,  # open
                    42000 + i + 50,  # high
                    42000 + i - 50,  # low
                    42000 + i + 25,  # close
                    100,  # volume
                ]
            )

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = ohlcv_data
        mock_exchange.rateLimit = 1000
        mock_exchange_class.return_value = mock_exchange

        # For concurrent testing, disable caching to avoid file corruption
        # (In real applications, cache writes should be serialized)
        feed = DataFeed(cache_enabled=False)

        def fetch_data(thread_id):
            """Function to run in parallel threads."""
            start_time = time.time()
            df = feed.get("BTC/USD", "1m", "2024-01-01", "2024-01-02")
            load_time = (time.time() - start_time) * 1000
            return thread_id, load_time, len(df)

        # Test with multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_data, i) for i in range(10)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # All requests should complete successfully and quickly
        for thread_id, load_time, data_len in results:
            assert (
                data_len == 1440
            ), f"Thread {thread_id} got {data_len} rows, expected 1440"
            # Note: Timing can vary significantly in test environments, so we use a generous limit
            assert (
                load_time < 5000
            ), f"Thread {thread_id} took {load_time:.2f}ms, too slow"
            print(f"Thread {thread_id} cache load: {load_time:.2f}ms")

        # Check that all threads completed successfully
        # Since caching is disabled, we focus on thread safety and performance
        stats = feed.get_cache_stats()
        assert stats["hits"] == 0  # No cache hits since caching is disabled
        assert stats["misses"] == 10  # All 10 threads missed cache (as expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
