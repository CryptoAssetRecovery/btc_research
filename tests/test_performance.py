"""
Performance benchmark tests for the BTC Research Engine.

This module provides performance tests to ensure the engine can handle
realistic data volumes and processing requirements efficiently.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from btc_research.core.backtester import Backtester
from btc_research.core.engine import Engine
from btc_research.core.registry import _clear_registry, register
from btc_research.indicators.ema import EMA
from btc_research.indicators.rsi import RSI


class TestEnginePerformance:
    """Test engine performance with various data sizes."""

    def setup_method(self):
        """Set up test environment."""
        _clear_registry()
        register("EMA")(EMA)
        register("RSI")(RSI)

    @pytest.fixture(params=[1000, 5000, 10000])
    def large_dataset(self, request):
        """Create large datasets for performance testing."""
        size = request.param
        dates = pd.date_range("2024-01-01", periods=size, freq="1h", tz="UTC")

        np.random.seed(42)  # For reproducible tests

        # Generate realistic random walk price data
        price = 45000.0
        prices = []

        for _ in range(size):
            change = np.random.normal(0.0001, 0.015)  # Small drift with volatility
            price *= 1 + change
            prices.append(price)

        data = pd.DataFrame(
            {
                "open": [p * np.random.uniform(0.999, 1.001) for p in prices],
                "high": [p * np.random.uniform(1.002, 1.008) for p in prices],
                "low": [p * np.random.uniform(0.992, 0.998) for p in prices],
                "close": prices,
                "volume": np.random.uniform(100, 1000, size),
            },
            index=dates,
        )

        # Ensure OHLC relationships
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data, size

    @patch("btc_research.core.engine.DataFeed")
    def test_engine_processing_speed(self, mock_datafeed_class, large_dataset):
        """Test engine processing speed with large datasets."""
        data, size = large_dataset

        # Set up mock datafeed
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed
        mock_datafeed.get.return_value = data

        config = {
            "version": "1.0",
            "name": "Performance Test",
            "symbol": "BTC/USD",
            "timeframes": {"entry": "1h"},
            "indicators": [
                {"id": "EMA_50", "type": "EMA", "timeframe": "1h", "length": 50},
                {"id": "EMA_200", "type": "EMA", "timeframe": "1h", "length": 200},
                {"id": "RSI_14", "type": "RSI", "timeframe": "1h", "length": 14},
            ],
            "logic": {
                "entry_long": ["EMA_50_trend == 'bull'", "RSI_14 < 30"],
                "exit_long": ["RSI_14 > 70"],
                "entry_short": [],
                "exit_short": [],
            },
            "backtest": {
                "cash": 10000,
                "commission": 0.001,
                "from": "2024-01-01",
                "to": "2024-12-31",
            },
        }

        # Measure processing time
        start_time = time.time()
        engine = Engine(config)
        result_df = engine.run()
        end_time = time.time()

        processing_time = end_time - start_time

        # Verify results
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == size

        # Performance benchmarks (these are reasonable for the data sizes)
        if size <= 1000:
            assert (
                processing_time < 5.0
            ), f"Processing {size} rows took {processing_time:.2f}s (expected < 5s)"
        elif size <= 5000:
            assert (
                processing_time < 15.0
            ), f"Processing {size} rows took {processing_time:.2f}s (expected < 15s)"
        else:  # 10000 rows
            assert (
                processing_time < 30.0
            ), f"Processing {size} rows took {processing_time:.2f}s (expected < 30s)"

        print(
            f"Processed {size} rows in {processing_time:.2f} seconds ({size/processing_time:.0f} rows/sec)"
        )

    @patch("btc_research.core.engine.DataFeed")
    def test_memory_efficiency(self, mock_datafeed_class, large_dataset):
        """Test that memory usage remains reasonable with large datasets."""
        data, size = large_dataset

        # Set up mock datafeed
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed
        mock_datafeed.get.return_value = data

        config = {
            "version": "1.0",
            "name": "Memory Test",
            "symbol": "BTC/USD",
            "timeframes": {"entry": "1h"},
            "indicators": [
                {"id": "EMA_20", "type": "EMA", "timeframe": "1h", "length": 20}
            ],
            "logic": {
                "entry_long": ["EMA_20_trend == 'bull'"],
                "exit_long": ["EMA_20_trend == 'bear'"],
                "entry_short": [],
                "exit_short": [],
            },
            "backtest": {
                "cash": 10000,
                "commission": 0.001,
                "from": "2024-01-01",
                "to": "2024-12-31",
            },
        }

        # Check that we can process the data without memory errors
        try:
            engine = Engine(config)
            result_df = engine.run()

            # Verify we didn't create excessive copies
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == size

            # Check memory usage is reasonable (columns should be proportional to input + indicators)
            expected_min_cols = 5  # OHLCV
            expected_max_cols = 15  # OHLCV + indicators + some room for growth
            assert expected_min_cols <= len(result_df.columns) <= expected_max_cols

        except MemoryError:
            pytest.fail(f"Memory error when processing {size} rows")


class TestBacktesterPerformance:
    """Test backtester performance."""

    @pytest.fixture
    def backtest_data(self):
        """Create realistic data for backtest performance testing."""
        size = 8760  # One year of hourly data
        dates = pd.date_range("2024-01-01", periods=size, freq="1h", tz="UTC")

        np.random.seed(42)

        # Create data with some realistic trading signals
        data = pd.DataFrame(
            {
                "open": np.random.uniform(44000, 46000, size),
                "high": np.random.uniform(44500, 46500, size),
                "low": np.random.uniform(43500, 45500, size),
                "close": np.random.uniform(44000, 46000, size),
                "volume": np.random.uniform(100, 1000, size),
                # Add some indicator columns for signal generation
                "EMA_trend": np.random.choice(["bull", "bear"], size),
                "RSI_value": np.random.uniform(0, 100, size),
            },
            index=dates,
        )

        # Fix OHLC relationships
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data

    def test_backtest_execution_speed(self, backtest_data):
        """Test backtesting speed with realistic data volume."""
        config = {
            "cash": 10000,
            "commission": 0.001,
            "slippage": 0.0,
            "logic": {
                "entry_long": ["EMA_trend == 'bull'", "RSI_value < 30"],
                "exit_long": ["RSI_value > 70"],
                "entry_short": [],
                "exit_short": [],
            },
        }

        # Measure backtest execution time
        start_time = time.time()
        backtester = Backtester(config)
        results = backtester.run(backtest_data)
        end_time = time.time()

        execution_time = end_time - start_time

        # Verify results structure
        assert isinstance(results, dict)
        assert "trades" in results
        assert "equity_curve" in results
        assert "total_return" in results

        # Performance benchmark (8760 rows should process quickly)
        assert (
            execution_time < 10.0
        ), f"Backtesting 8760 rows took {execution_time:.2f}s (expected < 10s)"

        print(f"Backtested {len(backtest_data)} rows in {execution_time:.2f} seconds")

    def test_trade_generation_efficiency(self, backtest_data):
        """Test efficiency of trade generation logic."""
        # Create data that should generate many trades
        backtest_data["frequent_signal"] = np.random.choice(
            [True, False], len(backtest_data), p=[0.1, 0.9]
        )

        config = {
            "cash": 10000,
            "commission": 0.001,
            "slippage": 0.0,
            "logic": {
                "entry_long": ["frequent_signal == True"],
                "exit_long": ["frequent_signal == False"],
                "entry_short": [],
                "exit_short": [],
            },
        }

        start_time = time.time()
        backtester = Backtester(config)
        results = backtester.run(backtest_data)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should still be fast even with many trades
        assert (
            execution_time < 15.0
        ), f"Backtesting with frequent signals took {execution_time:.2f}s (expected < 15s)"

        # Verify trades were generated
        trades = results["trades"]
        assert isinstance(trades, list)
        # Should have some trades with frequent signals
        if len(trades) > 0:
            assert len(trades) > 5, "Expected multiple trades with frequent signals"


class TestIndicatorPerformance:
    """Test indicator computation performance."""

    @pytest.fixture(params=[1000, 5000])
    def indicator_test_data(self, request):
        """Create data for indicator performance testing."""
        size = request.param
        dates = pd.date_range("2024-01-01", periods=size, freq="1h", tz="UTC")

        # Generate price data with some trends
        np.random.seed(42)
        price = 45000.0
        prices = []

        for i in range(size):
            # Add some cyclical patterns to make indicators more interesting
            trend = 0.0001 * np.sin(i * 0.01)
            noise = np.random.normal(0, 0.015)
            change = trend + noise
            price *= 1 + change
            prices.append(price)

        data = pd.DataFrame(
            {
                "open": [p * 0.9995 for p in prices],
                "high": [p * 1.005 for p in prices],
                "low": [p * 0.995 for p in prices],
                "close": prices,
                "volume": np.random.uniform(100, 1000, size),
            },
            index=dates,
        )

        return data, size

    def test_ema_computation_speed(self, indicator_test_data):
        """Test EMA indicator computation speed."""
        data, size = indicator_test_data

        # Test different EMA lengths
        for length in [20, 50, 200]:
            ema = EMA(length=length)

            start_time = time.time()
            result = ema.compute(data)
            end_time = time.time()

            computation_time = end_time - start_time

            # Verify result
            assert isinstance(result, pd.DataFrame)
            assert len(result) == size
            assert f"EMA_{length}" in result.columns
            assert f"EMA_{length}_trend" in result.columns

            # Performance check (should be fast for these data sizes)
            max_time = 2.0 if size <= 1000 else 5.0
            assert (
                computation_time < max_time
            ), f"EMA({length}) on {size} rows took {computation_time:.2f}s (expected < {max_time}s)"

    def test_rsi_computation_speed(self, indicator_test_data):
        """Test RSI indicator computation speed."""
        data, size = indicator_test_data

        # Test different RSI lengths
        for length in [14, 21, 30]:
            rsi = RSI(length=length)

            start_time = time.time()
            result = rsi.compute(data)
            end_time = time.time()

            computation_time = end_time - start_time

            # Verify result
            assert isinstance(result, pd.DataFrame)
            assert len(result) == size
            assert f"RSI_{length}" in result.columns
            assert f"RSI_{length}_overbought" in result.columns
            assert f"RSI_{length}_oversold" in result.columns

            # Performance check
            max_time = 2.0 if size <= 1000 else 5.0
            assert (
                computation_time < max_time
            ), f"RSI({length}) on {size} rows took {computation_time:.2f}s (expected < {max_time}s)"

    def test_multiple_indicators_performance(self, indicator_test_data):
        """Test performance when computing multiple indicators together."""
        data, size = indicator_test_data

        indicators = [EMA(length=20), EMA(length=50), RSI(length=14), RSI(length=21)]

        start_time = time.time()

        results = []
        for indicator in indicators:
            result = indicator.compute(data)
            results.append(result)

        # Combine results (simulating engine behavior)
        combined = data.copy()
        for result in results:
            combined = combined.join(result)

        end_time = time.time()
        computation_time = end_time - start_time

        # Verify combined result
        assert isinstance(combined, pd.DataFrame)
        assert len(combined) == size

        # Check that all indicator columns are present
        expected_cols = [
            "EMA_20",
            "EMA_20_trend",
            "EMA_50",
            "EMA_50_trend",
            "RSI_14",
            "RSI_14_overbought",
            "RSI_14_oversold",
            "RSI_21",
            "RSI_21_overbought",
            "RSI_21_oversold",
        ]
        for col in expected_cols:
            assert col in combined.columns, f"Missing column: {col}"

        # Performance check for multiple indicators
        max_time = 5.0 if size <= 1000 else 15.0
        assert (
            computation_time < max_time
        ), f"Multiple indicators on {size} rows took {computation_time:.2f}s (expected < {max_time}s)"


class TestScalabilityLimits:
    """Test system behavior at scalability limits."""

    def setup_method(self):
        """Set up test environment."""
        _clear_registry()
        register("EMA")(EMA)
        register("RSI")(RSI)

    @pytest.mark.slow
    def test_very_large_dataset_handling(self):
        """Test handling of very large datasets (marked as slow test)."""
        # Create a very large dataset (1 year of minute data)
        size = 525600  # 365 * 24 * 60

        # Only run this test if explicitly requested
        if not hasattr(pytest, "config") or not getattr(
            pytest.config.option, "runslow", False
        ):
            pytest.skip("Slow test skipped (use --runslow to run)")

        dates = pd.date_range("2024-01-01", periods=size, freq="1min", tz="UTC")

        # Generate data in chunks to avoid memory issues
        chunk_size = 10000
        chunks = []

        for i in range(0, size, chunk_size):
            chunk_dates = dates[i : i + chunk_size]
            chunk_data = pd.DataFrame(
                {
                    "open": np.random.uniform(44000, 46000, len(chunk_dates)),
                    "high": np.random.uniform(44500, 46500, len(chunk_dates)),
                    "low": np.random.uniform(43500, 45500, len(chunk_dates)),
                    "close": np.random.uniform(44000, 46000, len(chunk_dates)),
                    "volume": np.random.uniform(100, 1000, len(chunk_dates)),
                },
                index=chunk_dates,
            )
            chunks.append(chunk_data)

        data = pd.concat(chunks)

        # Test that indicators can handle very large datasets
        ema = EMA(length=50)

        start_time = time.time()
        result = ema.compute(data)
        end_time = time.time()

        computation_time = end_time - start_time

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == size

        # Should complete within reasonable time (5 minutes for 525k rows)
        assert (
            computation_time < 300.0
        ), f"Very large dataset processing took {computation_time:.2f}s (expected < 300s)"

        print(
            f"Processed {size} rows in {computation_time:.2f} seconds ({size/computation_time:.0f} rows/sec)"
        )
