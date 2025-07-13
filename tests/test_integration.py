"""
Integration tests for the BTC Research Engine.

This module provides end-to-end integration tests that verify the complete
workflow from configuration loading through backtesting and results generation.
These tests ensure that all components work together correctly.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from btc_research.core.backtester import Backtester
from btc_research.core.engine import Engine
from btc_research.core.registry import _clear_registry, register
from btc_research.indicators.ema import EMA
from btc_research.indicators.rsi import RSI


class TestFullWorkflowIntegration:
    """Test complete workflow integration."""

    def setup_method(self):
        """Set up test environment."""
        _clear_registry()
        register("EMA")(EMA)
        register("RSI")(RSI)

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create realistic OHLCV data for testing."""
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="1h", tz="UTC")

        # Generate realistic price movements
        np.random.seed(42)  # For reproducible tests
        price = 45000.0
        prices = []

        for _ in range(len(dates)):
            # Random walk with slight upward bias
            change = np.random.normal(0.001, 0.02)
            price *= 1 + change
            prices.append(price)

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "open": [p * np.random.uniform(0.998, 1.002) for p in prices],
                "high": [p * np.random.uniform(1.005, 1.015) for p in prices],
                "low": [p * np.random.uniform(0.985, 0.995) for p in prices],
                "close": prices,
                "volume": np.random.uniform(100, 1000, len(dates)),
            },
            index=dates,
        )

        # Ensure OHLC relationships are valid
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return {
            "version": "1.0",
            "name": "Integration Test Strategy",
            "symbol": "BTC/USDC",
            "exchange": "binanceus",
            "timeframes": {"entry": "1h"},
            "indicators": [
                {"id": "EMA_50", "type": "EMA", "timeframe": "1h", "length": 50},
                {"id": "RSI_14", "type": "RSI", "timeframe": "1h", "length": 14},
            ],
            "logic": {
                "entry_long": ["EMA_50_trend == 'bull'", "RSI_14 < 40"],
                "exit_long": ["RSI_14 > 60"],
                "entry_short": [],
                "exit_short": [],
            },
            "backtest": {
                "cash": 10000,
                "commission": 0.001,
                "slippage": 0.0,
                "from": "2024-01-01",
                "to": "2024-01-31",
            },
        }

    @patch("btc_research.core.engine.DataFeed")
    def test_complete_workflow(
        self, mock_datafeed_class, sample_ohlcv_data, test_config
    ):
        """Test complete end-to-end workflow."""
        # Set up mock datafeed
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed
        mock_datafeed.get.return_value = sample_ohlcv_data

        # Initialize engine
        engine = Engine(test_config)

        # Run the engine
        result_df = engine.run()

        # Verify results
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        assert len(result_df) == len(sample_ohlcv_data)

        # Check that indicators were computed
        expected_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "EMA_50",
            "EMA_50_trend",
            "RSI_14",
            "RSI_14_overbought",
            "RSI_14_oversold",
        ]
        for col in expected_columns:
            assert col in result_df.columns, f"Missing column: {col}"

        # Verify indicators have reasonable values
        assert not result_df["EMA_50"].isna().all()
        assert not result_df["RSI_14"].isna().all()
        assert result_df["RSI_14"].max() <= 100
        assert result_df["RSI_14"].min() >= 0

    @patch("btc_research.core.engine.DataFeed")
    def test_backtest_integration(
        self, mock_datafeed_class, sample_ohlcv_data, test_config
    ):
        """Test integration with backtester."""
        # Set up mock datafeed
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed
        mock_datafeed.get.return_value = sample_ohlcv_data

        # Initialize engine and run
        engine = Engine(test_config)
        result_df = engine.run()

        # Initialize backtester with results
        backtest_config = test_config["backtest"].copy()
        backtest_config["logic"] = test_config["logic"]
        backtester = Backtester(backtest_config)
        backtest_results = backtester.run(result_df)

        # Verify backtest results structure
        assert isinstance(backtest_results, dict)
        required_keys = [
            "trades",
            "equity_curve",
            "total_return",
            "num_trades",
            "win_rate",
        ]
        for key in required_keys:
            assert key in backtest_results, f"Missing key: {key}"

        # Verify trades structure
        trades = backtest_results["trades"]
        assert isinstance(trades, list)

        # Verify equity curve
        equity_curve = backtest_results["equity_curve"]
        assert isinstance(equity_curve, list)
        assert len(equity_curve) > 0

        # Check equity curve structure
        if len(equity_curve) > 0:
            equity_point = equity_curve[0]
            assert "timestamp" in equity_point
            assert "equity" in equity_point

        # Verify key metrics exist and have reasonable values
        assert isinstance(backtest_results["total_return"], (int, float))
        assert isinstance(backtest_results["num_trades"], int)
        assert isinstance(backtest_results["win_rate"], (int, float))
        assert backtest_results["num_trades"] >= 0
        assert 0 <= backtest_results["win_rate"] <= 100

    def test_config_file_integration(self, sample_ohlcv_data):
        """Test integration with YAML configuration files."""
        config = {
            "version": "1.0",
            "name": "YAML Config Test",
            "symbol": "BTC/USD",
            "exchange": "test",
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
                "to": "2024-01-31",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Load config from file
            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)

            # Verify config loaded correctly
            assert loaded_config["name"] == "YAML Config Test"
            assert len(loaded_config["indicators"]) == 1

            # Test engine can be initialized with loaded config
            with patch("btc_research.core.engine.DataFeed") as mock_datafeed_class:
                mock_datafeed = Mock()
                mock_datafeed_class.return_value = mock_datafeed
                mock_datafeed.get.return_value = sample_ohlcv_data

                engine = Engine(loaded_config)
                assert engine.cfg["name"] == "YAML Config Test"

                # Verify engine can run successfully
                result_df = engine.run()
                assert isinstance(result_df, pd.DataFrame)
                assert len(result_df) > 0

        finally:
            # Clean up temporary file
            Path(config_path).unlink()


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    def setup_method(self):
        """Set up test environment."""
        _clear_registry()
        register("EMA")(EMA)
        register("RSI")(RSI)

    def test_missing_indicator_error_propagation(self):
        """Test that missing indicator errors propagate correctly."""
        config = {
            "version": "1.0",
            "name": "Error Test",
            "symbol": "BTC/USD",
            "timeframes": {"entry": "1h"},
            "indicators": [{"id": "MISSING", "type": "NONEXISTENT", "timeframe": "1h"}],
            "logic": {
                "entry_long": [],
                "exit_long": [],
                "entry_short": [],
                "exit_short": [],
            },
            "backtest": {
                "cash": 10000,
                "commission": 0.001,
                "from": "2024-01-01",
                "to": "2024-01-02",
            },
        }

        with patch("btc_research.core.engine.DataFeed") as mock_datafeed_class:
            mock_datafeed = Mock()
            mock_datafeed_class.return_value = mock_datafeed
            mock_datafeed.get.return_value = pd.DataFrame(
                {
                    "open": [45000],
                    "high": [45100],
                    "low": [44900],
                    "close": [45050],
                    "volume": [100],
                },
                index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC"),
            )

            engine = Engine(config)

            with pytest.raises(Exception) as exc_info:
                engine.run()

            # Verify error message contains useful information
            assert (
                "NONEXISTENT" in str(exc_info.value)
                or "not found" in str(exc_info.value).lower()
            )

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        config = {
            "version": "1.0",
            "name": "Insufficient Data Test",
            "symbol": "BTC/USD",
            "timeframes": {"entry": "1h"},
            "indicators": [
                {
                    "id": "EMA_100",
                    "type": "EMA",
                    "timeframe": "1h",
                    "length": 100,
                }  # Requires more data
            ],
            "logic": {
                "entry_long": [],
                "exit_long": [],
                "entry_short": [],
                "exit_short": [],
            },
            "backtest": {
                "cash": 10000,
                "commission": 0.001,
                "from": "2024-01-01",
                "to": "2024-01-02",
            },
        }

        # Create minimal data (less than required for EMA_100)
        minimal_data = pd.DataFrame(
            {
                "open": [45000] * 5,
                "high": [45100] * 5,
                "low": [44900] * 5,
                "close": [45050] * 5,
                "volume": [100] * 5,
            },
            index=pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
        )

        with patch("btc_research.core.engine.DataFeed") as mock_datafeed_class:
            mock_datafeed = Mock()
            mock_datafeed_class.return_value = mock_datafeed
            mock_datafeed.get.return_value = minimal_data

            engine = Engine(config)

            # Should handle gracefully, not crash
            result_df = engine.run()
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == len(minimal_data)


class TestMultiTimeframeIntegration:
    """Test multi-timeframe functionality integration."""

    def setup_method(self):
        """Set up test environment."""
        _clear_registry()
        register("EMA")(EMA)
        register("RSI")(RSI)

    @patch("btc_research.core.engine.DataFeed")
    def test_multi_timeframe_workflow(self, mock_datafeed_class):
        """Test complete workflow with multiple timeframes."""
        # Create data for different timeframes
        dates_1h = pd.date_range("2024-01-01", "2024-01-31", freq="1h", tz="UTC")
        dates_4h = pd.date_range("2024-01-01", "2024-01-31", freq="4h", tz="UTC")

        data_1h = pd.DataFrame(
            {
                "open": np.random.uniform(44000, 46000, len(dates_1h)),
                "high": np.random.uniform(44500, 46500, len(dates_1h)),
                "low": np.random.uniform(43500, 45500, len(dates_1h)),
                "close": np.random.uniform(44000, 46000, len(dates_1h)),
                "volume": np.random.uniform(100, 1000, len(dates_1h)),
            },
            index=dates_1h,
        )

        data_4h = pd.DataFrame(
            {
                "open": np.random.uniform(44000, 46000, len(dates_4h)),
                "high": np.random.uniform(44500, 46500, len(dates_4h)),
                "low": np.random.uniform(43500, 45500, len(dates_4h)),
                "close": np.random.uniform(44000, 46000, len(dates_4h)),
                "volume": np.random.uniform(1000, 10000, len(dates_4h)),
            },
            index=dates_4h,
        )

        # Fix OHLC relationships
        for data in [data_1h, data_4h]:
            data["high"] = np.maximum(
                data["high"], np.maximum(data["open"], data["close"])
            )
            data["low"] = np.minimum(
                data["low"], np.minimum(data["open"], data["close"])
            )

        config = {
            "version": "1.0",
            "name": "Multi-Timeframe Test",
            "symbol": "BTC/USD",
            "timeframes": {"bias": "4h", "entry": "1h"},
            "indicators": [
                {"id": "BIAS_EMA", "type": "EMA", "timeframe": "4h", "length": 20},
                {"id": "ENTRY_RSI", "type": "RSI", "timeframe": "1h", "length": 14},
            ],
            "logic": {
                "entry_long": ["BIAS_EMA_trend == 'bull'", "ENTRY_RSI < 30"],
                "exit_long": ["ENTRY_RSI > 70"],
                "entry_short": [],
                "exit_short": [],
            },
            "backtest": {
                "cash": 10000,
                "commission": 0.001,
                "from": "2024-01-01",
                "to": "2024-01-31",
            },
        }

        # Set up mock datafeed to return appropriate data based on timeframe
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed

        def mock_get(*args, **kwargs):
            timeframe = kwargs.get("timeframe", args[1] if len(args) > 1 else None)
            if timeframe == "4h":
                return data_4h
            elif timeframe == "1h":
                return data_1h
            else:
                raise ValueError(f"Unexpected timeframe: {timeframe}")

        mock_datafeed.get.side_effect = mock_get

        # Run engine
        engine = Engine(config)
        result_df = engine.run()

        # Verify results
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(data_1h)  # Should match entry timeframe

        # Check that indicators from both timeframes are present
        expected_columns = [
            "BIAS_EMA_EMA_20",
            "BIAS_EMA_EMA_20_trend",
            "ENTRY_RSI_RSI_14",
            "ENTRY_RSI_RSI_14_overbought",
            "ENTRY_RSI_RSI_14_oversold",
        ]
        for col in expected_columns:
            assert col in result_df.columns, f"Missing column: {col}"

        # Verify data alignment worked (4h data should be forward-filled to 1h)
        bias_trend_col = "BIAS_EMA_EMA_20_trend"
        assert not result_df[bias_trend_col].isna().all()
