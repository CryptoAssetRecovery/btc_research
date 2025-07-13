"""
Unit tests for the Engine class.

This module provides comprehensive tests for the confluence engine including:
- Configuration validation
- Multi-timeframe data loading
- Indicator instantiation and computation
- Data alignment and column naming
- Error handling scenarios
- Integration with the demo.yaml configuration
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from btc_research.core.base_indicator import BaseIndicator
from btc_research.core.engine import Engine, EngineError
from btc_research.core.registry import _clear_registry, register


class TestEngine:
    """Test cases for the Engine class."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear registry to ensure clean state
        _clear_registry()

        # Register mock indicators for testing
        @register("TestRSI")
        class TestRSI(BaseIndicator):
            @classmethod
            def params(cls):
                return {"length": 14}

            def __init__(self, length=14):
                self.length = length

            def compute(self, df):
                result = pd.DataFrame(index=df.index)
                # Simple mock RSI calculation
                result[f"RSI_{self.length}"] = 50.0  # Constant for testing
                result[f"RSI_{self.length}_overbought"] = False
                result[f"RSI_{self.length}_oversold"] = False
                return result

        @register("TestEMA")
        class TestEMA(BaseIndicator):
            @classmethod
            def params(cls):
                return {"length": 200}

            def __init__(self, length=200):
                self.length = length

            def compute(self, df):
                result = pd.DataFrame(index=df.index)
                # Simple mock EMA calculation - use min_periods=1 for testing
                result[f"EMA_{self.length}"] = (
                    df["close"].rolling(2, min_periods=1).mean()
                )
                result[f"EMA_{self.length}_trend"] = "bull"
                return result

        self.test_config = {
            "symbol": "BTC/USD",
            "exchange": "binanceus",
            "timeframes": {"bias": "1h", "entry": "5m"},
            "indicators": [
                {"id": "EMA_200", "type": "TestEMA", "timeframe": "1h", "length": 200},
                {"id": "RSI_14", "type": "TestRSI", "timeframe": "5m", "length": 14},
            ],
            "backtest": {"from": "2024-01-01", "to": "2024-01-02"},
        }

        # Create mock OHLCV data
        self.mock_5m_data = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            },
            index=pd.date_range(
                "2024-01-01 00:00:00", periods=5, freq="5min", tz="UTC"
            ),
        )

        self.mock_1h_data = pd.DataFrame(
            {
                "open": [100.0, 102.0],
                "high": [102.0, 105.0],
                "low": [99.0, 101.0],
                "close": [101.0, 104.0],
                "volume": [5000.0, 6000.0],
            },
            index=pd.date_range("2024-01-01 00:00:00", periods=2, freq="1h", tz="UTC"),
        )

    def test_init_valid_config(self):
        """Test Engine initialization with valid configuration."""
        engine = Engine(self.test_config)
        assert engine.cfg == self.test_config
        assert len(engine.indicator_objects) == 0
        assert engine.datafeed is not None

    def test_init_missing_required_fields(self):
        """Test Engine initialization with missing required fields."""
        # Test missing symbol
        config = self.test_config.copy()
        del config["symbol"]
        with pytest.raises(
            EngineError, match="Missing required configuration field: symbol"
        ):
            Engine(config)

        # Test missing timeframes
        config = self.test_config.copy()
        del config["timeframes"]
        with pytest.raises(
            EngineError, match="Missing required configuration field: timeframes"
        ):
            Engine(config)

        # Test missing entry timeframe
        config = self.test_config.copy()
        del config["timeframes"]["entry"]
        with pytest.raises(
            EngineError, match="Configuration must specify 'entry' timeframe"
        ):
            Engine(config)

    def test_init_invalid_indicators(self):
        """Test Engine initialization with invalid indicators configuration."""
        # Test indicators not a list
        config = self.test_config.copy()
        config["indicators"] = "not a list"
        with pytest.raises(EngineError, match="Indicators must be specified as a list"):
            Engine(config)

        # Test indicator missing required fields
        config = self.test_config.copy()
        config["indicators"] = [{"id": "test"}]  # Missing type and timeframe
        with pytest.raises(
            EngineError, match="Indicator 0 missing required field: type"
        ):
            Engine(config)

    @patch("btc_research.core.engine.DataFeed")
    def test_load_timeframes(self, mock_datafeed_class):
        """Test _load_timeframes method."""
        # Set up mock
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed
        mock_datafeed.get.side_effect = [self.mock_1h_data, self.mock_5m_data]

        engine = Engine(self.test_config)
        data_by_tf = engine._load_timeframes()

        # Check that correct timeframes were loaded
        assert "1h" in data_by_tf
        assert "5m" in data_by_tf

        # Check that DataFeed.get was called correctly
        assert mock_datafeed.get.call_count == 2

        # Verify call arguments
        calls = mock_datafeed.get.call_args_list
        call_timeframes = [call[1]["timeframe"] for call in calls]
        assert "1h" in call_timeframes
        assert "5m" in call_timeframes

    @patch("btc_research.core.engine.DataFeed")
    def test_instantiate_indicators(self, mock_datafeed_class):
        """Test _instantiate_indicators method."""
        # Set up mock
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed

        engine = Engine(self.test_config)

        # Create test data with indicators
        data_by_tf = {"1h": self.mock_1h_data.copy(), "5m": self.mock_5m_data.copy()}

        engine._instantiate_indicators(data_by_tf)

        # Check that indicators were instantiated
        assert len(engine.indicator_objects) == 2

        # Check that indicator columns were added to the correct timeframes
        # The corrected naming logic: columns should not have duplicate ID prefixes
        # EMA_200 (column) == EMA_200 (ID) -> stays "EMA_200"
        # EMA_200_trend (column) != EMA_200 (ID) -> becomes "EMA_200_trend" (fixed behavior)
        assert "EMA_200" in data_by_tf["1h"].columns
        assert "EMA_200_trend" in data_by_tf["1h"].columns
        assert "RSI_14" in data_by_tf["5m"].columns
        assert "RSI_14_overbought" in data_by_tf["5m"].columns

    @patch("btc_research.core.engine.DataFeed")
    def test_run_complete_workflow(self, mock_datafeed_class):
        """Test complete engine workflow with run() method."""
        # Set up mock with timeframe-based return values
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed

        def mock_get(*args, **kwargs):
            timeframe = kwargs.get("timeframe", args[1] if len(args) > 1 else None)
            if timeframe == "1h":
                return self.mock_1h_data
            elif timeframe == "5m":
                return self.mock_5m_data
            else:
                raise ValueError(f"Unexpected timeframe: {timeframe}")

        mock_datafeed.get.side_effect = mock_get

        engine = Engine(self.test_config)
        result_df = engine.run()

        # Check basic structure
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(self.mock_5m_data)  # Should match entry timeframe
        assert isinstance(result_df.index, pd.DatetimeIndex)

        # Check OHLCV columns are present
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result_df.columns

        # Check indicator columns are present
        assert "RSI_14" in result_df.columns
        assert "EMA_200" in result_df.columns

    @patch("btc_research.core.engine.DataFeed")
    def test_multi_timeframe_alignment(self, mock_datafeed_class):
        """Test that multi-timeframe data is properly aligned using forward-fill."""
        # Set up mock with timeframe-based return values
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed

        def mock_get(*args, **kwargs):
            timeframe = kwargs.get("timeframe", args[1] if len(args) > 1 else None)
            if timeframe == "1h":
                return self.mock_1h_data
            elif timeframe == "5m":
                return self.mock_5m_data
            else:
                raise ValueError(f"Unexpected timeframe: {timeframe}")

        mock_datafeed.get.side_effect = mock_get

        engine = Engine(self.test_config)
        result_df = engine.run()

        # Check that higher timeframe data was forward-filled
        # The 1h EMA values should be repeated for each 5m period
        ema_col = "EMA_200"
        assert ema_col in result_df.columns

        # Values should be forward-filled (not NaN after the first hour)
        assert not result_df[ema_col].isna().all()

    @patch("btc_research.core.engine.DataFeed")
    def test_error_handling_missing_indicator(self, mock_datafeed_class):
        """Test error handling when indicator is not found in registry."""
        # Set up mock
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed
        mock_datafeed.get.return_value = self.mock_5m_data

        # Use indicator not in registry
        config = self.test_config.copy()
        config["indicators"] = [
            {"id": "MISSING", "type": "NonExistentIndicator", "timeframe": "5m"}
        ]

        engine = Engine(config)

        with pytest.raises(
            EngineError, match="Indicator NonExistentIndicator not found in registry"
        ):
            engine.run()

    @patch("btc_research.core.engine.DataFeed")
    def test_error_handling_data_load_failure(self, mock_datafeed_class):
        """Test error handling when data loading fails."""
        # Set up mock to raise exception
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed
        mock_datafeed.get.side_effect = Exception("Network error")

        engine = Engine(self.test_config)

        with pytest.raises(EngineError, match="Failed to load data for timeframe"):
            engine.run()

    def test_get_indicator_objects(self):
        """Test get_indicator_objects method."""
        engine = Engine(self.test_config)

        # Initially empty
        assert len(engine.get_indicator_objects()) == 0

        # Add mock objects
        engine.indicator_objects = ["mock1", "mock2"]
        objects = engine.get_indicator_objects()

        assert len(objects) == 2
        assert objects == ["mock1", "mock2"]

        # Should return a copy
        objects.append("mock3")
        assert len(engine.get_indicator_objects()) == 2

    def test_get_config(self):
        """Test get_config method."""
        engine = Engine(self.test_config)
        config_copy = engine.get_config()

        assert config_copy == self.test_config

        # Should return a copy
        config_copy["symbol"] = "ETH/USD"
        assert engine.cfg["symbol"] == "BTC/USD"

    def test_column_naming_with_conflicts(self):
        """Test proper column naming when there are conflicts between timeframes."""

        # Register a test indicator that creates conflicting column names
        @register("ConflictIndicator")
        class ConflictIndicator(BaseIndicator):
            @classmethod
            def params(cls):
                return {}

            def compute(self, df):
                result = pd.DataFrame(index=df.index)
                result["volume"] = df["volume"] * 2  # Creates conflict with OHLCV
                result["custom_signal"] = True
                return result

        config = {
            "symbol": "BTC/USD",
            "timeframes": {"entry": "5m"},
            "indicators": [
                {"id": "CONFLICT", "type": "ConflictIndicator", "timeframe": "1h"}
            ],
            "backtest": {"from": "2024-01-01", "to": "2024-01-02"},
        }

        with patch("btc_research.core.engine.DataFeed") as mock_datafeed_class:
            mock_datafeed = Mock()
            mock_datafeed_class.return_value = mock_datafeed
            mock_datafeed.get.side_effect = [self.mock_1h_data, self.mock_5m_data]

            engine = Engine(config)
            result_df = engine.run()

            # OHLCV volume should remain unchanged
            assert "volume" in result_df.columns
            # Custom signal should be present
            assert "CONFLICT_custom_signal" in result_df.columns


class TestEngineWithDemoConfig:
    """Test Engine with the actual demo.yaml configuration."""

    def setup_method(self):
        """Set up test environment with demo config."""
        # Clear registry and explicitly register the real indicators
        _clear_registry()

        # Force re-registration of indicators
        from btc_research.core.registry import register
        from btc_research.indicators.ema import EMA
        from btc_research.indicators.rsi import RSI

        # Explicitly register the classes
        register("RSI")(RSI)
        register("EMA")(EMA)

    def test_demo_config_loading(self):
        """Test that demo.yaml configuration loads correctly."""
        demo_config_path = (
            Path(__file__).parent.parent / "btc_research" / "config" / "demo.yaml"
        )

        with open(demo_config_path) as f:
            demo_config = yaml.safe_load(f)

        # Test that engine can be initialized with demo config
        engine = Engine(demo_config)
        assert engine.cfg["name"] == "EMA bias + RSI entry"
        assert engine.cfg["symbol"] == "BTC/USDC"
        assert len(engine.cfg["indicators"]) == 2

    @patch("btc_research.core.engine.DataFeed")
    def test_demo_config_execution(self, mock_datafeed_class):
        """Test that demo.yaml configuration can be executed successfully."""
        # Load demo config
        demo_config_path = (
            Path(__file__).parent.parent / "btc_research" / "config" / "demo.yaml"
        )
        with open(demo_config_path) as f:
            demo_config = yaml.safe_load(f)

        # Create realistic mock data
        dates_5m = pd.date_range("2024-01-01", "2024-01-02", freq="5min", tz="UTC")
        dates_1h = pd.date_range("2024-01-01", "2024-01-02", freq="1h", tz="UTC")

        mock_5m_data = pd.DataFrame(
            {
                "open": np.random.uniform(40000, 45000, len(dates_5m)),
                "high": np.random.uniform(40500, 45500, len(dates_5m)),
                "low": np.random.uniform(39500, 44500, len(dates_5m)),
                "close": np.random.uniform(40000, 45000, len(dates_5m)),
                "volume": np.random.uniform(100, 1000, len(dates_5m)),
            },
            index=dates_5m,
        )

        # Ensure OHLC relationships are valid
        mock_5m_data["high"] = np.maximum(
            mock_5m_data["high"],
            np.maximum(mock_5m_data["open"], mock_5m_data["close"]),
        )
        mock_5m_data["low"] = np.minimum(
            mock_5m_data["low"], np.minimum(mock_5m_data["open"], mock_5m_data["close"])
        )

        mock_1h_data = pd.DataFrame(
            {
                "open": np.random.uniform(40000, 45000, len(dates_1h)),
                "high": np.random.uniform(40500, 45500, len(dates_1h)),
                "low": np.random.uniform(39500, 44500, len(dates_1h)),
                "close": np.random.uniform(40000, 45000, len(dates_1h)),
                "volume": np.random.uniform(1000, 10000, len(dates_1h)),
            },
            index=dates_1h,
        )

        # Ensure OHLC relationships are valid
        mock_1h_data["high"] = np.maximum(
            mock_1h_data["high"],
            np.maximum(mock_1h_data["open"], mock_1h_data["close"]),
        )
        mock_1h_data["low"] = np.minimum(
            mock_1h_data["low"], np.minimum(mock_1h_data["open"], mock_1h_data["close"])
        )

        # Set up mock with timeframe-based return values
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed

        def mock_get(*args, **kwargs):
            timeframe = kwargs.get("timeframe", args[1] if len(args) > 1 else None)
            if timeframe == "1h":
                return mock_1h_data
            elif timeframe == "5m":
                return mock_5m_data
            else:
                raise ValueError(f"Unexpected timeframe: {timeframe}")

        mock_datafeed.get.side_effect = mock_get

        # Execute engine
        engine = Engine(demo_config)
        result_df = engine.run()

        # Verify result structure
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        assert len(result_df) == len(mock_5m_data)  # Should match entry timeframe (5m)

        # Check OHLCV columns
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result_df.columns

        # Check indicator columns from demo config
        assert "RSI_14" in result_df.columns  # RSI_14 column matches ID, so no prefix
        assert "EMA_200" in result_df.columns  # EMA_200 column matches ID, so no prefix
        assert "EMA_200_trend" in result_df.columns  # trend column gets proper naming (fixed)

        # Check that we have both indicator objects
        assert len(engine.get_indicator_objects()) == 2

        # Verify data types
        assert result_df["RSI_14"].dtype in [np.float64, np.float32]
        assert result_df["EMA_200_trend"].dtype == object  # String values

    def test_demo_config_structure(self):
        """Test that demo.yaml has the expected structure."""
        demo_config_path = (
            Path(__file__).parent.parent / "btc_research" / "config" / "demo.yaml"
        )

        with open(demo_config_path) as f:
            demo_config = yaml.safe_load(f)

        # Check required top-level fields
        assert "name" in demo_config
        assert "symbol" in demo_config
        assert "timeframes" in demo_config
        assert "indicators" in demo_config
        assert "backtest" in demo_config

        # Check timeframes structure
        assert "bias" in demo_config["timeframes"]
        assert "entry" in demo_config["timeframes"]
        assert demo_config["timeframes"]["bias"] == "1h"
        assert demo_config["timeframes"]["entry"] == "5m"

        # Check indicators structure
        indicators = demo_config["indicators"]
        assert len(indicators) == 2

        # Find EMA and RSI indicators
        ema_indicator = next(ind for ind in indicators if ind["type"] == "EMA")
        rsi_indicator = next(ind for ind in indicators if ind["type"] == "RSI")

        assert ema_indicator["id"] == "EMA_200"
        assert ema_indicator["timeframe"] == "1h"
        assert ema_indicator["length"] == 200

        assert rsi_indicator["id"] == "RSI_14"
        assert rsi_indicator["timeframe"] == "5m"
        assert rsi_indicator["length"] == 14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
