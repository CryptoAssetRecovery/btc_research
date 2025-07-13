"""
Comprehensive unit tests for EMA and RSI indicators.

This module tests the indicator implementations against the ta library baseline
to ensure accuracy and proper integration with the registry system.
"""

import numpy as np
import pandas as pd
import pytest
import ta.momentum as ta_momentum
import ta.trend as ta_trend

from btc_research.core.registry import RegistrationError, get
from btc_research.indicators.ema import EMA
from btc_research.indicators.rsi import RSI


class TestIndicatorRegistry:
    """Test indicator registration and retrieval."""

    def test_ema_registration(self):
        """Test that EMA indicator is properly registered."""
        ema_class = get("EMA")
        assert ema_class == EMA

    def test_rsi_registration(self):
        """Test that RSI indicator is properly registered."""
        rsi_class = get("RSI")
        assert rsi_class == RSI

    def test_invalid_indicator_raises_error(self):
        """Test that requesting non-existent indicator raises error."""
        with pytest.raises(RegistrationError):
            get("NONEXISTENT")


class TestEMAIndicator:
    """Test EMA indicator implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1H")
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data
        close_prices = []
        price = 50000.0
        for _ in range(100):
            change = np.random.normal(0, 0.02)  # 2% volatility
            price *= 1 + change
            close_prices.append(price)

        return pd.DataFrame(
            {
                "open": [p * 0.999 for p in close_prices],
                "high": [p * 1.01 for p in close_prices],
                "low": [p * 0.99 for p in close_prices],
                "close": close_prices,
                "volume": np.random.randint(100, 1000, 100),
            },
            index=dates,
        )

    def test_ema_params(self):
        """Test EMA default parameters."""
        params = EMA.params()
        assert params == {"length": 200}

    def test_ema_initialization(self):
        """Test EMA indicator initialization."""
        ema = EMA()
        assert ema.length == 200

        ema_custom = EMA(length=50)
        assert ema_custom.length == 50

    def test_ema_compute_output_format(self, sample_data):
        """Test EMA compute method output format."""
        ema = EMA(length=20)
        result = ema.compute(sample_data)

        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert result.index.equals(sample_data.index)

        # Check column names
        expected_cols = ["EMA_20", "EMA_20_trend"]
        assert list(result.columns) == expected_cols

        # Check data types
        assert result["EMA_20"].dtype == float
        assert result["EMA_20_trend"].dtype == object

    def test_ema_accuracy_against_ta_library(self, sample_data):
        """Test EMA calculation accuracy against ta library baseline."""
        ema = EMA(length=20)
        result = ema.compute(sample_data)

        # Calculate baseline using ta library directly
        baseline = ta_trend.EMAIndicator(
            close=sample_data["close"], window=20
        ).ema_indicator()

        # Compare values (allowing for small floating point differences)
        pd.testing.assert_series_equal(
            result["EMA_20"], baseline, check_names=False, rtol=1e-10
        )

    def test_ema_trend_detection(self, sample_data):
        """Test EMA trend detection logic."""
        ema = EMA(length=20)
        result = ema.compute(sample_data)

        # Check trend classification
        ema_values = result["EMA_20"]
        close_prices = sample_data["close"]
        trends = result["EMA_20_trend"]

        # Verify trend logic
        bull_mask = close_prices > ema_values
        bear_mask = close_prices < ema_values
        neutral_mask = close_prices == ema_values

        assert (trends[bull_mask] == "bull").all()
        assert (trends[bear_mask] == "bear").all()
        assert (trends[neutral_mask] == "neutral").all()

    def test_ema_insufficient_data(self):
        """Test EMA behavior with insufficient data."""
        # Create minimal data
        small_data = pd.DataFrame(
            {
                "close": [100.0],
                "open": [99.0],
                "high": [101.0],
                "low": [98.0],
                "volume": [1000],
            },
            index=[pd.Timestamp("2024-01-01")],
        )

        ema = EMA(length=20)
        result = ema.compute(small_data)

        assert len(result) == 1
        assert pd.isna(result["EMA_20"].iloc[0])
        assert result["EMA_20_trend"].iloc[0] is None


class TestRSIIndicator:
    """Test RSI indicator implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1H")
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data
        close_prices = []
        price = 50000.0
        for _ in range(50):
            change = np.random.normal(0, 0.02)  # 2% volatility
            price *= 1 + change
            close_prices.append(price)

        return pd.DataFrame(
            {
                "open": [p * 0.999 for p in close_prices],
                "high": [p * 1.01 for p in close_prices],
                "low": [p * 0.99 for p in close_prices],
                "close": close_prices,
                "volume": np.random.randint(100, 1000, 50),
            },
            index=dates,
        )

    def test_rsi_params(self):
        """Test RSI default parameters."""
        params = RSI.params()
        assert params == {"length": 14}

    def test_rsi_initialization(self):
        """Test RSI indicator initialization."""
        rsi = RSI()
        assert rsi.length == 14

        rsi_custom = RSI(length=21)
        assert rsi_custom.length == 21

    def test_rsi_compute_output_format(self, sample_data):
        """Test RSI compute method output format."""
        rsi = RSI(length=14)
        result = rsi.compute(sample_data)

        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert result.index.equals(sample_data.index)

        # Check column names
        expected_cols = ["RSI_14", "RSI_14_overbought", "RSI_14_oversold"]
        assert list(result.columns) == expected_cols

        # Check data types
        assert result["RSI_14"].dtype == float
        assert result["RSI_14_overbought"].dtype == bool
        assert result["RSI_14_oversold"].dtype == bool

    def test_rsi_accuracy_against_ta_library(self, sample_data):
        """Test RSI calculation accuracy against ta library baseline."""
        rsi = RSI(length=14)
        result = rsi.compute(sample_data)

        # Calculate baseline using ta library directly
        baseline = ta_momentum.RSIIndicator(close=sample_data["close"], window=14).rsi()

        # Compare values (allowing for small floating point differences)
        pd.testing.assert_series_equal(
            result["RSI_14"], baseline, check_names=False, rtol=1e-10
        )

    def test_rsi_signal_detection(self, sample_data):
        """Test RSI overbought/oversold signal detection."""
        rsi = RSI(length=14)
        result = rsi.compute(sample_data)

        rsi_values = result["RSI_14"]
        overbought = result["RSI_14_overbought"]
        oversold = result["RSI_14_oversold"]

        # Verify signal logic
        assert (overbought == (rsi_values > 70)).all()
        assert (oversold == (rsi_values < 30)).all()

        # Ensure mutual exclusivity when applicable
        extreme_mask = overbought | oversold
        if extreme_mask.any():
            assert not (overbought & oversold).any()

    def test_rsi_value_bounds(self, sample_data):
        """Test that RSI values are within expected bounds."""
        rsi = RSI(length=14)
        result = rsi.compute(sample_data)

        rsi_values = result["RSI_14"].dropna()

        # RSI should be between 0 and 100
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()

    def test_rsi_insufficient_data(self):
        """Test RSI behavior with insufficient data."""
        # Create data with less than required periods
        small_data = pd.DataFrame(
            {
                "close": [100.0, 101.0, 99.0],
                "open": [99.0, 100.0, 98.0],
                "high": [101.0, 102.0, 100.0],
                "low": [98.0, 99.0, 97.0],
                "volume": [1000, 1100, 900],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="1H"),
        )

        rsi = RSI(length=14)
        result = rsi.compute(small_data)

        assert len(result) == 3
        assert pd.isna(result["RSI_14"]).all()
        assert not result["RSI_14_overbought"].any()
        assert not result["RSI_14_oversold"].any()


class TestIndicatorIntegration:
    """Test integration scenarios with both indicators."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for integration testing."""
        dates = pd.date_range("2024-01-01", periods=300, freq="1H")
        np.random.seed(42)

        close_prices = []
        price = 50000.0
        for _ in range(300):
            change = np.random.normal(0, 0.015)
            price *= 1 + change
            close_prices.append(price)

        return pd.DataFrame(
            {
                "open": [p * 0.999 for p in close_prices],
                "high": [p * 1.008 for p in close_prices],
                "low": [p * 0.992 for p in close_prices],
                "close": close_prices,
                "volume": np.random.randint(500, 2000, 300),
            },
            index=dates,
        )

    def test_combined_indicator_usage(self, sample_data):
        """Test using both EMA and RSI indicators together."""
        ema = EMA(length=50)
        rsi = RSI(length=14)

        ema_result = ema.compute(sample_data)
        rsi_result = rsi.compute(sample_data)

        # Combine results (simulating engine behavior)
        combined = sample_data.join(ema_result).join(rsi_result)

        # Verify no column name conflicts
        expected_cols = [
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
        assert list(combined.columns) == expected_cols

    def test_registry_retrieval_and_instantiation(self, sample_data):
        """Test retrieving indicators from registry and using them."""
        # Get classes from registry
        ema_class = get("EMA")
        rsi_class = get("RSI")

        # Instantiate with custom parameters
        ema = ema_class(length=100)
        rsi = rsi_class(length=21)

        # Compute results
        ema_result = ema.compute(sample_data)
        rsi_result = rsi.compute(sample_data)

        # Verify correct column naming
        assert "EMA_100" in ema_result.columns
        assert "RSI_21" in rsi_result.columns

    def test_edge_case_handling(self):
        """Test edge cases across both indicators."""
        # Empty DataFrame
        empty_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        ema = EMA(length=20)
        rsi = RSI(length=14)

        # Should handle gracefully without errors
        ema_result = ema.compute(empty_data)
        rsi_result = rsi.compute(empty_data)

        assert len(ema_result) == 0
        assert len(rsi_result) == 0
        assert list(ema_result.columns) == ["EMA_20", "EMA_20_trend"]
        assert list(rsi_result.columns) == [
            "RSI_14",
            "RSI_14_overbought",
            "RSI_14_oversold",
        ]
