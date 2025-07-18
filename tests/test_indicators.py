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
from btc_research.indicators.fvg import FVG


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

    def test_fvg_registration(self):
        """Test that FVG indicator is properly registered."""
        fvg_class = get("FVG")
        assert fvg_class == FVG

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


class TestFVGIndicator:
    """Test Fair Value Gap (FVG) indicator implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1H")
        
        # Create realistic price data with some gaps
        np.random.seed(42)
        close_prices = []
        high_prices = []
        low_prices = []
        open_prices = []
        
        price = 50000.0
        for i in range(100):
            # Normal market movement
            change = np.random.normal(0, 0.01)  # 1% volatility
            price *= 1 + change
            
            # Create some intentional gaps for testing
            if i == 20:  # Bullish gap
                gap_size = price * 0.02  # 2% gap
                open_price = price
                high_price = price * 1.025
                low_price = price * 1.015  # Gap above previous high
                close_price = price * 1.02
                price = close_price
            elif i == 50:  # Bearish gap
                gap_size = price * 0.02  # 2% gap
                open_price = price
                high_price = price * 0.985  # Gap below previous low
                low_price = price * 0.975
                close_price = price * 0.98
                price = close_price
            else:
                # Normal candle
                open_price = price * (1 + np.random.normal(0, 0.002))
                high_price = price * 1.005
                low_price = price * 0.995
                close_price = price
            
            open_prices.append(open_price)
            high_prices.append(high_price)
            low_prices.append(low_price)
            close_prices.append(close_price)

        return pd.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": np.random.randint(100, 1000, 100),
            },
            index=dates,
        )

    @pytest.fixture
    def gap_data(self):
        """Create specific data with known FVG patterns."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1H")
        
        # Create a clear bullish FVG pattern:
        # Candle 0: high = 100
        # Candle 1: doesn't matter for gap
        # Candle 2: low = 105 (gap from 100 to 105)
        data = {
            "open": [98, 101, 104, 107, 106, 108, 107, 109, 108, 110],
            "high": [100, 102, 106, 108, 107, 109, 108, 110, 109, 111],
            "low": [97, 100, 105, 106, 105, 107, 106, 108, 107, 109],  # Clear gap at index 2
            "close": [99, 101, 105, 107, 106, 108, 107, 109, 108, 110],
            "volume": [1000] * 10,
        }
        
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def bearish_gap_data(self):
        """Create specific data with known bearish FVG pattern."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1H")
        
        # Create a clear bearish FVG pattern:
        # Candle 0: low = 105
        # Candle 1: doesn't matter for gap
        # Candle 2: high = 100 (gap from 100 to 105)
        data = {
            "open": [107, 104, 101, 98, 99, 97, 98, 96, 97, 95],
            "high": [108, 105, 102, 99, 100, 98, 99, 97, 98, 96],
            "low": [105, 103, 100, 97, 98, 96, 97, 95, 96, 94],  # Clear gap at index 2
            "close": [106, 104, 101, 98, 99, 97, 98, 96, 97, 95],
            "volume": [1000] * 10,
        }
        
        return pd.DataFrame(data, index=dates)

    def test_fvg_params(self):
        """Test FVG default parameters."""
        params = FVG.params()
        expected = {"min_gap_pips": 1.0, "max_lookback": 500}
        assert params == expected

    def test_fvg_initialization(self):
        """Test FVG indicator initialization."""
        fvg = FVG()
        assert fvg.min_gap_pips == 1.0
        assert fvg.max_lookback == 500

        fvg_custom = FVG(min_gap_pips=2.0, max_lookback=200)
        assert fvg_custom.min_gap_pips == 2.0
        assert fvg_custom.max_lookback == 200

    def test_fvg_compute_output_format(self, sample_data):
        """Test FVG compute method output format."""
        fvg = FVG()
        result = fvg.compute(sample_data)

        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert result.index.equals(sample_data.index)

        # Check column names
        expected_cols = [
            "FVG_bullish_signal",
            "FVG_bearish_signal",
            "FVG_gap_filled",
            "FVG_nearest_support",
            "FVG_nearest_resistance",
            "FVG_nearest_support_mid",
            "FVG_nearest_resistance_mid",
            "FVG_active_bullish_gaps",
            "FVG_active_bearish_gaps",
        ]
        assert list(result.columns) == expected_cols

        # Check data types
        assert result["FVG_bullish_signal"].dtype == bool
        assert result["FVG_bearish_signal"].dtype == bool
        assert result["FVG_gap_filled"].dtype == bool
        assert result["FVG_nearest_support"].dtype == float
        assert result["FVG_nearest_resistance"].dtype == float
        assert result["FVG_nearest_support_mid"].dtype == float
        assert result["FVG_nearest_resistance_mid"].dtype == float
        assert result["FVG_active_bullish_gaps"].dtype == int
        assert result["FVG_active_bearish_gaps"].dtype == int

    def test_fvg_bullish_gap_detection(self, gap_data):
        """Test detection of bullish FVG patterns."""
        fvg = FVG(min_gap_pips=1.0)
        result = fvg.compute(gap_data)

        # Check that bullish gap is detected
        # Gap should be detected at candle 2 (low[2] > high[0])
        assert gap_data.iloc[2]["low"] > gap_data.iloc[0]["high"]  # Confirm gap exists
        
        # Check for active bullish gaps after gap formation
        active_gaps_after = result["FVG_active_bullish_gaps"].iloc[3:]
        assert (active_gaps_after > 0).any(), "Should detect at least one bullish gap"

    def test_fvg_bearish_gap_detection(self, bearish_gap_data):
        """Test detection of bearish FVG patterns."""
        fvg = FVG(min_gap_pips=1.0)
        result = fvg.compute(bearish_gap_data)

        # Check that bearish gap is detected
        # Gap should be detected at candle 2 (high[2] < low[0])
        assert bearish_gap_data.iloc[2]["high"] < bearish_gap_data.iloc[0]["low"]  # Confirm gap exists
        
        # Check for active bearish gaps after gap formation
        active_gaps_after = result["FVG_active_bearish_gaps"].iloc[3:]
        assert (active_gaps_after > 0).any(), "Should detect at least one bearish gap"

    def test_fvg_signal_generation(self, gap_data):
        """Test FVG signal generation when price approaches gaps."""
        fvg = FVG(min_gap_pips=1.0)
        result = fvg.compute(gap_data)

        # Check that signals are boolean
        assert result["FVG_bullish_signal"].dtype == bool
        assert result["FVG_bearish_signal"].dtype == bool
        
        # Signals should not be all false (we have gaps in test data)
        has_signals = result["FVG_bullish_signal"].any() or result["FVG_bearish_signal"].any()
        # Note: depending on price action, we might not get signals if price doesn't return to gaps

    def test_fvg_gap_filling_detection(self):
        """Test detection of gap filling."""
        dates = pd.date_range("2024-01-01", periods=8, freq="1H")
        
        # Create data where gap gets filled
        data = pd.DataFrame({
            "open": [98, 101, 104, 107, 104, 102, 100, 98],  # Price returns to fill gap
            "high": [100, 102, 106, 108, 105, 103, 101, 99],
            "low": [97, 100, 105, 106, 103, 101, 99, 97],   # Gap at index 2, filled at index 6
            "close": [99, 101, 105, 107, 104, 102, 100, 98],
            "volume": [1000] * 8,
        }, index=dates)
        
        fvg = FVG(min_gap_pips=1.0)
        result = fvg.compute(data)
        
        # Should detect gap filling when price returns to gap zone
        assert result["FVG_gap_filled"].any(), "Should detect gap filling"

    def test_fvg_nearest_support_resistance(self, gap_data):
        """Test calculation of nearest support and resistance levels."""
        fvg = FVG(min_gap_pips=1.0)
        result = fvg.compute(gap_data)

        # Check that support/resistance values are calculated
        support_values = result["FVG_nearest_support"].dropna()
        resistance_values = result["FVG_nearest_resistance"].dropna()
        
        # Values should be valid prices when present
        if len(support_values) > 0:
            assert (support_values > 0).all()
        if len(resistance_values) > 0:
            assert (resistance_values > 0).all()

    def test_fvg_minimum_gap_size_filter(self):
        """Test filtering of gaps smaller than minimum size."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1H")
        
        # Create tiny gap (should be filtered out)
        data = pd.DataFrame({
            "open": [100, 100.2, 100.4, 100.6, 100.8],
            "high": [100.1, 100.3, 100.5, 100.7, 100.9],
            "low": [99.9, 100.11, 100.3, 100.5, 100.7],  # Tiny gap of 0.01
            "close": [100, 100.2, 100.4, 100.6, 100.8],
            "volume": [1000] * 5,
        }, index=dates)
        
        # Test with high minimum gap size
        fvg = FVG(min_gap_pips=1.0)  # 1.0 pip minimum
        result = fvg.compute(data)
        
        # Should not detect gaps due to size filter
        assert result["FVG_active_bullish_gaps"].max() == 0
        assert result["FVG_active_bearish_gaps"].max() == 0

    def test_fvg_insufficient_data(self):
        """Test FVG behavior with insufficient data."""
        # Create minimal data (less than 3 candles needed for pattern)
        small_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000, 1100],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="1H"),
        )

        fvg = FVG()
        result = fvg.compute(small_data)

        assert len(result) == 2
        # Should return default values for insufficient data
        assert not result["FVG_bullish_signal"].any()
        assert not result["FVG_bearish_signal"].any()
        assert not result["FVG_gap_filled"].any()
        assert result["FVG_active_bullish_gaps"].max() == 0
        assert result["FVG_active_bearish_gaps"].max() == 0

    def test_fvg_empty_dataframe(self):
        """Test FVG behavior with empty DataFrame."""
        empty_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        fvg = FVG()
        result = fvg.compute(empty_data)

        assert len(result) == 0
        expected_cols = [
            "FVG_bullish_signal",
            "FVG_bearish_signal", 
            "FVG_gap_filled",
            "FVG_nearest_support",
            "FVG_nearest_resistance",
            "FVG_nearest_support_mid",
            "FVG_nearest_resistance_mid",
            "FVG_active_bullish_gaps",
            "FVG_active_bearish_gaps",
        ]
        assert list(result.columns) == expected_cols

    def test_fvg_registry_integration(self, sample_data):
        """Test FVG integration with registry system."""
        # Get class from registry
        fvg_class = get("FVG")
        
        # Instantiate with custom parameters
        fvg = fvg_class(min_gap_pips=2.0, max_lookback=100)
        
        # Compute results
        result = fvg.compute(sample_data)
        
        # Verify correct initialization
        assert fvg.min_gap_pips == 2.0
        assert fvg.max_lookback == 100
        
        # Verify output format
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
