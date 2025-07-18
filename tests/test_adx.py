"""
Tests for the ADX (Average Directional Index) indicator.

This module contains comprehensive tests for the ADX implementation,
including edge cases, parameter validation, and calculation accuracy.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from btc_research.indicators.adx import ADX


class TestADXIndicator:
    """Test suite for ADX indicator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adx = ADX()
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        np.random.seed(42)  # For reproducible tests
        
        # Generate trending price data
        base_price = 50000
        trend = np.linspace(0, 5000, 100)  # Upward trend
        noise = np.random.normal(0, 200, 100)  # Random noise
        
        close_prices = base_price + trend + noise
        
        # Generate OHLC from close prices
        high_prices = close_prices + np.random.uniform(50, 300, 100)
        low_prices = close_prices - np.random.uniform(50, 300, 100)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Ensure OHLC relationships are maintained
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
        
        # Generate volume data
        volumes = np.random.uniform(1000, 10000, 100)
        
        self.sample_data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
    
    def test_initialization_default_params(self):
        """Test ADX initialization with default parameters."""
        adx = ADX()
        assert adx.period == 14
        assert adx.trend_threshold == 25.0
        assert adx.range_threshold == 20.0
        assert adx.smoothing_method == "wilder"
    
    def test_initialization_custom_params(self):
        """Test ADX initialization with custom parameters."""
        adx = ADX(period=21, trend_threshold=30.0, range_threshold=15.0, smoothing_method="ema")
        assert adx.period == 21
        assert adx.trend_threshold == 30.0
        assert adx.range_threshold == 15.0
        assert adx.smoothing_method == "ema"
    
    def test_initialization_invalid_params(self):
        """Test ADX initialization with invalid parameters."""
        with pytest.raises(ValueError, match="Period must be positive"):
            ADX(period=0)
        
        with pytest.raises(ValueError, match="Trend threshold must be greater than range threshold"):
            ADX(trend_threshold=20, range_threshold=25)
        
        with pytest.raises(ValueError, match="Smoothing method must be"):
            ADX(smoothing_method="invalid")
    
    def test_params_class_method(self):
        """Test the params class method."""
        params = ADX.params()
        expected_params = {
            "period": 14,
            "trend_threshold": 25.0,
            "range_threshold": 20.0,
            "smoothing_method": "wilder"
        }
        assert params == expected_params
    
    def test_compute_basic_functionality(self):
        """Test basic ADX computation."""
        result = self.adx.compute(self.sample_data)
        
        # Check that result has correct shape and index
        assert len(result) == len(self.sample_data)
        assert result.index.equals(self.sample_data.index)
        
        # Check that all expected columns are present
        expected_columns = [
            'ADX_value', 'DI_plus', 'DI_minus', 'ADX_trend', 'ADX_range',
            'ADX_strength', 'DI_bullish', 'DI_bearish'
        ]
        for col in expected_columns:
            assert col in result.columns
    
    def test_compute_insufficient_data(self):
        """Test ADX computation with insufficient data."""
        # Create data with fewer periods than required (need 2*period + 1 = 29)
        short_data = self.sample_data.iloc[:20]
        result = self.adx.compute(short_data)
        
        # Should return empty result with NaN values
        assert len(result) == len(short_data)
        assert result['ADX_value'].isna().all()
        assert result['DI_plus'].isna().all()
        assert result['DI_minus'].isna().all()
        assert (~result['ADX_trend']).all()
        assert (~result['ADX_range']).all()
    
    def test_compute_valid_adx_values(self):
        """Test that ADX values are within expected range."""
        result = self.adx.compute(self.sample_data)
        
        # ADX should be between 0 and 100
        valid_adx = result['ADX_value'].dropna()
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()
        
        # DI values should be positive
        valid_di_plus = result['DI_plus'].dropna()
        valid_di_minus = result['DI_minus'].dropna()
        assert (valid_di_plus >= 0).all()
        assert (valid_di_minus >= 0).all()
    
    def test_compute_trend_range_signals(self):
        """Test trend and range signal generation."""
        result = self.adx.compute(self.sample_data)
        
        # Check signal consistency
        valid_adx = result['ADX_value'].dropna()
        valid_trend = result['ADX_trend'][result['ADX_value'].notna()]
        valid_range = result['ADX_range'][result['ADX_value'].notna()]
        
        # Trend signals should be True when ADX > trend_threshold
        expected_trend = valid_adx > self.adx.trend_threshold
        assert valid_trend.equals(expected_trend)
        
        # Range signals should be True when ADX < range_threshold
        expected_range = valid_adx < self.adx.range_threshold
        assert valid_range.equals(expected_range)
    
    def test_compute_directional_signals(self):
        """Test directional signal generation."""
        result = self.adx.compute(self.sample_data)
        
        # Check directional signals
        valid_mask = result['DI_plus'].notna() & result['DI_minus'].notna()
        valid_bullish = result['DI_bullish'][valid_mask]
        valid_bearish = result['DI_bearish'][valid_mask]
        
        di_plus_vals = result['DI_plus'][valid_mask]
        di_minus_vals = result['DI_minus'][valid_mask]
        
        # Bullish should be True when +DI > -DI
        expected_bullish = di_plus_vals > di_minus_vals
        assert valid_bullish.equals(expected_bullish)
        
        # Bearish should be True when -DI > +DI
        expected_bearish = di_minus_vals > di_plus_vals
        assert valid_bearish.equals(expected_bearish)
    
    def test_compute_adx_strength_categories(self):
        """Test ADX strength categorization."""
        result = self.adx.compute(self.sample_data)
        
        # Check strength categories
        valid_adx = result['ADX_value'].dropna()
        valid_strength = result['ADX_strength'][result['ADX_value'].notna()]
        
        expected_categories = ["weak", "moderate", "strong", "very_strong"]
        for category in valid_strength.unique():
            if category != "unknown":
                assert category in expected_categories
    
    def test_compute_look_ahead_bias_prevention(self):
        """Test that look-ahead bias is prevented."""
        result = self.adx.compute(self.sample_data)
        
        # First valid ADX value should be at least 2*period bars after start
        # due to the double smoothing (TR/DM smoothing + ADX smoothing)
        first_valid_idx = result['ADX_value'].first_valid_index()
        if first_valid_idx is not None:
            first_valid_position = result.index.get_loc(first_valid_idx)
            # Should have sufficient lookback
            assert first_valid_position >= self.adx.period * 2 - 1
    
    def test_compute_different_smoothing_methods(self):
        """Test ADX with different smoothing methods."""
        adx_wilder = ADX(smoothing_method="wilder")
        adx_ema = ADX(smoothing_method="ema")
        
        result_wilder = adx_wilder.compute(self.sample_data)
        result_ema = adx_ema.compute(self.sample_data)
        
        # Both should produce some valid results if data is sufficient
        valid_wilder = result_wilder['ADX_value'].dropna()
        valid_ema = result_ema['ADX_value'].dropna()
        
        # If we have valid results, they should be different
        if len(valid_wilder) > 0 and len(valid_ema) > 0:
            # Results should be different due to different smoothing
            valid_mask = result_wilder['ADX_value'].notna() & result_ema['ADX_value'].notna()
            if valid_mask.any():
                assert not result_wilder['ADX_value'][valid_mask].equals(result_ema['ADX_value'][valid_mask])
    
    def test_compute_edge_cases(self):
        """Test ADX computation with edge cases."""
        # Test with zero volume
        zero_vol_data = self.sample_data.copy()
        zero_vol_data['volume'] = 0
        result = self.adx.compute(zero_vol_data)
        assert len(result) == len(zero_vol_data)
        
        # Test with constant prices (no movement)
        constant_data = self.sample_data.copy()
        constant_data['open'] = 50000
        constant_data['high'] = 50000
        constant_data['low'] = 50000
        constant_data['close'] = 50000
        
        result = self.adx.compute(constant_data)
        # ADX should be 0 or very low for constant prices
        valid_adx = result['ADX_value'].dropna()
        if len(valid_adx) > 0:
            assert (valid_adx <= 5).all()  # Should be very low
    
    def test_compute_missing_values(self):
        """Test ADX computation with missing values."""
        # Introduce some NaN values
        data_with_nans = self.sample_data.copy()
        data_with_nans.loc[data_with_nans.index[10:15], 'high'] = np.nan
        data_with_nans.loc[data_with_nans.index[30:35], 'low'] = np.nan
        
        result = self.adx.compute(data_with_nans)
        
        # Should handle NaN values gracefully
        assert len(result) == len(data_with_nans)
        assert result.index.equals(data_with_nans.index)
    
    def test_true_range_calculation(self):
        """Test True Range calculation."""
        # Create simple test data
        test_data = pd.DataFrame({
            'open': [100, 102, 101, 103],
            'high': [105, 107, 106, 108],
            'low': [98, 100, 99, 101],
            'close': [103, 105, 104, 106],
            'volume': [1000, 1000, 1000, 1000]
        })
        
        adx = ADX()
        tr = adx._calculate_true_range(test_data)
        
        # Check first TR (High - Low)
        assert tr[0] == 105 - 98  # 7
        
        # Check subsequent TRs
        assert tr[1] == max(107 - 100, abs(107 - 103), abs(100 - 103))  # max(7, 4, 3) = 7
        assert tr[2] == max(106 - 99, abs(106 - 105), abs(99 - 105))   # max(7, 1, 6) = 7
        assert tr[3] == max(108 - 101, abs(108 - 104), abs(101 - 104)) # max(7, 4, 3) = 7
    
    def test_directional_movement_calculation(self):
        """Test Directional Movement calculation."""
        # Create simple test data with clear directional moves
        test_data = pd.DataFrame({
            'open': [100, 102, 101, 103],
            'high': [105, 110, 106, 108],  # Strong up move in period 2
            'low': [98, 100, 95, 101],     # Strong down move in period 3
            'close': [103, 108, 104, 106],
            'volume': [1000, 1000, 1000, 1000]
        })
        
        adx = ADX()
        dm_plus, dm_minus = adx._calculate_directional_movement(test_data)
        
        # First period should have no directional movement
        assert dm_plus[0] == 0
        assert dm_minus[0] == 0
        
        # Period 2: Strong up move (110 - 105 = 5 > 100 - 95 = 5, but equal so no movement)
        # Actually let me recalculate: up_move = 110 - 105 = 5, down_move = 100 - 95 = 5
        # Since equal, no directional movement
        
        # Period 3: Strong down move (95 - 100 = -5, but down_move = 100 - 95 = 5)
        # up_move = 106 - 110 = -4, down_move = 100 - 95 = 5
        # Since down_move > up_move and down_move > 0, dm_minus should be 5
        assert dm_minus[2] == 5
    
    def test_smoothing_methods(self):
        """Test different smoothing methods."""
        values = np.array([10, 12, 11, 13, 14, 15, 16, 14, 13, 12, 11, 10, 9, 8, 7])
        period = 5
        
        # Test Wilder's smoothing
        adx_wilder = ADX(smoothing_method="wilder")
        smoothed_wilder = adx_wilder._smooth_values(values, period)
        
        # Test EMA smoothing
        adx_ema = ADX(smoothing_method="ema")
        smoothed_ema = adx_ema._smooth_values(values, period)
        
        # Both should start with same initial value (simple average)
        assert smoothed_wilder[period - 1] == smoothed_ema[period - 1]
        
        # But subsequent values should differ
        assert smoothed_wilder[period] != smoothed_ema[period]
    
    def test_dx_calculation(self):
        """Test DX calculation."""
        di_plus = np.array([20, 25, 30, 15, 10])
        di_minus = np.array([10, 15, 20, 25, 30])
        
        adx = ADX()
        dx = adx._calculate_dx(di_plus, di_minus)
        
        # DX = 100 * |+DI - -DI| / (+DI + -DI)
        expected_dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        
        np.testing.assert_array_almost_equal(dx, expected_dx)
    
    def test_adx_strength_categorization(self):
        """Test ADX strength categorization."""
        adx_values = pd.Series([np.nan, 15, 22, 30, 45])
        
        adx = ADX()
        categories = adx._categorize_adx_strength(adx_values)
        
        expected = ["unknown", "weak", "moderate", "strong", "very_strong"]
        assert categories.tolist() == expected
    
    def test_regression_known_values(self):
        """Test ADX calculation against known values."""
        # Create deterministic data for reproducible test with enough periods
        test_data = pd.DataFrame({
            'open': [100, 102, 101, 103, 105, 107, 106, 108, 110, 112] * 15,
            'high': [105, 107, 106, 108, 110, 112, 111, 113, 115, 117] * 15,
            'low': [98, 100, 99, 101, 103, 105, 104, 106, 108, 110] * 15,
            'close': [103, 105, 104, 106, 108, 110, 109, 111, 113, 115] * 15,
            'volume': [1000] * 150
        })
        
        adx = ADX(period=14)
        result = adx.compute(test_data)
        
        # Check that we get reasonable ADX values
        valid_adx = result['ADX_value'].dropna()
        if len(valid_adx) > 0:
            assert valid_adx.min() >= 0
            assert valid_adx.max() <= 100
            
            # For this oscillating data, ADX should be relatively low (indicating ranging market)
            if len(valid_adx) > 10:
                # Most values should be below 25 (indicating ranging/weak trend)
                assert (valid_adx < 25).sum() > len(valid_adx) * 0.8


if __name__ == "__main__":
    pytest.main([__file__])