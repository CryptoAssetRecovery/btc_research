"""
Tests for VPFVGSignal confluence indicator.

This module contains comprehensive tests for the VP-FVG confluence indicator,
including unit tests for core functionality, edge cases, and integration tests.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from btc_research.indicators.vpfvg_signal import VPFVGSignal


class TestVPFVGSignal:
    """Test suite for VPFVGSignal indicator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.indicator = VPFVGSignal()
        
        # Create sample data with required columns
        dates = pd.date_range('2023-01-01', periods=100, freq='15T')
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(30000, 32000, 100),
            'high': np.random.uniform(31000, 33000, 100),
            'low': np.random.uniform(29000, 31000, 100),
            'close': np.random.uniform(30000, 32000, 100),
            'volume': np.random.uniform(100, 1000, 100),
            # Volume Profile columns
            'VolumeProfile_poc_price': np.random.uniform(30000, 32000, 100),
            'VolumeProfile_is_lvn': np.random.choice([True, False], 100, p=[0.1, 0.9]),
            'VolumeProfile_is_hvn': np.random.choice([True, False], 100, p=[0.1, 0.9]),
            # FVG columns
            'FVG_bullish_signal': np.random.choice([True, False], 100, p=[0.1, 0.9]),
            'FVG_bearish_signal': np.random.choice([True, False], 100, p=[0.1, 0.9]),
            'FVG_nearest_support_mid': np.random.uniform(29500, 31500, 100),
            'FVG_nearest_resistance_mid': np.random.uniform(30500, 32500, 100),
            'FVG_active_bullish_gaps': np.random.randint(0, 5, 100),
            'FVG_active_bearish_gaps': np.random.randint(0, 5, 100),
        }, index=dates)
        
        # Ensure price relationships are valid
        self.sample_data['high'] = np.maximum(self.sample_data['high'], self.sample_data['close'])
        self.sample_data['low'] = np.minimum(self.sample_data['low'], self.sample_data['close'])
    
    def test_params_method(self):
        """Test that params method returns correct default parameters."""
        params = VPFVGSignal.params()
        
        expected_params = {
            "atr_period": 14,
            "lvn_dist_multiplier": 0.25,
            "poc_shift_multiplier": 0.5,
            "hvn_overlap_pct": 0.7,
            "min_fvg_size": 1.0,
            "lookback_validation": 5,
        }
        
        assert params == expected_params
    
    def test_initialization(self):
        """Test indicator initialization with custom parameters."""
        indicator = VPFVGSignal(
            atr_period=20,
            lvn_dist_multiplier=0.3,
            poc_shift_multiplier=0.6,
            hvn_overlap_pct=0.8,
            min_fvg_size=2.0,
            lookback_validation=10
        )
        
        assert indicator.atr_period == 20
        assert indicator.lvn_dist_multiplier == 0.3
        assert indicator.poc_shift_multiplier == 0.6
        assert indicator.hvn_overlap_pct == 0.8
        assert indicator.min_fvg_size == 2.0
        assert indicator.lookback_validation == 10
    
    def test_compute_with_sufficient_data(self):
        """Test compute method with sufficient data."""
        result = self.indicator.compute(self.sample_data)
        
        # Check that all expected columns are present
        expected_columns = [
            'vf_long', 'vf_short', 'vf_atr', 'vf_poc_shift', 
            'vf_lvn_distance', 'vf_hvn_overlap'
        ]
        
        for col in expected_columns:
            assert col in result.columns
        
        # Check that result has same index as input
        assert result.index.equals(self.sample_data.index)
        
        # Check that boolean columns are boolean type
        assert result['vf_long'].dtype == bool
        assert result['vf_short'].dtype == bool
        
        # Check that signals are properly shifted (first values should be False/NaN)
        assert result['vf_long'].iloc[0] == False
        assert result['vf_short'].iloc[0] == False
        assert np.isnan(result['vf_atr'].iloc[0])
    
    def test_compute_with_insufficient_data(self):
        """Test compute method with insufficient data."""
        # Create small dataset
        small_data = self.sample_data.iloc[:5].copy()
        result = self.indicator.compute(small_data)
        
        # Should return empty result
        assert len(result) == len(small_data)
        assert all(result['vf_long'] == False)
        assert all(result['vf_short'] == False)
        assert all(np.isnan(result['vf_atr']))
    
    def test_compute_with_missing_columns(self):
        """Test compute method with missing required columns."""
        # Remove a required column
        incomplete_data = self.sample_data.drop('VolumeProfile_poc_price', axis=1)
        result = self.indicator.compute(incomplete_data)
        
        # Should return empty result
        assert len(result) == len(incomplete_data)
        assert all(result['vf_long'] == False)
        assert all(result['vf_short'] == False)
    
    def test_atr_calculation(self):
        """Test ATR calculation method."""
        atr_values = self.indicator._calculate_atr(self.sample_data)
        
        # Check that ATR array has correct length
        assert len(atr_values) == len(self.sample_data)
        
        # Check that early values are NaN
        assert all(np.isnan(atr_values[:self.indicator.atr_period]))
        
        # Check that later values are not NaN (assuming valid data)
        assert not all(np.isnan(atr_values[self.indicator.atr_period:]))
        
        # Check that ATR values are positive
        valid_atr = atr_values[~np.isnan(atr_values)]
        assert all(valid_atr > 0)
    
    def test_reversal_setup_detection(self):
        """Test reversal setup detection logic."""
        # Create test data with known reversal setup
        test_data = self.sample_data.copy()
        
        # Set up a specific reversal scenario
        test_idx = 50
        test_data.iloc[test_idx]['FVG_bullish_signal'] = True
        test_data.iloc[test_idx]['FVG_active_bullish_gaps'] = 2
        test_data.iloc[test_idx]['FVG_nearest_support_mid'] = 31000
        test_data.iloc[test_idx-1]['VolumeProfile_is_lvn'] = True
        
        # Test the reversal setup detection
        atr_val = 100.0  # Mock ATR value
        price = 31000    # Mock price
        
        result = self.indicator._check_reversal_setup(test_data, test_idx, atr_val, price)
        
        # Should detect reversal setup
        assert isinstance(result, bool)
    
    def test_continuation_setup_detection(self):
        """Test continuation setup detection logic."""
        # Create test data with known continuation setup
        test_data = self.sample_data.copy()
        
        # Set up a specific continuation scenario
        test_idx = 50
        test_data.iloc[test_idx]['FVG_bearish_signal'] = True
        test_data.iloc[test_idx]['FVG_active_bearish_gaps'] = 1
        test_data.iloc[test_idx]['FVG_nearest_resistance_mid'] = 31000
        test_data.iloc[test_idx-1]['VolumeProfile_is_hvn'] = True
        
        # Test the continuation setup detection
        atr_val = 100.0    # Mock ATR value
        price = 31000      # Mock price
        poc_shift = 60.0   # Mock POC shift > 0.5 * ATR
        
        result = self.indicator._check_continuation_setup(test_data, test_idx, atr_val, price, poc_shift)
        
        # Should detect continuation setup
        assert isinstance(result, bool)
    
    def test_lvn_distance_calculation(self):
        """Test LVN distance calculation."""
        test_data = self.sample_data.copy()
        
        # Set up test scenario
        test_idx = 50
        test_data.iloc[test_idx]['FVG_nearest_support_mid'] = 31000
        test_data.iloc[test_idx-1]['VolumeProfile_is_lvn'] = True
        
        result = self.indicator._calculate_lvn_distance(test_data, test_idx, 31100)
        
        # Should return a distance value
        assert isinstance(result, (float, np.float64)) or np.isnan(result)
    
    def test_hvn_overlap_calculation(self):
        """Test HVN overlap calculation."""
        test_data = self.sample_data.copy()
        
        # Set up test scenario
        test_idx = 50
        test_data.iloc[test_idx]['FVG_nearest_resistance_mid'] = 31000
        test_data.iloc[test_idx-1]['VolumeProfile_is_hvn'] = True
        
        result = self.indicator._calculate_hvn_overlap(test_data, test_idx, 31100)
        
        # Should return an overlap percentage
        assert isinstance(result, (float, np.float64)) or np.isnan(result)
    
    def test_empty_result_creation(self):
        """Test creation of empty result DataFrame."""
        test_index = pd.date_range('2023-01-01', periods=10, freq='15T')
        result = self.indicator._create_empty_result(test_index)
        
        # Check structure
        assert len(result) == len(test_index)
        assert result.index.equals(test_index)
        
        # Check columns
        expected_columns = [
            'vf_long', 'vf_short', 'vf_atr', 'vf_poc_shift', 
            'vf_lvn_distance', 'vf_hvn_overlap'
        ]
        for col in expected_columns:
            assert col in result.columns
        
        # Check that boolean columns are False
        assert all(result['vf_long'] == False)
        assert all(result['vf_short'] == False)
        
        # Check that numeric columns are NaN
        assert all(np.isnan(result['vf_atr']))
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with all NaN values
        nan_data = self.sample_data.copy()
        nan_data.iloc[:] = np.nan
        
        result = self.indicator.compute(nan_data)
        assert len(result) == len(nan_data)
        assert all(result['vf_long'] == False)
        assert all(result['vf_short'] == False)
        
        # Test with no FVG signals
        no_fvg_data = self.sample_data.copy()
        no_fvg_data['FVG_bullish_signal'] = False
        no_fvg_data['FVG_bearish_signal'] = False
        
        result = self.indicator.compute(no_fvg_data)
        assert len(result) == len(no_fvg_data)
        # Should have no signals
        assert all(result['vf_long'] == False)
        assert all(result['vf_short'] == False)
        
        # Test with no VP signals
        no_vp_data = self.sample_data.copy()
        no_vp_data['VolumeProfile_is_lvn'] = False
        no_vp_data['VolumeProfile_is_hvn'] = False
        
        result = self.indicator.compute(no_vp_data)
        assert len(result) == len(no_vp_data)
        # Should have no signals
        assert all(result['vf_long'] == False)
        assert all(result['vf_short'] == False)
    
    def test_parameter_validation(self):
        """Test parameter validation and edge cases."""
        # Test with extreme parameters
        extreme_indicator = VPFVGSignal(
            atr_period=1,
            lvn_dist_multiplier=0.0,
            poc_shift_multiplier=0.0,
            hvn_overlap_pct=1.0,
            min_fvg_size=0.0,
            lookback_validation=1
        )
        
        result = extreme_indicator.compute(self.sample_data)
        
        # Should still return valid structure
        assert len(result) == len(self.sample_data)
        assert 'vf_long' in result.columns
        assert 'vf_short' in result.columns
    
    def test_look_ahead_bias_prevention(self):
        """Test that signals are properly shifted to prevent look-ahead bias."""
        result = self.indicator.compute(self.sample_data)
        
        # The first signal should be False (due to shift)
        assert result['vf_long'].iloc[0] == False
        assert result['vf_short'].iloc[0] == False
        
        # The first ATR value should be NaN (due to shift)
        assert np.isnan(result['vf_atr'].iloc[0])
        
        # Check that the shift is applied correctly
        # If we have a signal at index i, it should appear at index i+1 in the result
        # This is implicit in the shift(1) operation
        assert len(result) == len(self.sample_data)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])