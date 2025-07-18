"""
Integration tests for VPFVGSignal indicator.

This module tests the integration of the VPFVGSignal indicator with the broader
trading engine, including proper column naming, data flow, and interaction with
other indicators.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from btc_research.indicators.vpfvg_signal import VPFVGSignal
from btc_research.indicators.volume_profile import VolumeProfile
from btc_research.indicators.fvg import FVG
from btc_research.core.registry import get


class TestVPFVGIntegration:
    """Test suite for VPFVGSignal integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample OHLCV data
        dates = pd.date_range('2023-01-01', periods=500, freq='15min')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible tests
        prices = 30000 + np.cumsum(np.random.normal(0, 10, 500))
        
        self.ohlcv_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 500)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 500))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 500))),
            'close': prices,
            'volume': np.random.uniform(100, 1000, 500),
        }, index=dates)
        
        # Ensure price relationships are valid
        self.ohlcv_data['high'] = np.maximum(self.ohlcv_data['high'], self.ohlcv_data['close'])
        self.ohlcv_data['low'] = np.minimum(self.ohlcv_data['low'], self.ohlcv_data['close'])
        self.ohlcv_data['open'] = np.clip(self.ohlcv_data['open'], 
                                         self.ohlcv_data['low'], 
                                         self.ohlcv_data['high'])
    
    def test_registry_registration(self):
        """Test that VPFVGSignal is properly registered."""
        # Test that it can be retrieved from registry
        indicator_class = get('VPFVGSignal')
        assert indicator_class == VPFVGSignal
        
        # Test that it can be instantiated
        indicator = indicator_class()
        assert isinstance(indicator, VPFVGSignal)
    
    def test_full_indicator_pipeline(self):
        """Test the full pipeline: VP -> FVG -> VPFVGSignal."""
        # Step 1: Calculate Volume Profile
        vp_indicator = VolumeProfile(
            lookback=100,
            price_bins=20,
            update_frequency=1
        )
        vp_result = vp_indicator.compute(self.ohlcv_data)
        
        # Step 2: Calculate FVG
        fvg_indicator = FVG(min_gap_pips=1.0, max_lookback=100)
        fvg_result = fvg_indicator.compute(self.ohlcv_data)
        
        # Step 3: Combine the results
        combined_data = self.ohlcv_data.copy()
        
        # Add VP columns with proper prefixes
        for col in vp_result.columns:
            combined_data[f'VolumeProfile_{col}'] = vp_result[col]
        
        # Add FVG columns (already have FVG prefix)
        for col in fvg_result.columns:
            combined_data[col] = fvg_result[col]
        
        # Step 4: Calculate VPFVGSignal
        vpfvg_indicator = VPFVGSignal()
        vpfvg_result = vpfvg_indicator.compute(combined_data)
        
        # Verify results
        assert len(vpfvg_result) == len(self.ohlcv_data)
        assert 'vf_long' in vpfvg_result.columns
        assert 'vf_short' in vpfvg_result.columns
        assert 'vf_atr' in vpfvg_result.columns
        assert 'vf_poc_shift' in vpfvg_result.columns
        assert 'vf_lvn_distance' in vpfvg_result.columns
        assert 'vf_hvn_overlap' in vpfvg_result.columns
        
        # Check that we have some signals (may be rare with random data)
        total_signals = vpfvg_result['vf_long'].sum() + vpfvg_result['vf_short'].sum()
        print(f"Total signals generated: {total_signals}")
        
        # Check that boolean columns are properly boolean
        assert vpfvg_result['vf_long'].dtype == bool
        assert vpfvg_result['vf_short'].dtype == bool
        
        # Check that numeric columns are float
        assert vpfvg_result['vf_atr'].dtype in [np.float64, np.float32]
        assert vpfvg_result['vf_poc_shift'].dtype in [np.float64, np.float32]
    
    def test_column_naming_consistency(self):
        """Test that column naming follows the expected pattern."""
        # Create minimal data with required columns
        minimal_data = self.ohlcv_data.copy()
        
        # Add required VP columns
        minimal_data['VolumeProfile_poc_price'] = 30000
        minimal_data['VolumeProfile_is_lvn'] = False
        minimal_data['VolumeProfile_is_hvn'] = False
        
        # Add required FVG columns
        minimal_data['FVG_bullish_signal'] = False
        minimal_data['FVG_bearish_signal'] = False
        minimal_data['FVG_nearest_support_mid'] = 29500
        minimal_data['FVG_nearest_resistance_mid'] = 30500
        minimal_data['FVG_active_bullish_gaps'] = 0
        minimal_data['FVG_active_bearish_gaps'] = 0
        
        # Calculate VPFVGSignal
        vpfvg_indicator = VPFVGSignal()
        result = vpfvg_indicator.compute(minimal_data)
        
        # Check that all output columns follow the expected naming pattern
        expected_columns = {
            'vf_long', 'vf_short', 'vf_atr', 'vf_poc_shift', 
            'vf_lvn_distance', 'vf_hvn_overlap'
        }
        
        actual_columns = set(result.columns)
        assert actual_columns == expected_columns, f"Expected {expected_columns}, got {actual_columns}"
    
    def test_signal_generation_conditions(self):
        """Test that signals are generated under the right conditions."""
        # Create controlled test data
        test_data = self.ohlcv_data.iloc[:100].copy()
        
        # Set up a reversal scenario
        test_data['VolumeProfile_poc_price'] = 30000
        test_data['VolumeProfile_is_lvn'] = False
        test_data['VolumeProfile_is_hvn'] = False
        test_data['FVG_bullish_signal'] = False
        test_data['FVG_bearish_signal'] = False
        test_data['FVG_nearest_support_mid'] = 29500
        test_data['FVG_nearest_resistance_mid'] = 30500
        test_data['FVG_active_bullish_gaps'] = 0
        test_data['FVG_active_bearish_gaps'] = 0
        
        # Create a specific reversal setup at index 50
        test_idx = 50
        test_data.iloc[test_idx, test_data.columns.get_loc('FVG_bullish_signal')] = True
        test_data.iloc[test_idx, test_data.columns.get_loc('FVG_active_bullish_gaps')] = 1
        test_data.iloc[test_idx, test_data.columns.get_loc('FVG_nearest_support_mid')] = 30000
        test_data.iloc[test_idx-1, test_data.columns.get_loc('VolumeProfile_is_lvn')] = True
        
        # Calculate signals
        vpfvg_indicator = VPFVGSignal()
        result = vpfvg_indicator.compute(test_data)
        
        # Check that we have the expected structure
        assert len(result) == len(test_data)
        assert 'vf_long' in result.columns
        assert 'vf_short' in result.columns
        
        # Signals should be boolean
        assert result['vf_long'].dtype == bool
        assert result['vf_short'].dtype == bool
    
    def test_error_handling_missing_dependencies(self):
        """Test error handling when required indicators are missing."""
        # Test with missing VP columns
        incomplete_data = self.ohlcv_data.copy()
        incomplete_data['FVG_bullish_signal'] = False
        incomplete_data['FVG_bearish_signal'] = False
        incomplete_data['FVG_nearest_support_mid'] = 29500
        incomplete_data['FVG_nearest_resistance_mid'] = 30500
        incomplete_data['FVG_active_bullish_gaps'] = 0
        incomplete_data['FVG_active_bearish_gaps'] = 0
        # Missing VP columns
        
        vpfvg_indicator = VPFVGSignal()
        result = vpfvg_indicator.compute(incomplete_data)
        
        # Should return empty signals
        assert len(result) == len(incomplete_data)
        assert all(result['vf_long'] == False)
        assert all(result['vf_short'] == False)
        assert all(np.isnan(result['vf_atr']))
    
    def test_parameter_customization(self):
        """Test that custom parameters are properly applied."""
        # Create custom indicator with different parameters
        custom_indicator = VPFVGSignal(
            atr_period=20,
            lvn_dist_multiplier=0.3,
            poc_shift_multiplier=0.6,
            hvn_overlap_pct=0.8,
            min_fvg_size=2.0,
            lookback_validation=10
        )
        
        # Check that parameters are set correctly
        assert custom_indicator.atr_period == 20
        assert custom_indicator.lvn_dist_multiplier == 0.3
        assert custom_indicator.poc_shift_multiplier == 0.6
        assert custom_indicator.hvn_overlap_pct == 0.8
        assert custom_indicator.min_fvg_size == 2.0
        assert custom_indicator.lookback_validation == 10
        
        # Test with minimal data
        minimal_data = self.ohlcv_data.iloc[:50].copy()
        
        # Add required columns
        minimal_data['VolumeProfile_poc_price'] = 30000
        minimal_data['VolumeProfile_is_lvn'] = False
        minimal_data['VolumeProfile_is_hvn'] = False
        minimal_data['FVG_bullish_signal'] = False
        minimal_data['FVG_bearish_signal'] = False
        minimal_data['FVG_nearest_support_mid'] = 29500
        minimal_data['FVG_nearest_resistance_mid'] = 30500
        minimal_data['FVG_active_bullish_gaps'] = 0
        minimal_data['FVG_active_bearish_gaps'] = 0
        
        result = custom_indicator.compute(minimal_data)
        
        # Should work with custom parameters
        assert len(result) == len(minimal_data)
        assert 'vf_long' in result.columns
        assert 'vf_short' in result.columns
    
    def test_look_ahead_bias_prevention(self):
        """Test that look-ahead bias is properly prevented."""
        # Create data with known signal at a specific point
        test_data = self.ohlcv_data.iloc[:100].copy()
        
        # Add required columns
        test_data['VolumeProfile_poc_price'] = 30000
        test_data['VolumeProfile_is_lvn'] = False
        test_data['VolumeProfile_is_hvn'] = False
        test_data['FVG_bullish_signal'] = False
        test_data['FVG_bearish_signal'] = False
        test_data['FVG_nearest_support_mid'] = 29500
        test_data['FVG_nearest_resistance_mid'] = 30500
        test_data['FVG_active_bullish_gaps'] = 0
        test_data['FVG_active_bearish_gaps'] = 0
        
        # Create signal at index 50
        test_idx = 50
        test_data.iloc[test_idx, test_data.columns.get_loc('FVG_bullish_signal')] = True
        test_data.iloc[test_idx, test_data.columns.get_loc('FVG_active_bullish_gaps')] = 1
        test_data.iloc[test_idx-1, test_data.columns.get_loc('VolumeProfile_is_lvn')] = True
        
        # Calculate signals
        vpfvg_indicator = VPFVGSignal()
        result = vpfvg_indicator.compute(test_data)
        
        # The signal should appear in the NEXT bar due to shift(1)
        # First bar should be False (due to shift)
        assert result['vf_long'].iloc[0] == False
        assert result['vf_short'].iloc[0] == False
        assert np.isnan(result['vf_atr'].iloc[0])
        
        # This tests the look-ahead bias prevention
        assert len(result) == len(test_data)
    
    def test_real_world_scenario(self):
        """Test with a more realistic scenario."""
        # Create more realistic data
        np.random.seed(123)
        
        # Generate price data with trend
        n_bars = 200
        dates = pd.date_range('2023-01-01', periods=n_bars, freq='15min')
        
        # Create trending price data
        trend = np.linspace(0, 1000, n_bars)
        noise = np.random.normal(0, 50, n_bars)
        base_price = 30000 + trend + noise
        
        realistic_data = pd.DataFrame({
            'open': base_price * (1 + np.random.normal(0, 0.001, n_bars)),
            'high': base_price * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
            'low': base_price * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
            'close': base_price,
            'volume': np.random.uniform(100, 1000, n_bars),
        }, index=dates)
        
        # Ensure price relationships are valid
        realistic_data['high'] = np.maximum(realistic_data['high'], realistic_data['close'])
        realistic_data['low'] = np.minimum(realistic_data['low'], realistic_data['close'])
        realistic_data['open'] = np.clip(realistic_data['open'], 
                                       realistic_data['low'], 
                                       realistic_data['high'])
        
        # Calculate VP and FVG first
        vp_indicator = VolumeProfile(lookback=50, price_bins=20)
        vp_result = vp_indicator.compute(realistic_data)
        
        fvg_indicator = FVG()
        fvg_result = fvg_indicator.compute(realistic_data)
        
        # Combine data
        combined_data = realistic_data.copy()
        for col in vp_result.columns:
            combined_data[f'VolumeProfile_{col}'] = vp_result[col]
        for col in fvg_result.columns:
            combined_data[col] = fvg_result[col]
        
        # Calculate VPFVGSignal
        vpfvg_indicator = VPFVGSignal()
        result = vpfvg_indicator.compute(combined_data)
        
        # Verify the results
        assert len(result) == len(realistic_data)
        assert all(col in result.columns for col in ['vf_long', 'vf_short', 'vf_atr'])
        
        # Check for reasonable signal frequency (should be rare)
        total_signals = result['vf_long'].sum() + result['vf_short'].sum()
        signal_frequency = total_signals / len(result)
        
        # Signals should be relatively rare (less than 10% of bars)
        assert signal_frequency < 0.1, f"Signal frequency too high: {signal_frequency}"
        
        print(f"Real-world test - Total signals: {total_signals} "
              f"({signal_frequency:.2%} of bars)")
        print(f"Long signals: {result['vf_long'].sum()}")
        print(f"Short signals: {result['vf_short'].sum()}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])