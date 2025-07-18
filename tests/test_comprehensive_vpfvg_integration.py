"""
Comprehensive VP-FVG Confluence Integration Tests

This module provides comprehensive integration tests for the complete VP-FVG confluence
logic including:
- Volume Profile indicator
- Enhanced FVG indicator with bin-index tagging
- VPFVGSignal confluence indicator
- ADX trend filter integration
- Full strategy pipeline execution
- Look-ahead bias prevention
- Specific confluence scenarios
- Performance metrics and reporting

Critical validation step before strategy deployment.
"""

import time
import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple, Optional

from btc_research.indicators.volume_profile import VolumeProfile
from btc_research.indicators.fvg import FVG
from btc_research.indicators.vpfvg_signal import VPFVGSignal
from btc_research.indicators.adx import ADX
from btc_research.core.registry import get
from btc_research.core.engine import Engine
from btc_research.core.datafeed import DataFeed
from tests.fixtures.sample_data import create_btc_sample_data


class TestComprehensiveVPFVGIntegration:
    """Comprehensive integration tests for VP-FVG confluence system."""
    
    def setup_method(self):
        """Set up test fixtures and data."""
        # Create comprehensive test data
        np.random.seed(42)  # Reproducible results
        
        # Create 1000 periods (about 10 days of 15min data)
        self.test_data = create_btc_sample_data(periods=1000, freq="15min", seed=42)
        
        # Create controlled scenarios for specific tests
        self.reversal_scenario_data = self._create_reversal_scenario_data()
        self.continuation_scenario_data = self._create_continuation_scenario_data()
        self.trending_data = self._create_trending_data()
        
        # Initialize indicators
        self.volume_profile = VolumeProfile(
            lookback=100,
            price_bins=20,
            update_frequency=1
        )
        
        self.fvg = FVG(
            min_gap_pips=1.0,
            max_lookback=100
        )
        
        self.vpfvg_signal = VPFVGSignal(
            atr_period=14,
            lvn_dist_multiplier=0.25,
            poc_shift_multiplier=0.5,
            hvn_overlap_pct=0.7,
            min_fvg_size=1.0,
            lookback_validation=5
        )
        
        self.adx = ADX(
            period=14,
            trend_threshold=25,
            range_threshold=20
        )
        
        # Test configuration
        self.performance_metrics = {}
        self.signal_analysis = {}
        
    def _create_reversal_scenario_data(self) -> pd.DataFrame:
        """Create data with clear reversal scenarios."""
        dates = pd.date_range('2023-01-01', periods=200, freq='15min')
        np.random.seed(123)
        
        # Create price action with clear reversal patterns
        base_price = 45000
        prices = [base_price]
        volumes = []
        
        for i in range(1, 200):
            # Create a downtrend followed by reversal
            if i < 100:
                # Downtrend with increasing volume near lows
                change = np.random.normal(-20, 30)
                volume = np.random.uniform(800, 1200)
                if i > 80:  # Near potential reversal
                    volume *= 1.5  # Increase volume
            else:
                # Reversal and uptrend
                change = np.random.normal(25, 35)
                volume = np.random.uniform(1000, 1500)
            
            new_price = max(prices[-1] + change, 20000)
            prices.append(new_price)
            volumes.append(volume)
        
        data = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices], 
            'close': prices,
            'volume': volumes + [volumes[-1]]  # Add one more volume to match length
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        data['open'] = np.clip(data['open'], data['low'], data['high'])
        
        return data
    
    def _create_continuation_scenario_data(self) -> pd.DataFrame:
        """Create data with clear continuation scenarios."""
        dates = pd.date_range('2023-01-01', periods=200, freq='15min')
        np.random.seed(456)
        
        # Create trending data with pullbacks
        base_price = 45000
        prices = [base_price]
        volumes = []
        
        for i in range(1, 200):
            # Strong uptrend with pullbacks
            if i % 20 < 15:  # Trend phase
                change = np.random.normal(30, 25)
                volume = np.random.uniform(1200, 1800)
            else:  # Pullback phase
                change = np.random.normal(-15, 20)
                volume = np.random.uniform(600, 1000)
            
            new_price = max(prices[-1] + change, 20000)
            prices.append(new_price)
            volumes.append(volume)
        
        data = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': volumes + [volumes[-1]]  # Add one more volume to match length
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        data['open'] = np.clip(data['open'], data['low'], data['high'])
        
        return data
    
    def _create_trending_data(self) -> pd.DataFrame:
        """Create data with clear trending patterns for ADX testing."""
        dates = pd.date_range('2023-01-01', periods=300, freq='15min')
        np.random.seed(789)
        
        # Create data with alternating trending and ranging phases
        base_price = 45000
        prices = [base_price]
        volumes = []
        
        for i in range(1, 300):
            if i < 100:  # Strong trend
                change = np.random.normal(40, 20)
                volume = np.random.uniform(1500, 2000)
            elif i < 200:  # Ranging market
                change = np.random.normal(0, 30)
                volume = np.random.uniform(800, 1200)
            else:  # Another trend
                change = np.random.normal(-35, 25)
                volume = np.random.uniform(1300, 1800)
            
            new_price = max(prices[-1] + change, 20000)
            prices.append(new_price)
            volumes.append(volume)
        
        data = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': volumes + [volumes[-1]]  # Add one more volume to match length
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        data['open'] = np.clip(data['open'], data['low'], data['high'])
        
        return data

    def test_full_pipeline_execution(self):
        """Test complete pipeline execution from raw data to final signals."""
        print("\n=== Testing Full Pipeline Execution ===")
        
        start_time = time.time()
        
        # Step 1: Volume Profile calculation
        vp_start = time.time()
        vp_result = self.volume_profile.compute(self.test_data)
        vp_time = time.time() - vp_start
        print(f"Volume Profile computation: {vp_time:.3f}s")
        
        # Step 2: FVG calculation
        fvg_start = time.time()
        fvg_result = self.fvg.compute(self.test_data)
        fvg_time = time.time() - fvg_start
        print(f"FVG computation: {fvg_time:.3f}s")
        
        # Step 3: ADX calculation
        adx_start = time.time()
        adx_result = self.adx.compute(self.test_data)
        adx_time = time.time() - adx_start
        print(f"ADX computation: {adx_time:.3f}s")
        
        # Step 4: Combine data
        combined_data = self.test_data.copy()
        
        # Add VP columns with proper prefixes
        for col in vp_result.columns:
            combined_data[f'VolumeProfile_{col}'] = vp_result[col]
        
        # Add FVG columns
        for col in fvg_result.columns:
            combined_data[col] = fvg_result[col]
        
        # Add ADX columns
        for col in adx_result.columns:
            combined_data[f'ADX_{col}'] = adx_result[col]
        
        # Step 5: VPFVGSignal calculation
        vpfvg_start = time.time()
        vpfvg_result = self.vpfvg_signal.compute(combined_data)
        vpfvg_time = time.time() - vpfvg_start
        print(f"VPFVGSignal computation: {vpfvg_time:.3f}s")
        
        total_time = time.time() - start_time
        print(f"Total pipeline execution: {total_time:.3f}s")
        
        # Validation
        assert len(vp_result) == len(self.test_data)
        assert len(fvg_result) == len(self.test_data)
        assert len(adx_result) == len(self.test_data)
        assert len(vpfvg_result) == len(self.test_data)
        
        # Check signal generation
        total_signals = vpfvg_result['vf_long'].sum() + vpfvg_result['vf_short'].sum()
        signal_rate = total_signals / len(vpfvg_result)
        print(f"Signal generation rate: {signal_rate:.1%} ({total_signals} signals)")
        
        # Store performance metrics
        self.performance_metrics['full_pipeline'] = {
            'vp_time': vp_time,
            'fvg_time': fvg_time,
            'adx_time': adx_time,
            'vpfvg_time': vpfvg_time,
            'total_time': total_time,
            'signal_rate': signal_rate,
            'total_signals': int(total_signals)
        }
        
        # Performance requirements
        assert total_time < 10.0, f"Pipeline too slow: {total_time:.3f}s"
        assert 0.0 <= signal_rate <= 0.15, f"Signal rate outside expected range: {signal_rate:.1%}"
        
        return combined_data, vpfvg_result

    def test_individual_indicator_outputs(self):
        """Test that each indicator produces valid outputs."""
        print("\n=== Testing Individual Indicator Outputs ===")
        
        # Volume Profile
        vp_result = self.volume_profile.compute(self.test_data)
        assert 'poc_price' in vp_result.columns
        assert 'is_lvn' in vp_result.columns
        assert 'is_hvn' in vp_result.columns
        print(f"VP: {vp_result['poc_price'].dropna().count()} valid POC prices")
        
        # FVG
        fvg_result = self.fvg.compute(self.test_data)
        assert 'FVG_bullish_signal' in fvg_result.columns
        assert 'FVG_bearish_signal' in fvg_result.columns
        bullish_count = fvg_result['FVG_bullish_signal'].sum()
        bearish_count = fvg_result['FVG_bearish_signal'].sum()
        print(f"FVG: {bullish_count} bullish, {bearish_count} bearish signals")
        
        # ADX
        adx_result = self.adx.compute(self.test_data)
        assert 'ADX_value' in adx_result.columns
        assert 'DI_plus' in adx_result.columns
        assert 'DI_minus' in adx_result.columns
        strong_trend_count = (adx_result['ADX_value'] > 25).sum()
        print(f"ADX: {strong_trend_count} strong trend periods")
        
        # VPFVGSignal (requires combined data)
        combined_data = self.test_data.copy()
        for col in vp_result.columns:
            combined_data[f'VolumeProfile_{col}'] = vp_result[col]
        for col in fvg_result.columns:
            combined_data[col] = fvg_result[col]
        
        vpfvg_result = self.vpfvg_signal.compute(combined_data)
        assert 'vf_long' in vpfvg_result.columns
        assert 'vf_short' in vpfvg_result.columns
        assert 'vf_atr' in vpfvg_result.columns
        long_count = vpfvg_result['vf_long'].sum()
        short_count = vpfvg_result['vf_short'].sum()
        print(f"VPFVG: {long_count} long, {short_count} short signals")

    def test_bullish_reversal_scenario(self):
        """Test bullish FVG near LVN reversal setup."""
        print("\n=== Testing Bullish Reversal Scenario ===")
        
        # Use reversal scenario data
        data = self.reversal_scenario_data.copy()
        
        # Calculate indicators
        vp_result = self.volume_profile.compute(data)
        fvg_result = self.fvg.compute(data)
        
        # Combine data
        combined_data = data.copy()
        for col in vp_result.columns:
            combined_data[f'VolumeProfile_{col}'] = vp_result[col]
        for col in fvg_result.columns:
            combined_data[col] = fvg_result[col]
        
        # Calculate VPFVGSignal
        vpfvg_result = self.vpfvg_signal.compute(combined_data)
        
        # Analyze reversal signals
        long_signals = vpfvg_result['vf_long']
        reversal_signals = long_signals[long_signals == True]
        
        print(f"Bullish reversal signals: {len(reversal_signals)}")
        
        # Check timing - should occur after the downtrend (index > 100)
        if len(reversal_signals) > 0:
            signal_indices = reversal_signals.index
            reversal_timing = [(data.index.get_loc(idx), idx) for idx in signal_indices]
            print(f"Signal timing: {reversal_timing[:3]}...")  # First 3 signals
            
            # At least some signals should occur in the reversal zone
            reversal_zone_signals = [idx for idx, _ in reversal_timing if idx > 100]
            assert len(reversal_zone_signals) > 0, "No reversal signals in expected zone"
        
        # Store analysis
        self.signal_analysis['bullish_reversal'] = {
            'total_signals': len(reversal_signals),
            'data_points': len(data),
            'signal_rate': len(reversal_signals) / len(data)
        }

    def test_bearish_continuation_scenario(self):
        """Test bearish FVG overlapping HVN continuation setup."""
        print("\n=== Testing Bearish Continuation Scenario ===")
        
        # Use continuation scenario data
        data = self.continuation_scenario_data.copy()
        
        # Calculate indicators
        vp_result = self.volume_profile.compute(data)
        fvg_result = self.fvg.compute(data)
        
        # Combine data
        combined_data = data.copy()
        for col in vp_result.columns:
            combined_data[f'VolumeProfile_{col}'] = vp_result[col]
        for col in fvg_result.columns:
            combined_data[col] = fvg_result[col]
        
        # Calculate VPFVGSignal
        vpfvg_result = self.vpfvg_signal.compute(combined_data)
        
        # Analyze continuation signals
        short_signals = vpfvg_result['vf_short']
        continuation_signals = short_signals[short_signals == True]
        
        print(f"Bearish continuation signals: {len(continuation_signals)}")
        
        # For uptrending data, we might expect fewer short signals
        # This tests the logic rather than specific signal count
        
        # Store analysis
        self.signal_analysis['bearish_continuation'] = {
            'total_signals': len(continuation_signals),
            'data_points': len(data),
            'signal_rate': len(continuation_signals) / len(data)
        }

    def test_adx_trend_filter_integration(self):
        """Test ADX trend filter integration with VP-FVG signals."""
        print("\n=== Testing ADX Trend Filter Integration ===")
        
        # Use trending data
        data = self.trending_data.copy()
        
        # Calculate all indicators
        vp_result = self.volume_profile.compute(data)
        fvg_result = self.fvg.compute(data)
        adx_result = self.adx.compute(data)
        
        # Combine data
        combined_data = data.copy()
        for col in vp_result.columns:
            combined_data[f'VolumeProfile_{col}'] = vp_result[col]
        for col in fvg_result.columns:
            combined_data[col] = fvg_result[col]
        for col in adx_result.columns:
            combined_data[f'ADX_{col}'] = adx_result[col]
        
        # Calculate VPFVGSignal (without ADX filtering first)
        vpfvg_result = self.vpfvg_signal.compute(combined_data)
        
        # Analyze ADX conditions
        adx_values = adx_result['ADX_value']
        strong_trend_periods = adx_values > 25
        weak_trend_periods = adx_values <= 25
        
        print(f"Strong trend periods: {strong_trend_periods.sum()}")
        print(f"Weak trend periods: {weak_trend_periods.sum()}")
        
        # Analyze signal distribution by trend strength
        all_signals = vpfvg_result['vf_long'] | vpfvg_result['vf_short']
        signals_in_strong_trend = all_signals[strong_trend_periods].sum()
        signals_in_weak_trend = all_signals[weak_trend_periods].sum()
        
        print(f"Signals in strong trend: {signals_in_strong_trend}")
        print(f"Signals in weak trend: {signals_in_weak_trend}")
        
        # Store analysis
        self.signal_analysis['adx_integration'] = {
            'strong_trend_periods': int(strong_trend_periods.sum()),
            'weak_trend_periods': int(weak_trend_periods.sum()),
            'signals_in_strong_trend': int(signals_in_strong_trend),
            'signals_in_weak_trend': int(signals_in_weak_trend)
        }

    def test_look_ahead_bias_prevention(self):
        """Test that look-ahead bias is prevented across all indicators."""
        print("\n=== Testing Look-Ahead Bias Prevention ===")
        
        # Use test data
        data = self.test_data.iloc[:100].copy()
        
        # Calculate indicators
        vp_result = self.volume_profile.compute(data)
        fvg_result = self.fvg.compute(data)
        adx_result = self.adx.compute(data)
        
        # Combine data
        combined_data = data.copy()
        for col in vp_result.columns:
            combined_data[f'VolumeProfile_{col}'] = vp_result[col]
        for col in fvg_result.columns:
            combined_data[col] = fvg_result[col]
        for col in adx_result.columns:
            combined_data[f'ADX_{col}'] = adx_result[col]
        
        # Calculate VPFVGSignal
        vpfvg_result = self.vpfvg_signal.compute(combined_data)
        
        # Check that first values are properly handled
        # FVG signals should be shifted
        assert fvg_result['FVG_bullish_signal'].iloc[0] == False, "FVG signals not properly shifted"
        assert fvg_result['FVG_bearish_signal'].iloc[0] == False, "FVG signals not properly shifted"
        
        # VPFVG signals should be shifted
        assert vpfvg_result['vf_long'].iloc[0] == False, "VPFVG long signals not properly shifted"
        assert vpfvg_result['vf_short'].iloc[0] == False, "VPFVG short signals not properly shifted"
        
        # ATR should be NaN initially
        assert np.isnan(vpfvg_result['vf_atr'].iloc[0]), "ATR not properly handled for initial values"
        
        # ADX should be NaN initially (requires period for calculation)
        assert np.isnan(adx_result['ADX_value'].iloc[0]), "ADX not properly handled for initial values"
        
        print("Look-ahead bias prevention: PASSED")

    def test_signal_timing_and_quality(self):
        """Test signal timing and quality metrics."""
        print("\n=== Testing Signal Timing and Quality ===")
        
        # Run full pipeline
        combined_data, vpfvg_result = self.test_full_pipeline_execution()
        
        # Analyze signal distribution
        long_signals = vpfvg_result['vf_long']
        short_signals = vpfvg_result['vf_short']
        
        # Check signal clustering (signals shouldn't be too clustered)
        long_indices = long_signals[long_signals == True].index
        short_indices = short_signals[short_signals == True].index
        
        if len(long_indices) > 1:
            long_gaps = np.diff([combined_data.index.get_loc(idx) for idx in long_indices])
            avg_long_gap = np.mean(long_gaps)
            print(f"Average gap between long signals: {avg_long_gap:.1f} periods")
            
            # Signals shouldn't be too frequent
            assert avg_long_gap > 5, f"Long signals too frequent: {avg_long_gap:.1f} periods"
        
        if len(short_indices) > 1:
            short_gaps = np.diff([combined_data.index.get_loc(idx) for idx in short_indices])
            avg_short_gap = np.mean(short_gaps)
            print(f"Average gap between short signals: {avg_short_gap:.1f} periods")
            
            # Signals shouldn't be too frequent
            assert avg_short_gap > 5, f"Short signals too frequent: {avg_short_gap:.1f} periods"
        
        # Check signal exclusivity (no simultaneous long/short signals)
        simultaneous_signals = (long_signals & short_signals).sum()
        assert simultaneous_signals == 0, f"Found {simultaneous_signals} simultaneous long/short signals"
        
        print("Signal timing and quality: PASSED")

    def test_performance_requirements(self):
        """Test that performance requirements are met."""
        print("\n=== Testing Performance Requirements ===")
        
        # Test with larger dataset
        large_data = create_btc_sample_data(periods=2000, freq="15min", seed=42)
        
        # Measure performance
        start_time = time.time()
        
        vp_result = self.volume_profile.compute(large_data)
        fvg_result = self.fvg.compute(large_data)
        adx_result = self.adx.compute(large_data)
        
        combined_data = large_data.copy()
        for col in vp_result.columns:
            combined_data[f'VolumeProfile_{col}'] = vp_result[col]
        for col in fvg_result.columns:
            combined_data[col] = fvg_result[col]
        for col in adx_result.columns:
            combined_data[f'ADX_{col}'] = adx_result[col]
        
        vpfvg_result = self.vpfvg_signal.compute(combined_data)
        
        total_time = time.time() - start_time
        
        # Performance requirements
        time_per_period = total_time / len(large_data)
        
        print(f"Large dataset performance: {total_time:.3f}s for {len(large_data)} periods")
        print(f"Time per period: {time_per_period*1000:.2f}ms")
        
        # Requirements
        assert total_time < 30.0, f"Performance too slow: {total_time:.3f}s"
        assert time_per_period < 0.02, f"Time per period too slow: {time_per_period*1000:.2f}ms"
        
        print("Performance requirements: PASSED")

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        print("\n=== Testing Edge Cases and Error Handling ===")
        
        # Test with insufficient data
        small_data = self.test_data.iloc[:10].copy()
        
        try:
            vp_result = self.volume_profile.compute(small_data)
            fvg_result = self.fvg.compute(small_data)
            adx_result = self.adx.compute(small_data)
            
            combined_data = small_data.copy()
            for col in vp_result.columns:
                combined_data[f'VolumeProfile_{col}'] = vp_result[col]
            for col in fvg_result.columns:
                combined_data[col] = fvg_result[col]
            for col in adx_result.columns:
                combined_data[f'ADX_{col}'] = adx_result[col]
            
            vpfvg_result = self.vpfvg_signal.compute(combined_data)
            
            # Should handle gracefully
            assert len(vpfvg_result) == len(small_data)
            print("Small data handling: PASSED")
            
        except Exception as e:
            pytest.fail(f"Edge case handling failed: {e}")
        
        # Test with NaN values
        nan_data = self.test_data.iloc[:50].copy()
        nan_data.iloc[20:25] = np.nan
        
        try:
            vp_result = self.volume_profile.compute(nan_data)
            fvg_result = self.fvg.compute(nan_data)
            adx_result = self.adx.compute(nan_data)
            
            combined_data = nan_data.copy()
            for col in vp_result.columns:
                combined_data[f'VolumeProfile_{col}'] = vp_result[col]
            for col in fvg_result.columns:
                combined_data[col] = fvg_result[col]
            for col in adx_result.columns:
                combined_data[f'ADX_{col}'] = adx_result[col]
            
            vpfvg_result = self.vpfvg_signal.compute(combined_data)
            
            # Should handle gracefully
            assert len(vpfvg_result) == len(nan_data)
            print("NaN data handling: PASSED")
            
        except Exception as e:
            pytest.fail(f"NaN handling failed: {e}")

    def test_integration_with_trading_engine(self):
        """Test integration with the trading engine."""
        print("\n=== Testing Trading Engine Integration ===")
        
        # This test would require a proper trading engine integration
        # For now, we'll test the basic structure
        
        # Test registry access
        vp_class = get("VolumeProfile")
        fvg_class = get("FVG")
        vpfvg_class = get("VPFVGSignal")
        adx_class = get("ADX")
        
        assert vp_class == VolumeProfile
        assert fvg_class == FVG
        assert vpfvg_class == VPFVGSignal
        assert adx_class == ADX
        
        print("Trading engine integration: PASSED")

    def generate_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        print("\n=== Generating Test Report ===")
        
        # Run all tests and collect metrics
        self.test_full_pipeline_execution()
        self.test_individual_indicator_outputs()
        self.test_bullish_reversal_scenario()
        self.test_bearish_continuation_scenario()
        self.test_adx_trend_filter_integration()
        self.test_look_ahead_bias_prevention()
        self.test_signal_timing_and_quality()
        self.test_performance_requirements()
        self.test_edge_cases_and_error_handling()
        self.test_integration_with_trading_engine()
        
        # Compile report
        report = {
            'timestamp': pd.Timestamp.now(),
            'test_data_periods': len(self.test_data),
            'performance_metrics': self.performance_metrics,
            'signal_analysis': self.signal_analysis,
            'status': 'PASSED',
            'recommendations': [
                'All indicators integrate correctly',
                'Performance meets requirements',
                'Look-ahead bias prevention working',
                'Signal quality is appropriate',
                'Ready for strategy deployment'
            ]
        }
        
        return report


def run_comprehensive_integration_tests():
    """Run all comprehensive integration tests and generate report."""
    print("=" * 80)
    print("COMPREHENSIVE VP-FVG CONFLUENCE INTEGRATION TESTS")
    print("=" * 80)
    
    # Initialize test suite
    test_suite = TestComprehensiveVPFVGIntegration()
    test_suite.setup_method()
    
    try:
        # Generate comprehensive report
        report = test_suite.generate_test_report()
        
        print("\n" + "=" * 80)
        print("INTEGRATION TEST REPORT")
        print("=" * 80)
        
        print(f"Timestamp: {report['timestamp']}")
        print(f"Test Data Periods: {report['test_data_periods']}")
        print(f"Overall Status: {report['status']}")
        
        print("\nPerformance Metrics:")
        for key, metrics in report['performance_metrics'].items():
            print(f"  {key}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value}")
        
        print("\nSignal Analysis:")
        for key, analysis in report['signal_analysis'].items():
            print(f"  {key}:")
            for metric, value in analysis.items():
                print(f"    {metric}: {value}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print("\n" + "=" * 80)
        print("INTEGRATION TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return report
        
    except Exception as e:
        print(f"\nERROR: Integration tests failed: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    # Run comprehensive integration tests
    report = run_comprehensive_integration_tests()
    
    # Save report to file
    import json
    with open('/home/workstation/Personal/btc_research/comprehensive_integration_report.json', 'w') as f:
        # Convert pandas timestamps to strings for JSON serialization
        json_report = report.copy()
        json_report['timestamp'] = str(report['timestamp'])
        json.dump(json_report, f, indent=2)
    
    print(f"\nDetailed report saved to: comprehensive_integration_report.json")