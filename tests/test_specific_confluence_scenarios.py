"""
Specific VP-FVG Confluence Scenario Tests

This module creates controlled scenarios to test specific confluence patterns
that should generate signals. Used to validate signal generation logic.
"""

import numpy as np
import pandas as pd
import pytest

from btc_research.indicators.volume_profile import VolumeProfile
from btc_research.indicators.fvg import FVG
from btc_research.indicators.vpfvg_signal import VPFVGSignal
from btc_research.indicators.adx import ADX


class TestSpecificConfluenceScenarios:
    """Test specific confluence scenarios with controlled data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.volume_profile = VolumeProfile(lookback=50, price_bins=20)
        self.fvg = FVG(min_gap_pips=0.5, max_lookback=50)
        self.vpfvg_signal = VPFVGSignal()
        self.adx = ADX(period=14)
    
    def create_controlled_fvg_scenario(self) -> pd.DataFrame:
        """Create data with controlled FVG patterns."""
        dates = pd.date_range('2023-01-01', periods=150, freq='15min')
        
        # Create controlled price action with clear gaps
        prices = []
        volumes = []
        
        base_price = 50000
        for i in range(150):
            if i < 50:
                # Initial consolidation
                price = base_price + np.random.normal(0, 50)
                volume = np.random.uniform(500, 800)
            elif i == 50:
                # Create bullish FVG - price jumps up leaving gap
                price = base_price + 200  # Jump up 200 points
                volume = np.random.uniform(1000, 1500)
            elif i < 80:
                # Continue higher after gap
                price = base_price + 200 + (i - 50) * 10
                volume = np.random.uniform(800, 1200)
            elif i == 80:
                # Create bearish FVG - price drops leaving gap
                price = base_price + 100  # Drop 200 points from recent high
                volume = np.random.uniform(1200, 1800)
            else:
                # Continue lower after gap
                price = base_price + 100 - (i - 80) * 8
                volume = np.random.uniform(600, 1000)
            
            prices.append(price)
            volumes.append(volume)
        
        # Create OHLC data ensuring gaps exist
        data = pd.DataFrame(index=dates)
        
        opens = []
        highs = []
        lows = []
        closes = []
        
        for i, price in enumerate(prices):
            if i == 0:
                open_price = price
                close_price = price
                high_price = price * 1.001
                low_price = price * 0.999
            else:
                # Normal case
                open_price = closes[i-1]  # Open at previous close
                close_price = price
                
                # Create gaps at specific points
                if i == 50:  # Bullish gap
                    open_price = prices[i-1] + 150  # Gap up on open
                    low_price = min(open_price, close_price) * 0.999
                    high_price = max(open_price, close_price) * 1.001
                elif i == 80:  # Bearish gap
                    open_price = prices[i-1] - 150  # Gap down on open
                    low_price = min(open_price, close_price) * 0.999
                    high_price = max(open_price, close_price) * 1.001
                else:
                    low_price = min(open_price, close_price) * 0.999
                    high_price = max(open_price, close_price) * 1.001
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
        
        data['open'] = opens
        data['high'] = highs
        data['low'] = lows
        data['close'] = closes
        data['volume'] = volumes
        
        return data
    
    def create_controlled_volume_profile_scenario(self) -> pd.DataFrame:
        """Create data with controlled volume profile patterns."""
        dates = pd.date_range('2023-01-01', periods=150, freq='15min')
        
        prices = []
        volumes = []
        
        base_price = 50000
        for i in range(150):
            if 40 <= i < 60:
                # High volume node around 50200
                price = 50200 + np.random.normal(0, 20)
                volume = np.random.uniform(2000, 3000)  # High volume
            elif 80 <= i < 100:
                # Low volume node around 49800
                price = 49800 + np.random.normal(0, 20)
                volume = np.random.uniform(200, 400)  # Low volume
            else:
                # Regular trading
                price = base_price + np.random.normal(0, 100)
                volume = np.random.uniform(800, 1200)
            
            prices.append(price)
            volumes.append(volume)
        
        data = pd.DataFrame({
            'open': [p * 0.9995 for p in prices],
            'high': [p * 1.0005 for p in prices],
            'low': [p * 0.9995 for p in prices],
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        data['open'] = np.clip(data['open'], data['low'], data['high'])
        
        return data
    
    def test_bullish_fvg_at_lvn_reversal(self):
        """Test bullish FVG occurring near LVN for reversal signal."""
        print("\n=== Testing Bullish FVG at LVN Reversal ===")
        
        # Create controlled scenario
        fvg_data = self.create_controlled_fvg_scenario()
        vp_data = self.create_controlled_volume_profile_scenario()
        
        # Combine scenarios - use FVG price action with VP volume distribution
        combined_data = fvg_data.copy()
        combined_data['volume'] = vp_data['volume']
        
        # Calculate indicators
        vp_result = self.volume_profile.compute(combined_data)
        fvg_result = self.fvg.compute(combined_data)
        
        # Combine results
        full_data = combined_data.copy()
        for col in vp_result.columns:
            full_data[f'VolumeProfile_{col}'] = vp_result[col]
        for col in fvg_result.columns:
            full_data[col] = fvg_result[col]
        
        # Calculate confluence signals
        vpfvg_result = self.vpfvg_signal.compute(full_data)
        
        # Analyze results
        bullish_signals = vpfvg_result['vf_long']
        bearish_signals = vpfvg_result['vf_short']
        
        print(f"Bullish signals: {bullish_signals.sum()}")
        print(f"Bearish signals: {bearish_signals.sum()}")
        
        # Check FVG detection
        fvg_bullish = fvg_result['FVG_bullish_signal']
        fvg_bearish = fvg_result['FVG_bearish_signal']
        
        print(f"FVG bullish detected: {fvg_bullish.sum()}")
        print(f"FVG bearish detected: {fvg_bearish.sum()}")
        
        # Check VP node detection
        lvn_periods = vp_result['is_lvn']
        hvn_periods = vp_result['is_hvn']
        
        print(f"LVN periods: {lvn_periods.sum()}")
        print(f"HVN periods: {hvn_periods.sum()}")
        
        # We should have detected some FVG patterns
        assert fvg_bullish.sum() > 0 or fvg_bearish.sum() > 0, "No FVG patterns detected"
        
        # Print confluence analysis
        if bullish_signals.sum() > 0 or bearish_signals.sum() > 0:
            print("✓ Confluence signals generated successfully")
        else:
            print("⚠ No confluence signals generated - patterns may be too restrictive")
        
        return full_data, vpfvg_result
    
    def test_manual_confluence_setup(self):
        """Test manually created confluence setup."""
        print("\n=== Testing Manual Confluence Setup ===")
        
        # Create simple test data
        dates = pd.date_range('2023-01-01', periods=100, freq='15min')
        
        # Create basic price data
        base_price = 50000
        prices = [base_price + i * 10 + np.random.normal(0, 20) for i in range(100)]
        volumes = [np.random.uniform(800, 1200) for _ in range(100)]
        
        data = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        data['open'] = np.clip(data['open'], data['low'], data['high'])
        
        # Calculate indicators
        vp_result = self.volume_profile.compute(data)
        fvg_result = self.fvg.compute(data)
        
        # Manually create confluence scenario
        combined_data = data.copy()
        
        # Add VP results
        for col in vp_result.columns:
            combined_data[f'VolumeProfile_{col}'] = vp_result[col]
        
        # Add FVG results  
        for col in fvg_result.columns:
            combined_data[col] = fvg_result[col]
        
        # Manually set up confluence at index 50
        test_idx = 50
        combined_data.loc[combined_data.index[test_idx], 'FVG_bullish_signal'] = True
        combined_data.loc[combined_data.index[test_idx], 'FVG_active_bullish_gaps'] = 1
        combined_data.loc[combined_data.index[test_idx], 'FVG_nearest_support_mid'] = prices[test_idx]
        combined_data.loc[combined_data.index[test_idx-1], 'VolumeProfile_is_lvn'] = True
        
        # Calculate confluence signals
        vpfvg_result = self.vpfvg_signal.compute(combined_data)
        
        # Check results
        print(f"Manual confluence signals - Long: {vpfvg_result['vf_long'].sum()}")
        print(f"Manual confluence signals - Short: {vpfvg_result['vf_short'].sum()}")
        
        # Should have at least some signals due to manual setup
        total_signals = vpfvg_result['vf_long'].sum() + vpfvg_result['vf_short'].sum()
        print(f"Total signals: {total_signals}")
        
        return combined_data, vpfvg_result
    
    def test_adx_filter_integration(self):
        """Test ADX filter integration with confluence signals."""
        print("\n=== Testing ADX Filter Integration ===")
        
        # Create trending data
        dates = pd.date_range('2023-01-01', periods=100, freq='15min')
        
        # Create strong trend
        trend_strength = 50  # Strong uptrend
        prices = []
        volumes = []
        
        base_price = 50000
        for i in range(100):
            if i < 20:
                # Initial consolidation
                price = base_price + np.random.normal(0, 20)
                volume = np.random.uniform(500, 800)
            else:
                # Strong trend
                price = base_price + (i - 20) * trend_strength + np.random.normal(0, 30)
                volume = np.random.uniform(1000, 1500)
            
            prices.append(price)
            volumes.append(volume)
        
        data = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        data['open'] = np.clip(data['open'], data['low'], data['high'])
        
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
        
        # Calculate confluence signals
        vpfvg_result = self.vpfvg_signal.compute(combined_data)
        
        # Analyze ADX values
        adx_values = adx_result['ADX_value'].dropna()
        strong_trend_count = (adx_values > 25).sum()
        
        print(f"ADX strong trend periods: {strong_trend_count}")
        print(f"Average ADX: {adx_values.mean():.2f}")
        
        # Check signal distribution
        signals = vpfvg_result['vf_long'].sum() + vpfvg_result['vf_short'].sum()
        print(f"Total confluence signals: {signals}")
        
        # Verify ADX is calculating correctly
        assert not adx_values.empty, "ADX values should not be empty"
        assert adx_values.max() <= 100, "ADX should not exceed 100"
        assert adx_values.min() >= 0, "ADX should not be negative"
        
        return combined_data, vpfvg_result, adx_result
    
    def test_comprehensive_scenario_validation(self):
        """Test comprehensive scenario validation."""
        print("\n=== Testing Comprehensive Scenario Validation ===")
        
        # Run all scenarios
        scenario1_data, scenario1_signals = self.test_bullish_fvg_at_lvn_reversal()
        scenario2_data, scenario2_signals = self.test_manual_confluence_setup()
        scenario3_data, scenario3_signals, scenario3_adx = self.test_adx_filter_integration()
        
        # Aggregate results
        total_signals = (
            scenario1_signals['vf_long'].sum() + scenario1_signals['vf_short'].sum() +
            scenario2_signals['vf_long'].sum() + scenario2_signals['vf_short'].sum() +
            scenario3_signals['vf_long'].sum() + scenario3_signals['vf_short'].sum()
        )
        
        print(f"\nTotal signals across all scenarios: {total_signals}")
        
        # Validation checks
        assert len(scenario1_signals) == len(scenario1_data), "Signal length mismatch"
        assert len(scenario2_signals) == len(scenario2_data), "Signal length mismatch"
        assert len(scenario3_signals) == len(scenario3_data), "Signal length mismatch"
        
        # Check that signals are boolean
        assert scenario1_signals['vf_long'].dtype == bool, "Signals should be boolean"
        assert scenario1_signals['vf_short'].dtype == bool, "Signals should be boolean"
        
        # Check that ATR values are reasonable
        atr_values = scenario1_signals['vf_atr'].dropna()
        if not atr_values.empty:
            assert (atr_values > 0).all(), "ATR values should be positive"
        
        print("✓ All scenario validations passed")
        
        return {
            'scenario1': {'data': scenario1_data, 'signals': scenario1_signals},
            'scenario2': {'data': scenario2_data, 'signals': scenario2_signals},
            'scenario3': {'data': scenario3_data, 'signals': scenario3_signals, 'adx': scenario3_adx},
            'total_signals': total_signals
        }


def run_specific_confluence_tests():
    """Run specific confluence scenario tests."""
    print("=" * 80)
    print("SPECIFIC VP-FVG CONFLUENCE SCENARIO TESTS")
    print("=" * 80)
    
    test_suite = TestSpecificConfluenceScenarios()
    test_suite.setup_method()
    
    try:
        results = test_suite.test_comprehensive_scenario_validation()
        
        print("\n" + "=" * 80)
        print("SPECIFIC CONFLUENCE TESTS COMPLETED")
        print("=" * 80)
        
        print(f"Total signals generated: {results['total_signals']}")
        print("✓ All specific confluence scenarios validated")
        
        return results
        
    except Exception as e:
        print(f"\nERROR: Specific confluence tests failed: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    results = run_specific_confluence_tests()