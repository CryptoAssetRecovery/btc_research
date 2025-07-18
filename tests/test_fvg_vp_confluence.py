"""
Test suite for FVG-VP confluence analysis functionality.

This test suite validates the integration between Fair Value Gap (FVG) and 
Volume Profile (VP) indicators for confluence analysis.
"""

import numpy as np
import pandas as pd
import pytest
from btc_research.indicators.fvg import FVG
from btc_research.indicators.volume_profile import VolumeProfile


class TestFVGVPConfluence:
    """Test FVG-VP confluence analysis functionality."""

    def create_test_data(self, n_periods=50):
        """Create test data with intentional gaps for testing."""
        dates = pd.date_range('2023-01-01', periods=n_periods, freq='15min')
        np.random.seed(42)
        
        data = []
        base_price = 50000
        
        for i in range(n_periods):
            if i == 0:
                open_price = base_price
            else:
                open_price = data[-1]['close']
            
            # Add intentional gaps
            if i == 10:  # Bullish gap
                open_price += 150
            elif i == 30:  # Bearish gap  
                open_price -= 100
            
            # Random price movement
            price_change = np.random.uniform(-30, 30)
            high_price = open_price + np.random.uniform(0, 50)
            low_price = open_price - np.random.uniform(0, 50)
            close_price = open_price + price_change
            volume = np.random.uniform(500, 1500)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def test_fvg_with_vp_bins(self):
        """Test FVG indicator with VP bin information."""
        df = self.create_test_data()
        
        # Create VP bin edges
        price_min = df['low'].min()
        price_max = df['high'].max()
        vp_bin_edges = np.linspace(price_min, price_max, 21)  # 20 bins
        vp_bin_centers = (vp_bin_edges[:-1] + vp_bin_edges[1:]) / 2
        
        # Test FVG with VP bins
        fvg = FVG(min_gap_pips=10.0, max_lookback=500)
        result = fvg.compute(df, vp_bin_edges=vp_bin_edges, vp_bin_centers=vp_bin_centers)
        
        # Verify new columns exist
        expected_columns = ['FVG_bullish_bin_idx', 'FVG_bearish_bin_idx', 'FVG_lvn_dist', 'FVG_hvn_dist']
        for col in expected_columns:
            assert col in result.columns, f"Column {col} missing from result"
        
        # Verify bin indices are within valid range
        valid_bullish_bins = result['FVG_bullish_bin_idx'] >= 0
        valid_bearish_bins = result['FVG_bearish_bin_idx'] >= 0
        
        if valid_bullish_bins.any():
            max_bullish_bin = result.loc[valid_bullish_bins, 'FVG_bullish_bin_idx'].max()
            assert max_bullish_bin < len(vp_bin_edges) - 1, "Bullish bin index out of range"
        
        if valid_bearish_bins.any():
            max_bearish_bin = result.loc[valid_bearish_bins, 'FVG_bearish_bin_idx'].max()
            assert max_bearish_bin < len(vp_bin_edges) - 1, "Bearish bin index out of range"

    def test_fvg_without_vp_bins(self):
        """Test FVG indicator without VP bin information (backward compatibility)."""
        df = self.create_test_data()
        
        # Test FVG without VP bins
        fvg = FVG()
        result = fvg.compute(df)
        
        # Verify new columns exist but with default values
        assert 'FVG_bullish_bin_idx' in result.columns
        assert 'FVG_bearish_bin_idx' in result.columns
        assert 'FVG_lvn_dist' in result.columns
        assert 'FVG_hvn_dist' in result.columns
        
        # Verify bin indices are -1 (no VP bins provided)
        assert (result['FVG_bullish_bin_idx'] == -1).all()
        assert (result['FVG_bearish_bin_idx'] == -1).all()

    def test_vp_fvg_integration(self):
        """Test full VP-FVG integration."""
        df = self.create_test_data()
        
        # Create VP indicator
        vp = VolumeProfile(lookback=20, price_bins=15)
        vp_result = vp.compute(df)
        
        # Create VP bin edges for FVG
        price_min = df['low'].min()
        price_max = df['high'].max()
        vp_bin_edges = np.linspace(price_min, price_max, 16)  # 15 bins
        vp_bin_centers = (vp_bin_edges[:-1] + vp_bin_edges[1:]) / 2
        
        # Create FVG indicator with VP bins
        fvg = FVG(min_gap_pips=5.0)
        fvg_result = fvg.compute(df, vp_bin_edges=vp_bin_edges, vp_bin_centers=vp_bin_centers)
        
        # Verify both indicators produced results
        assert vp_result.shape[0] == df.shape[0]
        assert fvg_result.shape[0] == df.shape[0]
        
        # Verify confluence analysis is possible
        combined_data = pd.concat([
            vp_result[['poc_price', 'vah_price', 'val_price']],
            fvg_result[['FVG_bullish_bin_idx', 'FVG_bearish_bin_idx']]
        ], axis=1)
        
        assert combined_data.shape[0] == df.shape[0]
        assert not combined_data.empty

    def test_bin_index_calculation(self):
        """Test bin index calculation logic."""
        # Create simple test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='15min'),
            'open': [50000, 50010, 50020, 50030, 50040],
            'high': [50050, 50060, 50070, 50080, 50090],
            'low': [49950, 49960, 49970, 49980, 49990],
            'close': [50025, 50035, 50045, 50055, 50065],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }).set_index('timestamp')
        
        # Create VP bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        vp_bin_edges = np.linspace(price_min, price_max, 6)  # 5 bins
        vp_bin_centers = (vp_bin_edges[:-1] + vp_bin_edges[1:]) / 2
        
        # Test FVG bin index calculation
        fvg = FVG(min_gap_pips=1.0)
        
        # Test _get_bin_index method directly
        fvg.vp_bin_edges = vp_bin_edges
        
        # Test with a price in the middle
        mid_price = (price_min + price_max) / 2
        bin_idx = fvg._get_bin_index(mid_price)
        assert 0 <= bin_idx < len(vp_bin_edges) - 1
        
        # Test with edge prices
        min_bin_idx = fvg._get_bin_index(price_min)
        max_bin_idx = fvg._get_bin_index(price_max)
        assert min_bin_idx == 0
        assert max_bin_idx == len(vp_bin_edges) - 2

    def test_fvg_params_updated(self):
        """Test that FVG params include VP bin parameters."""
        params = FVG.params()
        
        assert 'vp_bin_edges' in params
        assert 'vp_bin_centers' in params
        assert params['vp_bin_edges'] is None
        assert params['vp_bin_centers'] is None

    def test_fvg_initialization_with_vp_bins(self):
        """Test FVG initialization with VP bin parameters."""
        vp_bin_edges = np.array([1, 2, 3, 4, 5])
        vp_bin_centers = np.array([1.5, 2.5, 3.5, 4.5])
        
        fvg = FVG(min_gap_pips=2.0, vp_bin_edges=vp_bin_edges, vp_bin_centers=vp_bin_centers)
        
        assert fvg.min_gap_pips == 2.0
        assert np.array_equal(fvg.vp_bin_edges, vp_bin_edges)
        assert np.array_equal(fvg.vp_bin_centers, vp_bin_centers)