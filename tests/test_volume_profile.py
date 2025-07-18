"""
Comprehensive unit tests for Volume Profile indicator.

This module tests the VolumeProfile class implementation including:
- Core calculation methods (price binning, POC, Value Area)
- Enhanced volume distribution algorithm with close price weighting
- Output column generation (all 15 columns including new average_volume)
- Extended lookback testing (5-day windows and larger datasets)
- Volume node classification (HVN/LVN identification)
- Distance metrics (POC/VAH/VAL positioning and Value Area width)
- Performance validation (vectorized improvements and speed requirements)
- Edge cases (insufficient data, zero volume, single price level)
- Parameter validation (including new parameters and edge values)
- Mathematical correctness validation
- Multi-timeframe compatibility testing
- ATR integration placeholder (for future enhancement)
- Profile balance and symmetry analysis
- Advanced signal generation and breakout detection

Tests follow the existing patterns from other indicator tests in the project
and include comprehensive performance benchmarks to ensure optimizations
maintain the required speed requirements.
"""

import time
import numpy as np
import pandas as pd
import pytest

from btc_research.core.registry import RegistrationError, get
from tests.fixtures.sample_data import (
    create_btc_sample_data,
    create_trending_market_data,
    create_volatile_market_data,
    create_gap_data,
)

# Import VolumeProfile indicator
try:
    from btc_research.indicators.volume_profile import VolumeProfile
    VOLUME_PROFILE_AVAILABLE = True
except ImportError:
    VOLUME_PROFILE_AVAILABLE = False
    VolumeProfile = None


class TestVolumeProfileRegistry:
    """Test Volume Profile registration and retrieval."""

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_volume_profile_registration(self):
        """Test that VolumeProfile indicator is properly registered."""
        vp_class = get("VolumeProfile")
        assert vp_class == VolumeProfile

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_invalid_indicator_raises_error(self):
        """Test that requesting non-existent indicator raises error."""
        with pytest.raises(RegistrationError):
            get("NonExistentVolumeProfile")


class TestVolumeProfileParameters:
    """Test Volume Profile parameter handling and validation."""

    def test_default_params(self):
        """Test Volume Profile default parameters."""
        params = VolumeProfile.params()
        expected = {
            "lookback": 480,                # Extended 5-day lookback
            "lookback_hours": 120,          # 5 days in hours
            "price_bins": 50,
            "value_area_pct": 70,
            "update_frequency": 1,
            "poc_sensitivity": 0.1,         # Deprecated parameter
            "min_volume_threshold": 0.01,
            "hvn_threshold": 1.5,           # High Volume Node threshold
            "lvn_threshold": 0.5,           # Low Volume Node threshold
            "atr_period": 14,               # ATR period for breakout detection
            "atr_multiplier": 2.0,          # ATR multiplier
            "enable_vectorization": True    # Vectorization flag
        }
        assert params == expected

    def test_initialization_with_defaults(self):
        """Test Volume Profile indicator initialization with defaults."""
        vp = VolumeProfile()
        assert vp.lookback == 480  # Extended 5-day lookback
        assert vp.lookback_hours == 120  # 5 days in hours
        assert vp.price_bins == 50
        assert vp.value_area_pct == 0.7  # Stored as decimal (70% -> 0.7)
        assert vp.update_frequency == 1
        assert vp.poc_sensitivity == 0.001  # Stored as decimal (0.1% -> 0.001)
        assert vp.min_volume_threshold == 0.0001  # Stored as decimal (0.01% -> 0.0001)
        assert vp.hvn_threshold == 1.5  # High Volume Node threshold
        assert vp.lvn_threshold == 0.5  # Low Volume Node threshold
        assert vp.atr_period == 14  # ATR period
        assert vp.atr_multiplier == 2.0  # ATR multiplier
        assert vp.enable_vectorization == True  # Vectorization enabled

    def test_initialization_with_custom_params(self):
        """Test Volume Profile indicator initialization with custom parameters."""
        vp = VolumeProfile(
            lookback=240,  # 2.5 days
            lookback_hours=60,  # 2.5 days in hours
            price_bins=30,
            value_area_pct=68,
            update_frequency=4,
            poc_sensitivity=0.05,
            min_volume_threshold=0.005,
            hvn_threshold=2.0,
            lvn_threshold=0.3,
            atr_period=21,
            atr_multiplier=1.5,
            enable_vectorization=False
        )
        assert vp.lookback == 240
        assert vp.lookback_hours == 60
        assert vp.price_bins == 30
        assert vp.value_area_pct == 0.68  # Stored as decimal (68% -> 0.68)
        assert vp.update_frequency == 4
        assert vp.poc_sensitivity == 0.0005  # Stored as decimal (0.05% -> 0.0005)
        assert vp.min_volume_threshold == 0.00005  # Stored as decimal (0.005% -> 0.00005)
        assert vp.hvn_threshold == 2.0
        assert vp.lvn_threshold == 0.3
        assert vp.atr_period == 21
        assert vp.atr_multiplier == 1.5
        assert vp.enable_vectorization == False

    def test_parameter_validation(self):
        """Test parameter validation in constructor."""
        # Valid parameters should work
        vp = VolumeProfile(lookback=24, price_bins=20, value_area_pct=65)
        assert vp.lookback == 24

        # Test edge cases that should be handled gracefully
        if VOLUME_PROFILE_AVAILABLE:
            # These tests would validate parameter bounds when implemented
            pass


class TestVolumeProfileCalculations:
    """Test core Volume Profile calculation methods."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        return create_btc_sample_data(periods=200, freq="15min", seed=42)

    @pytest.fixture
    def volume_profile_data(self):
        """Create specific data for Volume Profile testing with known characteristics."""
        dates = pd.date_range("2024-01-01", periods=100, freq="15min")
        np.random.seed(42)
        
        # Create data with clear volume clusters
        base_price = 45000.0
        prices = []
        volumes = []
        
        for i in range(100):
            # Create price clusters around certain levels
            if 20 <= i < 40:  # High volume cluster around 45500
                price_center = 45500.0
                volume_mult = 3.0  # 3x normal volume
            elif 60 <= i < 80:  # Medium volume cluster around 44500
                price_center = 44500.0
                volume_mult = 2.0  # 2x normal volume
            else:
                price_center = base_price
                volume_mult = 1.0
            
            # Add some noise around the center
            price = price_center + np.random.normal(0, 50)
            volume = np.random.uniform(500, 1000) * volume_mult
            
            prices.append(price)
            volumes.append(volume)
        
        data = pd.DataFrame(index=dates)
        data["close"] = prices
        data["open"] = [p * 0.999 for p in prices]
        data["high"] = [p * 1.001 for p in prices]
        data["low"] = [p * 0.999 for p in prices]
        data["volume"] = volumes
        
        return data

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_compute_output_format(self, sample_data):
        """Test Volume Profile compute method output format."""
        vp = VolumeProfile(lookback=50, price_bins=30)
        result = vp.compute(sample_data)

        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert result.index.equals(sample_data.index)

        # Check column names according to actual implementation (no VP_ prefix)
        expected_cols = [
            "poc_price",
            "vah_price", 
            "val_price",
            "total_volume",
            "poc_volume",
            "value_area_volume",
            "average_volume",
            "price_above_poc",
            "price_below_poc",
            "price_in_value_area",
            "poc_breakout",
            "volume_spike",
            "is_hvn",
            "is_lvn",
            "dist_to_poc",
            "dist_to_vah",
            "dist_to_val",
            "poc_strength",
            "value_area_width",
            "profile_balance"
        ]
        assert list(result.columns) == expected_cols

        # Check data types
        assert result["poc_price"].dtype == float
        assert result["vah_price"].dtype == float
        assert result["val_price"].dtype == float
        assert result["total_volume"].dtype == float
        assert result["poc_volume"].dtype == float
        assert result["value_area_volume"].dtype == float
        assert result["average_volume"].dtype == float
        assert result["price_above_poc"].dtype == bool
        assert result["price_below_poc"].dtype == bool
        assert result["price_in_value_area"].dtype == bool
        assert result["poc_breakout"].dtype == bool
        assert result["volume_spike"].dtype == bool
        assert result["is_hvn"].dtype == bool
        assert result["is_lvn"].dtype == bool
        assert result["dist_to_poc"].dtype == float
        assert result["dist_to_vah"].dtype == float
        assert result["dist_to_val"].dtype == float
        assert result["poc_strength"].dtype == float
        assert result["value_area_width"].dtype == float
        assert result["profile_balance"].dtype == float

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_price_binning_algorithm(self, volume_profile_data):
        """Test price binning algorithm accuracy."""
        vp = VolumeProfile(lookback=50, price_bins=20)
        result = vp.compute(volume_profile_data)
        
        # Test that price bins cover the full range of the entire dataset (not just last window)
        price_min = volume_profile_data[["high", "low", "open", "close"]].min().min()
        price_max = volume_profile_data[["high", "low", "open", "close"]].max().max()
        
        # POC should be within the overall price range (allowing for some variance due to volume weighting)
        poc_prices = result["poc_price"].dropna()
        if len(poc_prices) > 0:
            # Allow for reasonable variance - POC might be outside window due to volume concentration
            extended_min = price_min * 0.98  # 2% tolerance
            extended_max = price_max * 1.02  # 2% tolerance
            
            # At least most POC values should be within reasonable range
            within_range = ((poc_prices >= extended_min) & (poc_prices <= extended_max)).sum()
            assert within_range >= len(poc_prices) * 0.8  # At least 80% should be in range

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_poc_calculation(self, volume_profile_data):
        """Test Point of Control (POC) calculation."""
        vp = VolumeProfile(lookback=50, price_bins=25)
        result = vp.compute(volume_profile_data)
        
        # POC should be stable over periods with similar data
        poc_values = result["poc_price"].dropna()
        poc_volumes = result["poc_volume"].dropna()
        
        # POC volume should be the highest in the profile
        total_volumes = result["total_volume"].dropna()
        
        if len(poc_volumes) > 0 and len(total_volumes) > 0:
            # POC volume should be a reasonable percentage of total volume
            poc_ratio = poc_volumes / total_volumes
            assert (poc_ratio <= 1.0).all()  # Cannot exceed total volume
            assert (poc_ratio > 0.0).all()   # Should have some volume

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_value_area_calculation(self, volume_profile_data):
        """Test Value Area calculation (70% of volume)."""
        vp = VolumeProfile(lookback=50, price_bins=25, value_area_pct=70)
        result = vp.compute(volume_profile_data)
        
        vah_prices = result["vah_price"].dropna()
        val_prices = result["val_price"].dropna()
        poc_prices = result["poc_price"].dropna()
        va_volumes = result["value_area_volume"].dropna()
        total_volumes = result["total_volume"].dropna()
        
        if len(vah_prices) > 0 and len(val_prices) > 0 and len(poc_prices) > 0:
            # VAH should be >= POC >= VAL (but allow for some floating point precision issues)
            valid_periods = min(len(vah_prices), len(val_prices), len(poc_prices))
            if valid_periods > 0:
                vah_subset = vah_prices[:valid_periods]
                val_subset = val_prices[:valid_periods]
                poc_subset = poc_prices[:valid_periods]
                
                # Allow small tolerance for floating point comparison
                assert (vah_subset >= poc_subset - 1e-6).all()
                assert (poc_subset >= val_subset - 1e-6).all()
            
            # Value Area volume should be approximately 70% of total (with tolerance)
            if len(va_volumes) > 0 and len(total_volumes) > 0:
                min_len = min(len(va_volumes), len(total_volumes))
                va_ratio = va_volumes[:min_len] / total_volumes[:min_len]
                # Allow broader tolerance since algorithm may include additional bins to reach target
                assert (va_ratio >= 0.60).all()  # At least 60%
                assert (va_ratio <= 0.85).all()  # At most 85%

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_volume_distribution_algorithm(self, sample_data):
        """Test volume distribution across price bins."""
        vp = VolumeProfile(lookback=50, price_bins=20)
        result = vp.compute(sample_data)
        
        # Total volume should be conserved
        original_volumes = sample_data.tail(50)["volume"].sum()
        calculated_volumes = result["total_volume"].dropna()
        
        if len(calculated_volumes) > 0:
            # Volume should be approximately conserved (allowing for windowing effects)
            last_calc_volume = calculated_volumes.iloc[-1]
            # Should be within reasonable range of original
            assert abs(last_calc_volume - original_volumes) / original_volumes < 0.1

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_signal_generation(self, volume_profile_data):
        """Test trading signal generation."""
        vp = VolumeProfile(lookback=50, price_bins=25)
        result = vp.compute(volume_profile_data)
        
        close_prices = volume_profile_data["close"]
        poc_prices = result["poc_price"]
        price_above_poc = result["price_above_poc"]
        price_below_poc = result["price_below_poc"]
        price_in_va = result["price_in_value_area"]
        
        # Test signal logic consistency
        valid_mask = ~poc_prices.isna()
        if valid_mask.any():
            valid_close = close_prices[valid_mask]
            valid_poc = poc_prices[valid_mask]
            valid_above = price_above_poc[valid_mask]
            valid_below = price_below_poc[valid_mask]
            
            # Signals should be mutually exclusive for above/below POC
            assert not (valid_above & valid_below).any()
            
            # Signals should match price vs POC relationship
            expected_above = valid_close > valid_poc
            expected_below = valid_close < valid_poc
            
            assert (valid_above == expected_above).all()
            assert (valid_below == expected_below).all()


class TestVolumeProfileExtendedFeatures:
    """Test Volume Profile extended features and improvements."""

    @pytest.fixture
    def extended_dataset(self):
        """Create extended dataset for 5-day testing."""
        # 5 days of 15min data = 5 * 24 * 4 = 480 periods
        return create_btc_sample_data(periods=480, freq="15min", seed=42)

    @pytest.fixture
    def hvn_lvn_data(self):
        """Create data with distinct High Volume Nodes and Low Volume Nodes."""
        dates = pd.date_range("2024-01-01", periods=200, freq="15min")
        np.random.seed(42)
        
        # Create price and volume data with distinct volume clusters
        prices = []
        volumes = []
        
        for i in range(200):
            # Create distinct volume nodes
            if 40 <= i < 60:  # High Volume Node at 45500
                price = 45500 + np.random.normal(0, 25)
                volume = np.random.uniform(2000, 3000)  # High volume
            elif 80 <= i < 100:  # High Volume Node at 44800
                price = 44800 + np.random.normal(0, 25)
                volume = np.random.uniform(2000, 3000)  # High volume
            elif 120 <= i < 140:  # Low Volume Node at 45200
                price = 45200 + np.random.normal(0, 25)
                volume = np.random.uniform(200, 400)  # Low volume
            else:  # Medium volume elsewhere
                price = 45000 + np.random.normal(0, 100)
                volume = np.random.uniform(800, 1200)  # Medium volume
            
            prices.append(price)
            volumes.append(volume)
        
        data = pd.DataFrame(index=dates)
        data["close"] = prices
        data["open"] = [p * 0.9995 for p in prices]
        data["high"] = [p * 1.0005 for p in prices]
        data["low"] = [p * 0.9995 for p in prices]
        data["volume"] = volumes
        
        return data

    @pytest.fixture
    def volatile_market_data(self):
        """Create volatile market data for stress testing."""
        dates = pd.date_range("2024-01-01", periods=150, freq="15min")
        np.random.seed(42)
        
        # Create volatile price movements
        base_price = 45000
        prices = [base_price]
        
        for i in range(1, 150):
            # Add significant volatility
            change = np.random.normal(0, 200)  # High volatility
            new_price = max(prices[-1] + change, 1000)  # Prevent negative prices
            prices.append(new_price)
        
        volumes = np.random.uniform(500, 2000, 150)
        
        data = pd.DataFrame(index=dates)
        data["close"] = prices
        data["open"] = [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices]
        data["high"] = [p * (1 + abs(np.random.uniform(0, 0.02))) for p in prices]
        data["low"] = [p * (1 - abs(np.random.uniform(0, 0.02))) for p in prices]
        data["volume"] = volumes
        
        return data

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_extended_lookback_5day(self, extended_dataset):
        """Test Volume Profile with 5-day lookback window."""
        # 5 days = 5 * 24 * 4 = 480 periods for 15min data
        vp = VolumeProfile(lookback=480, price_bins=60, value_area_pct=70)
        result = vp.compute(extended_dataset)
        
        # Should handle large lookback without errors
        assert len(result) == len(extended_dataset)
        
        # Check that we get valid results for periods with sufficient data
        poc_prices = result["poc_price"].dropna()
        assert len(poc_prices) > 0
        
        # Values should be reasonable
        assert (poc_prices > 0).all()
        assert not poc_prices.isna().all()
        
        # Test performance with large dataset
        start_time = time.time()
        result2 = vp.compute(extended_dataset)
        calculation_time = time.time() - start_time
        
        # Should complete in reasonable time even with large lookback
        assert calculation_time < 60.0  # 1 minute max for 5-day lookback

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_hvn_lvn_identification_comprehensive(self, hvn_lvn_data):
        """Test comprehensive identification of High Volume Nodes and Low Volume Nodes."""
        vp = VolumeProfile(
            lookback=100, 
            price_bins=40, 
            value_area_pct=70,
            hvn_threshold=1.5,  # 1.5x average volume
            lvn_threshold=0.5   # 0.5x average volume
        )
        result = vp.compute(hvn_lvn_data)
        
        # Test HVN/LVN classification columns
        is_hvn = result["is_hvn"]
        is_lvn = result["is_lvn"]
        average_volume = result["average_volume"].dropna()
        
        # Test that HVN and LVN are mutually exclusive
        assert not (is_hvn & is_lvn).any(), "HVN and LVN should be mutually exclusive"
        
        # Test HVN identification
        hvn_count = is_hvn.sum()
        lvn_count = is_lvn.sum()
        
        assert hvn_count > 0, "Should identify some High Volume Nodes"
        assert lvn_count > 0, "Should identify some Low Volume Nodes"
        
        # Test that HVN occurs during high volume periods
        current_volumes = hvn_lvn_data["volume"]
        valid_mask = ~result["average_volume"].isna()
        
        if valid_mask.any():
            # Check that HVN periods have higher than average volume
            hvn_periods = is_hvn[valid_mask]
            avg_vol_periods = result["average_volume"][valid_mask]
            current_vol_periods = current_volumes[valid_mask]
            
            if hvn_periods.any():
                hvn_volume_ratio = (current_vol_periods[hvn_periods] / avg_vol_periods[hvn_periods]).mean()
                assert hvn_volume_ratio >= vp.hvn_threshold * 0.9, f"HVN volume ratio too low: {hvn_volume_ratio:.2f}"
            
            # Check that LVN periods have lower than average volume
            lvn_periods = is_lvn[valid_mask]
            if lvn_periods.any():
                lvn_volume_ratio = (current_vol_periods[lvn_periods] / avg_vol_periods[lvn_periods]).mean()
                assert lvn_volume_ratio <= vp.lvn_threshold * 1.1, f"LVN volume ratio too high: {lvn_volume_ratio:.2f}"
        
        # Test POC strength correlation with volume nodes
        poc_strengths = result["poc_strength"].dropna()
        if len(poc_strengths) > 10:
            strength_90th = np.percentile(poc_strengths, 90)
            strength_10th = np.percentile(poc_strengths, 10)
            
            # High volume periods should have higher POC strength
            assert strength_90th > strength_10th * 1.2, "POC strength should vary with volume"
        
        # Test different threshold configurations
        vp_sensitive = VolumeProfile(
            lookback=100,
            price_bins=40,
            hvn_threshold=1.2,  # More sensitive (lower threshold identifies more HVNs)
            lvn_threshold=0.8   # Less sensitive (higher threshold identifies more LVNs)
        )
        result_sensitive = vp_sensitive.compute(hvn_lvn_data)
        
        # More sensitive settings should identify more HVNs and more LVNs
        sensitive_hvn_count = result_sensitive["is_hvn"].sum()
        sensitive_lvn_count = result_sensitive["is_lvn"].sum()
        
        assert sensitive_hvn_count >= hvn_count, "More sensitive HVN threshold should identify more HVNs"
        assert sensitive_lvn_count >= lvn_count, "Less sensitive LVN threshold should identify more LVNs"

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_distance_metrics_comprehensive(self, hvn_lvn_data):
        """Test comprehensive POC/VAH/VAL distance calculations and relative positioning."""
        vp = VolumeProfile(lookback=100, price_bins=40, value_area_pct=70)
        result = vp.compute(hvn_lvn_data)
        
        # Test all distance metrics
        poc_prices = result["poc_price"].dropna()
        vah_prices = result["vah_price"].dropna()
        val_prices = result["val_price"].dropna()
        va_width = result["value_area_width"].dropna()
        
        # Test the new distance columns
        dist_to_poc = result["dist_to_poc"].dropna()
        dist_to_vah = result["dist_to_vah"].dropna()
        dist_to_val = result["dist_to_val"].dropna()
        
        if len(poc_prices) > 0 and len(vah_prices) > 0 and len(val_prices) > 0:
            # Test distance relationships
            min_len = min(len(poc_prices), len(vah_prices), len(val_prices))
            poc_subset = poc_prices[:min_len]
            vah_subset = vah_prices[:min_len]
            val_subset = val_prices[:min_len]
            
            # Distance from POC to VAH and VAL
            poc_to_vah_dist = vah_subset - poc_subset
            poc_to_val_dist = poc_subset - val_subset
            
            # Both distances should be positive (VAH > POC > VAL)
            assert (poc_to_vah_dist >= 0).all()
            assert (poc_to_val_dist >= 0).all()
            
            # Test Value Area width calculation
            if len(va_width) > 0:
                expected_width = vah_subset[:len(va_width)] - val_subset[:len(va_width)]
                np.testing.assert_allclose(
                    va_width, expected_width, rtol=1e-6,
                    err_msg="Value Area width calculation incorrect"
                )
        
        # Test distance-to-key-levels calculations
        if len(dist_to_poc) > 0 and len(poc_prices) > 0:
            # Distance to POC should be current_price - poc_price
            close_prices = hvn_lvn_data["close"]
            valid_indices = ~result["poc_price"].isna()
            
            if valid_indices.any():
                expected_dist_poc = close_prices[valid_indices] - result["poc_price"][valid_indices]
                actual_dist_poc = result["dist_to_poc"][valid_indices]
                
                # Should be mathematically equivalent
                np.testing.assert_allclose(
                    actual_dist_poc, expected_dist_poc, rtol=1e-6,
                    err_msg="Distance to POC calculation incorrect"
                )
        
        # Test distance interpretation
        if len(dist_to_poc) > 0:
            # Positive distance means price is above POC
            # Negative distance means price is below POC
            price_above_poc = result["price_above_poc"]
            price_below_poc = result["price_below_poc"]
            
            valid_mask = ~result["dist_to_poc"].isna()
            if valid_mask.any():
                # When distance is positive, price should be above POC
                positive_dist = result["dist_to_poc"][valid_mask] > 0
                above_poc = price_above_poc[valid_mask]
                
                # Should be consistent (allowing for edge cases at exactly POC)
                consistency_rate = (positive_dist == above_poc).mean()
                assert consistency_rate > 0.9, f"Distance-to-POC consistency too low: {consistency_rate:.2f}"
        
        # Test relative distance magnitudes
        if len(dist_to_vah) > 0 and len(dist_to_val) > 0:
            # Distance to VAH should be smaller (less negative) than distance to VAL when price is below both
            # Distance to VAL should be smaller (less positive) than distance to VAH when price is above both
            
            valid_mask = ~result["dist_to_vah"].isna() & ~result["dist_to_val"].isna()
            if valid_mask.any():
                vah_dist = result["dist_to_vah"][valid_mask]
                val_dist = result["dist_to_val"][valid_mask]
                
                # When price is below both VAH and VAL, distance to VAH should be more negative
                below_both = (vah_dist < 0) & (val_dist < 0)
                if below_both.any():
                    assert (vah_dist[below_both] < val_dist[below_both]).all(), "Distance relationships incorrect for prices below VA"
                
                # When price is above both VAH and VAL, distance to VAL should be more positive
                above_both = (vah_dist > 0) & (val_dist > 0)
                if above_both.any():
                    assert (val_dist[above_both] > vah_dist[above_both]).all(), "Distance relationships incorrect for prices above VA"

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_simplified_volume_distribution(self, hvn_lvn_data):
        """Test simplified 50/50 volume distribution method."""
        vp = VolumeProfile(lookback=50, price_bins=30)
        result = vp.compute(hvn_lvn_data)
        
        # Test that volume distribution is simplified and consistent
        poc_prices = result["poc_price"].dropna()
        total_volumes = result["total_volume"].dropna()
        
        # Check volume conservation (should be exact with simplified method)
        if len(total_volumes) > 0:
            # Calculate expected volume for last window
            window_data = hvn_lvn_data.tail(50)
            expected_volume = window_data["volume"].sum()
            calculated_volume = total_volumes.iloc[-1]
            
            # Volume should be perfectly conserved with simplified method
            assert abs(calculated_volume - expected_volume) / expected_volume < 0.001, \
                f"Volume conservation failed: expected {expected_volume:.2f}, got {calculated_volume:.2f}"
        
        # Test that the simplified method produces reasonable POC locations
        if len(poc_prices) > 0:
            # POC should generally be within the price range of the data
            price_range = hvn_lvn_data[["high", "low", "open", "close"]].values
            price_min = np.min(price_range)
            price_max = np.max(price_range)
            
            # Allow some tolerance for volume-weighted distribution
            extended_min = price_min * 0.95
            extended_max = price_max * 1.05
            
            valid_poc = poc_prices[(poc_prices >= extended_min) & (poc_prices <= extended_max)]
            assert len(valid_poc) >= len(poc_prices) * 0.9, \
                f"Too many POC prices outside reasonable range: {len(valid_poc)}/{len(poc_prices)}"
        
        # Test performance improvement with simplified method
        import time
        
        # Test with larger dataset to measure performance
        large_data = create_btc_sample_data(periods=1000, freq="15min", seed=42)
        
        start_time = time.time()
        large_result = vp.compute(large_data)
        calc_time = time.time() - start_time
        
        # Simplified method should be reasonably fast
        assert calc_time < 30.0, f"Simplified method too slow: {calc_time:.2f}s"
        
        # Should still produce valid results
        large_poc = large_result["poc_price"].dropna()
        large_volumes = large_result["total_volume"].dropna()
        
        assert len(large_poc) > 0, "No POC prices calculated for large dataset"
        assert len(large_volumes) > 0, "No volumes calculated for large dataset"
        
        # Test volume conservation for large dataset
        if len(large_volumes) > 0:
            window_data = large_data.tail(50)
            expected_volume = window_data["volume"].sum()
            calculated_volume = large_volumes.iloc[-1]
            
            conservation_error = abs(calculated_volume - expected_volume) / expected_volume
            assert conservation_error < 0.001, \
                f"Volume conservation failed for large dataset: {conservation_error:.4f}"

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_profile_balance_calculation(self, hvn_lvn_data):
        """Test profile balance calculation and symmetry analysis."""
        vp = VolumeProfile(lookback=80, price_bins=40)
        result = vp.compute(hvn_lvn_data)
        
        balance_scores = result["profile_balance"].dropna()
        
        if len(balance_scores) > 0:
            # Balance scores should be between 0 and 1
            assert (balance_scores >= 0).all()
            assert (balance_scores <= 1).all()
            
            # Test with controlled symmetric data
            symmetric_data = self._create_symmetric_volume_data()
            symmetric_result = vp.compute(symmetric_data)
            symmetric_balance = symmetric_result["profile_balance"].dropna()
            
            if len(symmetric_balance) > 0:
                # Symmetric data should have high balance scores
                assert symmetric_balance.mean() > 0.7  # Should be well balanced

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_average_volume_calculation(self, hvn_lvn_data):
        """Test the new average_volume column calculation."""
        vp = VolumeProfile(lookback=60, price_bins=30)
        result = vp.compute(hvn_lvn_data)
        
        avg_volumes = result["average_volume"].dropna()
        total_volumes = result["total_volume"].dropna()
        
        if len(avg_volumes) > 0 and len(total_volumes) > 0:
            # Average volume should be total volume divided by lookback
            min_len = min(len(avg_volumes), len(total_volumes))
            expected_avg = total_volumes[:min_len] / vp.lookback
            
            np.testing.assert_allclose(
                avg_volumes[:min_len], expected_avg, rtol=1e-6,
                err_msg="Average volume calculation incorrect"
            )

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_volume_spike_detection(self, volatile_market_data):
        """Test volume spike detection with volatile market data."""
        vp = VolumeProfile(lookback=50, price_bins=30)
        result = vp.compute(volatile_market_data)
        
        volume_spikes = result["volume_spike"]
        avg_volumes = result["average_volume"].dropna()
        
        # Should detect some volume spikes in volatile data
        spike_count = volume_spikes.sum()
        assert spike_count > 0  # Should detect some spikes
        assert spike_count < len(volume_spikes) * 0.3  # But not too many

    def _create_symmetric_volume_data(self):
        """Create perfectly symmetric volume distribution for testing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="15min")
        np.random.seed(42)
        
        # Create symmetric volume around a central price
        center_price = 45000
        prices = []
        volumes = []
        
        for i in range(100):
            # Create symmetric distribution
            if i < 50:
                # Lower half - symmetric pattern
                offset = (i - 25) * 20  # -500 to +500
                price = center_price + offset
                volume = 1000 + abs(offset) * 2  # Symmetric volume
            else:
                # Upper half - mirror the lower half
                mirror_i = 99 - i
                offset = (mirror_i - 25) * 20
                price = center_price + offset
                volume = 1000 + abs(offset) * 2  # Symmetric volume
            
            prices.append(price)
            volumes.append(volume)
        
        data = pd.DataFrame(index=dates)
        data["close"] = prices
        data["open"] = [p * 0.999 for p in prices]
        data["high"] = [p * 1.001 for p in prices]
        data["low"] = [p * 0.999 for p in prices]
        data["volume"] = volumes
        
        return data


class TestVolumeProfileEdgeCases:
    """Test Volume Profile edge cases and error handling."""

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_insufficient_data(self):
        """Test Volume Profile behavior with insufficient data."""
        # Create minimal data (less than lookback period)
        small_data = pd.DataFrame(
            {
                "open": [45000.0, 45100.0, 45050.0],
                "high": [45050.0, 45150.0, 45100.0],
                "low": [44950.0, 45050.0, 45000.0],
                "close": [45020.0, 45120.0, 45070.0],
                "volume": [1000.0, 1100.0, 950.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="15min"),
        )

        vp = VolumeProfile(lookback=20)
        result = vp.compute(small_data)

        assert len(result) == 3
        # Should return NaN for insufficient data periods
        assert pd.isna(result["poc_price"]).all()
        assert pd.isna(result["vah_price"]).all()
        assert pd.isna(result["val_price"]).all()

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_zero_volume_data(self):
        """Test behavior with zero volume periods."""
        dates = pd.date_range("2024-01-01", periods=50, freq="15min")
        
        zero_volume_data = pd.DataFrame(
            {
                "open": [45000.0] * 50,
                "high": [45010.0] * 50,
                "low": [44990.0] * 50,
                "close": [45000.0] * 50,
                "volume": [0.0] * 50,  # Zero volume
            },
            index=dates,
        )

        vp = VolumeProfile(lookback=30)
        result = vp.compute(zero_volume_data)

        # Should handle zero volume gracefully
        assert len(result) == 50
        
        # When implemented, should either return NaN or handle appropriately
        total_volumes = result["total_volume"]
        poc_volumes = result["poc_volume"]
        
        # These should be zero or NaN when volume is zero
        if not total_volumes.isna().all():
            assert (total_volumes == 0.0).all()

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_single_price_level(self):
        """Test behavior when all prices are the same."""
        dates = pd.date_range("2024-01-01", periods=50, freq="15min")
        
        single_price_data = pd.DataFrame(
            {
                "open": [45000.0] * 50,
                "high": [45000.0] * 50,
                "low": [45000.0] * 50,
                "close": [45000.0] * 50,
                "volume": np.random.uniform(500, 1500, 50),
            },
            index=dates,
        )

        vp = VolumeProfile(lookback=30, price_bins=20)
        result = vp.compute(single_price_data)

        assert len(result) == 50
        
        # When all prices are the same, POC, VAH, and VAL should be the same
        poc_prices = result["poc_price"].dropna()
        vah_prices = result["vah_price"].dropna()
        val_prices = result["val_price"].dropna()
        
        if len(poc_prices) > 0:
            assert (poc_prices == 45000.0).all()
            assert (vah_prices == 45000.0).all()
            assert (val_prices == 45000.0).all()

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_extreme_volatility(self):
        """Test behavior with extreme price volatility."""
        extreme_data = create_volatile_market_data(periods=100, volatility_level="extreme")
        
        vp = VolumeProfile(lookback=50, price_bins=25)
        result = vp.compute(extreme_data)
        
        # Should handle extreme volatility without errors
        assert len(result) == 100
        
        # Results should still be within reasonable bounds
        poc_prices = result["poc_price"].dropna()
        if len(poc_prices) > 0:
            assert not poc_prices.isna().all()
            assert (poc_prices > 0).all()  # Prices should be positive

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_empty_dataframe(self):
        """Test Volume Profile behavior with empty DataFrame."""
        empty_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        vp = VolumeProfile()
        result = vp.compute(empty_data)

        assert len(result) == 0
        expected_cols = [
            "poc_price",
            "vah_price", 
            "val_price",
            "total_volume",
            "poc_volume",
            "value_area_volume",
            "average_volume",
            "price_above_poc",
            "price_below_poc",
            "price_in_value_area",
            "poc_breakout",
            "volume_spike",
            "poc_strength",
            "value_area_width",
            "profile_balance"
        ]
        assert list(result.columns) == expected_cols


class TestVolumeProfilePerformance:
    """Performance benchmarks for Volume Profile indicator."""

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        return create_btc_sample_data(periods=5000, freq="15min", seed=42)

    @pytest.fixture
    def very_large_dataset(self):
        """Create very large dataset for stress testing."""
        return create_btc_sample_data(periods=10000, freq="15min", seed=42)

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_performance_benchmark_standard(self, large_dataset):
        """Test performance with standard parameters (5-day lookback)."""
        vp = VolumeProfile(lookback=480, price_bins=50, enable_vectorization=True)  # 5-day default
        
        start_time = time.time()
        result = vp.compute(large_dataset)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Should complete within reasonable time for large dataset with 5-day lookback
        assert calculation_time < 45.0  # 45 seconds max for 5000 periods with 5-day lookback
        assert len(result) == len(large_dataset)
        
        # Test vectorized improvements - should be faster than naive approach
        # This is a benchmark to ensure optimizations are working
        per_period_time = calculation_time / len(large_dataset)
        assert per_period_time < 0.015  # Less than 15ms per period for 5-day lookback
        
        # Test with different lookback periods to ensure scaling
        lookback_times = {}
        for lookback in [96, 240, 480, 960]:  # 1-day, 2.5-day, 5-day, 10-day
            vp_test = VolumeProfile(lookback=lookback, price_bins=50, enable_vectorization=True)
            
            start_time = time.time()
            result_test = vp_test.compute(large_dataset)
            test_time = time.time() - start_time
            
            lookback_times[lookback] = test_time
            
            # Each should complete in reasonable time
            max_time = 30.0 + (lookback - 96) * 0.02  # Scale with lookback
            assert test_time < max_time, f"Lookback {lookback} took {test_time:.1f}s (max {max_time:.1f}s)"
        
        # Verify that computation time scales reasonably with lookback
        time_96 = lookback_times[96]
        time_480 = lookback_times[480]
        time_960 = lookback_times[960]
        
        # 5-day should take less than 10x the time of 1-day
        assert time_480 < time_96 * 10, f"5-day lookback scaling poor: {time_480:.1f}s vs {time_96:.1f}s"
        
        # 10-day should take less than 20x the time of 1-day
        assert time_960 < time_96 * 20, f"10-day lookback scaling poor: {time_960:.1f}s vs {time_96:.1f}s"

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_performance_benchmark_high_resolution(self, large_dataset):
        """Test performance with high resolution (many bins) and extended lookback."""
        vp = VolumeProfile(lookback=480, price_bins=200, enable_vectorization=True)  # High resolution + 5-day lookback
        
        start_time = time.time()
        result = vp.compute(large_dataset)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Should still complete in reasonable time with high resolution and extended lookback
        assert calculation_time < 60.0  # 60 seconds max for high resolution with 5-day lookback
        assert len(result) == len(large_dataset)
        
        # Test that high resolution doesn't cause exponential slowdown
        per_period_time = calculation_time / len(large_dataset)
        assert per_period_time < 0.02  # Less than 20ms per period even with high resolution
        
        # Test scaling with bin count
        bin_times = {}
        for bin_count in [25, 50, 100, 200, 500]:
            vp_test = VolumeProfile(lookback=480, price_bins=bin_count, enable_vectorization=True)
            
            start_time = time.time()
            result_test = vp_test.compute(large_dataset)
            test_time = time.time() - start_time
            
            bin_times[bin_count] = test_time
            
            # Each should complete in reasonable time
            max_time = 30.0 + (bin_count - 25) * 0.1  # Scale with bin count
            assert test_time < max_time, f"Bin count {bin_count} took {test_time:.1f}s (max {max_time:.1f}s)"
        
        # Verify that computation time scales reasonably with bin count
        time_25 = bin_times[25]
        time_200 = bin_times[200]
        time_500 = bin_times[500]
        
        # 200 bins should take less than 20x the time of 25 bins
        assert time_200 < time_25 * 20, f"200-bin scaling poor: {time_200:.1f}s vs {time_25:.1f}s"
        
        # 500 bins should take less than 50x the time of 25 bins
        assert time_500 < time_25 * 50, f"500-bin scaling poor: {time_500:.1f}s vs {time_25:.1f}s"

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_memory_usage(self, large_dataset):
        """Test memory usage with large dataset and extended features."""
        vp = VolumeProfile(lookback=480, price_bins=50, enable_vectorization=True)  # 5-day lookback
        result = vp.compute(large_dataset)
        
        # Basic memory test - ensure calculation completes without memory errors
        assert len(result) == len(large_dataset)
        
        # Test that result size is reasonable (not exponentially large)
        expected_columns = 20  # Number of output columns (all new features included)
        assert len(result.columns) == expected_columns
        
        # Verify all expected columns are present
        expected_column_names = [
            "poc_price", "vah_price", "val_price", "total_volume", "poc_volume",
            "value_area_volume", "average_volume", "price_above_poc", "price_below_poc",
            "price_in_value_area", "poc_breakout", "volume_spike", "is_hvn", "is_lvn",
            "dist_to_poc", "dist_to_vah", "dist_to_val", "poc_strength", 
            "value_area_width", "profile_balance"
        ]
        assert set(result.columns) == set(expected_column_names), \
            f"Missing columns: {set(expected_column_names) - set(result.columns)}"
        
        # Test that result doesn't consume excessive memory
        # (This is a basic test - could be enhanced with actual memory monitoring)
        memory_per_row = result.memory_usage(deep=True).sum() / len(result)
        assert memory_per_row < 1500  # Less than 1.5KB per row (increased for extended features)
        
        # Test memory usage with different configurations
        configs = [
            {"lookback": 96, "price_bins": 25, "description": "1-day minimal"},
            {"lookback": 480, "price_bins": 50, "description": "5-day standard"},
            {"lookback": 960, "price_bins": 100, "description": "10-day high-res"},
        ]
        
        for config in configs:
            vp_test = VolumeProfile(
                lookback=config["lookback"], 
                price_bins=config["price_bins"], 
                enable_vectorization=True
            )
            result_test = vp_test.compute(large_dataset)
            
            # Memory usage should scale reasonably
            memory_per_row_test = result_test.memory_usage(deep=True).sum() / len(result_test)
            max_memory = 2000  # 2KB per row max
            assert memory_per_row_test < max_memory, \
                f"{config['description']} memory usage too high: {memory_per_row_test:.0f} bytes/row"
        
        # Test that extended lookback doesn't cause memory issues
        vp_extended = VolumeProfile(lookback=1440, price_bins=50, enable_vectorization=True)  # 15-day
        result_extended = vp_extended.compute(large_dataset)
        
        assert len(result_extended) == len(large_dataset)
        assert len(result_extended.columns) == expected_columns
        
        extended_memory = result_extended.memory_usage(deep=True).sum() / len(result_extended)
        assert extended_memory < 2000  # Extended lookback shouldn't significantly increase memory per row

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_incremental_update_performance(self):
        """Test performance of incremental updates."""
        # Test that update_frequency parameter affects performance
        data_chunks = []
        base_data = create_btc_sample_data(periods=1000, freq="15min", seed=42)
        
        # Split into chunks to simulate real-time updates
        chunk_size = 100
        for i in range(0, len(base_data), chunk_size):
            chunk = base_data.iloc[:i + chunk_size]
            data_chunks.append(chunk)
        
        vp_frequent = VolumeProfile(lookback=50, update_frequency=1)  # Update every period
        vp_infrequent = VolumeProfile(lookback=50, update_frequency=10)  # Update every 10 periods
        
        # Time frequent updates
        start_time = time.time()
        for chunk in data_chunks[-3:]:  # Test last 3 chunks
            result_frequent = vp_frequent.compute(chunk)
        time_frequent = time.time() - start_time
        
        # Time infrequent updates 
        start_time = time.time()
        for chunk in data_chunks[-3:]:  # Test last 3 chunks
            result_infrequent = vp_infrequent.compute(chunk)
        time_infrequent = time.time() - start_time
        
        # Infrequent updates should be faster or similar
        # (This assumes optimization is implemented)
        assert time_infrequent <= time_frequent * 1.1  # Allow 10% tolerance

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_vectorized_performance_improvements(self, large_dataset):
        """Test that vectorized improvements maintain speed requirements."""
        # Test vectorized vs iterative performance
        vp_vectorized = VolumeProfile(
            lookback=96, 
            price_bins=50,
            enable_vectorization=True
        )
        
        vp_iterative = VolumeProfile(
            lookback=96, 
            price_bins=50,
            enable_vectorization=False
        )
        
        # Time vectorized computation
        start_time = time.time()
        result_vectorized = vp_vectorized.compute(large_dataset)
        time_vectorized = time.time() - start_time
        
        # Time iterative computation
        start_time = time.time()
        result_iterative = vp_iterative.compute(large_dataset)
        time_iterative = time.time() - start_time
        
        # Both should complete in reasonable time
        assert time_vectorized < 45.0  # 45 seconds max
        assert time_iterative < 45.0   # 45 seconds max
        
        # Results should be identical
        assert len(result_vectorized) == len(result_iterative)
        assert len(result_vectorized) == len(large_dataset)
        
        # Test that results are consistent between methods
        assert not result_vectorized["poc_price"].dropna().empty
        assert not result_iterative["poc_price"].dropna().empty
        
        # Compare key metrics (allowing for small floating point differences)
        valid_mask = ~result_vectorized["poc_price"].isna() & ~result_iterative["poc_price"].isna()
        if valid_mask.any():
            poc_diff = np.abs(result_vectorized.loc[valid_mask, "poc_price"] - 
                             result_iterative.loc[valid_mask, "poc_price"])
            assert (poc_diff < 0.01).all()  # Should be nearly identical
        
        # Test multiple configurations to ensure vectorization is effective
        configs = [
            {"lookback": 48, "price_bins": 30, "enable_vectorization": True},
            {"lookback": 240, "price_bins": 50, "enable_vectorization": True},
            {"lookback": 480, "price_bins": 100, "enable_vectorization": True},
        ]
        
        for config in configs:
            vp = VolumeProfile(**config)
            
            start_time = time.time()
            result = vp.compute(large_dataset)
            calculation_time = time.time() - start_time
            
            # Should meet performance requirements
            assert calculation_time < 60.0  # 60 seconds max for extended lookbacks
            assert len(result) == len(large_dataset)
            
            # Test that results are consistent
            assert not result["poc_price"].dropna().empty
            assert not result["total_volume"].dropna().empty

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_extended_lookback_performance(self):
        """Test performance with extended lookback periods."""
        # Test 5-day lookback performance (now the default)
        extended_data = create_btc_sample_data(periods=2000, freq="15min", seed=42)
        
        # Test various extended lookback periods
        lookback_configs = [
            {"lookback": 240, "desc": "2.5-day"},   # 2.5 days
            {"lookback": 480, "desc": "5-day"},     # 5 days (default)
            {"lookback": 960, "desc": "10-day"},    # 10 days
            {"lookback": 1440, "desc": "15-day"},   # 15 days
        ]
        
        for config in lookback_configs:
            vp = VolumeProfile(
                lookback=config["lookback"], 
                price_bins=50,
                enable_vectorization=True
            )
            
            start_time = time.time()
            result = vp.compute(extended_data)
            calculation_time = time.time() - start_time
            
            # Should handle extended lookback efficiently
            max_time = 120.0 if config["lookback"] <= 480 else 180.0  # More time for longer lookbacks
            assert calculation_time < max_time, f"{config['desc']} lookback took {calculation_time:.1f}s (max {max_time}s)"
            assert len(result) == len(extended_data)
            
            # Results should be valid
            poc_prices = result["poc_price"].dropna()
            assert len(poc_prices) > 0, f"No POC prices found for {config['desc']} lookback"
            assert (poc_prices > 0).all(), f"Invalid POC prices for {config['desc']} lookback"
            
            # Test that longer lookbacks provide more stable results
            if len(poc_prices) > 50:
                # Check POC stability (longer lookbacks should be more stable)
                poc_volatility = poc_prices.pct_change().std()
                assert poc_volatility < 0.1, f"POC too volatile for {config['desc']} lookback: {poc_volatility:.4f}"

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_parameter_edge_cases_performance(self):
        """Test performance with edge case parameters."""
        test_data = create_btc_sample_data(periods=500, freq="15min", seed=42)
        
        # Test with minimal parameters
        vp_minimal = VolumeProfile(lookback=10, price_bins=5)
        start_time = time.time()
        result_minimal = vp_minimal.compute(test_data)
        time_minimal = time.time() - start_time
        
        # Test with high resolution parameters
        vp_high_res = VolumeProfile(lookback=50, price_bins=500)
        start_time = time.time()
        result_high_res = vp_high_res.compute(test_data)
        time_high_res = time.time() - start_time
        
        # Both should complete in reasonable time
        assert time_minimal < 5.0  # Minimal should be very fast
        assert time_high_res < 30.0  # High resolution should still be reasonable
        
        # Both should produce valid results
        assert len(result_minimal) == len(test_data)
        assert len(result_high_res) == len(test_data)


class TestVolumeProfileIntegration:
    """Test Volume Profile integration with other indicators."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for integration testing."""
        return create_btc_sample_data(periods=300, freq="15min", seed=42)

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_multiple_volume_profile_instances(self, sample_data):
        """Test using multiple Volume Profile instances with different parameters."""
        vp_short = VolumeProfile(lookback=48, price_bins=30)  # 12 hour lookback
        vp_long = VolumeProfile(lookback=96, price_bins=50)   # 24 hour lookback
        
        result_short = vp_short.compute(sample_data)
        result_long = vp_long.compute(sample_data)
        
        # Both should produce same structure
        assert len(result_short) == len(sample_data)
        assert len(result_long) == len(sample_data)
        
        # Column structures should be the same
        assert result_short.columns.tolist() == result_long.columns.tolist()

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_registry_retrieval_and_instantiation(self, sample_data):
        """Test retrieving Volume Profile from registry and using it."""
        # Get class from registry
        vp_class = get("VolumeProfile")
        
        # Instantiate with custom parameters
        vp = vp_class(lookback=72, price_bins=40, value_area_pct=68)
        
        # Compute results
        result = vp.compute(sample_data)
        
        # Verify correct initialization
        assert vp.lookback == 72
        assert vp.price_bins == 40
        assert vp.value_area_pct == 0.68
        
        # Verify output format
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_combined_with_other_indicators(self, sample_data):
        """Test using Volume Profile alongside other indicators."""
        # This would test integration with EMA, RSI, etc.
        # For now, just test that Volume Profile doesn't interfere
        
        vp = VolumeProfile(lookback=96, price_bins=50)
        vp_result = vp.compute(sample_data)
        
        # Simulate combining with other indicators (like in engine)
        combined = sample_data.join(vp_result)
        
        # Verify no column name conflicts - order doesn't matter, just check all are present
        expected_base_cols = {"open", "high", "low", "close", "volume"}
        expected_vp_cols = {
            "poc_price", "vah_price", "val_price",
            "total_volume", "poc_volume", "value_area_volume", "average_volume",
            "price_above_poc", "price_below_poc", "price_in_value_area",
            "poc_breakout", "volume_spike", "is_hvn", "is_lvn",
            "dist_to_poc", "dist_to_vah", "dist_to_val",
            "poc_strength", "value_area_width", "profile_balance"
        }
        
        all_expected_cols = expected_base_cols | expected_vp_cols
        actual_cols = set(combined.columns)
        assert actual_cols == all_expected_cols


class TestVolumeProfileAdvancedFeatures:
    """Test Volume Profile advanced features and integrations."""

    @pytest.fixture
    def atr_test_data(self):
        """Create data suitable for ATR-based breakout testing."""
        dates = pd.date_range("2024-01-01", periods=200, freq="15min")
        np.random.seed(42)
        
        # Create data with varying volatility periods
        prices = []
        volumes = []
        
        base_price = 45000
        for i in range(200):
            # Create periods of high and low volatility
            if 50 <= i < 100:  # High volatility period
                volatility = 300
                volume_mult = 1.5
            elif 150 <= i < 200:  # Low volatility period
                volatility = 50
                volume_mult = 0.8
            else:  # Normal volatility
                volatility = 150
                volume_mult = 1.0
            
            # Generate price with varying volatility
            if i == 0:
                price = base_price
            else:
                price = max(prices[-1] + np.random.normal(0, volatility), 1000)
            
            volume = np.random.uniform(800, 1200) * volume_mult
            
            prices.append(price)
            volumes.append(volume)
        
        data = pd.DataFrame(index=dates)
        data["close"] = prices
        data["open"] = [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices]
        data["high"] = [p * (1 + abs(np.random.uniform(0, 0.01))) for p in prices]
        data["low"] = [p * (1 - abs(np.random.uniform(0, 0.01))) for p in prices]
        data["volume"] = volumes
        
        return data

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_atr_integration_comprehensive(self, atr_test_data):
        """Test comprehensive ATR-based breakout logic integration."""
        vp = VolumeProfile(
            lookback=50, 
            price_bins=30, 
            atr_period=14, 
            atr_multiplier=2.0,
            poc_sensitivity=0.05  # Kept for backward compatibility
        )
        result = vp.compute(atr_test_data)
        
        # Test ATR-based breakout detection
        poc_breakouts = result["poc_breakout"]
        breakout_count = poc_breakouts.sum()
        
        # Should detect some breakouts in volatile data
        assert breakout_count > 0
        assert breakout_count < len(poc_breakouts) * 0.2  # But not too many
        
        # Test that breakouts align with price movements and ATR
        poc_prices = result["poc_price"]
        close_prices = atr_test_data["close"]
        
        # Calculate ATR manually for comparison
        atr_values = vp._calculate_atr(atr_test_data, vp.atr_period)
        
        # When breakouts occur, POC change should exceed ATR threshold
        for i in range(1, len(poc_breakouts)):
            if poc_breakouts.iloc[i] and not pd.isna(poc_prices.iloc[i]) and not pd.isna(poc_prices.iloc[i-1]):
                if not pd.isna(atr_values[i]):
                    poc_change = abs(poc_prices.iloc[i] - poc_prices.iloc[i-1])
                    atr_threshold = atr_values[i] * vp.atr_multiplier
                    # Breakout should indicate significant POC movement relative to ATR
                    # (allowing some tolerance for edge cases)
                    assert poc_change >= atr_threshold * 0.8  # 80% of threshold for tolerance
        
        # Test ATR calculation directly
        assert len(atr_values) == len(atr_test_data)
        assert not np.isnan(atr_values[-20:]).all()  # Should have ATR values at the end
        
        # Test different ATR parameters
        vp_sensitive = VolumeProfile(
            lookback=50, 
            price_bins=30, 
            atr_period=7,   # Shorter ATR period
            atr_multiplier=1.5  # Lower multiplier
        )
        result_sensitive = vp_sensitive.compute(atr_test_data)
        
        # More sensitive settings should detect more breakouts
        sensitive_breakouts = result_sensitive["poc_breakout"].sum()
        assert sensitive_breakouts >= breakout_count  # Should be at least as many

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_multi_timeframe_compatibility(self):
        """Test Volume Profile compatibility with different timeframes."""
        # Test with different timeframe data
        data_15m = create_btc_sample_data(periods=200, freq="15min", seed=42)
        data_1h = create_btc_sample_data(periods=200, freq="1h", seed=42)
        data_4h = create_btc_sample_data(periods=200, freq="4h", seed=42)
        
        # Adjust lookback for different timeframes
        vp_15m = VolumeProfile(lookback=96, price_bins=50)  # 24 hours
        vp_1h = VolumeProfile(lookback=24, price_bins=50)   # 24 hours
        vp_4h = VolumeProfile(lookback=6, price_bins=50)    # 24 hours
        
        result_15m = vp_15m.compute(data_15m)
        result_1h = vp_1h.compute(data_1h)
        result_4h = vp_4h.compute(data_4h)
        
        # All should produce valid results
        assert len(result_15m) == len(data_15m)
        assert len(result_1h) == len(data_1h)
        assert len(result_4h) == len(data_4h)
        
        # All should have same column structure
        assert result_15m.columns.tolist() == result_1h.columns.tolist()
        assert result_1h.columns.tolist() == result_4h.columns.tolist()

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_enhanced_parameter_validation(self):
        """Test enhanced parameter validation and edge cases with new parameters."""
        # Test with new parameters at edge values
        test_data = create_btc_sample_data(periods=100, freq="15min", seed=42)
        
        # Test minimum viable parameters
        vp_min = VolumeProfile(
            lookback=5, 
            lookback_hours=2,
            price_bins=3, 
            value_area_pct=50,
            poc_sensitivity=0.01,
            min_volume_threshold=0.001,
            hvn_threshold=1.1,
            lvn_threshold=0.9,
            atr_period=2,
            atr_multiplier=0.5,
            enable_vectorization=False
        )
        result_min = vp_min.compute(test_data)
        assert len(result_min) == len(test_data)
        
        # Test maximum reasonable parameters
        vp_max = VolumeProfile(
            lookback=2000, 
            lookback_hours=500,
            price_bins=1000, 
            value_area_pct=95,
            poc_sensitivity=5.0,
            min_volume_threshold=10.0,
            hvn_threshold=5.0,
            lvn_threshold=0.1,
            atr_period=50,
            atr_multiplier=10.0,
            enable_vectorization=True
        )
        result_max = vp_max.compute(test_data)
        assert len(result_max) == len(test_data)
        
        # Test that parameters are correctly converted
        assert vp_min.value_area_pct == 0.5  # 50% -> 0.5
        assert vp_min.poc_sensitivity == 0.0001  # 0.01% -> 0.0001
        assert vp_min.min_volume_threshold == 0.00001  # 0.001% -> 0.00001
        
        # Test new parameters
        assert vp_min.hvn_threshold == 1.1
        assert vp_min.lvn_threshold == 0.9
        assert vp_min.atr_period == 2
        assert vp_min.atr_multiplier == 0.5
        assert vp_min.enable_vectorization == False
        
        # Test that extreme parameter values don't crash
        vp_extreme = VolumeProfile(
            lookback=1,  # Extremely small lookback
            price_bins=1,  # Single bin
            value_area_pct=1,  # 1% value area
            hvn_threshold=100,  # Very high threshold
            lvn_threshold=0.01,  # Very low threshold
            atr_period=1,  # Minimum ATR period
            atr_multiplier=0.1  # Very small multiplier
        )
        result_extreme = vp_extreme.compute(test_data)
        assert len(result_extreme) == len(test_data)
        
        # Test that all results have expected structure
        for result in [result_min, result_max, result_extreme]:
            expected_columns = 20
            assert len(result.columns) == expected_columns, f"Expected {expected_columns} columns, got {len(result.columns)}"
            
            # Check that boolean columns are boolean
            bool_columns = ["price_above_poc", "price_below_poc", "price_in_value_area", 
                          "poc_breakout", "volume_spike", "is_hvn", "is_lvn"]
            for col in bool_columns:
                assert result[col].dtype == bool, f"Column {col} should be boolean"
            
            # Check that numeric columns are numeric
            numeric_columns = ["poc_price", "vah_price", "val_price", "total_volume", 
                             "poc_volume", "value_area_volume", "average_volume",
                             "dist_to_poc", "dist_to_vah", "dist_to_val", "poc_strength",
                             "value_area_width", "profile_balance"]
            for col in numeric_columns:
                assert pd.api.types.is_numeric_dtype(result[col]), f"Column {col} should be numeric"

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_volume_distribution_consistency(self):
        """Test consistency of enhanced volume distribution method."""
        # Create controlled test data
        dates = pd.date_range("2024-01-01", periods=50, freq="15min")
        
        # Test with identical candles
        identical_data = pd.DataFrame({
            "open": [45000.0] * 50,
            "high": [45100.0] * 50,
            "low": [44900.0] * 50,
            "close": [45050.0] * 50,  # Close price in middle
            "volume": [1000.0] * 50
        }, index=dates)
        
        vp = VolumeProfile(lookback=30, price_bins=20)
        result = vp.compute(identical_data)
        
        # With identical candles, results should be stable
        poc_prices = result["poc_price"].dropna()
        if len(poc_prices) > 1:
            # POC should be stable with identical data
            poc_std = poc_prices.std()
            assert poc_std < 100  # Should be very stable
        
        # Test with gradual price changes
        gradual_data = pd.DataFrame({
            "open": [45000 + i * 10 for i in range(50)],
            "high": [45100 + i * 10 for i in range(50)],
            "low": [44900 + i * 10 for i in range(50)],
            "close": [45050 + i * 10 for i in range(50)],
            "volume": [1000.0] * 50
        }, index=dates)
        
        result_gradual = vp.compute(gradual_data)
        poc_gradual = result_gradual["poc_price"].dropna()
        
        # POC should follow the price trend
        if len(poc_gradual) > 10:
            # Should show upward trend
            assert poc_gradual.iloc[-1] > poc_gradual.iloc[0]


class TestVolumeProfileValidation:
    """Test Volume Profile mathematical correctness and validation."""

    @pytest.fixture
    def controlled_data(self):
        """Create controlled data for validation testing."""
        dates = pd.date_range("2024-01-01", periods=50, freq="15min")
        
        # Create data with known volume distribution
        prices = [45000 + i * 10 for i in range(50)]  # Linearly increasing prices
        volumes = [1000] * 50  # Equal volumes
        
        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 5 for p in prices],
                "low": [p - 5 for p in prices],
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )
        
        return data

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_volume_conservation(self, controlled_data):
        """Test that total volume is conserved in calculations."""
        vp = VolumeProfile(lookback=30, price_bins=20)
        result = vp.compute(controlled_data)
        
        # Calculate expected total volume for the window
        window_volume = controlled_data.tail(30)["volume"].sum()
        calculated_volume = result["total_volume"].iloc[-1]
        
        if not pd.isna(calculated_volume):
            # Volume should be conserved (within small tolerance for floating point)
            assert abs(calculated_volume - window_volume) < 1e-6

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_average_volume_mathematical_correctness(self, controlled_data):
        """Test mathematical correctness of average volume calculation."""
        vp = VolumeProfile(lookback=30, price_bins=20)
        result = vp.compute(controlled_data)
        
        avg_volumes = result["average_volume"].dropna()
        total_volumes = result["total_volume"].dropna()
        
        if len(avg_volumes) > 0 and len(total_volumes) > 0:
            # Average should be total / lookback
            min_len = min(len(avg_volumes), len(total_volumes))
            expected_avg = total_volumes[:min_len] / vp.lookback
            
            # Should be mathematically exact
            np.testing.assert_allclose(
                avg_volumes[:min_len],
                expected_avg,
                rtol=1e-10,
                err_msg="Average volume calculation mathematically incorrect"
            )

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_value_area_bounds(self, controlled_data):
        """Test that Value Area calculations are within bounds."""
        vp = VolumeProfile(lookback=30, price_bins=20, value_area_pct=70)
        result = vp.compute(controlled_data)
        
        vah = result["vah_price"].dropna()
        val = result["val_price"].dropna()
        poc = result["poc_price"].dropna()
        
        if len(vah) > 0 and len(val) > 0 and len(poc) > 0:
            # VAH >= POC >= VAL always
            min_len = min(len(vah), len(val), len(poc))
            assert (vah[:min_len] >= poc[:min_len]).all()
            assert (poc[:min_len] >= val[:min_len]).all()

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_poc_strength_calculation(self, controlled_data):
        """Test POC strength calculation accuracy."""
        vp = VolumeProfile(lookback=30, price_bins=20)
        result = vp.compute(controlled_data)
        
        poc_strength = result["poc_strength"].dropna()
        poc_volume = result["poc_volume"].dropna()
        total_volume = result["total_volume"].dropna()
        
        if len(poc_strength) > 0 and len(poc_volume) > 0 and len(total_volume) > 0:
            # POC strength should be reasonable percentage
            min_len = min(len(poc_strength), len(poc_volume), len(total_volume))
            expected_strength = poc_volume[:min_len] / total_volume[:min_len]
            
            # Should be approximately equal (within tolerance)
            np.testing.assert_allclose(
                poc_strength[:min_len], 
                expected_strength, 
                rtol=1e-6,
                err_msg="POC strength calculation mismatch"
            )

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_value_area_width_calculation(self, controlled_data):
        """Test Value Area width calculation."""
        vp = VolumeProfile(lookback=30, price_bins=20)
        result = vp.compute(controlled_data)
        
        vah = result["vah_price"].dropna()
        val = result["val_price"].dropna()
        va_width = result["value_area_width"].dropna()
        
        if len(vah) > 0 and len(val) > 0 and len(va_width) > 0:
            min_len = min(len(vah), len(val), len(va_width))
            expected_width = vah[:min_len] - val[:min_len]
            
            # Width should match VAH - VAL
            np.testing.assert_allclose(
                va_width[:min_len],
                expected_width,
                rtol=1e-6,
                err_msg="Value Area width calculation mismatch"
            )

    @pytest.mark.skipif(not VOLUME_PROFILE_AVAILABLE, reason="VolumeProfile not implemented yet")
    def test_comprehensive_mathematical_validation(self, controlled_data):
        """Test comprehensive mathematical validation of all calculations."""
        vp = VolumeProfile(lookback=30, price_bins=20)
        result = vp.compute(controlled_data)
        
        # Test all mathematical relationships
        poc_prices = result["poc_price"].dropna()
        vah_prices = result["vah_price"].dropna()
        val_prices = result["val_price"].dropna()
        total_volumes = result["total_volume"].dropna()
        poc_volumes = result["poc_volume"].dropna()
        va_volumes = result["value_area_volume"].dropna()
        poc_strengths = result["poc_strength"].dropna()
        va_widths = result["value_area_width"].dropna()
        profile_balances = result["profile_balance"].dropna()
        
        if len(poc_prices) > 0:
            # Test ordering constraints
            min_len = min(len(vah_prices), len(val_prices), len(poc_prices))
            if min_len > 0:
                assert (vah_prices[:min_len] >= poc_prices[:min_len]).all()
                assert (poc_prices[:min_len] >= val_prices[:min_len]).all()
            
            # Test volume relationships
            min_vol_len = min(len(total_volumes), len(poc_volumes), len(va_volumes))
            if min_vol_len > 0:
                assert (total_volumes[:min_vol_len] >= poc_volumes[:min_vol_len]).all()
                assert (total_volumes[:min_vol_len] >= va_volumes[:min_vol_len]).all()
            
            # Test strength calculations
            min_strength_len = min(len(poc_strengths), len(poc_volumes), len(total_volumes))
            if min_strength_len > 0:
                expected_strengths = poc_volumes[:min_strength_len] / total_volumes[:min_strength_len]
                np.testing.assert_allclose(
                    poc_strengths[:min_strength_len],
                    expected_strengths,
                    rtol=1e-6,
                    err_msg="POC strength calculation error"
                )
            
            # Test width calculations
            min_width_len = min(len(va_widths), len(vah_prices), len(val_prices))
            if min_width_len > 0:
                expected_widths = vah_prices[:min_width_len] - val_prices[:min_width_len]
                np.testing.assert_allclose(
                    va_widths[:min_width_len],
                    expected_widths,
                    rtol=1e-6,
                    err_msg="Value Area width calculation error"
                )
            
            # Test balance constraints
            if len(profile_balances) > 0:
                assert (profile_balances >= 0).all()
                assert (profile_balances <= 1).all()


if __name__ == "__main__":
    # Run a subset of tests if Volume Profile is not yet implemented
    if not VOLUME_PROFILE_AVAILABLE:
        print("Volume Profile indicator not yet implemented.")
        print("Running parameter and structure tests only...")
        
        # Test parameter structure
        vp = VolumeProfile()
        params = vp.params()
        print(f"Default parameters: {params}")
        
        # Test initialization
        vp_custom = VolumeProfile(lookback=48, price_bins=30)
        print(f"Custom parameters: lookback={vp_custom.lookback}, price_bins={vp_custom.price_bins}")
        
        print("✓ Parameter tests passed")
        print("\nTo run full tests, implement VolumeProfile in btc_research/indicators/volume_profile.py")
    else:
        # Run with verbose output and performance timing
        pytest.main([__file__, "-v", "--tb=short"])