"""
Enhanced Rolling Volume Profile indicator implementation.

This module provides an enhanced rolling Volume Profile indicator that analyzes volume
distribution across price levels within a rolling window. It identifies key
support/resistance levels using Point of Control (POC) and Value Area calculations.

Key enhancements:
- Extended 5-day lookback period for better support/resistance identification
- HVN/LVN (High/Low Volume Node) classification for volume-based signals
- Distance metrics for confluence analysis
- ATR-based POC breakout detection for volatility-adjusted signals
- Simplified 50/50 volume distribution for better performance
- Vectorized operations for improved calculation speed
"""

import numpy as np
import pandas as pd
from typing import Any

from btc_research.core.base_indicator import BaseIndicator
from btc_research.core.registry import register

# Configure pandas to suppress FutureWarnings
pd.set_option("future.no_silent_downcasting", True)


@register("VolumeProfile")
class VolumeProfile(BaseIndicator):
    """
    Enhanced Rolling Volume Profile indicator.

    Analyzes volume distribution across price levels within a rolling window to identify
    key support/resistance levels. The indicator calculates Point of Control (POC) as
    the price level with highest volume, and Value Area as the price range containing
    a specified percentage of total volume around the POC.

    Key concepts:
    - POC (Point of Control): Price level with the highest volume
    - Value Area: Price range containing specified percentage of volume around POC
    - Volume Distribution: How volume is spread across different price levels
    - HVN/LVN: High/Low Volume Nodes for volume-based signal classification
    - Distance Metrics: Price distance to key levels for confluence analysis
    - ATR-based Breakouts: Volatility-adjusted POC breakout detection

    Attributes:
        lookback (int): Number of candles for rolling window calculation (default: 480)
        lookback_hours (int): Lookback period in hours (default: 120)
        price_bins (int): Number of price levels to create for volume distribution
        value_area_pct (float): Percentage for Value Area calculation (typically 70%)
        update_frequency (int): Recalculate every N candles for performance
        poc_sensitivity (float): POC change threshold as percentage (deprecated)
        min_volume_threshold (float): Minimum volume per bin as percentage of total
        hvn_threshold (float): High Volume Node threshold as multiple of average
        lvn_threshold (float): Low Volume Node threshold as multiple of average
        atr_period (int): ATR period for POC breakout detection
        atr_multiplier (float): ATR multiplier for breakout threshold
        enable_vectorization (bool): Enable vectorized operations for performance
    """

    @classmethod
    def params(cls) -> dict[str, Any]:
        """Return default parameters for Volume Profile indicator."""
        return {
            "lookback": 480,              # Number of candles (5 days = 480 * 15m)
            "lookback_hours": 120,        # Lookback period in hours (5 days = 120 hours)
            "price_bins": 50,             # Number of price levels to create
            "value_area_pct": 70,         # Percentage for Value Area calculation
            "update_frequency": 1,        # Recalculate every N candles
            # poc_sensitivity parameter removed - use ATR-based detection instead
            "min_volume_threshold": 0.01, # Minimum volume per bin (percentage of total)
            "hvn_threshold": 1.5,         # High Volume Node threshold (multiple of average)
            "lvn_threshold": 0.5,         # Low Volume Node threshold (multiple of average)
            "atr_period": 14,             # ATR period for POC breakout detection
            "atr_multiplier": 2.0,        # ATR multiplier for breakout threshold
            "enable_vectorization": True   # Enable vectorized operations for performance
        }

    def __init__(self, lookback: int = 480, lookback_hours: int = 120, price_bins: int = 50, 
                 value_area_pct: float = 70, update_frequency: int = 1,
                 min_volume_threshold: float = 0.01, hvn_threshold: float = 1.5,
                 lvn_threshold: float = 0.5, atr_period: int = 14, atr_multiplier: float = 2.0,
                 enable_vectorization: bool = True):
        """
        Initialize Volume Profile indicator.

        Args:
            lookback (int): Number of candles for rolling window. Default is 480 (5 days for 15m).
            lookback_hours (int): Lookback period in hours. Default is 120 (5 days).
            price_bins (int): Number of price levels to create. Default is 50.
            value_area_pct (float): Percentage for Value Area calculation. Default is 70%.
            update_frequency (int): Recalculate every N candles. Default is 1.
            # poc_sensitivity parameter removed - use ATR-based detection instead
            min_volume_threshold (float): Minimum volume per bin as percentage. Default is 0.01%.
            hvn_threshold (float): High Volume Node threshold as multiple of average. Default is 1.5.
            lvn_threshold (float): Low Volume Node threshold as multiple of average. Default is 0.5.
            atr_period (int): ATR period for POC breakout detection. Default is 14.
            atr_multiplier (float): ATR multiplier for breakout threshold. Default is 2.0.
            enable_vectorization (bool): Enable vectorized operations. Default is True.
        """
        self.lookback = lookback
        self.lookback_hours = lookback_hours
        self.price_bins = price_bins
        self.value_area_pct = value_area_pct / 100.0  # Convert to decimal
        self.update_frequency = update_frequency
        # poc_sensitivity parameter removed - use ATR-based detection instead
        self.min_volume_threshold = min_volume_threshold / 100.0  # Convert to decimal
        self.hvn_threshold = hvn_threshold
        self.lvn_threshold = lvn_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.enable_vectorization = enable_vectorization

        # Cache for performance optimization
        self._last_calculation_index = -1
        self._cached_results = {}
        self._atr_cache = []

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Volume Profile metrics and trading signals.

        Args:
            df (pd.DataFrame): Input OHLCV DataFrame

        Returns:
            pd.DataFrame: DataFrame with Volume Profile metrics and signals
        """
        # Handle edge case with insufficient data
        if len(df) < max(3, self.lookback):
            return self._create_empty_result(df.index)
        
        # Use vectorized operations when enabled and data is large enough
        if self.enable_vectorization and len(df) > 1000:
            return self._compute_vectorized(df)
        else:
            return self._compute_iterative(df)
    
    def _compute_iterative(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Volume Profile using iterative method (original implementation).
        
        Args:
            df (pd.DataFrame): Input OHLCV DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with Volume Profile metrics and signals
        """

        # Initialize result arrays
        n = len(df)
        poc_price = np.full(n, np.nan)
        vah_price = np.full(n, np.nan)
        val_price = np.full(n, np.nan)
        total_volume = np.full(n, np.nan)
        poc_volume = np.full(n, np.nan)
        value_area_volume = np.full(n, np.nan)
        average_volume = np.full(n, np.nan)
        
        # Trading signals
        price_above_poc = np.zeros(n, dtype=bool)
        price_below_poc = np.zeros(n, dtype=bool)
        price_in_value_area = np.zeros(n, dtype=bool)
        poc_breakout = np.zeros(n, dtype=bool)
        volume_spike = np.zeros(n, dtype=bool)
        
        # HVN/LVN classification
        is_hvn = np.zeros(n, dtype=bool)
        is_lvn = np.zeros(n, dtype=bool)
        
        # Nearest HVN/LVN price tracking
        nearest_hvn_price = np.full(n, np.nan)
        nearest_lvn_price = np.full(n, np.nan)
        
        # Distance metrics
        dist_to_poc = np.full(n, np.nan)
        dist_to_vah = np.full(n, np.nan)
        dist_to_val = np.full(n, np.nan)
        
        # Additional analytics
        poc_strength = np.full(n, np.nan)
        value_area_width = np.full(n, np.nan)
        profile_balance = np.full(n, np.nan)
        
        # Calculate ATR for POC breakout detection
        atr_values = self._calculate_atr(df, self.atr_period)

        # Calculate for each position with sufficient lookback
        for i in range(self.lookback - 1, n):
            # Skip calculation if update frequency not met (for performance)
            if i > self.lookback - 1 and (i - self._last_calculation_index) < self.update_frequency:
                # Copy previous values
                if i > 0:
                    poc_price[i] = poc_price[i-1]
                    vah_price[i] = vah_price[i-1]
                    val_price[i] = val_price[i-1]
                    total_volume[i] = total_volume[i-1]
                    poc_volume[i] = poc_volume[i-1]
                    value_area_volume[i] = value_area_volume[i-1]
                    average_volume[i] = average_volume[i-1]
                    poc_strength[i] = poc_strength[i-1]
                    value_area_width[i] = value_area_width[i-1]
                    profile_balance[i] = profile_balance[i-1]
                continue

            # Get rolling window data
            start_idx = max(0, i - self.lookback + 1)
            window_data = df.iloc[start_idx:i+1].copy()
            
            if len(window_data) < 2:
                continue

            # Calculate volume profile for this window
            vp_results = self._calculate_volume_profile(window_data)
            
            if vp_results is None:
                continue

            # Store results
            poc_price[i] = vp_results['poc_price']
            vah_price[i] = vp_results['vah_price']
            val_price[i] = vp_results['val_price']
            total_volume[i] = vp_results['total_volume']
            poc_volume[i] = vp_results['poc_volume']
            value_area_volume[i] = vp_results['value_area_volume']
            average_volume[i] = vp_results['total_volume'] / len(window_data)  # Calculate average volume
            poc_strength[i] = vp_results['poc_strength']
            value_area_width[i] = vp_results['value_area_width']
            profile_balance[i] = vp_results['profile_balance']

            # Generate trading signals
            current_price = df.iloc[i]['close']
            current_volume = df.iloc[i]['volume']
            
            # Price position signals
            if not np.isnan(poc_price[i]):
                price_above_poc[i] = current_price > poc_price[i]
                price_below_poc[i] = current_price < poc_price[i]
                
                # Distance metrics
                dist_to_poc[i] = current_price - poc_price[i]
                
                # ATR-based POC breakout detection
                if i > 0 and not np.isnan(poc_price[i-1]) and not np.isnan(atr_values[i]):
                    poc_change = abs(poc_price[i] - poc_price[i-1])
                    atr_threshold = atr_values[i] * self.atr_multiplier
                    poc_breakout[i] = poc_change > atr_threshold

            # Value Area signals and distance metrics
            if not np.isnan(vah_price[i]) and not np.isnan(val_price[i]):
                price_in_value_area[i] = val_price[i] <= current_price <= vah_price[i]
                dist_to_vah[i] = current_price - vah_price[i]
                dist_to_val[i] = current_price - val_price[i]

            # Volume spike detection
            if not np.isnan(average_volume[i]) and average_volume[i] > 0:
                volume_spike[i] = current_volume > (average_volume[i] * 1.5)
                
            # HVN/LVN classification and nearest node tracking
            if vp_results is not None and 'bin_volumes' in vp_results:
                bin_volumes_array = vp_results['bin_volumes']
                bin_centers = vp_results.get('bin_centers', [])
                if len(bin_volumes_array) > 0 and len(bin_centers) == len(bin_volumes_array):
                    avg_bin_volume = np.mean(bin_volumes_array)
                    if avg_bin_volume > 0:
                        # Find the bin containing current price
                        current_bin_idx = vp_results.get('current_price_bin', -1)
                        if 0 <= current_bin_idx < len(bin_volumes_array):
                            current_bin_volume = bin_volumes_array[current_bin_idx]
                            is_hvn[i] = current_bin_volume > (avg_bin_volume * self.hvn_threshold)
                            is_lvn[i] = current_bin_volume < (avg_bin_volume * self.lvn_threshold)
                        
                        # Find nearest HVN and LVN nodes
                        hvn_nodes, lvn_nodes = self._find_hvn_lvn_nodes(bin_volumes_array, bin_centers, avg_bin_volume)
                        nearest_hvn_price[i], nearest_lvn_price[i] = self._find_nearest_nodes(
                            current_price, hvn_nodes, lvn_nodes
                        )

            self._last_calculation_index = i

        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Core metrics (Engine will add indicator ID prefix)
        # IMPORTANT: Shift all VP fields by one bar to prevent look-ahead bias
        result["poc_price"] = pd.Series(poc_price, index=df.index).shift(1)
        result["vah_price"] = pd.Series(vah_price, index=df.index).shift(1)
        result["val_price"] = pd.Series(val_price, index=df.index).shift(1)
        result["total_volume"] = pd.Series(total_volume, index=df.index).shift(1)
        result["poc_volume"] = pd.Series(poc_volume, index=df.index).shift(1)
        result["value_area_volume"] = pd.Series(value_area_volume, index=df.index).shift(1)
        result["average_volume"] = pd.Series(average_volume, index=df.index).shift(1)
        
        # Trading signals - recalculate based on shifted VP data
        current_close = df['close']
        result["price_above_poc"] = current_close > result["poc_price"]
        result["price_below_poc"] = current_close < result["poc_price"]
        result["price_in_value_area"] = (current_close >= result["val_price"]) & (current_close <= result["vah_price"])
        result["poc_breakout"] = pd.Series(poc_breakout, index=df.index).shift(1)
        result["volume_spike"] = pd.Series(volume_spike, index=df.index).shift(1)
        
        # HVN/LVN classification
        result["is_hvn"] = pd.Series(is_hvn, index=df.index).shift(1)
        result["is_lvn"] = pd.Series(is_lvn, index=df.index).shift(1)
        
        # Nearest HVN/LVN prices with forward-fill and look-ahead bias prevention
        result["nearest_hvn_price"] = pd.Series(nearest_hvn_price, index=df.index).shift(1).ffill()
        result["nearest_lvn_price"] = pd.Series(nearest_lvn_price, index=df.index).shift(1).ffill()
        
        # Distance metrics - recalculate based on shifted VP data
        result["dist_to_poc"] = current_close - result["poc_price"]
        result["dist_to_vah"] = current_close - result["vah_price"]
        result["dist_to_val"] = current_close - result["val_price"]
        
        # Additional analytics
        result["poc_strength"] = pd.Series(poc_strength, index=df.index).shift(1)
        result["value_area_width"] = pd.Series(value_area_width, index=df.index).shift(1)
        result["profile_balance"] = pd.Series(profile_balance, index=df.index).shift(1)

        return result
    
    def _compute_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Volume Profile using vectorized operations for better performance.
        
        This is a performance-optimized version that uses numpy vectorized operations
        to reduce computation time for large datasets.
        
        Args:
            df (pd.DataFrame): Input OHLCV DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with Volume Profile metrics and signals
        """
        # Initialize result arrays
        n = len(df)
        poc_price = np.full(n, np.nan)
        vah_price = np.full(n, np.nan)
        val_price = np.full(n, np.nan)
        total_volume = np.full(n, np.nan)
        poc_volume = np.full(n, np.nan)
        value_area_volume = np.full(n, np.nan)
        average_volume = np.full(n, np.nan)
        
        # Trading signals
        price_above_poc = np.zeros(n, dtype=bool)
        price_below_poc = np.zeros(n, dtype=bool)
        price_in_value_area = np.zeros(n, dtype=bool)
        poc_breakout = np.zeros(n, dtype=bool)
        volume_spike = np.zeros(n, dtype=bool)
        
        # HVN/LVN classification
        is_hvn = np.zeros(n, dtype=bool)
        is_lvn = np.zeros(n, dtype=bool)
        
        # Nearest HVN/LVN price tracking
        nearest_hvn_price = np.full(n, np.nan)
        nearest_lvn_price = np.full(n, np.nan)
        
        # Distance metrics
        dist_to_poc = np.full(n, np.nan)
        dist_to_vah = np.full(n, np.nan)
        dist_to_val = np.full(n, np.nan)
        
        # Additional analytics
        poc_strength = np.full(n, np.nan)
        value_area_width = np.full(n, np.nan)
        profile_balance = np.full(n, np.nan)
        
        # Calculate ATR for POC breakout detection (vectorized)
        atr_values = self._calculate_atr_vectorized(df, self.atr_period)
        
        # Pre-compute rolling windows for batch processing
        high_vals = df['high'].values
        low_vals = df['low'].values
        close_vals = df['close'].values
        volume_vals = df['volume'].values
        
        # Process in batches for better memory efficiency
        batch_size = min(100, n // 4) if n > 400 else n
        
        for batch_start in range(self.lookback - 1, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            
            # Process batch
            for i in range(batch_start, batch_end):
                # Skip calculation if update frequency not met
                if i > self.lookback - 1 and (i - self._last_calculation_index) < self.update_frequency:
                    if i > 0:
                        poc_price[i] = poc_price[i-1]
                        vah_price[i] = vah_price[i-1]
                        val_price[i] = val_price[i-1]
                        total_volume[i] = total_volume[i-1]
                        poc_volume[i] = poc_volume[i-1]
                        value_area_volume[i] = value_area_volume[i-1]
                        average_volume[i] = average_volume[i-1]
                        poc_strength[i] = poc_strength[i-1]
                        value_area_width[i] = value_area_width[i-1]
                        profile_balance[i] = profile_balance[i-1]
                    continue
                
                # Get rolling window indices
                start_idx = max(0, i - self.lookback + 1)
                
                # Calculate volume profile using vectorized window operations
                vp_results = self._calculate_volume_profile_vectorized(
                    high_vals[start_idx:i+1],
                    low_vals[start_idx:i+1], 
                    close_vals[start_idx:i+1],
                    volume_vals[start_idx:i+1]
                )
                
                if vp_results is None:
                    continue
                
                # Store results
                poc_price[i] = vp_results['poc_price']
                vah_price[i] = vp_results['vah_price']
                val_price[i] = vp_results['val_price']
                total_volume[i] = vp_results['total_volume']
                poc_volume[i] = vp_results['poc_volume']
                value_area_volume[i] = vp_results['value_area_volume']
                average_volume[i] = vp_results['total_volume'] / (i - start_idx + 1)
                poc_strength[i] = vp_results['poc_strength']
                value_area_width[i] = vp_results['value_area_width']
                profile_balance[i] = vp_results['profile_balance']
                
                # Generate trading signals
                current_price = close_vals[i]
                current_volume = volume_vals[i]
                
                # Price position signals
                if not np.isnan(poc_price[i]):
                    price_above_poc[i] = current_price > poc_price[i]
                    price_below_poc[i] = current_price < poc_price[i]
                    dist_to_poc[i] = current_price - poc_price[i]
                    
                    # ATR-based POC breakout detection
                    if i > 0 and not np.isnan(poc_price[i-1]) and not np.isnan(atr_values[i]):
                        poc_change = abs(poc_price[i] - poc_price[i-1])
                        atr_threshold = atr_values[i] * self.atr_multiplier
                        poc_breakout[i] = poc_change > atr_threshold
                
                # Value Area signals and distance metrics
                if not np.isnan(vah_price[i]) and not np.isnan(val_price[i]):
                    price_in_value_area[i] = val_price[i] <= current_price <= vah_price[i]
                    dist_to_vah[i] = current_price - vah_price[i]
                    dist_to_val[i] = current_price - val_price[i]
                
                # Volume spike detection
                if not np.isnan(average_volume[i]) and average_volume[i] > 0:
                    volume_spike[i] = current_volume > (average_volume[i] * 1.5)
                
                # HVN/LVN classification and nearest node tracking
                if vp_results is not None and 'bin_volumes' in vp_results:
                    bin_volumes_array = vp_results['bin_volumes']
                    bin_centers = vp_results.get('bin_centers', [])
                    if len(bin_volumes_array) > 0 and len(bin_centers) == len(bin_volumes_array):
                        avg_bin_volume = np.mean(bin_volumes_array)
                        if avg_bin_volume > 0:
                            # Find the bin containing current price
                            current_bin_idx = vp_results.get('current_price_bin', -1)
                            if 0 <= current_bin_idx < len(bin_volumes_array):
                                current_bin_volume = bin_volumes_array[current_bin_idx]
                                is_hvn[i] = current_bin_volume > (avg_bin_volume * self.hvn_threshold)
                                is_lvn[i] = current_bin_volume < (avg_bin_volume * self.lvn_threshold)
                            
                            # Find nearest HVN and LVN nodes
                            hvn_nodes, lvn_nodes = self._find_hvn_lvn_nodes(bin_volumes_array, bin_centers, avg_bin_volume)
                            nearest_hvn_price[i], nearest_lvn_price[i] = self._find_nearest_nodes(
                                current_price, hvn_nodes, lvn_nodes
                            )
                
                self._last_calculation_index = i
        
        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Core metrics (Engine will add indicator ID prefix)
        # IMPORTANT: Shift all VP fields by one bar to prevent look-ahead bias
        result["poc_price"] = pd.Series(poc_price, index=df.index).shift(1)
        result["vah_price"] = pd.Series(vah_price, index=df.index).shift(1)
        result["val_price"] = pd.Series(val_price, index=df.index).shift(1)
        result["total_volume"] = pd.Series(total_volume, index=df.index).shift(1)
        result["poc_volume"] = pd.Series(poc_volume, index=df.index).shift(1)
        result["value_area_volume"] = pd.Series(value_area_volume, index=df.index).shift(1)
        result["average_volume"] = pd.Series(average_volume, index=df.index).shift(1)
        
        # Trading signals - recalculate based on shifted VP data
        current_close = df['close']
        result["price_above_poc"] = current_close > result["poc_price"]
        result["price_below_poc"] = current_close < result["poc_price"]
        result["price_in_value_area"] = (current_close >= result["val_price"]) & (current_close <= result["vah_price"])
        result["poc_breakout"] = pd.Series(poc_breakout, index=df.index).shift(1)
        result["volume_spike"] = pd.Series(volume_spike, index=df.index).shift(1)
        
        # HVN/LVN classification
        result["is_hvn"] = pd.Series(is_hvn, index=df.index).shift(1)
        result["is_lvn"] = pd.Series(is_lvn, index=df.index).shift(1)
        
        # Nearest HVN/LVN prices with forward-fill and look-ahead bias prevention
        result["nearest_hvn_price"] = pd.Series(nearest_hvn_price, index=df.index).shift(1).ffill()
        result["nearest_lvn_price"] = pd.Series(nearest_lvn_price, index=df.index).shift(1).ffill()
        
        # Distance metrics - recalculate based on shifted VP data
        result["dist_to_poc"] = current_close - result["poc_price"]
        result["dist_to_vah"] = current_close - result["vah_price"]
        result["dist_to_val"] = current_close - result["val_price"]
        
        # Additional analytics
        result["poc_strength"] = pd.Series(poc_strength, index=df.index).shift(1)
        result["value_area_width"] = pd.Series(value_area_width, index=df.index).shift(1)
        result["profile_balance"] = pd.Series(profile_balance, index=df.index).shift(1)
        
        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """
        Calculate Average True Range (ATR) for POC breakout detection.
        
        Args:
            df (pd.DataFrame): Input OHLCV DataFrame
            period (int): ATR calculation period
            
        Returns:
            np.ndarray: ATR values
        """
        n = len(df)
        atr = np.full(n, np.nan)
        
        if n < period + 1:
            return atr
            
        # Calculate True Range
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]  # First TR is just high - low
        
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Calculate ATR using exponential moving average
        for i in range(period, n):
            if i == period:
                atr[i] = np.mean(tr[i-period+1:i+1])
            else:
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
                
        return atr
    
    def _calculate_atr_vectorized(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """
        Calculate Average True Range (ATR) using vectorized operations.
        
        Args:
            df (pd.DataFrame): Input OHLCV DataFrame
            period (int): ATR calculation period
            
        Returns:
            np.ndarray: ATR values
        """
        n = len(df)
        atr = np.full(n, np.nan)
        
        if n < period + 1:
            return atr
            
        # Get price arrays
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate True Range using vectorized operations
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        
        # Vectorized TR calculation for all periods at once
        if n > 1:
            tr[1:] = np.maximum.reduce([
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            ])
        
        # Calculate ATR using Wilder's EMA method (consistent with iterative version)
        for i in range(period, n):
            if i == period:
                atr[i] = np.mean(tr[i-period+1:i+1])
            else:
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    def _calculate_volume_profile_vectorized(self, high_vals: np.ndarray, low_vals: np.ndarray, 
                                           close_vals: np.ndarray, volume_vals: np.ndarray) -> dict:
        """
        Calculate volume profile metrics using vectorized operations.
        
        Args:
            high_vals (np.ndarray): High prices for window
            low_vals (np.ndarray): Low prices for window
            close_vals (np.ndarray): Close prices for window
            volume_vals (np.ndarray): Volume values for window
            
        Returns:
            dict: Volume profile metrics or None if calculation fails
        """
        try:
            # Filter out invalid data
            valid_mask = (~np.isnan(high_vals)) & (~np.isnan(low_vals)) & (~np.isnan(volume_vals)) & (volume_vals > 0)
            
            if not np.any(valid_mask):
                return None
            
            high_vals = high_vals[valid_mask]
            low_vals = low_vals[valid_mask]
            volume_vals = volume_vals[valid_mask]
            
            # Get price range for binning
            price_min = np.min(low_vals)
            price_max = np.max(high_vals)
            
            if price_min >= price_max or np.isnan(price_min) or np.isnan(price_max):
                return None
            
            # Create price bins
            bin_edges = np.linspace(price_min, price_max, self.price_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_volumes = np.zeros(self.price_bins)
            
            # Vectorized volume distribution
            for i in range(len(high_vals)):
                if volume_vals[i] <= 0:
                    continue
                    
                # Find bins that overlap with price range
                low_bin = np.searchsorted(bin_edges, low_vals[i], side='left') - 1
                high_bin = np.searchsorted(bin_edges, high_vals[i], side='right') - 1
                
                # Clamp to valid indices
                low_bin = max(0, min(low_bin, self.price_bins - 1))
                high_bin = max(0, min(high_bin, self.price_bins - 1))
                
                if low_bin > high_bin:
                    low_bin, high_bin = high_bin, low_bin
                
                # Distribute volume evenly across affected bins
                affected_bins = high_bin - low_bin + 1
                volume_per_bin = volume_vals[i] / affected_bins
                
                bin_volumes[low_bin:high_bin+1] += volume_per_bin
            
            # Calculate total volume
            total_vol = np.sum(bin_volumes)
            if total_vol == 0:
                return None
            
            # Apply minimum volume threshold to filter noise
            min_volume_for_bin = total_vol * self.min_volume_threshold
            filtered_bin_volumes = np.where(bin_volumes >= min_volume_for_bin, bin_volumes, 0)
            
            # Find POC (vectorized) after filtering
            poc_bin_idx = np.argmax(filtered_bin_volumes)
            poc_price_val = bin_centers[poc_bin_idx]
            poc_volume_val = filtered_bin_volumes[poc_bin_idx]
            
            # Calculate Value Area (optimized) using filtered volumes
            va_results = self._calculate_value_area_vectorized(filtered_bin_volumes, bin_centers, poc_bin_idx)
            
            # Calculate additional metrics
            poc_strength_val = poc_volume_val / total_vol if total_vol > 0 else 0
            value_area_width_val = va_results['vah'] - va_results['val'] if va_results else 0
            
            # Calculate profile balance using filtered volumes
            profile_balance_val = self._calculate_profile_balance_vectorized(filtered_bin_volumes, poc_bin_idx)
            
            # Find current price bin for HVN/LVN classification
            current_price = close_vals[-1] if len(close_vals) > 0 else np.nan
            current_bin_idx = -1
            if not np.isnan(current_price):
                current_bin_idx = np.searchsorted(bin_edges, current_price, side='right') - 1
                current_bin_idx = max(0, min(current_bin_idx, len(bin_volumes) - 1))
            
            return {
                'poc_price': poc_price_val,
                'vah_price': va_results['vah'] if va_results else np.nan,
                'val_price': va_results['val'] if va_results else np.nan,
                'total_volume': total_vol,
                'poc_volume': poc_volume_val,
                'value_area_volume': va_results['volume'] if va_results else np.nan,
                'poc_strength': poc_strength_val,
                'value_area_width': value_area_width_val,
                'profile_balance': profile_balance_val,
                'bin_volumes': filtered_bin_volumes,
                'bin_centers': bin_centers,
                'current_price_bin': current_bin_idx
            }
            
        except Exception:
            return None
    
    def _calculate_value_area_vectorized(self, bin_volumes: np.ndarray, bin_centers: np.ndarray, 
                                       poc_bin_idx: int) -> dict:
        """
        Calculate Value Area using vectorized operations.
        
        Args:
            bin_volumes (np.ndarray): Volume for each price bin
            bin_centers (np.ndarray): Price centers for each bin
            poc_bin_idx (int): Index of the POC bin
            
        Returns:
            dict: Value Area High, Low, and contained volume
        """
        total_volume = np.sum(bin_volumes)
        target_volume = total_volume * self.value_area_pct
        
        if total_volume == 0 or target_volume == 0:
            return None
        
        # Start from POC and expand outward
        accumulated_volume = bin_volumes[poc_bin_idx]
        lower_bound = poc_bin_idx
        upper_bound = poc_bin_idx
        
        # Expand using vectorized operations where possible
        while accumulated_volume < target_volume and (lower_bound > 0 or upper_bound < len(bin_volumes) - 1):
            # Get adjacent volumes
            lower_volume = bin_volumes[lower_bound - 1] if lower_bound > 0 else 0
            upper_volume = bin_volumes[upper_bound + 1] if upper_bound < len(bin_volumes) - 1 else 0
            
            # Choose expansion direction
            if lower_bound > 0 and upper_bound < len(bin_volumes) - 1:
                if lower_volume >= upper_volume:
                    lower_bound -= 1
                    accumulated_volume += bin_volumes[lower_bound]
                else:
                    upper_bound += 1
                    accumulated_volume += bin_volumes[upper_bound]
            elif lower_bound > 0:
                lower_bound -= 1
                accumulated_volume += bin_volumes[lower_bound]
            elif upper_bound < len(bin_volumes) - 1:
                upper_bound += 1
                accumulated_volume += bin_volumes[upper_bound]
            else:
                break
        
        return {
            'val': bin_centers[lower_bound],
            'vah': bin_centers[upper_bound],
            'volume': accumulated_volume
        }
    
    def _calculate_profile_balance_vectorized(self, bin_volumes: np.ndarray, poc_bin_idx: int) -> float:
        """
        Calculate profile balance using vectorized operations.
        
        Args:
            bin_volumes (np.ndarray): Volume for each price bin
            poc_bin_idx (int): Index of the POC bin
            
        Returns:
            float: Balance score (0-1, higher = more balanced)
        """
        if len(bin_volumes) < 3:
            return 0.5
        
        # Vectorized volume calculations
        left_volume = np.sum(bin_volumes[:poc_bin_idx])
        right_volume = np.sum(bin_volumes[poc_bin_idx + 1:])
        
        if left_volume == 0 and right_volume == 0:
            return 1.0
        elif left_volume == 0 or right_volume == 0:
            return 0.0
        else:
            return min(left_volume, right_volume) / max(left_volume, right_volume)

    def _create_empty_result(self, index: pd.Index) -> pd.DataFrame:
        """Create empty result DataFrame for edge cases."""
        result = pd.DataFrame(index=index)
        
        # Core metrics (Engine will add indicator ID prefix)
        result["poc_price"] = np.nan
        result["vah_price"] = np.nan
        result["val_price"] = np.nan
        result["total_volume"] = np.nan
        result["poc_volume"] = np.nan
        result["value_area_volume"] = np.nan
        result["average_volume"] = np.nan
        
        # Trading signals
        result["price_above_poc"] = False
        result["price_below_poc"] = False
        result["price_in_value_area"] = False
        result["poc_breakout"] = False
        result["volume_spike"] = False
        
        # HVN/LVN classification
        result["is_hvn"] = False
        result["is_lvn"] = False
        
        # Nearest HVN/LVN prices
        result["nearest_hvn_price"] = np.nan
        result["nearest_lvn_price"] = np.nan
        
        # Distance metrics
        result["dist_to_poc"] = np.nan
        result["dist_to_vah"] = np.nan
        result["dist_to_val"] = np.nan
        
        # Additional analytics
        result["poc_strength"] = np.nan
        result["value_area_width"] = np.nan
        result["profile_balance"] = np.nan
        
        return result

    def _calculate_volume_profile(self, window_data: pd.DataFrame) -> dict:
        """
        Calculate volume profile metrics for a given window of data.

        Args:
            window_data (pd.DataFrame): OHLCV data for the rolling window

        Returns:
            dict: Volume profile metrics or None if calculation fails
        """
        try:
            # Get price range for binning
            price_min = window_data['low'].min()
            price_max = window_data['high'].max()
            
            if price_min >= price_max or np.isnan(price_min) or np.isnan(price_max):
                return None

            # Create price bins
            bin_size = (price_max - price_min) / self.price_bins
            if bin_size == 0:
                return None
                
            bin_edges = np.linspace(price_min, price_max, self.price_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Initialize volume array for each bin
            bin_volumes = np.zeros(self.price_bins)
            
            # Distribute volume across bins for each candle
            for _, candle in window_data.iterrows():
                if pd.isna(candle['volume']) or candle['volume'] <= 0:
                    continue
                    
                # Distribute volume based on OHLC using triangular distribution
                candle_volume = self._distribute_candle_volume(
                    candle, bin_edges, bin_centers
                )
                bin_volumes += candle_volume

            # Calculate total volume
            total_vol = np.sum(bin_volumes)
            if total_vol == 0:
                return None

            # Apply minimum volume threshold to filter noise
            min_volume_for_bin = total_vol * self.min_volume_threshold
            filtered_bin_volumes = np.where(bin_volumes >= min_volume_for_bin, bin_volumes, 0)
            
            # Find Point of Control (highest volume bin) after filtering
            poc_bin_idx = np.argmax(filtered_bin_volumes)
            poc_price_val = bin_centers[poc_bin_idx]
            poc_volume_val = filtered_bin_volumes[poc_bin_idx]

            # Calculate Value Area using filtered volumes
            va_results = self._calculate_value_area(filtered_bin_volumes, bin_centers, poc_bin_idx)
            
            # Calculate additional metrics
            poc_strength_val = poc_volume_val / total_vol if total_vol > 0 else 0
            value_area_width_val = va_results['vah'] - va_results['val'] if va_results else 0
            
            # Calculate profile balance (symmetry around POC) using filtered volumes
            profile_balance_val = self._calculate_profile_balance(
                filtered_bin_volumes, poc_bin_idx
            )

            # Find current price bin for HVN/LVN classification
            current_price = window_data['close'].iloc[-1] if len(window_data) > 0 else np.nan
            current_bin_idx = -1
            if not np.isnan(current_price):
                current_bin_idx = np.searchsorted(bin_edges, current_price, side='right') - 1
                current_bin_idx = max(0, min(current_bin_idx, len(bin_volumes) - 1))
            
            return {
                'poc_price': poc_price_val,
                'vah_price': va_results['vah'] if va_results else np.nan,
                'val_price': va_results['val'] if va_results else np.nan,
                'total_volume': total_vol,
                'poc_volume': poc_volume_val,
                'value_area_volume': va_results['volume'] if va_results else np.nan,
                'poc_strength': poc_strength_val,
                'value_area_width': value_area_width_val,
                'profile_balance': profile_balance_val,
                'bin_volumes': filtered_bin_volumes,
                'bin_centers': bin_centers,
                'current_price_bin': current_bin_idx
            }

        except Exception:
            return None

    def _distribute_candle_volume(self, candle: pd.Series, bin_edges: np.ndarray, 
                                bin_centers: np.ndarray) -> np.ndarray:
        """
        Distribute a single candle's volume across price bins using simplified 50/50 distribution.

        Uses a simplified approach where volume is distributed 50/50 between high and low,
        removing the complex triangular weighting for better performance and clarity.

        Args:
            candle (pd.Series): Single candle data with OHLCV
            bin_edges (np.ndarray): Price bin edges
            bin_centers (np.ndarray): Price bin centers

        Returns:
            np.ndarray: Volume distribution across bins
        """
        volume_dist = np.zeros(len(bin_centers))
        
        # Get candle price range
        high_price = candle['high']
        low_price = candle['low']
        volume = candle['volume']
        
        if pd.isna(high_price) or pd.isna(low_price) or pd.isna(volume) or volume <= 0:
            return volume_dist
        
        # Find bins that overlap with this candle's price range
        # Use digitize to find which bins the high and low fall into
        low_bin = np.digitize(low_price, bin_edges) - 1
        high_bin = np.digitize(high_price, bin_edges) - 1
        
        # Clamp to valid bin indices
        low_bin = max(0, min(low_bin, len(bin_centers) - 1))
        high_bin = max(0, min(high_bin, len(bin_centers) - 1))
        
        if low_bin > high_bin:
            low_bin, high_bin = high_bin, low_bin
        
        # Calculate number of affected bins
        affected_bins = high_bin - low_bin + 1
        if affected_bins <= 0:
            affected_bins = 1
            
        # Simplified 50/50 distribution: spread volume evenly across affected bins
        # No complex weighting - just equal distribution
        volume_per_bin = volume / affected_bins
        
        for bin_idx in range(low_bin, high_bin + 1):
            volume_dist[bin_idx] = volume_per_bin
            
        return volume_dist

    def _calculate_value_area(self, bin_volumes: np.ndarray, bin_centers: np.ndarray, 
                            poc_bin_idx: int) -> dict:
        """
        Calculate Value Area (price range containing specified percentage of volume).

        The Value Area contains the specified percentage of total volume centered
        around the Point of Control.

        Args:
            bin_volumes (np.ndarray): Volume for each price bin
            bin_centers (np.ndarray): Price centers for each bin
            poc_bin_idx (int): Index of the POC bin

        Returns:
            dict: Value Area High, Low, and contained volume
        """
        total_volume = np.sum(bin_volumes)
        target_volume = total_volume * self.value_area_pct
        
        if total_volume == 0 or target_volume == 0:
            return None
        
        # Start from POC and expand outward
        accumulated_volume = bin_volumes[poc_bin_idx]
        lower_bound = poc_bin_idx
        upper_bound = poc_bin_idx
        
        # Expand the range until we reach target volume
        while accumulated_volume < target_volume and (lower_bound > 0 or upper_bound < len(bin_volumes) - 1):
            # Check which direction to expand (higher volume gets priority)
            expand_lower = False
            expand_upper = False
            
            lower_volume = bin_volumes[lower_bound - 1] if lower_bound > 0 else 0
            upper_volume = bin_volumes[upper_bound + 1] if upper_bound < len(bin_volumes) - 1 else 0
            
            if lower_bound > 0 and upper_bound < len(bin_volumes) - 1:
                # Both directions available, choose the one with higher volume
                if lower_volume >= upper_volume:
                    expand_lower = True
                else:
                    expand_upper = True
            elif lower_bound > 0:
                expand_lower = True
            elif upper_bound < len(bin_volumes) - 1:
                expand_upper = True
            else:
                break  # No more expansion possible
            
            # Perform expansion
            if expand_lower:
                lower_bound -= 1
                accumulated_volume += bin_volumes[lower_bound]
            elif expand_upper:
                upper_bound += 1
                accumulated_volume += bin_volumes[upper_bound]
        
        # Get Value Area High and Low prices
        val_price = bin_centers[lower_bound]
        vah_price = bin_centers[upper_bound]
        
        return {
            'val': val_price,
            'vah': vah_price,
            'volume': accumulated_volume
        }

    def _calculate_profile_balance(self, bin_volumes: np.ndarray, poc_bin_idx: int) -> float:
        """
        Calculate profile balance (symmetry of volume distribution around POC).

        Returns a value between 0 and 1, where 1 indicates perfect symmetry.

        Args:
            bin_volumes (np.ndarray): Volume for each price bin
            poc_bin_idx (int): Index of the POC bin

        Returns:
            float: Balance score (0-1, higher = more balanced)
        """
        if len(bin_volumes) < 3:
            return 0.5
        
        # Calculate volume on each side of POC
        left_volume = np.sum(bin_volumes[:poc_bin_idx])
        right_volume = np.sum(bin_volumes[poc_bin_idx + 1:])
        
        total_side_volume = left_volume + right_volume
        if total_side_volume == 0:
            return 1.0  # Perfect balance when no volume on sides
        
        # Calculate balance ratio
        if left_volume == 0 and right_volume == 0:
            return 1.0
        elif left_volume == 0 or right_volume == 0:
            return 0.0
        else:
            ratio = min(left_volume, right_volume) / max(left_volume, right_volume)
            return ratio
    
    def _find_hvn_lvn_nodes(self, bin_volumes: np.ndarray, bin_centers: np.ndarray, avg_bin_volume: float) -> tuple:
        """
        Find all HVN (High Volume Node) and LVN (Low Volume Node) price levels in the volume profile.
        
        Args:
            bin_volumes (np.ndarray): Volume for each price bin
            bin_centers (np.ndarray): Price centers for each bin
            avg_bin_volume (float): Average volume across all bins
            
        Returns:
            tuple: (hvn_nodes, lvn_nodes) where each is a list of price levels
        """
        hvn_nodes = []
        lvn_nodes = []
        
        # Find HVN nodes - bins with volume above threshold
        hvn_threshold = avg_bin_volume * self.hvn_threshold
        hvn_mask = bin_volumes > hvn_threshold
        hvn_nodes = bin_centers[hvn_mask].tolist()
        
        # Find LVN nodes - bins with volume below threshold (but not empty)
        lvn_threshold = avg_bin_volume * self.lvn_threshold
        lvn_mask = (bin_volumes < lvn_threshold) & (bin_volumes > 0)
        lvn_nodes = bin_centers[lvn_mask].tolist()
        
        return hvn_nodes, lvn_nodes
    
    def _find_nearest_nodes(self, current_price: float, hvn_nodes: list, lvn_nodes: list) -> tuple:
        """
        Find the nearest HVN and LVN price levels to the current price.
        
        Args:
            current_price (float): Current price to find nearest nodes for
            hvn_nodes (list): List of HVN price levels
            lvn_nodes (list): List of LVN price levels
            
        Returns:
            tuple: (nearest_hvn_price, nearest_lvn_price)
        """
        nearest_hvn_price = np.nan
        nearest_lvn_price = np.nan
        
        # Handle invalid current price
        if np.isnan(current_price) or np.isinf(current_price):
            return nearest_hvn_price, nearest_lvn_price
        
        # Find nearest HVN with better error handling
        if hvn_nodes and len(hvn_nodes) > 0:
            try:
                # Filter out NaN values from hvn_nodes
                valid_hvn_nodes = [hvn for hvn in hvn_nodes if not np.isnan(hvn) and not np.isinf(hvn)]
                if valid_hvn_nodes:
                    hvn_distances = [abs(current_price - hvn) for hvn in valid_hvn_nodes]
                    if hvn_distances:  # Check if distances list is not empty
                        nearest_hvn_idx = np.argmin(hvn_distances)
                        nearest_hvn_price = valid_hvn_nodes[nearest_hvn_idx]
            except (ValueError, TypeError, IndexError):
                # Fallback to NaN if calculation fails
                nearest_hvn_price = np.nan
        
        # Find nearest LVN with better error handling
        if lvn_nodes and len(lvn_nodes) > 0:
            try:
                # Filter out NaN values from lvn_nodes
                valid_lvn_nodes = [lvn for lvn in lvn_nodes if not np.isnan(lvn) and not np.isinf(lvn)]
                if valid_lvn_nodes:
                    lvn_distances = [abs(current_price - lvn) for lvn in valid_lvn_nodes]
                    if lvn_distances:  # Check if distances list is not empty
                        nearest_lvn_idx = np.argmin(lvn_distances)
                        nearest_lvn_price = valid_lvn_nodes[nearest_lvn_idx]
            except (ValueError, TypeError, IndexError):
                # Fallback to NaN if calculation fails
                nearest_lvn_price = np.nan
        
        return nearest_hvn_price, nearest_lvn_price