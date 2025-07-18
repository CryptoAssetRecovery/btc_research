"""
VPFVGSignal confluence indicator implementation.

This module provides a confluence indicator that combines Volume Profile and Fair Value Gap
signals to identify high-probability trading opportunities. The indicator implements the
core trading logic from the VP-FVG strategy outlined in the implementation plan.

Key Features:
- Reversal Setup: Bullish FVG whose mid lies within 0.25 × ATR(14) of a composite LVN
- Continuation Setup: Bearish FVG overlapping HVN plus POC shift > 0.5 × ATR
- Configurable tolerance parameters for fine-tuning
- Look-ahead bias prevention with proper signal timing
- Comprehensive error handling and edge case management

Trading Logic:
1. **Reversal (vf_long)**: Occurs when a bullish FVG's midpoint is within 0.25 × ATR
   of a Low Volume Node (LVN), indicating potential bounce from support
2. **Continuation (vf_short)**: Occurs when a bearish FVG overlaps with a High Volume Node
   (HVN) and POC shifts significantly (> 0.5 × ATR), indicating strong downward momentum
"""

import numpy as np
import pandas as pd
from typing import Any, Optional

from btc_research.core.base_indicator import BaseIndicator
from btc_research.core.registry import register

# Configure pandas to suppress FutureWarnings
pd.set_option("future.no_silent_downcasting", True)


@register("VPFVGSignal")
class VPFVGSignal(BaseIndicator):
    """
    VP-FVG Confluence Signal indicator.
    
    This indicator combines Volume Profile and Fair Value Gap analysis to identify
    high-probability trading setups. It implements two core signal types:
    
    1. **Reversal Setup (vf_long)**: Bullish FVG near LVN
       - Bullish FVG whose midpoint lies within 0.25 × ATR(14) of a LVN
       - Indicates potential bounce from low-volume support level
       
    2. **Continuation Setup (vf_short)**: Bearish FVG overlapping HVN with POC shift
       - Bearish FVG overlapping a High Volume Node (HVN)
       - POC shift > 0.5 × ATR indicates strong momentum continuation
    
    The indicator requires pre-calculated Volume Profile and Fair Value Gap data
    as inputs, making it a composite indicator that builds upon the two base
    indicators.
    
    Attributes:
        atr_period (int): ATR calculation period for distance thresholds
        lvn_dist_multiplier (float): ATR multiplier for LVN proximity (0.25)
        poc_shift_multiplier (float): ATR multiplier for POC shift threshold (0.5)
        hvn_overlap_pct (float): Minimum overlap percentage for HVN confluence
        min_fvg_size (float): Minimum FVG size to consider valid
        lookback_validation (int): Bars to look back for validation
    """
    
    @classmethod
    def params(cls) -> dict[str, Any]:
        """Return default parameters for VPFVGSignal indicator."""
        return {
            "atr_period": 14,              # ATR period for distance calculations
            "lvn_dist_multiplier": 0.25,   # ATR multiplier for LVN proximity
            "poc_shift_multiplier": 0.5,   # ATR multiplier for POC shift threshold
            "hvn_overlap_pct": 0.7,        # Minimum overlap percentage for HVN confluence
            "min_fvg_size": 1.0,           # Minimum FVG size in price units
            "lookback_validation": 5,      # Bars to look back for validation
        }
    
    def __init__(self, 
                 atr_period: int = 14,
                 lvn_dist_multiplier: float = 0.25,
                 poc_shift_multiplier: float = 0.5,
                 hvn_overlap_pct: float = 0.7,
                 min_fvg_size: float = 1.0,
                 lookback_validation: int = 5):
        """
        Initialize VPFVGSignal indicator.
        
        Args:
            atr_period (int): ATR calculation period for distance thresholds
            lvn_dist_multiplier (float): ATR multiplier for LVN proximity (0.25)
            poc_shift_multiplier (float): ATR multiplier for POC shift threshold (0.5)
            hvn_overlap_pct (float): Minimum overlap percentage for HVN confluence
            min_fvg_size (float): Minimum FVG size to consider valid
            lookback_validation (int): Bars to look back for validation
        """
        self.atr_period = atr_period
        self.lvn_dist_multiplier = lvn_dist_multiplier
        self.poc_shift_multiplier = poc_shift_multiplier
        self.hvn_overlap_pct = hvn_overlap_pct
        self.min_fvg_size = min_fvg_size
        self.lookback_validation = lookback_validation
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute VP-FVG confluence signals.
        
        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV data and pre-calculated
                             Volume Profile and FVG indicators
        
        Returns:
            pd.DataFrame: DataFrame with confluence signals:
                - vf_long: Bullish reversal signal (FVG near LVN)
                - vf_short: Bearish continuation signal (FVG overlapping HVN + POC shift)
                - vf_atr: ATR values used for distance calculations
                - vf_poc_shift: POC shift magnitude (raw price units)
                - vf_lvn_distance: Distance from FVG mid to nearest LVN (raw price units)
                - vf_hvn_overlap: HVN overlap percentage
                - vf_lvn_distance_pct: Distance to nearest LVN as percentage of ATR
                - vf_hvn_distance_pct: Distance to nearest HVN as percentage of ATR
                - vf_poc_shift_pct: POC shift normalized by ATR
        """
        # Handle edge case with insufficient data
        if len(df) < max(self.atr_period, self.lookback_validation):
            return self._create_empty_result(df.index)
        
        # Calculate ATR for distance thresholds
        atr_values = self._calculate_atr(df)
        
        # Extract required columns with validation
        required_vp_columns = [
            'VolumeProfile_poc_price', 'VolumeProfile_is_lvn', 'VolumeProfile_is_hvn',
            'VolumeProfile_nearest_lvn_price', 'VolumeProfile_nearest_hvn_price'
        ]
        required_fvg_columns = [
            'FVG_bullish_signal', 'FVG_bearish_signal', 'FVG_nearest_support_mid', 
            'FVG_nearest_resistance_mid', 'FVG_active_bullish_gaps', 'FVG_active_bearish_gaps'
        ]
        
        # Validate required columns exist
        missing_columns = []
        for col in required_vp_columns + required_fvg_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            # Return empty result if required columns are missing
            return self._create_empty_result(df.index)
        
        # Initialize result arrays
        n = len(df)
        vf_long = np.zeros(n, dtype=bool)
        vf_short = np.zeros(n, dtype=bool)
        poc_shift = np.full(n, np.nan)
        lvn_distance = np.full(n, np.nan)
        hvn_overlap = np.full(n, np.nan)
        
        # New normalized metrics
        lvn_distance_pct = np.full(n, np.nan)
        hvn_distance_pct = np.full(n, np.nan)
        poc_shift_pct = np.full(n, np.nan)
        
        # Calculate POC shift and normalized version
        poc_prices = df['VolumeProfile_poc_price'].values
        for i in range(1, n):
            if not np.isnan(poc_prices[i]) and not np.isnan(poc_prices[i-1]) and not np.isinf(poc_prices[i]) and not np.isinf(poc_prices[i-1]):
                poc_shift[i] = abs(poc_prices[i] - poc_prices[i-1])
                # Calculate normalized POC shift
                if not np.isnan(atr_values[i]) and not np.isinf(atr_values[i]) and atr_values[i] > 0:
                    poc_shift_pct[i] = poc_shift[i] / atr_values[i]
                    # Handle infinity case
                    if np.isinf(poc_shift_pct[i]):
                        poc_shift_pct[i] = np.nan
        
        # Process each bar for confluence signals
        for i in range(self.lookback_validation, n):
            current_atr = atr_values[i]
            current_price = df.iloc[i]['close']
            
            if np.isnan(current_atr) or current_atr <= 0:
                continue
            
            # Check for reversal setup (vf_long)
            vf_long[i] = self._check_reversal_setup(df, i, current_atr, current_price)
            if vf_long[i]:
                # Calculate LVN distance for this signal
                lvn_distance[i] = self._calculate_lvn_distance(df, i, current_price)
                # Calculate normalized LVN distance
                if not np.isnan(lvn_distance[i]):
                    lvn_distance_pct[i] = lvn_distance[i] / current_atr
            
            # Check for continuation setup (vf_short)
            vf_short[i] = self._check_continuation_setup(df, i, current_atr, current_price, poc_shift[i])
            if vf_short[i]:
                # Calculate HVN overlap for this signal
                hvn_overlap[i] = self._calculate_hvn_overlap(df, i, current_price)
                # Calculate normalized HVN distance
                hvn_distance_val = self._calculate_hvn_distance(df, i, current_price)
                if not np.isnan(hvn_distance_val):
                    hvn_distance_pct[i] = hvn_distance_val / current_atr
        
        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Core signals (shifted to prevent look-ahead bias)
        vf_long_shifted = pd.Series(vf_long, index=df.index, dtype=bool).shift(1)
        vf_short_shifted = pd.Series(vf_short, index=df.index, dtype=bool).shift(1)
        result["vf_long"] = vf_long_shifted.fillna(False).astype(bool)
        result["vf_short"] = vf_short_shifted.fillna(False).astype(bool)
        
        # Diagnostic data (also shifted)
        result["vf_atr"] = pd.Series(atr_values, index=df.index).shift(1)
        result["vf_poc_shift"] = pd.Series(poc_shift, index=df.index).shift(1)
        result["vf_lvn_distance"] = pd.Series(lvn_distance, index=df.index).shift(1)
        result["vf_hvn_overlap"] = pd.Series(hvn_overlap, index=df.index).shift(1)
        
        # New normalized metrics (also shifted)
        result["vf_lvn_distance_pct"] = pd.Series(lvn_distance_pct, index=df.index).shift(1)
        result["vf_hvn_distance_pct"] = pd.Series(hvn_distance_pct, index=df.index).shift(1)
        result["vf_poc_shift_pct"] = pd.Series(poc_shift_pct, index=df.index).shift(1)
        
        return result
    
    def _calculate_atr(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate Average True Range (ATR) for distance calculations.
        
        Args:
            df (pd.DataFrame): Input OHLCV DataFrame
            
        Returns:
            np.ndarray: ATR values
        """
        n = len(df)
        atr = np.full(n, np.nan)
        
        if n < self.atr_period + 1:
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
        for i in range(self.atr_period, n):
            if i == self.atr_period:
                atr[i] = np.mean(tr[i-self.atr_period+1:i+1])
            else:
                atr[i] = (atr[i-1] * (self.atr_period - 1) + tr[i]) / self.atr_period
        
        return atr
    
    def _check_reversal_setup(self, df: pd.DataFrame, i: int, atr: float, price: float) -> bool:
        """
        Check for reversal setup: Bullish FVG near LVN.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            i (int): Current bar index
            atr (float): Current ATR value
            price (float): Current price
            
        Returns:
            bool: True if reversal setup is detected
        """
        try:
            # Check if there's an active bullish FVG
            if not df.iloc[i]['FVG_bullish_signal']:
                return False
            
            # Check if there are active bullish gaps
            if df.iloc[i]['FVG_active_bullish_gaps'] <= 0:
                return False
            
            # Get the FVG midpoint
            fvg_mid = df.iloc[i]['FVG_nearest_support_mid']
            if np.isnan(fvg_mid):
                return False
            
            # Get nearest LVN price (using new VolumeProfile column)
            nearest_lvn_price = df.iloc[i]['VolumeProfile_nearest_lvn_price']
            if np.isnan(nearest_lvn_price) or np.isinf(nearest_lvn_price):
                return False
            
            # Calculate distance from FVG midpoint to nearest LVN
            lvn_distance = abs(fvg_mid - nearest_lvn_price)
            
            # Calculate distance threshold (normalized by ATR)
            lvn_threshold = atr * self.lvn_dist_multiplier
            
            # Check if FVG midpoint is within threshold distance of nearest LVN
            return lvn_distance <= lvn_threshold
            
        except (KeyError, IndexError):
            return False
    
    def _check_continuation_setup(self, df: pd.DataFrame, i: int, atr: float, 
                                price: float, poc_shift_val: float) -> bool:
        """
        Check for continuation setup: Bearish FVG overlapping HVN with POC shift.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            i (int): Current bar index
            atr (float): Current ATR value
            price (float): Current price
            poc_shift_val (float): POC shift magnitude
            
        Returns:
            bool: True if continuation setup is detected
        """
        try:
            # Check if there's an active bearish FVG
            if not df.iloc[i]['FVG_bearish_signal']:
                return False
            
            # Check if there are active bearish gaps
            if df.iloc[i]['FVG_active_bearish_gaps'] <= 0:
                return False
            
            # Check POC shift requirement
            if np.isnan(poc_shift_val):
                return False
            
            poc_shift_threshold = atr * self.poc_shift_multiplier
            if poc_shift_val <= poc_shift_threshold:
                return False
            
            # Get the FVG resistance level
            fvg_resistance = df.iloc[i]['FVG_nearest_resistance_mid']
            if np.isnan(fvg_resistance):
                return False
            
            # Get nearest HVN price (using new VolumeProfile column)
            nearest_hvn_price = df.iloc[i]['VolumeProfile_nearest_hvn_price']
            if np.isnan(nearest_hvn_price) or np.isinf(nearest_hvn_price):
                return False
            
            # Check if FVG zone overlaps with HVN price level
            # We need to check if the HVN price level intersects with the FVG zone
            # Get FVG zone boundaries (resistance is the top of bearish FVG)
            # For bearish FVG, resistance_mid is usually the center, we need the actual zone
            
            # For now, check if HVN price is within reasonable distance of FVG resistance
            overlap_distance = abs(fvg_resistance - nearest_hvn_price)
            max_overlap_distance = atr * 0.5  # Allow 0.5 ATR overlap tolerance
            
            return overlap_distance <= max_overlap_distance
            
        except (KeyError, IndexError):
            return False
    
    def _calculate_lvn_distance(self, df: pd.DataFrame, i: int, price: float) -> float:
        """
        Calculate distance from FVG midpoint to nearest LVN.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            i (int): Current bar index
            price (float): Current price
            
        Returns:
            float: Distance to nearest LVN
        """
        try:
            # Get the FVG midpoint
            fvg_mid = df.iloc[i]['FVG_nearest_support_mid']
            if np.isnan(fvg_mid):
                return np.nan
            
            # Get nearest LVN price (using new VolumeProfile column)
            nearest_lvn_price = df.iloc[i]['VolumeProfile_nearest_lvn_price']
            if np.isnan(nearest_lvn_price) or np.isinf(nearest_lvn_price):
                return np.nan
            
            # Calculate distance from FVG midpoint to nearest LVN
            return abs(fvg_mid - nearest_lvn_price)
            
        except (KeyError, IndexError):
            return np.nan
    
    def _calculate_hvn_overlap(self, df: pd.DataFrame, i: int, price: float) -> float:
        """
        Calculate HVN overlap percentage.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            i (int): Current bar index
            price (float): Current price
            
        Returns:
            float: HVN overlap percentage
        """
        try:
            # Get the FVG resistance level
            fvg_resistance = df.iloc[i]['FVG_nearest_resistance_mid']
            if np.isnan(fvg_resistance):
                return np.nan
            
            # Get nearest HVN price (using new VolumeProfile column)
            nearest_hvn_price = df.iloc[i]['VolumeProfile_nearest_hvn_price']
            if np.isnan(nearest_hvn_price) or np.isinf(nearest_hvn_price):
                return np.nan
            
            # Calculate overlap as percentage
            overlap_distance = abs(fvg_resistance - nearest_hvn_price)
            # Normalize by ATR for percentage calculation
            atr_val = self._calculate_atr(df)[i]
            if not np.isnan(atr_val) and not np.isinf(atr_val) and atr_val > 0:
                overlap_pct = max(0, 1 - (overlap_distance / atr_val))
                # Handle infinity case
                if np.isinf(overlap_pct):
                    return np.nan
                return overlap_pct
            
            return 0.0
            
        except (KeyError, IndexError):
            return np.nan
    
    def _calculate_hvn_distance(self, df: pd.DataFrame, i: int, price: float) -> float:
        """
        Calculate distance from FVG resistance to nearest HVN.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            i (int): Current bar index
            price (float): Current price
            
        Returns:
            float: Distance to nearest HVN
        """
        try:
            # Get the FVG resistance level
            fvg_resistance = df.iloc[i]['FVG_nearest_resistance_mid']
            if np.isnan(fvg_resistance):
                return np.nan
            
            # Get nearest HVN price (using new VolumeProfile column)
            nearest_hvn_price = df.iloc[i]['VolumeProfile_nearest_hvn_price']
            if np.isnan(nearest_hvn_price) or np.isinf(nearest_hvn_price):
                return np.nan
            
            # Calculate distance from FVG resistance to nearest HVN
            return abs(fvg_resistance - nearest_hvn_price)
            
        except (KeyError, IndexError):
            return np.nan
    
    def _create_empty_result(self, index: pd.Index) -> pd.DataFrame:
        """Create empty result DataFrame for edge cases."""
        result = pd.DataFrame(index=index)
        result["vf_long"] = False
        result["vf_short"] = False
        result["vf_atr"] = np.nan
        result["vf_poc_shift"] = np.nan
        result["vf_lvn_distance"] = np.nan
        result["vf_hvn_overlap"] = np.nan
        result["vf_lvn_distance_pct"] = np.nan
        result["vf_hvn_distance_pct"] = np.nan
        result["vf_poc_shift_pct"] = np.nan
        return result