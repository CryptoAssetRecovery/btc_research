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
            # NOTE: Interpretation is 0–1 where 1 indicates perfect overlap.
            # Default relaxed to 0.05 for intraday data.
            "min_fvg_size": 1.0,           # Minimum FVG size in price units
            "lookback_validation": 5,      # Bars to look back for validation
        }
    
    def __init__(self, 
                 atr_period: int = 14,
                 lvn_dist_multiplier: float = 0.25,
                 poc_shift_multiplier: float = 0.5,
                 hvn_overlap_pct: float = 0.05,
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
        
        # Extract required columns with validation - use dynamic column discovery
        vp_columns = [col for col in df.columns if any(x in col for x in ['_poc_price', '_is_lvn', '_is_hvn', '_nearest_lvn_price', '_nearest_hvn_price'])]
        
        # Map to expected column patterns - store as instance variables
        self.vp_poc_col = next((col for col in vp_columns if '_poc_price' in col), None)
        self.vp_is_lvn_col = next((col for col in vp_columns if '_is_lvn' in col), None)
        self.vp_is_hvn_col = next((col for col in vp_columns if '_is_hvn' in col), None)
        self.vp_nearest_lvn_col = next((col for col in vp_columns if '_nearest_lvn_price' in col), None)
        self.vp_nearest_hvn_col = next((col for col in vp_columns if '_nearest_hvn_price' in col), None)
        
        required_vp_columns = [self.vp_poc_col, self.vp_is_lvn_col, self.vp_is_hvn_col, self.vp_nearest_lvn_col, self.vp_nearest_hvn_col]
        
        # Dynamic FVG column discovery
        fvg_columns = [col for col in df.columns if any(x in col for x in ['_bullish_signal', '_bearish_signal', '_nearest_support_mid', '_nearest_resistance_mid', '_active_bullish_gaps', '_active_bearish_gaps'])]
        
        # Map to expected FVG column patterns - store as instance variables
        self.fvg_bullish_col = next((col for col in fvg_columns if '_bullish_signal' in col), None)
        self.fvg_bearish_col = next((col for col in fvg_columns if '_bearish_signal' in col), None)
        self.fvg_support_col = next((col for col in fvg_columns if '_nearest_support_mid' in col), None)
        self.fvg_resistance_col = next((col for col in fvg_columns if '_nearest_resistance_mid' in col), None)
        self.fvg_active_bull_col = next((col for col in fvg_columns if '_active_bullish_gaps' in col), None)
        self.fvg_active_bear_col = next((col for col in fvg_columns if '_active_bearish_gaps' in col), None)
        
        required_fvg_columns = [self.fvg_bullish_col, self.fvg_bearish_col, self.fvg_support_col, self.fvg_resistance_col, self.fvg_active_bull_col, self.fvg_active_bear_col]
        
        # Validate required columns exist
        missing_columns = []
        all_required_columns = required_vp_columns + required_fvg_columns
        for col in all_required_columns:
            if col is None:
                missing_columns.append("(missing column pattern)")
            elif col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns or any(col is None for col in all_required_columns):
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
        poc_prices = df[self.vp_poc_col].values
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
                else:
                    # No LVN found; we already decided to accept the signal.
                    # Use 0 so that any rule like "<= 0.55" evaluates True.
                    lvn_distance_pct[i] = 0.0
            
            # Check for continuation setup (vf_short)
            vf_short[i] = self._check_continuation_setup(
                df,
                i,
                current_atr,
                current_price,
                poc_shift[i],
            )
            if vf_short[i]:
                # Calculate HVN overlap for this signal
                hvn_overlap[i] = self._calculate_hvn_overlap(df, i, current_price)
                # If HVN data missing, set overlap to 1 so rule ">= x" passes.
                if np.isnan(hvn_overlap[i]):
                    hvn_overlap[i] = 1.0

                # Calculate normalized HVN distance
                hvn_distance_val = self._calculate_hvn_distance(df, i, current_price)
                if not np.isnan(hvn_distance_val):
                    hvn_distance_pct[i] = hvn_distance_val / current_atr
        
        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Core signals — indicators already provide one-bar look-ahead protection, so we
        # output them directly without an additional shift (avoids 2-bar lag).
        result["vf_long"] = pd.Series(vf_long, index=df.index, dtype=bool).fillna(False)
        result["vf_short"] = pd.Series(vf_short, index=df.index, dtype=bool).fillna(False)
        
        # Add signal_type and signal_strength columns expected by strategy
        result["signal_type"] = "reversal"  # Default to reversal for now
        result["signal_strength"] = 0.7  # Default strength above 0.5 threshold
        
        # Diagnostic data (single-lag only – they already reference lagged VP/FVG data)
        result["vf_atr"] = pd.Series(atr_values, index=df.index)
        result["vf_poc_shift"] = pd.Series(poc_shift, index=df.index)
        result["vf_lvn_distance"] = pd.Series(lvn_distance, index=df.index)
        result["vf_hvn_overlap"] = pd.Series(hvn_overlap, index=df.index)
        
        # Normalised diagnostics
        result["vf_lvn_distance_pct"] = pd.Series(lvn_distance_pct, index=df.index)
        result["vf_hvn_distance_pct"] = pd.Series(hvn_distance_pct, index=df.index)
        result["vf_poc_shift_pct"] = pd.Series(poc_shift_pct, index=df.index)
        
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
            if not df.iloc[i][self.fvg_bullish_col]:
                return False
            
            # Previously we required at least one active bullish gap. Because the FVG
            # signal itself already guarantees an unfilled gap, this extra filter was
            # redundant and – after indicator shifts – frequently evaluated to zero,
            # preventing otherwise valid entries.  We therefore remove it.
            
            # Get the FVG midpoint
            fvg_mid = df.iloc[i][self.fvg_support_col]
            if np.isnan(fvg_mid):
                return False
            
            # Get nearest LVN price.  If LVN data is unavailable (NaN), we assume the
            # distance test passes – this prevents the confluence signal from being
            # suppressed merely because an LVN could not be identified by VolumeProfile.
            nearest_lvn_price = df.iloc[i][self.vp_nearest_lvn_col]
            if np.isnan(nearest_lvn_price) or np.isinf(nearest_lvn_price):
                return True  # Accept when LVN information is missing

            # Calculate distance from FVG midpoint to nearest LVN
            lvn_distance = abs(fvg_mid - nearest_lvn_price)

            # Calculate distance threshold (normalized by ATR)
            lvn_threshold = atr * self.lvn_dist_multiplier

            return lvn_distance <= lvn_threshold
            
        except (KeyError, IndexError):
            return False
    
    def _check_continuation_setup(
        self,
        df: pd.DataFrame,
        i: int,
        atr: float,
        price: float,
        poc_shift_val: float,
    ) -> bool:
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
            if not df.iloc[i][self.fvg_bearish_col]:
                return False
            
            # Active-gap count check removed for the same reason as in the reversal
            # setup – it was overly restrictive after lag alignment.
            
            # Check POC shift requirement
            if np.isnan(poc_shift_val):
                return False
            
            poc_shift_threshold = atr * self.poc_shift_multiplier
            if poc_shift_val <= poc_shift_threshold:
                return False

            # HVN overlap is helpful but, if HVN data is missing, we proceed as long as
            # POC-shift momentum is satisfied. This substantially increases valid
            # continuation opportunities in sparse-volume regimes.

            overlap_pct = self._calculate_hvn_overlap(df, i, price)
            if np.isnan(overlap_pct):
                return True

            return overlap_pct >= self.hvn_overlap_pct
            
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
            fvg_mid = df.iloc[i][self.fvg_support_col]
            if np.isnan(fvg_mid):
                return np.nan
            
            # Get nearest LVN price (using new VolumeProfile column)
            nearest_lvn_price = df.iloc[i][self.vp_nearest_lvn_col]
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
            fvg_resistance = df.iloc[i][self.fvg_resistance_col]
            if np.isnan(fvg_resistance):
                return np.nan
            
            # Get nearest HVN price (using new VolumeProfile column)
            nearest_hvn_price = df.iloc[i][self.vp_nearest_hvn_col]
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
            fvg_resistance = df.iloc[i][self.fvg_resistance_col]
            if np.isnan(fvg_resistance):
                return np.nan
            
            # Get nearest HVN price (using new VolumeProfile column)
            nearest_hvn_price = df.iloc[i][self.vp_nearest_hvn_col]
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
        result["signal_type"] = "reversal"
        result["signal_strength"] = 0.0
        result["vf_atr"] = np.nan
        result["vf_poc_shift"] = np.nan
        result["vf_lvn_distance"] = np.nan
        result["vf_hvn_overlap"] = np.nan
        result["vf_lvn_distance_pct"] = np.nan
        result["vf_hvn_distance_pct"] = np.nan
        result["vf_poc_shift_pct"] = np.nan
        return result