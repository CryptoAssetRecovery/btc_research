"""
Fair Value Gap (FVG) indicator implementation.

This module provides a Fair Value Gap indicator based on ICT (Inner Circle Trader)
methodology. FVGs identify imbalance areas in price action where gaps occur between
three consecutive candles, creating potential support/resistance zones.
"""


import numpy as np
import pandas as pd

from btc_research.core.base_indicator import BaseIndicator
from btc_research.core.registry import register

# Configure pandas to suppress FutureWarnings
pd.set_option("future.no_silent_downcasting", True)


@register("FVG")
class FVG(BaseIndicator):
    """
    Fair Value Gap (FVG) indicator with Volume Profile confluence analysis.

    Detects 3-candle gap patterns where price leaves unfilled areas that often
    act as future support/resistance zones. Based on ICT trading methodology.
    Enhanced with Volume Profile bin-index tagging for confluence analysis.

    A Fair Value Gap occurs when:
    - Bullish FVG: low[i+2] > high[i] (gap up)
    - Bearish FVG: high[i+2] < low[i] (gap down)

    Key Features:
    - Look-ahead bias prevention: All signals are shifted by 1 period
    - Mid-price calculation: Provides gap center for distance calculations
    - Gap tracking: Monitors active gaps and detects when they are filled
    - VP bin-index tagging: Tags each gap with its corresponding VP bin for confluence
    - Distance metrics: Placeholder for HVN/LVN distance calculations

    Attributes:
        min_gap_pips (float): Minimum gap size to filter noise
        max_lookback (int): Maximum candles to look back for gap detection
        vp_bin_edges (np.ndarray): Volume Profile bin edges for confluence analysis
        vp_bin_centers (np.ndarray): Volume Profile bin centers for confluence analysis
    """

    @classmethod
    def params(cls):
        """Return default parameters for FVG indicator."""
        return {
            "min_gap_pips": 1.0,    # Minimum gap size in pips/points
            "max_lookback": 500,    # Maximum historical gaps to track
            "vp_bin_edges": None,   # Volume Profile bin edges for confluence analysis
            "vp_bin_centers": None  # Volume Profile bin centers for confluence analysis
        }

    def __init__(self, min_gap_pips: float = 1.0, max_lookback: int = 500, 
                 vp_bin_edges: np.ndarray = None, vp_bin_centers: np.ndarray = None):
        """
        Initialize FVG indicator.

        Args:
            min_gap_pips (float): Minimum gap size to consider valid. Default is 1.0.
            max_lookback (int): Maximum number of historical gaps to track. Default is 500.
            vp_bin_edges (np.ndarray): Volume Profile bin edges for bin tagging. Optional.
            vp_bin_centers (np.ndarray): Volume Profile bin centers for bin tagging. Optional.
        """
        self.min_gap_pips = min_gap_pips
        self.max_lookback = max_lookback
        self.vp_bin_edges = vp_bin_edges
        self.vp_bin_centers = vp_bin_centers

    def compute(self, df: pd.DataFrame, vp_bin_edges: np.ndarray = None, 
                vp_bin_centers: np.ndarray = None) -> pd.DataFrame:
        """
        Compute Fair Value Gaps and generate trading signals.

        Args:
            df (pd.DataFrame): Input OHLCV DataFrame
            vp_bin_edges (np.ndarray): Volume Profile bin edges for bin tagging. Optional.
            vp_bin_centers (np.ndarray): Volume Profile bin centers for bin tagging. Optional.

        Returns:
            pd.DataFrame: DataFrame with FVG zones and signals
        """
        # Handle edge case with insufficient data
        if len(df) < 3:
            return self._create_empty_result(df.index)

        # Use provided VP bin info or instance variables
        if vp_bin_edges is not None:
            self.vp_bin_edges = vp_bin_edges
        if vp_bin_centers is not None:
            self.vp_bin_centers = vp_bin_centers

        # Reset index to work with integer positions
        df_reset = df.reset_index()

        # Detect all FVG patterns
        bullish_gaps, bearish_gaps = self._detect_gaps(df_reset)

        # Track gap status and generate signals
        result = self._generate_signals(df_reset, bullish_gaps, bearish_gaps)

        # Restore original index
        result.index = df.index

        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range (ATR) for dynamic gap size filtering.
        
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

    def _get_bin_index(self, price: float) -> int:
        """
        Get the bin index for a given price using VP bin edges.
        
        Args:
            price (float): Price to find bin index for
            
        Returns:
            int: Bin index (-1 if VP bins not available)
        """
        if self.vp_bin_edges is None or len(self.vp_bin_edges) == 0:
            return -1
        
        # Use np.searchsorted to find which bin the price falls into
        bin_idx = np.searchsorted(self.vp_bin_edges, price, side='right') - 1
        
        # Clamp to valid bin range
        bin_idx = max(0, min(bin_idx, len(self.vp_bin_edges) - 2))
        
        return bin_idx

    def _extract_gap_bin_indices(self, gaps: list[dict], df: pd.DataFrame, gap_type: str) -> np.ndarray:
        """
        Extract bin indices for gaps at each time point.
        
        Args:
            gaps (list[dict]): List of gap dictionaries
            df (pd.DataFrame): Price dataframe
            gap_type (str): 'bullish' or 'bearish'
            
        Returns:
            np.ndarray: Bin indices for each time point (-1 if no gap)
        """
        n = len(df)
        bin_indices = np.full(n, -1, dtype=int)
        
        for i in range(n):
            # Find the most recent gap that was created at or before this time point
            # We need to check gap filled status dynamically since gap filling is tracked in _generate_signals
            current_price = df.iloc[i]["close"]
            current_high = df.iloc[i]["high"]
            current_low = df.iloc[i]["low"]
            
            active_gaps = []
            for gap in gaps:
                if gap["created_at"] <= i:
                    # Check if gap is filled at this time point
                    gap_filled = False
                    if gap_type == 'bullish':
                        gap_filled = current_low <= gap["bottom"]
                    else:  # bearish
                        gap_filled = current_high >= gap["top"]
                    
                    if not gap_filled:
                        active_gaps.append(gap)
            
            if active_gaps:
                # Use the most recent active gap
                most_recent_gap = max(active_gaps, key=lambda g: g["created_at"])
                bin_indices[i] = most_recent_gap["fvg_bin_idx"]
        
        return bin_indices

    def _create_empty_result(self, index: pd.Index) -> pd.DataFrame:
        """Create empty result DataFrame for edge cases."""
        result = pd.DataFrame(index=index)
        result["FVG_bullish_signal"] = False
        result["FVG_bearish_signal"] = False
        result["FVG_gap_filled"] = False
        result["FVG_nearest_support"] = float("nan")
        result["FVG_nearest_resistance"] = float("nan")
        result["FVG_nearest_support_mid"] = float("nan")
        result["FVG_nearest_resistance_mid"] = float("nan")
        result["FVG_active_bullish_gaps"] = 0
        result["FVG_active_bearish_gaps"] = 0
        result["FVG_bullish_bin_idx"] = -1
        result["FVG_bearish_bin_idx"] = -1
        result["FVG_lvn_dist"] = float("nan")
        result["FVG_hvn_dist"] = float("nan")
        return result

    def _detect_gaps(self, df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        """
        Detect all Fair Value Gaps in the price data.

        Args:
            df (pd.DataFrame): OHLCV data with integer index

        Returns:
            Tuple[List[dict], List[dict]]: (bullish_gaps, bearish_gaps)
        """
        bullish_gaps = []
        bearish_gaps = []
        
        # Calculate ATR for dynamic min_gap_pips
        atr_values = self._calculate_atr(df)

        # Iterate through dataframe looking for 3-candle patterns
        for i in range(len(df) - 2):
            # Get the three candles
            candle1 = df.iloc[i]
            candle3 = df.iloc[i + 2]
            
            # Calculate dynamic minimum gap size: max(0.5 × ATR, 0.1% of price)
            current_price = candle3["close"]
            atr_val = atr_values[i + 2] if i + 2 < len(atr_values) else np.nan
            
            if np.isnan(atr_val):
                # Fallback to static value if ATR not available
                dynamic_min_gap = self.min_gap_pips
            else:
                dynamic_min_gap = max(0.25 * atr_val, 0.0005 * current_price)   # 0.25 ATR or 5 bp

            # Check for bullish FVG: low[i+2] > high[i]
            if candle3["low"] > candle1["high"]:
                gap_size = candle3["low"] - candle1["high"]
                if gap_size >= dynamic_min_gap:
                    gap_top = candle3["low"]
                    gap_bottom = candle1["high"]
                    gap_mid = (gap_top + gap_bottom) / 2
                    gap_bin_idx = self._get_bin_index(gap_mid)
                    
                    bullish_gaps.append({
                        "start_idx": i,
                        "end_idx": i + 2,
                        "top": gap_top,
                        "bottom": gap_bottom,
                        "mid": gap_mid,
                        "size": gap_size,
                        "filled": False,
                        "created_at": i + 2,
                        "fvg_bin_idx": gap_bin_idx
                    })

            # Check for bearish FVG: high[i+2] < low[i]
            elif candle3["high"] < candle1["low"]:
                gap_size = candle1["low"] - candle3["high"]
                if gap_size >= dynamic_min_gap:
                    gap_top = candle1["low"]
                    gap_bottom = candle3["high"]
                    gap_mid = (gap_top + gap_bottom) / 2
                    gap_bin_idx = self._get_bin_index(gap_mid)
                    
                    bearish_gaps.append({
                        "start_idx": i,
                        "end_idx": i + 2,
                        "top": gap_top,
                        "bottom": gap_bottom,
                        "mid": gap_mid,
                        "size": gap_size,
                        "filled": False,
                        "created_at": i + 2,
                        "fvg_bin_idx": gap_bin_idx
                    })

        return bullish_gaps, bearish_gaps

    def _generate_signals(self, df: pd.DataFrame, bullish_gaps: list[dict],
                         bearish_gaps: list[dict]) -> pd.DataFrame:
        """
        Generate trading signals based on price interaction with FVG zones.

        Args:
            df (pd.DataFrame): OHLCV data
            bullish_gaps (List[dict]): List of detected bullish gaps
            bearish_gaps (List[dict]): List of detected bearish gaps

        Returns:
            pd.DataFrame: Result with signals and gap information
        """
        # Initialize result arrays
        n = len(df)
        bullish_signals = np.zeros(n, dtype=bool)
        bearish_signals = np.zeros(n, dtype=bool)
        gap_filled = np.zeros(n, dtype=bool)
        nearest_support = np.full(n, np.nan)
        nearest_resistance = np.full(n, np.nan)
        nearest_support_mid = np.full(n, np.nan)
        nearest_resistance_mid = np.full(n, np.nan)
        active_bullish = np.zeros(n, dtype=int)
        active_bearish = np.zeros(n, dtype=int)

        # Process each candle
        for i in range(n):
            current_price = df.iloc[i]["close"]
            current_high = df.iloc[i]["high"]
            current_low = df.iloc[i]["low"]

            # Track gap status and generate signals
            active_bull_gaps = []
            active_bear_gaps = []

            # Process bullish gaps
            for gap in bullish_gaps:
                if gap["created_at"] <= i:  # Gap exists at this point
                    # Check if gap is filled
                    if not gap["filled"] and current_low <= gap["bottom"]:
                        gap["filled"] = True
                        gap_filled[i] = True

                    # If gap is still active, track it
                    if not gap["filled"]:
                        active_bull_gaps.append(gap)

                        # Generate signal when price approaches gap
                        if (gap["bottom"] <= current_price <= gap["top"] or
                            (current_low <= gap["top"] and current_high >= gap["bottom"])):
                            bullish_signals[i] = True

            # Process bearish gaps
            for gap in bearish_gaps:
                if gap["created_at"] <= i:  # Gap exists at this point
                    # Check if gap is filled
                    if not gap["filled"] and current_high >= gap["top"]:
                        gap["filled"] = True
                        gap_filled[i] = True

                    # If gap is still active, track it
                    if not gap["filled"]:
                        active_bear_gaps.append(gap)

                        # Generate signal when price approaches gap
                        if (gap["bottom"] <= current_price <= gap["top"] or
                            (current_low <= gap["top"] and current_high >= gap["bottom"])):
                            bearish_signals[i] = True

            # Find nearest support (bullish gap below price) and resistance (bearish gap above)
            bull_supports = [g for g in active_bull_gaps if g["top"] < current_price]
            bear_resistances = [g for g in active_bear_gaps if g["bottom"] > current_price]

            if bull_supports:
                nearest_gap = max(bull_supports, key=lambda g: g["top"])
                nearest_support[i] = nearest_gap["top"]
                nearest_support_mid[i] = nearest_gap["mid"]

            if bear_resistances:
                nearest_gap = min(bear_resistances, key=lambda g: g["bottom"])
                nearest_resistance[i] = nearest_gap["bottom"]
                nearest_resistance_mid[i] = nearest_gap["mid"]

            # Count active gaps
            active_bullish[i] = len(active_bull_gaps)
            active_bearish[i] = len(active_bear_gaps)

        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        result["FVG_bullish_signal"] = pd.Series(bullish_signals, index=df.index).shift(1).fillna(False)
        result["FVG_bearish_signal"] = pd.Series(bearish_signals, index=df.index).shift(1).fillna(False)
        result["FVG_gap_filled"] = pd.Series(gap_filled, index=df.index).shift(1).fillna(False)
        result["FVG_nearest_support"] = pd.Series(nearest_support, index=df.index).shift(1)
        result["FVG_nearest_resistance"] = pd.Series(nearest_resistance, index=df.index).shift(1)
        result["FVG_nearest_support_mid"] = pd.Series(nearest_support_mid, index=df.index).shift(1)
        result["FVG_nearest_resistance_mid"] = pd.Series(nearest_resistance_mid, index=df.index).shift(1)
        result["FVG_active_bullish_gaps"] = pd.Series(active_bullish, index=df.index).shift(1).fillna(0).astype(int)
        result["FVG_active_bearish_gaps"] = pd.Series(active_bearish, index=df.index).shift(1).fillna(0).astype(int)

        # Add bin index columns for VP-FVG confluence analysis
        bullish_bin_idx = self._extract_gap_bin_indices(bullish_gaps, df, 'bullish')
        bearish_bin_idx = self._extract_gap_bin_indices(bearish_gaps, df, 'bearish')
        
        result["FVG_bullish_bin_idx"] = pd.Series(bullish_bin_idx, index=df.index).shift(1).fillna(-1).astype(int)
        result["FVG_bearish_bin_idx"] = pd.Series(bearish_bin_idx, index=df.index).shift(1).fillna(-1).astype(int)
        
        # Add distance placeholders for future HVN/LVN analysis
        result["FVG_lvn_dist"] = pd.Series(np.full(len(df), np.nan), index=df.index).shift(1)
        result["FVG_hvn_dist"] = pd.Series(np.full(len(df), np.nan), index=df.index).shift(1)

        return result
