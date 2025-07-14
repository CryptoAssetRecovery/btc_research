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


@register("FVG")
class FVG(BaseIndicator):
    """
    Fair Value Gap (FVG) indicator.

    Detects 3-candle gap patterns where price leaves unfilled areas that often
    act as future support/resistance zones. Based on ICT trading methodology.

    A Fair Value Gap occurs when:
    - Bullish FVG: low[i+2] > high[i] (gap up)
    - Bearish FVG: high[i+2] < low[i] (gap down)

    Attributes:
        min_gap_pips (float): Minimum gap size to filter noise
        max_lookback (int): Maximum candles to look back for gap detection
    """

    @classmethod
    def params(cls):
        """Return default parameters for FVG indicator."""
        return {
            "min_gap_pips": 1.0,    # Minimum gap size in pips/points
            "max_lookback": 500     # Maximum historical gaps to track
        }

    def __init__(self, min_gap_pips: float = 1.0, max_lookback: int = 500):
        """
        Initialize FVG indicator.

        Args:
            min_gap_pips (float): Minimum gap size to consider valid. Default is 1.0.
            max_lookback (int): Maximum number of historical gaps to track. Default is 500.
        """
        self.min_gap_pips = min_gap_pips
        self.max_lookback = max_lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Fair Value Gaps and generate trading signals.

        Args:
            df (pd.DataFrame): Input OHLCV DataFrame

        Returns:
            pd.DataFrame: DataFrame with FVG zones and signals
        """
        # Handle edge case with insufficient data
        if len(df) < 3:
            return self._create_empty_result(df.index)

        # Reset index to work with integer positions
        df_reset = df.reset_index()

        # Detect all FVG patterns
        bullish_gaps, bearish_gaps = self._detect_gaps(df_reset)

        # Track gap status and generate signals
        result = self._generate_signals(df_reset, bullish_gaps, bearish_gaps)

        # Restore original index
        result.index = df.index

        return result

    def _create_empty_result(self, index: pd.Index) -> pd.DataFrame:
        """Create empty result DataFrame for edge cases."""
        result = pd.DataFrame(index=index)
        result["FVG_bullish_signal"] = False
        result["FVG_bearish_signal"] = False
        result["FVG_gap_filled"] = False
        result["FVG_nearest_support"] = float("nan")
        result["FVG_nearest_resistance"] = float("nan")
        result["FVG_active_bullish_gaps"] = 0
        result["FVG_active_bearish_gaps"] = 0
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

        # Iterate through dataframe looking for 3-candle patterns
        for i in range(len(df) - 2):
            # Get the three candles
            candle1 = df.iloc[i]
            candle3 = df.iloc[i + 2]

            # Check for bullish FVG: low[i+2] > high[i]
            if candle3["low"] > candle1["high"]:
                gap_size = candle3["low"] - candle1["high"]
                if gap_size >= self.min_gap_pips:
                    bullish_gaps.append({
                        "start_idx": i,
                        "end_idx": i + 2,
                        "top": candle3["low"],
                        "bottom": candle1["high"],
                        "size": gap_size,
                        "filled": False,
                        "created_at": i + 2
                    })

            # Check for bearish FVG: high[i+2] < low[i]
            elif candle3["high"] < candle1["low"]:
                gap_size = candle1["low"] - candle3["high"]
                if gap_size >= self.min_gap_pips:
                    bearish_gaps.append({
                        "start_idx": i,
                        "end_idx": i + 2,
                        "top": candle1["low"],
                        "bottom": candle3["high"],
                        "size": gap_size,
                        "filled": False,
                        "created_at": i + 2
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
                nearest_support[i] = max(g["top"] for g in bull_supports)

            if bear_resistances:
                nearest_resistance[i] = min(g["bottom"] for g in bear_resistances)

            # Count active gaps
            active_bullish[i] = len(active_bull_gaps)
            active_bearish[i] = len(active_bear_gaps)

        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        result["FVG_bullish_signal"] = bullish_signals
        result["FVG_bearish_signal"] = bearish_signals
        result["FVG_gap_filled"] = gap_filled
        result["FVG_nearest_support"] = nearest_support
        result["FVG_nearest_resistance"] = nearest_resistance
        result["FVG_active_bullish_gaps"] = active_bullish
        result["FVG_active_bearish_gaps"] = active_bearish

        return result
