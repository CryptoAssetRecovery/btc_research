"""
Relative Strength Index (RSI) indicator implementation.

This module provides an RSI indicator using the ta library for calculations
and includes overbought/oversold signal detection. The RSI is a momentum
oscillator used to identify potential reversal points.
"""

import pandas as pd
import ta.momentum as ta_momentum

from btc_research.core.base_indicator import BaseIndicator
from btc_research.core.registry import register


@register("RSI")
class RSI(BaseIndicator):
    """
    Relative Strength Index (RSI) indicator.

    The RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions. Values above 70 typically indicate
    overbought conditions, while values below 30 indicate oversold conditions.

    Attributes:
        length (int): The number of periods for RSI calculation
    """

    @classmethod
    def params(cls):
        """Return default parameters for RSI indicator."""
        return {"length": 14}

    def __init__(self, length=14):
        """
        Initialize RSI indicator.

        Args:
            length (int): Number of periods for RSI calculation. Default is 14.
        """
        self.length = length

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RSI and overbought/oversold signals.

        Args:
            df (pd.DataFrame): Input OHLCV DataFrame with 'close' column

        Returns:
            pd.DataFrame: DataFrame with RSI values and signal columns
        """
        # Handle edge case with insufficient data
        if len(df) < self.length + 1:
            result = pd.DataFrame(index=df.index)
            result[f"RSI_{self.length}"] = float("nan")
            result[f"RSI_{self.length}_overbought"] = False
            result[f"RSI_{self.length}_oversold"] = False
            return result

        # Calculate RSI using ta library
        rsi_indicator = ta_momentum.RSIIndicator(close=df["close"], window=self.length)
        rsi_values = rsi_indicator.rsi()

        # Generate overbought/oversold signals
        overbought = rsi_values > 70
        oversold = rsi_values < 30

        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        result[f"RSI_{self.length}"] = rsi_values
        result[f"RSI_{self.length}_overbought"] = overbought
        result[f"RSI_{self.length}_oversold"] = oversold

        return result
