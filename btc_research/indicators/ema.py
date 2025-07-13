"""
Exponential Moving Average (EMA) indicator implementation.

This module provides an EMA indicator that uses the ta library for calculations
and includes trend detection capabilities. The EMA is widely used to identify
trend direction and provide support/resistance levels.
"""

import pandas as pd
import ta.trend as ta_trend

from btc_research.core.base_indicator import BaseIndicator
from btc_research.core.registry import register


@register("EMA")
class EMA(BaseIndicator):
    """
    Exponential Moving Average (EMA) indicator.

    The EMA gives more weight to recent prices, making it more responsive to new
    information compared to a simple moving average. This implementation includes
    trend detection based on price position relative to the EMA.

    Attributes:
        length (int): The number of periods for EMA calculation
    """

    @classmethod
    def params(cls):
        """Return default parameters for EMA indicator."""
        return {"length": 200}

    def __init__(self, length=200):
        """
        Initialize EMA indicator.

        Args:
            length (int): Number of periods for EMA calculation. Default is 200.
        """
        self.length = length

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute EMA and trend signals.

        Args:
            df (pd.DataFrame): Input OHLCV DataFrame with 'close' column

        Returns:
            pd.DataFrame: DataFrame with EMA values and trend signals
        """
        # Handle edge case with insufficient data
        if len(df) < 2:
            result = pd.DataFrame(index=df.index)
            result[f"EMA_{self.length}"] = float("nan")
            result[f"EMA_{self.length}_trend"] = None
            return result

        # Calculate EMA using ta library
        ema_indicator = ta_trend.EMAIndicator(close=df["close"], window=self.length)
        ema_values = ema_indicator.ema_indicator()

        # Determine trend based on price vs EMA
        trend = pd.Series(index=df.index, dtype="object")
        trend.loc[df["close"] > ema_values] = "bull"
        trend.loc[df["close"] < ema_values] = "bear"
        trend.loc[df["close"] == ema_values] = "neutral"

        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        result[f"EMA_{self.length}"] = ema_values
        result[f"EMA_{self.length}_trend"] = trend

        return result
