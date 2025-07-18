"""
Average Directional Index (ADX) trend filter implementation.

This module provides an ADX indicator that measures the strength of a trend
without indicating its direction. It's used as a regime filter to distinguish
between trending and ranging markets for trading strategy optimization.

The ADX is calculated from the Directional Movement Index (DMI) components:
- +DI (Positive Directional Indicator): Measures upward price movement
- -DI (Negative Directional Indicator): Measures downward price movement
- ADX: Smoothed average of the difference between +DI and -DI

Key concepts:
- ADX > 25: Indicates trending market (suitable for continuation setups)
- ADX < 20: Indicates ranging market (suitable for reversal setups)
- ADX rising: Trend is strengthening
- ADX falling: Trend is weakening
"""

import numpy as np
import pandas as pd
from typing import Any

from btc_research.core.base_indicator import BaseIndicator
from btc_research.core.registry import register


@register("ADX")
class ADX(BaseIndicator):
    """
    Average Directional Index (ADX) trend filter indicator.
    
    The ADX measures the strength of a trend without indicating its direction.
    It's derived from the Directional Movement Index (DMI) and provides a value
    between 0 and 100, where higher values indicate stronger trends.
    
    The calculation involves:
    1. True Range (TR) calculation
    2. Directional Movement (+DM, -DM) calculation
    3. Smoothed True Range (ATR) and Directional Indicators (+DI, -DI)
    4. Directional Index (DX) calculation
    5. Average Directional Index (ADX) smoothing
    
    Attributes:
        period (int): Period for ADX calculation (default: 14)
        trend_threshold (float): ADX threshold for trending markets (default: 25)
        range_threshold (float): ADX threshold for ranging markets (default: 20)
        smoothing_method (str): Method for smoothing ('wilder' or 'ema')
    """
    
    @classmethod
    def params(cls) -> dict[str, Any]:
        """Return default parameters for ADX indicator."""
        return {
            "period": 14,                    # Period for ADX calculation
            "trend_threshold": 25.0,         # ADX > this indicates trending market
            "range_threshold": 20.0,         # ADX < this indicates ranging market
            "smoothing_method": "wilder"     # Smoothing method: 'wilder' or 'ema'
        }
    
    def __init__(self, period: int = 14, trend_threshold: float = 25.0, 
                 range_threshold: float = 20.0, smoothing_method: str = "wilder"):
        """
        Initialize ADX indicator.
        
        Args:
            period (int): Period for ADX calculation. Default is 14.
            trend_threshold (float): ADX threshold for trending markets. Default is 25.0.
            range_threshold (float): ADX threshold for ranging markets. Default is 20.0.
            smoothing_method (str): Smoothing method ('wilder' or 'ema'). Default is 'wilder'.
        """
        if period < 1:
            raise ValueError("Period must be positive")
        if trend_threshold <= range_threshold:
            raise ValueError("Trend threshold must be greater than range threshold")
        if smoothing_method not in ["wilder", "ema"]:
            raise ValueError("Smoothing method must be 'wilder' or 'ema'")
            
        self.period = period
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        self.smoothing_method = smoothing_method
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ADX and directional indicators.
        
        Args:
            df (pd.DataFrame): Input OHLCV DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with ADX values and trend/range signals
        """
        # Handle edge case with insufficient data
        # Need at least 2*period + 1 for double smoothing
        if len(df) < (self.period * 2) + 1:
            return self._create_empty_result(df.index)
        
        # Calculate True Range and Directional Movement
        tr_values = self._calculate_true_range(df)
        dm_plus, dm_minus = self._calculate_directional_movement(df)
        
        # Calculate smoothed ATR and DI values
        atr_values = self._smooth_values(tr_values, self.period)
        di_plus_smooth = self._smooth_values(dm_plus, self.period)
        di_minus_smooth = self._smooth_values(dm_minus, self.period)
        
        # Calculate +DI and -DI
        di_plus = np.full(len(df), np.nan)
        di_minus = np.full(len(df), np.nan)
        
        # Only calculate where we have valid ATR values
        valid_atr = (~np.isnan(atr_values)) & (atr_values > 0)
        valid_di_plus = (~np.isnan(di_plus_smooth)) & valid_atr
        valid_di_minus = (~np.isnan(di_minus_smooth)) & valid_atr
        
        di_plus[valid_di_plus] = 100 * (di_plus_smooth[valid_di_plus] / atr_values[valid_di_plus])
        di_minus[valid_di_minus] = 100 * (di_minus_smooth[valid_di_minus] / atr_values[valid_di_minus])
        
        # Calculate DX (Directional Index)
        dx_values = self._calculate_dx(di_plus, di_minus)
        
        # Calculate ADX (smoothed DX)
        adx_values = self._smooth_values(dx_values, self.period)
        
        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Apply look-ahead bias prevention by shifting all values
        result[f"ADX_value"] = pd.Series(adx_values, index=df.index).shift(1)
        result[f"DI_plus"] = pd.Series(di_plus, index=df.index).shift(1)
        result[f"DI_minus"] = pd.Series(di_minus, index=df.index).shift(1)
        
        # Recalculate signals based on shifted ADX values
        shifted_adx = result[f"ADX_value"]
        result[f"ADX_trend"] = shifted_adx > self.trend_threshold
        result[f"ADX_range"] = shifted_adx < self.range_threshold
        
        # Additional useful signals
        result[f"ADX_strength"] = self._categorize_adx_strength(shifted_adx)
        result[f"DI_bullish"] = result[f"DI_plus"] > result[f"DI_minus"]
        result[f"DI_bearish"] = result[f"DI_minus"] > result[f"DI_plus"]
        
        return result
    
    def _calculate_true_range(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate True Range (TR) values.
        
        TR = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
        
        Args:
            df (pd.DataFrame): Input OHLCV DataFrame
            
        Returns:
            np.ndarray: True Range values
        """
        n = len(df)
        tr = np.zeros(n)
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # First period TR is just high - low
        tr[0] = high[0] - low[0]
        
        # Calculate TR for remaining periods
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        return tr
    
    def _calculate_directional_movement(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate Directional Movement (+DM and -DM).
        
        +DM = max(High - PrevHigh, 0) if High - PrevHigh > PrevLow - Low, else 0
        -DM = max(PrevLow - Low, 0) if PrevLow - Low > High - PrevHigh, else 0
        
        Args:
            df (pd.DataFrame): Input OHLCV DataFrame
            
        Returns:
            tuple[np.ndarray, np.ndarray]: +DM and -DM values
        """
        n = len(df)
        dm_plus = np.zeros(n)
        dm_minus = np.zeros(n)
        
        high = df['high'].values
        low = df['low'].values
        
        # First period has no directional movement
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Calculate DM for remaining periods
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            else:
                dm_plus[i] = 0
                
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
            else:
                dm_minus[i] = 0
        
        return dm_plus, dm_minus
    
    def _smooth_values(self, values: np.ndarray, period: int) -> np.ndarray:
        """
        Apply smoothing to values using specified method.
        
        Args:
            values (np.ndarray): Input values to smooth
            period (int): Smoothing period
            
        Returns:
            np.ndarray: Smoothed values
        """
        n = len(values)
        smoothed = np.full(n, np.nan)
        
        if n < period:
            return smoothed
        
        # Skip initial NaN values
        first_valid = 0
        while first_valid < n and np.isnan(values[first_valid]):
            first_valid += 1
        
        if first_valid >= n or (first_valid + period) > n:
            return smoothed
        
        if self.smoothing_method == "wilder":
            # Wilder's smoothing (similar to EMA but different alpha)
            # First value is simple average of first 'period' valid values
            start_idx = first_valid
            end_idx = start_idx + period
            
            # Make sure we have enough valid values for initial average
            valid_values = values[start_idx:end_idx]
            if len(valid_values) == period and not np.any(np.isnan(valid_values)):
                smoothed[end_idx - 1] = np.mean(valid_values)
                
                # Subsequent values use Wilder's smoothing
                for i in range(end_idx, n):
                    if not np.isnan(values[i]):
                        smoothed[i] = (smoothed[i-1] * (period - 1) + values[i]) / period
                    else:
                        smoothed[i] = smoothed[i-1]  # Forward fill NaN values
                        
        elif self.smoothing_method == "ema":
            # Exponential moving average
            alpha = 2.0 / (period + 1)
            
            # First value is simple average of first 'period' valid values
            start_idx = first_valid
            end_idx = start_idx + period
            
            # Make sure we have enough valid values for initial average
            valid_values = values[start_idx:end_idx]
            if len(valid_values) == period and not np.any(np.isnan(valid_values)):
                smoothed[end_idx - 1] = np.mean(valid_values)
                
                # Apply EMA formula
                for i in range(end_idx, n):
                    if not np.isnan(values[i]):
                        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
                    else:
                        smoothed[i] = smoothed[i-1]  # Forward fill NaN values
        
        return smoothed
    
    def _calculate_dx(self, di_plus: np.ndarray, di_minus: np.ndarray) -> np.ndarray:
        """
        Calculate Directional Index (DX) from +DI and -DI.
        
        DX = 100 * abs(+DI - -DI) / (+DI + -DI)
        
        Args:
            di_plus (np.ndarray): +DI values
            di_minus (np.ndarray): -DI values
            
        Returns:
            np.ndarray: DX values
        """
        n = len(di_plus)
        dx = np.full(n, np.nan)
        
        # Calculate DX where both DI values are valid
        valid_mask = (~np.isnan(di_plus)) & (~np.isnan(di_minus))
        
        di_sum = di_plus + di_minus
        di_diff = np.abs(di_plus - di_minus)
        
        # Avoid division by zero
        non_zero_sum = di_sum > 0
        valid_calc = valid_mask & non_zero_sum
        
        dx[valid_calc] = 100 * (di_diff[valid_calc] / di_sum[valid_calc])
        
        return dx
    
    def _categorize_adx_strength(self, adx_values: pd.Series) -> pd.Series:
        """
        Categorize ADX strength into discrete levels.
        
        Args:
            adx_values (pd.Series): ADX values
            
        Returns:
            pd.Series: Strength categories
        """
        def categorize(value):
            if pd.isna(value):
                return "unknown"
            elif value < 20:
                return "weak"
            elif value < 25:
                return "moderate"
            elif value < 40:
                return "strong"
            else:
                return "very_strong"
        
        return adx_values.apply(categorize)
    
    def _create_empty_result(self, index: pd.Index) -> pd.DataFrame:
        """Create empty result DataFrame for edge cases."""
        result = pd.DataFrame(index=index)
        
        # Core ADX values
        result[f"ADX_value"] = np.nan
        result[f"DI_plus"] = np.nan
        result[f"DI_minus"] = np.nan
        
        # Trend/range signals
        result[f"ADX_trend"] = False
        result[f"ADX_range"] = False
        
        # Additional signals
        result[f"ADX_strength"] = "unknown"
        result[f"DI_bullish"] = False
        result[f"DI_bearish"] = False
        
        return result