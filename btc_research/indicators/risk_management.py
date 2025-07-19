"""
R-Multiple Position Management Engine.

This module provides an ATR-based position management system that implements
professional risk management techniques for VP-FVG strategies.

Key Features:
- ATR-based initial stop losses
- Dynamic trailing stops
- R-multiple target management
- Break-even management at R=1
- Separate logic for reversal and continuation setups
"""

import numpy as np
import pandas as pd
from typing import Any, Optional

from btc_research.core.base_indicator import BaseIndicator
from btc_research.core.registry import register

# Configure pandas to suppress FutureWarnings
pd.set_option("future.no_silent_downcasting", True)


@register("RiskManagement")
class RiskManagement(BaseIndicator):
    """
    R-Multiple Position Management Engine.
    
    This indicator implements a comprehensive risk management system based on ATR 
    and R-multiple concepts. It provides dynamic stop losses, trailing stops, 
    and target management for both reversal and continuation setups.
    
    Position Management Rules:
    1. **Initial Stop**: High/Low of the FVG zone ± 0.5 × ATR
    2. **Trailing**: Move to break-even at R=1, then trail 1 ATR behind
    3. **Hard Target**: 2 × ATR (reversal) or next HVN/LVN (continuations)
    
    The system tracks position state and provides exit signals based on:
    - Stop loss hits
    - Target achievement
    - Trailing stop violations
    
    Attributes:
        atr_period (int): ATR calculation period
        initial_stop_atr_mult (float): ATR multiplier for initial stop
        trailing_stop_atr_mult (float): ATR multiplier for trailing stop
        target_atr_mult (float): ATR multiplier for target calculation
        breakeven_r_mult (float): R-multiple at which to move to breakeven
        enable_trailing (bool): Enable trailing stop functionality
    """
    
    @classmethod
    def params(cls) -> dict[str, Any]:
        """Return default parameters for Risk Management indicator."""
        return {
            "atr_period": 14,               # ATR period for calculations
            "initial_stop_atr_mult": 0.5,   # ATR multiplier for initial stop
            "trailing_stop_atr_mult": 1.0,  # ATR multiplier for trailing stop
            "target_atr_mult": 2.0,         # ATR multiplier for target
            "breakeven_r_mult": 1.0,        # R-multiple to trigger breakeven
            "enable_trailing": True,        # Enable trailing stop functionality
        }
    
    def __init__(self, atr_period: int = 14, initial_stop_atr_mult: float = 0.5,
                 trailing_stop_atr_mult: float = 1.0, target_atr_mult: float = 2.0,
                 breakeven_r_mult: float = 1.0, enable_trailing: bool = True):
        """
        Initialize Risk Management indicator.
        
        Args:
            atr_period (int): ATR calculation period. Default is 14.
            initial_stop_atr_mult (float): ATR multiplier for initial stop. Default is 0.5.
            trailing_stop_atr_mult (float): ATR multiplier for trailing stop. Default is 1.0.
            target_atr_mult (float): ATR multiplier for target. Default is 2.0.
            breakeven_r_mult (float): R-multiple to trigger breakeven. Default is 1.0.
            enable_trailing (bool): Enable trailing stop functionality. Default is True.
        """
        self.atr_period = atr_period
        self.initial_stop_atr_mult = initial_stop_atr_mult
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.breakeven_r_mult = breakeven_r_mult
        self.enable_trailing = enable_trailing
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute risk management signals and levels.
        
        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV data and VP-FVG signals
        
        Returns:
            pd.DataFrame: DataFrame with risk management levels and signals
        """
        # Handle edge case with insufficient data
        if len(df) < self.atr_period:
            return self._create_empty_result(df.index)
        
        # Calculate ATR for position sizing
        atr_values = self._calculate_atr(df)
        
        # Initialize result arrays
        n = len(df)
        
        # Position management levels
        long_stop_loss = np.full(n, np.nan)
        long_target = np.full(n, np.nan)
        long_trailing_stop = np.full(n, np.nan)
        short_stop_loss = np.full(n, np.nan)
        short_target = np.full(n, np.nan)
        short_trailing_stop = np.full(n, np.nan)
        
        # Position state tracking
        long_position_active = np.zeros(n, dtype=bool)
        short_position_active = np.zeros(n, dtype=bool)
        long_entry_price = np.full(n, np.nan)
        short_entry_price = np.full(n, np.nan)
        long_r_multiple = np.full(n, np.nan)
        short_r_multiple = np.full(n, np.nan)
        
        # Exit signals
        long_stop_hit = np.zeros(n, dtype=bool)
        long_target_hit = np.zeros(n, dtype=bool)
        short_stop_hit = np.zeros(n, dtype=bool)
        short_target_hit = np.zeros(n, dtype=bool)
        
        # Position state variables
        current_long_position = False
        current_short_position = False
        current_long_entry = np.nan
        current_short_entry = np.nan
        current_long_stop = np.nan
        current_short_stop = np.nan
        current_long_target = np.nan
        current_short_target = np.nan
        current_long_trailing = np.nan
        current_short_trailing = np.nan
        
        # Required columns for integration - flexible signal detection
        required_columns = ['close', 'high', 'low']
        
        # Detect available signal columns
        long_signal_cols = [col for col in df.columns if any(x in col.lower() for x in ['vf_long', 'bullish_signal', 'long_signal'])]
        short_signal_cols = [col for col in df.columns if any(x in col.lower() for x in ['vf_short', 'bearish_signal', 'short_signal'])]
        atr_signal_cols = [col for col in df.columns if 'vf_atr' in col.lower()]
        
        # Use first available signal columns
        long_signal_col = long_signal_cols[0] if long_signal_cols else None
        short_signal_col = short_signal_cols[0] if short_signal_cols else None
        atr_signal_col = atr_signal_cols[0] if atr_signal_cols else None
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns or (long_signal_col is None and short_signal_col is None):
            return self._create_empty_result(df.index)
        
        # Process each bar
        for i in range(self.atr_period, n):
            current_price = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_atr = atr_values[i]
            
            if np.isnan(current_atr) or current_atr <= 0:
                continue
            
            # Check for new long entry
            if not current_long_position and long_signal_col and df.iloc[i][long_signal_col]:
                current_long_position = True
                current_long_entry = current_price
                
                # Calculate initial stop and target for long position
                # Initial stop: Below FVG zone - 0.5 * ATR
                fvg_low = self._get_fvg_level(df, i, 'support')
                if not np.isnan(fvg_low):
                    current_long_stop = fvg_low - (current_atr * self.initial_stop_atr_mult)
                else:
                    current_long_stop = current_price - (current_atr * self.initial_stop_atr_mult)
                
                # Target: Entry + 2 * ATR (reversal setup)
                current_long_target = current_price + (current_atr * self.target_atr_mult)
                current_long_trailing = current_long_stop  # Initialize trailing stop
            
            # Check for new short entry
            if not current_short_position and short_signal_col and df.iloc[i][short_signal_col]:
                current_short_position = True
                current_short_entry = current_price
                
                # Calculate initial stop and target for short position
                # Initial stop: Above FVG zone + 0.5 * ATR
                fvg_high = self._get_fvg_level(df, i, 'resistance')
                if not np.isnan(fvg_high):
                    current_short_stop = fvg_high + (current_atr * self.initial_stop_atr_mult)
                else:
                    current_short_stop = current_price + (current_atr * self.initial_stop_atr_mult)
                
                # Target: Entry - 2 * ATR (continuation setup)
                current_short_target = current_price - (current_atr * self.target_atr_mult)
                current_short_trailing = current_short_stop  # Initialize trailing stop
            
            # Manage long position
            if current_long_position:
                # Calculate current R-multiple
                risk_amount = current_long_entry - current_long_stop
                if risk_amount > 0:
                    current_r_mult = (current_price - current_long_entry) / risk_amount
                else:
                    current_r_mult = 0
                
                # Update trailing stop
                if self.enable_trailing and current_r_mult >= self.breakeven_r_mult:
                    if current_r_mult >= self.breakeven_r_mult:
                        # Move to breakeven
                        new_trailing_stop = current_long_entry
                    else:
                        # Trail by 1 ATR
                        new_trailing_stop = current_price - (current_atr * self.trailing_stop_atr_mult)
                    
                    current_long_trailing = max(current_long_trailing, new_trailing_stop)
                
                # Check for exits
                if current_low <= current_long_trailing:
                    long_stop_hit[i] = True
                    current_long_position = False
                    current_long_entry = np.nan
                    current_long_stop = np.nan
                    current_long_target = np.nan
                    current_long_trailing = np.nan
                elif current_high >= current_long_target:
                    long_target_hit[i] = True
                    current_long_position = False
                    current_long_entry = np.nan
                    current_long_stop = np.nan
                    current_long_target = np.nan
                    current_long_trailing = np.nan
                else:
                    # Position still active
                    long_position_active[i] = True
                    long_entry_price[i] = current_long_entry
                    long_stop_loss[i] = current_long_stop
                    long_target[i] = current_long_target
                    long_trailing_stop[i] = current_long_trailing
                    long_r_multiple[i] = current_r_mult
            
            # Manage short position
            if current_short_position:
                # Calculate current R-multiple
                risk_amount = current_short_stop - current_short_entry
                if risk_amount > 0:
                    current_r_mult = (current_short_entry - current_price) / risk_amount
                else:
                    current_r_mult = 0
                
                # Update trailing stop
                if self.enable_trailing and current_r_mult >= self.breakeven_r_mult:
                    if current_r_mult >= self.breakeven_r_mult:
                        # Move to breakeven
                        new_trailing_stop = current_short_entry
                    else:
                        # Trail by 1 ATR
                        new_trailing_stop = current_price + (current_atr * self.trailing_stop_atr_mult)
                    
                    current_short_trailing = min(current_short_trailing, new_trailing_stop)
                
                # Check for exits
                if current_high >= current_short_trailing:
                    short_stop_hit[i] = True
                    current_short_position = False
                    current_short_entry = np.nan
                    current_short_stop = np.nan
                    current_short_target = np.nan
                    current_short_trailing = np.nan
                elif current_low <= current_short_target:
                    short_target_hit[i] = True
                    current_short_position = False
                    current_short_entry = np.nan
                    current_short_stop = np.nan
                    current_short_target = np.nan
                    current_short_trailing = np.nan
                else:
                    # Position still active
                    short_position_active[i] = True
                    short_entry_price[i] = current_short_entry
                    short_stop_loss[i] = current_short_stop
                    short_target[i] = current_short_target
                    short_trailing_stop[i] = current_short_trailing
                    short_r_multiple[i] = current_r_mult
        
        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Position management levels (shift to prevent look-ahead bias)
        result["long_stop_loss"] = pd.Series(long_stop_loss, index=df.index).shift(1)
        result["long_target"] = pd.Series(long_target, index=df.index).shift(1)
        result["long_trailing_stop"] = pd.Series(long_trailing_stop, index=df.index).shift(1)
        result["short_stop_loss"] = pd.Series(short_stop_loss, index=df.index).shift(1)
        result["short_target"] = pd.Series(short_target, index=df.index).shift(1)
        result["short_trailing_stop"] = pd.Series(short_trailing_stop, index=df.index).shift(1)

        # Immediate (unshifted) stop levels so the strategy can size the very first
        # bar of a new position.  These are read only by DynamicStrategy and are not
        # referenced in YAML logic, so look-ahead bias is not a concern.
        result["long_stop_loss_immediate"] = pd.Series(long_stop_loss, index=df.index)
        result["short_stop_loss_immediate"] = pd.Series(short_stop_loss, index=df.index)
        
        # Position state
        result["long_position_active"] = pd.Series(long_position_active, index=df.index).shift(1).fillna(False).astype(bool)
        result["short_position_active"] = pd.Series(short_position_active, index=df.index).shift(1).fillna(False).astype(bool)
        result["long_entry_price"] = pd.Series(long_entry_price, index=df.index).shift(1)
        result["short_entry_price"] = pd.Series(short_entry_price, index=df.index).shift(1)
        result["long_r_multiple"] = pd.Series(long_r_multiple, index=df.index).shift(1)
        result["short_r_multiple"] = pd.Series(short_r_multiple, index=df.index).shift(1)
        
        # Exit signals
        result["long_stop_hit"] = pd.Series(long_stop_hit, index=df.index).shift(1).fillna(False).astype(bool)
        result["long_target_hit"] = pd.Series(long_target_hit, index=df.index).shift(1).fillna(False).astype(bool)
        result["short_stop_hit"] = pd.Series(short_stop_hit, index=df.index).shift(1).fillna(False).astype(bool)
        result["short_target_hit"] = pd.Series(short_target_hit, index=df.index).shift(1).fillna(False).astype(bool)
        
        # Risk management exit signals (combined)
        result["exit_long_risk"] = result["long_stop_hit"] | result["long_target_hit"]
        result["exit_short_risk"] = result["short_stop_hit"] | result["short_target_hit"]
        
        return result
    
    def _calculate_atr(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate Average True Range (ATR)."""
        n = len(df)
        atr = np.full(n, np.nan)
        
        if n < self.atr_period + 1:
            return atr
        
        # Calculate True Range
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        
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
    
    def _get_fvg_level(self, df: pd.DataFrame, i: int, level_type: str) -> float:
        """
        Get FVG level for stop loss calculation.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            i (int): Current bar index
            level_type (str): 'support' or 'resistance'
            
        Returns:
            float: FVG level price
        """
        try:
            if level_type == 'support':
                # For long positions, use FVG support level
                if 'FVG_nearest_support' in df.columns:
                    return df.iloc[i]['FVG_nearest_support']
                else:
                    return np.nan
            elif level_type == 'resistance':
                # For short positions, use FVG resistance level
                if 'FVG_nearest_resistance' in df.columns:
                    return df.iloc[i]['FVG_nearest_resistance']
                else:
                    return np.nan
            else:
                return np.nan
        except (KeyError, IndexError):
            return np.nan
    
    def _create_empty_result(self, index: pd.Index) -> pd.DataFrame:
        """Create empty result DataFrame for edge cases."""
        result = pd.DataFrame(index=index)
        
        # Position management levels
        result["long_stop_loss"] = np.nan
        result["long_target"] = np.nan
        result["long_trailing_stop"] = np.nan
        result["short_stop_loss"] = np.nan
        result["short_target"] = np.nan
        result["short_trailing_stop"] = np.nan
        
        # Position state
        result["long_position_active"] = False
        result["short_position_active"] = False
        result["long_entry_price"] = np.nan
        result["short_entry_price"] = np.nan
        result["long_r_multiple"] = np.nan
        result["short_r_multiple"] = np.nan
        
        # Exit signals
        result["long_stop_hit"] = False
        result["long_target_hit"] = False
        result["short_stop_hit"] = False
        result["short_target_hit"] = False
        result["exit_long_risk"] = False
        result["exit_short_risk"] = False
        
        return result