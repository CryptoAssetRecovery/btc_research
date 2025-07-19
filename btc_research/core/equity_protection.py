"""
Equity Protection Module for First-Loss Stop Implementation.

This module provides comprehensive equity curve protection mechanisms to prevent
catastrophic drawdowns. It implements a first-loss stop that flattens all positions
and disables new entries when drawdown exceeds a configurable threshold.

Key Features:
- Real-time equity curve tracking
- Configurable drawdown threshold (default 25%)
- Automatic trading suspension during high drawdown
- Bias flip detection for re-enabling trading
- Comprehensive equity statistics and visualization
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TradingState(Enum):
    """Trading state enumeration."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DISABLED = "disabled"


class BiasDirection(Enum):
    """Market bias direction enumeration."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


@dataclass
class EquityStats:
    """Equity curve statistics."""
    current_equity: float
    peak_equity: float
    current_drawdown: float
    max_drawdown: float
    drawdown_percentage: float
    trading_state: TradingState
    bias_direction: BiasDirection
    equity_curve: List[float]
    drawdown_curve: List[float]
    timestamps: List[str]


class EquityProtection:
    """
    Equity Protection System with First-Loss Stop.
    
    This class implements a comprehensive equity protection mechanism that:
    1. Tracks equity curve and maximum drawdown in real-time
    2. Implements configurable first-loss stop threshold
    3. Disables new entries when drawdown threshold is exceeded
    4. Re-enables trading when market bias flips
    5. Provides equity curve visualization and statistics
    
    Attributes:
        drawdown_threshold (float): Drawdown threshold for first-loss stop (default 25%)
        bias_flip_threshold (float): Price change threshold for bias flip detection
        equity_smoothing (int): Number of periods for equity smoothing
        enable_bias_reset (bool): Whether to re-enable trading on bias flip
    """
    
    def __init__(self,
                 drawdown_threshold: float = 0.25,
                 bias_flip_threshold: float = 0.10,
                 equity_smoothing: int = 1,
                 enable_bias_reset: bool = True,
                 initial_equity: Optional[float] = None):
        """
        Initialize Equity Protection System.
        
        Args:
            drawdown_threshold (float): Drawdown threshold (0.25 = 25%)
            bias_flip_threshold (float): Price change for bias flip (0.10 = 10%)
            equity_smoothing (int): Periods for equity smoothing (1 = no smoothing)
            enable_bias_reset (bool): Enable trading reset on bias flip
            initial_equity (float, optional): Initial equity value to set as peak
        """
        self.drawdown_threshold = drawdown_threshold
        self.bias_flip_threshold = bias_flip_threshold
        self.equity_smoothing = equity_smoothing
        self.enable_bias_reset = enable_bias_reset
        
        # State tracking
        self.equity_curve = []
        self.drawdown_curve = []
        self.timestamps = []
        self.peak_equity = initial_equity or 0.0
        self.current_equity = initial_equity or 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.trading_state = TradingState.ACTIVE
        self.bias_direction = BiasDirection.NEUTRAL
        
        # Bias tracking
        self.last_bias_price = None
        self.bias_flip_count = 0
        self.last_bias_flip_time = None
        
        # Statistics
        self.drawdown_periods = 0
        self.recovery_periods = 0
        self.suspended_periods = 0
    
    def update_equity(self, 
                     current_equity: float, 
                     timestamp: Optional[str] = None) -> dict:
        """
        Update equity curve and check protection thresholds.
        
        Args:
            current_equity (float): Current portfolio equity
            timestamp (str, optional): Timestamp for tracking
            
        Returns:
            dict: Update result with protection status
        """
        if timestamp is None:
            timestamp = f"Period_{len(self.equity_curve)}"
        
        # Apply equity smoothing if enabled
        if self.equity_smoothing > 1 and len(self.equity_curve) >= self.equity_smoothing:
            recent_equity = self.equity_curve[-self.equity_smoothing:]
            smoothed_equity = np.mean(recent_equity + [current_equity])
            self.current_equity = smoothed_equity
        else:
            self.current_equity = current_equity
        
        # Update equity curve
        self.equity_curve.append(self.current_equity)
        self.timestamps.append(timestamp)
        
        # Update peak equity
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = self.peak_equity - self.current_equity
            drawdown_percentage = self.current_drawdown / self.peak_equity
        else:
            self.current_drawdown = 0.0
            drawdown_percentage = 0.0
        
        # Update maximum drawdown
        if drawdown_percentage > self.max_drawdown:
            self.max_drawdown = drawdown_percentage
        
        # Update drawdown curve
        self.drawdown_curve.append(drawdown_percentage)
        
        # Check protection thresholds
        protection_triggered = self._check_protection_thresholds()
        
        # Update statistics
        self._update_statistics()
        
        # Return update result
        return {
            'equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'drawdown': drawdown_percentage,
            'protection_active': self.trading_state != TradingState.ACTIVE,
            'protection_triggered': protection_triggered,
            'trading_disabled': self.should_disable_trading()
        }
    
    def update_bias(self, 
                   current_price: float, 
                   bias_indicator: Optional[str] = None) -> None:
        """
        Update market bias and check for bias flip.
        
        Args:
            current_price (float): Current market price
            bias_indicator (str, optional): External bias indicator ('bull'/'bear')
        """
        if bias_indicator:
            # Use external bias indicator
            new_bias = BiasDirection.BULL if bias_indicator.lower() == 'bull' else BiasDirection.BEAR
        else:
            # Calculate bias from price change
            if self.last_bias_price is None:
                self.last_bias_price = current_price
                return
            
            price_change = (current_price - self.last_bias_price) / self.last_bias_price
            
            if price_change > self.bias_flip_threshold:
                new_bias = BiasDirection.BULL
            elif price_change < -self.bias_flip_threshold:
                new_bias = BiasDirection.BEAR
            else:
                new_bias = self.bias_direction  # No change
        
        # Check for bias flip
        if new_bias != self.bias_direction and self.bias_direction != BiasDirection.NEUTRAL:
            self._handle_bias_flip(new_bias)
        
        self.bias_direction = new_bias
        self.last_bias_price = current_price
    
    def is_drawdown_exceeded(self) -> bool:
        """Check if drawdown threshold is exceeded."""
        if self.peak_equity <= 0:
            return False
        
        current_drawdown_pct = self.current_drawdown / self.peak_equity
        return current_drawdown_pct >= self.drawdown_threshold
    
    def should_disable_trading(self) -> bool:
        """Check if trading should be disabled."""
        return self.trading_state in [TradingState.SUSPENDED, TradingState.DISABLED]
    
    def should_allow_longs(self) -> bool:
        """Check if long positions should be allowed."""
        if self.should_disable_trading():
            return False
        return self.bias_direction in [BiasDirection.BULL, BiasDirection.NEUTRAL]
    
    def should_allow_shorts(self) -> bool:
        """Check if short positions should be allowed."""
        if self.should_disable_trading():
            return False
        return self.bias_direction in [BiasDirection.BEAR, BiasDirection.NEUTRAL]
    
    def force_suspend_trading(self) -> None:
        """Force suspend trading regardless of conditions."""
        self.trading_state = TradingState.SUSPENDED
    
    def force_resume_trading(self) -> None:
        """Force resume trading regardless of conditions."""
        self.trading_state = TradingState.ACTIVE
    
    def reset_on_bias_flip(self, new_bias: BiasDirection) -> None:
        """Reset trading state on bias flip."""
        if self.enable_bias_reset and self.trading_state == TradingState.SUSPENDED:
            self.trading_state = TradingState.ACTIVE
            self.bias_flip_count += 1
            self.last_bias_flip_time = len(self.timestamps) - 1
    
    def get_equity_stats(self) -> EquityStats:
        """Get comprehensive equity statistics."""
        return EquityStats(
            current_equity=self.current_equity,
            peak_equity=self.peak_equity,
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            drawdown_percentage=self.current_drawdown / self.peak_equity if self.peak_equity > 0 else 0.0,
            trading_state=self.trading_state,
            bias_direction=self.bias_direction,
            equity_curve=self.equity_curve.copy(),
            drawdown_curve=self.drawdown_curve.copy(),
            timestamps=self.timestamps.copy()
        )
    
    def _check_protection_thresholds(self) -> bool:
        """Check and update protection thresholds.
        
        Returns:
            bool: True if protection was triggered this update
        """
        protection_triggered = False
        
        if self.is_drawdown_exceeded():
            if self.trading_state == TradingState.ACTIVE:
                self.trading_state = TradingState.SUSPENDED
                self.suspended_periods = 0
                protection_triggered = True
        else:
            # Resume trading if drawdown is below threshold and no other constraints
            if self.trading_state == TradingState.SUSPENDED:
                self.trading_state = TradingState.ACTIVE
        
        return protection_triggered
    
    def _handle_bias_flip(self, new_bias: BiasDirection) -> None:
        """Handle bias flip event."""
        if self.enable_bias_reset:
            self.reset_on_bias_flip(new_bias)
    
    def _update_statistics(self) -> None:
        """Update internal statistics."""
        if self.trading_state == TradingState.SUSPENDED:
            self.suspended_periods += 1
        
        if len(self.drawdown_curve) > 1:
            if self.drawdown_curve[-1] > self.drawdown_curve[-2]:
                self.drawdown_periods += 1
            elif self.drawdown_curve[-1] < self.drawdown_curve[-2]:
                self.recovery_periods += 1
    
    def plot_equity_curve(self, 
                         title: str = "Equity Curve with Protection",
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> None:
        """
        Plot equity curve with protection indicators.
        
        Args:
            title (str): Plot title
            figsize (tuple): Figure size
            save_path (str, optional): Path to save plot
        """
        if not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not available - cannot plot equity curve")
            return
        
        if len(self.equity_curve) < 2:
            warnings.warn("Insufficient data for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot equity curve
        ax1.plot(self.equity_curve, label='Equity', linewidth=2)
        ax1.axhline(y=self.peak_equity, color='g', linestyle='--', alpha=0.7, label='Peak Equity')
        
        # Mark suspended periods
        suspended_mask = np.array([self.trading_state == TradingState.SUSPENDED] * len(self.equity_curve))
        if np.any(suspended_mask):
            ax1.fill_between(range(len(self.equity_curve)), 
                           min(self.equity_curve), max(self.equity_curve),
                           where=suspended_mask, alpha=0.3, color='red', label='Trading Suspended')
        
        ax1.set_title(title)
        ax1.set_ylabel('Equity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown curve
        drawdown_pct = np.array(self.drawdown_curve) * 100
        ax2.fill_between(range(len(drawdown_pct)), 0, drawdown_pct, 
                        alpha=0.6, color='red', label='Drawdown %')
        ax2.axhline(y=self.drawdown_threshold * 100, color='orange', 
                   linestyle='--', linewidth=2, label=f'Protection Threshold ({self.drawdown_threshold*100:.1f}%)')
        
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Drawdown %')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Invert y-axis so drawdown goes down
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_protection_report(self) -> str:
        """Generate comprehensive protection report."""
        stats = self.get_equity_stats()
        
        report = f"""
EQUITY PROTECTION REPORT
========================
Current State: {stats.trading_state.value.upper()}
Market Bias: {stats.bias_direction.value.upper()}

Equity Metrics:
- Current Equity: ${stats.current_equity:,.2f}
- Peak Equity: ${stats.peak_equity:,.2f}
- Current Drawdown: ${stats.current_drawdown:,.2f} ({stats.drawdown_percentage:.2%})
- Maximum Drawdown: {stats.max_drawdown:.2%}

Protection Settings:
- Drawdown Threshold: {self.drawdown_threshold:.2%}
- Bias Flip Threshold: {self.bias_flip_threshold:.2%}
- Equity Smoothing: {self.equity_smoothing} periods
- Bias Reset Enabled: {self.enable_bias_reset}

Statistics:
- Total Periods: {len(self.equity_curve)}
- Suspended Periods: {self.suspended_periods}
- Drawdown Periods: {self.drawdown_periods}
- Recovery Periods: {self.recovery_periods}
- Bias Flips: {self.bias_flip_count}

Trading Permissions:
- Allow Longs: {self.should_allow_longs()}
- Allow Shorts: {self.should_allow_shorts()}
- Trading Disabled: {self.should_disable_trading()}
"""
        
        return report.strip()


class BacktraderIntegration:
    """Integration utilities for Backtrader."""
    
    @staticmethod
    def create_equity_protection_analyzer(protection: EquityProtection):
        """Create Backtrader analyzer for equity protection."""
        
        class EquityProtectionAnalyzer:
            def __init__(self):
                self.protection = protection
            
            def next(self):
                # Update equity from broker
                current_equity = self.strategy.broker.getvalue()
                self.protection.update_equity(current_equity)
            
            def should_allow_trade(self, trade_type: str) -> bool:
                if trade_type.lower() == 'long':
                    return self.protection.should_allow_longs()
                elif trade_type.lower() == 'short':
                    return self.protection.should_allow_shorts()
                return not self.protection.should_disable_trading()
        
        return EquityProtectionAnalyzer


def create_equity_protection_column(df: pd.DataFrame,
                                  equity_column: str = 'equity',
                                  drawdown_threshold: float = 0.25) -> pd.Series:
    """
    Create equity protection signal column for DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with equity column
        equity_column (str): Name of equity column
        drawdown_threshold (float): Drawdown threshold
        
    Returns:
        pd.Series: Boolean series indicating when trading should be disabled
    """
    if equity_column not in df.columns:
        return pd.Series(False, index=df.index)
    
    protection = EquityProtection(drawdown_threshold=drawdown_threshold)
    disable_trading = []
    
    for equity in df[equity_column]:
        if not np.isnan(equity):
            protection.update_equity(equity)
        disable_trading.append(protection.should_disable_trading())
    
    return pd.Series(disable_trading, index=df.index)


def simulate_equity_protection(equity_curve: List[float],
                             drawdown_threshold: float = 0.25,
                             plot_results: bool = True) -> EquityProtection:
    """
    Simulate equity protection on historical equity curve.
    
    Args:
        equity_curve (List[float]): Historical equity values
        drawdown_threshold (float): Drawdown threshold
        plot_results (bool): Whether to plot results
        
    Returns:
        EquityProtection: Configured protection instance
    """
    protection = EquityProtection(drawdown_threshold=drawdown_threshold)
    
    for i, equity in enumerate(equity_curve):
        protection.update_equity(equity, f"Period_{i}")
    
    if plot_results:
        protection.plot_equity_curve(title="Equity Protection Simulation")
    
    return protection


# Example usage and testing
def example_usage():
    """Example usage of equity protection system."""
    print("Example Equity Protection Usage:")
    print("=" * 40)
    
    # Create protection instance
    protection = EquityProtection(drawdown_threshold=0.25)
    
    # Simulate equity curve
    np.random.seed(42)
    equity_values = [100000]  # Starting equity
    
    for i in range(1000):
        # Simulate returns with occasional large losses
        if i == 500:  # Simulate large loss
            return_rate = -0.30
        else:
            return_rate = np.random.normal(0.0005, 0.02)  # 0.05% daily return, 2% volatility
        
        new_equity = equity_values[-1] * (1 + return_rate)
        equity_values.append(new_equity)
        
        # Update protection
        protection.update_equity(new_equity, f"Day_{i}")
    
    # Generate report
    print(protection.generate_protection_report())
    
    # Plot results
    protection.plot_equity_curve(title="Example Equity Protection")
    
    return protection


if __name__ == "__main__":
    example_usage()