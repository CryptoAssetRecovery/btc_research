"""
Equity Protection Module for BTC Research Framework.

This module implements comprehensive equity curve protection mechanisms to prevent
catastrophic drawdowns. It provides first-loss stop functionality that flattens
all positions and disables new entries when drawdown exceeds configurable thresholds.

Key Features:
- Real-time equity curve tracking
- Configurable drawdown thresholds (default 25%)
- Automatic position flattening on threshold breach
- Trading disable/enable based on bias changes
- Comprehensive statistics and monitoring
- Backtrader integration for live trading protection

Usage:
    >>> from btc_research.core.equity_protection import EquityProtection
    >>> 
    >>> # Initialize with 25% drawdown threshold
    >>> protection = EquityProtection(drawdown_threshold=0.25)
    >>> 
    >>> # Update equity in trading loop
    >>> protection.update_equity(current_equity)
    >>> 
    >>> # Check if trading should be disabled
    >>> if protection.should_disable_trading():
    ...     # Flatten all positions and disable new entries
    ...     pass
    >>> 
    >>> # Re-enable after bias flip
    >>> protection.reset_on_bias_flip("bull")
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import backtrader as bt
    HAS_BACKTRADER = True
except ImportError:
    bt = None
    HAS_BACKTRADER = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False


__all__ = [
    "EquityProtection",
    "EquityProtectionError", 
    "EquityProtectionAnalyzer",
    "ProtectedStrategy",
    "create_equity_protection_config"
]


class EquityProtectionError(Exception):
    """Raised when there are issues with equity protection operations."""
    pass


class EquityProtection:
    """
    Comprehensive equity protection system with first-loss stop functionality.
    
    This class implements a sophisticated equity curve protection mechanism that
    prevents catastrophic drawdowns by:
    
    1. **Continuous Monitoring**: Tracks equity curve and calculates real-time drawdown
    2. **Threshold Protection**: Automatically disables trading when drawdown exceeds threshold
    3. **Position Flattening**: Provides hooks to flatten all positions on threshold breach
    4. **Bias-Based Recovery**: Re-enables trading only when market bias flips
    5. **Comprehensive Statistics**: Detailed equity curve analysis and reporting
    
    The system is designed to prevent the kind of catastrophic drawdowns that can
    occur with aggressive strategies, particularly in volatile markets like Bitcoin.
    """
    
    def __init__(
        self,
        drawdown_threshold: float = 0.25,
        enable_on_bias_flip: bool = True,
        smoothing_window: int = 5,
        min_equity_history: int = 10,
        debug: bool = False
    ):
        """
        Initialize equity protection system.
        
        Args:
            drawdown_threshold: Maximum drawdown before disabling trading (0.25 = 25%)
            enable_on_bias_flip: Whether to re-enable trading on bias flip
            smoothing_window: Window for equity curve smoothing (0 = no smoothing)
            min_equity_history: Minimum history needed before protection activates
            debug: Enable debug logging
        """
        self.drawdown_threshold = drawdown_threshold
        self.enable_on_bias_flip = enable_on_bias_flip
        self.smoothing_window = smoothing_window
        self.min_equity_history = min_equity_history
        self.debug = debug
        
        # Core tracking variables
        self.equity_history: List[Dict[str, Any]] = []
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Protection state
        self.protection_active = False
        self.trading_disabled = False
        self.protection_triggered_at = None
        self.protection_triggered_equity = None
        self.protection_triggered_drawdown = None
        
        # Bias tracking for recovery
        self.current_bias = None
        self.bias_history: List[Dict[str, Any]] = []
        self.last_bias_flip_time = None
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'protection_triggers': 0,
            'bias_flips': 0,
            'trading_disabled_periods': 0,
            'max_protection_duration': 0,
            'average_protection_duration': 0
        }
        
        if self.debug:
            print(f"EquityProtection initialized with {drawdown_threshold:.1%} threshold")
    
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update equity and check for protection triggers.
        
        Args:
            equity: Current portfolio equity value
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            Dictionary with update results and protection status
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Validate equity value
        if not isinstance(equity, (int, float)) or equity <= 0:
            raise EquityProtectionError(f"Invalid equity value: {equity}")
        
        # Store previous state for comparison
        previous_drawdown = self.current_drawdown
        previous_protection_active = self.protection_active
        
        # Update equity tracking
        self.current_equity = float(equity)
        self.stats['total_updates'] += 1
        
        # Update peak equity
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            if self.debug:
                print(f"New equity peak: ${self.peak_equity:.2f}")
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0
        
        # Update maximum drawdown
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
        
        # Store equity history
        equity_record = {
            'timestamp': timestamp,
            'equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'drawdown': self.current_drawdown,
            'protection_active': self.protection_active,
            'trading_disabled': self.trading_disabled
        }
        
        self.equity_history.append(equity_record)
        
        # Apply smoothing if configured
        if self.smoothing_window > 1 and len(self.equity_history) >= self.smoothing_window:
            smoothed_drawdown = self._calculate_smoothed_drawdown()
            equity_record['smoothed_drawdown'] = smoothed_drawdown
        else:
            equity_record['smoothed_drawdown'] = self.current_drawdown
        
        # Check for protection trigger
        should_trigger = self._should_trigger_protection()
        
        if should_trigger and not self.protection_active:
            self._trigger_protection(timestamp)
            
        # Prepare response
        response = {
            'timestamp': timestamp,
            'equity': self.current_equity,
            'drawdown': self.current_drawdown,
            'protection_active': self.protection_active,
            'trading_disabled': self.trading_disabled,
            'protection_triggered': should_trigger and not previous_protection_active,
            'drawdown_increased': self.current_drawdown > previous_drawdown
        }
        
        if self.debug and (should_trigger or self.current_drawdown > 0.1):
            print(f"Equity: ${self.current_equity:.2f}, Drawdown: {self.current_drawdown:.1%}, "
                  f"Protection: {self.protection_active}, Trading: {not self.trading_disabled}")
        
        return response
    
    def _should_trigger_protection(self) -> bool:
        """Check if protection should be triggered based on current conditions."""
        # Need minimum history before protection can activate
        if len(self.equity_history) < self.min_equity_history:
            return False
        
        # Check if already active
        if self.protection_active:
            return False
        
        # Use smoothed drawdown if available
        if self.smoothing_window > 1 and len(self.equity_history) >= self.smoothing_window:
            drawdown_to_check = self._calculate_smoothed_drawdown()
        else:
            drawdown_to_check = self.current_drawdown
        
        # Trigger if drawdown exceeds threshold
        return drawdown_to_check >= self.drawdown_threshold
    
    def _trigger_protection(self, timestamp: datetime) -> None:
        """Trigger equity protection mechanism."""
        self.protection_active = True
        self.trading_disabled = True
        self.protection_triggered_at = timestamp
        self.protection_triggered_equity = self.current_equity
        self.protection_triggered_drawdown = self.current_drawdown
        
        self.stats['protection_triggers'] += 1
        
        if self.debug:
            print(f"EQUITY PROTECTION TRIGGERED at {timestamp}")
            print(f"  Equity: ${self.current_equity:.2f}")
            print(f"  Drawdown: {self.current_drawdown:.1%}")
            print(f"  Peak: ${self.peak_equity:.2f}")
    
    def _calculate_smoothed_drawdown(self) -> float:
        """Calculate smoothed drawdown using moving average."""
        if len(self.equity_history) < self.smoothing_window:
            return self.current_drawdown
        
        recent_drawdowns = [
            record['drawdown'] for record in self.equity_history[-self.smoothing_window:]
        ]
        return np.mean(recent_drawdowns)
    
    def is_drawdown_exceeded(self) -> bool:
        """
        Check if drawdown threshold has been exceeded.
        
        Returns:
            True if drawdown exceeds threshold and protection is active
        """
        return self.protection_active
    
    def should_disable_trading(self) -> bool:
        """
        Check if trading should be disabled.
        
        Returns:
            True if trading should be disabled due to equity protection
        """
        return self.trading_disabled
    
    def should_flatten_positions(self) -> bool:
        """
        Check if all positions should be flattened immediately.
        
        Returns:
            True if positions should be flattened due to protection trigger
        """
        return self.protection_active and self.trading_disabled
    
    def update_bias(self, new_bias: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update market bias and check for bias flip recovery.
        
        Args:
            new_bias: New market bias ("bull", "bear", "neutral")
            timestamp: Optional timestamp
            
        Returns:
            Dictionary with bias update results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Validate bias value
        valid_biases = {"bull", "bear", "neutral"}
        if new_bias not in valid_biases:
            raise EquityProtectionError(f"Invalid bias: {new_bias}. Must be one of {valid_biases}")
        
        previous_bias = self.current_bias
        previous_trading_disabled = self.trading_disabled
        
        # Update bias
        self.current_bias = new_bias
        
        # Store bias history
        bias_record = {
            'timestamp': timestamp,
            'bias': new_bias,
            'previous_bias': previous_bias,
            'is_flip': previous_bias is not None and previous_bias != new_bias
        }
        
        self.bias_history.append(bias_record)
        
        # Check for bias flip
        if bias_record['is_flip']:
            self.last_bias_flip_time = timestamp
            self.stats['bias_flips'] += 1
            
            if self.debug:
                print(f"Bias flip detected: {previous_bias} -> {new_bias}")
            
            # Re-enable trading if protection allows it
            if self.enable_on_bias_flip and self.trading_disabled:
                self._check_bias_flip_recovery(timestamp)
        
        return {
            'timestamp': timestamp,
            'bias': new_bias,
            'previous_bias': previous_bias,
            'is_flip': bias_record['is_flip'],
            'trading_enabled': not self.trading_disabled,
            'trading_status_changed': previous_trading_disabled != self.trading_disabled
        }
    
    def _check_bias_flip_recovery(self, timestamp: datetime) -> None:
        """Check if bias flip should trigger recovery from protection."""
        if not self.protection_active:
            return
        
        # Re-enable trading on bias flip
        if self.enable_on_bias_flip:
            self.trading_disabled = False
            
            # Calculate protection duration
            if self.protection_triggered_at:
                duration = (timestamp - self.protection_triggered_at).total_seconds()
                if duration > self.stats['max_protection_duration']:
                    self.stats['max_protection_duration'] = duration
            
            if self.debug:
                print(f"Trading re-enabled due to bias flip to {self.current_bias}")
    
    def reset_on_bias_flip(self, new_bias: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Reset protection state on bias flip (legacy method name).
        
        Args:
            new_bias: New market bias
            timestamp: Optional timestamp
            
        Returns:
            Dictionary with reset results
        """
        return self.update_bias(new_bias, timestamp)
    
    def force_reset(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Force reset of protection state (emergency override).
        
        Args:
            timestamp: Optional timestamp
            
        Returns:
            Dictionary with reset results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        previous_state = {
            'protection_active': self.protection_active,
            'trading_disabled': self.trading_disabled
        }
        
        # Reset protection state
        self.protection_active = False
        self.trading_disabled = False
        
        if self.debug:
            print(f"Equity protection force reset at {timestamp}")
        
        return {
            'timestamp': timestamp,
            'reset_type': 'force',
            'previous_state': previous_state,
            'new_state': {
                'protection_active': self.protection_active,
                'trading_disabled': self.trading_disabled
            }
        }
    
    def get_equity_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive equity protection statistics.
        
        Returns:
            Dictionary with detailed statistics
        """
        stats = {
            # Current State
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            
            # Protection State
            'protection_active': self.protection_active,
            'trading_disabled': self.trading_disabled,
            'drawdown_threshold': self.drawdown_threshold,
            
            # Trigger Information
            'protection_triggered_at': self.protection_triggered_at,
            'protection_triggered_equity': self.protection_triggered_equity,
            'protection_triggered_drawdown': self.protection_triggered_drawdown,
            
            # Bias Information
            'current_bias': self.current_bias,
            'last_bias_flip_time': self.last_bias_flip_time,
            
            # Historical Statistics
            'total_updates': self.stats['total_updates'],
            'protection_triggers': self.stats['protection_triggers'],
            'bias_flips': self.stats['bias_flips'],
            'max_protection_duration': self.stats['max_protection_duration'],
            
            # Data Points
            'equity_history_length': len(self.equity_history),
            'bias_history_length': len(self.bias_history),
            
            # Configuration
            'smoothing_window': self.smoothing_window,
            'min_equity_history': self.min_equity_history,
            'enable_on_bias_flip': self.enable_on_bias_flip
        }
        
        return stats
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as pandas DataFrame.
        
        Returns:
            DataFrame with equity curve data
        """
        if not self.equity_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_history)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_drawdown_curve(self) -> pd.DataFrame:
        """
        Get drawdown curve as pandas DataFrame.
        
        Returns:
            DataFrame with drawdown curve data
        """
        df = self.get_equity_curve()
        if df.empty:
            return pd.DataFrame()
        
        # Calculate additional drawdown metrics
        df['drawdown_pct'] = df['drawdown'] * 100
        df['drawdown_amount'] = df['peak_equity'] - df['equity']
        df['recovery_factor'] = df['equity'] / df['peak_equity']
        
        return df[['drawdown', 'drawdown_pct', 'drawdown_amount', 'recovery_factor', 'protection_active']]
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (12, 8)) -> Optional[plt.Figure]:
        """
        Plot equity curve with protection levels.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure if available, None otherwise
        """
        if not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not available for plotting")
            return None
        
        if not self.equity_history:
            warnings.warn("No equity history available for plotting")
            return None
        
        df = self.get_equity_curve()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot equity curve
        ax1.plot(df.index, df['equity'], label='Equity', linewidth=2, color='blue')
        ax1.plot(df.index, df['peak_equity'], label='Peak Equity', linewidth=1, color='green', alpha=0.7)
        
        # Highlight protection periods
        protection_periods = df[df['protection_active']]
        if not protection_periods.empty:
            ax1.scatter(protection_periods.index, protection_periods['equity'], 
                       color='red', s=20, label='Protection Active', alpha=0.8)
        
        ax1.set_title('Equity Curve with Protection Levels')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown
        ax2.fill_between(df.index, df['drawdown'] * 100, 0, 
                        alpha=0.3, color='red', label='Drawdown')
        ax2.axhline(y=self.drawdown_threshold * 100, color='red', linestyle='--', 
                   label=f'Protection Threshold ({self.drawdown_threshold:.1%})')
        
        ax2.set_title('Drawdown with Protection Threshold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Drawdown should go down
        
        plt.tight_layout()
        return fig


class EquityProtectionAnalyzer:
    """
    Analyzer for equity protection performance and effectiveness.
    
    This class provides comprehensive analysis of equity protection system
    performance, including statistics on protection triggers, recovery times,
    and overall effectiveness in preventing catastrophic drawdowns.
    """
    
    def __init__(self, equity_protection: EquityProtection):
        """
        Initialize analyzer with equity protection instance.
        
        Args:
            equity_protection: EquityProtection instance to analyze
        """
        self.protection = equity_protection
    
    def analyze_protection_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze effectiveness of equity protection system.
        
        Returns:
            Dictionary with effectiveness analysis
        """
        df = self.protection.get_equity_curve()
        if df.empty:
            return {'error': 'No equity data available'}
        
        # Calculate protection effectiveness metrics
        analysis = {
            'protection_summary': {
                'total_triggers': self.protection.stats['protection_triggers'],
                'max_drawdown_observed': self.protection.max_drawdown,
                'max_drawdown_prevented': max(0, self.protection.max_drawdown - self.protection.drawdown_threshold),
                'protection_threshold': self.protection.drawdown_threshold,
                'effectiveness_ratio': min(1.0, self.protection.drawdown_threshold / max(self.protection.max_drawdown, 0.001))
            },
            
            'recovery_analysis': self._analyze_recovery_periods(df),
            'drawdown_analysis': self._analyze_drawdown_patterns(df),
            'bias_flip_analysis': self._analyze_bias_flip_effectiveness()
        }
        
        return analysis
    
    def _analyze_recovery_periods(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze recovery periods from protection triggers."""
        if df.empty or 'protection_active' not in df.columns:
            return {'error': 'No protection data available'}
        
        # Find protection periods
        protection_changes = df['protection_active'].diff()
        protection_starts = df[protection_changes == 1].index
        protection_ends = df[protection_changes == -1].index
        
        recovery_times = []
        for start in protection_starts:
            # Find corresponding end
            ends_after_start = protection_ends[protection_ends > start]
            if len(ends_after_start) > 0:
                end = ends_after_start[0]
                duration = (end - start).total_seconds() / 3600  # Hours
                recovery_times.append(duration)
        
        if recovery_times:
            return {
                'avg_recovery_time_hours': np.mean(recovery_times),
                'max_recovery_time_hours': np.max(recovery_times),
                'min_recovery_time_hours': np.min(recovery_times),
                'total_recovery_periods': len(recovery_times)
            }
        else:
            return {'no_recovery_periods': True}
    
    def _analyze_drawdown_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze drawdown patterns and protection effectiveness."""
        if df.empty:
            return {'error': 'No data available'}
        
        # Calculate drawdown statistics
        drawdowns = df['drawdown'].values
        
        # Find consecutive drawdown periods
        drawdown_periods = []
        current_period = []
        
        for i, dd in enumerate(drawdowns):
            if dd > 0:
                current_period.append(dd)
            else:
                if current_period:
                    drawdown_periods.append(current_period)
                    current_period = []
        
        if current_period:
            drawdown_periods.append(current_period)
        
        # Analyze drawdown periods
        if drawdown_periods:
            max_drawdowns = [max(period) for period in drawdown_periods]
            period_lengths = [len(period) for period in drawdown_periods]
            
            return {
                'total_drawdown_periods': len(drawdown_periods),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'avg_drawdown_period_length': np.mean(period_lengths),
                'drawdowns_exceeding_threshold': sum(1 for dd in max_drawdowns if dd >= self.protection.drawdown_threshold),
                'protection_effectiveness_pct': (1 - sum(1 for dd in max_drawdowns if dd >= self.protection.drawdown_threshold) / len(max_drawdowns)) * 100
            }
        else:
            return {'no_drawdown_periods': True}
    
    def _analyze_bias_flip_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of bias flip recovery mechanism."""
        if not self.protection.bias_history:
            return {'error': 'No bias data available'}
        
        bias_flips = [record for record in self.protection.bias_history if record['is_flip']]
        
        if not bias_flips:
            return {'no_bias_flips': True}
        
        # Calculate recovery success rate
        successful_recoveries = 0
        total_recoveries = 0
        
        for flip in bias_flips:
            # Check if trading was disabled at time of flip
            flip_time = flip['timestamp']
            
            # Find equity state at time of flip
            equity_at_flip = None
            for record in self.protection.equity_history:
                if record['timestamp'] >= flip_time:
                    equity_at_flip = record
                    break
            
            if equity_at_flip and equity_at_flip['trading_disabled']:
                total_recoveries += 1
                # Check if trading was subsequently enabled
                # This is a simplified check - in practice, you'd want more sophisticated analysis
                if self.protection.enable_on_bias_flip:
                    successful_recoveries += 1
        
        recovery_success_rate = (successful_recoveries / total_recoveries * 100) if total_recoveries > 0 else 0
        
        return {
            'total_bias_flips': len(bias_flips),
            'recovery_opportunities': total_recoveries,
            'successful_recoveries': successful_recoveries,
            'recovery_success_rate_pct': recovery_success_rate
        }
    
    def generate_protection_report(self) -> str:
        """
        Generate comprehensive protection effectiveness report.
        
        Returns:
            Formatted string report
        """
        stats = self.protection.get_equity_stats()
        analysis = self.analyze_protection_effectiveness()
        
        report = f"""
Equity Protection System Report
==============================

Current Status:
- Equity: ${stats['current_equity']:,.2f}
- Peak Equity: ${stats['peak_equity']:,.2f}
- Current Drawdown: {stats['current_drawdown']:.1%}
- Max Drawdown: {stats['max_drawdown']:.1%}
- Protection Active: {stats['protection_active']}
- Trading Disabled: {stats['trading_disabled']}

Protection Configuration:
- Drawdown Threshold: {stats['drawdown_threshold']:.1%}
- Bias Flip Recovery: {stats['enable_on_bias_flip']}
- Smoothing Window: {stats['smoothing_window']} periods
- Min History: {stats['min_equity_history']} updates

Protection Performance:
- Total Triggers: {stats['protection_triggers']}
- Total Updates: {stats['total_updates']}
- Bias Flips: {stats['bias_flips']}
- Max Protection Duration: {stats['max_protection_duration']:.1f} seconds

"""
        
        if 'protection_summary' in analysis:
            ps = analysis['protection_summary']
            report += f"""Protection Effectiveness:
- Effectiveness Ratio: {ps['effectiveness_ratio']:.2f}
- Max Drawdown Prevented: {ps['max_drawdown_prevented']:.1%}
- Protection Success: {'Yes' if ps['max_drawdown_prevented'] > 0 else 'No'}

"""
        
        if 'recovery_analysis' in analysis and 'avg_recovery_time_hours' in analysis['recovery_analysis']:
            ra = analysis['recovery_analysis']
            report += f"""Recovery Analysis:
- Average Recovery Time: {ra['avg_recovery_time_hours']:.1f} hours
- Max Recovery Time: {ra['max_recovery_time_hours']:.1f} hours
- Total Recovery Periods: {ra['total_recovery_periods']}

"""
        
        if 'bias_flip_analysis' in analysis and 'total_bias_flips' in analysis['bias_flip_analysis']:
            bfa = analysis['bias_flip_analysis']
            report += f"""Bias Flip Analysis:
- Total Bias Flips: {bfa['total_bias_flips']}
- Recovery Success Rate: {bfa.get('recovery_success_rate_pct', 0):.1f}%
- Recovery Opportunities: {bfa.get('recovery_opportunities', 0)}

"""
        
        return report.strip()


class ProtectedStrategy(bt.Strategy):
    """
    Backtrader strategy wrapper with integrated equity protection.
    
    This class extends Backtrader's strategy framework to include real-time
    equity protection monitoring and automatic position management based on
    drawdown thresholds.
    """
    
    params = (
        ('equity_protection', None),  # EquityProtection instance
        ('flatten_on_trigger', True),  # Flatten positions on protection trigger
        ('base_strategy_class', None),  # Base strategy class to wrap
        ('base_strategy_params', {}),  # Parameters for base strategy
    )
    
    def __init__(self):
        """Initialize protected strategy."""
        super().__init__()
        
        # Initialize equity protection
        if self.params.equity_protection is None:
            self.equity_protection = EquityProtection()
        else:
            self.equity_protection = self.params.equity_protection
        
        # Initialize base strategy if provided
        if self.params.base_strategy_class is not None:
            self.base_strategy = self.params.base_strategy_class(**self.params.base_strategy_params)
        else:
            self.base_strategy = None
        
        # Track protection state
        self.last_protection_check = None
        self.positions_flattened = False
    
    def next(self):
        """Process next bar with equity protection checks."""
        # Update equity protection
        current_equity = self.broker.getvalue()
        protection_update = self.equity_protection.update_equity(current_equity)
        
        # Check if protection was just triggered
        if protection_update['protection_triggered']:
            self._handle_protection_trigger()
        
        # Execute base strategy if trading is enabled
        if not self.equity_protection.should_disable_trading():
            if self.base_strategy and hasattr(self.base_strategy, 'next'):
                self.base_strategy.next()
        
        # Reset position flattening flag if protection is no longer active
        if not self.equity_protection.protection_active:
            self.positions_flattened = False
    
    def _handle_protection_trigger(self):
        """Handle equity protection trigger."""
        if self.params.flatten_on_trigger and not self.positions_flattened:
            # Flatten all positions
            if self.position:
                self.close()
                self.positions_flattened = True
                
                print(f"EQUITY PROTECTION: Flattened all positions due to {self.equity_protection.current_drawdown:.1%} drawdown")
    
    def notify_order(self, order):
        """Override order notifications to include protection status."""
        if hasattr(self.base_strategy, 'notify_order'):
            self.base_strategy.notify_order(order)
    
    def notify_trade(self, trade):
        """Override trade notifications to include protection status."""
        if hasattr(self.base_strategy, 'notify_trade'):
            self.base_strategy.notify_trade(trade)


def create_equity_protection_config(
    drawdown_threshold: float = 0.25,
    enable_on_bias_flip: bool = True,
    smoothing_window: int = 5,
    min_equity_history: int = 10
) -> Dict[str, Any]:
    """
    Create equity protection configuration dictionary.
    
    Args:
        drawdown_threshold: Maximum drawdown before protection triggers
        enable_on_bias_flip: Whether to re-enable trading on bias flip
        smoothing_window: Window for equity curve smoothing
        min_equity_history: Minimum history before protection activates
        
    Returns:
        Configuration dictionary for equity protection
    """
    return {
        'equity_protection': {
            'enabled': True,
            'drawdown_threshold': drawdown_threshold,
            'enable_on_bias_flip': enable_on_bias_flip,
            'smoothing_window': smoothing_window,
            'min_equity_history': min_equity_history,
            'flatten_positions_on_trigger': True,
            'disable_new_entries': True,
            'recovery_mechanism': 'bias_flip' if enable_on_bias_flip else 'manual'
        },
        'risk_management': {
            'max_drawdown_threshold': drawdown_threshold,
            'emergency_stop_enabled': True,
            'position_flattening_enabled': True,
            'new_entry_control': True
        }
    }


def integrate_equity_protection_with_strategy(
    strategy_config: Dict[str, Any],
    equity_protection_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Integrate equity protection configuration with existing strategy configuration.
    
    Args:
        strategy_config: Existing strategy configuration
        equity_protection_config: Optional equity protection configuration
        
    Returns:
        Updated strategy configuration with equity protection
    """
    if equity_protection_config is None:
        equity_protection_config = create_equity_protection_config()
    
    # Create a copy of the strategy config
    integrated_config = strategy_config.copy()
    
    # Add equity protection configuration
    integrated_config.update(equity_protection_config)
    
    # Add equity protection rules to logic if not present
    if 'logic' not in integrated_config:
        integrated_config['logic'] = {}
    
    # Add protection checks to entry conditions
    entry_protection_rule = "not equity_protection_active"
    
    if 'entry_long' in integrated_config['logic']:
        if entry_protection_rule not in integrated_config['logic']['entry_long']:
            integrated_config['logic']['entry_long'].append(entry_protection_rule)
    
    if 'entry_short' in integrated_config['logic']:
        if entry_protection_rule not in integrated_config['logic']['entry_short']:
            integrated_config['logic']['entry_short'].append(entry_protection_rule)
    
    # Add protection-based exit rules
    protection_exit_rule = "equity_protection_triggered"
    
    if 'exit_long' in integrated_config['logic']:
        if protection_exit_rule not in integrated_config['logic']['exit_long']:
            integrated_config['logic']['exit_long'].append(protection_exit_rule)
    
    if 'exit_short' in integrated_config['logic']:
        if protection_exit_rule not in integrated_config['logic']['exit_short']:
            integrated_config['logic']['exit_short'].append(protection_exit_rule)
    
    return integrated_config


# Example usage and integration functions
def create_protected_backtest_config() -> Dict[str, Any]:
    """
    Create a sample configuration with equity protection for backtesting.
    
    Returns:
        Complete configuration with equity protection
    """
    base_config = {
        'name': 'Protected Trading Strategy',
        'symbol': 'BTC/USD',
        'timeframes': {'entry': '1h'},
        'indicators': [
            {
                'id': 'protection_monitor',
                'type': 'EquityProtection',
                'timeframe': '1h',
                'drawdown_threshold': 0.25,
                'enable_on_bias_flip': True
            }
        ],
        'logic': {
            'entry_long': ['rsi < 30', 'not equity_protection_active'],
            'exit_long': ['rsi > 70', 'equity_protection_triggered'],
            'entry_short': ['rsi > 70', 'not equity_protection_active'],
            'exit_short': ['rsi < 30', 'equity_protection_triggered']
        },
        'backtest': {
            'from': '2024-01-01',
            'to': '2024-12-31',
            'initial_cash': 10000,
            'commission': 0.001
        }
    }
    
    # Add equity protection
    equity_protection_config = create_equity_protection_config()
    return integrate_equity_protection_with_strategy(base_config, equity_protection_config)


if __name__ == "__main__":
    # Example usage
    print("Equity Protection System Example")
    print("================================")
    
    # Create equity protection instance
    protection = EquityProtection(drawdown_threshold=0.25, debug=True)
    
    # Simulate equity updates
    equity_values = [10000, 10500, 11000, 9500, 8000, 7500, 7000, 8500, 9000, 9500]
    
    for i, equity in enumerate(equity_values):
        result = protection.update_equity(equity)
        print(f"Update {i+1}: Equity=${equity}, Drawdown={result['drawdown']:.1%}, "
              f"Protection={result['protection_active']}")
        
        # Simulate bias flip after protection trigger
        if result['protection_triggered']:
            print("  -> Simulating bias flip to bull")
            protection.update_bias("bull")
    
    # Print final statistics
    stats = protection.get_equity_stats()
    print(f"\nFinal Statistics:")
    print(f"  Max Drawdown: {stats['max_drawdown']:.1%}")
    print(f"  Protection Triggers: {stats['protection_triggers']}")
    print(f"  Trading Disabled: {stats['trading_disabled']}")
    
    # Generate report
    analyzer = EquityProtectionAnalyzer(protection)
    report = analyzer.generate_protection_report()
    print(f"\n{report}")