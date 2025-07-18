"""
Equity Protection Integration Utilities.

This module provides utilities to integrate the equity protection system with
existing trading strategies, backtesting frameworks, and live trading systems.
It simplifies the process of adding catastrophic drawdown protection to any
trading strategy.

Key Features:
- Easy integration with existing strategies
- Backtrader strategy wrapper
- Configuration helpers
- Visualization utilities
- Performance analysis tools

Usage:
    >>> from btc_research.utils.equity_protection_integration import (
    ...     add_equity_protection_to_config,
    ...     create_protected_strategy,
    ...     EquityProtectedBacktester
    ... )
    >>> 
    >>> # Add protection to existing config
    >>> protected_config = add_equity_protection_to_config(strategy_config)
    >>> 
    >>> # Create protected backtester
    >>> backtester = EquityProtectedBacktester(protected_config)
    >>> results = backtester.run(df)
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from btc_research.core.equity_protection import (
    EquityProtection,
    EquityProtectionAnalyzer,
    EquityProtectionError,
    ProtectedStrategy
)
from btc_research.core.backtester import Backtester, BacktesterError

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
    "add_equity_protection_to_config",
    "create_protected_strategy",
    "EquityProtectedBacktester",
    "EquityProtectionMonitor",
    "analyze_strategy_with_protection",
    "generate_protection_comparison_report"
]


def add_equity_protection_to_config(
    strategy_config: Dict[str, Any],
    drawdown_threshold: float = 0.25,
    enable_on_bias_flip: bool = True,
    smoothing_window: int = 5,
    recovery_mechanism: str = "bias_flip"
) -> Dict[str, Any]:
    """
    Add equity protection configuration to an existing strategy configuration.
    
    Args:
        strategy_config: Existing strategy configuration dictionary
        drawdown_threshold: Maximum drawdown before protection triggers (0.25 = 25%)
        enable_on_bias_flip: Whether to re-enable trading on bias flip
        smoothing_window: Window for equity curve smoothing
        recovery_mechanism: Recovery mechanism ("bias_flip", "manual", "time_based")
        
    Returns:
        Updated strategy configuration with equity protection
    """
    # Create a deep copy to avoid modifying original
    protected_config = strategy_config.copy()
    
    # Add equity protection configuration
    protected_config['equity_protection'] = {
        'enabled': True,
        'drawdown_threshold': drawdown_threshold,
        'enable_on_bias_flip': enable_on_bias_flip,
        'smoothing_window': smoothing_window,
        'min_equity_history': 10,
        'flatten_positions_on_trigger': True,
        'disable_new_entries': True,
        'recovery_mechanism': recovery_mechanism,
        'emergency_controls': {
            'force_reset_after_hours': 168,  # 7 days
            'max_consecutive_triggers': 3,
            'alert_on_trigger': True
        }
    }
    
    # Add protection-aware logic rules
    if 'logic' not in protected_config:
        protected_config['logic'] = {}
    
    # Add protection checks to entry conditions
    protection_entry_rule = "not equity_protection_active"
    
    for entry_type in ['entry_long', 'entry_short']:
        if entry_type in protected_config['logic']:
            if isinstance(protected_config['logic'][entry_type], list):
                if protection_entry_rule not in protected_config['logic'][entry_type]:
                    protected_config['logic'][entry_type].append(protection_entry_rule)
            else:
                # Convert single rule to list and add protection rule
                protected_config['logic'][entry_type] = [
                    protected_config['logic'][entry_type],
                    protection_entry_rule
                ]
    
    # Add protection-based exit rules
    protection_exit_rule = "equity_protection_triggered"
    
    for exit_type in ['exit_long', 'exit_short']:
        if exit_type in protected_config['logic']:
            if isinstance(protected_config['logic'][exit_type], list):
                if protection_exit_rule not in protected_config['logic'][exit_type]:
                    protected_config['logic'][exit_type].append(protection_exit_rule)
            else:
                protected_config['logic'][exit_type] = [
                    protected_config['logic'][exit_type],
                    protection_exit_rule
                ]
    
    # Add bias detection configuration if not present
    if 'bias_detection' not in protected_config:
        protected_config['bias_detection'] = {
            'method': 'price_ma',
            'ma_period': 50,
            'trend_threshold': 0.02,  # 2% move to confirm bias flip
            'min_bars_for_flip': 5    # Minimum bars to confirm bias flip
        }
    
    # Update risk management to account for protection
    if 'risk_management' not in protected_config:
        protected_config['risk_management'] = {}
    
    # Add protection-specific risk management
    protected_config['risk_management']['protection_mode'] = {
        'max_position_pct': 0.1,      # Reduce position size when recovering
        'risk_pct': 0.005,            # Reduce risk when recovering
        'require_stronger_signals': True
    }
    
    # Add monitoring configuration
    protected_config['monitoring'] = {
        'equity_protection_enabled': True,
        'track_drawdown_history': True,
        'generate_protection_report': True,
        'alert_on_large_drawdowns': True,
        'save_protection_events': True
    }
    
    return protected_config


def create_protected_strategy(
    base_strategy_class: Any,
    equity_protection: Optional[EquityProtection] = None,
    flatten_on_trigger: bool = True,
    **base_strategy_params
) -> type:
    """
    Create a protected version of an existing Backtrader strategy class.
    
    Args:
        base_strategy_class: Original Backtrader strategy class
        equity_protection: Optional EquityProtection instance
        flatten_on_trigger: Whether to flatten positions on protection trigger
        **base_strategy_params: Parameters for the base strategy
        
    Returns:
        New protected strategy class
    """
    if not HAS_BACKTRADER:
        raise EquityProtectionError("Backtrader is required for protected strategies")
    
    class DynamicProtectedStrategy(ProtectedStrategy):
        params = (
            ('equity_protection', equity_protection),
            ('flatten_on_trigger', flatten_on_trigger),
            ('base_strategy_class', base_strategy_class),
            ('base_strategy_params', base_strategy_params),
        )
    
    return DynamicProtectedStrategy


class EquityProtectedBacktester(Backtester):
    """
    Extended backtester with integrated equity protection.
    
    This class extends the standard Backtester to include equity protection
    monitoring, automatic position flattening, and comprehensive protection
    analysis in the backtest results.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        equity_protection: Optional[EquityProtection] = None,
        debug: bool = False
    ):
        """
        Initialize equity protected backtester.
        
        Args:
            config: Configuration with equity protection settings
            equity_protection: Optional EquityProtection instance
            debug: Enable debug output
        """
        super().__init__(config, debug=debug)
        
        # Initialize equity protection
        if equity_protection is not None:
            self.equity_protection = equity_protection
        else:
            protection_config = config.get('equity_protection', {})
            if protection_config.get('enabled', False):
                self.equity_protection = EquityProtection(
                    drawdown_threshold=protection_config.get('drawdown_threshold', 0.25),
                    enable_on_bias_flip=protection_config.get('enable_on_bias_flip', True),
                    smoothing_window=protection_config.get('smoothing_window', 5),
                    min_equity_history=protection_config.get('min_equity_history', 10),
                    debug=debug
                )
            else:
                self.equity_protection = None
        
        self.protection_analyzer = None
        if self.equity_protection is not None:
            self.protection_analyzer = EquityProtectionAnalyzer(self.equity_protection)
    
    def run(
        self,
        df: pd.DataFrame,
        cash: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute backtest with equity protection monitoring.
        
        Args:
            df: DataFrame with OHLCV data and indicator columns
            cash: Starting capital
            commission: Commission rate
            slippage: Slippage rate
            
        Returns:
            Dictionary with backtest results including protection analysis
        """
        # Run standard backtest
        results = super().run(df, cash, commission, slippage)
        
        # Add equity protection analysis if enabled
        if self.equity_protection is not None and self.protection_analyzer is not None:
            # Simulate equity protection during backtest
            self._simulate_equity_protection(df, results)
            
            # Add protection analysis to results
            protection_analysis = self.protection_analyzer.analyze_protection_effectiveness()
            results['equity_protection'] = {
                'enabled': True,
                'configuration': self.equity_protection.get_equity_stats(),
                'analysis': protection_analysis,
                'report': self.protection_analyzer.generate_protection_report()
            }
            
            # Add protection-specific metrics
            results['protection_metrics'] = self._calculate_protection_metrics(results)
        
        return results
    
    def _simulate_equity_protection(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        Simulate equity protection during backtest (post-hoc analysis).
        
        Args:
            df: Original DataFrame
            results: Backtest results dictionary
        """
        if 'equity_curve' not in results:
            return
        
        # Simulate protection monitoring using equity curve
        for equity_point in results['equity_curve']:
            timestamp_str = equity_point['timestamp']
            equity_value = equity_point['equity']
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            
            # Update equity protection
            protection_update = self.equity_protection.update_equity(equity_value, timestamp)
            
            # Simulate bias detection (simplified)
            if protection_update.get('protection_triggered', False):
                # Simulate bias flip after some time
                recovery_time = timestamp + timedelta(hours=24)  # Assume 24-hour recovery
                self.equity_protection.update_bias("bull", recovery_time)
    
    def _calculate_protection_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate protection-specific performance metrics.
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            Dictionary with protection metrics
        """
        if self.equity_protection is None:
            return {}
        
        stats = self.equity_protection.get_equity_stats()
        
        # Calculate protection effectiveness
        theoretical_max_drawdown = results.get('max_drawdown', 0)
        protection_threshold = stats['drawdown_threshold']
        
        protection_effectiveness = min(1.0, protection_threshold / max(theoretical_max_drawdown, 0.001))
        
        # Calculate recovery metrics
        recovery_metrics = {}
        if stats['protection_triggers'] > 0:
            recovery_metrics['avg_recovery_time'] = stats.get('max_protection_duration', 0) / stats['protection_triggers']
            recovery_metrics['protection_success_rate'] = min(1.0, protection_effectiveness)
        
        return {
            'protection_effectiveness': protection_effectiveness,
            'protection_triggers': stats['protection_triggers'],
            'max_drawdown_prevented': max(0, theoretical_max_drawdown - protection_threshold),
            'recovery_metrics': recovery_metrics,
            'protection_impact': {
                'theoretical_max_drawdown': theoretical_max_drawdown,
                'actual_max_drawdown': stats['max_drawdown'],
                'drawdown_reduction': theoretical_max_drawdown - stats['max_drawdown']
            }
        }


class EquityProtectionMonitor:
    """
    Real-time equity protection monitoring for live trading.
    
    This class provides real-time monitoring capabilities for live trading
    systems, including alerts, automatic position management, and logging.
    """
    
    def __init__(
        self,
        equity_protection: EquityProtection,
        alert_callback: Optional[callable] = None,
        position_manager: Optional[callable] = None,
        log_file: Optional[str] = None
    ):
        """
        Initialize equity protection monitor.
        
        Args:
            equity_protection: EquityProtection instance
            alert_callback: Optional callback function for alerts
            position_manager: Optional callback for position management
            log_file: Optional log file path
        """
        self.equity_protection = equity_protection
        self.alert_callback = alert_callback
        self.position_manager = position_manager
        self.log_file = log_file
        
        # Monitoring state
        self.last_alert_time = None
        self.alert_cooldown = timedelta(minutes=5)  # 5-minute cooldown between alerts
        
        # Event log
        self.event_log: List[Dict[str, Any]] = []
    
    def update_and_monitor(self, equity: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update equity and perform monitoring actions.
        
        Args:
            equity: Current equity value
            timestamp: Optional timestamp
            
        Returns:
            Dictionary with monitoring results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update equity protection
        protection_update = self.equity_protection.update_equity(equity, timestamp)
        
        # Log event
        event = {
            'timestamp': timestamp,
            'equity': equity,
            'drawdown': protection_update['drawdown'],
            'protection_active': protection_update['protection_active'],
            'protection_triggered': protection_update.get('protection_triggered', False)
        }
        
        self.event_log.append(event)
        
        # Handle protection trigger
        if protection_update.get('protection_triggered', False):
            self._handle_protection_trigger(timestamp, equity, protection_update['drawdown'])
        
        # Handle position flattening
        if self.equity_protection.should_flatten_positions():
            self._handle_position_flattening(timestamp, equity)
        
        # Write to log file if configured
        if self.log_file:
            self._write_to_log(event)
        
        return {
            'monitoring_result': event,
            'protection_update': protection_update,
            'actions_taken': self._get_actions_taken(protection_update)
        }
    
    def _handle_protection_trigger(self, timestamp: datetime, equity: float, drawdown: float) -> None:
        """Handle equity protection trigger."""
        # Send alert
        if self.alert_callback and self._should_send_alert(timestamp):
            alert_message = (
                f"EQUITY PROTECTION TRIGGERED\n"
                f"Time: {timestamp}\n"
                f"Equity: ${equity:,.2f}\n"
                f"Drawdown: {drawdown:.1%}\n"
                f"Action: All positions will be flattened, trading disabled"
            )
            self.alert_callback(alert_message, severity="critical")
            self.last_alert_time = timestamp
        
        # Log critical event
        print(f"CRITICAL: Equity protection triggered at {timestamp}")
        print(f"  Equity: ${equity:,.2f}")
        print(f"  Drawdown: {drawdown:.1%}")
    
    def _handle_position_flattening(self, timestamp: datetime, equity: float) -> None:
        """Handle position flattening."""
        if self.position_manager:
            try:
                self.position_manager("flatten_all")
                print(f"INFO: All positions flattened at {timestamp} due to equity protection")
            except Exception as e:
                print(f"ERROR: Failed to flatten positions: {e}")
    
    def _should_send_alert(self, timestamp: datetime) -> bool:
        """Check if alert should be sent based on cooldown."""
        if self.last_alert_time is None:
            return True
        
        return (timestamp - self.last_alert_time) > self.alert_cooldown
    
    def _get_actions_taken(self, protection_update: Dict[str, Any]) -> List[str]:
        """Get list of actions taken based on protection update."""
        actions = []
        
        if protection_update.get('protection_triggered', False):
            actions.append("protection_triggered")
        
        if self.equity_protection.should_disable_trading():
            actions.append("trading_disabled")
        
        if self.equity_protection.should_flatten_positions():
            actions.append("positions_flattened")
        
        return actions
    
    def _write_to_log(self, event: Dict[str, Any]) -> None:
        """Write event to log file."""
        try:
            with open(self.log_file, 'a') as f:
                log_line = (
                    f"{event['timestamp']}, "
                    f"{event['equity']:.2f}, "
                    f"{event['drawdown']:.4f}, "
                    f"{event['protection_active']}, "
                    f"{event.get('protection_triggered', False)}\n"
                )
                f.write(log_line)
        except Exception as e:
            print(f"Warning: Failed to write to log file: {e}")
    
    def get_monitoring_report(self) -> str:
        """Generate monitoring report."""
        stats = self.equity_protection.get_equity_stats()
        
        report = f"""
Equity Protection Monitoring Report
===================================

Current Status:
- Equity: ${stats['current_equity']:,.2f}
- Peak Equity: ${stats['peak_equity']:,.2f}
- Current Drawdown: {stats['current_drawdown']:.1%}
- Protection Active: {stats['protection_active']}
- Trading Disabled: {stats['trading_disabled']}

Protection Events:
- Total Updates: {stats['total_updates']}
- Protection Triggers: {stats['protection_triggers']}
- Bias Flips: {stats['bias_flips']}
- Event Log Length: {len(self.event_log)}

Configuration:
- Drawdown Threshold: {stats['drawdown_threshold']:.1%}
- Bias Flip Recovery: {stats['enable_on_bias_flip']}
- Smoothing Window: {stats['smoothing_window']}
"""
        
        return report.strip()


def analyze_strategy_with_protection(
    strategy_config: Dict[str, Any],
    df: pd.DataFrame,
    protection_thresholds: List[float] = [0.15, 0.20, 0.25, 0.30],
    cash: float = 10000.0
) -> Dict[str, Any]:
    """
    Analyze strategy performance with different protection thresholds.
    
    Args:
        strategy_config: Strategy configuration
        df: DataFrame with market data
        protection_thresholds: List of protection thresholds to test
        cash: Starting capital
        
    Returns:
        Dictionary with comparative analysis results
    """
    results = {}
    
    # Test without protection
    print("Testing strategy without protection...")
    base_backtester = Backtester(strategy_config)
    base_results = base_backtester.run(df, cash=cash)
    results['no_protection'] = base_results
    
    # Test with different protection thresholds
    for threshold in protection_thresholds:
        print(f"Testing strategy with {threshold:.1%} protection threshold...")
        
        # Add protection to config
        protected_config = add_equity_protection_to_config(
            strategy_config,
            drawdown_threshold=threshold
        )
        
        # Run protected backtest
        protected_backtester = EquityProtectedBacktester(protected_config)
        protected_results = protected_backtester.run(df, cash=cash)
        
        results[f'protection_{threshold:.0%}'] = protected_results
    
    # Calculate comparison metrics
    comparison = _calculate_protection_comparison(results)
    results['comparison'] = comparison
    
    return results


def _calculate_protection_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comparative metrics for protection analysis."""
    comparison = {}
    
    # Base case (no protection)
    base_results = results.get('no_protection', {})
    base_return = base_results.get('total_return', 0)
    base_drawdown = base_results.get('max_drawdown', 0)
    base_sharpe = base_results.get('sharpe_ratio', 0)
    
    comparison['base_metrics'] = {
        'return': base_return,
        'max_drawdown': base_drawdown,
        'sharpe_ratio': base_sharpe
    }
    
    # Protection case comparisons
    protection_comparisons = {}
    
    for key, result in results.items():
        if key.startswith('protection_'):
            threshold = key.replace('protection_', '').replace('%', '')
            
            protected_return = result.get('total_return', 0)
            protected_drawdown = result.get('max_drawdown', 0)
            protected_sharpe = result.get('sharpe_ratio', 0)
            
            protection_comparisons[threshold] = {
                'return': protected_return,
                'max_drawdown': protected_drawdown,
                'sharpe_ratio': protected_sharpe,
                'return_change': protected_return - base_return,
                'drawdown_reduction': base_drawdown - protected_drawdown,
                'sharpe_improvement': protected_sharpe - base_sharpe,
                'risk_adjusted_improvement': (protected_sharpe - base_sharpe) / max(base_sharpe, 0.1)
            }
    
    comparison['protection_comparisons'] = protection_comparisons
    
    return comparison


def generate_protection_comparison_report(analysis_results: Dict[str, Any]) -> str:
    """
    Generate comprehensive protection comparison report.
    
    Args:
        analysis_results: Results from analyze_strategy_with_protection
        
    Returns:
        Formatted comparison report
    """
    comparison = analysis_results.get('comparison', {})
    base_metrics = comparison.get('base_metrics', {})
    protection_comparisons = comparison.get('protection_comparisons', {})
    
    report = f"""
Equity Protection Comparison Report
===================================

Base Strategy (No Protection):
- Total Return: {base_metrics.get('return', 0):.2%}
- Max Drawdown: {base_metrics.get('max_drawdown', 0):.2%}
- Sharpe Ratio: {base_metrics.get('sharpe_ratio', 0):.2f}

Protection Threshold Comparisons:
"""
    
    for threshold, metrics in protection_comparisons.items():
        report += f"""
{threshold} Protection Threshold:
- Total Return: {metrics['return']:.2%} (change: {metrics['return_change']:+.2%})
- Max Drawdown: {metrics['max_drawdown']:.2%} (reduction: {metrics['drawdown_reduction']:+.2%})
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (improvement: {metrics['sharpe_improvement']:+.2f})
- Risk-Adjusted Improvement: {metrics['risk_adjusted_improvement']:+.1%}
"""
    
    # Add recommendations
    report += "\nRecommendations:\n"
    
    # Find best protection threshold
    best_threshold = None
    best_risk_adjusted = float('-inf')
    
    for threshold, metrics in protection_comparisons.items():
        if metrics['risk_adjusted_improvement'] > best_risk_adjusted:
            best_risk_adjusted = metrics['risk_adjusted_improvement']
            best_threshold = threshold
    
    if best_threshold:
        report += f"- Best protection threshold: {best_threshold} (risk-adjusted improvement: {best_risk_adjusted:+.1%})\n"
    
    # Add general recommendations
    report += "- Use equity protection for strategies with high drawdown potential\n"
    report += "- Consider bias-flip recovery for trending markets\n"
    report += "- Monitor protection effectiveness regularly\n"
    
    return report.strip()


def save_protection_config_template(file_path: str) -> None:
    """
    Save a template configuration file with equity protection settings.
    
    Args:
        file_path: Path to save the template configuration
    """
    template_config = {
        'name': 'Equity Protection Template',
        'description': 'Template configuration with equity protection',
        
        'equity_protection': {
            'enabled': True,
            'drawdown_threshold': 0.25,
            'enable_on_bias_flip': True,
            'smoothing_window': 5,
            'min_equity_history': 10,
            'flatten_positions_on_trigger': True,
            'disable_new_entries': True,
            'recovery_mechanism': 'bias_flip',
            'emergency_controls': {
                'force_reset_after_hours': 168,
                'max_consecutive_triggers': 3,
                'alert_on_trigger': True
            }
        },
        
        'bias_detection': {
            'method': 'price_ma',
            'ma_period': 50,
            'trend_threshold': 0.02,
            'min_bars_for_flip': 5
        },
        
        'risk_management': {
            'risk_pct': 0.015,
            'max_position_pct': 0.25,
            'protection_mode': {
                'max_position_pct': 0.1,
                'risk_pct': 0.005,
                'require_stronger_signals': True
            }
        },
        
        'alerts': {
            'equity_protection_triggered': {
                'enabled': True,
                'message': 'EQUITY PROTECTION TRIGGERED',
                'severity': 'critical'
            },
            'drawdown_warning': {
                'enabled': True,
                'threshold': 0.15,
                'message': 'High drawdown detected',
                'severity': 'warning'
            }
        }
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(template_config, f, default_flow_style=False, indent=2)
    
    print(f"Equity protection template saved to: {file_path}")


if __name__ == "__main__":
    # Example usage
    print("Equity Protection Integration Example")
    print("=====================================")
    
    # Create sample strategy config
    sample_config = {
        'name': 'Sample Strategy',
        'logic': {
            'entry_long': ['rsi < 30'],
            'exit_long': ['rsi > 70']
        },
        'risk_management': {
            'risk_pct': 0.02
        }
    }
    
    # Add equity protection
    protected_config = add_equity_protection_to_config(sample_config)
    
    print("Original config:")
    print(yaml.dump(sample_config, default_flow_style=False))
    
    print("\nProtected config:")
    print(yaml.dump(protected_config, default_flow_style=False))
    
    # Create protection monitor example
    equity_protection = EquityProtection(drawdown_threshold=0.25, debug=True)
    monitor = EquityProtectionMonitor(equity_protection)
    
    # Simulate some equity updates
    test_equity_values = [10000, 9500, 9000, 8500, 7000, 8000, 8500]
    
    for equity in test_equity_values:
        result = monitor.update_and_monitor(equity)
        print(f"Equity: ${equity}, Actions: {result['actions_taken']}")
    
    # Generate monitoring report
    report = monitor.get_monitoring_report()
    print(f"\n{report}")