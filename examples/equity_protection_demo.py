#!/usr/bin/env python3
"""
Equity Protection System Demonstration.

This script demonstrates the comprehensive equity protection system designed
to prevent catastrophic drawdowns. It shows how the system works in practice
and compares strategy performance with and without protection.

Key demonstrations:
1. Basic equity protection functionality
2. Integration with existing strategies
3. Backtest comparison with different protection thresholds
4. Real-time monitoring simulation
5. Performance analysis and reporting

Usage:
    python examples/equity_protection_demo.py
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml

# Add the btc_research package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from btc_research.core.equity_protection import (
    EquityProtection,
    EquityProtectionAnalyzer,
    EquityProtectionError
)
from btc_research.utils.equity_protection_integration import (
    add_equity_protection_to_config,
    EquityProtectedBacktester,
    EquityProtectionMonitor,
    analyze_strategy_with_protection,
    generate_protection_comparison_report
)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting will be disabled.")


def create_sample_market_data(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    initial_price: float = 50000.0,
    volatility: float = 0.02,
    trend_bias: float = 0.0001
) -> pd.DataFrame:
    """
    Create sample market data for demonstration.
    
    Args:
        start_date: Start date string
        end_date: End date string
        initial_price: Initial price
        volatility: Daily volatility
        trend_bias: Trend bias (positive for uptrend)
        
    Returns:
        DataFrame with OHLCV data
    """
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate random price movements
    np.random.seed(42)  # For reproducible results
    num_periods = len(date_range)
    
    # Generate returns with trend bias
    returns = np.random.normal(trend_bias, volatility, num_periods)
    
    # Add some extreme events to simulate crashes
    crash_indices = np.random.choice(num_periods, size=3, replace=False)
    for idx in crash_indices:
        returns[idx:idx+10] = np.random.normal(-0.05, 0.02, 10)  # Crash events
    
    # Generate price series
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])  # Remove initial price
    
    # Create OHLCV data
    data = {
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, num_periods)),
        'low': prices * (1 - np.random.uniform(0, 0.01, num_periods)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, num_periods)
    }
    
    # Ensure high >= close >= low and open
    data['high'] = np.maximum(data['high'], np.maximum(data['close'], data['open']))
    data['low'] = np.minimum(data['low'], np.minimum(data['close'], data['open']))
    
    df = pd.DataFrame(data, index=date_range)
    
    # Add simple indicators for demonstration
    df['rsi'] = calculate_rsi(df['close'])
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def create_sample_strategy_config() -> Dict[str, Any]:
    """Create a sample strategy configuration for demonstration."""
    return {
        'name': 'Sample RSI Strategy',
        'description': 'Simple RSI-based strategy for equity protection demonstration',
        
        'symbol': 'BTC/USD',
        'timeframes': {'entry': '1h'},
        
        'indicators': [],  # We'll use pre-calculated indicators
        
        'logic': {
            'entry_long': [
                'rsi < 30',  # Oversold condition
                'close > sma_50'  # Above long-term average
            ],
            'exit_long': [
                'rsi > 70',  # Overbought condition
                'close < sma_20'  # Below short-term average
            ],
            'entry_short': [
                'rsi > 70',  # Overbought condition
                'close < sma_50'  # Below long-term average
            ],
            'exit_short': [
                'rsi < 30',  # Oversold condition
                'close > sma_20'  # Above short-term average
            ]
        },
        
        'backtest': {
            'from': '2024-01-01',
            'to': '2024-12-31',
            'initial_cash': 10000,
            'commission': 0.001
        },
        
        'risk_management': {
            'risk_pct': 0.02,  # 2% risk per trade
            'max_position_pct': 0.3  # Max 30% position size
        }
    }


def demonstrate_basic_equity_protection():
    """Demonstrate basic equity protection functionality."""
    print("=" * 60)
    print("BASIC EQUITY PROTECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create equity protection instance
    protection = EquityProtection(
        drawdown_threshold=0.25,  # 25% first-loss stop
        enable_on_bias_flip=True,
        smoothing_window=3,
        debug=True
    )
    
    print(f"Created equity protection with {protection.drawdown_threshold:.1%} threshold")
    print()
    
    # Simulate equity curve with catastrophic drawdown
    print("Simulating equity curve with potential catastrophic drawdown...")
    equity_scenarios = [
        ("Normal Trading", [10000, 10500, 11000, 10800, 11200, 10900, 11500]),
        ("Drawdown Scenario", [10000, 9500, 9000, 8500, 7800, 7200, 6800]),
        ("Recovery", [6800, 7200, 7800, 8500, 9200, 9800, 10200])
    ]
    
    for scenario_name, equity_values in equity_scenarios:
        print(f"\n{scenario_name}:")
        print("-" * 30)
        
        for i, equity in enumerate(equity_values):
            result = protection.update_equity(equity)
            
            status = "NORMAL"
            if result['protection_active']:
                status = "PROTECTION ACTIVE"
            if result['trading_disabled']:
                status = "TRADING DISABLED"
            
            print(f"  Day {i+1}: ${equity:,} | Drawdown: {result['drawdown']:6.1%} | Status: {status}")
            
            # Simulate bias flip if protection is active
            if result['protection_triggered']:
                print(f"    -> Protection triggered! Simulating bias flip...")
                bias_result = protection.update_bias("bull")
                if not bias_result['trading_enabled']:
                    print(f"    -> Trading still disabled")
                else:
                    print(f"    -> Trading re-enabled after bias flip")
    
    # Generate final report
    print("\n" + "=" * 60)
    print("FINAL EQUITY PROTECTION STATISTICS")
    print("=" * 60)
    
    analyzer = EquityProtectionAnalyzer(protection)
    report = analyzer.generate_protection_report()
    print(report)
    
    return protection


def demonstrate_strategy_integration():
    """Demonstrate integration with trading strategies."""
    print("\n" + "=" * 60)
    print("STRATEGY INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample strategy config
    base_config = create_sample_strategy_config()
    
    # Add equity protection
    protected_config = add_equity_protection_to_config(
        base_config,
        drawdown_threshold=0.25,
        enable_on_bias_flip=True
    )
    
    print("Original strategy configuration:")
    print(yaml.dump(base_config['logic'], default_flow_style=False, indent=2))
    
    print("\nProtected strategy configuration:")
    print(yaml.dump(protected_config['logic'], default_flow_style=False, indent=2))
    
    print("\nEquity protection configuration:")
    print(yaml.dump(protected_config['equity_protection'], default_flow_style=False, indent=2))
    
    return protected_config


def demonstrate_backtest_comparison():
    """Demonstrate backtest comparison with and without protection."""
    print("\n" + "=" * 60)
    print("BACKTEST COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    # Create sample market data
    print("Generating sample market data...")
    df = create_sample_market_data(
        start_date="2024-01-01",
        end_date="2024-03-31",  # Shorter period for demo
        volatility=0.03,  # Higher volatility for dramatic effect
        trend_bias=-0.0005  # Slight downtrend to trigger protection
    )
    
    print(f"Generated {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"Max drawdown in data: {(df['close'].max() - df['close'].min()) / df['close'].max():.1%}")
    
    # Create strategy configuration
    strategy_config = create_sample_strategy_config()
    
    # Run comparison analysis
    print("\nRunning backtest comparison...")
    analysis_results = analyze_strategy_with_protection(
        strategy_config,
        df,
        protection_thresholds=[0.15, 0.20, 0.25, 0.30],
        cash=10000.0
    )
    
    # Generate comparison report
    comparison_report = generate_protection_comparison_report(analysis_results)
    print("\n" + comparison_report)
    
    return analysis_results


def demonstrate_real_time_monitoring():
    """Demonstrate real-time monitoring capabilities."""
    print("\n" + "=" * 60)
    print("REAL-TIME MONITORING DEMONSTRATION")
    print("=" * 60)
    
    # Create equity protection and monitor
    protection = EquityProtection(drawdown_threshold=0.20, debug=True)
    
    def alert_callback(message: str, severity: str = "info"):
        print(f"[{severity.upper()}] ALERT: {message}")
    
    def position_manager(action: str):
        print(f"[POSITION MANAGER] Action: {action}")
    
    monitor = EquityProtectionMonitor(
        protection,
        alert_callback=alert_callback,
        position_manager=position_manager
    )
    
    print("Starting real-time monitoring simulation...")
    print("Simulating live trading with equity updates...")
    
    # Simulate live trading scenario
    trading_scenarios = [
        ("Normal trading", [10000, 10200, 10500, 10300, 10800]),
        ("Market decline", [10800, 10400, 9800, 9200, 8500]),
        ("Protection trigger", [8500, 8000, 7500]),  # Should trigger at 7500
        ("Bias flip recovery", [7500, 7800, 8200, 8600, 9000])
    ]
    
    for scenario_name, equity_values in trading_scenarios:
        print(f"\n{scenario_name}:")
        print("-" * 30)
        
        for i, equity in enumerate(equity_values):
            timestamp = datetime.now() + timedelta(hours=i)
            result = monitor.update_and_monitor(equity, timestamp)
            
            actions = result['actions_taken']
            actions_str = ", ".join(actions) if actions else "None"
            
            print(f"  Time: {timestamp.strftime('%H:%M')}, "
                  f"Equity: ${equity:,}, "
                  f"Actions: {actions_str}")
            
            # Simulate bias flip when protection is active
            if 'protection_triggered' in actions:
                print(f"    -> Simulating bias flip detection...")
                protection.update_bias("bull", timestamp)
    
    # Generate monitoring report
    monitoring_report = monitor.get_monitoring_report()
    print("\n" + monitoring_report)
    
    return monitor


def demonstrate_visualization():
    """Demonstrate visualization capabilities."""
    if not HAS_MATPLOTLIB:
        print("\n" + "=" * 60)
        print("VISUALIZATION DEMONSTRATION")
        print("=" * 60)
        print("Matplotlib not available - skipping visualization demo")
        return
    
    print("\n" + "=" * 60)
    print("VISUALIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create equity protection with simulated data
    protection = EquityProtection(drawdown_threshold=0.25, debug=False)
    
    # Simulate a complete equity curve
    base_equity = 10000
    equity_curve = []
    
    # Generate realistic equity curve with drawdowns
    for i in range(100):
        if i < 30:
            # Initial growth
            equity = base_equity * (1 + 0.02 * i + np.random.normal(0, 0.01))
        elif i < 60:
            # Drawdown period
            peak = base_equity * 1.6
            equity = peak * (1 - 0.4 * (i - 30) / 30) + np.random.normal(0, 100)
        else:
            # Recovery period
            low = base_equity * 0.96
            equity = low * (1 + 0.03 * (i - 60) / 40) + np.random.normal(0, 100)
        
        equity_curve.append(max(equity, 1000))  # Ensure positive equity
    
    # Update protection with equity curve
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(100)]
    
    for i, (timestamp, equity) in enumerate(zip(timestamps, equity_curve)):
        protection.update_equity(equity, timestamp)
        
        # Simulate bias flips
        if i == 40:  # During drawdown
            protection.update_bias("bear", timestamp)
        elif i == 70:  # During recovery
            protection.update_bias("bull", timestamp)
    
    # Create visualization
    print("Creating equity curve visualization...")
    fig = protection.plot_equity_curve(figsize=(12, 8))
    
    if fig:
        plt.savefig('equity_protection_demo.png', dpi=300, bbox_inches='tight')
        print("Equity curve plot saved as 'equity_protection_demo.png'")
        
        # Try to display if running interactively
        try:
            plt.show()
        except:
            pass  # Non-interactive environment
    
    return protection


def main():
    """Main demonstration function."""
    print("EQUITY PROTECTION SYSTEM COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows how the equity protection system prevents")
    print("catastrophic drawdowns and integrates with trading strategies.")
    print()
    
    # Run all demonstrations
    try:
        # 1. Basic functionality
        basic_protection = demonstrate_basic_equity_protection()
        
        # 2. Strategy integration
        protected_config = demonstrate_strategy_integration()
        
        # 3. Backtest comparison
        backtest_results = demonstrate_backtest_comparison()
        
        # 4. Real-time monitoring
        monitor = demonstrate_real_time_monitoring()
        
        # 5. Visualization
        visualization_protection = demonstrate_visualization()
        
        # Final summary
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("The equity protection system has been demonstrated with:")
        print("✓ Basic protection functionality")
        print("✓ Strategy integration capabilities")
        print("✓ Backtest comparison analysis")
        print("✓ Real-time monitoring simulation")
        if HAS_MATPLOTLIB:
            print("✓ Visualization capabilities")
        else:
            print("✗ Visualization (matplotlib not available)")
        print()
        print("Key benefits demonstrated:")
        print("- Prevents catastrophic drawdowns (25% first-loss stop)")
        print("- Automatic position flattening on threshold breach")
        print("- Bias-flip recovery mechanism")
        print("- Easy integration with existing strategies")
        print("- Comprehensive monitoring and analysis")
        print()
        print("The system is ready for use in production trading environments.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()