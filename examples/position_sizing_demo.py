#!/usr/bin/env python3
"""
Position Sizing Demo for BTC Research Engine.

This script demonstrates how to use the new risk-per-trade position sizing
module with the existing VP-FVG strategy system. It shows the difference
between the old 95% capital allocation approach and the new risk-based approach.

Key Features Demonstrated:
1. Basic position sizing calculations
2. ATR-based stop loss integration
3. Risk management with the RiskManagement indicator
4. Backtest comparison between old and new approaches
5. Position sizing validation and error handling

Run with:
    python examples/position_sizing_demo.py
"""

import sys
import warnings
from pathlib import Path

# Add the btc_research package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from btc_research.core.datafeed import DataFeed
from btc_research.core.engine import Engine
from btc_research.core.backtester import Backtester, create_backtest_summary
from btc_research.core.position_sizing import (
    PositionSizer, 
    BacktraderPositionSizer, 
    calculate_atr_stop_loss,
    validate_position_sizing_config
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demo_basic_position_sizing():
    """Demonstrate basic position sizing calculations."""
    print("=== Basic Position Sizing Demo ===\n")
    
    # Create position sizer with 1% risk per trade
    sizer = PositionSizer(default_risk_pct=0.01, max_position_pct=0.2)
    
    # Example scenarios
    scenarios = [
        {
            "name": "Conservative BTC Long",
            "equity": 10000,
            "entry_price": 50000,
            "stop_price": 49000,
            "is_long": True
        },
        {
            "name": "Aggressive BTC Long",
            "equity": 10000,
            "entry_price": 50000,
            "stop_price": 48000,
            "is_long": True
        },
        {
            "name": "BTC Short",
            "equity": 10000,
            "entry_price": 50000,
            "stop_price": 51000,
            "is_long": False
        },
        {
            "name": "Small Account",
            "equity": 1000,
            "entry_price": 50000,
            "stop_price": 49000,
            "is_long": True
        }
    ]
    
    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print(f"  Equity: ${scenario['equity']:,}")
        print(f"  Entry: ${scenario['entry_price']:,}")
        print(f"  Stop: ${scenario['stop_price']:,}")
        print(f"  Direction: {'Long' if scenario['is_long'] else 'Short'}")
        
        try:
            # Calculate position size
            position_size = sizer.calculate_position_size(
                equity=scenario['equity'],
                entry_price=scenario['entry_price'],
                stop_price=scenario['stop_price'],
                is_long=scenario['is_long']
            )
            
            # Calculate metrics
            metrics = sizer.calculate_position_metrics(
                equity=scenario['equity'],
                entry_price=scenario['entry_price'],
                stop_price=scenario['stop_price'],
                position_size=position_size
            )
            
            print(f"  Position Size: {position_size:.6f} BTC")
            print(f"  Position Value: ${metrics['position_value']:,.2f}")
            print(f"  Position %: {metrics['position_pct']:.1%}")
            print(f"  Risk Amount: ${metrics['risk_amount']:.2f}")
            print(f"  Risk %: {metrics['risk_pct']:.2%}")
            print(f"  Stop Distance: {metrics['stop_distance_pct']:.2%}")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        print()


def demo_atr_stop_loss():
    """Demonstrate ATR-based stop loss calculation."""
    print("=== ATR-Based Stop Loss Demo ===\n")
    
    # Create sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    
    # Generate realistic BTC price data
    base_price = 50000
    price_data = []
    current_price = base_price
    
    for i in range(100):
        # Random walk with volatility
        change = np.random.normal(0, 0.02)  # 2% volatility
        current_price *= (1 + change)
        
        # Generate OHLC
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = current_price * (1 + np.random.normal(0, 0.005))
        close_price = current_price
        volume = np.random.uniform(100, 1000)
        
        price_data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(price_data, index=dates)
    
    # Calculate ATR-based stop losses
    print("ATR Stop Loss Examples:")
    
    atr_configs = [
        {"period": 14, "multiplier": 1.5, "name": "Conservative (1.5x ATR)"},
        {"period": 14, "multiplier": 2.0, "name": "Standard (2.0x ATR)"},
        {"period": 14, "multiplier": 2.5, "name": "Wide (2.5x ATR)"}
    ]
    
    current_price = df['close'].iloc[-1]
    print(f"Current Price: ${current_price:,.2f}")
    
    for config in atr_configs:
        try:
            # Long stop
            long_stop = calculate_atr_stop_loss(
                df, 
                period=config['period'],
                multiplier=config['multiplier'],
                is_long=True
            )
            
            # Short stop
            short_stop = calculate_atr_stop_loss(
                df,
                period=config['period'],
                multiplier=config['multiplier'],
                is_long=False
            )
            
            print(f"\n{config['name']}:")
            print(f"  Long Stop: ${long_stop:,.2f} ({(current_price - long_stop) / current_price:.2%} below)")
            print(f"  Short Stop: ${short_stop:,.2f} ({(short_stop - current_price) / current_price:.2%} above)")
            
        except Exception as e:
            print(f"  Error calculating {config['name']}: {e}")


def demo_risk_management_integration():
    """Demonstrate integration with existing risk management indicator."""
    print("=== Risk Management Integration Demo ===\n")
    
    # Create a simple configuration for testing
    config = {
        "data": {
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "start": "2024-01-01",
            "end": "2024-01-02",
            "source": "binanceus"
        },
        "indicators": {
            "RiskManagement": {
                "atr_period": 14,
                "initial_stop_atr_mult": 0.5,
                "target_atr_mult": 2.0,
                "enable_trailing": True
            }
        },
        "logic": {
            "entry_long": ["close > open"],  # Simple long condition
            "exit_long": ["exit_long_risk == True"],  # Exit on risk management signal
            "entry_short": ["close < open"],  # Simple short condition
            "exit_short": ["exit_short_risk == True"]  # Exit on risk management signal
        }
    }
    
    try:
        # Create datafeed and engine
        with DataFeed() as feed:
            # Get sample data
            df = feed.get(
                symbol="BTC/USD",
                timeframe="1h", 
                start="2024-01-01",
                end="2024-01-02",
                source="binanceus"
            )
            
            if len(df) == 0:
                print("No data available for demo (offline mode)")
                return
            
            # Run indicators
            engine = Engine(config)
            result_df = engine.run(df)
            
            print(f"Generated {len(result_df)} data points with indicators")
            
            # Show available risk management columns
            risk_columns = [col for col in result_df.columns if 'risk' in col.lower() or 'stop' in col.lower()]
            print(f"Risk Management Columns: {risk_columns}")
            
            # Demo position sizing with risk management data
            if len(result_df) > 20:  # Ensure we have enough data
                sizer = PositionSizer(default_risk_pct=0.01)
                
                # Get a sample row with risk management data
                sample_row = result_df.iloc[-1]
                current_price = sample_row['close']
                
                # Try to get stop loss from risk management
                long_stop = sample_row.get('long_stop_loss', None)
                short_stop = sample_row.get('short_stop_loss', None)
                
                print(f"\nSample Position Sizing:")
                print(f"Current Price: ${current_price:,.2f}")
                
                if long_stop and not pd.isna(long_stop):
                    print(f"Long Stop: ${long_stop:,.2f}")
                    try:
                        equity = 10000
                        position_size = sizer.calculate_position_size(
                            equity=equity,
                            entry_price=current_price,
                            stop_price=long_stop,
                            is_long=True
                        )
                        risk_amount = (current_price - long_stop) * position_size
                        print(f"Long Position Size: {position_size:.6f} BTC")
                        print(f"Risk Amount: ${risk_amount:.2f} ({risk_amount/equity:.2%})")
                    except Exception as e:
                        print(f"Long position sizing error: {e}")
                
                if short_stop and not pd.isna(short_stop):
                    print(f"Short Stop: ${short_stop:,.2f}")
                    try:
                        equity = 10000
                        position_size = sizer.calculate_position_size(
                            equity=equity,
                            entry_price=current_price,
                            stop_price=short_stop,
                            is_long=False
                        )
                        risk_amount = (short_stop - current_price) * position_size
                        print(f"Short Position Size: {position_size:.6f} BTC")
                        print(f"Risk Amount: ${risk_amount:.2f} ({risk_amount/equity:.2%})")
                    except Exception as e:
                        print(f"Short position sizing error: {e}")
            
    except Exception as e:
        print(f"Integration demo error: {e}")


def demo_backtest_comparison():
    """Compare old vs new position sizing approaches."""
    print("=== Backtest Comparison Demo ===\n")
    
    # Create test configuration
    config = {
        "data": {
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "start": "2024-01-01",
            "end": "2024-01-15",
            "source": "binanceus"
        },
        "indicators": {
            "RiskManagement": {
                "atr_period": 14,
                "initial_stop_atr_mult": 1.0,
                "target_atr_mult": 2.0
            }
        },
        "logic": {
            "entry_long": ["close > open"],  # Simple trend following
            "exit_long": ["exit_long_risk == True"],
            "entry_short": [],  # No short positions for this demo
            "exit_short": []
        }
    }
    
    try:
        # Get data
        with DataFeed() as feed:
            df = feed.get(
                symbol="BTC/USD",
                timeframe="1h",
                start="2024-01-01", 
                end="2024-01-15",
                source="binanceus"
            )
            
            if len(df) == 0:
                print("No data available for backtest demo (offline mode)")
                return
            
            # Run indicators
            engine = Engine(config)
            result_df = engine.run(df)
            
            print(f"Running backtests on {len(result_df)} data points...\n")
            
            # Test old approach (without position sizing)
            print("=== Old Approach (Fixed Size) ===")
            old_backtester = Backtester(config, debug=False, use_position_sizing=False)
            old_stats = old_backtester.run(result_df, cash=10000, commission=0.001)
            print(create_backtest_summary(old_stats))
            
            # Test new approach (with position sizing)
            print("\n=== New Approach (Risk-Based Sizing) ===")
            new_backtester = Backtester(config, debug=False, use_position_sizing=True, risk_pct=0.01)
            new_stats = new_backtester.run(result_df, cash=10000, commission=0.001)
            print(create_backtest_summary(new_stats))
            
            # Compare results
            print("\n=== Comparison ===")
            print(f"Return Difference: {new_stats['total_return'] - old_stats['total_return']:.2%}")
            print(f"Sharpe Improvement: {new_stats['sharpe_ratio'] - old_stats['sharpe_ratio']:.2f}")
            print(f"Drawdown Change: {new_stats['max_drawdown'] - old_stats['max_drawdown']:.2%}")
            
            # Risk analysis
            print(f"\nRisk Analysis:")
            print(f"Old Max Drawdown: {old_stats['max_drawdown']:.2%}")
            print(f"New Max Drawdown: {new_stats['max_drawdown']:.2%}")
            
            if new_stats['max_drawdown'] < old_stats['max_drawdown']:
                print("✓ New approach shows better risk control")
            else:
                print("⚠ New approach shows higher risk")
                
    except Exception as e:
        print(f"Backtest comparison error: {e}")


def demo_config_validation():
    """Demonstrate position sizing configuration validation."""
    print("=== Configuration Validation Demo ===\n")
    
    test_configs = [
        {
            "name": "Valid Configuration",
            "config": {
                "risk_pct": 0.01,
                "max_position_pct": 0.2,
                "atr_period": 14,
                "atr_multiplier": 2.0
            },
            "should_pass": True
        },
        {
            "name": "Invalid Risk Percentage",
            "config": {
                "risk_pct": 1.5,  # > 100%
                "max_position_pct": 0.2
            },
            "should_pass": False
        },
        {
            "name": "Invalid ATR Period",
            "config": {
                "risk_pct": 0.01,
                "atr_period": 0  # Must be > 0
            },
            "should_pass": False
        },
        {
            "name": "Missing Parameters (Uses Defaults)",
            "config": {},
            "should_pass": True
        }
    ]
    
    for test in test_configs:
        print(f"Testing: {test['name']}")
        try:
            validated_config = validate_position_sizing_config(test['config'])
            print(f"  ✓ Validation passed")
            print(f"  Risk %: {validated_config['risk_pct']:.2%}")
            print(f"  Max Position %: {validated_config['max_position_pct']:.1%}")
            print(f"  ATR Period: {validated_config['atr_period']}")
            
            if not test['should_pass']:
                print(f"  ⚠ Expected failure but passed")
                
        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
            
            if test['should_pass']:
                print(f"  ⚠ Expected success but failed")
        
        print()


def main():
    """Run all position sizing demos."""
    print("BTC Research Engine - Position Sizing Demo")
    print("=" * 50)
    print()
    
    try:
        # Run all demos
        demo_basic_position_sizing()
        print("\n" + "=" * 50 + "\n")
        
        demo_atr_stop_loss()
        print("\n" + "=" * 50 + "\n")
        
        demo_config_validation()
        print("\n" + "=" * 50 + "\n")
        
        demo_risk_management_integration()
        print("\n" + "=" * 50 + "\n")
        
        demo_backtest_comparison()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()