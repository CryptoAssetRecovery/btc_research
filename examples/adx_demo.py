#!/usr/bin/env python3
"""
ADX (Average Directional Index) Indicator Demo

This script demonstrates the ADX trend filter indicator with different
market conditions:
1. Strong trending market (ADX > 25)
2. Ranging market (ADX < 20)
3. Transitioning market (ADX between 20-25)

The ADX is used for regime-aware trading to filter setups based on market conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from btc_research.indicators.adx import ADX


def create_trending_data(periods=200, base_price=50000, trend_strength=10000):
    """Create synthetic trending market data."""
    np.random.seed(42)
    
    # Strong upward trend
    trend = np.linspace(0, trend_strength, periods)
    noise = np.random.normal(0, 100, periods)
    close_prices = base_price + trend + noise
    
    # Create OHLC data
    high_prices = close_prices + np.random.uniform(10, 100, periods)
    low_prices = close_prices - np.random.uniform(10, 100, periods)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Ensure OHLC relationships
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    volumes = np.random.uniform(1000, 5000, periods)
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='15min')
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


def create_ranging_data(periods=200, base_price=50000, range_size=1000):
    """Create synthetic ranging market data."""
    np.random.seed(123)
    
    # Oscillating around base price
    t = np.linspace(0, 4 * np.pi, periods)
    oscillation = np.sin(t) * (range_size / 2)
    noise = np.random.normal(0, 50, periods)
    close_prices = base_price + oscillation + noise
    
    # Create OHLC data
    high_prices = close_prices + np.random.uniform(10, 50, periods)
    low_prices = close_prices - np.random.uniform(10, 50, periods)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Ensure OHLC relationships
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    volumes = np.random.uniform(1000, 5000, periods)
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='15min')
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


def create_transitioning_data(periods=200, base_price=50000):
    """Create synthetic transitioning market data (trending then ranging)."""
    np.random.seed(456)
    
    # First half: trending up
    trend_periods = periods // 2
    trend = np.linspace(0, 3000, trend_periods)
    trend_noise = np.random.normal(0, 80, trend_periods)
    trend_prices = base_price + trend + trend_noise
    
    # Second half: ranging
    range_periods = periods - trend_periods
    t = np.linspace(0, 3 * np.pi, range_periods)
    oscillation = np.sin(t) * 300
    range_noise = np.random.normal(0, 60, range_periods)
    range_prices = trend_prices[-1] + oscillation + range_noise
    
    close_prices = np.concatenate([trend_prices, range_prices])
    
    # Create OHLC data
    high_prices = close_prices + np.random.uniform(10, 80, periods)
    low_prices = close_prices - np.random.uniform(10, 80, periods)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Ensure OHLC relationships
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    volumes = np.random.uniform(1000, 5000, periods)
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='15min')
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


def analyze_market_regime(data, title):
    """Analyze market regime using ADX."""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Calculate ADX
    adx = ADX(period=14, trend_threshold=25.0, range_threshold=20.0)
    result = adx.compute(data)
    
    # Get valid ADX values
    valid_adx = result['ADX_value'].dropna()
    
    if len(valid_adx) == 0:
        print("Insufficient data for ADX calculation")
        return None
    
    # Statistics
    print(f"ADX Statistics:")
    print(f"  Mean ADX: {valid_adx.mean():.2f}")
    print(f"  Min ADX: {valid_adx.min():.2f}")
    print(f"  Max ADX: {valid_adx.max():.2f}")
    print(f"  Std ADX: {valid_adx.std():.2f}")
    
    # Regime analysis
    trend_periods = result['ADX_trend'].sum()
    range_periods = result['ADX_range'].sum()
    neutral_periods = len(result) - trend_periods - range_periods
    
    print(f"\nRegime Analysis:")
    print(f"  Trending periods (ADX > 25): {trend_periods} ({trend_periods/len(result)*100:.1f}%)")
    print(f"  Ranging periods (ADX < 20): {range_periods} ({range_periods/len(result)*100:.1f}%)")
    print(f"  Neutral periods (20 ≤ ADX ≤ 25): {neutral_periods} ({neutral_periods/len(result)*100:.1f}%)")
    
    # Directional bias
    valid_mask = result['DI_plus'].notna() & result['DI_minus'].notna()
    if valid_mask.any():
        bullish_periods = result['DI_bullish'][valid_mask].sum()
        bearish_periods = result['DI_bearish'][valid_mask].sum()
        
        print(f"\nDirectional Bias:")
        print(f"  Bullish periods (+DI > -DI): {bullish_periods} ({bullish_periods/valid_mask.sum()*100:.1f}%)")
        print(f"  Bearish periods (-DI > +DI): {bearish_periods} ({bearish_periods/valid_mask.sum()*100:.1f}%)")
    
    # Strength distribution
    strength_counts = result['ADX_strength'].value_counts()
    print(f"\nADX Strength Distribution:")
    for strength, count in strength_counts.items():
        print(f"  {strength}: {count} ({count/len(result)*100:.1f}%)")
    
    return result


def plot_adx_analysis(data, result, title):
    """Plot ADX analysis charts."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Price with trend/range periods
    axes[0].plot(data.index, data['close'], 'b-', alpha=0.7, label='Close Price')
    
    # Highlight trending periods
    trend_mask = result['ADX_trend']
    if trend_mask.any():
        axes[0].fill_between(data.index, data['close'].min(), data['close'].max(), 
                           where=trend_mask, alpha=0.2, color='green', label='Trending (ADX > 25)')
    
    # Highlight ranging periods
    range_mask = result['ADX_range']
    if range_mask.any():
        axes[0].fill_between(data.index, data['close'].min(), data['close'].max(), 
                           where=range_mask, alpha=0.2, color='red', label='Ranging (ADX < 20)')
    
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: ADX with thresholds
    valid_adx = result['ADX_value'].dropna()
    if len(valid_adx) > 0:
        axes[1].plot(result.index, result['ADX_value'], 'purple', linewidth=2, label='ADX')
        axes[1].axhline(y=25, color='green', linestyle='--', alpha=0.7, label='Trend Threshold (25)')
        axes[1].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Range Threshold (20)')
        axes[1].fill_between(result.index, 20, 25, alpha=0.1, color='yellow', label='Neutral Zone')
    
    axes[1].set_ylabel('ADX Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 100)
    
    # Plot 3: Directional Indicators
    axes[2].plot(result.index, result['DI_plus'], 'g-', label='+DI', alpha=0.8)
    axes[2].plot(result.index, result['DI_minus'], 'r-', label='-DI', alpha=0.8)
    
    axes[2].set_ylabel('DI Value')
    axes[2].set_xlabel('Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main demonstration function."""
    print("ADX (Average Directional Index) Trend Filter Demo")
    print("=" * 50)
    
    # Create different market conditions
    print("\nGenerating synthetic market data...")
    
    trending_data = create_trending_data(periods=150)
    ranging_data = create_ranging_data(periods=150)
    transitioning_data = create_transitioning_data(periods=150)
    
    # Analyze each market condition
    trending_result = analyze_market_regime(trending_data, "TRENDING MARKET ANALYSIS")
    ranging_result = analyze_market_regime(ranging_data, "RANGING MARKET ANALYSIS")
    transitioning_result = analyze_market_regime(transitioning_data, "TRANSITIONING MARKET ANALYSIS")
    
    # Trading implications
    print("\n" + "=" * 50)
    print("TRADING IMPLICATIONS")
    print("=" * 50)
    print("""
ADX Trend Filter Usage in Regime-Aware Trading:

1. TRENDING MARKETS (ADX > 25):
   - Favor continuation setups
   - Use trend-following strategies
   - Look for pullbacks to enter in trend direction
   - Avoid counter-trend trades

2. RANGING MARKETS (ADX < 20):
   - Favor reversal setups
   - Use mean-reversion strategies
   - Trade between support/resistance levels
   - Avoid breakout strategies

3. NEUTRAL MARKETS (20 ≤ ADX ≤ 25):
   - Exercise caution
   - Wait for clear regime identification
   - Use smaller position sizes
   - Combine with other confluence factors

4. DIRECTIONAL BIAS:
   - +DI > -DI: Bullish directional bias
   - -DI > +DI: Bearish directional bias
   - Use DI crossovers for entry timing
""")
    
    # Create visualizations
    print("\nGenerating visualization plots...")
    
    if trending_result is not None:
        fig1 = plot_adx_analysis(trending_data, trending_result, "Trending Market - ADX Analysis")
        plt.show()
    
    if ranging_result is not None:
        fig2 = plot_adx_analysis(ranging_data, ranging_result, "Ranging Market - ADX Analysis")
        plt.show()
    
    if transitioning_result is not None:
        fig3 = plot_adx_analysis(transitioning_data, transitioning_result, "Transitioning Market - ADX Analysis")
        plt.show()
    
    print("\nDemo completed! The ADX indicator is now ready for use in regime-aware trading strategies.")


if __name__ == "__main__":
    main()