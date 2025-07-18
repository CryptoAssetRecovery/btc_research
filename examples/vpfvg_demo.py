#!/usr/bin/env python3
"""
VPFVGSignal Confluence Indicator Demo

This script demonstrates how to use the VPFVGSignal indicator to identify
high-probability trading opportunities by combining Volume Profile and Fair Value Gap analysis.

The demo shows:
1. How to compute the required base indicators (Volume Profile and FVG)
2. How to use the VPFVGSignal indicator
3. How to interpret the signals and diagnostic data
4. Example of signal analysis and filtering

Usage:
    python examples/vpfvg_demo.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# Import the required indicators
from btc_research.indicators.volume_profile import VolumeProfile
from btc_research.indicators.fvg import FVG
from btc_research.indicators.vpfvg_signal import VPFVGSignal


def generate_sample_data(n_bars: int = 1000) -> pd.DataFrame:
    """
    Generate sample OHLCV data for demonstration purposes.
    
    Args:
        n_bars: Number of bars to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate dates
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='15min')
    
    # Generate price data with trend and volatility
    trend = np.linspace(0, 2000, n_bars)  # Upward trend
    volatility = 100 + 50 * np.sin(np.linspace(0, 10, n_bars))  # Varying volatility
    
    # Random walk with trend
    returns = np.random.normal(0, 1, n_bars)
    price_changes = returns * volatility * 0.01
    
    # Base price series
    base_price = 30000 + trend + np.cumsum(price_changes)
    
    # Generate OHLC from base price
    data = pd.DataFrame({
        'open': base_price * (1 + np.random.normal(0, 0.001, n_bars)),
        'high': base_price * (1 + np.abs(np.random.normal(0, 0.003, n_bars))),
        'low': base_price * (1 - np.abs(np.random.normal(0, 0.003, n_bars))),
        'close': base_price,
        'volume': np.random.uniform(100, 1000, n_bars),
    }, index=dates)
    
    # Ensure price relationships are valid
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    data['open'] = np.clip(data['open'], data['low'], data['high'])
    
    return data


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all required indicators for the VPFVGSignal.
    
    Args:
        data: OHLCV DataFrame
        
    Returns:
        DataFrame with all indicator data
    """
    print("Computing Volume Profile...")
    vp_indicator = VolumeProfile(
        lookback=200,           # 200 bars lookback
        price_bins=30,          # 30 price bins
        value_area_pct=70,      # 70% value area
        hvn_threshold=1.5,      # High Volume Node threshold
        lvn_threshold=0.5,      # Low Volume Node threshold
        atr_period=14,          # ATR period
        enable_vectorization=True
    )
    vp_result = vp_indicator.compute(data)
    
    print("Computing Fair Value Gaps...")
    fvg_indicator = FVG(
        min_gap_pips=5.0,       # Minimum gap size
        max_lookback=500        # Max gaps to track
    )
    fvg_result = fvg_indicator.compute(data)
    
    print("Computing VPFVGSignal...")
    # Combine the data
    combined_data = data.copy()
    
    # Add VP columns with proper prefixes
    for col in vp_result.columns:
        combined_data[f'VolumeProfile_{col}'] = vp_result[col]
    
    # Add FVG columns (already prefixed)
    for col in fvg_result.columns:
        combined_data[col] = fvg_result[col]
    
    # Calculate VPFVGSignal
    vpfvg_indicator = VPFVGSignal(
        atr_period=14,              # ATR period
        lvn_dist_multiplier=0.25,   # LVN distance threshold
        poc_shift_multiplier=0.5,   # POC shift threshold
        hvn_overlap_pct=0.7,        # HVN overlap requirement
        lookback_validation=5       # Bars to look back
    )
    vpfvg_result = vpfvg_indicator.compute(combined_data)
    
    # Add VPFVGSignal columns
    for col in vpfvg_result.columns:
        combined_data[f'VPFVGSignal_{col}'] = vpfvg_result[col]
    
    return combined_data


def analyze_signals(data: pd.DataFrame) -> None:
    """
    Analyze the generated signals and provide statistics.
    
    Args:
        data: DataFrame with all indicator data
    """
    print("\n" + "="*60)
    print("SIGNAL ANALYSIS")
    print("="*60)
    
    # Extract signal columns
    long_signals = data['VPFVGSignal_vf_long']
    short_signals = data['VPFVGSignal_vf_short']
    atr_values = data['VPFVGSignal_vf_atr']
    poc_shift = data['VPFVGSignal_vf_poc_shift']
    lvn_distance = data['VPFVGSignal_vf_lvn_distance']
    hvn_overlap = data['VPFVGSignal_vf_hvn_overlap']
    
    # Basic statistics
    total_bars = len(data)
    long_count = long_signals.sum()
    short_count = short_signals.sum()
    total_signals = long_count + short_count
    
    print(f"Total bars analyzed: {total_bars}")
    print(f"Long signals: {long_count} ({long_count/total_bars:.2%})")
    print(f"Short signals: {short_count} ({short_count/total_bars:.2%})")
    print(f"Total signals: {total_signals} ({total_signals/total_bars:.2%})")
    print(f"Signal frequency: {total_signals/total_bars:.2%} of bars")
    
    # ATR statistics
    valid_atr = atr_values.dropna()
    if len(valid_atr) > 0:
        print(f"\nATR Statistics:")
        print(f"  Mean ATR: {valid_atr.mean():.2f}")
        print(f"  Median ATR: {valid_atr.median():.2f}")
        print(f"  ATR Range: {valid_atr.min():.2f} - {valid_atr.max():.2f}")
    
    # POC shift statistics
    valid_poc_shift = poc_shift.dropna()
    if len(valid_poc_shift) > 0:
        print(f"\nPOC Shift Statistics:")
        print(f"  Mean POC shift: {valid_poc_shift.mean():.2f}")
        print(f"  Median POC shift: {valid_poc_shift.median():.2f}")
        print(f"  POC shift Range: {valid_poc_shift.min():.2f} - {valid_poc_shift.max():.2f}")
    
    # Find signal occurrences
    long_indices = data[long_signals].index
    short_indices = data[short_signals].index
    
    print(f"\nSignal Timing:")
    if len(long_indices) > 0:
        print(f"  First long signal: {long_indices[0]}")
        print(f"  Last long signal: {long_indices[-1]}")
    
    if len(short_indices) > 0:
        print(f"  First short signal: {short_indices[0]}")
        print(f"  Last short signal: {short_indices[-1]}")
    
    # Volume Profile statistics
    vp_poc = data['VolumeProfile_poc_price'].dropna()
    vp_lvn = data['VolumeProfile_is_lvn'].sum()
    vp_hvn = data['VolumeProfile_is_hvn'].sum()
    
    print(f"\nVolume Profile Statistics:")
    print(f"  POC price range: {vp_poc.min():.2f} - {vp_poc.max():.2f}")
    print(f"  LVN occurrences: {vp_lvn}")
    print(f"  HVN occurrences: {vp_hvn}")
    
    # FVG statistics
    fvg_bullish = data['FVG_bullish_signal'].sum()
    fvg_bearish = data['FVG_bearish_signal'].sum()
    fvg_active_bull = data['FVG_active_bullish_gaps'].sum()
    fvg_active_bear = data['FVG_active_bearish_gaps'].sum()
    
    print(f"\nFVG Statistics:")
    print(f"  Bullish FVG signals: {fvg_bullish}")
    print(f"  Bearish FVG signals: {fvg_bearish}")
    print(f"  Total active bullish gaps: {fvg_active_bull}")
    print(f"  Total active bearish gaps: {fvg_active_bear}")


def plot_signals(data: pd.DataFrame, start_idx: int = 0, end_idx: int = 500) -> None:
    """
    Plot price data with signals for visualization.
    
    Args:
        data: DataFrame with all indicator data
        start_idx: Start index for plotting
        end_idx: End index for plotting
    """
    # Select data range
    plot_data = data.iloc[start_idx:end_idx]
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Price and signals
    ax1.plot(plot_data.index, plot_data['close'], label='Close Price', linewidth=1)
    ax1.plot(plot_data.index, plot_data['VolumeProfile_poc_price'], 
             label='POC Price', alpha=0.7, linewidth=1)
    
    # Mark long signals
    long_signals = plot_data['VPFVGSignal_vf_long']
    if long_signals.any():
        long_points = plot_data[long_signals]
        ax1.scatter(long_points.index, long_points['close'], 
                   color='green', marker='^', s=100, label='Long Signal', zorder=5)
    
    # Mark short signals
    short_signals = plot_data['VPFVGSignal_vf_short']
    if short_signals.any():
        short_points = plot_data[short_signals]
        ax1.scatter(short_points.index, short_points['close'], 
                   color='red', marker='v', s=100, label='Short Signal', zorder=5)
    
    ax1.set_title('Price Action with VP-FVG Confluence Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ATR and POC Shift
    ax2.plot(plot_data.index, plot_data['VPFVGSignal_vf_atr'], 
             label='ATR', color='blue', alpha=0.7)
    ax2.plot(plot_data.index, plot_data['VPFVGSignal_vf_poc_shift'], 
             label='POC Shift', color='orange', alpha=0.7)
    
    ax2.set_title('ATR and POC Shift')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: HVN/LVN indicators
    hvn_points = plot_data[plot_data['VolumeProfile_is_hvn']]
    lvn_points = plot_data[plot_data['VolumeProfile_is_lvn']]
    
    ax3.plot(plot_data.index, plot_data['close'], label='Close Price', alpha=0.5)
    
    if len(hvn_points) > 0:
        ax3.scatter(hvn_points.index, hvn_points['close'], 
                   color='purple', marker='s', s=50, label='HVN', alpha=0.7)
    
    if len(lvn_points) > 0:
        ax3.scatter(lvn_points.index, lvn_points['close'], 
                   color='yellow', marker='o', s=50, label='LVN', alpha=0.7)
    
    ax3.set_title('High Volume Nodes (HVN) and Low Volume Nodes (LVN)')
    ax3.set_ylabel('Price')
    ax3.set_xlabel('Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('/home/workstation/Personal/btc_research/vpfvg_demo_chart.png', 
                dpi=300, bbox_inches='tight')
    print(f"\nChart saved as: /home/workstation/Personal/btc_research/vpfvg_demo_chart.png")


def main():
    """Main demonstration function."""
    print("VPFVGSignal Confluence Indicator Demo")
    print("="*60)
    
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data(n_bars=1000)
    print(f"Generated {len(data)} bars of data")
    
    # Compute indicators
    print("\nComputing indicators...")
    full_data = compute_indicators(data)
    print("All indicators computed successfully")
    
    # Analyze signals
    analyze_signals(full_data)
    
    # Show example signals
    print("\n" + "="*60)
    print("EXAMPLE SIGNALS")
    print("="*60)
    
    # Find some actual signals to display
    long_signals = full_data[full_data['VPFVGSignal_vf_long']]
    short_signals = full_data[full_data['VPFVGSignal_vf_short']]
    
    if len(long_signals) > 0:
        print(f"\nExample Long Signal:")
        example_long = long_signals.iloc[0]
        print(f"  Time: {example_long.name}")
        print(f"  Price: {example_long['close']:.2f}")
        print(f"  ATR: {example_long['VPFVGSignal_vf_atr']:.2f}")
        print(f"  LVN Distance: {example_long['VPFVGSignal_vf_lvn_distance']:.2f}")
        print(f"  POC Price: {example_long['VolumeProfile_poc_price']:.2f}")
    
    if len(short_signals) > 0:
        print(f"\nExample Short Signal:")
        example_short = short_signals.iloc[0]
        print(f"  Time: {example_short.name}")
        print(f"  Price: {example_short['close']:.2f}")
        print(f"  ATR: {example_short['VPFVGSignal_vf_atr']:.2f}")
        print(f"  HVN Overlap: {example_short['VPFVGSignal_vf_hvn_overlap']:.2f}")
        print(f"  POC Shift: {example_short['VPFVGSignal_vf_poc_shift']:.2f}")
    
    # Plot visualization
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    try:
        plot_signals(full_data, start_idx=200, end_idx=700)
    except Exception as e:
        print(f"Could not create plot: {e}")
        print("This is normal in headless environments")
    
    # Configuration example
    print("\n" + "="*60)
    print("CONFIGURATION EXAMPLE")
    print("="*60)
    
    print("To use this indicator in a YAML configuration:")
    print("""
indicators:
  - id: "VolumeProfile"
    type: "VolumeProfile"
    timeframe: "15m"
    lookback: 200
    price_bins: 30
    
  - id: "FVG"
    type: "FVG"
    timeframe: "15m"
    min_gap_pips: 5.0
    
  - id: "VPFVGSignal"
    type: "VPFVGSignal"
    timeframe: "15m"
    atr_period: 14
    lvn_dist_multiplier: 0.25
    poc_shift_multiplier: 0.5

strategies:
  - id: "VP_FVG_Strategy"
    type: "simple"
    entry_conditions:
      - "VPFVGSignal_vf_long == True"
    exit_conditions:
      - "VPFVGSignal_vf_long == False"
    """)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()