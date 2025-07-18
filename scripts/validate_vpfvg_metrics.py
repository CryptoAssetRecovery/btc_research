"""
VP-FVG Confluence Strategy Metrics Validation Script.

This script validates that the VP-FVG indicators are producing expected
metric ranges and distributions. It helps ensure the strategy is working
as intended before running optimizations.

Expected Ranges:
- vf_lvn_distance_pct: peaks near 0.1-0.2, long tail to 2+
- vf_poc_shift_pct: median ~ 0.3 in calm weeks, > 1 on volatile breakouts
- If blank charts appear → NaNs are still feeding through

Usage:
    python scripts/validate_vpfvg_metrics.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure pandas to suppress FutureWarnings
pd.set_option("future.no_silent_downcasting", True)


def load_sample_data():
    """Load or generate sample OHLCV data for testing."""
    # For demonstration, create synthetic data
    # In practice, you would load real market data
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15T')
    
    # Generate price data with some volatility
    close_prices = 30000 + np.cumsum(np.random.normal(0, 50, n_samples))
    high_prices = close_prices + np.random.exponential(20, n_samples)
    low_prices = close_prices - np.random.exponential(20, n_samples)
    open_prices = close_prices + np.random.normal(0, 10, n_samples)
    volume = np.random.exponential(1000, n_samples)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    return df.set_index('timestamp')


def simulate_vpfvg_indicators(df):
    """Simulate VP-FVG indicator outputs for validation."""
    n = len(df)
    
    # Simulate ATR
    atr_values = np.random.exponential(100, n)
    
    # Simulate LVN distance as percentage of ATR
    # Should peak near 0.1-0.2 with long tail to 2+
    lvn_distance_pct = np.random.lognormal(mean=-1.5, sigma=0.8, size=n)
    lvn_distance_pct[lvn_distance_pct > 3] = np.nan  # Some NaN values
    
    # Simulate POC shift as percentage of ATR
    # Should have median ~ 0.3 in calm periods, > 1 in volatile periods
    poc_shift_pct = np.random.lognormal(mean=-1.2, sigma=0.6, size=n)
    poc_shift_pct[poc_shift_pct > 5] = np.nan  # Some NaN values
    
    # Add some high volatility periods
    volatile_periods = np.random.choice(n, size=n//10, replace=False)
    poc_shift_pct[volatile_periods] = np.random.lognormal(mean=0.2, sigma=0.5, size=len(volatile_periods))
    
    # Simulate HVN overlap
    hvn_overlap = np.random.beta(2, 3, n)  # Skewed toward lower values
    hvn_overlap[hvn_overlap < 0.1] = np.nan  # Some NaN values
    
    # Simulate boolean signals
    vf_long = np.random.choice([True, False], size=n, p=[0.05, 0.95])
    vf_short = np.random.choice([True, False], size=n, p=[0.03, 0.97])
    
    # Create DataFrame with simulated indicators
    indicators_df = pd.DataFrame({
        'VPFVGSignal_vf_atr': atr_values,
        'VPFVGSignal_vf_lvn_distance_pct': lvn_distance_pct,
        'VPFVGSignal_vf_poc_shift_pct': poc_shift_pct,
        'VPFVGSignal_vf_hvn_overlap': hvn_overlap,
        'VPFVGSignal_vf_long': vf_long,
        'VPFVGSignal_vf_short': vf_short,
    }, index=df.index)
    
    return indicators_df


def validate_metrics(indicators_df):
    """Validate that metrics are within expected ranges."""
    print("=" * 60)
    print("VP-FVG METRICS VALIDATION REPORT")
    print("=" * 60)
    
    # Check for NaN values
    print("\\n1. NaN VALUE ANALYSIS:")
    print("-" * 30)
    
    for col in indicators_df.columns:
        nan_count = indicators_df[col].isna().sum()
        nan_pct = (nan_count / len(indicators_df)) * 100
        print(f"{col}: {nan_count} NaN values ({nan_pct:.1f}%)")
    
    # Validate LVN distance distribution
    print("\\n2. LVN DISTANCE PERCENTAGE ANALYSIS:")
    print("-" * 40)
    
    lvn_dist = indicators_df['VPFVGSignal_vf_lvn_distance_pct'].dropna()
    if len(lvn_dist) > 0:
        print(f"Count: {len(lvn_dist)}")
        print(f"Mean: {lvn_dist.mean():.3f}")
        print(f"Median: {lvn_dist.median():.3f}")
        print(f"Std: {lvn_dist.std():.3f}")
        print(f"Min: {lvn_dist.min():.3f}")
        print(f"Max: {lvn_dist.max():.3f}")
        print(f"25th percentile: {lvn_dist.quantile(0.25):.3f}")
        print(f"75th percentile: {lvn_dist.quantile(0.75):.3f}")
        print(f"95th percentile: {lvn_dist.quantile(0.95):.3f}")
        
        # Check expected ranges
        peak_range = lvn_dist[(lvn_dist >= 0.1) & (lvn_dist <= 0.2)]
        long_tail = lvn_dist[lvn_dist > 2.0]
        
        print(f"\\nExpected peak range (0.1-0.2): {len(peak_range)} values ({len(peak_range)/len(lvn_dist)*100:.1f}%)")
        print(f"Long tail (>2.0): {len(long_tail)} values ({len(long_tail)/len(lvn_dist)*100:.1f}%)")
        
        # Validation status
        if len(peak_range) > 0 and lvn_dist.median() < 1.0:
            print("✓ LVN distance distribution looks reasonable")
        else:
            print("⚠ LVN distance distribution may need adjustment")
    else:
        print("❌ No valid LVN distance data found - all values are NaN!")
    
    # Validate POC shift distribution
    print("\\n3. POC SHIFT PERCENTAGE ANALYSIS:")
    print("-" * 40)
    
    poc_shift = indicators_df['VPFVGSignal_vf_poc_shift_pct'].dropna()
    if len(poc_shift) > 0:
        print(f"Count: {len(poc_shift)}")
        print(f"Mean: {poc_shift.mean():.3f}")
        print(f"Median: {poc_shift.median():.3f}")
        print(f"Std: {poc_shift.std():.3f}")
        print(f"Min: {poc_shift.min():.3f}")
        print(f"Max: {poc_shift.max():.3f}")
        print(f"25th percentile: {poc_shift.quantile(0.25):.3f}")
        print(f"75th percentile: {poc_shift.quantile(0.75):.3f}")
        print(f"95th percentile: {poc_shift.quantile(0.95):.3f}")
        
        # Check expected ranges
        calm_median = poc_shift.median()
        volatile_count = len(poc_shift[poc_shift > 1.0])
        
        print(f"\\nCalm period median: {calm_median:.3f} (expected ~ 0.3)")
        print(f"Volatile breakouts (>1.0): {volatile_count} values ({volatile_count/len(poc_shift)*100:.1f}%)")
        
        # Validation status
        if 0.2 <= calm_median <= 0.5:
            print("✓ POC shift distribution looks reasonable")
        else:
            print("⚠ POC shift distribution may need adjustment")
    else:
        print("❌ No valid POC shift data found - all values are NaN!")
    
    # Validate HVN overlap
    print("\\n4. HVN OVERLAP ANALYSIS:")
    print("-" * 30)
    
    hvn_overlap = indicators_df['VPFVGSignal_vf_hvn_overlap'].dropna()
    if len(hvn_overlap) > 0:
        print(f"Count: {len(hvn_overlap)}")
        print(f"Mean: {hvn_overlap.mean():.3f}")
        print(f"Median: {hvn_overlap.median():.3f}")
        print(f"Min: {hvn_overlap.min():.3f}")
        print(f"Max: {hvn_overlap.max():.3f}")
        
        # Check for reasonable overlap values
        if 0.0 <= hvn_overlap.min() <= hvn_overlap.max() <= 1.0:
            print("✓ HVN overlap values are within valid range [0, 1]")
        else:
            print("⚠ HVN overlap values outside expected range [0, 1]")
    else:
        print("❌ No valid HVN overlap data found - all values are NaN!")
    
    # Signal frequency analysis
    print("\\n5. SIGNAL FREQUENCY ANALYSIS:")
    print("-" * 40)
    
    long_signals = indicators_df['VPFVGSignal_vf_long'].sum()
    short_signals = indicators_df['VPFVGSignal_vf_short'].sum()
    total_bars = len(indicators_df)
    
    print(f"Long signals: {long_signals} ({long_signals/total_bars*100:.2f}%)")
    print(f"Short signals: {short_signals} ({short_signals/total_bars*100:.2f}%)")
    
    # Reasonable signal frequency (not too many, not too few)
    if 1 <= long_signals/total_bars*100 <= 10:
        print("✓ Long signal frequency looks reasonable")
    else:
        print("⚠ Long signal frequency may need adjustment")
    
    if 1 <= short_signals/total_bars*100 <= 10:
        print("✓ Short signal frequency looks reasonable")
    else:
        print("⚠ Short signal frequency may need adjustment")


def create_validation_charts(indicators_df):
    """Create validation charts for visual inspection."""
    print("\\n6. CREATING VALIDATION CHARTS:")
    print("-" * 40)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VP-FVG Metrics Validation Charts', fontsize=16)
    
    # Chart 1: LVN Distance Distribution
    ax1 = axes[0, 0]
    lvn_dist = indicators_df['VPFVGSignal_vf_lvn_distance_pct'].dropna()
    if len(lvn_dist) > 0:
        ax1.hist(lvn_dist, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title('LVN Distance Percentage Distribution')
        ax1.set_xlabel('LVN Distance (% of ATR)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Expected peak start')
        ax1.axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='Expected peak end')
        ax1.axvline(x=2.0, color='orange', linestyle='--', alpha=0.7, label='Long tail start')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No valid data\\n(All NaN)', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('LVN Distance - NO DATA')
    
    # Chart 2: POC Shift Distribution
    ax2 = axes[0, 1]
    poc_shift = indicators_df['VPFVGSignal_vf_poc_shift_pct'].dropna()
    if len(poc_shift) > 0:
        ax2.hist(poc_shift, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('POC Shift Percentage Distribution')
        ax2.set_xlabel('POC Shift (% of ATR)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0.3, color='red', linestyle='--', alpha=0.7, label='Expected median')
        ax2.axvline(x=1.0, color='orange', linestyle='--', alpha=0.7, label='Volatile threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No valid data\\n(All NaN)', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('POC Shift - NO DATA')
    
    # Chart 3: HVN Overlap Distribution
    ax3 = axes[1, 0]
    hvn_overlap = indicators_df['VPFVGSignal_vf_hvn_overlap'].dropna()
    if len(hvn_overlap) > 0:
        ax3.hist(hvn_overlap, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_title('HVN Overlap Distribution')
        ax3.set_xlabel('HVN Overlap (0-1)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No valid data\\n(All NaN)', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('HVN Overlap - NO DATA')
    
    # Chart 4: Signal Timing
    ax4 = axes[1, 1]
    signal_data = indicators_df[['VPFVGSignal_vf_long', 'VPFVGSignal_vf_short']].iloc[-200:]  # Last 200 bars
    
    # Create time series plot of signals
    x = range(len(signal_data))
    long_signals = signal_data['VPFVGSignal_vf_long'].astype(int)
    short_signals = signal_data['VPFVGSignal_vf_short'].astype(int) * -1  # Negative for plotting
    
    ax4.plot(x, long_signals, 'go', markersize=3, label='Long signals', alpha=0.7)
    ax4.plot(x, short_signals, 'ro', markersize=3, label='Short signals', alpha=0.7)
    ax4.set_title('Signal Timing (Last 200 bars)')
    ax4.set_xlabel('Time (bars)')
    ax4.set_ylabel('Signal Type')
    ax4.set_ylim(-1.5, 1.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the chart
    output_dir = Path('validation_output')
    output_dir.mkdir(exist_ok=True)
    
    chart_path = output_dir / 'vpfvg_metrics_validation.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Validation charts saved to: {chart_path}")
    
    # Show the plot
    plt.show()


def main():
    """Main validation function."""
    print("VP-FVG Confluence Strategy Metrics Validation")
    print("=" * 60)
    
    # Load sample data
    print("Loading sample data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    
    # Simulate indicators (replace with actual indicator computation)
    print("\\nSimulating VP-FVG indicators...")
    indicators_df = simulate_vpfvg_indicators(df)
    print("Indicators simulated successfully")
    
    # Validate metrics
    validate_metrics(indicators_df)
    
    # Create validation charts
    create_validation_charts(indicators_df)
    
    print("\\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("\\nNext steps:")
    print("1. If charts are blank → NaN values are feeding through")
    print("2. Check metric ranges against expected values")
    print("3. Investigate any anomalies before running backtests")
    print("4. Only proceed with optimization after validation passes")


if __name__ == "__main__":
    main()