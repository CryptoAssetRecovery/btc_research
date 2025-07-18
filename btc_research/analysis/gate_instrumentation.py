"""
Gate Instrumentation and Analysis Utilities.

This module provides comprehensive analysis tools for understanding VP-FVG
strategy gate filtering behavior. It helps identify which gates are most
restrictive and provides recommendations for parameter adjustments.

Key Features:
- Individual gate pass rate analysis
- Combined gate success rate calculation
- Health benchmark comparisons
- Parameter adjustment recommendations
- Timeframe-specific debugging
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GateAnalysisResult:
    """Results of gate analysis."""
    gate_stats: Dict[str, float]
    combined_stats: Dict[str, float]
    recommendations: List[str]
    health_check: Dict[str, bool]
    most_restrictive_gates: List[str]
    total_bars: int


def analyze_gate_statistics(df: pd.DataFrame) -> GateAnalysisResult:
    """
    Analyze gate filtering statistics for VP-FVG strategy.
    
    Args:
        df (pd.DataFrame): DataFrame with all indicator columns
        
    Returns:
        GateAnalysisResult: Comprehensive gate analysis results
    """
    total_bars = len(df)
    
    # Initialize results
    gate_stats = {}
    combined_stats = {}
    recommendations = []
    health_check = {}
    most_restrictive_gates = []
    
    # Long gate analysis
    long_gates = _analyze_long_gates(df)
    gate_stats.update(long_gates)
    
    # Short gate analysis  
    short_gates = _analyze_short_gates(df)
    gate_stats.update(short_gates)
    
    # Combined gate analysis
    combined_stats = _analyze_combined_gates(df)
    
    # Health check against benchmarks
    health_check = _perform_health_check(gate_stats, combined_stats)
    
    # Identify most restrictive gates
    most_restrictive_gates = _identify_restrictive_gates(gate_stats)
    
    # Generate recommendations
    recommendations = _generate_recommendations(gate_stats, combined_stats, health_check)
    
    return GateAnalysisResult(
        gate_stats=gate_stats,
        combined_stats=combined_stats,
        recommendations=recommendations,
        health_check=health_check,
        most_restrictive_gates=most_restrictive_gates,
        total_bars=total_bars
    )


def _analyze_long_gates(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze individual long entry gates."""
    gate_stats = {}
    
    # Long gate 1: VP-FVG Long Signal
    if 'VPFVGSignal_vf_long' in df.columns:
        gate_stats['long_vf_signal'] = df['VPFVGSignal_vf_long'].fillna(False).mean()
    
    # Long gate 2: ADX Range (< 20)
    if 'ADX_1h_ADX_value' in df.columns:
        gate_stats['long_adx_range'] = (df['ADX_1h_ADX_value'] < 20).fillna(False).mean()
    
    # Long gate 3: No bearish DI
    if 'ADX_1h_DI_bearish' in df.columns:
        gate_stats['long_no_bearish_di'] = (~df['ADX_1h_DI_bearish']).fillna(False).mean()
    
    # Long gate 4: LVN distance
    if 'VPFVGSignal_vf_lvn_distance_pct' in df.columns:
        gate_stats['long_lvn_distance'] = (df['VPFVGSignal_vf_lvn_distance_pct'] <= 0.25).fillna(False).mean()
    
    # Long gate 5: Valid ATR
    if 'VPFVGSignal_vf_atr' in df.columns:
        gate_stats['long_valid_atr'] = (df['VPFVGSignal_vf_atr'] > 0).fillna(False).mean()
    
    return gate_stats


def _analyze_short_gates(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze individual short entry gates."""
    gate_stats = {}
    
    # Short gate 1: VP-FVG Short Signal
    if 'VPFVGSignal_vf_short' in df.columns:
        gate_stats['short_vf_signal'] = df['VPFVGSignal_vf_short'].fillna(False).mean()
    
    # Short gate 2: ADX Trend (>= 20)
    if 'ADX_1h_ADX_value' in df.columns:
        gate_stats['short_adx_trend'] = (df['ADX_1h_ADX_value'] >= 20).fillna(False).mean()
    
    # Short gate 3: Bearish DI
    if 'ADX_1h_DI_bearish' in df.columns:
        gate_stats['short_bearish_di'] = df['ADX_1h_DI_bearish'].fillna(False).mean()
    
    # Short gate 4: HVN overlap
    if 'VPFVGSignal_vf_hvn_overlap' in df.columns:
        gate_stats['short_hvn_overlap'] = (df['VPFVGSignal_vf_hvn_overlap'] >= 0.15).fillna(False).mean()
    
    # Short gate 5: Valid ATR
    if 'VPFVGSignal_vf_atr' in df.columns:
        gate_stats['short_valid_atr'] = (df['VPFVGSignal_vf_atr'] > 0).fillna(False).mean()
    
    return gate_stats


def _analyze_combined_gates(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze combined gate pass rates."""
    combined_stats = {}
    
    # Combined long gates
    long_conditions = []
    if 'VPFVGSignal_vf_long' in df.columns:
        long_conditions.append(df['VPFVGSignal_vf_long'].fillna(False))
    if 'ADX_1h_ADX_value' in df.columns:
        long_conditions.append(df['ADX_1h_ADX_value'] < 20)
    if 'ADX_1h_DI_bearish' in df.columns:
        long_conditions.append(~df['ADX_1h_DI_bearish'].fillna(True))
    if 'VPFVGSignal_vf_lvn_distance_pct' in df.columns:
        long_conditions.append(df['VPFVGSignal_vf_lvn_distance_pct'] <= 0.25)
    if 'VPFVGSignal_vf_atr' in df.columns:
        long_conditions.append(df['VPFVGSignal_vf_atr'] > 0)
    
    if long_conditions:
        combined_long = pd.concat(long_conditions, axis=1).fillna(False).all(axis=1)
        combined_stats['combined_long'] = combined_long.mean()
        combined_stats['long_signal_count'] = combined_long.sum()
    
    # Combined short gates
    short_conditions = []
    if 'VPFVGSignal_vf_short' in df.columns:
        short_conditions.append(df['VPFVGSignal_vf_short'].fillna(False))
    if 'ADX_1h_ADX_value' in df.columns:
        short_conditions.append(df['ADX_1h_ADX_value'] >= 20)
    if 'ADX_1h_DI_bearish' in df.columns:
        short_conditions.append(df['ADX_1h_DI_bearish'].fillna(False))
    if 'VPFVGSignal_vf_hvn_overlap' in df.columns:
        short_conditions.append(df['VPFVGSignal_vf_hvn_overlap'] >= 0.15)
    if 'VPFVGSignal_vf_atr' in df.columns:
        short_conditions.append(df['VPFVGSignal_vf_atr'] > 0)
    
    if short_conditions:
        combined_short = pd.concat(short_conditions, axis=1).fillna(False).all(axis=1)
        combined_stats['combined_short'] = combined_short.mean()
        combined_stats['short_signal_count'] = combined_short.sum()
    
    # Total signals
    total_signals = (combined_stats.get('long_signal_count', 0) + 
                    combined_stats.get('short_signal_count', 0))
    combined_stats['total_signals'] = total_signals
    
    return combined_stats


def _perform_health_check(gate_stats: Dict[str, float], 
                         combined_stats: Dict[str, float]) -> Dict[str, bool]:
    """Perform health check against expected benchmarks."""
    health_check = {}
    
    # Healthy numbers: vf_long raw ≥ 1%, after filters ≥ 0.1%
    health_check['long_vf_signal_healthy'] = gate_stats.get('long_vf_signal', 0) >= 0.01
    health_check['short_vf_signal_healthy'] = gate_stats.get('short_vf_signal', 0) >= 0.01
    
    # Combined signals should be ≥ 0.1%
    health_check['combined_long_healthy'] = combined_stats.get('combined_long', 0) >= 0.001
    health_check['combined_short_healthy'] = combined_stats.get('combined_short', 0) >= 0.001
    
    # Total signals should generate reasonable trade count (~24 trades/year on 15m)
    # For a year of 15m data (~35,000 bars), expect at least 24 signals
    expected_min_signals = 24  # Minimum for reasonable strategy
    health_check['total_signals_healthy'] = combined_stats.get('total_signals', 0) >= expected_min_signals
    
    return health_check


def _identify_restrictive_gates(gate_stats: Dict[str, float]) -> List[str]:
    """Identify gates with very low pass rates (<1%)."""
    restrictive_gates = []
    
    for gate_name, pass_rate in gate_stats.items():
        if pass_rate < 0.01:  # Less than 1%
            restrictive_gates.append(gate_name)
    
    return restrictive_gates


def _generate_recommendations(gate_stats: Dict[str, float],
                            combined_stats: Dict[str, float],
                            health_check: Dict[str, bool]) -> List[str]:
    """Generate specific parameter adjustment recommendations."""
    recommendations = []
    
    # Check long signal health
    if not health_check.get('long_vf_signal_healthy', True):
        recommendations.append("Long VP-FVG signal rate too low (<1%) - consider loosening VP parameters")
    
    if not health_check.get('combined_long_healthy', True):
        recommendations.append("Combined long signals too restrictive - consider:")
        if gate_stats.get('long_lvn_distance', 0) < 0.05:
            recommendations.append("  - Increase lvn_dist_multiplier from 0.25 to 0.35")
        if gate_stats.get('long_adx_range', 0) < 0.3:
            recommendations.append("  - Increase ADX range threshold from 20 to 25")
    
    # Check short signal health
    if not health_check.get('short_vf_signal_healthy', True):
        recommendations.append("Short VP-FVG signal rate too low (<1%) - consider loosening VP parameters")
    
    if not health_check.get('combined_short_healthy', True):
        recommendations.append("Combined short signals too restrictive - consider:")
        if gate_stats.get('short_hvn_overlap', 0) < 0.05:
            recommendations.append("  - Decrease hvn_overlap_pct from 0.15 to 0.10")
        if gate_stats.get('short_adx_trend', 0) < 0.3:
            recommendations.append("  - Decrease ADX trend threshold from 20 to 15")
    
    # Overall signal count
    if not health_check.get('total_signals_healthy', True):
        total_signals = combined_stats.get('total_signals', 0)
        recommendations.append(f"Total signals ({total_signals}) too low - strategy may be over-fitted")
        recommendations.append("Consider implementing the vol-adaptive FVG detection improvements")
    
    return recommendations


def print_gate_analysis_report(result: GateAnalysisResult) -> None:
    """Print comprehensive gate analysis report."""
    print("=" * 70)
    print("VP-FVG STRATEGY GATE ANALYSIS REPORT")
    print("=" * 70)
    
    print(f"\nTotal bars analyzed: {result.total_bars:,}")
    
    # Individual gate statistics
    print("\n1. INDIVIDUAL GATE PASS RATES:")
    print("-" * 40)
    
    print("\nLong Entry Gates:")
    for gate_name, pass_rate in result.gate_stats.items():
        if gate_name.startswith('long_'):
            description = _get_gate_description(gate_name)
            print(f"  {description}: {pass_rate:.3%} ({int(pass_rate * result.total_bars):,} bars)")
    
    print("\nShort Entry Gates:")
    for gate_name, pass_rate in result.gate_stats.items():
        if gate_name.startswith('short_'):
            description = _get_gate_description(gate_name)
            print(f"  {description}: {pass_rate:.3%} ({int(pass_rate * result.total_bars):,} bars)")
    
    # Combined gate statistics
    print("\n2. COMBINED GATE SUCCESS RATES:")
    print("-" * 40)
    
    long_combined = result.combined_stats.get('combined_long', 0)
    short_combined = result.combined_stats.get('combined_short', 0)
    long_signals = result.combined_stats.get('long_signal_count', 0)
    short_signals = result.combined_stats.get('short_signal_count', 0)
    total_signals = result.combined_stats.get('total_signals', 0)
    
    print(f"Long entry (all gates): {long_combined:.4%} ({long_signals:,} signals)")
    print(f"Short entry (all gates): {short_combined:.4%} ({short_signals:,} signals)")
    print(f"Total signals: {total_signals:,}")
    
    # Health check
    print("\n3. HEALTH CHECK VS BENCHMARKS:")
    print("-" * 40)
    
    benchmarks = {
        'long_vf_signal_healthy': 'Long VP-FVG signal ≥ 1%',
        'short_vf_signal_healthy': 'Short VP-FVG signal ≥ 1%',
        'combined_long_healthy': 'Combined long signals ≥ 0.1%',
        'combined_short_healthy': 'Combined short signals ≥ 0.1%',
        'total_signals_healthy': 'Total signals ≥ 24/year'
    }
    
    for check_name, description in benchmarks.items():
        status = "✓" if result.health_check.get(check_name, False) else "✗"
        print(f"  {status} {description}")
    
    # Most restrictive gates
    print("\n4. MOST RESTRICTIVE GATES (<1% pass rate):")
    print("-" * 40)
    
    if result.most_restrictive_gates:
        for gate in result.most_restrictive_gates:
            description = _get_gate_description(gate)
            pass_rate = result.gate_stats.get(gate, 0)
            print(f"  ⚠ {description}: {pass_rate:.3%}")
    else:
        print("  ✓ No extremely restrictive gates found")
    
    # Recommendations
    print("\n5. PARAMETER ADJUSTMENT RECOMMENDATIONS:")
    print("-" * 40)
    
    if result.recommendations:
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  ✓ Gate filtering appears healthy - no adjustments needed")
    
    print("\n" + "=" * 70)


def _get_gate_description(gate_name: str) -> str:
    """Get human-readable description for gate name."""
    descriptions = {
        'long_vf_signal': 'VP-FVG Long Signal',
        'long_adx_range': 'ADX Range (<20)',
        'long_no_bearish_di': 'No Bearish DI',
        'long_lvn_distance': 'LVN Distance (≤25% ATR)',
        'long_valid_atr': 'Valid ATR (>0)',
        'short_vf_signal': 'VP-FVG Short Signal',
        'short_adx_trend': 'ADX Trend (≥20)',
        'short_bearish_di': 'Bearish DI',
        'short_hvn_overlap': 'HVN Overlap (≥15%)',
        'short_valid_atr': 'Valid ATR (>0)',
    }
    return descriptions.get(gate_name, gate_name)


def quick_gate_summary(df: pd.DataFrame) -> None:
    """Quick summary of gate statistics."""
    result = analyze_gate_statistics(df)
    
    print("QUICK GATE SUMMARY:")
    print("=" * 30)
    print(f"Total bars: {result.total_bars:,}")
    print(f"Long signals: {result.combined_stats.get('long_signal_count', 0):,}")
    print(f"Short signals: {result.combined_stats.get('short_signal_count', 0):,}")
    print(f"Total signals: {result.combined_stats.get('total_signals', 0):,}")
    print(f"Signal rate: {result.combined_stats.get('total_signals', 0) / result.total_bars:.4%}")
    
    if result.most_restrictive_gates:
        print(f"Most restrictive: {', '.join(result.most_restrictive_gates)}")


def analyze_gate_progression(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze how signals are filtered at each gate stage."""
    # Create progression DataFrame
    progression = pd.DataFrame(index=['Long', 'Short'], columns=[
        'Initial Signal', 'After ADX', 'After Direction', 'After Distance', 'Final'
    ])
    
    # Long progression
    if 'VPFVGSignal_vf_long' in df.columns:
        long_initial = df['VPFVGSignal_vf_long'].fillna(False).sum()
        progression.loc['Long', 'Initial Signal'] = long_initial
        
        # After ADX filter
        long_after_adx = df['VPFVGSignal_vf_long'].fillna(False) & (df['ADX_1h_ADX_value'] < 20)
        progression.loc['Long', 'After ADX'] = long_after_adx.sum()
        
        # After direction filter
        long_after_dir = long_after_adx & (~df['ADX_1h_DI_bearish'].fillna(True))
        progression.loc['Long', 'After Direction'] = long_after_dir.sum()
        
        # After distance filter
        long_after_dist = long_after_dir & (df['VPFVGSignal_vf_lvn_distance_pct'] <= 0.25)
        progression.loc['Long', 'After Distance'] = long_after_dist.sum()
        
        # Final (with ATR filter)
        long_final = long_after_dist & (df['VPFVGSignal_vf_atr'] > 0)
        progression.loc['Long', 'Final'] = long_final.sum()
    
    # Short progression
    if 'VPFVGSignal_vf_short' in df.columns:
        short_initial = df['VPFVGSignal_vf_short'].fillna(False).sum()
        progression.loc['Short', 'Initial Signal'] = short_initial
        
        # After ADX filter
        short_after_adx = df['VPFVGSignal_vf_short'].fillna(False) & (df['ADX_1h_ADX_value'] >= 20)
        progression.loc['Short', 'After ADX'] = short_after_adx.sum()
        
        # After direction filter
        short_after_dir = short_after_adx & df['ADX_1h_DI_bearish'].fillna(False)
        progression.loc['Short', 'After Direction'] = short_after_dir.sum()
        
        # After distance filter
        short_after_dist = short_after_dir & (df['VPFVGSignal_vf_hvn_overlap'] >= 0.15)
        progression.loc['Short', 'After Distance'] = short_after_dist.sum()
        
        # Final (with ATR filter)
        short_final = short_after_dist & (df['VPFVGSignal_vf_atr'] > 0)
        progression.loc['Short', 'Final'] = short_final.sum()
    
    return progression.fillna(0).astype(int)


def debug_specific_timeframe(df: pd.DataFrame, 
                           start_date: str, 
                           end_date: str) -> None:
    """Debug gate conditions for specific timeframe."""
    # Filter DataFrame to specific timeframe
    df_period = df[start_date:end_date]
    
    print(f"DEBUGGING PERIOD: {start_date} to {end_date}")
    print("=" * 50)
    
    # Quick analysis for this period
    result = analyze_gate_statistics(df_period)
    
    print(f"Period length: {len(df_period):,} bars")
    print(f"Long signals: {result.combined_stats.get('long_signal_count', 0):,}")
    print(f"Short signals: {result.combined_stats.get('short_signal_count', 0):,}")
    
    # Show most restrictive gates for this period
    if result.most_restrictive_gates:
        print("\nMost restrictive gates in this period:")
        for gate in result.most_restrictive_gates:
            pass_rate = result.gate_stats.get(gate, 0)
            print(f"  - {_get_gate_description(gate)}: {pass_rate:.3%}")


# Example usage functions
def example_usage():
    """Example usage of gate instrumentation."""
    print("Example Gate Instrumentation Usage:")
    print("=" * 40)
    
    # Example code (commented out as it requires actual data)
    """
    # Basic analysis
    result = analyze_gate_statistics(df)
    print_gate_analysis_report(result)
    
    # Quick summary
    quick_gate_summary(df)
    
    # Gate progression analysis
    progression = analyze_gate_progression(df)
    print("Signal filtering progression:")
    print(progression)
    
    # Debug specific period
    debug_specific_timeframe(df, '2022-01-01', '2022-12-31')
    """
    
    print("Import this module and use the functions above with your DataFrame")
    print("containing VP-FVG indicator columns.")