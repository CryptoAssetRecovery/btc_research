#!/usr/bin/env python3
"""
Enhanced Monte Carlo Robustness Testing Framework Demo

This example demonstrates the comprehensive Monte Carlo robustness testing
capabilities including advanced bootstrap methods, data perturbation testing,
synthetic data generation, stress testing, and comprehensive scoring.

Usage:
    python enhanced_monte_carlo_robustness_demo.py
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Dict, Any

from btc_research.optimization.robustness.monte_carlo import MonteCarloRobustnessTest
from btc_research.optimization.robustness.data_perturbation import DataPerturbationTest
from btc_research.optimization.robustness.synthetic_data import SyntheticDataTest
from btc_research.optimization.robustness.stress_tests import StressTestFramework
from btc_research.optimization.robustness.parameter_sensitivity import ParameterSensitivityTest
from btc_research.optimization.robustness.robustness_metrics import RobustnessMetrics, RobustnessScoring
from btc_research.optimization.robustness.comprehensive_framework import ComprehensiveRobustnessFramework

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_sample_data(n_periods: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic sample trading data for demonstration.
    
    Args:
        n_periods: Number of time periods
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(random_seed)
    
    # Generate price path with realistic characteristics
    initial_price = 50000  # Starting price (e.g., BTC-USD)
    dt = 1/365  # Daily time step
    
    # Market parameters
    mu = 0.1  # Annual drift
    sigma = 0.4  # Annual volatility
    
    # Generate log-normal price path
    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_periods)
    log_prices = np.cumsum(returns)
    prices = initial_price * np.exp(log_prices)
    
    # Create OHLC data
    data = []
    for i, close_price in enumerate(prices):
        # Add some intraday volatility
        daily_vol = sigma * np.sqrt(dt) * 0.5
        high = close_price * (1 + abs(np.random.normal(0, daily_vol)))
        low = close_price * (1 - abs(np.random.normal(0, daily_vol)))
        
        if i == 0:
            open_price = close_price
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, daily_vol * 0.3))
        
        # Ensure OHLC consistency
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Generate volume (log-normal distribution)
        volume = np.random.lognormal(10, 1)
        
        data.append({
            'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


def simple_moving_average_strategy(
    data: pd.DataFrame, 
    parameters: Dict[str, Any]
) -> Dict[str, float]:
    """
    Simple moving average crossover strategy for demonstration.
    
    Args:
        data: OHLCV data
        parameters: Strategy parameters
        
    Returns:
        Dictionary with performance metrics
    """
    short_window = parameters.get('short_window', 10)
    long_window = parameters.get('long_window', 50)
    
    if len(data) < long_window:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_trades': 0
        }
    
    # Calculate moving averages
    short_ma = data['close'].rolling(window=short_window).mean()
    long_ma = data['close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals = np.where(short_ma > long_ma, 1, -1)
    signals = pd.Series(signals, index=data.index)
    
    # Calculate returns
    returns = data['close'].pct_change()
    strategy_returns = signals.shift(1) * returns
    
    # Remove NaN values
    strategy_returns = strategy_returns.dropna()
    
    if len(strategy_returns) == 0:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_trades': 0
        }
    
    # Calculate performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    # Calculate max drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calculate win rate
    winning_trades = (strategy_returns > 0).sum()
    total_trades = len(strategy_returns)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Count number of position changes
    position_changes = (signals != signals.shift(1)).sum()
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': position_changes
    }


def demonstrate_enhanced_monte_carlo_testing():
    """
    Demonstrate the enhanced Monte Carlo robustness testing framework.
    """
    logger.info("🚀 Starting Enhanced Monte Carlo Robustness Testing Demo")
    
    # Generate sample data
    logger.info("📊 Generating sample trading data...")
    data = generate_sample_data(n_periods=500, random_seed=42)
    logger.info(f"Generated {len(data)} periods of data from {data.index[0]} to {data.index[-1]}")
    
    # Define strategy parameters
    base_parameters = {
        'short_window': 10,
        'long_window': 50
    }
    
    logger.info(f"📋 Testing strategy with parameters: {base_parameters}")
    
    # Test baseline strategy performance
    logger.info("🔍 Testing baseline strategy performance...")
    baseline_result = simple_moving_average_strategy(data, base_parameters)
    logger.info(f"Baseline performance: {baseline_result}")
    
    # 1. Enhanced Monte Carlo Bootstrap Testing
    logger.info("\n🎲 Running Enhanced Monte Carlo Bootstrap Testing...")
    
    mc_test = MonteCarloRobustnessTest(
        data=data,
        random_seed=42,
        enable_parallel=True,
        max_workers=4
    )
    
    # Run comprehensive Monte Carlo test suite
    logger.info("Running comprehensive Monte Carlo test suite...")
    mc_suite_results = mc_test.run_advanced_monte_carlo_suite(
        parameters=base_parameters,
        backtest_function=simple_moving_average_strategy,
        n_simulations=200,  # Reduced for demo
        test_methods=['basic_bootstrap', 'block_bootstrap', 'circular_bootstrap', 'stationary_bootstrap']
    )
    
    logger.info("Monte Carlo Suite Results:")
    for method, result in mc_suite_results.items():
        logger.info(f"  {method}: Success Rate = {result.success_rate:.3f}, "
                   f"Mean Return = {result.summary_stats.get('total_return', {}).get('mean', 0):.4f}")
    
    # Run Monte Carlo convergence analysis
    logger.info("Running Monte Carlo convergence analysis...")
    convergence_result = mc_test.calculate_monte_carlo_convergence(
        parameters=base_parameters,
        backtest_function=simple_moving_average_strategy,
        max_simulations=500,
        convergence_threshold=0.001
    )
    
    logger.info(f"Convergence Analysis:")
    logger.info(f"  Converged: {convergence_result['converged']}")
    logger.info(f"  Optimal simulations: {convergence_result['optimal_n_simulations']}")
    logger.info(f"  Final metric mean: {convergence_result['final_metric_mean']:.4f}")
    
    # Run Monte Carlo diagnostic tests
    logger.info("Running Monte Carlo diagnostic tests...")
    diagnostic_result = mc_test.run_monte_carlo_diagnostic(
        parameters=base_parameters,
        backtest_function=simple_moving_average_strategy,
        n_simulations=200
    )
    
    logger.info(f"Diagnostic Quality Score: {diagnostic_result.get('overall_quality_score', 0):.3f}")
    
    # 2. Enhanced Data Perturbation Testing
    logger.info("\n🌪️ Running Enhanced Data Perturbation Testing...")
    
    perturbation_test = DataPerturbationTest(
        data=data,
        random_seed=42
    )
    
    # Run comprehensive perturbation suite (this would need the extensions to be properly integrated)
    # For demo, we'll run the main perturbation test
    perturbation_result = perturbation_test.run_test(
        parameters=base_parameters,
        backtest_function=simple_moving_average_strategy,
        n_simulations=200,
        perturbation_types=['price_noise', 'volume_noise', 'regime_change']
    )
    
    logger.info(f"Data Perturbation Results:")
    logger.info(f"  Success Rate: {perturbation_result.success_rate:.3f}")
    logger.info(f"  Mean Return: {perturbation_result.summary_stats.get('total_return', {}).get('mean', 0):.4f}")
    
    # Run specific noise analysis
    noise_analysis = perturbation_test.run_price_noise_test(
        parameters=base_parameters,
        backtest_function=simple_moving_average_strategy,
        noise_levels=[0.001, 0.005, 0.01, 0.02],
        n_simulations_per_level=50
    )
    
    logger.info(f"Price Noise Analysis completed with {len(noise_analysis['noise_results'])} noise levels")
    
    # 3. Synthetic Data Testing
    logger.info("\n🔬 Running Synthetic Data Testing...")
    
    synthetic_test = SyntheticDataTest(
        data=data,
        random_seed=42
    )
    
    # Run permutation test
    permutation_result = synthetic_test.run_permutation_test(
        parameters=base_parameters,
        backtest_function=simple_moving_average_strategy,
        n_simulations=200
    )
    
    logger.info(f"Permutation Test Results:")
    logger.info(f"  P-value: {permutation_result['significance_test']['p_value']:.4f}")
    logger.info(f"  Significant at 5%: {permutation_result['significance_test']['significant_at_05']}")
    logger.info(f"  Overfitting detected: {permutation_result['significance_test']['overfitting_detected']}")
    
    # Run overfitting detection
    overfitting_result = synthetic_test.run_overfitting_detection(
        parameters=base_parameters,
        backtest_function=simple_moving_average_strategy,
        n_simulations=200
    )
    
    logger.info(f"Overfitting Detection:")
    logger.info(f"  Overfitting Score: {overfitting_result['overfitting_score']:.3f}")
    logger.info(f"  Overfitting Detected: {overfitting_result['overfitting_detected']}")
    
    # 4. Stress Testing
    logger.info("\n⚡ Running Stress Testing...")
    
    stress_test = StressTestFramework(
        data=data,
        random_seed=42
    )
    
    stress_result = stress_test.run_test(
        parameters=base_parameters,
        backtest_function=simple_moving_average_strategy,
        n_simulations=200,
        stress_scenarios=['flash_crash', 'high_volatility', 'liquidity_crisis']
    )
    
    logger.info(f"Stress Test Results:")
    logger.info(f"  Survival Rate: {stress_result.success_rate:.3f}")
    logger.info(f"  Worst Case Return: {stress_result.worst_case_scenario.get('total_return', 0):.4f}")
    
    # Run flash crash specific test
    flash_crash_result = stress_test.run_flash_crash_test(
        parameters=base_parameters,
        backtest_function=simple_moving_average_strategy,
        crash_magnitudes=[0.1, 0.2, 0.3],
        recovery_times=[5, 20, 50],
        n_simulations=20
    )
    
    logger.info(f"Flash Crash Test completed with {len(flash_crash_result['crash_results'])} scenarios")
    
    # 5. Parameter Sensitivity Analysis
    logger.info("\n🎯 Running Parameter Sensitivity Analysis...")
    
    sensitivity_test = ParameterSensitivityTest(
        data=data,
        random_seed=42
    )
    
    sensitivity_result = sensitivity_test.run_test(
        parameters=base_parameters,
        backtest_function=simple_moving_average_strategy,
        n_simulations=50,  # Reduced for demo
        sensitivity_range=0.3
    )
    
    logger.info(f"Parameter Sensitivity Results:")
    logger.info(f"  Success Rate: {sensitivity_result.success_rate:.3f}")
    for param, stats in sensitivity_result.summary_stats.items():
        if isinstance(stats, dict) and 'sensitivity_coefficient' in stats:
            logger.info(f"  {param} sensitivity: {stats['sensitivity_coefficient']:.4f}")
    
    # 6. Comprehensive Scoring
    logger.info("\n📊 Calculating Comprehensive Robustness Score...")
    
    # Collect all results for comprehensive analysis
    all_results = {
        'monte_carlo_results': mc_suite_results['basic_bootstrap'],
        'perturbation_results': perturbation_result,
        'synthetic_results': permutation_result,
        'stress_results': stress_result,
        'sensitivity_results': sensitivity_result
    }
    
    # Calculate comprehensive metrics
    tail_risk_metrics = RobustnessMetrics.calculate_tail_risk_metrics(
        results=mc_suite_results['basic_bootstrap'].results
    )
    
    stability_metrics = RobustnessMetrics.calculate_stability_metrics(
        results=mc_suite_results['basic_bootstrap'].results,
        baseline_results=baseline_result
    )
    
    overfitting_metrics = RobustnessMetrics.calculate_overfitting_metrics(
        original_results=baseline_result,
        oos_results=perturbation_result.results,
        synthetic_results=permutation_result['permutation_results']
    )
    
    # Create comprehensive robustness assessment
    robustness_data = {
        'tail_risk_metrics': tail_risk_metrics,
        'stability_metrics': stability_metrics,
        'overfitting_metrics': overfitting_metrics,
        'stress_results': stress_result,
    }
    
    # Initialize scoring system
    scorer = RobustnessScoring()\n    comprehensive_scores = scorer.calculate_comprehensive_score(robustness_data)
    
    logger.info(f"\\n🏆 COMPREHENSIVE ROBUSTNESS ASSESSMENT")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Robustness Score: {comprehensive_scores['overall_robustness_score']:.3f}")
    logger.info(f"Robustness Grade: {comprehensive_scores['robustness_grade']}")
    logger.info(f"")
    logger.info(f"Component Scores:")
    for component, score in comprehensive_scores.items():
        if component not in ['overall_robustness_score', 'robustness_grade']:
            logger.info(f"  {component.replace('_', ' ').title()}: {score:.3f}")
    
    # Generate detailed report
    detailed_report = scorer.generate_robustness_report(
        component_scores=comprehensive_scores,
        detailed_results=all_results
    )
    
    logger.info(f"\\n📋 DETAILED ROBUSTNESS REPORT")
    logger.info(detailed_report)
    
    # 7. Summary and Recommendations
    logger.info(f"\\n💡 SUMMARY AND RECOMMENDATIONS")
    logger.info(f"{'='*60}")
    
    overall_score = comprehensive_scores['overall_robustness_score']
    
    if overall_score >= 0.8:
        logger.info("✅ EXCELLENT: Strategy shows high robustness across all dimensions")
        logger.info("   Recommended for live trading with standard risk management")
    elif overall_score >= 0.6:
        logger.info("✅ GOOD: Strategy shows adequate robustness with some weaknesses")
        logger.info("   Recommended for live trading with enhanced risk management")
    elif overall_score >= 0.4:
        logger.info("⚠️ MODERATE: Strategy shows mixed robustness results")
        logger.info("   Further optimization recommended before live trading")
    else:
        logger.info("❌ POOR: Strategy shows low robustness")
        logger.info("   Significant revisions required before considering live trading")
    
    # Key insights
    logger.info(f"\\nKey Insights:")
    logger.info(f"• Monte Carlo convergence: {'✅' if convergence_result['converged'] else '❌'}")
    logger.info(f"• Overfitting risk: {'⚠️ High' if overfitting_result['overfitting_detected'] else '✅ Low'}")
    logger.info(f"• Stress resilience: {'✅ High' if stress_result.success_rate > 0.7 else '⚠️ Low'}")
    logger.info(f"• Parameter sensitivity: {'✅ Low' if sensitivity_result.success_rate > 0.8 else '⚠️ High'}")
    
    logger.info(f"\\n🎉 Enhanced Monte Carlo Robustness Testing Demo Complete!")
    
    return {
        'baseline_result': baseline_result,
        'monte_carlo_results': mc_suite_results,
        'perturbation_result': perturbation_result,
        'synthetic_results': {
            'permutation': permutation_result,
            'overfitting': overfitting_result
        },
        'stress_result': stress_result,
        'sensitivity_result': sensitivity_result,
        'comprehensive_scores': comprehensive_scores,
        'detailed_report': detailed_report
    }


if __name__ == "__main__":
    try:
        results = demonstrate_enhanced_monte_carlo_testing()
        print(f"\\n✅ Demo completed successfully!")
        print(f"Final robustness score: {results['comprehensive_scores']['overall_robustness_score']:.3f}")
        print(f"Robustness grade: {results['comprehensive_scores']['robustness_grade']}")
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)