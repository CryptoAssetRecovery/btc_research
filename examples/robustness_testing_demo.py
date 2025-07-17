"""
Comprehensive robustness testing framework demonstration.

This example demonstrates how to use the Monte Carlo robustness testing
framework to validate trading strategy optimization results and detect
overfitting through various stress testing methods.
"""

import logging
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import robustness testing components
from btc_research.optimization.robustness import (
    MonteCarloRobustnessTest,
    DataPerturbationTest,
    SyntheticDataTest,
    StressTestFramework,
    RobustnessMetrics,
    RobustnessScoring
)


def generate_sample_data(n_periods: int = 1000) -> pd.DataFrame:
    """
    Generate sample OHLCV data for demonstration.
    
    Args:
        n_periods: Number of periods to generate
        
    Returns:
        DataFrame with sample market data
    """
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    
    # Base price and trends
    base_price = 50000
    trend = 0.0002  # Slight upward trend
    volatility = 0.02
    
    # Generate returns with some autocorrelation
    returns = []
    prev_return = 0
    
    for i in range(n_periods):
        # Add trend and mean reversion
        mean_return = trend - 0.1 * prev_return
        return_val = np.random.normal(mean_return, volatility)
        returns.append(return_val)
        prev_return = return_val
    
    # Calculate prices
    prices = [base_price]
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    prices = prices[1:]  # Remove initial price
    
    # Generate OHLCV data
    data = []
    for i, close_price in enumerate(prices):
        # Generate realistic OHLC
        volatility_factor = abs(returns[i]) * 10  # Higher volatility = wider range
        
        open_price = close_price * (1 + np.random.normal(0, volatility_factor * 0.5))
        
        high_base = max(open_price, close_price)
        low_base = min(open_price, close_price)
        
        high_price = high_base * (1 + np.random.exponential(volatility_factor))
        low_price = low_base * (1 - np.random.exponential(volatility_factor))
        
        # Volume with some correlation to price movement
        base_volume = 1000
        volume_factor = 1 + abs(returns[i]) * 5  # Higher on volatile days
        volume = base_volume * volume_factor * np.random.uniform(0.5, 2.0)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    # Create DataFrame with datetime index
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='1H')
    
    return df


def simple_backtest_function(data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, float]:
    """
    Simple moving average crossover strategy for demonstration.
    
    Args:
        data: OHLCV data
        parameters: Strategy parameters
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Extract parameters
        fast_period = parameters.get('fast_period', 10)
        slow_period = parameters.get('slow_period', 20)
        
        if len(data) < max(fast_period, slow_period) + 1:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
        
        # Calculate moving averages
        data = data.copy()
        data['fast_ma'] = data['close'].rolling(window=fast_period).mean()
        data['slow_ma'] = data['close'].rolling(window=slow_period).mean()
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1
        data.loc[data['fast_ma'] <= data['slow_ma'], 'signal'] = -1
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        data['strategy_returns'] = data['signal'].shift(1) * data['returns']
        
        # Drop NaN values
        strategy_returns = data['strategy_returns'].dropna()
        
        if len(strategy_returns) == 0:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
        
        # Calculate performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        
        # Sharpe ratio (annualized)
        if strategy_returns.std() > 0:
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(24 * 365)  # Hourly data
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(data[data['signal'].diff() != 0]) // 2,
            'win_rate': len(strategy_returns[strategy_returns > 0]) / len(strategy_returns) if len(strategy_returns) > 0 else 0.0
        }
        
    except Exception as e:
        logger.warning(f"Backtest failed: {e}")
        return {'total_return': float('-inf'), 'sharpe_ratio': float('-inf'), 'max_drawdown': 1.0}


def demonstrate_monte_carlo_testing():
    """Demonstrate Monte Carlo robustness testing."""
    logger.info("=== Monte Carlo Robustness Testing ===")
    
    # Generate sample data
    data = generate_sample_data(1000)
    
    # Define strategy parameters
    parameters = {
        'fast_period': 10,
        'slow_period': 20
    }
    
    # Initialize Monte Carlo test
    mc_test = MonteCarloRobustnessTest(
        data=data,
        random_seed=42,
        noise_level=0.01,
        parameter_variance=0.1,
        enable_parallel=False  # Disable for demo
    )
    
    logger.info("Running Monte Carlo simulations...")
    
    # Run standard Monte Carlo test
    mc_result = mc_test.run_test(
        parameters=parameters,
        backtest_function=simple_backtest_function,
        n_simulations=200,
        test_type="data_noise"
    )
    
    logger.info(f"Monte Carlo Results:")
    logger.info(f"  Success Rate: {mc_result.success_rate:.2%}")
    logger.info(f"  Number of Simulations: {mc_result.n_simulations}")
    logger.info(f"  Best Case Return: {mc_result.best_case_scenario.get('total_return', 0):.2%}")
    logger.info(f"  Worst Case Return: {mc_result.worst_case_scenario.get('total_return', 0):.2%}")
    
    # Run trade sequence resampling test
    logger.info("\\nRunning trade sequence resampling test...")
    
    trade_seq_result = mc_test.run_trade_sequence_test(
        parameters=parameters,
        backtest_function=simple_backtest_function,
        n_simulations=100,
        block_size=20,
        preserve_correlations=True
    )
    
    logger.info(f"Trade Sequence Results:")
    logger.info(f"  Success Rate: {trade_seq_result.success_rate:.2%}")
    logger.info(f"  Enhanced Statistics Available: {len(trade_seq_result.summary_stats)}")
    
    # Run statistical significance test
    logger.info("\\nRunning statistical significance test...")
    
    stat_sig_result = mc_test.run_statistical_significance_test(
        parameters=parameters,
        backtest_function=simple_backtest_function,
        n_simulations=500,
        confidence_level=0.95
    )
    
    logger.info(f"Statistical Significance:")
    mc_stats = stat_sig_result['monte_carlo_result']
    stat_tests = stat_sig_result['statistical_tests']
    
    for metric, test_info in stat_tests.items():
        if 'zero_test' in test_info:
            p_value = test_info['zero_test']['p_value']
            significant = test_info['zero_test']['significant']
            logger.info(f"  {metric}: p-value={p_value:.4f}, significant={significant}")
    
    return mc_result, trade_seq_result, stat_sig_result


def demonstrate_data_perturbation_testing():
    """Demonstrate data perturbation testing."""
    logger.info("\\n=== Data Perturbation Testing ===")
    
    # Generate sample data
    data = generate_sample_data(800)
    
    # Define strategy parameters
    parameters = {
        'fast_period': 8,
        'slow_period': 21
    }
    
    # Initialize data perturbation test
    perturbation_test = DataPerturbationTest(
        data=data,
        random_seed=42
    )
    
    # Run comprehensive perturbation test
    logger.info("Running comprehensive data perturbation test...")
    
    perturbation_result = perturbation_test.run_test(
        parameters=parameters,
        backtest_function=simple_backtest_function,
        n_simulations=300,
        perturbation_types=['price_noise', 'volume_noise', 'missing_data', 'regime_change']
    )
    
    logger.info(f"Data Perturbation Results:")
    logger.info(f"  Success Rate: {perturbation_result.success_rate:.2%}")
    logger.info(f"  Number of Simulations: {perturbation_result.n_simulations}")
    
    # Run specific noise level testing
    logger.info("\\nRunning systematic price noise test...")
    
    noise_result = perturbation_test.run_price_noise_test(
        parameters=parameters,
        backtest_function=simple_backtest_function,
        noise_levels=[0.001, 0.005, 0.01, 0.02, 0.05],
        n_simulations_per_level=50,
        noise_type="gaussian"
    )
    
    logger.info(f"Price Noise Analysis:")
    sensitivity_analysis = noise_result['sensitivity_analysis']
    logger.info(f"  Noise Tolerance: {sensitivity_analysis.get('noise_tolerance', 0):.3f}")
    logger.info(f"  Critical Noise Level: {sensitivity_analysis.get('critical_noise_level', 0):.3f}")
    
    return perturbation_result, noise_result


def demonstrate_synthetic_data_testing():
    """Demonstrate synthetic data generation testing."""
    logger.info("\\n=== Synthetic Data Testing ===")
    
    # Generate sample data
    data = generate_sample_data(600)
    
    # Define strategy parameters
    parameters = {
        'fast_period': 12,
        'slow_period': 26
    }
    
    # Initialize synthetic data test
    synthetic_test = SyntheticDataTest(
        data=data,
        random_seed=42
    )
    
    # Run permutation test for overfitting detection
    logger.info("Running permutation test for overfitting detection...")
    
    permutation_result = synthetic_test.run_permutation_test(
        parameters=parameters,
        backtest_function=simple_backtest_function,
        n_simulations=500,
        preserve_properties=['returns_distribution']
    )
    
    significance_test = permutation_result['significance_test']
    logger.info(f"Permutation Test Results:")
    logger.info(f"  Baseline Return: {significance_test['baseline_return']:.3f}")
    logger.info(f"  Permutation Mean: {significance_test['permutation_mean']:.3f}")
    logger.info(f"  P-value: {significance_test['p_value']:.4f}")
    logger.info(f"  Significant at 5%: {significance_test['significant_at_05']}")
    logger.info(f"  Overfitting Detected: {significance_test['overfitting_detected']}")
    
    # Run comprehensive overfitting detection
    logger.info("\\nRunning comprehensive overfitting detection...")
    
    overfitting_result = synthetic_test.run_overfitting_detection(
        parameters=parameters,
        backtest_function=simple_backtest_function,
        detection_methods=['permutation', 'data_mining_bias'],
        n_simulations=400
    )
    
    logger.info(f"Overfitting Detection:")
    logger.info(f"  Overfitting Score: {overfitting_result['overfitting_score']:.3f}")
    logger.info(f"  Overfitting Detected: {overfitting_result['overfitting_detected']}")
    
    return permutation_result, overfitting_result


def demonstrate_stress_testing():
    """Demonstrate stress testing framework."""
    logger.info("\\n=== Stress Testing ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    # Define strategy parameters
    parameters = {
        'fast_period': 15,
        'slow_period': 30
    }
    
    # Initialize stress testing framework
    stress_test = StressTestFramework(
        data=data,
        random_seed=42
    )
    
    # Run comprehensive stress test
    logger.info("Running comprehensive stress testing...")
    
    stress_result = stress_test.run_test(
        parameters=parameters,
        backtest_function=simple_backtest_function,
        n_simulations=200,
        stress_scenarios=['flash_crash', 'high_volatility', 'liquidity_crisis']
    )
    
    logger.info(f"Stress Test Results:")
    logger.info(f"  Survival Rate: {stress_result.success_rate:.2%}")
    logger.info(f"  Number of Simulations: {stress_result.n_simulations}")
    
    # Run specific flash crash test
    logger.info("\\nRunning flash crash resilience test...")
    
    flash_crash_result = stress_test.run_flash_crash_test(
        parameters=parameters,
        backtest_function=simple_backtest_function,
        crash_magnitudes=[0.1, 0.2, 0.3],
        recovery_times=[5, 20, 50],
        n_simulations=30
    )
    
    resilience_analysis = flash_crash_result['resilience_analysis']
    logger.info(f"Flash Crash Resilience:")
    logger.info(f"  Crash Tolerance: {resilience_analysis.get('crash_tolerance', 0):.1%}")
    logger.info(f"  Overall Resilience: {resilience_analysis.get('overall_resilience', 0):.3f}")
    
    return stress_result, flash_crash_result


def demonstrate_robustness_metrics():
    """Demonstrate comprehensive robustness metrics and scoring."""
    logger.info("\\n=== Robustness Metrics and Scoring ===")
    
    # Generate sample results for demonstration
    sample_results = []
    for i in range(100):
        # Simulate various performance outcomes
        base_return = np.random.normal(0.05, 0.15)
        
        sample_results.append({
            'total_return': base_return,
            'sharpe_ratio': np.random.normal(1.2, 0.8),
            'max_drawdown': abs(np.random.normal(0.1, 0.05)),
        })
    
    # Calculate tail risk metrics
    logger.info("Calculating tail risk metrics...")
    
    tail_metrics = RobustnessMetrics.calculate_tail_risk_metrics(
        results=sample_results,
        metrics=['total_return', 'sharpe_ratio'],
        confidence_levels=[0.90, 0.95, 0.99]
    )
    
    logger.info(f"Tail Risk Metrics for total_return:")
    if 'total_return' in tail_metrics:
        tr_metrics = tail_metrics['total_return']
        logger.info(f"  VaR 95%: {tr_metrics.get('var_95%', 0):.3f}")
        logger.info(f"  CVaR 95%: {tr_metrics.get('cvar_95%', 0):.3f}")
        logger.info(f"  Maximum Loss: {tr_metrics.get('maximum_loss', 0):.3f}")
        logger.info(f"  Expected Tail Loss: {tr_metrics.get('expected_tail_loss', 0):.3f}")
    
    # Calculate stability metrics
    logger.info("\\nCalculating stability metrics...")
    
    stability_metrics = RobustnessMetrics.calculate_stability_metrics(
        results=sample_results,
        baseline_results={'total_return': 0.08, 'sharpe_ratio': 1.5}
    )
    
    logger.info(f"Stability Metrics:")
    logger.info(f"  Success Rate: {stability_metrics.get('success_rate', 0):.2%}")
    logger.info(f"  Consistency Score: {stability_metrics.get('consistency_score', 0):.3f}")
    logger.info(f"  Overall Stability Score: {stability_metrics.get('overall_stability_score', 0):.3f}")
    
    # Comprehensive robustness scoring
    logger.info("\\nCalculating comprehensive robustness score...")
    
    scorer = RobustnessScoring()
    
    # Simulate comprehensive robustness results
    robustness_results = {
        'stability_metrics': stability_metrics,
        'tail_risk_metrics': tail_metrics,
        'stress_results': {'success_rate': 0.75},
        'overfitting_metrics': {'overfitting_score': 0.3}
    }
    
    component_scores = scorer.calculate_comprehensive_score(robustness_results)
    
    logger.info(f"Component Scores:")
    for component, score in component_scores.items():
        if component != 'robustness_grade':
            logger.info(f"  {component}: {score:.3f}")
    
    logger.info(f"\\nOverall Robustness: {component_scores['overall_robustness_score']:.3f} ({component_scores['robustness_grade']})")
    
    # Generate comprehensive report
    logger.info("\\nGenerating comprehensive robustness report...")
    
    report = scorer.generate_robustness_report(component_scores, robustness_results)
    logger.info(f"\\n{report}")
    
    return tail_metrics, stability_metrics, component_scores


def main():
    """Main demonstration function."""
    logger.info("Starting Comprehensive Robustness Testing Framework Demonstration")
    logger.info("=" * 80)
    
    try:
        # Demonstrate each component
        mc_results = demonstrate_monte_carlo_testing()
        perturbation_results = demonstrate_data_perturbation_testing()
        synthetic_results = demonstrate_synthetic_data_testing()
        stress_results = demonstrate_stress_testing()
        metrics_results = demonstrate_robustness_metrics()
        
        logger.info("\\n" + "=" * 80)
        logger.info("DEMONSTRATION SUMMARY")
        logger.info("=" * 80)
        
        logger.info("✓ Monte Carlo Testing: Completed with statistical analysis")
        logger.info("✓ Data Perturbation Testing: Assessed strategy sensitivity to data quality")
        logger.info("✓ Synthetic Data Testing: Detected overfitting using permutation tests")
        logger.info("✓ Stress Testing: Evaluated resilience under extreme conditions")
        logger.info("✓ Robustness Metrics: Calculated comprehensive robustness scoring")
        
        logger.info("\\nThe robustness testing framework provides:")
        logger.info("• Statistical validation of strategy performance")
        logger.info("• Overfitting detection through multiple methods")
        logger.info("• Stress testing under extreme market conditions")
        logger.info("• Comprehensive robustness scoring and grading")
        logger.info("• Parallel execution support for large-scale testing")
        
        logger.info("\\nFramework is ready for integration with optimization algorithms")
        logger.info("for robust parameter selection and strategy validation.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()