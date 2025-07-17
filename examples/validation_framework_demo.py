#!/usr/bin/env python3
"""
Comprehensive demo of the BTC Research Validation Framework.

This example demonstrates how to use the enhanced validation framework
for robust strategy optimization and validation.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from btc_research.core.datafeed import DataFeed
from btc_research.core.engine import Engine
from btc_research.optimization.validators import (
    # Data splitting
    TimeSeriesDataSplitter, SplitConfig,
    
    # Validators
    WalkForwardValidator, WalkForwardConfig, WindowType,
    EnhancedTimeSeriesSplitValidator, SplitStrategy, TimeSeriesSplitConfig,
    PurgedCrossValidator,
    
    # Metrics and analysis
    ValidationMetricsCalculator, ValidationSummaryGenerator,
    StabilityAnalyzer, OverfittingDetector,
    
    # Integration
    BacktesterIntegrator, ValidationPipeline, ValidationBacktestFunction,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(periods: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for demonstration."""
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='1H')
    
    # Random walk with trend and volatility
    returns = np.random.normal(0.0001, 0.02, periods)  # Small positive drift with volatility
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    
    # Generate high/low with realistic spreads
    spread = np.random.uniform(0.001, 0.01, periods)
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + spread)
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - spread)
    
    # Generate volume
    data['volume'] = np.random.lognormal(10, 1, periods)
    
    return data


def create_sample_strategy_config() -> Dict[str, Any]:
    """Create sample strategy configuration."""
    return {
        "strategy": {
            "name": "SimpleMovingAverageCrossover",
            "parameters": {
                "fast_period": 10,
                "slow_period": 30,
                "stop_loss": 0.02,
                "take_profit": 0.04,
            }
        },
        "execution": {
            "initial_capital": 10000,
            "commission": 0.001,
            "slippage": 0.0005,
        }
    }


def demo_data_splitting():
    """Demonstrate time series data splitting utilities."""
    logger.info("=== Data Splitting Demo ===")
    
    # Create sample data
    data = create_sample_data(500)
    logger.info(f"Created sample data with {len(data)} samples")
    
    # Initialize data splitter
    splitter = TimeSeriesDataSplitter(random_seed=42)
    
    # Demo 1: Basic ratio-based splitting
    logger.info("\n1. Basic ratio-based splitting:")
    config = SplitConfig(
        train_ratio=0.6,
        validation_ratio=0.2, 
        test_ratio=0.2,
        gap_days=1  # 1 day gap between splits
    )
    
    split_result = splitter.split_by_ratio(data, config)
    summary = split_result.get_summary()
    
    logger.info(f"Split sizes: {summary['split_sizes']}")
    logger.info(f"Split ratios: {summary['split_ratios']}")
    logger.info(f"Data leakage detected: {summary['has_data_leakage']}")
    
    # Demo 2: Date-based splitting
    logger.info("\n2. Date-based splitting:")
    train_end = data.index[int(len(data) * 0.6)]
    val_end = data.index[int(len(data) * 0.8)]
    
    date_split = splitter.split_by_date(data, train_end, val_end, gap_days=1)
    date_summary = date_split.get_summary()
    logger.info(f"Date split sizes: {date_summary['split_sizes']}")
    
    # Demo 3: Walk-forward splits
    logger.info("\n3. Walk-forward splits:")
    wf_splits = splitter.create_walk_forward_splits(
        data=data,
        window_size_days=30,
        step_size_days=10,
        validation_days=10,
        gap_days=1
    )
    logger.info(f"Created {len(wf_splits)} walk-forward splits")
    
    return data


def demo_walk_forward_validation(data: pd.DataFrame):
    """Demonstrate enhanced walk-forward validation."""
    logger.info("\n=== Walk-Forward Validation Demo ===")
    
    # Demo 1: Rolling window validation
    logger.info("\n1. Rolling window validation:")
    rolling_config = WalkForwardConfig(
        window_type=WindowType.ROLLING,
        training_window_days=60,
        validation_window_days=20,
        step_size_days=15,
        gap_days=1,
        min_training_samples=100,
    )
    
    rolling_validator = WalkForwardValidator(data, rolling_config)
    split_info = rolling_validator.get_split_info()
    logger.info(f"Rolling validation: {split_info['total_splits']} splits created")
    
    # Demo 2: Expanding window validation
    logger.info("\n2. Expanding window validation:")
    expanding_config = WalkForwardConfig(
        window_type=WindowType.EXPANDING,
        training_window_days=60,  # Initial window size
        validation_window_days=20,
        step_size_days=15,
        gap_days=1,
    )
    
    expanding_validator = WalkForwardValidator(data, expanding_config)
    expanding_info = expanding_validator.get_split_info()
    logger.info(f"Expanding validation: {expanding_info['total_splits']} splits created")
    
    return rolling_validator, expanding_validator


def demo_enhanced_time_series_validation(data: pd.DataFrame):
    """Demonstrate enhanced time series split validation."""
    logger.info("\n=== Enhanced Time Series Validation Demo ===")
    
    # Demo 1: Purged cross-validation
    logger.info("\n1. Purged cross-validation:")
    purged_config = TimeSeriesSplitConfig(
        strategy=SplitStrategy.PURGED,
        n_splits=5,
        test_size_ratio=0.2,
        purge_pct=0.02,
        embargo_pct=0.01,
    )
    
    purged_validator = EnhancedTimeSeriesSplitValidator(data, purged_config)
    purged_splits = purged_validator.split_data()
    logger.info(f"Purged CV: {len(purged_splits)} splits created")
    
    # Demo 2: Combinatorial purged CV
    logger.info("\n2. Combinatorial purged cross-validation:")
    combinatorial_config = TimeSeriesSplitConfig(
        strategy=SplitStrategy.COMBINATORIAL,
        n_splits=4,
        test_size_ratio=0.15,
        purge_pct=0.01,
        embargo_pct=0.01,
        enable_combinatorial=True,
        max_combinations=6,
    )
    
    combinatorial_validator = EnhancedTimeSeriesSplitValidator(data, combinatorial_config)
    combinatorial_splits = combinatorial_validator.split_data()
    logger.info(f"Combinatorial CV: {len(combinatorial_splits)} splits created")
    
    return purged_validator, combinatorial_validator


def create_mock_backtest_function():
    """Create a mock backtest function for demonstration."""
    def mock_backtest(data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Mock backtest function that generates realistic metrics."""
        # Simulate strategy performance based on parameters
        fast_period = parameters.get('fast_period', 10)
        slow_period = parameters.get('slow_period', 30)
        
        # Simple performance simulation (in reality, this would run actual backtest)
        period_ratio = fast_period / slow_period
        base_return = 0.1 * period_ratio  # Base return based on period ratio
        
        # Add some noise and market regime effects
        market_factor = np.random.normal(0, 0.05)  # Market noise
        data_length_factor = min(len(data) / 1000, 1.0)  # More data = better performance
        
        total_return = base_return + market_factor + data_length_factor * 0.02
        
        # Generate correlated metrics
        volatility = max(0.01, abs(market_factor) * 2 + 0.1)
        sharpe_ratio = total_return / volatility if volatility > 0 else 0
        max_drawdown = min(-0.01, total_return * -0.3 + np.random.normal(0, 0.02))
        win_rate = min(0.8, max(0.3, 0.6 + total_return))
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': volatility,
            'profit_factor': max(0.5, 1.0 + total_return),
            'num_trades': int(len(data) / 10),
        }
    
    return mock_backtest


def demo_validation_metrics():
    """Demonstrate validation metrics and analysis."""
    logger.info("\n=== Validation Metrics Demo ===")
    
    # Create sample validation results (normally from actual validation)
    fold_results = []
    np.random.seed(42)
    
    for i in range(8):  # 8 folds
        # Simulate decreasing performance to show degradation detection
        degradation_factor = 1 - i * 0.05
        noise = np.random.normal(0, 0.02)
        
        fold_results.append({
            'fold': i,
            'total_return': 0.12 * degradation_factor + noise,
            'sharpe_ratio': 1.5 * degradation_factor + noise * 0.5,
            'max_drawdown': -0.08 * (1 / degradation_factor) + noise,
            'win_rate': 0.65 * degradation_factor + abs(noise) * 0.1,
            'train_samples': 500,
            'val_samples': 100,
        })
    
    # Create mock validation result
    from btc_research.optimization.types import ValidationResult, ValidationMethod
    
    # Calculate basic statistics
    mean_metrics = {}
    std_metrics = {}
    confidence_intervals = {}
    
    for key in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
        values = [r[key] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        mean_metrics[key] = mean_val
        std_metrics[key] = std_val
        confidence_intervals[key] = (mean_val - 1.96 * std_val, mean_val + 1.96 * std_val)
    
    validation_result = ValidationResult(
        method=ValidationMethod.WALK_FORWARD,
        fold_results=fold_results,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        confidence_intervals=confidence_intervals,
        n_splits=len(fold_results),
        data_split_info={}
    )
    
    # Demo validation metrics
    metrics_calculator = ValidationMetricsCalculator()
    comprehensive_metrics = metrics_calculator.calculate_comprehensive_metrics(validation_result)
    
    logger.info("\n1. Basic Statistics:")
    basic_stats = comprehensive_metrics['basic_statistics']
    if 'total_return' in basic_stats:
        return_stats = basic_stats['total_return']
        logger.info(f"Total Return - Mean: {return_stats['mean']:.4f}, Std: {return_stats['std']:.4f}")
    
    logger.info("\n2. Stability Metrics:")
    stability_metrics = comprehensive_metrics['stability_metrics']
    if 'total_return' in stability_metrics:
        return_stability = stability_metrics['total_return']
        logger.info(f"Coefficient of Variation: {return_stability['coefficient_of_variation']:.4f}")
        logger.info(f"Stability Score: {return_stability['stability_score']:.4f}")
    
    # Demo stability analysis
    logger.info("\n3. Stability Analysis:")
    stability_analyzer = StabilityAnalyzer()
    stability_analysis = stability_analyzer.analyze_stability(validation_result)
    logger.info(f"Stability Class: {stability_analysis['stability_class']}")
    logger.info(f"Stability Trend: {stability_analysis['stability_trend']}")
    
    # Demo overfitting detection
    logger.info("\n4. Overfitting Detection:")
    overfitting_detector = OverfittingDetector()
    overfitting_analysis = overfitting_detector.detect_overfitting(validation_result)
    logger.info(f"Overfitting Risk: {overfitting_analysis['overfitting_risk']}")
    logger.info(f"Risk Indicators: {len(overfitting_analysis['indicators'])}")
    
    # Demo comprehensive summary
    logger.info("\n5. Comprehensive Summary:")
    summary_generator = ValidationSummaryGenerator()
    comprehensive_summary = summary_generator.generate_comprehensive_summary(validation_result)
    
    overall_score = comprehensive_summary['overall_validation_score']
    logger.info(f"Overall Validation Score: {overall_score['overall_score']:.3f}")
    logger.info(f"Classification: {overall_score['classification']}")
    
    return validation_result


def demo_integration_pipeline(data: pd.DataFrame):
    """Demonstrate the complete validation pipeline with integration."""
    logger.info("\n=== Integration Pipeline Demo ===")
    
    # Create strategy configuration
    strategy_config = create_sample_strategy_config()
    parameters = strategy_config["strategy"]["parameters"]
    
    # Create mock components (in real usage, these would be actual instances)
    mock_backtest_function = create_mock_backtest_function()
    
    # Demo 1: Direct validation with backtest function
    logger.info("\n1. Direct validation with backtest function:")
    
    validator = WalkForwardValidator(
        data, 
        WalkForwardConfig(
            window_type=WindowType.ROLLING,
            training_window_days=30,
            validation_window_days=10,
            step_size_days=10,
        )
    )
    
    validation_result = validator.validate(parameters, mock_backtest_function)
    logger.info(f"Validation completed with {validation_result.n_splits} folds")
    logger.info(f"Mean total return: {validation_result.mean_metrics.get('total_return', 0):.4f}")
    logger.info(f"Stability score: {validation_result.stability_score:.4f}")
    
    # Demo 2: Using ValidationBacktestFunction wrapper
    logger.info("\n2. Using ValidationBacktestFunction wrapper:")
    
    backtest_wrapper = ValidationBacktestFunction(
        strategy_config=strategy_config,
        use_existing_indicators=False  # Simplified for demo
    )
    
    # Replace the __call__ method with our mock function for demo
    backtest_wrapper.__call__ = lambda data, params: mock_backtest_function(data, params)
    
    # Test parameter variations
    param_variations = [
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 10, "slow_period": 30},
        {"fast_period": 15, "slow_period": 40},
    ]
    
    logger.info("\n3. Testing parameter variations:")
    for i, params in enumerate(param_variations):
        result = backtest_wrapper(data, params)
        logger.info(f"Params {i+1}: Return={result['total_return']:.4f}, "
                   f"Sharpe={result['sharpe_ratio']:.4f}")
    
    return validation_result


def main():
    """Run the complete validation framework demonstration."""
    logger.info("Starting BTC Research Validation Framework Demo")
    logger.info("=" * 60)
    
    try:
        # Demo 1: Data splitting utilities
        data = demo_data_splitting()
        
        # Demo 2: Walk-forward validation
        rolling_validator, expanding_validator = demo_walk_forward_validation(data)
        
        # Demo 3: Enhanced time series validation
        purged_validator, combinatorial_validator = demo_enhanced_time_series_validation(data)
        
        # Demo 4: Validation metrics and analysis
        validation_result = demo_validation_metrics()
        
        # Demo 5: Integration pipeline
        pipeline_result = demo_integration_pipeline(data)
        
        logger.info("\n" + "=" * 60)
        logger.info("Demo completed successfully!")
        logger.info("Key takeaways:")
        logger.info("1. Data splitting utilities prevent temporal leakage")
        logger.info("2. Multiple validation strategies provide robust assessment")
        logger.info("3. Comprehensive metrics detect overfitting and instability")
        logger.info("4. Integration utilities simplify workflow implementation")
        logger.info("5. Thread-safe design enables parallel optimization")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)