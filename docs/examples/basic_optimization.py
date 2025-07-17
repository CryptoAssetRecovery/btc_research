#!/usr/bin/env python3
"""
Basic Optimization Example

This example demonstrates how to optimize a simple RSI strategy using
the new optimization framework with Bayesian optimization and validation.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from btc_research.optimization import (
    optimize_strategy,
    OptimizationFramework,
    BayesianOptimizer,
    WalkForwardValidator,
    MonteCarloRobustnessTest,
    ParameterSpec,
    ParameterType,
    OptimizationMetric,
)
from btc_research.optimization.integration import BacktestObjective
from tests.fixtures.sample_data import create_btc_sample_data


def main():
    """Run basic optimization example."""
    print("=" * 60)
    print("BTC Research - Basic Optimization Example")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n1. Creating sample market data...")
    market_data = create_btc_sample_data(
        periods=2000,  # ~3 months of hourly data
        freq="1h",
        seed=42
    )
    print(f"   Created {len(market_data)} data points")
    print(f"   Date range: {market_data.index[0]} to {market_data.index[-1]}")
    
    # Step 2: Define parameters to optimize
    print("\n2. Defining parameter search space...")
    parameter_specs = [
        ParameterSpec(
            name="rsi_period",
            param_type=ParameterType.INTEGER,
            low=10,
            high=30,
            description="RSI calculation period"
        ),
        ParameterSpec(
            name="rsi_oversold",
            param_type=ParameterType.FLOAT,
            low=20.0,
            high=40.0,
            description="RSI oversold threshold for entry"
        ),
        ParameterSpec(
            name="rsi_overbought",
            param_type=ParameterType.FLOAT,
            low=60.0,
            high=80.0,
            description="RSI overbought threshold for exit"
        ),
    ]
    
    for spec in parameter_specs:
        print(f"   {spec.name}: {spec.param_type.value} [{spec.low}, {spec.high}]")
    
    # Step 3: Create objective function
    print("\n3. Creating objective function...")
    
    def rsi_strategy_objective(params):
        """
        Objective function for RSI strategy optimization.
        
        This function simulates a simple RSI mean reversion strategy
        and returns the Sharpe ratio as the objective to maximize.
        """
        rsi_period = params["rsi_period"]
        rsi_oversold = params["rsi_oversold"]
        rsi_overbought = params["rsi_overbought"]
        
        # Calculate RSI
        def calculate_rsi(prices, period):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        # Get price data
        close_prices = market_data['close']
        rsi = calculate_rsi(close_prices, rsi_period)
        
        # Generate signals
        long_signals = (rsi < rsi_oversold).astype(int)
        exit_signals = (rsi > rsi_overbought).astype(int)
        
        # Calculate positions
        position = 0
        positions = []
        for i in range(len(market_data)):
            if long_signals.iloc[i] == 1 and position == 0:
                position = 1  # Enter long
            elif exit_signals.iloc[i] == 1 and position == 1:
                position = 0  # Exit long
            positions.append(position)
        
        positions = pd.Series(positions, index=market_data.index)
        
        # Calculate returns
        returns = close_prices.pct_change() * positions.shift(1)
        returns = returns.dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return -1.0  # Invalid strategy
        
        # Calculate Sharpe ratio (annualized)
        annual_return = returns.mean() * 24 * 365  # Hourly to annual
        annual_volatility = returns.std() * np.sqrt(24 * 365)
        
        if annual_volatility == 0:
            return -1.0
        
        sharpe_ratio = annual_return / annual_volatility
        
        # Add some noise to simulate realistic backtest variations
        noise = np.random.normal(0, 0.01)
        return sharpe_ratio + noise
    
    # Step 4: Run Bayesian optimization
    print("\n4. Running Bayesian optimization...")
    print("   This will intelligently search for the best parameters...")
    
    optimizer = BayesianOptimizer(
        parameter_specs=parameter_specs,
        objective_function=rsi_strategy_objective,
        metric=OptimizationMetric.SHARPE_RATIO,
        n_initial_points=10,  # Random exploration points
        random_seed=42
    )
    
    optimization_result = optimizer.optimize(
        max_iterations=50,
        convergence_threshold=0.01,
        timeout_seconds=300  # 5 minute timeout
    )
    
    print(f"\n   Optimization completed!")
    print(f"   Best parameters found: {optimization_result.best_parameters}")
    print(f"   Best Sharpe ratio: {optimization_result.best_score:.4f}")
    print(f"   Total evaluations: {optimization_result.total_evaluations}")
    print(f"   Optimization time: {optimization_result.optimization_time:.2f} seconds")
    
    # Step 5: Validate results with walk-forward analysis
    print("\n5. Validating with walk-forward analysis...")
    print("   This tests out-of-sample performance to avoid overfitting...")
    
    validator = WalkForwardValidator(
        data=market_data,
        window_size=500,  # ~3 weeks training window
        step_size=100,    # ~4 day step forward
        min_train_size=300
    )
    
    validation_result = validator.validate(
        parameters=optimization_result.best_parameters,
        backtest_function=lambda data, params: {
            "sharpe_ratio": rsi_strategy_objective(params)
        }
    )
    
    print(f"\n   Validation completed!")
    print(f"   Number of folds: {len(validation_result.fold_results)}")
    print(f"   Mean out-of-sample Sharpe: {validation_result.mean_metrics['sharpe_ratio']:.4f}")
    print(f"   Standard deviation: {validation_result.std_metrics['sharpe_ratio']:.4f}")
    
    # Calculate confidence interval
    ci_lower, ci_upper = validation_result.confidence_intervals['sharpe_ratio']
    print(f"   95% Confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Step 6: Test robustness
    print("\n6. Testing robustness with Monte Carlo simulation...")
    print("   This tests strategy stability under different market conditions...")
    
    robustness_test = MonteCarloRobustnessTest(
        data=market_data,
        perturbation_methods=["price_noise", "bootstrap"],
        noise_level=0.005,  # 0.5% price noise
        random_seed=42
    )
    
    robustness_result = robustness_test.run_test(
        parameters=optimization_result.best_parameters,
        backtest_function=lambda data, params: {
            "sharpe_ratio": rsi_strategy_objective(params)
        },
        n_simulations=100
    )
    
    print(f"\n   Robustness test completed!")
    print(f"   Mean robust Sharpe: {robustness_result.mean_metrics['sharpe_ratio']:.4f}")
    print(f"   Robustness std dev: {robustness_result.std_metrics['sharpe_ratio']:.4f}")
    print(f"   VaR 95%: {robustness_result.var_metrics['sharpe_ratio_0.95']:.4f}")
    print(f"   Expected Shortfall 95%: {robustness_result.es_metrics['sharpe_ratio_0.95']:.4f}")
    
    # Step 7: Compare with grid search
    print("\n7. Comparing with traditional grid search...")
    
    from btc_research.optimization.optimizers import GridSearchOptimizer
    
    # Use smaller search space for grid search demo
    grid_parameter_specs = [
        ParameterSpec("rsi_period", ParameterType.INTEGER, low=12, high=18, step=2),
        ParameterSpec("rsi_oversold", ParameterType.FLOAT, low=25.0, high=35.0, step=5.0),
        ParameterSpec("rsi_overbought", ParameterType.FLOAT, low=65.0, high=75.0, step=5.0),
    ]
    
    grid_optimizer = GridSearchOptimizer(
        parameter_specs=grid_parameter_specs,
        objective_function=rsi_strategy_objective,
        metric=OptimizationMetric.SHARPE_RATIO
    )
    
    grid_result = grid_optimizer.optimize()
    
    print(f"\n   Grid search completed!")
    print(f"   Best parameters: {grid_result.best_parameters}")
    print(f"   Best Sharpe ratio: {grid_result.best_score:.4f}")
    print(f"   Total evaluations: {grid_result.total_evaluations}")
    print(f"   Time: {grid_result.optimization_time:.2f} seconds")
    
    # Step 8: Summary and recommendations
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    
    print(f"\nMethod Comparison:")
    print(f"{'Method':<20} {'Best Score':<12} {'Evaluations':<12} {'Time (s)':<10}")
    print("-" * 54)
    print(f"{'Bayesian':<20} {optimization_result.best_score:<12.4f} {optimization_result.total_evaluations:<12} {optimization_result.optimization_time:<10.2f}")
    print(f"{'Grid Search':<20} {grid_result.best_score:<12.4f} {grid_result.total_evaluations:<12} {grid_result.optimization_time:<10.2f}")
    
    efficiency_gain = grid_result.total_evaluations / optimization_result.total_evaluations
    print(f"\nBayesian optimization is {efficiency_gain:.1f}x more efficient!")
    
    print(f"\nValidation Results:")
    print(f"In-sample Sharpe:     {optimization_result.best_score:.4f}")
    print(f"Out-of-sample Sharpe: {validation_result.mean_metrics['sharpe_ratio']:.4f}")
    
    overfitting_measure = (optimization_result.best_score - validation_result.mean_metrics['sharpe_ratio']) / optimization_result.best_score
    if overfitting_measure > 0.2:
        print(f"⚠️  Warning: Possible overfitting detected ({overfitting_measure:.1%} degradation)")
    else:
        print(f"✅ Good generalization ({overfitting_measure:.1%} degradation)")
    
    print(f"\nRobustness Assessment:")
    robustness_ratio = robustness_result.std_metrics['sharpe_ratio'] / robustness_result.mean_metrics['sharpe_ratio']
    if robustness_ratio < 0.3:
        print(f"✅ Strategy is robust (CV = {robustness_ratio:.2f})")
    elif robustness_ratio < 0.5:
        print(f"⚠️  Strategy is moderately robust (CV = {robustness_ratio:.2f})")
    else:
        print(f"❌ Strategy may be unstable (CV = {robustness_ratio:.2f})")
    
    print(f"\nRecommended Parameters:")
    for param_name, param_value in optimization_result.best_parameters.items():
        print(f"  {param_name}: {param_value}")
    
    print(f"\nNext Steps:")
    print("1. Test these parameters on more recent data")
    print("2. Implement risk management rules")
    print("3. Consider transaction costs and slippage")
    print("4. Monitor performance in live trading")
    
    return optimization_result


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        result = main()
        print("\n✅ Example completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)