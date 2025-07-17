"""
End-to-end integration tests for optimization framework.

Tests complete optimization workflows including integration with
backtesting system, CLI interface, and real-world scenarios.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import tempfile
import yaml
import os

from btc_research.optimization import (
    optimize_strategy,
    OptimizationFramework,
    BacktestObjective,
    BayesianOptimizer,
    WalkForwardValidator,
    MonteCarloRobustnessTest,
)
from btc_research.optimization.types import (
    ParameterSpec,
    ParameterType,
    OptimizationMetric,
)
from tests.fixtures.sample_data import create_btc_sample_data, SAMPLE_CONFIGS


class TestEndToEndOptimization(unittest.TestCase):
    """Test complete optimization workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = create_btc_sample_data(periods=1000, freq="1h")
        
        # Parameter specifications for testing
        self.parameter_specs = [
            ParameterSpec("rsi_period", ParameterType.INTEGER, low=10, high=30),
            ParameterSpec("rsi_oversold", ParameterType.FLOAT, low=20.0, high=40.0),
            ParameterSpec("rsi_overbought", ParameterType.FLOAT, low=60.0, high=80.0),
        ]
        
        # Mock backtester that returns realistic results
        self.mock_backtest_results = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "num_trades": 45
        }
    
    @patch('btc_research.optimization.integration.Backtester')
    @patch('btc_research.optimization.integration.DataFeed')
    def test_optimize_strategy_basic(self, mock_datafeed_class, mock_backtester_class):
        """Test basic strategy optimization workflow."""
        # Mock data feed
        mock_datafeed = Mock()
        mock_datafeed.get_data.return_value = self.data
        mock_datafeed_class.return_value = mock_datafeed
        
        # Mock backtester
        mock_backtester = Mock()
        mock_backtester.run.return_value = self.mock_backtest_results
        mock_backtester_class.return_value = mock_backtester
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(SAMPLE_CONFIGS["simple_ema"], f)
            config_path = f.name
        
        try:
            result = optimize_strategy(
                config_path=config_path,
                parameter_specs=self.parameter_specs,
                optimizer=BayesianOptimizer,
                validator=WalkForwardValidator,
                max_iterations=10
            )
            
            # Verify result structure
            self.assertIsNotNone(result.best_parameters)
            self.assertIsNotNone(result.best_score)
            self.assertEqual(result.total_evaluations, 10)
            self.assertGreater(result.optimization_time, 0)
            
            # Check that parameters are within bounds
            for spec in self.parameter_specs:
                param_value = result.best_parameters[spec.name]
                if spec.param_type == ParameterType.INTEGER:
                    self.assertGreaterEqual(param_value, spec.low)
                    self.assertLessEqual(param_value, spec.high)
                elif spec.param_type == ParameterType.FLOAT:
                    self.assertGreaterEqual(param_value, spec.low)
                    self.assertLessEqual(param_value, spec.high)
            
        finally:
            os.unlink(config_path)
    
    @patch('btc_research.optimization.integration.Backtester')
    @patch('btc_research.optimization.integration.DataFeed')
    def test_optimization_framework_complete_workflow(self, mock_datafeed_class, mock_backtester_class):
        """Test complete optimization framework workflow."""
        # Setup mocks
        mock_datafeed = Mock()
        mock_datafeed.get_data.return_value = self.data
        mock_datafeed_class.return_value = mock_datafeed
        
        mock_backtester = Mock()
        mock_backtester.run.return_value = self.mock_backtest_results
        mock_backtester_class.return_value = mock_backtester
        
        # Create optimization framework
        framework = OptimizationFramework(
            data=self.data,
            optimizer_class=BayesianOptimizer,
            validator_class=WalkForwardValidator,
            robustness_test_class=MonteCarloRobustnessTest
        )
        
        # Run optimization
        optimization_result = framework.optimize(
            parameter_specs=self.parameter_specs,
            max_iterations=5,
            metric=OptimizationMetric.SHARPE_RATIO
        )
        
        # Run validation
        validation_result = framework.validate(
            parameters=optimization_result.best_parameters,
            backtest_function=lambda data, params: self.mock_backtest_results
        )
        
        # Run robustness test
        robustness_result = framework.test_robustness(
            parameters=optimization_result.best_parameters,
            backtest_function=lambda data, params: self.mock_backtest_results,
            n_simulations=10
        )
        
        # Verify all results
        self.assertIsNotNone(optimization_result)
        self.assertIsNotNone(validation_result)
        self.assertIsNotNone(robustness_result)
        
        # Check optimization result
        self.assertEqual(optimization_result.total_evaluations, 5)
        self.assertIn("algorithm", optimization_result.metadata)
        
        # Check validation result
        self.assertGreater(len(validation_result.fold_results), 0)
        self.assertIn("sharpe_ratio", validation_result.mean_metrics)
        
        # Check robustness result
        self.assertEqual(len(robustness_result.simulation_results), 10)
        self.assertIn("sharpe_ratio", robustness_result.mean_metrics)
    
    def test_backtest_objective_function(self):
        """Test BacktestObjective function integration."""
        # Mock config
        config = SAMPLE_CONFIGS["rsi_mean_reversion"].copy()
        
        # Create BacktestObjective
        objective = BacktestObjective(
            config=config,
            data=self.data,
            metric=OptimizationMetric.SHARPE_RATIO
        )
        
        # Test parameter evaluation
        with patch('btc_research.optimization.integration.Backtester') as mock_backtester_class:
            mock_backtester = Mock()
            mock_backtester.run.return_value = self.mock_backtest_results
            mock_backtester_class.return_value = mock_backtester
            
            parameters = {"rsi_period": 14, "rsi_oversold": 30}
            score = objective(parameters)
            
            self.assertIsInstance(score, float)
            self.assertEqual(score, self.mock_backtest_results["sharpe_ratio"])
            
            # Verify backtester was called with updated config
            mock_backtester.run.assert_called_once()
    
    @patch('btc_research.optimization.integration.Backtester')
    def test_parameter_injection_into_config(self, mock_backtester_class):
        """Test that parameters are correctly injected into strategy config."""
        # Setup mock
        mock_backtester = Mock()
        mock_backtester.run.return_value = self.mock_backtest_results
        mock_backtester_class.return_value = mock_backtester
        
        config = SAMPLE_CONFIGS["rsi_mean_reversion"].copy()
        objective = BacktestObjective(config=config, data=self.data)
        
        # Call with specific parameters
        parameters = {"rsi_period": 21, "rsi_oversold": 25}
        objective(parameters)
        
        # Get the config that was passed to backtester
        call_args = mock_backtester_class.call_args
        updated_config = call_args[1]['config']
        
        # Check that parameters were injected correctly
        # This would depend on the specific parameter mapping logic
        self.assertIn("indicators", updated_config)
    
    def test_optimization_with_multiple_metrics(self):
        """Test optimization considering multiple performance metrics."""
        with patch('btc_research.optimization.integration.Backtester') as mock_backtester_class:
            mock_backtester = Mock()
            
            # Return different results for different parameter combinations
            def varying_backtest(*args, **kwargs):
                # Simulate parameter sensitivity
                config = kwargs.get('config', {})
                base_results = self.mock_backtest_results.copy()
                
                # Add some parameter-dependent variation
                if 'rsi_period' in str(config):
                    variation = np.random.normal(0, 0.1)
                    base_results["sharpe_ratio"] += variation
                    base_results["total_return"] += variation * 0.5
                
                return base_results
            
            mock_backtester.run.side_effect = varying_backtest
            mock_backtester_class.return_value = mock_backtester
            
            # Test optimization with different metrics
            metrics_to_test = [
                OptimizationMetric.SHARPE_RATIO,
                OptimizationMetric.TOTAL_RETURN,
                OptimizationMetric.PROFIT_FACTOR
            ]
            
            results = {}
            for metric in metrics_to_test:
                objective = BacktestObjective(
                    config=SAMPLE_CONFIGS["rsi_mean_reversion"],
                    data=self.data,
                    metric=metric
                )
                
                optimizer = BayesianOptimizer(
                    parameter_specs=self.parameter_specs[:2],  # Use fewer params for speed
                    objective_function=objective,
                    metric=metric,
                    n_initial_points=3,
                    random_seed=42
                )
                
                result = optimizer.optimize(max_iterations=5)
                results[metric] = result
            
            # Results should vary based on optimization metric
            self.assertEqual(len(results), len(metrics_to_test))
            
            for metric, result in results.items():
                self.assertIsNotNone(result.best_parameters)
                self.assertEqual(result.metadata["optimization_metric"], metric.value)
    
    def test_optimization_with_constraints(self):
        """Test optimization with parameter constraints and relationships."""
        # Create parameters with interdependencies
        constrained_specs = [
            ParameterSpec("rsi_oversold", ParameterType.FLOAT, low=20.0, high=40.0),
            ParameterSpec("rsi_overbought", ParameterType.FLOAT, low=60.0, high=80.0),
        ]
        
        def constrained_objective(params):
            # Add constraint: overbought > oversold + 20
            if params["rsi_overbought"] <= params["rsi_oversold"] + 20:
                return -1000  # Heavy penalty for constraint violation
            
            # Normal evaluation
            return self.mock_backtest_results["sharpe_ratio"]
        
        optimizer = BayesianOptimizer(
            parameter_specs=constrained_specs,
            objective_function=constrained_objective,
            metric=OptimizationMetric.SHARPE_RATIO,
            random_seed=42
        )
        
        result = optimizer.optimize(max_iterations=10)
        
        # Check that final result respects constraints
        params = result.best_parameters
        self.assertGreater(
            params["rsi_overbought"], 
            params["rsi_oversold"] + 20
        )
    
    def test_optimization_convergence_detection(self):
        """Test that optimization detects convergence appropriately."""
        def converging_objective(params):
            # Function that converges to specific value
            target_rsi = 14
            penalty = abs(params["rsi_period"] - target_rsi) * 0.1
            return 1.0 - penalty
        
        optimizer = BayesianOptimizer(
            parameter_specs=[self.parameter_specs[0]],  # Just RSI period
            objective_function=converging_objective,
            metric=OptimizationMetric.SHARPE_RATIO,
            random_seed=42
        )
        
        result = optimizer.optimize(
            max_iterations=50,
            convergence_threshold=0.01
        )
        
        # Should converge early
        self.assertLess(result.total_evaluations, 50)
        self.assertAlmostEqual(result.best_parameters["rsi_period"], 14, delta=2)
    
    def test_optimization_timeout_handling(self):
        """Test optimization timeout behavior."""
        def slow_objective(params):
            import time
            time.sleep(0.01)  # Small delay
            return 0.5
        
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs[:1],
            objective_function=slow_objective,
            metric=OptimizationMetric.SHARPE_RATIO
        )
        
        result = optimizer.optimize(
            max_iterations=100,
            timeout_seconds=0.1  # Very short timeout
        )
        
        # Should timeout before completing all iterations
        self.assertLess(result.total_evaluations, 100)
        self.assertLess(result.optimization_time, 0.2)
    
    def test_error_recovery_and_robustness(self):
        """Test that optimization handles errors gracefully."""
        def unreliable_objective(params):
            # Fail randomly 20% of the time
            if np.random.random() < 0.2:
                raise ValueError("Simulated evaluation failure")
            return params["rsi_period"] * 0.01
        
        optimizer = BayesianOptimizer(
            parameter_specs=[self.parameter_specs[0]],
            objective_function=unreliable_objective,
            metric=OptimizationMetric.SHARPE_RATIO,
            random_seed=42
        )
        
        result = optimizer.optimize(max_iterations=20)
        
        # Should complete despite some failures
        self.assertIsNotNone(result.best_parameters)
        self.assertGreater(result.total_evaluations, 0)
        # May have fewer evaluations due to failures
        self.assertLessEqual(result.total_evaluations, 20)
    
    def test_reproducibility_across_runs(self):
        """Test that optimization results are reproducible."""
        def deterministic_objective(params):
            return params["rsi_period"] * 0.01 + params["rsi_oversold"] * 0.005
        
        # Run optimization twice with same seed
        results = []
        for _ in range(2):
            optimizer = BayesianOptimizer(
                parameter_specs=self.parameter_specs[:2],
                objective_function=deterministic_objective,
                metric=OptimizationMetric.SHARPE_RATIO,
                n_initial_points=3,
                random_seed=42
            )
            
            result = optimizer.optimize(max_iterations=5)
            results.append(result)
        
        # Results should be identical
        result1, result2 = results
        self.assertEqual(result1.total_evaluations, result2.total_evaluations)
        
        # Parameters should be very close
        for param_name in result1.best_parameters:
            self.assertAlmostEqual(
                result1.best_parameters[param_name],
                result2.best_parameters[param_name],
                places=4
            )


class TestOptimizationFrameworkIntegration(unittest.TestCase):
    """Test OptimizationFramework class integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = create_btc_sample_data(periods=500, freq="1h")
        self.parameter_specs = [
            ParameterSpec("param1", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("param2", ParameterType.INTEGER, low=1, high=20),
        ]
    
    def test_framework_initialization(self):
        """Test OptimizationFramework initialization."""
        framework = OptimizationFramework(
            data=self.data,
            optimizer_class=BayesianOptimizer,
            validator_class=WalkForwardValidator,
            robustness_test_class=MonteCarloRobustnessTest
        )
        
        self.assertTrue(framework.data.equals(self.data))
        self.assertEqual(framework.optimizer_class, BayesianOptimizer)
        self.assertEqual(framework.validator_class, WalkForwardValidator)
        self.assertEqual(framework.robustness_test_class, MonteCarloRobustnessTest)
    
    def test_framework_component_creation(self):
        """Test that framework creates components correctly."""
        framework = OptimizationFramework(
            data=self.data,
            optimizer_class=BayesianOptimizer,
            validator_class=WalkForwardValidator
        )
        
        def mock_objective(params):
            return sum(params.values())
        
        # Create optimizer
        optimizer = framework._create_optimizer(
            parameter_specs=self.parameter_specs,
            objective_function=mock_objective,
            metric=OptimizationMetric.SHARPE_RATIO
        )
        
        self.assertIsInstance(optimizer, BayesianOptimizer)
        self.assertEqual(optimizer.parameter_specs, self.parameter_specs)
        
        # Create validator
        validator = framework._create_validator()
        self.assertIsInstance(validator, WalkForwardValidator)
        self.assertTrue(validator.data.equals(self.data))


if __name__ == "__main__":
    unittest.main()