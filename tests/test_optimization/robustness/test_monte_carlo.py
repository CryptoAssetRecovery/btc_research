"""
Unit tests for MonteCarloRobustnessTest.

Tests the Monte Carlo robustness testing including data perturbation,
bootstrap sampling, and statistical risk metrics.
"""

import unittest
from unittest.mock import Mock
import pandas as pd
import numpy as np

from btc_research.optimization.robustness.monte_carlo import MonteCarloRobustnessTest
from btc_research.optimization.types import RobustnessResult
from tests.fixtures.sample_data import create_btc_sample_data


class TestMonteCarloRobustnessTest(unittest.TestCase):
    """Test cases for MonteCarloRobustnessTest."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = create_btc_sample_data(periods=500, freq="1h")
        
        # Mock backtest function with some variance
        def mock_backtest(data, params):
            base_return = params.get("param1", 0) * 0.01
            noise = np.random.normal(0, 0.05)  # Add noise for variation
            return {
                "total_return": base_return + noise,
                "sharpe_ratio": (base_return + noise) / 0.15,
                "max_drawdown": -abs(np.random.uniform(0.02, 0.1))
            }
        
        self.backtest_function = mock_backtest
        self.parameters = {"param1": 10, "param2": 0.5}
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        test = MonteCarloRobustnessTest(self.data)
        
        self.assertTrue(test.data.equals(self.data))
        self.assertEqual(test.perturbation_methods, ["price_noise", "bootstrap"])
        self.assertEqual(test.noise_level, 0.01)
        self.assertIsNone(test.random_seed)
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        test = MonteCarloRobustnessTest(
            data=self.data,
            perturbation_methods=["price_noise", "volume_noise"],
            noise_level=0.02,
            random_seed=42
        )
        
        self.assertEqual(test.perturbation_methods, ["price_noise", "volume_noise"])
        self.assertEqual(test.noise_level, 0.02)
        self.assertEqual(test.random_seed, 42)
    
    def test_price_noise_perturbation(self):
        """Test price noise perturbation method."""
        test = MonteCarloRobustnessTest(
            self.data, 
            perturbation_methods=["price_noise"],
            noise_level=0.01,
            random_seed=42
        )
        
        original_data = self.data.copy()
        perturbed_data = test._apply_price_noise(original_data)
        
        # Should maintain same structure
        self.assertEqual(len(perturbed_data), len(original_data))
        self.assertEqual(list(perturbed_data.columns), list(original_data.columns))
        
        # Prices should be slightly different
        for col in ['open', 'high', 'low', 'close']:
            if col in original_data.columns:
                self.assertFalse(np.array_equal(original_data[col], perturbed_data[col]))
                # Should be within reasonable bounds
                relative_change = abs(perturbed_data[col] / original_data[col] - 1)
                self.assertLess(relative_change.max(), 0.05)  # Max 5% change
    
    def test_bootstrap_perturbation(self):
        """Test bootstrap resampling perturbation."""
        test = MonteCarloRobustnessTest(
            self.data,
            perturbation_methods=["bootstrap"],
            random_seed=42
        )
        
        original_data = self.data.copy()
        perturbed_data = test._apply_bootstrap(original_data)
        
        # Should maintain same length
        self.assertEqual(len(perturbed_data), len(original_data))
        self.assertEqual(list(perturbed_data.columns), list(original_data.columns))
        
        # Data should be resampled (some rows duplicated, others missing)
        # This is hard to test directly, so we check statistical properties
        original_mean = original_data['close'].mean()
        perturbed_mean = perturbed_data['close'].mean()
        # Means should be reasonably close
        self.assertLess(abs(original_mean - perturbed_mean) / original_mean, 0.1)
    
    def test_volume_noise_perturbation(self):
        """Test volume noise perturbation method."""
        test = MonteCarloRobustnessTest(
            self.data,
            perturbation_methods=["volume_noise"],
            noise_level=0.02,
            random_seed=42
        )
        
        original_data = self.data.copy()
        perturbed_data = test._apply_volume_noise(original_data)
        
        # Volume should be modified, prices unchanged
        if 'volume' in original_data.columns:
            self.assertFalse(np.array_equal(original_data['volume'], perturbed_data['volume']))
            # Prices should remain the same
            for col in ['open', 'high', 'low', 'close']:
                if col in original_data.columns:
                    np.testing.assert_array_equal(original_data[col], perturbed_data[col])
    
    def test_run_test_basic(self):
        """Test basic robustness test execution."""
        test = MonteCarloRobustnessTest(
            self.data,
            perturbation_methods=["price_noise"],
            random_seed=42
        )
        
        result = test.run_test(
            parameters=self.parameters,
            backtest_function=self.backtest_function,
            n_simulations=10
        )
        
        self.assertIsInstance(result, RobustnessResult)
        self.assertEqual(result.parameters, self.parameters)
        self.assertEqual(len(result.simulation_results), 10)
        
        # Check that statistics were calculated
        self.assertIn("total_return", result.mean_metrics)
        self.assertIn("sharpe_ratio", result.mean_metrics)
        self.assertIn("total_return", result.std_metrics)
        self.assertIn("sharpe_ratio", result.std_metrics)
        
        # Check VaR and ES metrics
        self.assertIn("total_return_0.95", result.var_metrics)
        self.assertIn("total_return_0.95", result.es_metrics)
    
    def test_run_test_multiple_perturbations(self):
        """Test robustness test with multiple perturbation methods."""
        test = MonteCarloRobustnessTest(
            self.data,
            perturbation_methods=["price_noise", "bootstrap", "volume_noise"],
            random_seed=42
        )
        
        result = test.run_test(
            parameters=self.parameters,
            backtest_function=self.backtest_function,
            n_simulations=15
        )
        
        self.assertEqual(len(result.simulation_results), 15)
        
        # Should have metadata about perturbation methods
        self.assertIn("perturbation_methods", result.metadata)
        self.assertEqual(
            result.metadata["perturbation_methods"], 
            ["price_noise", "bootstrap", "volume_noise"]
        )
    
    def test_statistical_metrics_calculation(self):
        """Test calculation of statistical risk metrics."""
        test = MonteCarloRobustnessTest(self.data, random_seed=42)
        
        # Create deterministic simulation results for testing
        simulation_results = [
            {"return": 0.1, "sharpe": 1.0},
            {"return": 0.05, "sharpe": 0.5},
            {"return": 0.15, "sharpe": 1.5},
            {"return": -0.02, "sharpe": -0.2},
            {"return": 0.08, "sharpe": 0.8}
        ]
        
        var_metrics, es_metrics = test._calculate_risk_metrics(simulation_results)
        
        # Check VaR calculation (5th percentile for 95% confidence)
        returns = [r["return"] for r in simulation_results]
        returns_sorted = sorted(returns)
        expected_var_95 = returns_sorted[0]  # Worst return for small sample
        
        self.assertIn("return_0.95", var_metrics)
        self.assertEqual(var_metrics["return_0.95"], expected_var_95)
        
        # ES should be even worse than VaR
        self.assertIn("return_0.95", es_metrics)
        self.assertLessEqual(es_metrics["return_0.95"], var_metrics["return_0.95"])
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        test1 = MonteCarloRobustnessTest(
            self.data,
            perturbation_methods=["price_noise"],
            random_seed=42
        )
        
        test2 = MonteCarloRobustnessTest(
            self.data,
            perturbation_methods=["price_noise"],
            random_seed=42
        )
        
        # Use deterministic backtest for comparison
        def deterministic_backtest(data, params):
            return {"return": data['close'].mean() * 0.001}
        
        result1 = test1.run_test(
            parameters=self.parameters,
            backtest_function=deterministic_backtest,
            n_simulations=5
        )
        
        result2 = test2.run_test(
            parameters=self.parameters,
            backtest_function=deterministic_backtest,
            n_simulations=5
        )
        
        # Results should be very similar (allowing for small numerical differences)
        self.assertAlmostEqual(
            result1.mean_metrics["return"],
            result2.mean_metrics["return"],
            places=4
        )
    
    def test_error_handling(self):
        """Test handling of backtest function errors."""
        def failing_backtest(data, params):
            if np.random.random() < 0.3:  # Fail 30% of the time
                raise ValueError("Simulated backtest failure")
            return {"return": 0.05, "sharpe": 0.5}
        
        test = MonteCarloRobustnessTest(self.data, random_seed=42)
        
        result = test.run_test(
            parameters=self.parameters,
            backtest_function=failing_backtest,
            n_simulations=10
        )
        
        # Should complete despite some failures
        self.assertIsInstance(result, RobustnessResult)
        # Might have fewer successful simulations due to failures
        self.assertGreater(len(result.simulation_results), 0)
        self.assertLessEqual(len(result.simulation_results), 10)
    
    def test_performance_large_simulations(self):
        """Test performance with larger number of simulations."""
        test = MonteCarloRobustnessTest(
            self.data,
            perturbation_methods=["price_noise"],
            random_seed=42
        )
        
        import time
        start_time = time.time()
        
        result = test.run_test(
            parameters=self.parameters,
            backtest_function=self.backtest_function,
            n_simulations=100
        )
        
        elapsed_time = time.time() - start_time
        
        self.assertIsInstance(result, RobustnessResult)
        self.assertEqual(len(result.simulation_results), 100)
        self.assertLess(elapsed_time, 30)  # Should complete within 30 seconds
    
    def test_metadata_completeness(self):
        """Test that result metadata is complete."""
        test = MonteCarloRobustnessTest(
            self.data,
            perturbation_methods=["price_noise", "bootstrap"],
            noise_level=0.015
        )
        
        result = test.run_test(
            parameters=self.parameters,
            backtest_function=self.backtest_function,
            n_simulations=20
        )
        
        expected_metadata = [
            "n_simulations", "perturbation_methods", "noise_level",
            "successful_simulations", "failed_simulations"
        ]
        
        for key in expected_metadata:
            self.assertIn(key, result.metadata)
        
        self.assertEqual(result.metadata["n_simulations"], 20)
        self.assertEqual(result.metadata["noise_level"], 0.015)
        self.assertEqual(
            result.metadata["perturbation_methods"],
            ["price_noise", "bootstrap"]
        )


if __name__ == "__main__":
    unittest.main()