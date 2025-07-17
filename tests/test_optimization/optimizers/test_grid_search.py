"""
Unit tests for GridSearchOptimizer.

Tests the exhaustive grid search optimization algorithm for correctness,
parameter handling, and edge cases.
"""

import unittest
from unittest.mock import Mock
import numpy as np
from typing import Dict, Any

from btc_research.optimization.optimizers.grid_search import GridSearchOptimizer
from btc_research.optimization.types import (
    ParameterSpec,
    ParameterType,
    OptimizationMetric,
    OptimizationResult,
)


class TestGridSearchOptimizer(unittest.TestCase):
    """Test cases for GridSearchOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_specs = [
            ParameterSpec("int_param", ParameterType.INTEGER, low=1, high=3),
            ParameterSpec("float_param", ParameterType.FLOAT, low=0.1, high=0.3, step=0.1),
            ParameterSpec("cat_param", ParameterType.CATEGORICAL, choices=["a", "b"]),
        ]
        
        # Mock objective function that returns parameter sum
        def mock_objective(params: Dict[str, Any]) -> float:
            score = 0.0
            if "int_param" in params:
                score += params["int_param"]
            if "float_param" in params:
                score += params["float_param"]
            if "cat_param" in params:
                score += {"a": 0.1, "b": 0.2}[params["cat_param"]]
            return score
        
        self.objective_function = mock_objective
        self.metric = OptimizationMetric.SHARPE_RATIO
    
    def test_initialization(self):
        """Test successful initialization."""
        optimizer = GridSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            maximize=True,
            random_seed=42
        )
        
        self.assertEqual(optimizer.parameter_specs, self.parameter_specs)
        self.assertEqual(optimizer.objective_function, self.objective_function)
        self.assertEqual(optimizer.metric, self.metric)
        self.assertTrue(optimizer.maximize)
        self.assertEqual(optimizer.random_seed, 42)
    
    def test_generate_parameter_grid_integer(self):
        """Test parameter grid generation for integer parameters."""
        int_spec = ParameterSpec("test", ParameterType.INTEGER, low=1, high=4, step=1)
        optimizer = GridSearchOptimizer(
            parameter_specs=[int_spec],
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        grid = optimizer._generate_parameter_grid()
        expected_values = [1, 2, 3, 4]
        
        self.assertEqual(len(grid), len(expected_values))
        for params in grid:
            self.assertIn(params["test"], expected_values)
    
    def test_generate_parameter_grid_float(self):
        """Test parameter grid generation for float parameters."""
        float_spec = ParameterSpec("test", ParameterType.FLOAT, low=0.1, high=0.4, step=0.1)
        optimizer = GridSearchOptimizer(
            parameter_specs=[float_spec],
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        grid = optimizer._generate_parameter_grid()
        expected_values = [0.1, 0.2, 0.3, 0.4]
        
        self.assertEqual(len(grid), len(expected_values))
        for params in grid:
            self.assertAlmostEqual(params["test"], expected_values[grid.index(params)], places=5)
    
    def test_generate_parameter_grid_categorical(self):
        """Test parameter grid generation for categorical parameters."""
        cat_spec = ParameterSpec("test", ParameterType.CATEGORICAL, choices=["x", "y", "z"])
        optimizer = GridSearchOptimizer(
            parameter_specs=[cat_spec],
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        grid = optimizer._generate_parameter_grid()
        expected_choices = ["x", "y", "z"]
        
        self.assertEqual(len(grid), len(expected_choices))
        for params in grid:
            self.assertIn(params["test"], expected_choices)
    
    def test_generate_parameter_grid_boolean(self):
        """Test parameter grid generation for boolean parameters."""
        bool_spec = ParameterSpec("test", ParameterType.BOOLEAN)
        optimizer = GridSearchOptimizer(
            parameter_specs=[bool_spec],
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        grid = optimizer._generate_parameter_grid()
        expected_values = [True, False]
        
        self.assertEqual(len(grid), 2)
        values = [params["test"] for params in grid]
        self.assertIn(True, values)
        self.assertIn(False, values)
    
    def test_generate_parameter_grid_multiple_parameters(self):
        """Test parameter grid generation with multiple parameters."""
        specs = [
            ParameterSpec("int_param", ParameterType.INTEGER, low=1, high=2),
            ParameterSpec("cat_param", ParameterType.CATEGORICAL, choices=["a", "b"]),
        ]
        
        optimizer = GridSearchOptimizer(
            parameter_specs=specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        grid = optimizer._generate_parameter_grid()
        
        # Should have 2 * 2 = 4 combinations
        self.assertEqual(len(grid), 4)
        
        # Check all combinations are present
        expected_combinations = [
            {"int_param": 1, "cat_param": "a"},
            {"int_param": 1, "cat_param": "b"},
            {"int_param": 2, "cat_param": "a"},
            {"int_param": 2, "cat_param": "b"},
        ]
        
        for expected in expected_combinations:
            self.assertIn(expected, grid)
    
    def test_suggest_parameters(self):
        """Test parameter suggestion functionality."""
        optimizer = GridSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            random_seed=42
        )
        
        # First suggestion should work
        params1 = optimizer.suggest_parameters()
        self.assertIsInstance(params1, dict)
        self.assertIn("int_param", params1)
        self.assertIn("float_param", params1)
        self.assertIn("cat_param", params1)
        
        # Should be able to suggest multiple times until exhausted
        params2 = optimizer.suggest_parameters()
        self.assertNotEqual(params1, params2)
    
    def test_suggest_parameters_exhausted(self):
        """Test behavior when all parameter combinations are exhausted."""
        # Create a small grid for easy exhaustion
        small_specs = [
            ParameterSpec("param", ParameterType.CATEGORICAL, choices=["a", "b"])
        ]
        
        optimizer = GridSearchOptimizer(
            parameter_specs=small_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        # Exhaust all combinations
        params1 = optimizer.suggest_parameters()
        params2 = optimizer.suggest_parameters()
        
        # Third suggestion should raise StopIteration or return None
        with self.assertRaises((StopIteration, ValueError)):
            optimizer.suggest_parameters()
    
    def test_optimize_maximize(self):
        """Test optimization in maximize mode."""
        # Simple case where we know the optimal solution
        specs = [
            ParameterSpec("value", ParameterType.INTEGER, low=1, high=5)
        ]
        
        def objective(params):
            return params["value"]  # Maximum should be 5
        
        optimizer = GridSearchOptimizer(
            parameter_specs=specs,
            objective_function=objective,
            metric=self.metric,
            maximize=True
        )
        
        result = optimizer.optimize()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.best_parameters["value"], 5)
        self.assertEqual(result.best_score, 5.0)
        self.assertEqual(result.total_evaluations, 5)  # Should evaluate all 5 values
        self.assertGreater(result.optimization_time, 0)
    
    def test_optimize_minimize(self):
        """Test optimization in minimize mode."""
        specs = [
            ParameterSpec("value", ParameterType.INTEGER, low=1, high=5)
        ]
        
        def objective(params):
            return params["value"]  # Minimum should be 1
        
        optimizer = GridSearchOptimizer(
            parameter_specs=specs,
            objective_function=objective,
            metric=self.metric,
            maximize=False
        )
        
        result = optimizer.optimize()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.best_parameters["value"], 1)
        self.assertEqual(result.best_score, -1.0)  # Negated for minimize mode
        self.assertEqual(result.total_evaluations, 5)
    
    def test_optimize_with_max_iterations(self):
        """Test optimization with max iterations limit."""
        optimizer = GridSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        # Limit iterations to less than total grid size
        result = optimizer.optimize(max_iterations=3)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.total_evaluations, 3)
        self.assertIsNotNone(result.best_parameters)
        self.assertIsNotNone(result.best_score)
    
    def test_optimize_with_timeout(self):
        """Test optimization with timeout."""
        def slow_objective(params):
            import time
            time.sleep(0.01)  # Small delay
            return 1.0
        
        optimizer = GridSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=slow_objective,
            metric=self.metric
        )
        
        # Very short timeout
        result = optimizer.optimize(timeout_seconds=0.05)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertLess(result.optimization_time, 0.1)  # Should timeout quickly
        self.assertGreater(result.total_evaluations, 0)  # Should complete at least some
    
    def test_optimize_empty_grid(self):
        """Test optimization with parameters that create empty grid."""
        # Parameters with no valid combinations
        invalid_specs = [
            ParameterSpec("param", ParameterType.INTEGER, low=5, high=1)  # Invalid range
        ]
        
        optimizer = GridSearchOptimizer(
            parameter_specs=invalid_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        # Should handle gracefully
        result = optimizer.optimize()
        self.assertEqual(result.total_evaluations, 0)
    
    def test_convergence_history(self):
        """Test that convergence history is properly tracked."""
        optimizer = GridSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            maximize=True
        )
        
        result = optimizer.optimize(max_iterations=5)
        
        self.assertIsInstance(result.convergence_history, list)
        self.assertGreater(len(result.convergence_history), 0)
        
        # Check convergence history structure
        for entry in result.convergence_history:
            self.assertIn("iteration", entry)
            self.assertIn("best_score", entry)
            self.assertIn("current_score", entry)
    
    def test_objective_function_exception_handling(self):
        """Test handling of objective function exceptions."""
        def failing_objective(params):
            if params.get("int_param", 0) == 2:
                raise ValueError("Simulated failure")
            return 1.0
        
        optimizer = GridSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=failing_objective,
            metric=self.metric,
            maximize=True
        )
        
        # Should not crash and should skip failed evaluations
        result = optimizer.optimize()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.best_parameters)
        # Should have fewer evaluations due to failures
        self.assertGreater(result.total_evaluations, 0)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        optimizer1 = GridSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            random_seed=42
        )
        
        optimizer2 = GridSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            random_seed=42
        )
        
        result1 = optimizer1.optimize()
        result2 = optimizer2.optimize()
        
        # Results should be identical with same seed
        self.assertEqual(result1.best_parameters, result2.best_parameters)
        self.assertEqual(result1.best_score, result2.best_score)
        self.assertEqual(result1.total_evaluations, result2.total_evaluations)
    
    def test_metadata_tracking(self):
        """Test that optimization metadata is properly tracked."""
        optimizer = GridSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        result = optimizer.optimize()
        
        self.assertIn("grid_size", result.metadata)
        self.assertIn("search_space_coverage", result.metadata)
        self.assertIn("algorithm", result.metadata)
        self.assertEqual(result.metadata["algorithm"], "grid_search")
        
        # Grid size should match total possible combinations
        expected_grid_size = 3 * 3 * 2  # int_param * float_param * cat_param
        self.assertEqual(result.metadata["grid_size"], expected_grid_size)


if __name__ == "__main__":
    unittest.main()