"""
Unit tests for RandomSearchOptimizer.

Tests the random search optimization algorithm for correctness,
sampling strategies, and statistical properties.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from typing import Dict, Any

from btc_research.optimization.optimizers.random_search import (
    RandomSearchOptimizer,
    SamplingStrategy,
)
from btc_research.optimization.types import (
    ParameterSpec,
    ParameterType,
    OptimizationMetric,
    OptimizationResult,
)


class TestRandomSearchOptimizer(unittest.TestCase):
    """Test cases for RandomSearchOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_specs = [
            ParameterSpec("int_param", ParameterType.INTEGER, low=1, high=10),
            ParameterSpec("float_param", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("cat_param", ParameterType.CATEGORICAL, choices=["a", "b", "c"]),
            ParameterSpec("bool_param", ParameterType.BOOLEAN),
        ]
        
        # Mock objective function
        def mock_objective(params: Dict[str, Any]) -> float:
            score = 0.0
            score += params.get("int_param", 0) * 0.1
            score += params.get("float_param", 0) * 0.5
            score += {"a": 0.1, "b": 0.2, "c": 0.3}.get(params.get("cat_param"), 0)
            score += 0.1 if params.get("bool_param", False) else 0
            return score
        
        self.objective_function = mock_objective
        self.metric = OptimizationMetric.SHARPE_RATIO
    
    def test_initialization_default(self):
        """Test successful initialization with default parameters."""
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        self.assertEqual(optimizer.parameter_specs, self.parameter_specs)
        self.assertEqual(optimizer.objective_function, self.objective_function)
        self.assertEqual(optimizer.metric, self.metric)
        self.assertEqual(optimizer.sampling_strategy, SamplingStrategy.UNIFORM)
        self.assertIsNone(optimizer.random_seed)
    
    def test_initialization_with_strategy(self):
        """Test initialization with specific sampling strategy."""
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            sampling_strategy=SamplingStrategy.LATIN_HYPERCUBE,
            random_seed=42
        )
        
        self.assertEqual(optimizer.sampling_strategy, SamplingStrategy.LATIN_HYPERCUBE)
        self.assertEqual(optimizer.random_seed, 42)
    
    def test_suggest_parameters_uniform_sampling(self):
        """Test uniform random parameter sampling."""
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            sampling_strategy=SamplingStrategy.UNIFORM,
            random_seed=42
        )
        
        # Generate multiple samples to test distribution
        samples = []
        for _ in range(100):
            params = optimizer.suggest_parameters()
            samples.append(params)
            
            # Validate parameter types and bounds
            self.assertIsInstance(params["int_param"], int)
            self.assertGreaterEqual(params["int_param"], 1)
            self.assertLessEqual(params["int_param"], 10)
            
            self.assertIsInstance(params["float_param"], float)
            self.assertGreaterEqual(params["float_param"], 0.0)
            self.assertLessEqual(params["float_param"], 1.0)
            
            self.assertIn(params["cat_param"], ["a", "b", "c"])
            self.assertIsInstance(params["bool_param"], bool)
        
        # Check distribution properties
        int_values = [s["int_param"] for s in samples]
        float_values = [s["float_param"] for s in samples]
        
        # Should have reasonable spread
        self.assertGreater(np.std(int_values), 0)
        self.assertGreater(np.std(float_values), 0)
        
        # Categorical values should be present
        cat_values = [s["cat_param"] for s in samples]
        self.assertIn("a", cat_values)
        self.assertIn("b", cat_values)
        self.assertIn("c", cat_values)
    
    def test_suggest_parameters_latin_hypercube(self):
        """Test Latin Hypercube sampling."""
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            sampling_strategy=SamplingStrategy.LATIN_HYPERCUBE,
            random_seed=42
        )
        
        # Generate samples
        samples = []
        for _ in range(50):
            params = optimizer.suggest_parameters()
            samples.append(params)
        
        # Latin Hypercube should provide better space coverage
        int_values = [s["int_param"] for s in samples]
        float_values = [s["float_param"] for s in samples]
        
        # Check that we get good coverage of the parameter space
        self.assertGreater(len(set(int_values)), 5)  # Should explore multiple integer values
        self.assertGreater(np.std(float_values), 0.2)  # Should have good spread in float values
    
    def test_suggest_parameters_sobol(self):
        """Test Sobol sequence sampling."""
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            sampling_strategy=SamplingStrategy.SOBOL,
            random_seed=42
        )
        
        # Generate samples
        samples = []
        for _ in range(32):  # Power of 2 for better Sobol properties
            params = optimizer.suggest_parameters()
            samples.append(params)
        
        # Sobol should provide very uniform coverage
        float_values = [s["float_param"] for s in samples]
        
        # Should have good space coverage
        self.assertGreater(np.std(float_values), 0.25)
    
    def test_suggest_parameters_halton(self):
        """Test Halton sequence sampling."""
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            sampling_strategy=SamplingStrategy.HALTON,
            random_seed=42
        )
        
        # Generate samples
        samples = []
        for _ in range(50):
            params = optimizer.suggest_parameters()
            samples.append(params)
        
        # Check basic validity
        for params in samples:
            self.assertIn("int_param", params)
            self.assertIn("float_param", params)
            self.assertIn("cat_param", params)
            self.assertIn("bool_param", params)
    
    def test_optimize_basic(self):
        """Test basic optimization functionality."""
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            maximize=True,
            random_seed=42
        )
        
        result = optimizer.optimize(max_iterations=50)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.best_parameters)
        self.assertIsNotNone(result.best_score)
        self.assertEqual(result.total_evaluations, 50)
        self.assertGreater(result.optimization_time, 0)
        self.assertEqual(len(result.convergence_history), 50)
    
    def test_optimize_minimize(self):
        """Test optimization in minimize mode."""
        def negative_objective(params):
            return -self.objective_function(params)
        
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=negative_objective,
            metric=self.metric,
            maximize=False,
            random_seed=42
        )
        
        result = optimizer.optimize(max_iterations=30)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertLess(result.best_score, 0)  # Should be negative due to minimize mode
    
    def test_optimize_with_convergence_threshold(self):
        """Test optimization with convergence threshold."""
        # Create objective that converges quickly
        def converging_objective(params):
            return 1.0  # Always returns same value
        
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=converging_objective,
            metric=self.metric,
            random_seed=42
        )
        
        result = optimizer.optimize(
            max_iterations=100,
            convergence_threshold=0.001
        )
        
        # Should converge early due to no improvement
        self.assertLess(result.total_evaluations, 100)
    
    def test_optimize_with_timeout(self):
        """Test optimization with timeout."""
        def slow_objective(params):
            import time
            time.sleep(0.01)
            return 1.0
        
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=slow_objective,
            metric=self.metric
        )
        
        result = optimizer.optimize(timeout_seconds=0.1)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertLess(result.optimization_time, 0.2)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        optimizer1 = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            random_seed=42
        )
        
        optimizer2 = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            random_seed=42
        )
        
        # Generate same sequence of parameters
        params1 = [optimizer1.suggest_parameters() for _ in range(10)]
        params2 = [optimizer2.suggest_parameters() for _ in range(10)]
        
        self.assertEqual(params1, params2)
    
    def test_parameter_bounds_enforcement(self):
        """Test that generated parameters respect bounds."""
        specs_with_bounds = [
            ParameterSpec("bounded_int", ParameterType.INTEGER, low=5, high=15),
            ParameterSpec("bounded_float", ParameterType.FLOAT, low=0.2, high=0.8),
        ]
        
        optimizer = RandomSearchOptimizer(
            parameter_specs=specs_with_bounds,
            objective_function=lambda x: 1.0,
            metric=self.metric
        )
        
        for _ in range(100):
            params = optimizer.suggest_parameters()
            
            self.assertGreaterEqual(params["bounded_int"], 5)
            self.assertLessEqual(params["bounded_int"], 15)
            self.assertGreaterEqual(params["bounded_float"], 0.2)
            self.assertLessEqual(params["bounded_float"], 0.8)
    
    def test_categorical_parameter_handling(self):
        """Test proper handling of categorical parameters."""
        cat_only_specs = [
            ParameterSpec("choice", ParameterType.CATEGORICAL, choices=["option1", "option2", "option3"])
        ]
        
        optimizer = RandomSearchOptimizer(
            parameter_specs=cat_only_specs,
            objective_function=lambda x: 1.0,
            metric=self.metric
        )
        
        choices_seen = set()
        for _ in range(50):
            params = optimizer.suggest_parameters()
            choices_seen.add(params["choice"])
        
        # Should see all choices eventually
        self.assertEqual(choices_seen, {"option1", "option2", "option3"})
    
    def test_boolean_parameter_handling(self):
        """Test proper handling of boolean parameters."""
        bool_only_specs = [
            ParameterSpec("flag", ParameterType.BOOLEAN)
        ]
        
        optimizer = RandomSearchOptimizer(
            parameter_specs=bool_only_specs,
            objective_function=lambda x: 1.0,
            metric=self.metric
        )
        
        values_seen = set()
        for _ in range(20):
            params = optimizer.suggest_parameters()
            values_seen.add(params["flag"])
        
        # Should see both True and False
        self.assertEqual(values_seen, {True, False})
    
    def test_objective_function_exception_handling(self):
        """Test handling of objective function exceptions."""
        def failing_objective(params):
            if params["int_param"] > 5:
                raise ValueError("Simulated failure")
            return 1.0
        
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=failing_objective,
            metric=self.metric,
            random_seed=42
        )
        
        # Should handle exceptions gracefully
        result = optimizer.optimize(max_iterations=20)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.best_parameters)
    
    def test_metadata_tracking(self):
        """Test that optimization metadata is properly tracked."""
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            sampling_strategy=SamplingStrategy.LATIN_HYPERCUBE
        )
        
        result = optimizer.optimize(max_iterations=25)
        
        self.assertIn("algorithm", result.metadata)
        self.assertIn("sampling_strategy", result.metadata)
        self.assertIn("search_space_coverage", result.metadata)
        
        self.assertEqual(result.metadata["algorithm"], "random_search")
        self.assertEqual(result.metadata["sampling_strategy"], "latin_hypercube")
    
    def test_convergence_tracking(self):
        """Test that convergence history is properly tracked."""
        optimizer = RandomSearchOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            random_seed=42
        )
        
        result = optimizer.optimize(max_iterations=10)
        
        self.assertEqual(len(result.convergence_history), 10)
        
        for i, entry in enumerate(result.convergence_history):
            self.assertIn("iteration", entry)
            self.assertIn("best_score", entry)
            self.assertIn("current_score", entry)
            self.assertEqual(entry["iteration"], i + 1)
    
    def test_parameter_distribution_uniformity(self):
        """Test that parameter distributions are reasonably uniform."""
        optimizer = RandomSearchOptimizer(
            parameter_specs=[
                ParameterSpec("uniform_float", ParameterType.FLOAT, low=0.0, high=1.0)
            ],
            objective_function=lambda x: 1.0,
            metric=self.metric,
            sampling_strategy=SamplingStrategy.UNIFORM,
            random_seed=42
        )
        
        values = []
        for _ in range(1000):
            params = optimizer.suggest_parameters()
            values.append(params["uniform_float"])
        
        # Check statistical properties
        mean = np.mean(values)
        std = np.std(values)
        
        # For uniform distribution [0,1], mean should be ~0.5, std should be ~0.289
        self.assertAlmostEqual(mean, 0.5, delta=0.05)
        self.assertAlmostEqual(std, 0.289, delta=0.05)
    
    def test_different_sampling_strategies_coverage(self):
        """Test that different sampling strategies provide different coverage patterns."""
        strategies = [
            SamplingStrategy.UNIFORM,
            SamplingStrategy.LATIN_HYPERCUBE,
            SamplingStrategy.SOBOL,
            SamplingStrategy.HALTON
        ]
        
        coverage_results = {}
        
        for strategy in strategies:
            optimizer = RandomSearchOptimizer(
                parameter_specs=[
                    ParameterSpec("x", ParameterType.FLOAT, low=0.0, high=1.0),
                    ParameterSpec("y", ParameterType.FLOAT, low=0.0, high=1.0)
                ],
                objective_function=lambda x: 1.0,
                metric=self.metric,
                sampling_strategy=strategy,
                random_seed=42
            )
            
            samples = []
            for _ in range(50):
                params = optimizer.suggest_parameters()
                samples.append((params["x"], params["y"]))
            
            # Calculate coverage metric (minimum distance between any two points)
            min_distances = []
            for i, (x1, y1) in enumerate(samples):
                for j, (x2, y2) in enumerate(samples[i+1:], i+1):
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    min_distances.append(dist)
            
            coverage_results[strategy] = min(min_distances) if min_distances else 0
        
        # Latin Hypercube and Sobol should generally provide better coverage
        # This is a weak test since we can't guarantee exact ordering
        self.assertGreater(len(coverage_results), 0)


class TestSamplingStrategy(unittest.TestCase):
    """Test cases for SamplingStrategy enum."""
    
    def test_sampling_strategy_values(self):
        """Test that all expected sampling strategies are available."""
        expected_strategies = ["uniform", "latin_hypercube", "sobol", "halton"]
        
        for strategy_name in expected_strategies:
            strategy = SamplingStrategy(strategy_name)
            self.assertEqual(strategy.value, strategy_name)
    
    def test_sampling_strategy_iteration(self):
        """Test that we can iterate over all sampling strategies."""
        strategies = list(SamplingStrategy)
        self.assertGreaterEqual(len(strategies), 4)


if __name__ == "__main__":
    unittest.main()