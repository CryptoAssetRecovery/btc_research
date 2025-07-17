"""
Unit tests for BayesianOptimizer.

Tests the Bayesian optimization algorithm for correctness,
acquisition functions, and Gaussian process modeling.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import Dict, Any

from btc_research.optimization.optimizers.bayesian import (
    BayesianOptimizer,
    AcquisitionFunction,
)
from btc_research.optimization.types import (
    ParameterSpec,
    ParameterType,
    OptimizationMetric,
    OptimizationResult,
)


class TestBayesianOptimizer(unittest.TestCase):
    """Test cases for BayesianOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_specs = [
            ParameterSpec("x1", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("x2", ParameterType.FLOAT, low=-1.0, high=1.0),
            ParameterSpec("n", ParameterType.INTEGER, low=1, high=10),
        ]
        
        # Simple quadratic objective function
        def mock_objective(params: Dict[str, Any]) -> float:
            x1 = params.get("x1", 0)
            x2 = params.get("x2", 0) 
            n = params.get("n", 1)
            # Quadratic function with maximum at (0.7, 0.3)
            return -(2*(x1-0.7)**2 + (x2-0.3)**2) + n * 0.01
        
        self.objective_function = mock_objective
        self.metric = OptimizationMetric.SHARPE_RATIO
    
    def test_initialization_default(self):
        """Test successful initialization with default parameters."""
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        self.assertEqual(optimizer.parameter_specs, self.parameter_specs)
        self.assertEqual(optimizer.objective_function, self.objective_function)
        self.assertEqual(optimizer.metric, self.metric)
        self.assertEqual(optimizer.acquisition_function, AcquisitionFunction.EXPECTED_IMPROVEMENT)
        self.assertEqual(optimizer.n_initial_points, 5)
        self.assertEqual(optimizer.xi, 0.01)
        self.assertEqual(optimizer.kappa, 2.576)
    
    def test_initialization_with_parameters(self):
        """Test initialization with custom parameters."""
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            acquisition_function=AcquisitionFunction.UPPER_CONFIDENCE_BOUND,
            n_initial_points=10,
            xi=0.05,
            kappa=1.96,
            random_seed=42
        )
        
        self.assertEqual(optimizer.acquisition_function, AcquisitionFunction.UPPER_CONFIDENCE_BOUND)
        self.assertEqual(optimizer.n_initial_points, 10)
        self.assertEqual(optimizer.xi, 0.05)
        self.assertEqual(optimizer.kappa, 1.96)
        self.assertEqual(optimizer.random_seed, 42)
    
    @patch('btc_research.optimization.optimizers.bayesian.GaussianProcessRegressor')
    def test_initialization_gaussian_process(self, mock_gp_class):
        """Test that Gaussian Process is properly initialized."""
        mock_gp = Mock()
        mock_gp_class.return_value = mock_gp
        
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        # Should create GP with appropriate kernel
        mock_gp_class.assert_called_once()
        args, kwargs = mock_gp_class.call_args
        self.assertIn('kernel', kwargs)
        self.assertIn('alpha', kwargs)
        self.assertIn('random_state', kwargs)
    
    def test_normalize_parameters(self):
        """Test parameter normalization to [0,1] range."""
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        params = {"x1": 0.5, "x2": 0.0, "n": 5}
        normalized = optimizer._normalize_parameters(params)
        
        # x1: 0.5 in [0,1] -> 0.5
        # x2: 0.0 in [-1,1] -> 0.5
        # n: 5 in [1,10] -> 4/9 ≈ 0.444
        expected = [0.5, 0.5, 4/9]
        
        np.testing.assert_array_almost_equal(normalized, expected, decimal=3)
    
    def test_denormalize_parameters(self):
        """Test parameter denormalization from [0,1] range."""
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        normalized = np.array([0.5, 0.75, 0.0])
        params = optimizer._denormalize_parameters(normalized)
        
        # x1: 0.5 -> 0.5
        # x2: 0.75 -> 0.5 (0.75 * 2 - 1)
        # n: 0.0 -> 1 (rounded)
        expected = {"x1": 0.5, "x2": 0.5, "n": 1}
        
        self.assertAlmostEqual(params["x1"], expected["x1"], places=3)
        self.assertAlmostEqual(params["x2"], expected["x2"], places=3)
        self.assertEqual(params["n"], expected["n"])
    
    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize -> denormalize is consistent."""
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        original_params = {"x1": 0.3, "x2": -0.5, "n": 7}
        normalized = optimizer._normalize_parameters(original_params)
        recovered_params = optimizer._denormalize_parameters(normalized)
        
        self.assertAlmostEqual(recovered_params["x1"], original_params["x1"], places=3)
        self.assertAlmostEqual(recovered_params["x2"], original_params["x2"], places=3)
        self.assertEqual(recovered_params["n"], original_params["n"])
    
    @patch('btc_research.optimization.optimizers.bayesian.GaussianProcessRegressor')
    def test_suggest_parameters_initial_phase(self, mock_gp_class):
        """Test parameter suggestion during initial random phase."""
        mock_gp = Mock()
        mock_gp_class.return_value = mock_gp
        
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            n_initial_points=3,
            random_seed=42
        )
        
        # First few suggestions should be random
        for i in range(3):
            params = optimizer.suggest_parameters()
            self.assertIsInstance(params, dict)
            self.assertIn("x1", params)
            self.assertIn("x2", params)
            self.assertIn("n", params)
            
            # Should respect bounds
            self.assertGreaterEqual(params["x1"], 0.0)
            self.assertLessEqual(params["x1"], 1.0)
            self.assertGreaterEqual(params["x2"], -1.0)
            self.assertLessEqual(params["x2"], 1.0)
            self.assertGreaterEqual(params["n"], 1)
            self.assertLessEqual(params["n"], 10)
        
        # GP should not be fitted yet during initial phase
        mock_gp.fit.assert_not_called()
    
    @patch('btc_research.optimization.optimizers.bayesian.GaussianProcessRegressor')
    @patch('btc_research.optimization.optimizers.bayesian.minimize')
    def test_suggest_parameters_bayesian_phase(self, mock_minimize, mock_gp_class):
        """Test parameter suggestion during Bayesian optimization phase."""
        mock_gp = Mock()
        mock_gp_class.return_value = mock_gp
        mock_gp.predict.return_value = (np.array([0.5]), np.array([0.1]))
        
        # Mock scipy minimize result
        mock_result = Mock()
        mock_result.x = np.array([0.5, 0.5, 0.5])
        mock_result.fun = -1.0
        mock_minimize.return_value = mock_result
        
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            n_initial_points=2,
            random_seed=42
        )
        
        # Do initial suggestions and evaluations
        for i in range(3):  # One more than n_initial_points to trigger Bayesian phase
            params = optimizer.suggest_parameters()
            score = optimizer.evaluate_parameters(params)
            optimizer.update_with_result(params, score)
        
        # GP should be fitted when we enter Bayesian phase
        mock_gp.fit.assert_called()
        
        # Acquisition function optimization should be called
        mock_minimize.assert_called()
    
    def test_acquisition_function_expected_improvement(self):
        """Test Expected Improvement acquisition function."""
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT
        )
        
        # Mock GP predictions
        mu = np.array([0.5, 0.3, 0.8])
        sigma = np.array([0.1, 0.2, 0.05])
        f_best = 0.6
        
        ei_values = optimizer._expected_improvement(mu, sigma, f_best)
        
        self.assertEqual(len(ei_values), 3)
        self.assertGreater(ei_values[2], ei_values[1])  # Lower uncertainty, higher mean
    
    def test_acquisition_function_upper_confidence_bound(self):
        """Test Upper Confidence Bound acquisition function."""
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            acquisition_function=AcquisitionFunction.UPPER_CONFIDENCE_BOUND,
            kappa=2.0
        )
        
        mu = np.array([0.5, 0.3, 0.8])
        sigma = np.array([0.1, 0.2, 0.05])
        
        ucb_values = optimizer._upper_confidence_bound(mu, sigma)
        
        self.assertEqual(len(ucb_values), 3)
        # UCB = mu + kappa * sigma
        expected = mu + 2.0 * sigma
        np.testing.assert_array_almost_equal(ucb_values, expected)
    
    def test_acquisition_function_probability_improvement(self):
        """Test Probability of Improvement acquisition function."""
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            acquisition_function=AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT
        )
        
        mu = np.array([0.5, 0.3, 0.8])
        sigma = np.array([0.1, 0.2, 0.05])
        f_best = 0.6
        
        pi_values = optimizer._probability_of_improvement(mu, sigma, f_best)
        
        self.assertEqual(len(pi_values), 3)
        self.assertTrue(all(0 <= val <= 1 for val in pi_values))  # Probabilities in [0,1]
    
    @patch('btc_research.optimization.optimizers.bayesian.GaussianProcessRegressor')
    def test_update_with_result(self, mock_gp_class):
        """Test updating optimizer with evaluation results."""
        mock_gp = Mock()
        mock_gp_class.return_value = mock_gp
        
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        params = {"x1": 0.5, "x2": 0.0, "n": 5}
        score = 0.75
        
        initial_count = optimizer.iteration_count
        optimizer.update_with_result(params, score)
        
        # Should increment iteration count
        self.assertEqual(optimizer.iteration_count, initial_count + 1)
        
        # Should store the evaluation
        self.assertEqual(len(optimizer.X_), 1)
        self.assertEqual(len(optimizer.y_), 1)
        self.assertEqual(optimizer.y_[0], score)
    
    @patch('btc_research.optimization.optimizers.bayesian.GaussianProcessRegressor')
    def test_optimize_basic(self, mock_gp_class):
        """Test basic optimization workflow."""
        mock_gp = Mock()
        mock_gp_class.return_value = mock_gp
        mock_gp.predict.return_value = (np.array([0.5]), np.array([0.1]))
        
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            n_initial_points=3,
            random_seed=42
        )
        
        result = optimizer.optimize(max_iterations=10)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.best_parameters)
        self.assertIsNotNone(result.best_score)
        self.assertEqual(result.total_evaluations, 10)
        self.assertGreater(result.optimization_time, 0)
        self.assertEqual(len(result.convergence_history), 10)
    
    def test_optimize_convergence(self):
        """Test optimization convergence detection."""
        # Create simple objective that has clear optimum
        def simple_objective(params):
            return -(params["x1"] - 0.7)**2
        
        specs = [ParameterSpec("x1", ParameterType.FLOAT, low=0.0, high=1.0)]
        
        optimizer = BayesianOptimizer(
            parameter_specs=specs,
            objective_function=simple_objective,
            metric=self.metric,
            n_initial_points=3,
            random_seed=42
        )
        
        result = optimizer.optimize(
            max_iterations=50,
            convergence_threshold=0.01
        )
        
        # Should find solution close to optimal
        self.assertAlmostEqual(result.best_parameters["x1"], 0.7, delta=0.1)
    
    def test_optimize_with_timeout(self):
        """Test optimization with timeout."""
        def slow_objective(params):
            import time
            time.sleep(0.01)
            return 1.0
        
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=slow_objective,
            metric=self.metric,
            n_initial_points=2
        )
        
        result = optimizer.optimize(timeout_seconds=0.1)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertLess(result.optimization_time, 0.2)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        def deterministic_objective(params):
            return params["x1"] + params["x2"] + params["n"] * 0.01
        
        optimizer1 = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=deterministic_objective,
            metric=self.metric,
            n_initial_points=3,
            random_seed=42
        )
        
        optimizer2 = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=deterministic_objective,
            metric=self.metric,
            n_initial_points=3,
            random_seed=42
        )
        
        # Generate initial suggestions - should be identical
        params1 = [optimizer1.suggest_parameters() for _ in range(3)]
        params2 = [optimizer2.suggest_parameters() for _ in range(3)]
        
        # Should be identical with same seed
        for p1, p2 in zip(params1, params2):
            for key in p1:
                if isinstance(p1[key], float):
                    self.assertAlmostEqual(p1[key], p2[key], places=5)
                else:
                    self.assertEqual(p1[key], p2[key])
    
    def test_objective_function_exception_handling(self):
        """Test handling of objective function exceptions."""
        def failing_objective(params):
            if params["x1"] > 0.8:
                raise ValueError("Simulated failure")
            return params["x1"]
        
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=failing_objective,
            metric=self.metric,
            n_initial_points=3,
            random_seed=42
        )
        
        # Should handle exceptions gracefully
        result = optimizer.optimize(max_iterations=10)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.best_parameters)
    
    def test_metadata_tracking(self):
        """Test that optimization metadata is properly tracked."""
        optimizer = BayesianOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            n_initial_points=3
        )
        
        result = optimizer.optimize(max_iterations=8)
        
        self.assertIn("algorithm", result.metadata)
        self.assertIn("acquisition_function", result.metadata)
        self.assertIn("n_initial_points", result.metadata)
        self.assertIn("gp_kernel", result.metadata)
        
        self.assertEqual(result.metadata["algorithm"], "bayesian_optimization")
        self.assertEqual(result.metadata["acquisition_function"], "expected_improvement")
        self.assertEqual(result.metadata["n_initial_points"], 3)
    
    def test_categorical_parameter_handling(self):
        """Test proper handling of categorical parameters."""
        cat_specs = [
            ParameterSpec("x", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("method", ParameterType.CATEGORICAL, choices=["a", "b", "c"])
        ]
        
        def cat_objective(params):
            base = params["x"]
            bonus = {"a": 0.0, "b": 0.1, "c": 0.2}[params["method"]]
            return base + bonus
        
        optimizer = BayesianOptimizer(
            parameter_specs=cat_specs,
            objective_function=cat_objective,
            metric=self.metric,
            n_initial_points=4,
            random_seed=42
        )
        
        result = optimizer.optimize(max_iterations=12)
        
        # Should find that method "c" is optimal
        self.assertEqual(result.best_parameters["method"], "c")
        self.assertGreater(result.best_parameters["x"], 0.8)  # Should maximize x as well
    
    def test_boolean_parameter_handling(self):
        """Test proper handling of boolean parameters."""
        bool_specs = [
            ParameterSpec("x", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("enable", ParameterType.BOOLEAN)
        ]
        
        def bool_objective(params):
            base = params["x"]
            bonus = 0.5 if params["enable"] else 0.0
            return base + bonus
        
        optimizer = BayesianOptimizer(
            parameter_specs=bool_specs,
            objective_function=bool_objective,
            metric=self.metric,
            n_initial_points=3,
            random_seed=42
        )
        
        result = optimizer.optimize(max_iterations=10)
        
        # Should find that enable=True is optimal
        self.assertTrue(result.best_parameters["enable"])
        self.assertGreater(result.best_parameters["x"], 0.8)


class TestAcquisitionFunction(unittest.TestCase):
    """Test cases for AcquisitionFunction enum."""
    
    def test_acquisition_function_values(self):
        """Test that all expected acquisition functions are available."""
        expected_functions = [
            "expected_improvement",
            "upper_confidence_bound", 
            "probability_of_improvement"
        ]
        
        for func_name in expected_functions:
            func = AcquisitionFunction(func_name)
            self.assertEqual(func.value, func_name)
    
    def test_acquisition_function_iteration(self):
        """Test that we can iterate over all acquisition functions."""
        functions = list(AcquisitionFunction)
        self.assertGreaterEqual(len(functions), 3)


if __name__ == "__main__":
    unittest.main()