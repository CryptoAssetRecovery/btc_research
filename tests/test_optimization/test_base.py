"""
Unit tests for base optimization framework classes.

Tests for abstract base classes and core interfaces to ensure
proper implementation contracts and error handling.
"""

import unittest
from unittest.mock import Mock, patch
from abc import ABC
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable

from btc_research.optimization.base import (
    BaseOptimizer,
    BaseValidator,
    BaseRobustnessTest,
    BaseStatisticsTest,
    ObjectiveFunction,
)
from btc_research.optimization.types import (
    ParameterSpec,
    ParameterType,
    OptimizationMetric,
    OptimizationResult,
    ValidationResult,
    RobustnessResult,
    StatisticsResult,
)
from tests.fixtures.sample_data import create_btc_sample_data


class ConcreteOptimizer(BaseOptimizer):
    """Concrete implementation for testing BaseOptimizer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._suggestion_count = 0
    
    def optimize(self, max_iterations=100, **kwargs):
        """Simple optimization that tries random parameter combinations."""
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None
        
        for i in range(max_iterations):
            params = self.suggest_parameters()
            score = self.evaluate_parameters(params)
            self.update_with_result(params, score)
            
            if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                best_score = score
                best_params = params
        
        return OptimizationResult(
            best_parameters=best_params or {},
            best_score=best_score,
            total_evaluations=max_iterations,
            optimization_time=0.0,
            convergence_history=[],
            metadata={}
        )
    
    def suggest_parameters(self):
        """Suggest random parameter values within bounds."""
        self._suggestion_count += 1
        params = {}
        
        for spec in self.parameter_specs:
            if spec.param_type == ParameterType.INTEGER:
                low = spec.low or 1
                high = spec.high or 100
                params[spec.name] = np.random.randint(low, high + 1)
            elif spec.param_type == ParameterType.FLOAT:
                low = spec.low or 0.0
                high = spec.high or 1.0
                params[spec.name] = np.random.uniform(low, high)
            elif spec.param_type == ParameterType.CATEGORICAL:
                params[spec.name] = np.random.choice(spec.choices or ["a", "b"])
            elif spec.param_type == ParameterType.BOOLEAN:
                params[spec.name] = np.random.choice([True, False])
        
        return params


class ConcreteValidator(BaseValidator):
    """Concrete implementation for testing BaseValidator."""
    
    def split_data(self):
        """Simple 80/20 split."""
        split_point = int(len(self.data) * 0.8)
        train_data = self.data.iloc[:split_point]
        val_data = self.data.iloc[split_point:]
        return [(train_data, val_data)]
    
    def validate(self, parameters, backtest_function):
        """Simple validation using single split."""
        splits = self.split_data()
        fold_results = []
        
        for train_data, val_data in splits:
            # Run backtest on validation data
            result = backtest_function(val_data, parameters)
            fold_results.append(result)
        
        mean_metrics, std_metrics, confidence_intervals = self._calculate_summary_statistics(fold_results)
        
        return ValidationResult(
            parameters=parameters,
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            confidence_intervals=confidence_intervals,
            metadata={"n_folds": len(splits)}
        )


class ConcreteRobustnessTest(BaseRobustnessTest):
    """Concrete implementation for testing BaseRobustnessTest."""
    
    def run_test(self, parameters, backtest_function, n_simulations=10, **kwargs):
        """Simple robustness test with data perturbation."""
        results = []
        
        for i in range(n_simulations):
            # Add small random noise to data for perturbation
            perturbed_data = self.data.copy()
            noise_factor = 0.001  # 0.1% noise
            for col in ['open', 'high', 'low', 'close']:
                if col in perturbed_data.columns:
                    noise = np.random.normal(1, noise_factor, len(perturbed_data))
                    perturbed_data[col] *= noise
            
            # Run backtest with perturbed data
            result = backtest_function(perturbed_data, parameters)
            results.append(result)
        
        # Calculate statistics
        if results:
            all_metrics = set()
            for result in results:
                all_metrics.update(result.keys())
            
            mean_metrics = {}
            std_metrics = {}
            for metric in all_metrics:
                values = [result.get(metric, 0.0) for result in results]
                mean_metrics[metric] = np.mean(values)
                std_metrics[metric] = np.std(values)
        else:
            mean_metrics = {}
            std_metrics = {}
        
        var_metrics, es_metrics = self._calculate_risk_metrics(results)
        
        return RobustnessResult(
            parameters=parameters,
            simulation_results=results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            var_metrics=var_metrics,
            es_metrics=es_metrics,
            metadata={"n_simulations": n_simulations}
        )


class ConcreteStatisticsTest(BaseStatisticsTest):
    """Concrete implementation for testing BaseStatisticsTest."""
    
    def run_test(self, sample1, sample2=None, **kwargs):
        """Simple t-test implementation."""
        self._validate_samples(sample1, sample2)
        
        if sample2 is None:
            # One-sample test against zero
            statistic = np.mean(sample1) / (np.std(sample1) / np.sqrt(len(sample1)))
            p_value = 0.05  # Dummy p-value
        else:
            # Two-sample test
            mean_diff = np.mean(sample1) - np.mean(sample2)
            pooled_std = np.sqrt((np.var(sample1) + np.var(sample2)) / 2)
            statistic = mean_diff / (pooled_std * np.sqrt(2 / len(sample1)))
            p_value = 0.05  # Dummy p-value
        
        significant = p_value < self.alpha
        
        return StatisticsResult(
            test_name="t_test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            confidence_level=self.confidence_level,
            metadata={"sample1_size": len(sample1), "sample2_size": len(sample2) if sample2 else 0}
        )


class TestBaseOptimizer(unittest.TestCase):
    """Test cases for BaseOptimizer abstract class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_specs = [
            ParameterSpec("int_param", ParameterType.INTEGER, low=1, high=10),
            ParameterSpec("float_param", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("cat_param", ParameterType.CATEGORICAL, choices=["a", "b", "c"]),
            ParameterSpec("bool_param", ParameterType.BOOLEAN),
        ]
        
        self.objective_function = Mock(return_value=0.5)
        self.metric = OptimizationMetric.SHARPE_RATIO
    
    def test_initialization_valid(self):
        """Test successful initialization with valid parameters."""
        optimizer = ConcreteOptimizer(
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
        self.assertEqual(optimizer.iteration_count, 0)
        self.assertIsNone(optimizer.best_result)
        self.assertEqual(optimizer.all_results, [])
    
    def test_initialization_empty_specs(self):
        """Test initialization fails with empty parameter specs."""
        with self.assertRaises(ValueError) as context:
            ConcreteOptimizer(
                parameter_specs=[],
                objective_function=self.objective_function,
                metric=self.metric
            )
        self.assertIn("At least one parameter specification is required", str(context.exception))
    
    def test_evaluate_parameters_maximize(self):
        """Test parameter evaluation in maximize mode."""
        optimizer = ConcreteOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            maximize=True
        )
        
        params = {"test": "value"}
        score = optimizer.evaluate_parameters(params)
        
        self.objective_function.assert_called_once_with(params)
        self.assertEqual(score, 0.5)  # Should return objective function value as-is
    
    def test_evaluate_parameters_minimize(self):
        """Test parameter evaluation in minimize mode."""
        optimizer = ConcreteOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            maximize=False
        )
        
        params = {"test": "value"}
        score = optimizer.evaluate_parameters(params)
        
        self.objective_function.assert_called_once_with(params)
        self.assertEqual(score, -0.5)  # Should negate objective function value
    
    def test_evaluate_parameters_exception_handling(self):
        """Test parameter evaluation handles exceptions gracefully."""
        failing_objective = Mock(side_effect=Exception("Test error"))
        
        optimizer = ConcreteOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=failing_objective,
            metric=self.metric,
            maximize=True
        )
        
        params = {"test": "value"}
        score = optimizer.evaluate_parameters(params)
        
        self.assertEqual(score, float('-inf'))  # Should return worst score for maximize
        
        # Test minimize mode
        optimizer.maximize = False
        score = optimizer.evaluate_parameters(params)
        self.assertEqual(score, float('inf'))  # Should return worst score for minimize
    
    def test_update_with_result(self):
        """Test updating optimizer state with results."""
        optimizer = ConcreteOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        initial_count = optimizer.iteration_count
        optimizer.update_with_result({"param": "value"}, 0.7)
        
        self.assertEqual(optimizer.iteration_count, initial_count + 1)
    
    def test_parameter_validation_integer(self):
        """Test validation of integer parameters."""
        optimizer = ConcreteOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        # Valid integer
        valid_params = {"int_param": 5, "float_param": 0.5, "cat_param": "a", "bool_param": True}
        self.assertTrue(optimizer._validate_parameters(valid_params))
        
        # Invalid type
        invalid_params = {"int_param": 5.5, "float_param": 0.5, "cat_param": "a", "bool_param": True}
        self.assertFalse(optimizer._validate_parameters(invalid_params))
        
        # Out of bounds
        invalid_params = {"int_param": 15, "float_param": 0.5, "cat_param": "a", "bool_param": True}
        self.assertFalse(optimizer._validate_parameters(invalid_params))
    
    def test_parameter_validation_float(self):
        """Test validation of float parameters."""
        optimizer = ConcreteOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        # Valid float
        valid_params = {"int_param": 5, "float_param": 0.5, "cat_param": "a", "bool_param": True}
        self.assertTrue(optimizer._validate_parameters(valid_params))
        
        # Out of bounds
        invalid_params = {"int_param": 5, "float_param": 1.5, "cat_param": "a", "bool_param": True}
        self.assertFalse(optimizer._validate_parameters(invalid_params))
    
    def test_parameter_validation_categorical(self):
        """Test validation of categorical parameters."""
        optimizer = ConcreteOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        # Valid choice
        valid_params = {"int_param": 5, "float_param": 0.5, "cat_param": "b", "bool_param": True}
        self.assertTrue(optimizer._validate_parameters(valid_params))
        
        # Invalid choice
        invalid_params = {"int_param": 5, "float_param": 0.5, "cat_param": "z", "bool_param": True}
        self.assertFalse(optimizer._validate_parameters(invalid_params))
    
    def test_parameter_validation_boolean(self):
        """Test validation of boolean parameters."""
        optimizer = ConcreteOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        # Valid boolean
        valid_params = {"int_param": 5, "float_param": 0.5, "cat_param": "a", "bool_param": False}
        self.assertTrue(optimizer._validate_parameters(valid_params))
        
        # Invalid type
        invalid_params = {"int_param": 5, "float_param": 0.5, "cat_param": "a", "bool_param": "true"}
        self.assertFalse(optimizer._validate_parameters(invalid_params))
    
    def test_parameter_validation_missing_parameter(self):
        """Test validation fails when parameters are missing."""
        optimizer = ConcreteOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        # Missing parameter
        incomplete_params = {"int_param": 5, "float_param": 0.5}  # Missing cat_param and bool_param
        self.assertFalse(optimizer._validate_parameters(incomplete_params))
    
    def test_optimize_integration(self):
        """Test the full optimization workflow."""
        np.random.seed(42)  # For reproducible tests
        
        optimizer = ConcreteOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            maximize=True,
            random_seed=42
        )
        
        result = optimizer.optimize(max_iterations=5)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.best_parameters)
        self.assertEqual(result.total_evaluations, 5)
        self.assertEqual(optimizer.iteration_count, 5)


class TestBaseValidator(unittest.TestCase):
    """Test cases for BaseValidator abstract class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = create_btc_sample_data(periods=100)
        self.backtest_function = Mock(return_value={"sharpe": 0.5, "return": 0.1})
    
    def test_initialization_valid(self):
        """Test successful initialization with valid data."""
        validator = ConcreteValidator(self.data, random_seed=42)
        
        self.assertTrue(validator.data.equals(self.data))
        self.assertEqual(validator.date_column, "timestamp")
        self.assertEqual(validator.random_seed, 42)
    
    def test_initialization_empty_data(self):
        """Test initialization fails with empty data."""
        empty_data = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            ConcreteValidator(empty_data)
        self.assertIn("Data cannot be empty", str(context.exception))
    
    def test_initialization_invalid_date_column(self):
        """Test initialization fails with invalid date column."""
        data_no_date = pd.DataFrame({"price": [1, 2, 3]})
        
        with self.assertRaises(ValueError) as context:
            ConcreteValidator(data_no_date, date_column="missing_column")
        self.assertIn("Date column 'missing_column' not found", str(context.exception))
    
    def test_split_data(self):
        """Test data splitting functionality."""
        validator = ConcreteValidator(self.data)
        splits = validator.split_data()
        
        self.assertIsInstance(splits, list)
        self.assertEqual(len(splits), 1)  # ConcreteValidator creates one split
        
        train_data, val_data = splits[0]
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(val_data, pd.DataFrame)
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(val_data), 0)
        self.assertEqual(len(train_data) + len(val_data), len(self.data))
    
    def test_validate(self):
        """Test parameter validation workflow."""
        validator = ConcreteValidator(self.data)
        parameters = {"param1": 10, "param2": 0.5}
        
        result = validator.validate(parameters, self.backtest_function)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.parameters, parameters)
        self.assertEqual(len(result.fold_results), 1)
        self.assertIn("sharpe", result.mean_metrics)
        self.assertIn("return", result.mean_metrics)
        self.assertEqual(result.metadata["n_folds"], 1)
        
        # Verify backtest function was called
        self.backtest_function.assert_called_once()
    
    def test_calculate_summary_statistics_single_fold(self):
        """Test summary statistics calculation with single fold."""
        validator = ConcreteValidator(self.data)
        fold_results = [{"metric1": 0.5, "metric2": 0.3}]
        
        mean_metrics, std_metrics, confidence_intervals = validator._calculate_summary_statistics(fold_results)
        
        self.assertEqual(mean_metrics["metric1"], 0.5)
        self.assertEqual(mean_metrics["metric2"], 0.3)
        self.assertEqual(std_metrics["metric1"], 0.0)  # No variation with single fold
        self.assertEqual(std_metrics["metric2"], 0.0)
        self.assertEqual(confidence_intervals["metric1"], (0.5, 0.5))
        self.assertEqual(confidence_intervals["metric2"], (0.3, 0.3))
    
    def test_calculate_summary_statistics_multiple_folds(self):
        """Test summary statistics calculation with multiple folds."""
        validator = ConcreteValidator(self.data)
        fold_results = [
            {"metric1": 0.4, "metric2": 0.2},
            {"metric1": 0.6, "metric2": 0.4}
        ]
        
        mean_metrics, std_metrics, confidence_intervals = validator._calculate_summary_statistics(fold_results)
        
        self.assertEqual(mean_metrics["metric1"], 0.5)
        self.assertEqual(mean_metrics["metric2"], 0.3)
        self.assertGreater(std_metrics["metric1"], 0)
        self.assertGreater(std_metrics["metric2"], 0)
        
        # Check confidence intervals are reasonable
        ci_metric1 = confidence_intervals["metric1"]
        self.assertLess(ci_metric1[0], mean_metrics["metric1"])
        self.assertGreater(ci_metric1[1], mean_metrics["metric1"])
    
    def test_calculate_summary_statistics_empty(self):
        """Test summary statistics with empty results."""
        validator = ConcreteValidator(self.data)
        
        mean_metrics, std_metrics, confidence_intervals = validator._calculate_summary_statistics([])
        
        self.assertEqual(mean_metrics, {})
        self.assertEqual(std_metrics, {})
        self.assertEqual(confidence_intervals, {})


class TestBaseRobustnessTest(unittest.TestCase):
    """Test cases for BaseRobustnessTest abstract class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = create_btc_sample_data(periods=50)
        self.backtest_function = Mock(return_value={"sharpe": 0.5, "return": 0.1})
    
    def test_initialization(self):
        """Test successful initialization."""
        test = ConcreteRobustnessTest(self.data, random_seed=42)
        
        self.assertTrue(test.data.equals(self.data))
        self.assertEqual(test.random_seed, 42)
    
    def test_run_test(self):
        """Test robustness test execution."""
        test = ConcreteRobustnessTest(self.data, random_seed=42)
        parameters = {"param1": 10}
        
        result = test.run_test(parameters, self.backtest_function, n_simulations=5)
        
        self.assertIsInstance(result, RobustnessResult)
        self.assertEqual(result.parameters, parameters)
        self.assertEqual(len(result.simulation_results), 5)
        self.assertIn("sharpe", result.mean_metrics)
        self.assertIn("return", result.mean_metrics)
        self.assertEqual(result.metadata["n_simulations"], 5)
        
        # Verify backtest function was called multiple times
        self.assertEqual(self.backtest_function.call_count, 5)
    
    def test_calculate_risk_metrics(self):
        """Test VaR and ES calculation."""
        test = ConcreteRobustnessTest(self.data)
        
        # Create test results with known distribution
        results = [
            {"return": val} for val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ]
        
        var_metrics, es_metrics = test._calculate_risk_metrics(results)
        
        # Check that VaR and ES were calculated for different confidence levels
        self.assertIn("return_0.95", var_metrics)
        self.assertIn("return_0.99", var_metrics)
        self.assertIn("return_0.95", es_metrics)
        self.assertIn("return_0.99", es_metrics)
        
        # VaR should be lower percentiles, ES should be even lower
        self.assertLess(var_metrics["return_0.95"], 0.5)  # 5th percentile
        self.assertLess(es_metrics["return_0.95"], var_metrics["return_0.95"])
    
    def test_calculate_risk_metrics_empty(self):
        """Test risk metrics calculation with empty results."""
        test = ConcreteRobustnessTest(self.data)
        
        var_metrics, es_metrics = test._calculate_risk_metrics([])
        
        self.assertEqual(var_metrics, {})
        self.assertEqual(es_metrics, {})


class TestBaseStatisticsTest(unittest.TestCase):
    """Test cases for BaseStatisticsTest abstract class."""
    
    def test_initialization_valid(self):
        """Test successful initialization with valid confidence level."""
        test = ConcreteStatisticsTest(confidence_level=0.95)
        
        self.assertEqual(test.confidence_level, 0.95)
        self.assertEqual(test.alpha, 0.05)
    
    def test_initialization_invalid_confidence_level(self):
        """Test initialization fails with invalid confidence level."""
        with self.assertRaises(ValueError) as context:
            ConcreteStatisticsTest(confidence_level=1.5)
        self.assertIn("Confidence level must be between 0 and 1", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            ConcreteStatisticsTest(confidence_level=0.0)
        self.assertIn("Confidence level must be between 0 and 1", str(context.exception))
    
    def test_run_test_one_sample(self):
        """Test one-sample statistical test."""
        test = ConcreteStatisticsTest(confidence_level=0.95)
        sample1 = [0.1, 0.2, 0.15, 0.18, 0.12]
        
        result = test.run_test(sample1)
        
        self.assertIsInstance(result, StatisticsResult)
        self.assertEqual(result.test_name, "t_test")
        self.assertIsInstance(result.statistic, float)
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.significant, bool)
        self.assertEqual(result.confidence_level, 0.95)
        self.assertEqual(result.metadata["sample1_size"], 5)
        self.assertEqual(result.metadata["sample2_size"], 0)
    
    def test_run_test_two_sample(self):
        """Test two-sample statistical test."""
        test = ConcreteStatisticsTest(confidence_level=0.99)
        sample1 = [0.1, 0.2, 0.15, 0.18, 0.12]
        sample2 = [0.05, 0.08, 0.12, 0.07, 0.09]
        
        result = test.run_test(sample1, sample2)
        
        self.assertIsInstance(result, StatisticsResult)
        self.assertEqual(result.test_name, "t_test")
        self.assertIsInstance(result.statistic, float)
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.significant, bool)
        self.assertEqual(result.confidence_level, 0.99)
        self.assertEqual(result.metadata["sample1_size"], 5)
        self.assertEqual(result.metadata["sample2_size"], 5)
    
    def test_validate_samples_valid(self):
        """Test sample validation with valid inputs."""
        test = ConcreteStatisticsTest()
        
        # Should not raise any exceptions
        test._validate_samples([1, 2, 3])
        test._validate_samples([1.0, 2.5, 3.7], [4, 5, 6])
    
    def test_validate_samples_empty_sample1(self):
        """Test sample validation fails with empty sample1."""
        test = ConcreteStatisticsTest()
        
        with self.assertRaises(ValueError) as context:
            test._validate_samples([])
        self.assertIn("Sample1 cannot be empty", str(context.exception))
    
    def test_validate_samples_non_numeric_sample1(self):
        """Test sample validation fails with non-numeric sample1."""
        test = ConcreteStatisticsTest()
        
        with self.assertRaises(ValueError) as context:
            test._validate_samples([1, 2, "three"])
        self.assertIn("Sample1 must contain only numeric values", str(context.exception))
    
    def test_validate_samples_empty_sample2(self):
        """Test sample validation fails with empty sample2."""
        test = ConcreteStatisticsTest()
        
        with self.assertRaises(ValueError) as context:
            test._validate_samples([1, 2, 3], [])
        self.assertIn("Sample2 cannot be empty if provided", str(context.exception))
    
    def test_validate_samples_non_numeric_sample2(self):
        """Test sample validation fails with non-numeric sample2."""
        test = ConcreteStatisticsTest()
        
        with self.assertRaises(ValueError) as context:
            test._validate_samples([1, 2, 3], [4, 5, "six"])
        self.assertIn("Sample2 must contain only numeric values", str(context.exception))


class TestObjectiveFunction(unittest.TestCase):
    """Test cases for ObjectiveFunction type alias."""
    
    def test_objective_function_callable(self):
        """Test that ObjectiveFunction is properly typed as callable."""
        def sample_objective(params: Dict[str, Any]) -> float:
            return sum(params.values()) if params else 0.0
        
        # This should be a valid ObjectiveFunction
        objective: ObjectiveFunction = sample_objective
        
        # Test the function works as expected
        result = objective({"a": 1, "b": 2})
        self.assertEqual(result, 3.0)
        
        result = objective({})
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()