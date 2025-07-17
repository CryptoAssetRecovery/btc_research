"""
Base abstract classes for the optimization framework.

This module defines the core interfaces that all optimization components
must implement to ensure consistency and interoperability.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from btc_research.optimization.types import (
    OptimizationMetric,
    OptimizationResult,
    ParameterSpec,
    RobustnessResult,
    StatisticsResult,
    ValidationResult,
)

__all__ = [
    "BaseOptimizer",
    "BaseValidator", 
    "BaseRobustnessTest",
    "BaseStatisticsTest",
    "ObjectiveFunction",
]


# Type alias for objective functions
ObjectiveFunction = Callable[[Dict[str, Any]], float]


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization algorithms.
    
    Defines the interface that all optimizers must implement to work
    with the optimization framework. Optimizers search for the best
    parameter combinations using different strategies.
    """
    
    def __init__(
        self,
        parameter_specs: List[ParameterSpec],
        objective_function: ObjectiveFunction,
        metric: OptimizationMetric,
        maximize: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the optimizer.
        
        Args:
            parameter_specs: List of parameter specifications defining search space
            objective_function: Function to optimize (takes params dict, returns score)
            metric: Primary optimization metric
            maximize: Whether to maximize (True) or minimize (False) the objective
            random_seed: Random seed for reproducibility
        """
        self.parameter_specs = parameter_specs
        self.objective_function = objective_function
        self.metric = metric
        self.maximize = maximize
        self.random_seed = random_seed
        
        # Validation
        if not parameter_specs:
            raise ValueError("At least one parameter specification is required")
        
        # Internal state
        self._best_result: Optional[OptimizationResult] = None
        self._all_results: List[OptimizationResult] = []
        self._iteration_count = 0
        
    @abstractmethod
    def optimize(
        self,
        max_iterations: int = 100,
        timeout_seconds: Optional[float] = None,
        convergence_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Run the optimization algorithm.
        
        Args:
            max_iterations: Maximum number of iterations/evaluations
            timeout_seconds: Maximum time to run optimization
            convergence_threshold: Stop if improvement falls below this threshold
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Best optimization result found
        """
        pass
    
    @abstractmethod
    def suggest_parameters(self) -> Dict[str, Any]:
        """
        Suggest next parameter combination to evaluate.
        
        Returns:
            Dictionary of parameter names and suggested values
        """
        pass
    
    def evaluate_parameters(self, parameters: Dict[str, Any]) -> float:
        """
        Evaluate a parameter combination using the objective function.
        
        Args:
            parameters: Parameter values to evaluate
            
        Returns:
            Objective function value
        """
        try:
            score = self.objective_function(parameters)
            if not self.maximize:
                score = -score
            return score
        except Exception as e:
            # Return worst possible score for failed evaluations
            return float('-inf') if self.maximize else float('inf')
    
    def update_with_result(self, parameters: Dict[str, Any], score: float) -> None:
        """
        Update optimizer state with evaluation result.
        
        Args:
            parameters: Parameter values that were evaluated
            score: Objective function result
        """
        self._iteration_count += 1
        # Subclasses can override this to update internal models
        pass
    
    @property
    def best_result(self) -> Optional[OptimizationResult]:
        """Get the best result found so far."""
        return self._best_result
    
    @property
    def all_results(self) -> List[OptimizationResult]:
        """Get all evaluation results."""
        return self._all_results.copy()
    
    @property
    def iteration_count(self) -> int:
        """Get current iteration count."""
        return self._iteration_count
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate that parameter values are within specified bounds.
        
        Args:
            parameters: Parameter values to validate
            
        Returns:
            True if all parameters are valid
        """
        for spec in self.parameter_specs:
            if spec.name not in parameters:
                return False
            
            value = parameters[spec.name]
            
            # Type-specific validation
            if spec.param_type.value == "integer":
                if not isinstance(value, int):
                    return False
                if spec.low is not None and value < spec.low:
                    return False
                if spec.high is not None and value > spec.high:
                    return False
                    
            elif spec.param_type.value == "float":
                if not isinstance(value, (int, float)):
                    return False
                if spec.low is not None and value < spec.low:
                    return False
                if spec.high is not None and value > spec.high:
                    return False
                    
            elif spec.param_type.value == "categorical":
                if spec.choices and value not in spec.choices:
                    return False
                    
            elif spec.param_type.value == "boolean":
                if not isinstance(value, bool):
                    return False
        
        return True


class BaseValidator(ABC):
    """
    Abstract base class for validation strategies.
    
    Validators assess the robustness and generalizability of optimization
    results by testing performance on out-of-sample data.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        date_column: str = "timestamp",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the validator.
        
        Args:
            data: Time series data for validation
            date_column: Name of the datetime column
            random_seed: Random seed for reproducibility
        """
        self.data = data
        self.date_column = date_column
        self.random_seed = random_seed
        
        # Validate data
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        if date_column not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(f"Date column '{date_column}' not found in data")
    
    @abstractmethod
    def split_data(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into training and validation sets.
        
        Returns:
            List of (train_data, validation_data) tuples
        """
        pass
    
    @abstractmethod
    def validate(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
    ) -> ValidationResult:
        """
        Validate parameter combination using the validation strategy.
        
        Args:
            parameters: Parameter values to validate
            backtest_function: Function that runs backtest and returns metrics
            
        Returns:
            Validation result with cross-fold performance
        """
        pass
    
    def _calculate_summary_statistics(
        self, 
        fold_results: List[Dict[str, float]]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Tuple[float, float]]]:
        """
        Calculate mean, std, and confidence intervals across folds.
        
        Args:
            fold_results: List of metric dictionaries from each fold
            
        Returns:
            Tuple of (mean_metrics, std_metrics, confidence_intervals)
        """
        if not fold_results:
            return {}, {}, {}
        
        # Get all metric names
        all_metrics = set()
        for result in fold_results:
            all_metrics.update(result.keys())
        
        mean_metrics = {}
        std_metrics = {}
        confidence_intervals = {}
        
        for metric in all_metrics:
            values = [result.get(metric, 0.0) for result in fold_results]
            
            mean_val = sum(values) / len(values)
            
            # Calculate standard deviation
            if len(values) > 1:
                variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
                std_val = variance ** 0.5
            else:
                std_val = 0.0
            
            # Calculate 95% confidence interval (assuming normal distribution)
            if len(values) > 1:
                margin = 1.96 * std_val / (len(values) ** 0.5)  # 95% CI
                ci = (mean_val - margin, mean_val + margin)
            else:
                ci = (mean_val, mean_val)
            
            mean_metrics[metric] = mean_val
            std_metrics[metric] = std_val
            confidence_intervals[metric] = ci
        
        return mean_metrics, std_metrics, confidence_intervals


class BaseRobustnessTest(ABC):
    """
    Abstract base class for robustness testing.
    
    Robustness tests assess how sensitive a strategy is to variations
    in market conditions, parameter values, or other factors.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the robustness test.
        
        Args:
            data: Historical data for testing
            random_seed: Random seed for reproducibility
        """
        self.data = data
        self.random_seed = random_seed
    
    @abstractmethod
    def run_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 1000,
        **kwargs: Any,
    ) -> RobustnessResult:
        """
        Run robustness test on parameter combination.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of simulations to run
            **kwargs: Test-specific parameters
            
        Returns:
            Robustness test result
        """
        pass
    
    def _calculate_risk_metrics(
        self, 
        results: List[Dict[str, float]]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate Value at Risk and Expected Shortfall.
        
        Args:
            results: List of simulation results
            
        Returns:
            Tuple of (VaR_dict, ES_dict) for different confidence levels
        """
        if not results:
            return {}, {}
        
        # Get all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.keys())
        
        var_results = {}
        es_results = {}
        confidence_levels = [0.95, 0.99]
        
        for metric in all_metrics:
            values = [result.get(metric, 0.0) for result in results]
            values.sort()
            
            for conf_level in confidence_levels:
                # Value at Risk (percentile)
                var_idx = int((1 - conf_level) * len(values))
                if var_idx < len(values):
                    var_value = values[var_idx]
                else:
                    var_value = values[0] if values else 0.0
                
                # Expected Shortfall (mean of tail)
                tail_values = values[:var_idx + 1] if var_idx >= 0 else [values[0]]
                es_value = sum(tail_values) / len(tail_values) if tail_values else 0.0
                
                var_results[f"{metric}_{conf_level}"] = var_value
                es_results[f"{metric}_{conf_level}"] = es_value
        
        return var_results, es_results


class BaseStatisticsTest(ABC):
    """
    Abstract base class for statistical significance testing.
    
    Statistical tests help determine if observed performance differences
    are statistically significant rather than due to random chance.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the statistical test.
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    @abstractmethod
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Run statistical test on sample data.
        
        Args:
            sample1: First sample (e.g., strategy returns)
            sample2: Second sample for comparison tests (optional)
            **kwargs: Test-specific parameters
            
        Returns:
            Statistical test result
        """
        pass
    
    def _validate_samples(
        self, 
        sample1: List[float], 
        sample2: Optional[List[float]] = None
    ) -> None:
        """
        Validate input samples.
        
        Args:
            sample1: First sample
            sample2: Second sample (optional)
        """
        if not sample1:
            raise ValueError("Sample1 cannot be empty")
        
        if any(not isinstance(x, (int, float)) for x in sample1):
            raise ValueError("Sample1 must contain only numeric values")
        
        if sample2 is not None:
            if not sample2:
                raise ValueError("Sample2 cannot be empty if provided")
            
            if any(not isinstance(x, (int, float)) for x in sample2):
                raise ValueError("Sample2 must contain only numeric values")