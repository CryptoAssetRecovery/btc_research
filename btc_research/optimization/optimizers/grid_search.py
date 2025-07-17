"""
Grid search optimization for exhaustive parameter space exploration.

Grid search evaluates all possible combinations of parameter values
within specified ranges. While computationally expensive, it guarantees
finding the global optimum within the discrete search space.
"""

import itertools
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from btc_research.optimization.base import BaseOptimizer, ObjectiveFunction
from btc_research.optimization.types import (
    OptimizationMethod,
    OptimizationMetric,
    OptimizationResult,
    ParameterSpec,
    ParameterType,
)

__all__ = ["GridSearchOptimizer"]


class GridSearchOptimizer(BaseOptimizer):
    """
    Grid search optimization implementation.
    
    This optimizer evaluates all possible combinations of parameter values
    within the specified search space. It's exhaustive but guaranteed to
    find the best combination within the discrete grid.
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
        Initialize grid search optimizer.
        
        Args:
            parameter_specs: List of parameter specifications defining search space
            objective_function: Function to optimize (takes params dict, returns score)
            metric: Primary optimization metric
            maximize: Whether to maximize (True) or minimize (False) the objective
            random_seed: Random seed for reproducibility (not used in grid search)
        """
        super().__init__(parameter_specs, objective_function, metric, maximize, random_seed)
        
        # Generate all parameter combinations
        self._parameter_combinations = list(self._generate_parameter_combinations())
        self._current_index = 0
        
        if not self._parameter_combinations:
            raise ValueError("No parameter combinations generated from specifications")
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all possible parameter combinations from specifications.
        
        Returns:
            List of parameter dictionaries
        """
        param_values = []
        param_names = []
        
        for spec in self.parameter_specs:
            param_names.append(spec.name)
            
            if spec.param_type == ParameterType.INTEGER:
                if spec.step is not None:
                    values = list(range(int(spec.low), int(spec.high) + 1, int(spec.step)))
                else:
                    values = list(range(int(spec.low), int(spec.high) + 1))
                    
            elif spec.param_type == ParameterType.FLOAT:
                if spec.step is not None:
                    # Generate float range with step
                    values = []
                    current = spec.low
                    while current <= spec.high:
                        values.append(current)
                        current += spec.step
                else:
                    # Default to 10 values between low and high
                    n_values = 10
                    step = (spec.high - spec.low) / (n_values - 1)
                    values = [spec.low + i * step for i in range(n_values)]
                    
            elif spec.param_type == ParameterType.CATEGORICAL:
                values = spec.choices.copy()
                
            elif spec.param_type == ParameterType.BOOLEAN:
                values = [True, False]
                
            else:
                raise ValueError(f"Unsupported parameter type: {spec.param_type}")
            
            param_values.append(values)
        
        # Generate all combinations
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """
        Suggest next parameter combination to evaluate.
        
        Returns:
            Dictionary of parameter names and suggested values
        """
        if self._current_index >= len(self._parameter_combinations):
            # All combinations have been evaluated
            raise StopIteration("All parameter combinations have been evaluated")
        
        params = self._parameter_combinations[self._current_index]
        self._current_index += 1
        
        return params
    
    def optimize(
        self,
        max_iterations: int = 100,
        timeout_seconds: Optional[float] = None,
        convergence_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Run grid search optimization.
        
        Args:
            max_iterations: Maximum number of iterations (ignored, uses all combinations)
            timeout_seconds: Maximum time to run optimization
            convergence_threshold: Convergence threshold (not applicable to grid search)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Best optimization result found
        """
        start_time = datetime.now()
        
        # Reset state
        self._current_index = 0
        self._all_results.clear()
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None
        best_metrics = None
        
        total_combinations = len(self._parameter_combinations)
        
        # Evaluate all combinations or until timeout
        for i in range(total_combinations):
            # Check timeout
            if timeout_seconds:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    print(f"Grid search stopped due to timeout after {i} evaluations")
                    break
            
            try:
                # Get next parameter combination
                params = self.suggest_parameters()
                
                # Evaluate parameters
                score = self.evaluate_parameters(params)
                
                # Update best result
                if self.maximize and score > best_score:
                    best_score = score
                    best_params = params.copy()
                elif not self.maximize and score < best_score:
                    best_score = score
                    best_params = params.copy()
                
                # Store result (simplified - in full implementation would call backtest)
                self.update_with_result(params, score)
                
            except StopIteration:
                # All combinations evaluated
                break
            except Exception as e:
                print(f"Warning: Evaluation failed for parameters {params}: {e}")
                continue
        
        end_time = datetime.now()
        
        if best_params is None:
            raise ValueError("No valid parameter combinations found")
        
        # Create optimization result
        result = OptimizationResult(
            parameters=best_params,
            metrics={"objective_value": best_score},  # Simplified
            objective_value=best_score,
            in_sample_metrics={"objective_value": best_score},
            method=OptimizationMethod.GRID_SEARCH,
            metric=self.metric,
            start_time=start_time,
            end_time=end_time,
            iterations=self._current_index,
            convergence_achieved=True,  # Grid search always converges
            diagnostics={
                "total_combinations": total_combinations,
                "evaluated_combinations": self._current_index,
                "completion_rate": self._current_index / total_combinations,
            }
        )
        
        self._best_result = result
        return result
    
    def get_remaining_combinations(self) -> int:
        """
        Get number of remaining parameter combinations to evaluate.
        
        Returns:
            Number of combinations not yet evaluated
        """
        return max(0, len(self._parameter_combinations) - self._current_index)
    
    def get_total_combinations(self) -> int:
        """
        Get total number of parameter combinations.
        
        Returns:
            Total number of combinations in the grid
        """
        return len(self._parameter_combinations)
    
    def get_progress(self) -> float:
        """
        Get optimization progress as a percentage.
        
        Returns:
            Progress percentage (0.0 to 1.0)
        """
        if not self._parameter_combinations:
            return 1.0
        
        return self._current_index / len(self._parameter_combinations)
    
    def reset(self) -> None:
        """Reset the optimizer to start from the beginning."""
        self._current_index = 0
        self._all_results.clear()
        self._best_result = None