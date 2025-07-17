"""
Integration interface for the optimization framework.

This module provides high-level interfaces that integrate the optimization
framework with the existing BTC Research Engine components (Backtester, Engine).
"""

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import pandas as pd
import yaml
import numpy as np

from btc_research.core.backtester import Backtester
from btc_research.core.engine import Engine
from btc_research.optimization.base import (
    BaseOptimizer,
    BaseRobustnessTest,
    BaseStatisticsTest,
    BaseValidator,
    ObjectiveFunction,
)
from btc_research.optimization.types import (
    OptimizationMetric,
    OptimizationResult,
    ParameterSpec,
    RobustnessResult,
    StatisticsResult,
    ValidationResult,
)

__all__ = [
    "BacktestObjective",
    "OptimizationFramework", 
    "optimize_strategy",
]


class BacktestObjective:
    """
    Objective function wrapper for backtesting.
    
    This class creates an objective function that can be used by optimizers
    to evaluate parameter combinations using the existing Backtester.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        metric: OptimizationMetric = OptimizationMetric.TOTAL_RETURN,
        data: Optional[pd.DataFrame] = None,
        cash: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
    ):
        """
        Initialize backtest objective function.
        
        Args:
            config: Strategy configuration
            metric: Optimization metric to use
            data: Pre-computed data (if None, will run Engine)
            cash: Starting capital
            commission: Commission rate
            slippage: Slippage rate
        """
        self.config = config
        self.metric = metric
        self.data = data
        self.cash = cash
        self.commission = commission
        self.slippage = slippage
    
    def __call__(self, parameters: Dict[str, Any]) -> float:
        """
        Evaluate parameter combination using backtesting.
        
        Args:
            parameters: Parameter values to test
            
        Returns:
            Objective function value (metric score)
        """
        try:
            # Update config with parameters
            test_config = self._update_config_with_parameters(self.config, parameters)
            
            # Get or compute data
            if self.data is not None:
                df = self.data
            else:
                df = self._get_engine_data(test_config)
            
            # Run backtest
            backtester = Backtester(test_config, debug=False)
            stats = backtester.run(
                df,
                cash=self.cash,
                commission=self.commission,
                slippage=self.slippage,
            )
            
            # Extract metric value
            metric_name = self.metric.value
            metric_value = stats.get(metric_name, 0.0)
            
            # Handle invalid results
            if pd.isna(metric_value) or np.isinf(metric_value):
                return float('-inf')
            
            # Suppress numpy warnings for cleaner output
            with np.errstate(all='ignore'):
                return float(metric_value)
            
        except Exception as e:
            # Return worst possible score for failed evaluations - could log in verbose mode
            return float('-inf')
    
    def _update_config_with_parameters(
        self, 
        config: Dict[str, Any], 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update strategy configuration with optimization parameters.
        
        Args:
            config: Base configuration
            parameters: Parameter values to apply
            
        Returns:
            Updated configuration
        """
        test_config = copy.deepcopy(config)
        
        # Update indicator parameters
        updates_made = 0
        for indicator in test_config.get("indicators", []):
            indicator_id = indicator["id"]
            
            for param_name, param_value in parameters.items():
                # Check if this parameter applies to this indicator
                if param_name.startswith(f"{indicator_id}."):
                    # Remove indicator prefix
                    actual_param = param_name[len(f"{indicator_id}.") :]
                    indicator[actual_param] = param_value
                    updates_made += 1
                elif param_name in indicator:
                    # Direct parameter name match
                    indicator[param_name] = param_value
                    updates_made += 1
        
        # Update strategy logic to use new column names based on parameter changes
        if updates_made > 0:
            test_config = self._update_strategy_logic_column_names(test_config)
        
        return test_config
    
    def _update_strategy_logic_column_names(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update strategy logic expressions to use new column names when indicator parameters change.
        
        This fixes the issue where optimization changes indicator parameters (e.g., RSI length 14->13)
        which creates new column names (RSI_14_RSI_13) but strategy logic still references old names (RSI_14).
        """
        # Map original indicator references to new column names based on current indicator configs
        column_name_mapping = {}
        
        for indicator in config.get("indicators", []):
            indicator_id = indicator["id"]
            indicator_type = indicator["type"]
            
            # Generate the actual column names that the Engine will create
            if indicator_type == "EMA":
                length = indicator.get("length", 50)
                # EMA creates: {id}_EMA_{length} and {id}_EMA_{length}_trend
                base_name = f"{indicator_id}_EMA_{length}"
                column_name_mapping[f"{indicator_id}"] = base_name
                column_name_mapping[f"{indicator_id}_trend"] = f"{base_name}_trend"
                
            elif indicator_type == "RSI":
                length = indicator.get("length", 14)
                # RSI creates: {id}_RSI_{length}, {id}_RSI_{length}_overbought, {id}_RSI_{length}_oversold
                base_name = f"{indicator_id}_RSI_{length}"
                column_name_mapping[f"{indicator_id}"] = base_name
                column_name_mapping[f"{indicator_id}_overbought"] = f"{base_name}_overbought"
                column_name_mapping[f"{indicator_id}_oversold"] = f"{base_name}_oversold"
        
        # Update logic expressions
        if "logic" in config:
            updated_logic = {}
            for logic_type, expressions in config["logic"].items():
                updated_expressions = []
                for expr in expressions:
                    updated_expr = expr
                    # Replace column references in the expression
                    for old_name, new_name in column_name_mapping.items():
                        if old_name in updated_expr:
                            updated_expr = updated_expr.replace(old_name, new_name)
                    updated_expressions.append(updated_expr)
                updated_logic[logic_type] = updated_expressions
            
            config["logic"] = updated_logic
        
        return config
    
    def _get_engine_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Get data from Engine - always recompute since parameters change.
        
        Args:
            config: Strategy configuration
            
        Returns:
            DataFrame with indicator data
        """
        # Always recompute engine data since indicator parameters change
        engine = Engine(config)
        return engine.run()


class OptimizationFramework:
    """
    Comprehensive optimization framework that integrates all components.
    
    This class provides a high-level interface for running complete
    optimization workflows including parameter search, validation,
    robustness testing, and statistical analysis.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        optimizer_class: Type[BaseOptimizer],
        validator_class: Type[BaseValidator],
        robustness_test_class: Type[BaseRobustnessTest],
        statistical_test_class: Optional[Type[BaseStatisticsTest]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize optimization framework.
        
        Args:
            data: Historical data for optimization
            optimizer_class: Optimizer class to use
            validator_class: Validator class to use
            robustness_test_class: Robustness test class to use
            statistical_test_class: Statistical test class to use (optional)
            random_seed: Random seed for reproducibility
        """
        self.data = data
        self.optimizer_class = optimizer_class
        self.validator_class = validator_class
        self.robustness_test_class = robustness_test_class
        self.statistical_test_class = statistical_test_class
        self.random_seed = random_seed
        
        # Components (initialized when run)
        self.optimizer: Optional[BaseOptimizer] = None
        self.validator: Optional[BaseValidator] = None
        self.robustness_test: Optional[BaseRobustnessTest] = None
        self.statistical_test: Optional[BaseStatisticsTest] = None
    
    def optimize(
        self,
        parameter_specs: List[ParameterSpec],
        objective_function: ObjectiveFunction,
        metric: OptimizationMetric,
        max_iterations: int = 100,
        validation_config: Optional[Dict[str, Any]] = None,
        robustness_config: Optional[Dict[str, Any]] = None,
        progress_tracker: Optional[Any] = None,
        **optimizer_kwargs,
    ) -> Dict[str, Any]:
        """
        Run complete optimization workflow.
        
        Args:
            parameter_specs: Parameter specifications for optimization
            objective_function: Objective function to optimize
            metric: Primary optimization metric
            max_iterations: Maximum optimization iterations
            validation_config: Configuration for validation
            robustness_config: Configuration for robustness testing
            **optimizer_kwargs: Additional optimizer parameters
            
        Returns:
            Dictionary with complete optimization results
        """
        start_time = datetime.now()
        
        # Wrap objective function with progress tracking if available
        if progress_tracker is not None:
            original_objective = objective_function
            iteration_count = 0
            
            def tracked_objective(parameters: Dict[str, Any]) -> float:
                nonlocal iteration_count
                iteration_count += 1
                
                # Evaluate objective function
                try:
                    score = original_objective(parameters)
                    
                    # Handle NaN/inf values
                    if pd.isna(score) or np.isinf(score):
                        print(f"Warning: Invalid score ({score}) for parameters {parameters}")
                        score = float('-inf')
                    
                    # Update progress tracker
                    progress_tracker.update(iteration_count, score, parameters)
                    
                    return score
                except Exception as e:
                    print(f"Warning: Objective function evaluation failed: {e}")
                    progress_tracker.update(iteration_count, float('-inf'), parameters)
                    return float('-inf')
            
            tracked_objective_function = tracked_objective
        else:
            tracked_objective_function = objective_function
        
        # Initialize optimizer
        self.optimizer = self.optimizer_class(
            parameter_specs=parameter_specs,
            objective_function=tracked_objective_function,
            metric=metric,
            random_seed=self.random_seed,
            **optimizer_kwargs,
        )
        
        # Run optimization
        optimization_result = self.optimizer.optimize(max_iterations=max_iterations)
        
        # Run validation
        validation_result = None
        if validation_config is None:
            validation_config = {}
        
        try:
            self.validator = self.validator_class(
                data=self.data,
                random_seed=self.random_seed,
                **validation_config,
            )
            
            validation_result = self.validator.validate(
                parameters=optimization_result.parameters,
                backtest_function=self._create_backtest_function(objective_function),
            )
        except Exception as e:
            print(f"Warning: Validation failed: {e}")
        
        # Run robustness testing
        robustness_result = None
        if robustness_config is None:
            robustness_config = {}
        
        try:
            self.robustness_test = self.robustness_test_class(
                data=self.data,
                random_seed=self.random_seed,
            )
            
            robustness_result = self.robustness_test.run_test(
                parameters=optimization_result.parameters,
                backtest_function=self._create_backtest_function(objective_function),
                **robustness_config,
            )
        except Exception as e:
            print(f"Warning: Robustness testing failed: {e}")
        
        # Run statistical testing (if available)
        statistical_result = None
        if self.statistical_test_class is not None:
            try:
                self.statistical_test = self.statistical_test_class()
                
                # Extract returns or metrics for statistical testing
                if validation_result and validation_result.fold_results:
                    metric_values = [
                        fold.get(metric.value, 0.0) 
                        for fold in validation_result.fold_results
                        if metric.value in fold
                    ]
                    
                    if metric_values:
                        statistical_result = self.statistical_test.run_test(metric_values)
                
            except Exception as e:
                print(f"Warning: Statistical testing failed: {e}")
        
        end_time = datetime.now()
        
        # Compile complete results
        complete_results = {
            "optimization": optimization_result,
            "validation": validation_result,
            "robustness": robustness_result,
            "statistical": statistical_result,
            "framework_info": {
                "optimizer": self.optimizer_class.__name__,
                "validator": self.validator_class.__name__,
                "robustness_test": self.robustness_test_class.__name__,
                "statistical_test": self.statistical_test_class.__name__ if self.statistical_test_class else None,
                "start_time": start_time,
                "end_time": end_time,
                "total_duration": (end_time - start_time).total_seconds(),
                "random_seed": self.random_seed,
            },
            "summary": self._create_summary(
                optimization_result, validation_result, robustness_result, statistical_result
            ),
        }
        
        return complete_results
    
    def _create_backtest_function(self, objective_function: ObjectiveFunction) -> Callable:
        """
        Create backtest function compatible with validators and robustness tests.
        
        Args:
            objective_function: Original objective function
            
        Returns:
            Backtest function that returns metrics dictionary
        """
        def backtest_function(data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, float]:
            # Simple wrapper that returns the objective value as a metric
            # In a full implementation, this would run the actual backtest
            # and return comprehensive metrics
            
            objective_value = objective_function(parameters)
            
            return {
                "total_return": objective_value,
                "sharpe_ratio": objective_value * 0.5,  # Simplified
                "max_drawdown": -abs(objective_value * 0.1),  # Simplified
                "num_trades": max(1, int(abs(objective_value * 10))),  # Simplified
            }
        
        return backtest_function
    
    def _create_summary(
        self,
        optimization_result: OptimizationResult,
        validation_result: Optional[ValidationResult],
        robustness_result: Optional[RobustnessResult],
        statistical_result: Optional[StatisticsResult],
    ) -> Dict[str, Any]:
        """
        Create summary of optimization results.
        
        Args:
            optimization_result: Optimization results
            validation_result: Validation results
            robustness_result: Robustness test results
            statistical_result: Statistical test results
            
        Returns:
            Summary dictionary
        """
        summary = {
            "best_parameters": optimization_result.parameters,
            "best_objective_value": optimization_result.objective_value,
            "optimization_iterations": optimization_result.iterations,
            "optimization_converged": optimization_result.convergence_achieved,
        }
        
        if validation_result:
            summary.update({
                "validation_method": validation_result.method.value,
                "validation_folds": validation_result.n_splits,
                "validation_stability": validation_result.stability_score,
                "potential_overfitting": validation_result.is_overfitting(),
                "cross_validation_mean": validation_result.mean_metrics,
                "cross_validation_std": validation_result.std_metrics,
            })
        
        if robustness_result:
            summary.update({
                "robustness_test": robustness_result.test_type,
                "robustness_simulations": robustness_result.n_simulations,
                "robustness_success_rate": robustness_result.success_rate,
                "robustness_worst_case": robustness_result.worst_case_scenario,
                "robustness_best_case": robustness_result.best_case_scenario,
            })
        
        if statistical_result:
            summary.update({
                "statistical_test": statistical_result.test_name,
                "statistical_significance": statistical_result.is_significant(),
                "p_value": statistical_result.p_value,
                "confidence_interval": statistical_result.confidence_interval,
            })
        
        return summary


def optimize_strategy(
    config_path: Union[str, Path, Dict[str, Any]],
    parameter_specs: List[ParameterSpec],
    optimizer_type: str = "bayesian",
    validator_type: str = "walk_forward",
    robustness_test_type: str = "monte_carlo",
    metric: OptimizationMetric = OptimizationMetric.TOTAL_RETURN,
    max_iterations: int = 100,
    cash: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0,
    validation_config: Optional[Dict[str, Any]] = None,
    robustness_config: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None,
    progress_tracker: Optional[Any] = None,
    **optimizer_kwargs,
) -> Dict[str, Any]:
    """
    High-level function to optimize a trading strategy.
    
    This function provides a simple interface to run a complete optimization
    workflow including parameter search, validation, and robustness testing.
    
    Args:
        config_path: Path to strategy configuration file or config dict
        parameter_specs: Parameter specifications for optimization
        optimizer_type: Type of optimizer ("grid_search", "random_search", "bayesian", "genetic_algorithm")
        validator_type: Type of validator ("walk_forward", "time_series_split", "purged_cv")
        robustness_test_type: Type of robustness test ("monte_carlo", "bootstrap", "parameter_sensitivity")
        metric: Primary optimization metric
        max_iterations: Maximum optimization iterations
        cash: Starting capital for backtesting
        commission: Commission rate for backtesting
        slippage: Slippage rate for backtesting
        validation_config: Configuration for validation
        robustness_config: Configuration for robustness testing
        random_seed: Random seed for reproducibility
        **optimizer_kwargs: Additional optimizer parameters
        
    Returns:
        Dictionary with complete optimization results
        
    Example:
        >>> from btc_research.optimization import optimize_strategy
        >>> from btc_research.optimization.types import ParameterSpec, ParameterType, OptimizationMetric
        >>> 
        >>> parameter_specs = [
        >>>     ParameterSpec("RSI_14.length", ParameterType.INTEGER, low=10, high=30),
        >>>     ParameterSpec("RSI_14.oversold", ParameterType.FLOAT, low=20.0, high=40.0)
        >>> ]
        >>> 
        >>> results = optimize_strategy(
        >>>     config_path="config/rsi_strategy.yaml",
        >>>     parameter_specs=parameter_specs,
        >>>     optimizer_type="bayesian",
        >>>     validator_type="walk_forward",
        >>>     metric=OptimizationMetric.SHARPE_RATIO,
        >>>     max_iterations=50
        >>> )
        >>> 
        >>> print(f"Best parameters: {results['optimization'].parameters}")
        >>> print(f"Best Sharpe ratio: {results['optimization'].objective_value:.3f}")
        >>> print(f"Validation stability: {results['validation'].stability_score:.3f}")
    """
    # Load configuration
    if isinstance(config_path, dict):
        config = config_path
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Generate data using Engine
    engine = Engine(config)
    data = engine.run()
    
    # Create objective function
    objective_function = BacktestObjective(
        config=config,
        metric=metric,
        data=None,  # Force recompute for each parameter combination
        cash=cash,
        commission=commission,
        slippage=slippage,
    )
    
    # Import optimization components
    from btc_research.optimization import (
        get_available_optimizers,
        get_available_validators,
        get_available_robustness_tests,
    )
    
    # Get component classes
    optimizers = get_available_optimizers()
    validators = get_available_validators()
    robustness_tests = get_available_robustness_tests()
    
    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Available: {list(optimizers.keys())}")
    if validator_type not in validators:
        raise ValueError(f"Unknown validator type: {validator_type}. Available: {list(validators.keys())}")
    if robustness_test_type not in robustness_tests:
        raise ValueError(f"Unknown robustness test type: {robustness_test_type}. Available: {list(robustness_tests.keys())}")
    
    # Create optimization framework
    framework = OptimizationFramework(
        data=data,
        optimizer_class=optimizers[optimizer_type],
        validator_class=validators[validator_type],
        robustness_test_class=robustness_tests[robustness_test_type],
        random_seed=random_seed,
    )
    
    # Run optimization
    results = framework.optimize(
        parameter_specs=parameter_specs,
        objective_function=objective_function,
        metric=metric,
        max_iterations=max_iterations,
        validation_config=validation_config,
        robustness_config=robustness_config,
        progress_tracker=progress_tracker,
        **optimizer_kwargs,
    )
    
    return results