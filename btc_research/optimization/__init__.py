"""
Optimization framework for BTC Research Engine.

This module provides a comprehensive optimization framework to replace the
overfitting-prone grid search approach with robust validation and testing methods.

Key Features:
- Multiple optimization algorithms (Grid Search, Random Search, Bayesian, Genetic)
- Advanced validation strategies (Walk-Forward, Time Series Split, Purged CV)
- Robustness testing (Monte Carlo, Bootstrap, Parameter Sensitivity)
- Statistical significance testing (Hypothesis tests, Performance metrics)
- Integration with existing Backtester and Engine

Example Usage:
    >>> from btc_research.optimization import optimize_strategy
    >>> from btc_research.optimization.optimizers import BayesianOptimizer
    >>> from btc_research.optimization.validators import WalkForwardValidator
    >>> 
    >>> # Define parameter search space
    >>> from btc_research.optimization.types import ParameterSpec, ParameterType
    >>> parameter_specs = [
    >>>     ParameterSpec("rsi_period", ParameterType.INTEGER, low=10, high=30),
    >>>     ParameterSpec("rsi_oversold", ParameterType.FLOAT, low=20.0, high=40.0)
    >>> ]
    >>> 
    >>> # Run optimization with validation
    >>> result = optimize_strategy(
    >>>     config_path="config/strategy.yaml",
    >>>     parameter_specs=parameter_specs,
    >>>     optimizer=BayesianOptimizer,
    >>>     validator=WalkForwardValidator,
    >>>     max_iterations=100
    >>> )
"""

# Core types and data structures
from btc_research.optimization.types import (
    OptimizationMetric,
    OptimizationMethod,
    ValidationMethod,
    ParameterType,
    ParameterSpec,
    OptimizationResult,
    ValidationResult,
    RobustnessResult,
    StatisticsResult,
)

# Base abstract classes
from btc_research.optimization.base import (
    BaseOptimizer,
    BaseValidator,
    BaseRobustnessTest,
    BaseStatisticsTest,
    ObjectiveFunction,
)

# Optimizer implementations
from btc_research.optimization.optimizers import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    GeneticAlgorithmOptimizer,
)

# Validation strategies
from btc_research.optimization.validators import (
    WalkForwardValidator,
    TimeSeriesSplitValidator,
    PurgedCrossValidator,
)

# Robustness testing
from btc_research.optimization.robustness import (
    MonteCarloRobustnessTest,
    BootstrapRobustnessTest,
    ParameterSensitivityTest,
)

# Statistical testing
from btc_research.optimization.statistics import (
    TTestStatistics,
    WilcoxonTestStatistics,
    KolmogorovSmirnovTestStatistics,
    SharpeRatioTestStatistics,
    DrawdownTestStatistics,
    ReturnDistributionTestStatistics,
)

# Integration interface
from btc_research.optimization.integration import (
    optimize_strategy,
    OptimizationFramework,
    BacktestObjective,
)

__version__ = "1.0.0"

__all__ = [
    # Core types
    "OptimizationMetric",
    "OptimizationMethod", 
    "ValidationMethod",
    "ParameterType",
    "ParameterSpec",
    "OptimizationResult",
    "ValidationResult",
    "RobustnessResult",
    "StatisticsResult",
    
    # Base classes
    "BaseOptimizer",
    "BaseValidator",
    "BaseRobustnessTest", 
    "BaseStatisticsTest",
    "ObjectiveFunction",
    
    # Optimizers
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "BayesianOptimizer",
    "GeneticAlgorithmOptimizer",
    
    # Validators
    "WalkForwardValidator",
    "TimeSeriesSplitValidator",
    "PurgedCrossValidator",
    
    # Robustness tests
    "MonteCarloRobustnessTest",
    "BootstrapRobustnessTest",
    "ParameterSensitivityTest",
    
    # Statistical tests
    "TTestStatistics",
    "WilcoxonTestStatistics",
    "KolmogorovSmirnovTestStatistics",
    "SharpeRatioTestStatistics",
    "DrawdownTestStatistics", 
    "ReturnDistributionTestStatistics",
    
    # Integration
    "optimize_strategy",
    "OptimizationFramework",
    "BacktestObjective",
]


def get_available_optimizers():
    """Get list of available optimization algorithms."""
    return {
        "grid_search": GridSearchOptimizer,
        "random_search": RandomSearchOptimizer,
        "bayesian": BayesianOptimizer,
        "genetic_algorithm": GeneticAlgorithmOptimizer,
    }


def get_available_validators():
    """Get list of available validation strategies."""
    return {
        "walk_forward": WalkForwardValidator,
        "time_series_split": TimeSeriesSplitValidator,
        "purged_cv": PurgedCrossValidator,
    }


def get_available_robustness_tests():
    """Get list of available robustness tests."""
    return {
        "monte_carlo": MonteCarloRobustnessTest,
        "bootstrap": BootstrapRobustnessTest,
        "parameter_sensitivity": ParameterSensitivityTest,
    }


def get_available_statistical_tests():
    """Get list of available statistical tests."""
    return {
        "t_test": TTestStatistics,
        "wilcoxon": WilcoxonTestStatistics,
        "kolmogorov_smirnov": KolmogorovSmirnovTestStatistics,
        "sharpe_ratio": SharpeRatioTestStatistics,
        "drawdown": DrawdownTestStatistics,
        "return_distribution": ReturnDistributionTestStatistics,
    }


def create_optimization_framework(
    data,
    optimizer_type: str = "bayesian",
    validator_type: str = "walk_forward",
    robustness_test_type: str = "monte_carlo",
    **kwargs
) -> "OptimizationFramework":
    """
    Create a complete optimization framework with specified components.
    
    Args:
        data: Historical data for optimization
        optimizer_type: Type of optimizer to use
        validator_type: Type of validator to use
        robustness_test_type: Type of robustness test to use
        **kwargs: Additional parameters for components
        
    Returns:
        Configured optimization framework
    """
    # Get component classes
    optimizers = get_available_optimizers()
    validators = get_available_validators()
    robustness_tests = get_available_robustness_tests()
    
    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    if validator_type not in validators:
        raise ValueError(f"Unknown validator type: {validator_type}")
    if robustness_test_type not in robustness_tests:
        raise ValueError(f"Unknown robustness test type: {robustness_test_type}")
    
    # Create optimization framework
    framework = OptimizationFramework(
        data=data,
        optimizer_class=optimizers[optimizer_type],
        validator_class=validators[validator_type],
        robustness_test_class=robustness_tests[robustness_test_type],
        **kwargs
    )
    
    return framework