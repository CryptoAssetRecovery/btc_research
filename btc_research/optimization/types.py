"""
Type definitions and data classes for the optimization framework.

This module provides common types and data structures used throughout
the optimization framework for type safety and consistency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "OptimizationMetric",
    "OptimizationMethod", 
    "ValidationMethod",
    "ParameterType",
    "ParameterSpec",
    "OptimizationResult",
    "ValidationResult",
    "RobustnessResult",
    "StatisticsResult",
]


class OptimizationMetric(Enum):
    """Supported optimization metrics."""
    
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    EXPECTANCY = "expectancy"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


class OptimizationMethod(Enum):
    """Supported optimization methods."""
    
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"


class ValidationMethod(Enum):
    """Supported validation methods."""
    
    WALK_FORWARD = "walk_forward"
    TIME_SERIES_SPLIT = "time_series_split"
    PURGED_CROSS_VALIDATION = "purged_cross_validation"
    COMBINATORIAL_PURGED_CV = "combinatorial_purged_cv"
    MONTE_CARLO_CV = "monte_carlo_cv"


class ParameterType(Enum):
    """Supported parameter types for optimization."""
    
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class ParameterSpec:
    """
    Specification for an optimization parameter.
    
    Defines the search space and constraints for a single parameter
    in the optimization process.
    """
    
    name: str
    param_type: ParameterType
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    step: Optional[Union[int, float]] = None
    log_scale: bool = False
    
    def __post_init__(self) -> None:
        """Validate parameter specification."""
        if self.param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
            if self.low is None or self.high is None:
                raise ValueError(f"Parameter {self.name}: low and high must be specified for {self.param_type.value}")
            if self.low >= self.high:
                raise ValueError(f"Parameter {self.name}: low must be less than high")
        
        if self.param_type == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"Parameter {self.name}: choices must be specified for categorical parameters")
        
        if self.param_type == ParameterType.BOOLEAN:
            if self.choices is None:
                self.choices = [True, False]


@dataclass
class OptimizationResult:
    """
    Complete result from an optimization run.
    
    Contains all relevant information about the optimization process,
    including parameter values, performance metrics, and metadata.
    """
    
    # Core results
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    objective_value: float
    
    # Metadata (required fields)
    method: OptimizationMethod
    metric: OptimizationMetric
    
    # Validation results
    in_sample_metrics: Dict[str, float]
    out_of_sample_metrics: Optional[Dict[str, float]] = None
    validation_scores: Optional[List[float]] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    iterations: int = 0
    convergence_achieved: bool = False
    
    # Additional data
    equity_curve: Optional[pd.DataFrame] = None
    trade_list: Optional[pd.DataFrame] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Calculate duration if end_time is set."""
        if self.end_time and self.duration_seconds is None:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    @property
    def is_valid(self) -> bool:
        """Check if the optimization result is valid."""
        return (
            bool(self.parameters) and
            bool(self.metrics) and
            not np.isnan(self.objective_value) and
            not np.isinf(self.objective_value)
        )
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get parameter value by name."""
        return self.parameters.get(name, default)
    
    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get metric value by name."""
        return self.metrics.get(name, default)


@dataclass
class ValidationResult:
    """
    Result from a validation strategy.
    
    Contains metrics for each fold/split and summary statistics
    for model stability assessment.
    """
    
    method: ValidationMethod
    fold_results: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    confidence_intervals: Dict[str, tuple[float, float]]
    
    # Metadata
    n_splits: int
    data_split_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def stability_score(self) -> float:
        """
        Calculate stability score based on coefficient of variation.
        
        Lower values indicate more stable performance across folds.
        """
        if not self.fold_results:
            return float('inf')
        
        # Use primary metric (usually the first one) for stability calculation
        primary_metric = next(iter(self.mean_metrics.keys()))
        mean_val = self.mean_metrics[primary_metric]
        std_val = self.std_metrics[primary_metric]
        
        if mean_val == 0:
            return float('inf')
        
        return abs(std_val / mean_val)  # Coefficient of variation
    
    def is_overfitting(self, threshold: float = 0.3) -> bool:
        """
        Check if results indicate potential overfitting.
        
        Args:
            threshold: Stability threshold above which overfitting is suspected
            
        Returns:
            True if stability score exceeds threshold
        """
        return self.stability_score > threshold


@dataclass
class RobustnessResult:
    """
    Result from robustness testing.
    
    Contains results from Monte Carlo simulations and other
    robustness tests to assess strategy stability.
    """
    
    test_type: str
    n_simulations: int
    results: List[Dict[str, float]]
    summary_stats: Dict[str, Dict[str, float]]  # metric -> {mean, std, percentiles}
    
    # Risk metrics
    value_at_risk: Dict[str, float]  # VaR at different confidence levels
    expected_shortfall: Dict[str, float]  # ES at different confidence levels
    
    # Stability metrics
    success_rate: float  # Percentage of simulations that met criteria
    worst_case_scenario: Dict[str, float]
    best_case_scenario: Dict[str, float]
    
    def get_percentile(self, metric: str, percentile: float) -> float:
        """Get percentile value for a specific metric."""
        values = [result[metric] for result in self.results if metric in result]
        if not values:
            return np.nan
        return np.percentile(values, percentile)


@dataclass 
class StatisticsResult:
    """
    Result from statistical significance testing.
    
    Contains p-values, confidence intervals, and other statistical
    measures for hypothesis testing.
    """
    
    test_name: str
    statistic: float
    p_value: float
    confidence_level: float
    confidence_interval: tuple[float, float]
    
    # Additional test-specific results
    effect_size: Optional[float] = None
    power: Optional[float] = None
    critical_value: Optional[float] = None
    
    # Metadata
    sample_size: int = 0
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        return self.p_value < alpha
    
    def get_interpretation(self, alpha: float = 0.05) -> str:
        """Get human-readable interpretation of the test result."""
        if self.is_significant(alpha):
            return f"Statistically significant (p={self.p_value:.4f} < {alpha})"
        else:
            return f"Not statistically significant (p={self.p_value:.4f} >= {alpha})"