"""
Optimization algorithms for strategy parameter optimization.

This submodule provides various optimization methods for finding
optimal parameter combinations for trading strategies.
"""

from btc_research.optimization.optimizers.grid_search import GridSearchOptimizer
from btc_research.optimization.optimizers.random_search import (
    RandomSearchOptimizer,
    SamplingStrategy,
)
from btc_research.optimization.optimizers.bayesian import (
    BayesianOptimizer,
    AcquisitionFunction,
)
from btc_research.optimization.optimizers.genetic_algorithm import (
    GeneticAlgorithmOptimizer,
    SelectionMethod,
    CrossoverMethod,
    MutationMethod,
)

__all__ = [
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "BayesianOptimizer", 
    "GeneticAlgorithmOptimizer",
    # Enum classes for configuration
    "SamplingStrategy",
    "AcquisitionFunction",
    "SelectionMethod",
    "CrossoverMethod",
    "MutationMethod",
]