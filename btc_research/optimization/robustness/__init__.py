"""
Robustness testing for optimization framework.

This submodule provides comprehensive robustness tests to assess how sensitive
optimized strategies are to variations in market conditions, parameters,
data quality, and other factors. It includes Monte Carlo testing, stress testing,
synthetic data generation, and comprehensive robustness metrics.
"""

from btc_research.optimization.robustness.monte_carlo import MonteCarloRobustnessTest
from btc_research.optimization.robustness.bootstrap import BootstrapRobustnessTest
from btc_research.optimization.robustness.parameter_sensitivity import ParameterSensitivityTest
from btc_research.optimization.robustness.data_perturbation import DataPerturbationTest
from btc_research.optimization.robustness.synthetic_data import SyntheticDataTest
from btc_research.optimization.robustness.stress_tests import StressTestFramework
from btc_research.optimization.robustness.robustness_metrics import RobustnessMetrics, RobustnessScoring
from btc_research.optimization.robustness.comprehensive_framework import ComprehensiveRobustnessFramework

__all__ = [
    "MonteCarloRobustnessTest",
    "BootstrapRobustnessTest", 
    "ParameterSensitivityTest",
    "DataPerturbationTest",
    "SyntheticDataTest", 
    "StressTestFramework",
    "RobustnessMetrics",
    "RobustnessScoring",
    "ComprehensiveRobustnessFramework",
]