"""
Statistical significance testing for optimization framework.

This submodule provides comprehensive statistical tests to assess the significance
of optimization results and compare strategy performance using both classical
and modern statistical methods.
"""

# Basic hypothesis tests
from btc_research.optimization.statistics.hypothesis_tests import (
    TTestStatistics,
    WilcoxonTestStatistics,
    KolmogorovSmirnovTestStatistics,
    MultipleComparisonCorrection,
    PowerAnalysis,
)

# Performance-specific tests
from btc_research.optimization.statistics.performance_tests import (
    SharpeRatioTestStatistics,
    DrawdownTestStatistics,
    ReturnDistributionTestStatistics,
)

# Advanced statistical tests
from btc_research.optimization.statistics.advanced_tests import (
    BootstrapTest,
    PermutationTest,
    DieboldMarianoTest,
    WhiteRealityCheck,
)

# Model selection and comparison
from btc_research.optimization.statistics.model_selection import (
    InformationCriteria,
    CrossValidationStatistics,
    ModelAveraging,
    OverfittingDetection,
    StrategyRankingStatistics,
)

# Bayesian analysis
from btc_research.optimization.statistics.bayesian_analysis import (
    BayesianModelComparison,
    BayesianParameterEstimation,
    BayesianOptimizationUncertainty,
    PriorSensitivityAnalysis,
)

# Time series specific tests
from btc_research.optimization.statistics.time_series_tests import (
    StationarityTests,
    AutocorrelationAnalysis,
    RegimeChangeDetection,
    VolatilityClusteringTests,
    MarketEfficiencyTests,
)

# Bootstrap-enhanced performance tests
from btc_research.optimization.statistics.bootstrap_performance import (
    BootstrapPerformanceTests,
)

# Comprehensive reporting
from btc_research.optimization.statistics.statistical_reports import (
    StatisticalReportGenerator,
    MultipleTestingCorrector,
    EffectSizeCalculator,
    PowerAnalysisReporter,
    VisualizationHelper,
)

__all__ = [
    # Basic hypothesis tests
    "TTestStatistics",
    "WilcoxonTestStatistics", 
    "KolmogorovSmirnovTestStatistics",
    "MultipleComparisonCorrection",
    "PowerAnalysis",
    
    # Performance tests
    "SharpeRatioTestStatistics",
    "DrawdownTestStatistics",
    "ReturnDistributionTestStatistics",
    "BootstrapPerformanceTests",
    
    # Advanced tests
    "BootstrapTest",
    "PermutationTest",
    "DieboldMarianoTest",
    "WhiteRealityCheck",
    
    # Model selection
    "InformationCriteria",
    "CrossValidationStatistics",
    "ModelAveraging",
    "OverfittingDetection",
    "StrategyRankingStatistics",
    
    # Bayesian analysis
    "BayesianModelComparison",
    "BayesianParameterEstimation",
    "BayesianOptimizationUncertainty",
    "PriorSensitivityAnalysis",
    
    # Time series tests
    "StationarityTests",
    "AutocorrelationAnalysis",
    "RegimeChangeDetection",
    "VolatilityClusteringTests",
    "MarketEfficiencyTests",
    
    # Reporting and analysis
    "StatisticalReportGenerator",
    "MultipleTestingCorrector",
    "EffectSizeCalculator",
    "PowerAnalysisReporter",
    "VisualizationHelper",
]