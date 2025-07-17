"""
Comprehensive validation framework for optimization strategies.

This submodule provides various validation methods, metrics, and integration
utilities to assess the robustness and generalizability of optimized trading
strategies while preventing overfitting and data leakage.

Key Components:
- Data splitting utilities with gap handling
- Walk-forward validation with rolling/expanding windows
- Purged cross-validation for time series
- Enhanced time series split validation
- Comprehensive validation metrics and stability analysis
- Backtester integration utilities
- Validation pipelines for complete workflows

Example Usage:
    >>> from btc_research.optimization.validators import (
    ...     WalkForwardValidator, WalkForwardConfig, WindowType,
    ...     ValidationPipeline, BacktesterIntegrator
    ... )
    >>> 
    >>> # Configure walk-forward validation
    >>> config = WalkForwardConfig(
    ...     window_type=WindowType.EXPANDING,
    ...     training_window_days=90,
    ...     validation_window_days=30
    ... )
    >>> validator = WalkForwardValidator(data, config)
    >>> 
    >>> # Set up validation pipeline
    >>> integrator = BacktesterIntegrator(datafeed, engine)
    >>> pipeline = ValidationPipeline(integrator, [validator])
    >>> 
    >>> # Run comprehensive validation
    >>> report = pipeline.run_comprehensive_validation(
    ...     strategy_config, parameters, data
    ... )
"""

# Core data splitting utilities
from btc_research.optimization.validators.data_splitter import (
    TimeSeriesDataSplitter,
    DataSplitResult,
    SplitConfig,
)

# Enhanced walk-forward validation
from btc_research.optimization.validators.walk_forward import (
    WalkForwardValidator,
    WindowType,
    WalkForwardConfig,
)

# Original validators (maintained for compatibility)
from btc_research.optimization.validators.time_series_split import TimeSeriesSplitValidator
from btc_research.optimization.validators.purged_cv import PurgedCrossValidator

# Enhanced time series validation
from btc_research.optimization.validators.enhanced_time_series_split import (
    EnhancedTimeSeriesSplitValidator,
    SplitStrategy,
    TimeSeriesSplitConfig,
)

# Comprehensive validation metrics
from btc_research.optimization.validators.validation_metrics import (
    ValidationMetricsCalculator,
    StabilityAnalyzer,
    PerformanceDegradationDetector,
    OverfittingDetector,
    ValidationSummaryGenerator,
)

# Integration utilities
from btc_research.optimization.validators.backtester_integration import (
    BacktesterIntegrator,
    ValidationPipeline,
    OptimizationValidationWrapper,
    ValidationBacktestFunction,
)

__all__ = [
    # Data splitting
    "TimeSeriesDataSplitter",
    "DataSplitResult", 
    "SplitConfig",
    
    # Walk-forward validation
    "WalkForwardValidator",
    "WindowType",
    "WalkForwardConfig",
    
    # Original validators
    "TimeSeriesSplitValidator", 
    "PurgedCrossValidator",
    
    # Enhanced validators
    "EnhancedTimeSeriesSplitValidator",
    "SplitStrategy",
    "TimeSeriesSplitConfig",
    
    # Validation metrics
    "ValidationMetricsCalculator",
    "StabilityAnalyzer", 
    "PerformanceDegradationDetector",
    "OverfittingDetector",
    "ValidationSummaryGenerator",
    
    # Integration utilities
    "BacktesterIntegrator",
    "ValidationPipeline",
    "OptimizationValidationWrapper",
    "ValidationBacktestFunction",
]