"""
Enhanced time series split validation for trading strategy optimization.

This enhanced version provides purged k-fold cross-validation with overlapping
period handling, temporal dependencies, and comprehensive validation metrics
to prevent data leakage and overfitting.
"""

import logging
import threading
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from btc_research.optimization.base import BaseValidator
from btc_research.optimization.types import ValidationMethod, ValidationResult
from btc_research.optimization.validators.data_splitter import TimeSeriesDataSplitter

# Set up logging
logger = logging.getLogger(__name__)

__all__ = ["EnhancedTimeSeriesSplitValidator", "SplitStrategy", "TimeSeriesSplitConfig"]


class SplitStrategy(Enum):
    """Time series split strategies."""
    EXPANDING = "expanding"  # Growing training set
    ROLLING = "rolling"      # Fixed-size training set
    PURGED = "purged"       # Purged cross-validation
    COMBINATORIAL = "combinatorial"  # Combinatorial purged CV


class TimeSeriesSplitConfig:
    """Configuration for enhanced time series split validation."""
    
    def __init__(
        self,
        strategy: SplitStrategy = SplitStrategy.EXPANDING,
        n_splits: int = 5,
        test_size_ratio: float = 0.2,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.01,
        min_training_samples: int = 100,
        min_test_samples: int = 20,
        overlap_tolerance: float = 0.05,
        enable_combinatorial: bool = False,
        max_combinations: Optional[int] = None,
    ):
        """
        Initialize time series split configuration.
        
        Args:
            strategy: Split strategy to use
            n_splits: Number of splits to create
            test_size_ratio: Proportion of data for testing
            purge_pct: Percentage of data to purge around test sets
            embargo_pct: Percentage of data to embargo after test sets
            min_training_samples: Minimum samples in training set
            min_test_samples: Minimum samples in test set
            overlap_tolerance: Tolerance for overlapping periods
            enable_combinatorial: Whether to use combinatorial purged CV
            max_combinations: Maximum number of combinations for combinatorial CV
        """
        self.strategy = strategy
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.min_training_samples = min_training_samples
        self.min_test_samples = min_test_samples
        self.overlap_tolerance = overlap_tolerance
        self.enable_combinatorial = enable_combinatorial
        self.max_combinations = max_combinations
        
        # Validation
        if n_splits < 2:
            raise ValueError("Number of splits must be at least 2")
        if not 0 < test_size_ratio < 1:
            raise ValueError("Test size ratio must be between 0 and 1")
        if not 0 <= purge_pct < 1:
            raise ValueError("Purge percentage must be between 0 and 1")
        if not 0 <= embargo_pct < 1:
            raise ValueError("Embargo percentage must be between 0 and 1")


class EnhancedTimeSeriesSplitValidator(BaseValidator):
    """
    Enhanced time series split validation implementation.
    
    This validator provides multiple strategies for time series cross-validation:
    1. Expanding window: Each split adds more historical data
    2. Rolling window: Fixed-size training windows
    3. Purged CV: Addresses temporal data leakage
    4. Combinatorial purged CV: Multiple non-overlapping test sets
    
    Features comprehensive validation metrics and overfitting detection.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[TimeSeriesSplitConfig] = None,
        date_column: str = "timestamp",
        random_seed: Optional[int] = None,
        thread_safe: bool = True,
    ):
        """
        Initialize enhanced time series split validator.
        
        Args:
            data: Time series data for validation
            config: Split configuration (uses defaults if None)
            date_column: Name of the datetime column
            random_seed: Random seed for reproducibility
            thread_safe: Whether to enable thread-safe operations
        """
        super().__init__(data, date_column, random_seed)
        
        self.config = config if config is not None else TimeSeriesSplitConfig()
        self.thread_safe = thread_safe
        
        # Initialize data splitter
        self.data_splitter = TimeSeriesDataSplitter(
            date_column=date_column,
            random_seed=random_seed,
            thread_safe=thread_safe,
        )
        
        # Thread safety
        self._lock = threading.RLock() if thread_safe else None
        
        # Ensure data is sorted by date
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data = self.data.sort_index()
        else:
            self.data = self.data.sort_values(date_column)
        
        # Caches
        self._splits_cache: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None
        self._validation_metrics_cache: Optional[Dict[str, Any]] = None
    
    def split_data(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create time series training/validation splits based on strategy.
        
        Returns:
            List of (train_data, validation_data) tuples
        """
        with self._get_lock():
            if self._splits_cache is not None:
                return self._splits_cache
            
            # Route to appropriate split strategy
            if self.config.strategy == SplitStrategy.EXPANDING:
                splits = self._create_expanding_splits()
            elif self.config.strategy == SplitStrategy.ROLLING:
                splits = self._create_rolling_splits()
            elif self.config.strategy == SplitStrategy.PURGED:
                splits = self._create_purged_splits()
            elif self.config.strategy == SplitStrategy.COMBINATORIAL:
                splits = self._create_combinatorial_splits()
            else:
                raise ValueError(f"Unknown split strategy: {self.config.strategy}")
            
            if not splits:
                raise ValueError("No valid splits could be created with current configuration")
            
            # Validate splits for data leakage
            self._validate_splits_integrity(splits)
            
            logger.info(f"Created {len(splits)} time series splits using {self.config.strategy.value} strategy")
            self._splits_cache = splits
            return splits
    
    def _create_expanding_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create expanding window splits."""
        data_length = len(self.data)
        test_size = max(self.config.min_test_samples, int(data_length * self.config.test_size_ratio))
        
        splits = []
        
        for split_idx in range(self.config.n_splits):
            # Calculate test set position
            test_end = data_length - (self.config.n_splits - split_idx - 1) * (test_size // self.config.n_splits)
            test_start = test_end - test_size
            test_start = max(0, test_start)
            
            # Training set includes all data before test set
            train_end = test_start
            train_start = 0
            
            # Skip if training set is too small
            if train_end - train_start < self.config.min_training_samples:
                continue
            
            # Create splits
            train_data = self.data.iloc[train_start:train_end].copy()
            test_data = self.data.iloc[test_start:test_end].copy()
            
            if len(train_data) >= self.config.min_training_samples and len(test_data) >= self.config.min_test_samples:
                splits.append((train_data, test_data))
        
        return splits
    
    def _create_rolling_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create rolling window splits."""
        data_length = len(self.data)
        test_size = max(self.config.min_test_samples, int(data_length * self.config.test_size_ratio))
        
        # Calculate training window size
        available_for_training = data_length - test_size * self.config.n_splits
        train_size = max(self.config.min_training_samples, available_for_training // self.config.n_splits)
        
        splits = []
        
        for split_idx in range(self.config.n_splits):
            # Calculate positions
            test_start = split_idx * (test_size + (data_length - test_size * self.config.n_splits) // self.config.n_splits)
            test_end = test_start + test_size
            
            train_start = max(0, test_start - train_size)
            train_end = test_start
            
            # Skip if windows are invalid
            if (test_end > data_length or 
                train_end - train_start < self.config.min_training_samples or
                test_end - test_start < self.config.min_test_samples):
                continue
            
            # Create splits
            train_data = self.data.iloc[train_start:train_end].copy()
            test_data = self.data.iloc[test_start:test_end].copy()
            
            splits.append((train_data, test_data))
        
        return splits
    
    def _create_purged_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create purged cross-validation splits."""
        data_length = len(self.data)
        test_size = max(self.config.min_test_samples, int(data_length * self.config.test_size_ratio))
        purge_size = max(1, int(data_length * self.config.purge_pct))
        embargo_size = max(1, int(data_length * self.config.embargo_pct))
        
        splits = []
        
        for fold in range(self.config.n_splits):
            # Calculate test set position
            test_start = int(fold * (data_length - test_size) / (self.config.n_splits - 1)) if self.config.n_splits > 1 else 0
            test_end = test_start + test_size
            test_end = min(test_end, data_length)
            test_start = test_end - test_size
            test_start = max(0, test_start)
            
            # Create purged training set
            train_indices = []
            
            # Add data before test set (with purging)
            purge_start = max(0, test_start - purge_size)
            if purge_start > 0:
                train_indices.extend(range(0, purge_start))
            
            # Add data after test set (with embargo)
            embargo_end = min(data_length, test_end + embargo_size)
            if embargo_end < data_length:
                train_indices.extend(range(embargo_end, data_length))
            
            # Create splits
            if train_indices:
                train_data = self.data.iloc[train_indices].copy()
                test_data = self.data.iloc[test_start:test_end].copy()
                
                if (len(train_data) >= self.config.min_training_samples and 
                    len(test_data) >= self.config.min_test_samples):
                    splits.append((train_data, test_data))
        
        return splits
    
    def _create_combinatorial_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create combinatorial purged cross-validation splits."""
        # Start with basic purged splits
        base_splits = self._create_purged_splits()
        
        if not self.config.enable_combinatorial or len(base_splits) < 2:
            return base_splits
        
        combinatorial_splits = []
        max_combinations = self.config.max_combinations or min(10, len(base_splits) * (len(base_splits) - 1) // 2)
        
        # Create combinations of test sets
        from itertools import combinations
        
        test_combinations = list(combinations(range(len(base_splits)), 2))[:max_combinations]
        
        for combo in test_combinations:
            # Combine test sets from selected folds
            test_indices = set()
            for fold_idx in combo:
                _, test_data = base_splits[fold_idx]
                if isinstance(test_data.index, pd.DatetimeIndex):
                    test_indices.update(test_data.index)
                else:
                    test_indices.update(test_data.index.tolist())
            
            # Create training set excluding test indices and applying purging
            train_indices = []
            for idx in self.data.index:
                if idx not in test_indices:
                    train_indices.append(idx)
            
            if train_indices and test_indices:
                train_data = self.data.loc[train_indices].copy()
                test_data = self.data.loc[list(test_indices)].copy()
                
                # Sort by index
                train_data = train_data.sort_index()
                test_data = test_data.sort_index()
                
                if (len(train_data) >= self.config.min_training_samples and 
                    len(test_data) >= self.config.min_test_samples):
                    combinatorial_splits.append((train_data, test_data))
        
        return combinatorial_splits
    
    def _validate_splits_integrity(self, splits: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> None:
        """Validate splits for data leakage and integrity."""
        leakage_warnings = []
        
        for i, (train_data, test_data) in enumerate(splits):
            # Check for temporal leakage
            if isinstance(train_data.index, pd.DatetimeIndex) and isinstance(test_data.index, pd.DatetimeIndex):
                train_max = train_data.index.max()
                test_min = test_data.index.min()
                
                if train_max >= test_min:
                    leakage_warnings.append(f"Split {i}: Training data extends into test period")
            
            # Check for index overlap
            train_indices = set(train_data.index)
            test_indices = set(test_data.index)
            overlap = train_indices.intersection(test_indices)
            
            if overlap:
                leakage_warnings.append(f"Split {i}: {len(overlap)} overlapping indices between train and test")
        
        if leakage_warnings:
            for warning in leakage_warnings:
                logger.warning(warning)
            
            if len(leakage_warnings) > len(splits) * self.config.overlap_tolerance:
                raise ValueError("Too many splits have data leakage issues")
    
    def validate(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
    ) -> ValidationResult:
        """
        Validate parameters using enhanced time series cross-validation.
        
        Args:
            parameters: Parameter values to validate
            backtest_function: Function that runs backtest and returns metrics
            
        Returns:
            Enhanced validation result with comprehensive metrics
        """
        with self._get_lock():
            splits = self.split_data()
            fold_results = []
            
            for i, (train_data, test_data) in enumerate(splits):
                try:
                    # Run backtest on test data
                    metrics = backtest_function(test_data, parameters)
                    
                    # Add fold information and metadata
                    metrics_with_fold = metrics.copy()
                    metrics_with_fold['fold'] = i
                    metrics_with_fold['train_samples'] = len(train_data)
                    metrics_with_fold['test_samples'] = len(test_data)
                    metrics_with_fold['strategy'] = self.config.strategy.value
                    
                    # Add temporal information
                    if isinstance(train_data.index, pd.DatetimeIndex):
                        metrics_with_fold['train_start'] = train_data.index.min()
                        metrics_with_fold['train_end'] = train_data.index.max()
                        metrics_with_fold['test_start'] = test_data.index.min()
                        metrics_with_fold['test_end'] = test_data.index.max()
                    
                    # Calculate coverage metrics
                    total_samples = len(self.data)
                    metrics_with_fold['train_coverage'] = len(train_data) / total_samples
                    metrics_with_fold['test_coverage'] = len(test_data) / total_samples
                    
                    fold_results.append(metrics_with_fold)
                    
                except Exception as e:
                    logger.warning(f"Time series split fold {i} failed: {e}")
                    continue
            
            if not fold_results:
                raise ValueError("All time series split folds failed")
            
            # Calculate summary statistics
            mean_metrics, std_metrics, confidence_intervals = self._calculate_summary_statistics(fold_results)
            
            # Calculate enhanced validation metrics
            validation_metrics = self._calculate_validation_metrics(fold_results)
            
            # Create comprehensive data split info
            data_split_info = {
                "strategy": self.config.strategy.value,
                "n_splits": self.config.n_splits,
                "test_size_ratio": self.config.test_size_ratio,
                "purge_pct": self.config.purge_pct,
                "embargo_pct": self.config.embargo_pct,
                "total_samples": len(self.data),
                "successful_splits": len(fold_results),
                "enable_combinatorial": self.config.enable_combinatorial,
                "date_range": {
                    "start": self.data.index.min() if isinstance(self.data.index, pd.DatetimeIndex)
                            else self.data[self.date_column].min(),
                    "end": self.data.index.max() if isinstance(self.data.index, pd.DatetimeIndex)
                          else self.data[self.date_column].max(),
                },
                "validation_metrics": validation_metrics,
            }
            
            return ValidationResult(
                method=ValidationMethod.TIME_SERIES_SPLIT,
                fold_results=fold_results,
                mean_metrics=mean_metrics,
                std_metrics=std_metrics,
                confidence_intervals=confidence_intervals,
                n_splits=len(fold_results),
                data_split_info=data_split_info,
            )
    
    def _calculate_validation_metrics(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive validation metrics."""
        if not fold_results:
            return {}
        
        # Extract numeric metrics
        numeric_metrics = {}
        for result in fold_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and not key.startswith(('fold', 'train_', 'test_')):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        validation_metrics = {}
        
        # Calculate consistency and stability metrics
        for metric_name, values in numeric_metrics.items():
            if len(values) < 2:
                continue
            
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array, ddof=1)
            
            # Coefficient of variation
            cv = abs(std_val / mean_val) if mean_val != 0 else float('inf')
            
            # Consistency ratio (values within 1 std)
            within_1std = np.sum(np.abs(values_array - mean_val) <= std_val) / len(values)
            
            # Performance trend
            x = np.arange(len(values))
            if len(values) > 2:
                slope = np.polyfit(x, values_array, 1)[0]
            else:
                slope = 0.0
            
            validation_metrics[metric_name] = {
                "coefficient_of_variation": cv,
                "consistency_ratio": within_1std,
                "trend_slope": slope,
                "min_value": float(np.min(values_array)),
                "max_value": float(np.max(values_array)),
                "median_value": float(np.median(values_array)),
            }
        
        # Overall validation assessment
        main_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        stability_scores = []
        for metric in main_metrics:
            if metric in validation_metrics:
                cv = validation_metrics[metric]['coefficient_of_variation']
                if not np.isinf(cv):
                    stability_scores.append(cv)
        
        overall_stability = np.mean(stability_scores) if stability_scores else float('inf')
        
        validation_metrics['overall'] = {
            "stability_score": overall_stability,
            "is_stable": overall_stability < 0.25,
            "num_metrics_analyzed": len(stability_scores),
            "overfitting_risk": "high" if overall_stability > 0.5 else "medium" if overall_stability > 0.25 else "low",
        }
        
        return validation_metrics
    
    def detect_overfitting(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """
        Detect potential overfitting in validation results.
        
        Args:
            validation_result: Result from validate() method
            
        Returns:
            Dictionary with overfitting analysis
        """
        validation_metrics = validation_result.data_split_info.get('validation_metrics', {})
        overall_metrics = validation_metrics.get('overall', {})
        
        overfitting_indicators = []
        risk_score = 0
        
        # Check stability score
        stability_score = overall_metrics.get('stability_score', float('inf'))
        if stability_score > 0.5:
            overfitting_indicators.append("High performance variability across folds")
            risk_score += 3
        elif stability_score > 0.25:
            overfitting_indicators.append("Moderate performance variability")
            risk_score += 1
        
        # Check for declining performance trends
        declining_metrics = []
        for metric_name, metric_info in validation_metrics.items():
            if metric_name != 'overall' and isinstance(metric_info, dict):
                slope = metric_info.get('trend_slope', 0)
                if slope < -0.01:  # Significant decline
                    declining_metrics.append(metric_name)
                    risk_score += 2
        
        if declining_metrics:
            overfitting_indicators.append(f"Declining performance trends in: {', '.join(declining_metrics)}")
        
        # Check number of successful folds
        if validation_result.n_splits < 3:
            overfitting_indicators.append("Insufficient number of validation folds")
            risk_score += 2
        
        # Determine overall risk level
        if risk_score >= 5:
            risk_level = "high"
        elif risk_score >= 3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "overfitting_risk": risk_level,
            "risk_score": risk_score,
            "indicators": overfitting_indicators,
            "recommendations": self._get_overfitting_recommendations(risk_level, overfitting_indicators),
            "stability_score": stability_score,
            "successful_folds": validation_result.n_splits,
        }
    
    def _get_overfitting_recommendations(self, risk_level: str, indicators: List[str]) -> List[str]:
        """Generate recommendations based on overfitting risk."""
        recommendations = []
        
        if risk_level == "high":
            recommendations.extend([
                "Consider simplifying the model or reducing parameter complexity",
                "Increase training data size or use longer time periods",
                "Implement regularization techniques",
                "Use more conservative parameter bounds",
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Monitor performance on out-of-sample data carefully",
                "Consider using ensemble methods for stability",
                "Validate on different market regimes",
            ])
        
        if any("variability" in indicator for indicator in indicators):
            recommendations.append("Use stability-weighted parameter selection")
        
        if any("declining" in indicator for indicator in indicators):
            recommendations.append("Investigate why performance degrades over time")
        
        if any("Insufficient" in indicator for indicator in indicators):
            recommendations.append("Increase the number of validation folds")
        
        return recommendations
    
    def _get_lock(self):
        """Get thread lock if thread safety is enabled."""
        if self.thread_safe and self._lock is not None:
            return self._lock
        else:
            # Return a dummy context manager if thread safety is disabled
            class DummyLock:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return DummyLock()