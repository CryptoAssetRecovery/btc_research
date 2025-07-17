"""
Walk-forward validation for trading strategy optimization.

Walk-forward analysis tests strategy performance by moving training windows
forward through time and validating on subsequent data. This enhanced version
supports both rolling and expanding window approaches, performance stability
metrics, and comprehensive validation to prevent overfitting.
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

__all__ = ["WalkForwardValidator", "WindowType", "WalkForwardConfig"]


class WindowType(Enum):
    """Types of walk-forward windows."""
    ROLLING = "rolling"  # Fixed-size window that slides forward
    EXPANDING = "expanding"  # Growing window that includes all past data


class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    
    def __init__(
        self,
        window_type: WindowType = WindowType.ROLLING,
        training_window_days: int = 90,
        validation_window_days: int = 30,
        step_size_days: int = 30,
        gap_days: int = 0,
        min_training_samples: int = 100,
        min_validation_samples: int = 20,
        max_windows: Optional[int] = None,
        reoptimize_each_window: bool = False,
    ):
        """
        Initialize walk-forward configuration.
        
        Args:
            window_type: Type of window (rolling or expanding)
            training_window_days: Days for training window (rolling only)
            validation_window_days: Days for validation window
            step_size_days: Days to step forward between windows
            gap_days: Gap days between training and validation
            min_training_samples: Minimum samples in training set
            min_validation_samples: Minimum samples in validation set
            max_windows: Maximum number of windows to create
            reoptimize_each_window: Whether to reoptimize in each window
        """
        self.window_type = window_type
        self.training_window_days = training_window_days
        self.validation_window_days = validation_window_days
        self.step_size_days = step_size_days
        self.gap_days = gap_days
        self.min_training_samples = min_training_samples
        self.min_validation_samples = min_validation_samples
        self.max_windows = max_windows
        self.reoptimize_each_window = reoptimize_each_window
        
        # Validation
        if training_window_days <= 0:
            raise ValueError("Training window days must be positive")
        if validation_window_days <= 0:
            raise ValueError("Validation window days must be positive")
        if step_size_days <= 0:
            raise ValueError("Step size days must be positive")
        if gap_days < 0:
            raise ValueError("Gap days cannot be negative")


class WalkForwardValidator(BaseValidator):
    """
    Enhanced walk-forward validation implementation.
    
    This validator creates multiple training/validation splits by moving
    training windows forward through time. Supports both rolling (fixed-size)
    and expanding (growing) windows, with comprehensive stability metrics
    and performance analysis.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[WalkForwardConfig] = None,
        date_column: str = "timestamp",
        random_seed: Optional[int] = None,
        thread_safe: bool = True,
    ):
        """
        Initialize enhanced walk-forward validator.
        
        Args:
            data: Time series data for validation
            config: Walk-forward configuration (uses defaults if None)
            date_column: Name of the datetime column
            random_seed: Random seed for reproducibility
            thread_safe: Whether to enable thread-safe operations
        """
        super().__init__(data, date_column, random_seed)
        
        self.config = config if config is not None else WalkForwardConfig()
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
        self._stability_metrics_cache: Optional[Dict[str, Any]] = None
    
    def split_data(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward training/validation splits with rolling or expanding windows.
        
        Returns:
            List of (train_data, validation_data) tuples
        """
        with self._get_lock():
            if self._splits_cache is not None:
                return self._splits_cache
            
            splits = []
            
            # Get datetime information
            if isinstance(self.data.index, pd.DatetimeIndex):
                dates = self.data.index
            else:
                dates = pd.to_datetime(self.data[self.date_column])
            
            start_date = dates.min()
            end_date = dates.max()
            
            # Calculate total available days
            total_days = (end_date - start_date).days
            min_required_days = self.config.training_window_days + self.config.validation_window_days + self.config.gap_days
            
            if total_days < min_required_days:
                raise ValueError(
                    f"Insufficient data: {total_days} days available, "
                    f"but need at least {min_required_days} days"
                )
            
            # Create walk-forward splits
            current_start = start_date
            window_count = 0
            
            while True:
                # Check max windows limit
                if self.config.max_windows and window_count >= self.config.max_windows:
                    break
                
                # Define training window based on window type
                if self.config.window_type == WindowType.ROLLING:
                    train_start = current_start
                    train_end = train_start + timedelta(days=self.config.training_window_days)
                else:  # EXPANDING
                    train_start = start_date
                    train_end = current_start + timedelta(days=self.config.training_window_days)
                
                # Add gap between training and validation
                val_start = train_end + timedelta(days=self.config.gap_days)
                val_end = val_start + timedelta(days=self.config.validation_window_days)
                
                # Check if we have enough data for this split
                if val_end > end_date:
                    break
                
                # Extract training and validation data
                if isinstance(self.data.index, pd.DatetimeIndex):
                    train_data = self.data[
                        (self.data.index >= train_start) & (self.data.index < train_end)
                    ]
                    val_data = self.data[
                        (self.data.index >= val_start) & (self.data.index < val_end)
                    ]
                else:
                    train_mask = (dates >= train_start) & (dates < train_end)
                    val_mask = (dates >= val_start) & (dates < val_end)
                    train_data = self.data[train_mask]
                    val_data = self.data[val_mask]
                
                # Check minimum sample requirements
                if (len(train_data) >= self.config.min_training_samples and 
                    len(val_data) >= self.config.min_validation_samples):
                    
                    splits.append((train_data.copy(), val_data.copy()))
                    window_count += 1
                    
                    logger.debug(
                        f"Created walk-forward split {window_count}: "
                        f"train={len(train_data)} samples, val={len(val_data)} samples"
                    )
                
                # Move to next position
                current_start += timedelta(days=self.config.step_size_days)
            
            if not splits:
                raise ValueError(
                    "No valid splits generated. Check data range and window parameters."
                )
            
            logger.info(f"Created {len(splits)} walk-forward splits ({self.config.window_type.value} windows)")
            self._splits_cache = splits
            return splits
    
    def validate(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        optimization_function: Optional[Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> ValidationResult:
        """
        Validate parameters using enhanced walk-forward analysis.
        
        Args:
            parameters: Parameter values to validate
            backtest_function: Function that runs backtest and returns metrics
            optimization_function: Optional function to re-optimize parameters for each window
            
        Returns:
            Enhanced validation result with stability metrics
        """
        with self._get_lock():
            splits = self.split_data()
            fold_results = []
            
            for i, (train_data, val_data) in enumerate(splits):
                try:
                    # Determine which parameters to use
                    fold_parameters = parameters
                    
                    # Re-optimize parameters for this window if requested
                    if self.config.reoptimize_each_window and optimization_function:
                        try:
                            optimized_params = optimization_function(train_data, parameters)
                            fold_parameters = optimized_params
                            logger.debug(f"Re-optimized parameters for fold {i}")
                        except Exception as e:
                            logger.warning(f"Re-optimization failed for fold {i}: {e}, using original parameters")
                    
                    # Run backtest on validation data
                    metrics = backtest_function(val_data, fold_parameters)
                    
                    # Add fold information and metadata to results
                    metrics_with_fold = metrics.copy()
                    metrics_with_fold['fold'] = i
                    metrics_with_fold['train_samples'] = len(train_data)
                    metrics_with_fold['val_samples'] = len(val_data)
                    metrics_with_fold['window_type'] = self.config.window_type.value
                    metrics_with_fold['reoptimized'] = self.config.reoptimize_each_window
                    
                    # Add timing information
                    if isinstance(train_data.index, pd.DatetimeIndex):
                        metrics_with_fold['train_start'] = train_data.index.min()
                        metrics_with_fold['train_end'] = train_data.index.max()
                        metrics_with_fold['val_start'] = val_data.index.min()
                        metrics_with_fold['val_end'] = val_data.index.max()
                    
                    fold_results.append(metrics_with_fold)
                    
                except Exception as e:
                    logger.warning(f"Walk-forward fold {i} failed: {e}")
                    continue
            
            if not fold_results:
                raise ValueError("All walk-forward folds failed")
            
            # Calculate summary statistics
            mean_metrics, std_metrics, confidence_intervals = self._calculate_summary_statistics(fold_results)
            
            # Calculate stability and performance metrics
            stability_metrics = self._calculate_stability_metrics(fold_results)
            
            # Create comprehensive data split info
            data_split_info = {
                "window_type": self.config.window_type.value,
                "training_window_days": self.config.training_window_days,
                "validation_window_days": self.config.validation_window_days,
                "step_size_days": self.config.step_size_days,
                "gap_days": self.config.gap_days,
                "total_splits": len(splits),
                "successful_splits": len(fold_results),
                "reoptimize_each_window": self.config.reoptimize_each_window,
                "date_range": {
                    "start": self.data.index.min() if isinstance(self.data.index, pd.DatetimeIndex) 
                            else self.data[self.date_column].min(),
                    "end": self.data.index.max() if isinstance(self.data.index, pd.DatetimeIndex)
                          else self.data[self.date_column].max(),
                },
                "stability_metrics": stability_metrics,
            }
            
            return ValidationResult(
                method=ValidationMethod.WALK_FORWARD,
                fold_results=fold_results,
                mean_metrics=mean_metrics,
                std_metrics=std_metrics,
                confidence_intervals=confidence_intervals,
                n_splits=len(fold_results),
                data_split_info=data_split_info,
            )
    
    def get_split_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the walk-forward splits.
        
        Returns:
            Dictionary with enhanced split information
        """
        with self._get_lock():
            splits = self.split_data()
            
            split_info = []
            for i, (train_data, val_data) in enumerate(splits):
                # Get date ranges
                if isinstance(train_data.index, pd.DatetimeIndex):
                    train_start = train_data.index.min()
                    train_end = train_data.index.max()
                    val_start = val_data.index.min()
                    val_end = val_data.index.max()
                else:
                    train_start = train_data[self.date_column].min()
                    train_end = train_data[self.date_column].max()
                    val_start = val_data[self.date_column].min()
                    val_end = val_data[self.date_column].max()
                
                # Calculate coverage ratios
                total_samples = len(self.data)
                train_coverage = len(train_data) / total_samples if total_samples > 0 else 0
                val_coverage = len(val_data) / total_samples if total_samples > 0 else 0
                
                split_info.append({
                    "fold": i,
                    "train_start": train_start,
                    "train_end": train_end,
                    "train_samples": len(train_data),
                    "train_coverage": train_coverage,
                    "val_start": val_start,
                    "val_end": val_end,
                    "val_samples": len(val_data),
                    "val_coverage": val_coverage,
                    "gap_days": self.config.gap_days,
                })
            
            return {
                "total_splits": len(splits),
                "window_type": self.config.window_type.value,
                "training_window_days": self.config.training_window_days,
                "validation_window_days": self.config.validation_window_days,
                "step_size_days": self.config.step_size_days,
                "gap_days": self.config.gap_days,
                "total_data_samples": len(self.data),
                "data_date_range": {
                    "start": self.data.index.min() if isinstance(self.data.index, pd.DatetimeIndex)
                            else self.data[self.date_column].min(),
                    "end": self.data.index.max() if isinstance(self.data.index, pd.DatetimeIndex)
                          else self.data[self.date_column].max(),
                },
                "splits": split_info,
            }
    
    def _calculate_stability_metrics(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance stability metrics across walk-forward windows.
        
        Args:
            fold_results: Results from each fold
            
        Returns:
            Dictionary with stability metrics
        """
        if not fold_results:
            return {}
        
        # Extract all numeric metrics
        numeric_metrics = {}
        for result in fold_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and not key.startswith(('fold', 'train_', 'val_')):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        stability_metrics = {}
        
        for metric_name, values in numeric_metrics.items():
            if len(values) < 2:
                continue
            
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array, ddof=1)
            
            # Coefficient of variation (stability score)
            cv = abs(std_val / mean_val) if mean_val != 0 else float('inf')
            
            # Performance trend (linear regression slope)
            x = np.arange(len(values))
            if len(values) > 2:
                slope = np.polyfit(x, values_array, 1)[0]
                trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            else:
                slope = 0.0
                trend_direction = "stable"
            
            # Consistency (percentage of windows within 1 std of mean)
            within_1std = np.sum(np.abs(values_array - mean_val) <= std_val) / len(values)
            
            stability_metrics[metric_name] = {
                "coefficient_of_variation": cv,
                "trend_slope": slope,
                "trend_direction": trend_direction,
                "consistency_ratio": within_1std,
                "min_value": float(np.min(values_array)),
                "max_value": float(np.max(values_array)),
                "range": float(np.max(values_array) - np.min(values_array)),
            }
        
        # Overall stability score (average CV across main metrics)
        main_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        stability_scores = []
        for metric in main_metrics:
            if metric in stability_metrics:
                cv = stability_metrics[metric]['coefficient_of_variation']
                if not np.isinf(cv):
                    stability_scores.append(cv)
        
        overall_stability = np.mean(stability_scores) if stability_scores else float('inf')
        
        stability_metrics['overall'] = {
            "stability_score": overall_stability,
            "num_metrics_analyzed": len(stability_scores),
            "is_stable": overall_stability < 0.3,  # Threshold for stability
        }
        
        return stability_metrics
    
    def calculate_performance_degradation(
        self, 
        fold_results: List[Dict[str, Any]], 
        metric: str = 'total_return'
    ) -> Dict[str, Any]:
        """
        Detect performance degradation patterns across windows.
        
        Args:
            fold_results: Results from each fold
            metric: Metric to analyze for degradation
            
        Returns:
            Dictionary with degradation analysis
        """
        if not fold_results or metric not in fold_results[0]:
            return {"error": f"Metric '{metric}' not found in results"}
        
        values = [result[metric] for result in fold_results if metric in result]
        if len(values) < 3:
            return {"error": "Insufficient data for degradation analysis"}
        
        values_array = np.array(values)
        x = np.arange(len(values))
        
        # Linear trend analysis
        slope, intercept = np.polyfit(x, values_array, 1)
        
        # Detect significant drops
        drops = []
        for i in range(1, len(values)):
            pct_change = (values[i] - values[i-1]) / abs(values[i-1]) if values[i-1] != 0 else 0
            if pct_change < -0.1:  # 10% drop threshold
                drops.append({
                    "window": i,
                    "drop_percentage": pct_change,
                    "from_value": values[i-1],
                    "to_value": values[i],
                })
        
        # Rolling performance analysis
        window_size = min(3, len(values) // 2)
        rolling_means = []
        for i in range(window_size, len(values) + 1):
            rolling_mean = np.mean(values[i-window_size:i])
            rolling_means.append(rolling_mean)
        
        # Performance consistency
        if len(rolling_means) > 1:
            rolling_std = np.std(rolling_means)
            rolling_cv = rolling_std / np.mean(rolling_means) if np.mean(rolling_means) != 0 else float('inf')
        else:
            rolling_cv = 0.0
        
        return {
            "metric_analyzed": metric,
            "trend_slope": slope,
            "trend_direction": "declining" if slope < -0.01 else "improving" if slope > 0.01 else "stable",
            "significant_drops": drops,
            "num_significant_drops": len(drops),
            "rolling_consistency": rolling_cv,
            "early_performance": np.mean(values[:len(values)//3]) if len(values) >= 3 else values[0],
            "late_performance": np.mean(values[-len(values)//3:]) if len(values) >= 3 else values[-1],
            "performance_ratio": (np.mean(values[-len(values)//3:]) / np.mean(values[:len(values)//3])) 
                               if len(values) >= 3 and np.mean(values[:len(values)//3]) != 0 else 1.0,
        }
    
    def get_validation_summary(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """
        Generate a comprehensive validation summary with recommendations.
        
        Args:
            validation_result: Result from validate() method
            
        Returns:
            Dictionary with validation summary and recommendations
        """
        stability_metrics = validation_result.data_split_info.get('stability_metrics', {})
        overall_stability = stability_metrics.get('overall', {})
        
        # Performance assessment
        performance_assessment = {
            "stability_score": overall_stability.get('stability_score', float('inf')),
            "is_stable": overall_stability.get('is_stable', False),
            "num_successful_windows": validation_result.n_splits,
            "consistency_issues": [],
            "recommendations": [],
        }
        
        # Check for consistency issues
        if validation_result.stability_score > 0.5:
            performance_assessment["consistency_issues"].append("High performance variability across windows")
            performance_assessment["recommendations"].append("Consider using more stable parameters or longer training windows")
        
        if validation_result.n_splits < 5:
            performance_assessment["consistency_issues"].append("Insufficient number of validation windows")
            performance_assessment["recommendations"].append("Increase data period or reduce window sizes for more validation windows")
        
        # Check for degradation
        for metric_name, metric_info in stability_metrics.items():
            if metric_name != 'overall' and isinstance(metric_info, dict):
                if metric_info.get('trend_direction') == 'decreasing':
                    performance_assessment["consistency_issues"].append(f"Declining trend in {metric_name}")
                    performance_assessment["recommendations"].append(f"Investigate why {metric_name} is declining over time")
        
        # Overall recommendation
        if performance_assessment["is_stable"] and not performance_assessment["consistency_issues"]:
            performance_assessment["overall_recommendation"] = "Parameters appear robust for walk-forward validation"
        elif performance_assessment["is_stable"]:
            performance_assessment["overall_recommendation"] = "Parameters are stable but have some consistency issues"
        else:
            performance_assessment["overall_recommendation"] = "Parameters show instability - consider re-optimization"
        
        return {
            "validation_method": "walk_forward",
            "window_type": self.config.window_type.value,
            "performance_assessment": performance_assessment,
            "key_metrics": {
                "mean_return": validation_result.mean_metrics.get('total_return', 0),
                "return_std": validation_result.std_metrics.get('total_return', 0),
                "mean_sharpe": validation_result.mean_metrics.get('sharpe_ratio', 0),
                "sharpe_std": validation_result.std_metrics.get('sharpe_ratio', 0),
            },
            "validation_config": {
                "training_window_days": self.config.training_window_days,
                "validation_window_days": self.config.validation_window_days,
                "step_size_days": self.config.step_size_days,
                "gap_days": self.config.gap_days,
                "reoptimize_each_window": self.config.reoptimize_each_window,
            },
        }
    
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