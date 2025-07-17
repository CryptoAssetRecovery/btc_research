"""
Purged cross-validation for trading strategy optimization.

Purged cross-validation addresses the data leakage issue in traditional
cross-validation when applied to time series data. It ensures that
information from the test set cannot leak into the training set by
introducing purging gaps.
"""

from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from btc_research.optimization.base import BaseValidator
from btc_research.optimization.types import ValidationMethod, ValidationResult

__all__ = ["PurgedCrossValidator"]


class PurgedCrossValidator(BaseValidator):
    """
    Purged cross-validation implementation.
    
    This validator implements purged cross-validation as described in
    "Advances in Financial Machine Learning" by Marcos López de Prado.
    It addresses the data leakage problem in time series by:
    
    1. Creating non-overlapping folds
    2. Purging training data that comes after test data
    3. Adding embargo periods to prevent look-ahead bias
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.01,
        test_size_pct: float = 0.2,
        date_column: str = "timestamp",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize purged cross-validator.
        
        Args:
            data: Time series data for validation
            n_splits: Number of cross-validation splits
            purge_pct: Percentage of data to purge around test sets
            embargo_pct: Percentage of data to embargo after test sets
            test_size_pct: Percentage of data to use for each test set
            date_column: Name of the datetime column
            random_seed: Random seed for reproducibility
        """
        super().__init__(data, date_column, random_seed)
        
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.test_size_pct = test_size_pct
        
        # Validate parameters
        if n_splits < 2:
            raise ValueError("Number of splits must be at least 2")
        if not 0 <= purge_pct < 1:
            raise ValueError("Purge percentage must be between 0 and 1")
        if not 0 <= embargo_pct < 1:
            raise ValueError("Embargo percentage must be between 0 and 1")
        if not 0 < test_size_pct < 1:
            raise ValueError("Test size percentage must be between 0 and 1")
        
        # Ensure data is sorted by date
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data = self.data.sort_index()
        else:
            self.data = self.data.sort_values(date_column)
        
        self._splits_cache: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None
    
    def split_data(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create purged cross-validation training/validation splits.
        
        Returns:
            List of (train_data, validation_data) tuples
        """
        if self._splits_cache is not None:
            return self._splits_cache
        
        data_length = len(self.data)
        test_size = max(1, int(data_length * self.test_size_pct))
        purge_size = max(1, int(data_length * self.purge_pct))
        embargo_size = max(1, int(data_length * self.embargo_pct))
        
        splits = []
        
        # Calculate test set positions for each fold
        for fold in range(self.n_splits):
            # Evenly distribute test sets across the data
            test_start = int(fold * (data_length - test_size) / (self.n_splits - 1)) if self.n_splits > 1 else 0
            test_end = test_start + test_size
            
            # Ensure we don't exceed data bounds
            test_end = min(test_end, data_length)
            test_start = test_end - test_size
            test_start = max(0, test_start)
            
            # Skip if test set is invalid
            if test_start >= test_end:
                continue
            
            # Create test set
            test_data = self.data.iloc[test_start:test_end].copy()
            
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
            
            # Create training set
            if train_indices:
                train_data = self.data.iloc[train_indices].copy()
            else:
                # Skip this fold if no training data available
                continue
            
            # Skip if either set is empty or training set is too small
            if len(train_data) < 50 or len(test_data) == 0:
                continue
            
            splits.append((train_data, test_data))
        
        if not splits:
            raise ValueError(
                "No valid splits generated. Check data size and purging parameters."
            )
        
        self._splits_cache = splits
        return splits
    
    def validate(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
    ) -> ValidationResult:
        """
        Validate parameters using purged cross-validation.
        
        Args:
            parameters: Parameter values to validate
            backtest_function: Function that runs backtest and returns metrics
            
        Returns:
            Validation result with performance across all folds
        """
        splits = self.split_data()
        fold_results = []
        
        for i, (train_data, test_data) in enumerate(splits):
            try:
                # Run backtest on test data using parameters
                metrics = backtest_function(test_data, parameters)
                
                # Add fold information to results
                metrics_with_fold = metrics.copy()
                metrics_with_fold['fold'] = i
                metrics_with_fold['train_samples'] = len(train_data)
                metrics_with_fold['test_samples'] = len(test_data)
                
                # Calculate data usage statistics
                total_samples = len(self.data)
                metrics_with_fold['train_coverage'] = len(train_data) / total_samples
                metrics_with_fold['test_coverage'] = len(test_data) / total_samples
                
                fold_results.append(metrics_with_fold)
                
            except Exception as e:
                # Handle failed backtests
                print(f"Warning: Purged CV fold {i} failed: {e}")
                continue
        
        if not fold_results:
            raise ValueError("All purged cross-validation folds failed")
        
        # Calculate summary statistics
        mean_metrics, std_metrics, confidence_intervals = self._calculate_summary_statistics(fold_results)
        
        # Create data split info
        data_split_info = {
            "n_splits": self.n_splits,
            "purge_pct": self.purge_pct,
            "embargo_pct": self.embargo_pct,
            "test_size_pct": self.test_size_pct,
            "total_samples": len(self.data),
            "successful_splits": len(fold_results),
            "date_range": {
                "start": self.data.index.min() if isinstance(self.data.index, pd.DatetimeIndex)
                        else self.data[self.date_column].min(),
                "end": self.data.index.max() if isinstance(self.data.index, pd.DatetimeIndex)
                      else self.data[self.date_column].max(),
            }
        }
        
        return ValidationResult(
            method=ValidationMethod.PURGED_CROSS_VALIDATION,
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            confidence_intervals=confidence_intervals,
            n_splits=len(fold_results),
            data_split_info=data_split_info,
        )
    
    def get_split_info(self) -> Dict[str, Any]:
        """
        Get information about the purged cross-validation splits.
        
        Returns:
            Dictionary with split information
        """
        splits = self.split_data()
        
        split_info = []
        for i, (train_data, test_data) in enumerate(splits):
            # Get date ranges
            if isinstance(train_data.index, pd.DatetimeIndex):
                train_start = train_data.index.min()
                train_end = train_data.index.max()
                test_start = test_data.index.min()
                test_end = test_data.index.max()
            else:
                train_start = train_data[self.date_column].min()
                train_end = train_data[self.date_column].max()
                test_start = test_data[self.date_column].min()
                test_end = test_data[self.date_column].max()
            
            # Check for data leakage (training data after test data)
            has_leakage = False
            if isinstance(self.data.index, pd.DatetimeIndex):
                has_leakage = train_end > test_start
            else:
                has_leakage = train_end > test_start
            
            split_info.append({
                "fold": i,
                "train_start": train_start,
                "train_end": train_end,
                "train_samples": len(train_data),
                "test_start": test_start,
                "test_end": test_end,
                "test_samples": len(test_data),
                "train_coverage": len(train_data) / len(self.data),
                "test_coverage": len(test_data) / len(self.data),
                "has_potential_leakage": has_leakage,
            })
        
        return {
            "total_splits": len(splits),
            "n_splits": self.n_splits,
            "purge_pct": self.purge_pct,
            "embargo_pct": self.embargo_pct,
            "test_size_pct": self.test_size_pct,
            "splits": split_info,
        }
    
    def calculate_purge_embargo_sizes(self) -> Dict[str, int]:
        """
        Calculate actual purge and embargo sizes in samples.
        
        Returns:
            Dictionary with calculated sizes
        """
        data_length = len(self.data)
        
        return {
            "data_length": data_length,
            "purge_size_samples": max(1, int(data_length * self.purge_pct)),
            "embargo_size_samples": max(1, int(data_length * self.embargo_pct)),
            "test_size_samples": max(1, int(data_length * self.test_size_pct)),
            "purge_size_pct": self.purge_pct,
            "embargo_size_pct": self.embargo_pct,
            "test_size_pct": self.test_size_pct,
        }