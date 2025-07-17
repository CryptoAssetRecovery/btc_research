"""
Time series split validation for trading strategy optimization.

Time series split validation creates multiple chronological splits where
each subsequent split includes more historical data for training. This
provides a more robust assessment of how a strategy performs as more
data becomes available.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from btc_research.optimization.base import BaseValidator
from btc_research.optimization.types import ValidationMethod, ValidationResult

__all__ = ["TimeSeriesSplitValidator"]


class TimeSeriesSplitValidator(BaseValidator):
    """
    Time series split validation implementation.
    
    This validator creates multiple training/validation splits where:
    1. The first split uses the earliest portion of data for training
    2. Each subsequent split adds more historical data to training
    3. Validation is always performed on the next chronological portion
    
    This approach helps assess model stability as more data becomes available.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        test_size_ratio: float = 0.2,
        gap_size: int = 0,
        min_training_samples: int = 100,
        date_column: str = "timestamp",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize time series split validator.
        
        Args:
            data: Time series data for validation
            n_splits: Number of splits to create
            test_size_ratio: Proportion of data to use for testing in each split
            gap_size: Number of samples to skip between train and test (to avoid look-ahead)
            min_training_samples: Minimum number of samples required in training set
            date_column: Name of the datetime column
            random_seed: Random seed for reproducibility
        """
        super().__init__(data, date_column, random_seed)
        
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.gap_size = gap_size
        self.min_training_samples = min_training_samples
        
        # Validate parameters
        if n_splits < 2:
            raise ValueError("Number of splits must be at least 2")
        if not 0 < test_size_ratio < 1:
            raise ValueError("Test size ratio must be between 0 and 1")
        if gap_size < 0:
            raise ValueError("Gap size cannot be negative")
        
        # Ensure data is sorted by date
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data = self.data.sort_index()
        else:
            self.data = self.data.sort_values(date_column)
        
        self._splits_cache: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None
    
    def split_data(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create time series training/validation splits.
        
        Returns:
            List of (train_data, validation_data) tuples
        """
        if self._splits_cache is not None:
            return self._splits_cache
        
        data_length = len(self.data)
        test_size = max(1, int(data_length * self.test_size_ratio))
        
        splits = []
        
        # Calculate split points
        for split_idx in range(self.n_splits):
            # Calculate test end position for this split
            # Later splits have test sets further into the data
            test_end = data_length - (self.n_splits - split_idx - 1) * (test_size // self.n_splits)
            test_start = test_end - test_size
            
            # Ensure we don't go below 0
            test_start = max(0, test_start)
            
            # Calculate training end (with gap)
            train_end = test_start - self.gap_size
            
            # Skip if training set is too small
            if train_end < self.min_training_samples:
                continue
            
            # Create training and test sets
            train_data = self.data.iloc[:train_end].copy()
            test_data = self.data.iloc[test_start:test_end].copy()
            
            # Skip if either set is empty
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            splits.append((train_data, test_data))
        
        if not splits:
            raise ValueError(
                "No valid splits generated. Check data size and split parameters."
            )
        
        self._splits_cache = splits
        return splits
    
    def validate(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
    ) -> ValidationResult:
        """
        Validate parameters using time series splits.
        
        Args:
            parameters: Parameter values to validate
            backtest_function: Function that runs backtest and returns metrics
            
        Returns:
            Validation result with performance across all splits
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
                
                # Calculate training period coverage
                train_start_idx = 0
                train_end_idx = len(train_data) - 1
                total_data_length = len(self.data)
                
                metrics_with_fold['train_coverage'] = len(train_data) / total_data_length
                
                fold_results.append(metrics_with_fold)
                
            except Exception as e:
                # Handle failed backtests
                print(f"Warning: Time series split fold {i} failed: {e}")
                continue
        
        if not fold_results:
            raise ValueError("All time series splits failed")
        
        # Calculate summary statistics
        mean_metrics, std_metrics, confidence_intervals = self._calculate_summary_statistics(fold_results)
        
        # Create data split info
        data_split_info = {
            "n_splits": self.n_splits,
            "test_size_ratio": self.test_size_ratio,
            "gap_size": self.gap_size,
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
            method=ValidationMethod.TIME_SERIES_SPLIT,
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            confidence_intervals=confidence_intervals,
            n_splits=len(fold_results),
            data_split_info=data_split_info,
        )
    
    def get_split_info(self) -> Dict[str, Any]:
        """
        Get information about the time series splits.
        
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
            
            split_info.append({
                "fold": i,
                "train_start": train_start,
                "train_end": train_end,
                "train_samples": len(train_data),
                "test_start": test_start,
                "test_end": test_end,
                "test_samples": len(test_data),
                "train_coverage": len(train_data) / len(self.data),
            })
        
        return {
            "total_splits": len(splits),
            "n_splits": self.n_splits,
            "test_size_ratio": self.test_size_ratio,
            "gap_size": self.gap_size,
            "splits": split_info,
        }