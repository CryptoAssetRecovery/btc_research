"""
Time series data splitting utilities for validation frameworks.

This module provides utilities for splitting time series data into clean
training, validation, and test sets while preventing data leakage and
ensuring temporal ordering is preserved.
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

__all__ = ["TimeSeriesDataSplitter", "DataSplitResult", "SplitConfig"]

# Set up logging
logger = logging.getLogger(__name__)


class DataSplitResult:
    """
    Result container for time series data splits.
    
    Contains the split datasets along with metadata about the split operation.
    """
    
    def __init__(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        metadata: Dict[str, Any],
    ):
        """
        Initialize split result.
        
        Args:
            train: Training dataset
            validation: Validation dataset  
            test: Test dataset
            metadata: Split metadata and statistics
        """
        self.train = train
        self.validation = validation
        self.test = test
        self.metadata = metadata
    
    @property
    def total_samples(self) -> int:
        """Total number of samples across all splits."""
        return len(self.train) + len(self.validation) + len(self.test)
    
    @property
    def split_ratios(self) -> Dict[str, float]:
        """Calculate actual split ratios."""
        total = self.total_samples
        if total == 0:
            return {"train": 0.0, "validation": 0.0, "test": 0.0}
        
        return {
            "train": len(self.train) / total,
            "validation": len(self.validation) / total,
            "test": len(self.test) / total,
        }
    
    def has_data_leakage(self) -> bool:
        """
        Check for potential data leakage between splits.
        
        Returns:
            True if temporal ordering is violated
        """
        if any(len(df) == 0 for df in [self.train, self.validation, self.test]):
            return False
        
        # Get datetime indices or columns
        train_dates = self._get_dates(self.train)
        val_dates = self._get_dates(self.validation) 
        test_dates = self._get_dates(self.test)
        
        if any(dates is None for dates in [train_dates, val_dates, test_dates]):
            logger.warning("Cannot check data leakage - no datetime information found")
            return False
        
        # Check temporal ordering
        train_max = train_dates.max()
        val_min = val_dates.min()
        val_max = val_dates.max()
        test_min = test_dates.min()
        
        # Training should come before validation, validation before test
        return train_max >= val_min or val_max >= test_min
    
    def _get_dates(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract datetime information from DataFrame."""
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.to_series()
        
        # Look for common datetime column names
        date_columns = ['timestamp', 'date', 'datetime', 'time']
        for col in date_columns:
            if col in df.columns:
                return pd.to_datetime(df[col])
        
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the split."""
        summary = {
            "split_sizes": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
                "total": self.total_samples,
            },
            "split_ratios": self.split_ratios,
            "has_data_leakage": self.has_data_leakage(),
            "metadata": self.metadata,
        }
        
        # Add date ranges if available
        for split_name, df in [("train", self.train), ("validation", self.validation), ("test", self.test)]:
            dates = self._get_dates(df)
            if dates is not None and len(dates) > 0:
                summary[f"{split_name}_date_range"] = {
                    "start": dates.min(),
                    "end": dates.max(),
                }
        
        return summary


class SplitConfig:
    """Configuration for time series data splitting."""
    
    def __init__(
        self,
        train_ratio: float = 0.6,
        validation_ratio: float = 0.2,
        test_ratio: float = 0.2,
        gap_days: int = 0,
        min_samples_per_split: int = 10,
        preserve_frequency: bool = True,
    ):
        """
        Initialize split configuration.
        
        Args:
            train_ratio: Proportion of data for training (0-1)
            validation_ratio: Proportion of data for validation (0-1)
            test_ratio: Proportion of data for testing (0-1)
            gap_days: Days to skip between splits to prevent lookahead bias
            min_samples_per_split: Minimum samples required in each split
            preserve_frequency: Whether to preserve DataFrame frequency information
        """
        # Validate ratios
        total_ratio = train_ratio + validation_ratio + test_ratio
        if not 0.95 <= total_ratio <= 1.05:  # Allow small floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        if any(ratio <= 0 for ratio in [train_ratio, validation_ratio, test_ratio]):
            raise ValueError("All split ratios must be positive")
        
        # Normalize ratios to ensure they sum to exactly 1.0
        self.train_ratio = train_ratio / total_ratio
        self.validation_ratio = validation_ratio / total_ratio
        self.test_ratio = test_ratio / total_ratio
        
        self.gap_days = max(0, gap_days)
        self.min_samples_per_split = max(1, min_samples_per_split)
        self.preserve_frequency = preserve_frequency


class TimeSeriesDataSplitter:
    """
    Comprehensive time series data splitter with gap handling and validation.
    
    This class provides robust methods for splitting time series data while
    ensuring no data leakage occurs and temporal ordering is preserved.
    Supports both percentage-based and date-based splitting with configurable
    gap periods.
    """
    
    def __init__(
        self,
        date_column: str = "timestamp",
        random_seed: Optional[int] = None,
        thread_safe: bool = True,
    ):
        """
        Initialize the time series data splitter.
        
        Args:
            date_column: Name of the datetime column (if not using DatetimeIndex)
            random_seed: Random seed for reproducibility
            thread_safe: Whether to enable thread-safe operations
        """
        self.date_column = date_column
        self.random_seed = random_seed
        self.thread_safe = thread_safe
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Thread safety
        self._lock = threading.RLock() if thread_safe else None
        
        # Caching for repeated operations
        self._cache: Dict[str, Any] = {}
    
    def split_by_ratio(
        self,
        data: pd.DataFrame,
        config: Optional[SplitConfig] = None,
    ) -> DataSplitResult:
        """
        Split data by ratio with proper temporal ordering.
        
        Args:
            data: Time series data to split
            config: Split configuration (uses defaults if None)
            
        Returns:
            DataSplitResult containing train, validation, and test sets
            
        Raises:
            ValueError: If data is insufficient or invalid
        """
        if config is None:
            config = SplitConfig()
        
        with self._get_lock():
            return self._split_by_ratio_impl(data, config)
    
    def split_by_date(
        self,
        data: pd.DataFrame,
        train_end: Union[str, datetime],
        validation_end: Union[str, datetime],
        gap_days: int = 0,
    ) -> DataSplitResult:
        """
        Split data by specific dates.
        
        Args:
            data: Time series data to split
            train_end: End date for training data
            validation_end: End date for validation data
            gap_days: Days to skip between splits
            
        Returns:
            DataSplitResult containing train, validation, and test sets
        """
        with self._get_lock():
            return self._split_by_date_impl(data, train_end, validation_end, gap_days)
    
    def create_walk_forward_splits(
        self,
        data: pd.DataFrame,
        window_size_days: int,
        step_size_days: int,
        validation_days: int,
        gap_days: int = 0,
        min_samples: int = 10,
    ) -> List[DataSplitResult]:
        """
        Create multiple walk-forward splits for validation.
        
        Args:
            data: Time series data to split
            window_size_days: Size of training window in days
            step_size_days: Step size between windows in days
            validation_days: Size of validation window in days
            gap_days: Gap between training and validation
            min_samples: Minimum samples per split
            
        Returns:
            List of DataSplitResult objects for each walk
        """
        with self._get_lock():
            return self._create_walk_forward_splits_impl(
                data, window_size_days, step_size_days, validation_days, gap_days, min_samples
            )
    
    def validate_split_quality(self, result: DataSplitResult) -> Dict[str, Any]:
        """
        Validate the quality of a data split.
        
        Args:
            result: Split result to validate
            
        Returns:
            Dictionary with validation metrics and warnings
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "metrics": {},
        }
        
        # Check for empty splits
        split_sizes = result.split_ratios
        for split_name, ratio in split_sizes.items():
            if ratio == 0:
                validation_results["warnings"].append(f"Empty {split_name} split")
                validation_results["is_valid"] = False
        
        # Check for data leakage
        if result.has_data_leakage():
            validation_results["warnings"].append("Potential data leakage detected")
            validation_results["is_valid"] = False
        
        # Check split balance
        expected_ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}
        ratio_deviations = {}
        for split_name, actual_ratio in split_sizes.items():
            expected = expected_ratios.get(split_name, 0.33)
            deviation = abs(actual_ratio - expected) / expected
            ratio_deviations[f"{split_name}_deviation"] = deviation
            
            if deviation > 0.5:  # More than 50% deviation
                validation_results["warnings"].append(
                    f"{split_name} split ratio deviates significantly from expected"
                )
        
        validation_results["metrics"].update(ratio_deviations)
        validation_results["metrics"]["total_samples"] = result.total_samples
        
        return validation_results
    
    def _split_by_ratio_impl(self, data: pd.DataFrame, config: SplitConfig) -> DataSplitResult:
        """Internal implementation of ratio-based splitting."""
        # Validate input data
        self._validate_data(data)
        
        # Sort data by date
        data_sorted = self._sort_by_date(data)
        data_length = len(data_sorted)
        
        # Check minimum data requirements
        min_total_samples = config.min_samples_per_split * 3
        if data_length < min_total_samples:
            raise ValueError(
                f"Insufficient data: {data_length} samples, need at least {min_total_samples}"
            )
        
        # Calculate split points
        train_size = int(data_length * config.train_ratio)
        val_size = int(data_length * config.validation_ratio)
        
        # Adjust for gap days
        gap_samples = self._calculate_gap_samples(data_sorted, config.gap_days)
        
        # Calculate actual split indices with gaps
        train_end = train_size
        val_start = train_end + gap_samples
        val_end = val_start + val_size
        test_start = val_end + gap_samples
        
        # Ensure we don't exceed data bounds
        if test_start >= data_length:
            # Adjust split sizes to accommodate gaps
            available_for_splits = data_length - 2 * gap_samples
            if available_for_splits < min_total_samples:
                raise ValueError("Insufficient data after accounting for gaps")
            
            train_size = int(available_for_splits * config.train_ratio)
            val_size = int(available_for_splits * config.validation_ratio)
            
            train_end = train_size
            val_start = train_end + gap_samples
            val_end = val_start + val_size
            test_start = val_end + gap_samples
        
        # Create splits
        train_data = data_sorted.iloc[:train_end].copy()
        validation_data = data_sorted.iloc[val_start:val_end].copy()
        test_data = data_sorted.iloc[test_start:].copy()
        
        # Validate split sizes
        for split_data, split_name in [(train_data, "train"), (validation_data, "validation"), (test_data, "test")]:
            if len(split_data) < config.min_samples_per_split:
                raise ValueError(f"{split_name} split has insufficient samples: {len(split_data)}")
        
        # Create metadata
        metadata = {
            "split_method": "ratio",
            "config": config.__dict__,
            "original_length": data_length,
            "gap_samples": gap_samples,
            "split_indices": {
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
            },
        }
        
        return DataSplitResult(train_data, validation_data, test_data, metadata)
    
    def _split_by_date_impl(
        self, 
        data: pd.DataFrame, 
        train_end: Union[str, datetime], 
        validation_end: Union[str, datetime],
        gap_days: int
    ) -> DataSplitResult:
        """Internal implementation of date-based splitting."""
        # Validate and parse dates
        train_end_dt = pd.to_datetime(train_end)
        validation_end_dt = pd.to_datetime(validation_end)
        
        if train_end_dt >= validation_end_dt:
            raise ValueError("train_end must be before validation_end")
        
        # Validate input data
        self._validate_data(data)
        
        # Sort data by date
        data_sorted = self._sort_by_date(data)
        
        # Get datetime series
        dates = self._get_datetime_series(data_sorted)
        
        # Add gap periods
        gap_delta = timedelta(days=gap_days)
        train_end_with_gap = train_end_dt - gap_delta
        val_start_with_gap = train_end_dt + gap_delta
        val_end_with_gap = validation_end_dt - gap_delta
        test_start_with_gap = validation_end_dt + gap_delta
        
        # Create masks for each split
        train_mask = dates <= train_end_with_gap
        val_mask = (dates >= val_start_with_gap) & (dates <= val_end_with_gap)
        test_mask = dates >= test_start_with_gap
        
        # Create splits
        train_data = data_sorted[train_mask].copy()
        validation_data = data_sorted[val_mask].copy()
        test_data = data_sorted[test_mask].copy()
        
        # Check for empty splits
        if any(len(df) == 0 for df in [train_data, validation_data, test_data]):
            raise ValueError("One or more splits are empty - check date ranges and gap settings")
        
        # Create metadata
        metadata = {
            "split_method": "date",
            "train_end": train_end_dt,
            "validation_end": validation_end_dt,
            "gap_days": gap_days,
            "date_ranges": {
                "train": (dates[train_mask].min(), dates[train_mask].max()),
                "validation": (dates[val_mask].min(), dates[val_mask].max()),
                "test": (dates[test_mask].min(), dates[test_mask].max()),
            },
        }
        
        return DataSplitResult(train_data, validation_data, test_data, metadata)
    
    def _create_walk_forward_splits_impl(
        self,
        data: pd.DataFrame,
        window_size_days: int,
        step_size_days: int,
        validation_days: int,
        gap_days: int,
        min_samples: int,
    ) -> List[DataSplitResult]:
        """Internal implementation of walk-forward splits."""
        # Validate input data
        self._validate_data(data)
        
        # Sort data by date
        data_sorted = self._sort_by_date(data)
        dates = self._get_datetime_series(data_sorted)
        
        start_date = dates.min()
        end_date = dates.max()
        
        splits = []
        current_start = start_date
        
        while True:
            # Define current window
            train_end = current_start + timedelta(days=window_size_days)
            val_start = train_end + timedelta(days=gap_days)
            val_end = val_start + timedelta(days=validation_days)
            
            # Check if we have enough data
            if val_end > end_date:
                break
            
            # Create masks
            train_mask = (dates >= current_start) & (dates < train_end)
            val_mask = (dates >= val_start) & (dates < val_end)
            
            # Create splits
            train_data = data_sorted[train_mask].copy()
            val_data = data_sorted[val_mask].copy()
            test_data = pd.DataFrame()  # No test data in walk-forward
            
            # Check minimum sample requirements
            if len(train_data) >= min_samples and len(val_data) >= min_samples:
                metadata = {
                    "split_method": "walk_forward",
                    "window_index": len(splits),
                    "train_start": current_start,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "gap_days": gap_days,
                }
                
                splits.append(DataSplitResult(train_data, val_data, test_data, metadata))
            
            # Move to next window
            current_start += timedelta(days=step_size_days)
        
        if not splits:
            raise ValueError("No valid walk-forward splits could be created")
        
        return splits
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        # Check for datetime information
        has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
        has_datetime_column = self.date_column in data.columns
        
        if not has_datetime_index and not has_datetime_column:
            raise ValueError(f"No datetime information found. Need DatetimeIndex or '{self.date_column}' column")
    
    def _sort_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sort data by datetime."""
        if isinstance(data.index, pd.DatetimeIndex):
            return data.sort_index()
        else:
            return data.sort_values(self.date_column)
    
    def _get_datetime_series(self, data: pd.DataFrame) -> pd.Series:
        """Get datetime series from data."""
        if isinstance(data.index, pd.DatetimeIndex):
            return data.index.to_series()
        else:
            return pd.to_datetime(data[self.date_column])
    
    def _calculate_gap_samples(self, data: pd.DataFrame, gap_days: int) -> int:
        """Calculate number of samples equivalent to gap_days."""
        if gap_days == 0:
            return 0
        
        dates = self._get_datetime_series(data)
        if len(dates) < 2:
            return 0
        
        # Estimate average time between samples
        time_diffs = dates.diff().dropna()
        avg_diff = time_diffs.mean()
        
        # Convert gap_days to number of samples
        gap_delta = timedelta(days=gap_days)
        gap_samples = int(gap_delta / avg_diff)
        
        return max(0, gap_samples)
    
    def _get_lock(self):
        """Get thread lock if thread safety is enabled."""
        return self._lock if self._lock is not None else threading.Lock()