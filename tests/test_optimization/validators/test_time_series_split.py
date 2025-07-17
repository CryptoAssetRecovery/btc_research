"""
Unit tests for TimeSeriesSplitValidator.

Tests the time series split validation strategy with proper
chronological ordering and gap handling.
"""

import unittest
from unittest.mock import Mock
import pandas as pd
import numpy as np

from btc_research.optimization.validators.time_series_split import TimeSeriesSplitValidator
from btc_research.optimization.types import ValidationResult
from tests.fixtures.sample_data import create_btc_sample_data


class TestTimeSeriesSplitValidator(unittest.TestCase):
    """Test cases for TimeSeriesSplitValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = create_btc_sample_data(periods=1000, freq="1h")
        self.mock_backtest = Mock(return_value={"sharpe": 0.5, "return": 0.1})
        self.parameters = {"param1": 10}
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        validator = TimeSeriesSplitValidator(self.data)
        
        self.assertEqual(validator.n_splits, 5)
        self.assertEqual(validator.test_size, 0.2)
        self.assertEqual(validator.gap, 0)
        self.assertFalse(validator.expanding_window)
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        validator = TimeSeriesSplitValidator(
            data=self.data,
            n_splits=3,
            test_size=0.3,
            gap=24,  # 1 day gap
            expanding_window=True,
            random_seed=42
        )
        
        self.assertEqual(validator.n_splits, 3)
        self.assertEqual(validator.test_size, 0.3)
        self.assertEqual(validator.gap, 24)
        self.assertTrue(validator.expanding_window)
        self.assertEqual(validator.random_seed, 42)
    
    def test_split_data_basic(self):
        """Test basic data splitting."""
        validator = TimeSeriesSplitValidator(self.data, n_splits=3)
        splits = validator.split_data()
        
        self.assertEqual(len(splits), 3)
        
        for train_data, val_data in splits:
            self.assertIsInstance(train_data, pd.DataFrame)
            self.assertIsInstance(val_data, pd.DataFrame)
            self.assertGreater(len(train_data), 0)
            self.assertGreater(len(val_data), 0)
            # Train should come before validation
            self.assertLess(train_data.index[-1], val_data.index[0])
    
    def test_split_data_expanding_window(self):
        """Test expanding window functionality."""
        validator = TimeSeriesSplitValidator(
            self.data, 
            n_splits=3, 
            expanding_window=True
        )
        splits = validator.split_data()
        
        # Training sets should be expanding
        train_sizes = [len(train) for train, _ in splits]
        for i in range(1, len(train_sizes)):
            self.assertGreater(train_sizes[i], train_sizes[i-1])
    
    def test_split_data_with_gap(self):
        """Test data splitting with gap between train and validation."""
        gap_size = 50
        validator = TimeSeriesSplitValidator(
            self.data, 
            n_splits=2, 
            gap=gap_size
        )
        splits = validator.split_data()
        
        for train_data, val_data in splits:
            # Calculate actual gap
            train_end_idx = self.data.index.get_loc(train_data.index[-1])
            val_start_idx = self.data.index.get_loc(val_data.index[0])
            actual_gap = val_start_idx - train_end_idx - 1
            
            self.assertGreaterEqual(actual_gap, gap_size - 1)  # Allow for rounding
    
    def test_validate(self):
        """Test validation workflow."""
        validator = TimeSeriesSplitValidator(self.data, n_splits=3)
        result = validator.validate(self.parameters, self.mock_backtest)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(len(result.fold_results), 3)
        self.assertEqual(self.mock_backtest.call_count, 3)
        
        # Check metadata
        self.assertIn("n_splits", result.metadata)
        self.assertIn("test_size", result.metadata)
        self.assertIn("gap", result.metadata)


if __name__ == "__main__":
    unittest.main()