"""
Unit tests for PurgedCrossValidator.

Tests the purged cross-validation strategy designed specifically
for financial time series with data leakage prevention.
"""

import unittest
from unittest.mock import Mock
import pandas as pd
import numpy as np

from btc_research.optimization.validators.purged_cv import PurgedCrossValidator
from btc_research.optimization.types import ValidationResult
from tests.fixtures.sample_data import create_btc_sample_data


class TestPurgedCrossValidator(unittest.TestCase):
    """Test cases for PurgedCrossValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = create_btc_sample_data(periods=1000, freq="1h")
        self.mock_backtest = Mock(return_value={"sharpe": 0.5, "return": 0.1})
        self.parameters = {"param1": 10}
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        validator = PurgedCrossValidator(self.data)
        
        self.assertEqual(validator.n_splits, 5)
        self.assertEqual(validator.purge_length, 24)  # 1 day default
        self.assertEqual(validator.embargo_length, 12)  # 12 hours default
        self.assertFalse(validator.shuffle)
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        validator = PurgedCrossValidator(
            data=self.data,
            n_splits=3,
            purge_length=48,
            embargo_length=24,
            shuffle=True,
            random_seed=42
        )
        
        self.assertEqual(validator.n_splits, 3)
        self.assertEqual(validator.purge_length, 48)
        self.assertEqual(validator.embargo_length, 24)
        self.assertTrue(validator.shuffle)
        self.assertEqual(validator.random_seed, 42)
    
    def test_split_data_basic(self):
        """Test basic purged cross-validation splitting."""
        validator = PurgedCrossValidator(self.data, n_splits=3, purge_length=10)
        splits = validator.split_data()
        
        self.assertEqual(len(splits), 3)
        
        for train_data, val_data in splits:
            self.assertIsInstance(train_data, pd.DataFrame)
            self.assertIsInstance(val_data, pd.DataFrame)
            self.assertGreater(len(train_data), 0)
            self.assertGreater(len(val_data), 0)
    
    def test_purge_functionality(self):
        """Test that purging removes data points correctly."""
        purge_length = 20
        validator = PurgedCrossValidator(
            self.data, 
            n_splits=2, 
            purge_length=purge_length,
            embargo_length=0  # No embargo for this test
        )
        splits = validator.split_data()
        
        for train_data, val_data in splits:
            # Find the gap between training and validation
            all_train_indices = set(self.data.index.get_indexer(train_data.index))
            all_val_indices = set(self.data.index.get_indexer(val_data.index))
            
            # There should be no overlap
            self.assertEqual(len(all_train_indices & all_val_indices), 0)
            
            # Check that the purge gap exists
            if len(all_train_indices) > 0 and len(all_val_indices) > 0:
                max_train_idx = max(all_train_indices)
                min_val_idx = min(all_val_indices)
                gap = min_val_idx - max_train_idx
                self.assertGreaterEqual(gap, purge_length)
    
    def test_embargo_functionality(self):
        """Test embargo period implementation."""
        embargo_length = 15
        validator = PurgedCrossValidator(
            self.data,
            n_splits=2,
            purge_length=0,  # No purge for this test
            embargo_length=embargo_length
        )
        splits = validator.split_data()
        
        # Embargo should prevent validation data immediately after training periods
        for train_data, val_data in splits:
            # Check that embargo period is respected
            train_indices = set(self.data.index.get_indexer(train_data.index))
            val_indices = set(self.data.index.get_indexer(val_data.index))
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                max_train_idx = max(train_indices)
                min_val_idx = min(val_indices)
                gap = min_val_idx - max_train_idx
                self.assertGreaterEqual(gap, embargo_length)
    
    def test_validate(self):
        """Test validation workflow."""
        validator = PurgedCrossValidator(self.data, n_splits=3)
        result = validator.validate(self.parameters, self.mock_backtest)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(len(result.fold_results), 3)
        self.assertEqual(self.mock_backtest.call_count, 3)
        
        # Check metadata
        self.assertIn("n_splits", result.metadata)
        self.assertIn("purge_length", result.metadata)
        self.assertIn("embargo_length", result.metadata)
    
    def test_no_data_leakage(self):
        """Test that no data leakage occurs between folds."""
        validator = PurgedCrossValidator(
            self.data, 
            n_splits=3, 
            purge_length=10, 
            embargo_length=5
        )
        splits = validator.split_data()
        
        for i, (train_data, val_data) in enumerate(splits):
            train_timestamps = set(train_data.index)
            val_timestamps = set(val_data.index)
            
            # No overlap between training and validation
            self.assertEqual(len(train_timestamps & val_timestamps), 0)
            
            # Validation should not contain any training data
            for val_time in val_timestamps:
                self.assertNotIn(val_time, train_timestamps)


if __name__ == "__main__":
    unittest.main()