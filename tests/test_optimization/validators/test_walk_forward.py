"""
Unit tests for WalkForwardValidator.

Tests the walk-forward validation strategy for time series data,
including proper time ordering and realistic backtesting scenarios.
"""

import unittest
from unittest.mock import Mock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from btc_research.optimization.validators.walk_forward import WalkForwardValidator
from btc_research.optimization.types import ValidationResult
from tests.fixtures.sample_data import create_btc_sample_data


class TestWalkForwardValidator(unittest.TestCase):
    """Test cases for WalkForwardValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample time series data
        self.data = create_btc_sample_data(periods=1000, freq="1h")
        
        # Mock backtest function that returns realistic metrics
        def mock_backtest(data, params):
            # Simple mock that varies based on data length and params
            base_return = len(data) * 0.0001  # Longer periods = higher returns
            param_bonus = params.get("param1", 0) * 0.01
            volatility = 0.02 + np.random.normal(0, 0.005)
            
            total_return = base_return + param_bonus + np.random.normal(0, 0.01)
            sharpe = total_return / volatility if volatility > 0 else 0
            max_drawdown = -abs(np.random.normal(0.05, 0.02))
            
            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "win_rate": np.random.uniform(0.4, 0.7)
            }
        
        self.backtest_function = mock_backtest
        self.parameters = {"param1": 10, "param2": 0.5}
    
    def test_initialization_default(self):
        """Test successful initialization with default parameters."""
        validator = WalkForwardValidator(self.data)
        
        self.assertTrue(validator.data.equals(self.data))
        self.assertEqual(validator.date_column, "timestamp")
        self.assertEqual(validator.window_size, 252)  # Default 1 year
        self.assertEqual(validator.step_size, 21)     # Default 1 month
        self.assertEqual(validator.min_train_size, 100)
        self.assertEqual(validator.max_train_size, None)
        self.assertIsNone(validator.random_seed)
    
    def test_initialization_with_parameters(self):
        """Test initialization with custom parameters."""
        validator = WalkForwardValidator(
            data=self.data,
            window_size=500,
            step_size=50,
            min_train_size=200,
            max_train_size=800,
            date_column="timestamp",
            random_seed=42
        )
        
        self.assertEqual(validator.window_size, 500)
        self.assertEqual(validator.step_size, 50)
        self.assertEqual(validator.min_train_size, 200)
        self.assertEqual(validator.max_train_size, 800)
        self.assertEqual(validator.random_seed, 42)
    
    def test_initialization_invalid_parameters(self):
        """Test initialization fails with invalid parameters."""
        # Window size too small
        with self.assertRaises(ValueError):
            WalkForwardValidator(self.data, window_size=10)
        
        # Step size too small
        with self.assertRaises(ValueError):
            WalkForwardValidator(self.data, step_size=0)
        
        # Min train size too small
        with self.assertRaises(ValueError):
            WalkForwardValidator(self.data, min_train_size=5)
        
        # Max train size smaller than min
        with self.assertRaises(ValueError):
            WalkForwardValidator(
                self.data, 
                min_train_size=200, 
                max_train_size=100
            )
    
    def test_split_data_basic(self):
        """Test basic data splitting functionality."""
        validator = WalkForwardValidator(
            data=self.data,
            window_size=200,
            step_size=50,
            min_train_size=100
        )
        
        splits = validator.split_data()
        
        self.assertIsInstance(splits, list)
        self.assertGreater(len(splits), 0)
        
        for train_data, val_data in splits:
            self.assertIsInstance(train_data, pd.DataFrame)
            self.assertIsInstance(val_data, pd.DataFrame)
            self.assertGreater(len(train_data), 0)
            self.assertGreater(len(val_data), 0)
            
            # Training data should come before validation data
            train_end = train_data.index[-1]
            val_start = val_data.index[0]
            self.assertLess(train_end, val_start)
    
    def test_split_data_time_ordering(self):
        """Test that splits maintain proper time ordering."""
        validator = WalkForwardValidator(
            data=self.data,
            window_size=150,
            step_size=30
        )
        
        splits = validator.split_data()
        
        previous_val_end = None
        for i, (train_data, val_data) in enumerate(splits):
            # Each split should have proper internal ordering
            self.assertTrue(train_data.index.is_monotonic_increasing)
            self.assertTrue(val_data.index.is_monotonic_increasing)
            
            # Training should come before validation
            self.assertLess(train_data.index[-1], val_data.index[0])
            
            # Validation periods should advance over time
            if previous_val_end is not None:
                self.assertGreater(val_data.index[0], previous_val_end)
            
            previous_val_end = val_data.index[-1]
    
    def test_split_data_window_sizes(self):
        """Test that splits respect window size constraints."""
        window_size = 100
        validator = WalkForwardValidator(
            data=self.data,
            window_size=window_size,
            step_size=25
        )
        
        splits = validator.split_data()
        
        for train_data, val_data in splits:
            # Training data should not exceed max_train_size if specified
            if validator.max_train_size:
                self.assertLessEqual(len(train_data), validator.max_train_size)
            
            # Training data should meet minimum size
            self.assertGreaterEqual(len(train_data), validator.min_train_size)
            
            # Validation data should be reasonable size
            self.assertGreater(len(val_data), 0)
            self.assertLessEqual(len(val_data), window_size)
    
    def test_split_data_with_max_train_size(self):
        """Test data splitting with maximum training size limit."""
        validator = WalkForwardValidator(
            data=self.data,
            window_size=200,
            step_size=50,
            min_train_size=100,
            max_train_size=300
        )
        
        splits = validator.split_data()
        
        for train_data, val_data in splits:
            self.assertLessEqual(len(train_data), 300)
            self.assertGreaterEqual(len(train_data), 100)
    
    def test_split_data_insufficient_data(self):
        """Test behavior with insufficient data for splitting."""
        # Create very small dataset
        small_data = self.data.iloc[:50]
        
        validator = WalkForwardValidator(
            data=small_data,
            window_size=100,  # Larger than available data
            step_size=25,
            min_train_size=30
        )
        
        splits = validator.split_data()
        
        # Should return empty list or handle gracefully
        if len(splits) > 0:
            # If splits are created, they should still be valid
            for train_data, val_data in splits:
                self.assertGreater(len(train_data), 0)
                self.assertGreater(len(val_data), 0)
    
    def test_validate_basic(self):
        """Test basic validation functionality."""
        validator = WalkForwardValidator(
            data=self.data,
            window_size=200,
            step_size=100  # Large step for fewer folds
        )
        
        result = validator.validate(self.parameters, self.backtest_function)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.parameters, self.parameters)
        self.assertGreater(len(result.fold_results), 0)
        
        # Check that all expected metrics are present
        for fold_result in result.fold_results:
            self.assertIn("total_return", fold_result)
            self.assertIn("sharpe_ratio", fold_result)
            self.assertIn("max_drawdown", fold_result)
            self.assertIn("volatility", fold_result)
            self.assertIn("win_rate", fold_result)
        
        # Check summary statistics
        self.assertIn("total_return", result.mean_metrics)
        self.assertIn("sharpe_ratio", result.mean_metrics)
        self.assertIn("total_return", result.std_metrics)
        self.assertIn("sharpe_ratio", result.std_metrics)
        self.assertIn("total_return", result.confidence_intervals)
    
    def test_validate_multiple_folds(self):
        """Test validation with multiple walk-forward folds."""
        validator = WalkForwardValidator(
            data=self.data,
            window_size=150,
            step_size=50,  # Should create multiple folds
            min_train_size=100
        )
        
        result = validator.validate(self.parameters, self.backtest_function)
        
        # Should have multiple folds
        self.assertGreaterEqual(len(result.fold_results), 2)
        
        # Each fold should have consistent metrics
        first_fold_metrics = set(result.fold_results[0].keys())
        for fold_result in result.fold_results[1:]:
            self.assertEqual(set(fold_result.keys()), first_fold_metrics)
        
        # Metadata should track fold information
        self.assertIn("n_folds", result.metadata)
        self.assertEqual(result.metadata["n_folds"], len(result.fold_results))
        self.assertIn("window_size", result.metadata)
        self.assertIn("step_size", result.metadata)
    
    def test_validate_backtest_function_calls(self):
        """Test that backtest function is called correctly for each fold."""
        # Track calls to backtest function
        call_log = []
        
        def tracking_backtest(data, params):
            call_log.append({
                "data_length": len(data),
                "data_start": data.index[0],
                "data_end": data.index[-1],
                "params": params.copy()
            })
            return {"return": 0.1, "sharpe": 0.5}
        
        validator = WalkForwardValidator(
            data=self.data,
            window_size=200,
            step_size=100
        )
        
        result = validator.validate(self.parameters, tracking_backtest)
        
        # Should have called backtest for each fold
        self.assertEqual(len(call_log), len(result.fold_results))
        
        # Each call should have correct parameters
        for call in call_log:
            self.assertEqual(call["params"], self.parameters)
            self.assertGreater(call["data_length"], 0)
        
        # Calls should be in chronological order
        for i in range(1, len(call_log)):
            self.assertGreater(call_log[i]["data_start"], call_log[i-1]["data_start"])
    
    def test_validate_with_failing_backtest(self):
        """Test validation handles backtest function failures gracefully."""
        def failing_backtest(data, params):
            if len(data) > 150:  # Fail on larger datasets
                raise ValueError("Simulated backtest failure")
            return {"return": 0.05, "sharpe": 0.3}
        
        validator = WalkForwardValidator(
            data=self.data,
            window_size=200,
            step_size=50
        )
        
        # Should handle failures gracefully
        result = validator.validate(self.parameters, failing_backtest)
        
        # Should still return a result, possibly with fewer folds
        self.assertIsInstance(result, ValidationResult)
        # Some folds might succeed
        self.assertGreaterEqual(len(result.fold_results), 0)
    
    def test_validate_edge_cases(self):
        """Test validation with edge case parameters."""
        # Very small window size
        validator_small = WalkForwardValidator(
            data=self.data,
            window_size=50,
            step_size=25,
            min_train_size=30
        )
        
        result_small = validator_small.validate(self.parameters, self.backtest_function)
        self.assertIsInstance(result_small, ValidationResult)
        
        # Large step size (few folds)
        validator_large_step = WalkForwardValidator(
            data=self.data,
            window_size=200,
            step_size=300  # Larger than window
        )
        
        result_large_step = validator_large_step.validate(self.parameters, self.backtest_function)
        self.assertIsInstance(result_large_step, ValidationResult)
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger datasets."""
        # Create larger dataset
        large_data = create_btc_sample_data(periods=5000, freq="1h")
        
        validator = WalkForwardValidator(
            data=large_data,
            window_size=500,
            step_size=100
        )
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        result = validator.validate(self.parameters, self.backtest_function)
        elapsed_time = time.time() - start_time
        
        self.assertIsInstance(result, ValidationResult)
        self.assertLess(elapsed_time, 30)  # Should complete within 30 seconds
        self.assertGreater(len(result.fold_results), 5)  # Should create multiple folds
    
    def test_validate_metadata_completeness(self):
        """Test that validation metadata is complete and accurate."""
        validator = WalkForwardValidator(
            data=self.data,
            window_size=200,
            step_size=75,
            min_train_size=150,
            max_train_size=400
        )
        
        result = validator.validate(self.parameters, self.backtest_function)
        
        # Check metadata completeness
        expected_metadata = [
            "n_folds", "window_size", "step_size", 
            "min_train_size", "max_train_size",
            "data_start", "data_end", "total_data_points"
        ]
        
        for key in expected_metadata:
            self.assertIn(key, result.metadata)
        
        # Check metadata accuracy
        self.assertEqual(result.metadata["window_size"], 200)
        self.assertEqual(result.metadata["step_size"], 75)
        self.assertEqual(result.metadata["min_train_size"], 150)
        self.assertEqual(result.metadata["max_train_size"], 400)
        self.assertEqual(result.metadata["total_data_points"], len(self.data))
    
    def test_datetime_index_handling(self):
        """Test proper handling of datetime indices."""
        # Ensure data has datetime index
        self.assertIsInstance(self.data.index, pd.DatetimeIndex)
        
        validator = WalkForwardValidator(
            data=self.data,
            window_size=200,
            step_size=50
        )
        
        splits = validator.split_data()
        
        for train_data, val_data in splits:
            # Both should maintain datetime index
            self.assertIsInstance(train_data.index, pd.DatetimeIndex)
            self.assertIsInstance(val_data.index, pd.DatetimeIndex)
            
            # Indices should be timezone-aware if original was
            if self.data.index.tz is not None:
                self.assertIsNotNone(train_data.index.tz)
                self.assertIsNotNone(val_data.index.tz)
    
    def test_data_integrity_preservation(self):
        """Test that original data is not modified during validation."""
        original_data = self.data.copy()
        
        validator = WalkForwardValidator(
            data=self.data,
            window_size=200,
            step_size=100
        )
        
        result = validator.validate(self.parameters, self.backtest_function)
        
        # Original data should be unchanged
        pd.testing.assert_frame_equal(self.data, original_data)
        
        # Validator's internal data should also be unchanged
        pd.testing.assert_frame_equal(validator.data, original_data)
    
    def test_reproducibility(self):
        """Test that validation results are reproducible."""
        def deterministic_backtest(data, params):
            # Deterministic function of data and params
            return {
                "return": len(data) * 0.0001 + params.get("param1", 0) * 0.001,
                "sharpe": 0.5,
                "drawdown": -0.05
            }
        
        validator1 = WalkForwardValidator(
            data=self.data,
            window_size=200,
            step_size=100,
            random_seed=42
        )
        
        validator2 = WalkForwardValidator(
            data=self.data,
            window_size=200,
            step_size=100,
            random_seed=42
        )
        
        result1 = validator1.validate(self.parameters, deterministic_backtest)
        result2 = validator2.validate(self.parameters, deterministic_backtest)
        
        # Results should be identical
        self.assertEqual(len(result1.fold_results), len(result2.fold_results))
        
        for fold1, fold2 in zip(result1.fold_results, result2.fold_results):
            for metric in fold1.keys():
                self.assertAlmostEqual(fold1[metric], fold2[metric], places=6)


if __name__ == "__main__":
    unittest.main()