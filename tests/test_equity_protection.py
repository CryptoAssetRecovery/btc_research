"""
Test suite for the equity protection system.

This module contains comprehensive tests for the equity protection functionality,
including unit tests for core components and integration tests for the complete
system.
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from btc_research.core.equity_protection import (
    EquityProtection,
    EquityProtectionAnalyzer,
    EquityProtectionError,
    create_equity_protection_config,
    integrate_equity_protection_with_strategy
)
from btc_research.utils.equity_protection_integration import (
    add_equity_protection_to_config,
    EquityProtectionMonitor
)


class TestEquityProtection(unittest.TestCase):
    """Test cases for the EquityProtection class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.protection = EquityProtection(
            drawdown_threshold=0.25,
            enable_on_bias_flip=True,
            smoothing_window=3,
            min_equity_history=5,
            debug=False
        )
    
    def test_initialization(self):
        """Test equity protection initialization."""
        self.assertEqual(self.protection.drawdown_threshold, 0.25)
        self.assertTrue(self.protection.enable_on_bias_flip)
        self.assertEqual(self.protection.smoothing_window, 3)
        self.assertEqual(self.protection.min_equity_history, 5)
        self.assertEqual(self.protection.current_equity, 0.0)
        self.assertEqual(self.protection.peak_equity, 0.0)
        self.assertEqual(self.protection.current_drawdown, 0.0)
        self.assertFalse(self.protection.protection_active)
        self.assertFalse(self.protection.trading_disabled)
    
    def test_equity_update_normal(self):
        """Test normal equity updates without protection trigger."""
        equity_values = [10000, 10500, 11000, 10800, 11200]
        
        for equity in equity_values:
            result = self.protection.update_equity(equity)
            
            # Check return values
            self.assertIsInstance(result, dict)
            self.assertIn('equity', result)
            self.assertIn('drawdown', result)
            self.assertIn('protection_active', result)
            self.assertIn('trading_disabled', result)
            
            # Check that protection is not triggered
            self.assertFalse(result['protection_active'])
            self.assertFalse(result['trading_disabled'])
        
        # Check final state
        self.assertEqual(self.protection.current_equity, 11200)
        self.assertEqual(self.protection.peak_equity, 11200)
        self.assertEqual(self.protection.current_drawdown, 0.0)
    
    def test_equity_update_with_drawdown(self):
        """Test equity updates with drawdown but no protection trigger."""
        equity_values = [10000, 11000, 10500, 9800, 9200]  # ~16% drawdown
        
        for equity in equity_values:
            result = self.protection.update_equity(equity)
        
        # Check drawdown calculation
        expected_drawdown = (11000 - 9200) / 11000  # ~16.4%
        self.assertAlmostEqual(self.protection.current_drawdown, expected_drawdown, places=3)
        
        # Protection should not be triggered (threshold is 25%)
        self.assertFalse(self.protection.protection_active)
        self.assertFalse(self.protection.trading_disabled)
    
    def test_protection_trigger(self):
        """Test protection trigger when drawdown exceeds threshold."""
        equity_values = [10000, 12000, 11000, 9500, 8800, 8000]  # ~33% drawdown
        
        protection_triggered = False
        for equity in equity_values:
            result = self.protection.update_equity(equity)
            if result.get('protection_triggered', False):
                protection_triggered = True
        
        # Protection should be triggered
        self.assertTrue(protection_triggered)
        self.assertTrue(self.protection.protection_active)
        self.assertTrue(self.protection.trading_disabled)
        
        # Check final drawdown
        expected_drawdown = (12000 - 8000) / 12000  # ~33%
        self.assertAlmostEqual(self.protection.current_drawdown, expected_drawdown, places=3)
        self.assertGreater(self.protection.current_drawdown, self.protection.drawdown_threshold)
    
    def test_bias_flip_recovery(self):
        """Test bias flip recovery mechanism."""
        # First trigger protection
        equity_values = [10000, 12000, 8000]  # ~33% drawdown
        for equity in equity_values:
            self.protection.update_equity(equity)
        
        # Verify protection is active
        self.assertTrue(self.protection.protection_active)
        self.assertTrue(self.protection.trading_disabled)
        
        # Trigger bias flip
        bias_result = self.protection.update_bias("bull")
        
        # Check bias flip results
        self.assertIsInstance(bias_result, dict)
        self.assertIn('bias', bias_result)
        self.assertIn('is_flip', bias_result)
        self.assertIn('trading_enabled', bias_result)
        
        # Trading should be re-enabled
        self.assertTrue(bias_result['trading_enabled'])
        self.assertFalse(self.protection.trading_disabled)
        
        # Protection should still be active (equity hasn't recovered)
        self.assertTrue(self.protection.protection_active)
    
    def test_invalid_equity_values(self):
        """Test handling of invalid equity values."""
        invalid_values = [-1000, 0, None, "invalid", np.nan]
        
        for invalid_value in invalid_values:
            with self.assertRaises(EquityProtectionError):
                self.protection.update_equity(invalid_value)
    
    def test_invalid_bias_values(self):
        """Test handling of invalid bias values."""
        invalid_biases = ["invalid", "up", "down", None, 123]
        
        for invalid_bias in invalid_biases:
            with self.assertRaises(EquityProtectionError):
                self.protection.update_bias(invalid_bias)
    
    def test_smoothing_functionality(self):
        """Test equity curve smoothing functionality."""
        protection_with_smoothing = EquityProtection(
            drawdown_threshold=0.25,
            smoothing_window=5,
            min_equity_history=3
        )
        
        # Add some volatile equity values
        equity_values = [10000, 12000, 9000, 11000, 8000, 10000, 7000]
        
        for equity in equity_values:
            result = protection_with_smoothing.update_equity(equity)
            
            # Check that smoothed drawdown is calculated
            if len(protection_with_smoothing.equity_history) >= 5:
                self.assertIn('smoothed_drawdown', 
                            protection_with_smoothing.equity_history[-1])
    
    def test_equity_stats(self):
        """Test equity statistics generation."""
        # Add some equity data
        equity_values = [10000, 11000, 10500, 9800, 9200, 8500, 9000]
        
        for equity in equity_values:
            self.protection.update_equity(equity)
        
        # Get stats
        stats = self.protection.get_equity_stats()
        
        # Check required fields
        required_fields = [
            'current_equity', 'peak_equity', 'current_drawdown', 'max_drawdown',
            'protection_active', 'trading_disabled', 'drawdown_threshold',
            'total_updates', 'equity_history_length'
        ]
        
        for field in required_fields:
            self.assertIn(field, stats)
        
        # Check values
        self.assertEqual(stats['current_equity'], 9000)
        self.assertEqual(stats['peak_equity'], 11000)
        self.assertEqual(stats['total_updates'], len(equity_values))
        self.assertEqual(stats['equity_history_length'], len(equity_values))
    
    def test_equity_curve_dataframe(self):
        """Test equity curve DataFrame generation."""
        # Add some equity data
        equity_values = [10000, 11000, 10500, 9800, 9200]
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(len(equity_values))]
        
        for equity, timestamp in zip(equity_values, timestamps):
            self.protection.update_equity(equity, timestamp)
        
        # Get equity curve
        df = self.protection.get_equity_curve()
        
        # Check DataFrame properties
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(equity_values))
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
        
        # Check required columns
        required_columns = ['equity', 'peak_equity', 'drawdown', 'protection_active', 'trading_disabled']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check values
        self.assertEqual(df['equity'].iloc[-1], 9200)
        self.assertEqual(df['peak_equity'].iloc[-1], 11000)
    
    def test_drawdown_curve_dataframe(self):
        """Test drawdown curve DataFrame generation."""
        # Add some equity data
        equity_values = [10000, 11000, 10500, 9800, 9200]
        
        for equity in equity_values:
            self.protection.update_equity(equity)
        
        # Get drawdown curve
        df = self.protection.get_drawdown_curve()
        
        # Check DataFrame properties
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(equity_values))
        
        # Check required columns
        required_columns = ['drawdown', 'drawdown_pct', 'drawdown_amount', 'recovery_factor', 'protection_active']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check calculations
        final_drawdown = df['drawdown'].iloc[-1]
        final_drawdown_pct = df['drawdown_pct'].iloc[-1]
        self.assertAlmostEqual(final_drawdown * 100, final_drawdown_pct, places=2)


class TestEquityProtectionAnalyzer(unittest.TestCase):
    """Test cases for the EquityProtectionAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.protection = EquityProtection(drawdown_threshold=0.25, debug=False)
        self.analyzer = EquityProtectionAnalyzer(self.protection)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, EquityProtectionAnalyzer)
        self.assertEqual(self.analyzer.protection, self.protection)
    
    def test_protection_effectiveness_analysis(self):
        """Test protection effectiveness analysis."""
        # Add equity data that triggers protection
        equity_values = [10000, 12000, 11000, 9500, 8800, 8000, 9000]
        
        for equity in equity_values:
            self.protection.update_equity(equity)
        
        # Add bias flip
        self.protection.update_bias("bull")
        
        # Run analysis
        analysis = self.analyzer.analyze_protection_effectiveness()
        
        # Check analysis structure
        self.assertIsInstance(analysis, dict)
        self.assertIn('protection_summary', analysis)
        self.assertIn('recovery_analysis', analysis)
        self.assertIn('drawdown_analysis', analysis)
        self.assertIn('bias_flip_analysis', analysis)
        
        # Check protection summary
        ps = analysis['protection_summary']
        self.assertIn('total_triggers', ps)
        self.assertIn('max_drawdown_observed', ps)
        self.assertIn('protection_threshold', ps)
        self.assertIn('effectiveness_ratio', ps)
    
    def test_protection_report_generation(self):
        """Test protection report generation."""
        # Add some equity data
        equity_values = [10000, 11000, 10500, 9800, 9200]
        
        for equity in equity_values:
            self.protection.update_equity(equity)
        
        # Generate report
        report = self.analyzer.generate_protection_report()
        
        # Check report properties
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)  # Should be substantial
        self.assertIn('Equity Protection System Report', report)
        self.assertIn('Current Status', report)
        self.assertIn('Protection Configuration', report)


class TestEquityProtectionIntegration(unittest.TestCase):
    """Test cases for equity protection integration utilities."""
    
    def test_add_protection_to_config(self):
        """Test adding protection to strategy configuration."""
        # Create base config
        base_config = {
            'name': 'Test Strategy',
            'logic': {
                'entry_long': ['rsi < 30'],
                'exit_long': ['rsi > 70']
            }
        }
        
        # Add protection
        protected_config = add_equity_protection_to_config(base_config)
        
        # Check that protection was added
        self.assertIn('equity_protection', protected_config)
        self.assertTrue(protected_config['equity_protection']['enabled'])
        
        # Check that protection rules were added to logic
        self.assertIn('not equity_protection_active', protected_config['logic']['entry_long'])
        self.assertIn('equity_protection_triggered', protected_config['logic']['exit_long'])
    
    def test_create_equity_protection_config(self):
        """Test equity protection configuration creation."""
        config = create_equity_protection_config(
            drawdown_threshold=0.20,
            enable_on_bias_flip=False,
            smoothing_window=10
        )
        
        # Check configuration structure
        self.assertIn('equity_protection', config)
        self.assertIn('risk_management', config)
        
        # Check values
        ep_config = config['equity_protection']
        self.assertEqual(ep_config['drawdown_threshold'], 0.20)
        self.assertFalse(ep_config['enable_on_bias_flip'])
        self.assertEqual(ep_config['smoothing_window'], 10)
    
    def test_integrate_equity_protection_with_strategy(self):
        """Test strategy integration function."""
        # Create base strategy config
        strategy_config = {
            'name': 'Test Strategy',
            'logic': {
                'entry_long': ['signal == 1'],
                'exit_long': ['signal == -1']
            }
        }
        
        # Create protection config
        protection_config = create_equity_protection_config(drawdown_threshold=0.30)
        
        # Integrate
        integrated_config = integrate_equity_protection_with_strategy(
            strategy_config, protection_config
        )
        
        # Check integration
        self.assertIn('equity_protection', integrated_config)
        self.assertIn('risk_management', integrated_config)
        
        # Check that original config wasn't modified
        self.assertNotIn('equity_protection', strategy_config)
        
        # Check that protection rules were added
        self.assertIn('not equity_protection_active', integrated_config['logic']['entry_long'])


class TestEquityProtectionMonitor(unittest.TestCase):
    """Test cases for the EquityProtectionMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.protection = EquityProtection(drawdown_threshold=0.25, debug=False)
        self.alerts = []
        self.position_actions = []
        
        def alert_callback(message: str, severity: str = "info"):
            self.alerts.append({'message': message, 'severity': severity})
        
        def position_manager(action: str):
            self.position_actions.append(action)
        
        self.monitor = EquityProtectionMonitor(
            self.protection,
            alert_callback=alert_callback,
            position_manager=position_manager
        )
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertIsInstance(self.monitor, EquityProtectionMonitor)
        self.assertEqual(self.monitor.protection, self.protection)
        self.assertIsNotNone(self.monitor.alert_callback)
        self.assertIsNotNone(self.monitor.position_manager)
    
    def test_monitor_normal_operation(self):
        """Test monitor during normal operation."""
        equity_values = [10000, 10500, 11000, 10800]
        
        for equity in equity_values:
            result = self.monitor.update_and_monitor(equity)
            
            # Check result structure
            self.assertIn('monitoring_result', result)
            self.assertIn('protection_update', result)
            self.assertIn('actions_taken', result)
            
            # No actions should be taken during normal operation
            self.assertEqual(len(result['actions_taken']), 0)
        
        # No alerts or position actions should be triggered
        self.assertEqual(len(self.alerts), 0)
        self.assertEqual(len(self.position_actions), 0)
    
    def test_monitor_protection_trigger(self):
        """Test monitor when protection is triggered."""
        # Create scenario that triggers protection
        equity_values = [10000, 12000, 11000, 9500, 8800, 8000]
        
        for equity in equity_values:
            result = self.monitor.update_and_monitor(equity)
        
        # Check that alerts were generated
        self.assertGreater(len(self.alerts), 0)
        
        # Check that position actions were triggered
        self.assertGreater(len(self.position_actions), 0)
        self.assertIn('flatten_all', self.position_actions)
        
        # Check that critical alert was sent
        critical_alerts = [a for a in self.alerts if a['severity'] == 'critical']
        self.assertGreater(len(critical_alerts), 0)
    
    def test_monitor_report_generation(self):
        """Test monitor report generation."""
        # Add some equity data
        equity_values = [10000, 11000, 10500, 9800]
        
        for equity in equity_values:
            self.monitor.update_and_monitor(equity)
        
        # Generate report
        report = self.monitor.get_monitoring_report()
        
        # Check report properties
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)
        self.assertIn('Equity Protection Monitoring Report', report)
        self.assertIn('Current Status', report)


class TestEquityProtectionEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_minimum_equity_history_requirement(self):
        """Test that protection doesn't trigger before minimum history."""
        protection = EquityProtection(
            drawdown_threshold=0.25,
            min_equity_history=10
        )
        
        # Add equity that would trigger protection but with insufficient history
        equity_values = [10000, 12000, 8000]  # 33% drawdown
        
        for equity in equity_values:
            result = protection.update_equity(equity)
        
        # Protection should not be triggered due to insufficient history
        self.assertFalse(protection.protection_active)
        self.assertFalse(protection.trading_disabled)
    
    def test_zero_drawdown_threshold(self):
        """Test behavior with zero drawdown threshold."""
        protection = EquityProtection(drawdown_threshold=0.0)
        
        # Any decline should trigger protection
        equity_values = [10000, 9999]
        
        for equity in equity_values:
            result = protection.update_equity(equity)
        
        # Protection should be triggered immediately
        self.assertTrue(protection.protection_active)
        self.assertTrue(protection.trading_disabled)
    
    def test_very_high_drawdown_threshold(self):
        """Test behavior with very high drawdown threshold."""
        protection = EquityProtection(drawdown_threshold=0.99)  # 99% threshold
        
        # Even extreme drawdown shouldn't trigger
        equity_values = [10000, 2000]  # 80% drawdown
        
        for equity in equity_values:
            result = protection.update_equity(equity)
        
        # Protection should not be triggered
        self.assertFalse(protection.protection_active)
        self.assertFalse(protection.trading_disabled)
    
    def test_recovery_without_bias_flip(self):
        """Test recovery when bias flip is disabled."""
        protection = EquityProtection(
            drawdown_threshold=0.25,
            enable_on_bias_flip=False
        )
        
        # Trigger protection
        equity_values = [10000, 12000, 8000]
        for equity in equity_values:
            protection.update_equity(equity)
        
        # Verify protection is active
        self.assertTrue(protection.protection_active)
        self.assertTrue(protection.trading_disabled)
        
        # Try bias flip
        protection.update_bias("bull")
        
        # Trading should still be disabled
        self.assertTrue(protection.trading_disabled)
    
    def test_force_reset_functionality(self):
        """Test force reset functionality."""
        protection = EquityProtection(drawdown_threshold=0.25)
        
        # Trigger protection
        equity_values = [10000, 12000, 8000]
        for equity in equity_values:
            protection.update_equity(equity)
        
        # Verify protection is active
        self.assertTrue(protection.protection_active)
        self.assertTrue(protection.trading_disabled)
        
        # Force reset
        reset_result = protection.force_reset()
        
        # Check reset result
        self.assertIsInstance(reset_result, dict)
        self.assertIn('reset_type', reset_result)
        self.assertEqual(reset_result['reset_type'], 'force')
        
        # Protection should be disabled
        self.assertFalse(protection.protection_active)
        self.assertFalse(protection.trading_disabled)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)