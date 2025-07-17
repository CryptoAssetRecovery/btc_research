"""
Tests for CLI integration and backward compatibility.

Tests that ensure the new optimization framework integrates properly
with existing CLI interface and maintains backward compatibility.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import yaml
import os

from btc_research.cli.optimise import main as optimize_main
from btc_research.optimization.cli_integration import (
    create_optimization_config,
    run_optimization_cli,
    validate_cli_parameters,
)
from tests.fixtures.sample_data import SAMPLE_CONFIGS


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration with optimization framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.config_data = SAMPLE_CONFIGS["rsi_mean_reversion"].copy()
        
        # Add optimization section
        self.config_data["optimization"] = {
            "method": "bayesian",
            "max_iterations": 10,
            "parameters": [
                {
                    "name": "rsi_period",
                    "type": "integer",
                    "low": 10,
                    "high": 30
                },
                {
                    "name": "rsi_oversold",
                    "type": "float", 
                    "low": 20.0,
                    "high": 40.0
                }
            ]
        }
        
        # Create temp file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.yaml', 
            delete=False
        )
        yaml.dump(self.config_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_create_optimization_config(self):
        """Test optimization config creation from CLI parameters."""
        cli_args = {
            'method': 'bayesian',
            'max_iterations': 50,
            'metric': 'sharpe_ratio',
            'validation': 'walk_forward',
            'validation_window': 252,
            'robustness_test': True,
            'n_simulations': 100
        }
        
        config = create_optimization_config(cli_args)
        
        self.assertEqual(config['method'], 'bayesian')
        self.assertEqual(config['max_iterations'], 50)
        self.assertEqual(config['metric'], 'sharpe_ratio')
        self.assertEqual(config['validation']['method'], 'walk_forward')
        self.assertEqual(config['validation']['window_size'], 252)
        self.assertTrue(config['robustness_test']['enabled'])
        self.assertEqual(config['robustness_test']['n_simulations'], 100)
    
    def test_validate_cli_parameters(self):
        """Test CLI parameter validation."""
        # Valid parameters
        valid_params = {
            'method': 'bayesian',
            'max_iterations': 100,
            'metric': 'sharpe_ratio'
        }
        
        # Should not raise exception
        validate_cli_parameters(valid_params)
        
        # Invalid method
        invalid_method = valid_params.copy()
        invalid_method['method'] = 'invalid_method'
        
        with self.assertRaises(ValueError):
            validate_cli_parameters(invalid_method)
        
        # Invalid metric
        invalid_metric = valid_params.copy()
        invalid_metric['metric'] = 'invalid_metric'
        
        with self.assertRaises(ValueError):
            validate_cli_parameters(invalid_metric)
        
        # Invalid iterations
        invalid_iterations = valid_params.copy()
        invalid_iterations['max_iterations'] = -1
        
        with self.assertRaises(ValueError):
            validate_cli_parameters(invalid_iterations)
    
    @patch('btc_research.optimization.cli_integration.optimize_strategy')
    @patch('btc_research.optimization.cli_integration.load_config')
    def test_run_optimization_cli(self, mock_load_config, mock_optimize):
        """Test CLI optimization execution."""
        # Mock config loading
        mock_load_config.return_value = self.config_data
        
        # Mock optimization result
        mock_result = Mock()
        mock_result.best_parameters = {"rsi_period": 14, "rsi_oversold": 30}
        mock_result.best_score = 1.25
        mock_result.total_evaluations = 10
        mock_result.optimization_time = 45.2
        mock_optimize.return_value = mock_result
        
        # Run CLI optimization
        result = run_optimization_cli(
            config_path=self.temp_file.name,
            method="bayesian",
            max_iterations=10,
            metric="sharpe_ratio"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.best_parameters["rsi_period"], 14)
        self.assertEqual(result.best_score, 1.25)
        
        # Verify optimize_strategy was called correctly
        mock_optimize.assert_called_once()
        call_args = mock_optimize.call_args
        self.assertIn('parameter_specs', call_args.kwargs)
        self.assertIn('max_iterations', call_args.kwargs)
    
    @patch('btc_research.cli.optimise.sys.argv')
    @patch('btc_research.optimization.cli_integration.run_optimization_cli')
    def test_cli_main_function_optimization(self, mock_run_opt, mock_argv):
        """Test main CLI function with optimization parameters."""
        # Mock command line arguments
        mock_argv.__getitem__.side_effect = lambda x: [
            'optimize',
            '--config', self.temp_file.name,
            '--method', 'bayesian',
            '--max-iterations', '20',
            '--metric', 'total_return'
        ][x]
        mock_argv.__len__.return_value = 7
        
        # Mock optimization result
        mock_result = Mock()
        mock_result.best_parameters = {"param": "value"}
        mock_run_opt.return_value = mock_result
        
        # This would test the actual CLI main function
        # For now, we test the integration component
        result = run_optimization_cli(
            config_path=self.temp_file.name,
            method="bayesian",
            max_iterations=20,
            metric="total_return"
        )
        
        mock_run_opt.assert_called_once()
    
    def test_backward_compatibility_grid_search(self):
        """Test that existing grid search configurations still work."""
        # Legacy grid search config
        legacy_config = self.config_data.copy()
        legacy_config["optimization"] = {
            "method": "grid_search",
            "parameters": [
                {
                    "name": "rsi_period",
                    "type": "integer",
                    "values": [10, 14, 20, 30]
                }
            ]
        }
        
        # Save legacy config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(legacy_config, f)
            legacy_config_path = f.name
        
        try:
            with patch('btc_research.optimization.cli_integration.optimize_strategy') as mock_opt:
                mock_result = Mock()
                mock_result.best_parameters = {"rsi_period": 14}
                mock_opt.return_value = mock_result
                
                # Should work with legacy config
                result = run_optimization_cli(
                    config_path=legacy_config_path,
                    method="grid_search"
                )
                
                self.assertIsNotNone(result)
                mock_opt.assert_called_once()
                
        finally:
            os.unlink(legacy_config_path)
    
    def test_parameter_specification_parsing(self):
        """Test parsing of parameter specifications from config."""
        from btc_research.optimization.cli_integration import parse_parameter_specs
        
        param_configs = [
            {
                "name": "int_param",
                "type": "integer",
                "low": 1,
                "high": 100
            },
            {
                "name": "float_param",
                "type": "float",
                "low": 0.0,
                "high": 1.0,
                "step": 0.1
            },
            {
                "name": "cat_param",
                "type": "categorical",
                "choices": ["option1", "option2", "option3"]
            },
            {
                "name": "bool_param",
                "type": "boolean"
            }
        ]
        
        param_specs = parse_parameter_specs(param_configs)
        
        self.assertEqual(len(param_specs), 4)
        
        # Check integer parameter
        int_spec = param_specs[0]
        self.assertEqual(int_spec.name, "int_param")
        self.assertEqual(int_spec.param_type.value, "integer")
        self.assertEqual(int_spec.low, 1)
        self.assertEqual(int_spec.high, 100)
        
        # Check categorical parameter
        cat_spec = param_specs[2]
        self.assertEqual(cat_spec.name, "cat_param")
        self.assertEqual(cat_spec.param_type.value, "categorical")
        self.assertEqual(cat_spec.choices, ["option1", "option2", "option3"])
    
    def test_cli_output_formatting(self):
        """Test CLI output formatting for optimization results."""
        from btc_research.optimization.cli_integration import format_optimization_results
        
        # Mock optimization result
        mock_result = Mock()
        mock_result.best_parameters = {
            "rsi_period": 14,
            "rsi_oversold": 25.5,
            "enable_feature": True
        }
        mock_result.best_score = 1.456789
        mock_result.total_evaluations = 50
        mock_result.optimization_time = 123.45
        mock_result.metadata = {
            "algorithm": "bayesian_optimization",
            "n_initial_points": 5
        }
        
        formatted_output = format_optimization_results(mock_result)
        
        self.assertIn("Best Parameters:", formatted_output)
        self.assertIn("rsi_period: 14", formatted_output)
        self.assertIn("rsi_oversold: 25.5", formatted_output)
        self.assertIn("enable_feature: True", formatted_output)
        self.assertIn("Best Score: 1.457", formatted_output)  # Should be rounded
        self.assertIn("Total Evaluations: 50", formatted_output)
        self.assertIn("Optimization Time: 2.06 minutes", formatted_output)
    
    def test_error_handling_in_cli(self):
        """Test error handling in CLI integration."""
        from btc_research.optimization.cli_integration import handle_optimization_error
        
        # Test different error types
        errors_to_test = [
            FileNotFoundError("Config file not found"),
            ValueError("Invalid parameter specification"),
            RuntimeError("Optimization failed"),
        ]
        
        for error in errors_to_test:
            formatted_error = handle_optimization_error(error)
            
            self.assertIn("Error:", formatted_error)
            self.assertIn(str(error), formatted_error)
            self.assertIn(type(error).__name__, formatted_error)
    
    def test_config_migration_helper(self):
        """Test helper for migrating old grid search configs."""
        from btc_research.optimization.cli_integration import migrate_grid_config
        
        old_config = {
            "grid_search": {
                "rsi_period": [10, 14, 20],
                "ema_length": [20, 50, 100]
            }
        }
        
        new_config = migrate_grid_config(old_config)
        
        self.assertIn("optimization", new_config)
        self.assertEqual(new_config["optimization"]["method"], "grid_search")
        self.assertIn("parameters", new_config["optimization"])
        
        # Check parameter conversion
        params = new_config["optimization"]["parameters"]
        rsi_param = next(p for p in params if p["name"] == "rsi_period")
        self.assertEqual(rsi_param["type"], "categorical")
        self.assertEqual(rsi_param["choices"], [10, 14, 20])
    
    @patch('btc_research.optimization.cli_integration.DataFeed')
    def test_cli_data_handling(self, mock_datafeed_class):
        """Test CLI data loading and validation."""
        from btc_research.optimization.cli_integration import load_and_validate_data
        
        # Mock data feed
        mock_datafeed = Mock()
        mock_data = Mock()
        mock_data.empty = False
        mock_data.__len__ = Mock(return_value=1000)
        mock_datafeed.get_data.return_value = mock_data
        mock_datafeed_class.return_value = mock_datafeed
        
        config = self.config_data.copy()
        
        data = load_and_validate_data(config)
        
        self.assertIsNotNone(data)
        mock_datafeed.get_data.assert_called_once()
        
        # Test with empty data
        mock_data.empty = True
        mock_data.__len__ = Mock(return_value=0)
        
        with self.assertRaises(ValueError):
            load_and_validate_data(config)
    
    def test_progress_reporting(self):
        """Test progress reporting during optimization."""
        from btc_research.optimization.cli_integration import OptimizationProgressReporter
        
        reporter = OptimizationProgressReporter(total_iterations=10)
        
        # Test progress updates
        reporter.update(iteration=1, best_score=0.5, current_score=0.3)
        reporter.update(iteration=5, best_score=0.8, current_score=0.7)
        reporter.update(iteration=10, best_score=1.2, current_score=0.9)
        
        # Should complete without errors
        self.assertEqual(reporter.total_iterations, 10)
        self.assertEqual(reporter.current_iteration, 10)
        self.assertEqual(reporter.best_score, 1.2)
    
    def test_configuration_validation(self):
        """Test comprehensive configuration validation."""
        from btc_research.optimization.cli_integration import validate_optimization_config
        
        # Valid config
        valid_config = {
            "optimization": {
                "method": "bayesian",
                "max_iterations": 50,
                "parameters": [
                    {
                        "name": "param1",
                        "type": "float",
                        "low": 0.0,
                        "high": 1.0
                    }
                ]
            }
        }
        
        # Should pass validation
        validate_optimization_config(valid_config)
        
        # Missing optimization section
        invalid_config1 = {"backtest": {}}
        with self.assertRaises(ValueError):
            validate_optimization_config(invalid_config1)
        
        # Missing parameters
        invalid_config2 = {
            "optimization": {
                "method": "bayesian",
                "max_iterations": 50
            }
        }
        with self.assertRaises(ValueError):
            validate_optimization_config(invalid_config2)
        
        # Invalid parameter specification
        invalid_config3 = {
            "optimization": {
                "method": "bayesian",
                "max_iterations": 50,
                "parameters": [
                    {
                        "name": "param1",
                        "type": "invalid_type"
                    }
                ]
            }
        }
        with self.assertRaises(ValueError):
            validate_optimization_config(invalid_config3)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing workflows."""
    
    def test_legacy_config_support(self):
        """Test that legacy configuration formats are supported."""
        from btc_research.optimization.cli_integration import is_legacy_config, convert_legacy_config
        
        # Legacy grid search config
        legacy_config = {
            "grid_search": {
                "parameter1": [1, 2, 3],
                "parameter2": [0.1, 0.2, 0.3]
            },
            "backtest": {"cash": 10000}
        }
        
        self.assertTrue(is_legacy_config(legacy_config))
        
        # Convert to new format
        new_config = convert_legacy_config(legacy_config)
        
        self.assertIn("optimization", new_config)
        self.assertEqual(new_config["optimization"]["method"], "grid_search")
        self.assertIn("parameters", new_config["optimization"])
    
    def test_cli_argument_compatibility(self):
        """Test that old CLI arguments still work."""
        from btc_research.optimization.cli_integration import map_legacy_arguments
        
        # Legacy arguments
        legacy_args = {
            'grid_search': True,
            'param1_values': [1, 2, 3],
            'param2_range': (0.1, 0.3, 0.1),
            'metric': 'sharpe'
        }
        
        # Map to new format
        new_args = map_legacy_arguments(legacy_args)
        
        self.assertEqual(new_args['method'], 'grid_search')
        self.assertEqual(new_args['metric'], 'sharpe_ratio')  # Normalized metric name
        self.assertIn('parameters', new_args)


if __name__ == "__main__":
    unittest.main()