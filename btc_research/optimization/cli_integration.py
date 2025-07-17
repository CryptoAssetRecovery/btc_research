"""
CLI Integration module for the optimization framework.

This module provides a comprehensive bridge between the command-line interface
and the optimization framework, handling configuration parsing, result formatting,
progress tracking, and error management for CLI operations.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml
from tqdm import tqdm

from btc_research.core.backtester import Backtester
from btc_research.core.engine import Engine
from btc_research.optimization.integration import (
    BacktestObjective,
    OptimizationFramework,
    optimize_strategy,
)
from btc_research.optimization.types import (
    OptimizationMetric,
    OptimizationMethod,
    ParameterSpec,
    ParameterType,
    ValidationMethod,
)

__all__ = [
    "CLIOptimizationConfig",
    "CLIResultFormatter", 
    "CLIProgressTracker",
    "parse_parameter_specs",
    "run_cli_optimization",
    "export_results",
]


class CLIOptimizationConfig:
    """
    Configuration parser and validator for CLI optimization requests.
    
    This class handles parsing command-line arguments and configuration files
    into the format expected by the optimization framework.
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        method: str = "bayesian",
        validation: str = "walk-forward",
        robustness: Optional[List[str]] = None,
        statistics: Optional[List[str]] = None,
        metric: str = "total_return",
        max_iterations: Optional[int] = None,
        grid_params: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize CLI optimization configuration.
        
        Args:
            config_path: Path to strategy configuration file
            method: Optimization method
            validation: Validation strategy
            robustness: List of robustness tests to run
            statistics: List of statistical tests to run
            metric: Primary optimization metric
            max_iterations: Maximum optimization iterations
            grid_params: Grid search parameter string (legacy compatibility)
            **kwargs: Additional optimization parameters
        """
        self.config_path = Path(config_path)
        self.grid_params = grid_params
        self.kwargs = kwargs
        
        # Load strategy configuration first
        self.strategy_config = self._load_strategy_config()
        
        # Load YAML defaults, then apply CLI parameters as overrides
        yaml_defaults = self._load_optimization_defaults()
        
        # Apply YAML defaults only when CLI parameters are not explicitly provided
        # CLI defaults from function signature
        CLI_DEFAULTS = {
            "method": "bayesian",
            "validation": "walk-forward",  # Note: CLI uses walk-forward, internal uses walk_forward
            "metric": "total_return",
            "max_iterations": 100,  # Fallback when neither CLI nor YAML provide value
        }
        
        # Use YAML default if CLI param is not explicitly provided, otherwise use CLI param
        self.method = yaml_defaults.get("method", method) if method == CLI_DEFAULTS["method"] else method
        # Handle validation name mapping (CLI uses walk-forward, internal uses walk_forward)
        cli_validation = validation if validation != "walk-forward" else "walk_forward"
        yaml_validation = yaml_defaults.get("validation", cli_validation)
        self.validation = yaml_validation if validation == CLI_DEFAULTS["validation"] else cli_validation
        
        self.robustness = yaml_defaults.get("robustness", robustness or []) if robustness is None else robustness
        self.statistics = statistics or []  # No YAML support for statistics yet
        self.metric = yaml_defaults.get("metric", metric) if metric == CLI_DEFAULTS["metric"] else metric
        
        # Handle max_iterations: None means not provided via CLI, so use YAML default or fallback
        if max_iterations is None:
            self.max_iterations = yaml_defaults.get("max_iterations", CLI_DEFAULTS["max_iterations"])
        else:
            self.max_iterations = max_iterations
        
        # Store validation config if provided in YAML
        if "validation_config" in yaml_defaults:
            self.kwargs["validation_config"] = yaml_defaults["validation_config"]
        
        # Parse optimization parameters
        self.parameter_specs = self._parse_optimization_parameters()
        
        # Validate configuration
        self._validate_config()
    
    def _load_strategy_config(self) -> Dict[str, Any]:
        """Load and validate strategy configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ["name", "symbol", "indicators"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in configuration: {field}")
        
        return config
    
    def _parse_optimization_parameters(self) -> List[ParameterSpec]:
        """Parse optimization parameters from various sources."""
        parameter_specs = []
        
        # Check for optimization section in config file
        if "optimization" in self.strategy_config:
            opt_config = self.strategy_config["optimization"]
            
            if "parameters" in opt_config:
                for param_config in opt_config["parameters"]:
                    spec = self._create_parameter_spec_from_config(param_config)
                    parameter_specs.append(spec)
        
        # Parse legacy grid parameters if provided
        if self.grid_params:
            grid_specs = self._parse_grid_parameters(self.grid_params)
            parameter_specs.extend(grid_specs)
        
        # Parse parameter specs from kwargs
        if "parameter_specs" in self.kwargs:
            parameter_specs.extend(self.kwargs["parameter_specs"])
        
        if not parameter_specs:
            raise ValueError("No optimization parameters specified. Use --grid or add optimization section to config.")
        
        return parameter_specs
    
    def _create_parameter_spec_from_config(self, param_config: Dict[str, Any]) -> ParameterSpec:
        """Create ParameterSpec from configuration dictionary."""
        name = param_config["name"]
        param_type = ParameterType(param_config["type"])
        
        spec_args = {"name": name, "param_type": param_type}
        
        if param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
            spec_args["low"] = param_config["low"]
            spec_args["high"] = param_config["high"]
            if "step" in param_config:
                spec_args["step"] = param_config["step"]
            if "log_scale" in param_config:
                spec_args["log_scale"] = param_config["log_scale"]
        
        elif param_type == ParameterType.CATEGORICAL:
            spec_args["choices"] = param_config["choices"]
        
        return ParameterSpec(**spec_args)
    
    def _parse_grid_parameters(self, grid_string: str) -> List[ParameterSpec]:
        """Parse legacy grid parameter string into ParameterSpec objects."""
        parameter_specs = []
        
        if not grid_string:
            return parameter_specs
        
        for param_spec in grid_string.split(";"):
            if "=" not in param_spec:
                continue
            
            param_name, values_str = param_spec.split("=", 1)
            param_name = param_name.strip()
            
            # Parse values
            values = []
            for value_str in values_str.split(","):
                value_str = value_str.strip()
                try:
                    if "." in value_str:
                        values.append(float(value_str))
                    else:
                        values.append(int(value_str))
                except ValueError:
                    values.append(value_str)
            
            # Determine parameter type and create spec
            if all(isinstance(v, int) for v in values):
                spec = ParameterSpec(
                    name=param_name,
                    param_type=ParameterType.INTEGER,
                    low=min(values),
                    high=max(values)
                )
            elif all(isinstance(v, (int, float)) for v in values):
                spec = ParameterSpec(
                    name=param_name,
                    param_type=ParameterType.FLOAT,
                    low=float(min(values)),
                    high=float(max(values))
                )
            else:
                spec = ParameterSpec(
                    name=param_name,
                    param_type=ParameterType.CATEGORICAL,
                    choices=values
                )
            
            parameter_specs.append(spec)
        
        return parameter_specs
    
    def _load_optimization_defaults(self) -> Dict[str, Any]:
        """Load optimization defaults from YAML configuration."""
        defaults = {}
        
        # Check for optimization defaults section in config file
        if "optimization" in self.strategy_config:
            opt_config = self.strategy_config["optimization"]
            
            if "defaults" in opt_config:
                yaml_defaults = opt_config["defaults"]
                
                # Map YAML defaults to expected parameter names
                if "method" in yaml_defaults:
                    defaults["method"] = yaml_defaults["method"]
                if "validation" in yaml_defaults:
                    defaults["validation"] = yaml_defaults["validation"]
                if "metric" in yaml_defaults:
                    defaults["metric"] = yaml_defaults["metric"]
                if "max_iterations" in yaml_defaults:
                    defaults["max_iterations"] = yaml_defaults["max_iterations"]
                if "robustness_tests" in yaml_defaults:
                    # Extract robustness test types from the list
                    robustness_tests = []
                    for test in yaml_defaults["robustness_tests"]:
                        if isinstance(test, dict) and "type" in test:
                            robustness_tests.append(test["type"])
                        elif isinstance(test, str):
                            robustness_tests.append(test)
                    defaults["robustness"] = robustness_tests
                if "validation_config" in yaml_defaults:
                    defaults["validation_config"] = yaml_defaults["validation_config"]
        
        return defaults
    
    def _validate_config(self) -> None:
        """Validate optimization configuration."""
        # Validate optimization method
        from btc_research.optimization import get_available_optimizers
        available_optimizers = get_available_optimizers()
        if self.method not in available_optimizers:
            raise ValueError(f"Unknown optimization method: {self.method}. Available: {list(available_optimizers.keys())}")
        
        # Validate validation method
        from btc_research.optimization import get_available_validators
        available_validators = get_available_validators()
        if self.validation not in available_validators:
            raise ValueError(f"Unknown validation method: {self.validation}. Available: {list(available_validators.keys())}")
        
        # Validate metric
        try:
            OptimizationMetric(self.metric)
        except ValueError:
            available_metrics = [m.value for m in OptimizationMetric]
            raise ValueError(f"Unknown metric: {self.metric}. Available: {available_metrics}")
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration dictionary."""
        return {
            "config_path": self.strategy_config,
            "parameter_specs": self.parameter_specs,
            "optimizer_type": self.method,
            "validator_type": self.validation,
            "robustness_test_type": self.robustness[0] if self.robustness else "monte_carlo",
            "metric": OptimizationMetric(self.metric),
            "max_iterations": self.max_iterations,
            **self.kwargs
        }


class CLIProgressTracker:
    """
    Progress tracker for CLI optimization operations.
    
    Provides real-time feedback during long-running optimizations including
    progress bars, timing information, and intermediate results.
    """
    
    def __init__(self, total_iterations: int, verbose: bool = False):
        """
        Initialize progress tracker.
        
        Args:
            total_iterations: Total number of optimization iterations
            verbose: Whether to show verbose output
        """
        self.total_iterations = total_iterations
        self.verbose = verbose
        self.start_time = time.time()
        self.current_iteration = 0
        self.best_score = float('-inf')
        self.best_params = {}
        self.last_update_time = time.time()
        
        # Initialize progress bar - always show for better UX
        self.pbar = tqdm(
            total=total_iterations,
            desc="Optimization",
            unit="iter",
            disable=False,  # Always show progress bar
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    def update(self, iteration: int, score: float, parameters: Dict[str, Any]) -> None:
        """
        Update progress with current iteration results.
        
        Args:
            iteration: Current iteration number
            score: Current objective function value
            parameters: Current parameter values
        """
        self.current_iteration = iteration
        current_time = time.time()
        
        # Update best result if improved
        if score > self.best_score:
            self.best_score = score
            self.best_params = parameters.copy()
            
            # Always update the progress bar with best score
            self.pbar.set_postfix({
                "best": f"{self.best_score:.4f}",
                "current": f"{score:.4f}"
            })
        
        # Update progress bar only if we haven't exceeded total
        if iteration <= self.total_iterations:
            self.pbar.update(1)
        
        # Print periodic detailed updates in verbose mode
        if self.verbose and (current_time - self.last_update_time) > 5.0:  # Every 5 seconds
            elapsed = current_time - self.start_time
            remaining = (elapsed / max(1, iteration)) * (self.total_iterations - iteration)
            print(f"\\nDetailed Progress: Iteration {iteration}/{self.total_iterations}")
            print(f"  Current score: {score:.4f}")
            print(f"  Best score: {self.best_score:.4f}")
            print(f"  Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
            print(f"  Current parameters: {parameters}")
            print(f"  Best parameters: {self.best_params}")
            self.last_update_time = current_time
    
    def finish(self) -> None:
        """Finish progress tracking."""
        self.pbar.close()
        elapsed = time.time() - self.start_time
        
        print(f"\\nOptimization completed in {elapsed:.1f}s")
        print(f"Best score: {self.best_score:.4f}")
        if self.verbose:
            print(f"Best parameters: {self.best_params}")
        
        # Show final statistics
        avg_time_per_iteration = elapsed / max(1, self.current_iteration)
        print(f"Average time per iteration: {avg_time_per_iteration:.2f}s")


class CLIResultFormatter:
    """
    Result formatter for CLI output.
    
    Handles formatting optimization results for display in the terminal
    and export to various file formats.
    """
    
    def __init__(self, results: Dict[str, Any], verbose: bool = False):
        """
        Initialize result formatter.
        
        Args:
            results: Complete optimization results
            verbose: Whether to show verbose output
        """
        self.results = results
        self.verbose = verbose
    
    def print_summary(self) -> None:
        """Print optimization summary to console."""
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        # Framework information
        framework_info = self.results.get("framework_info", {})
        print(f"Optimizer: {framework_info.get('optimizer', 'Unknown')}")
        print(f"Validator: {framework_info.get('validator', 'Unknown')}")
        print(f"Duration: {framework_info.get('total_duration', 0):.1f}s")
        
        # Optimization results
        opt_result = self.results.get("optimization")
        if opt_result:
            print(f"\nBest Parameters:")
            for param, value in opt_result.parameters.items():
                print(f"  {param}: {value}")
            
            print(f"\nObjective Value: {opt_result.objective_value:.4f}")
            print(f"Iterations: {opt_result.iterations}")
            print(f"Converged: {opt_result.convergence_achieved}")
        
        # Validation results
        validation_result = self.results.get("validation")
        if validation_result:
            print(f"\nValidation Results:")
            print(f"  Method: {validation_result.method.value}")
            print(f"  Folds: {validation_result.n_splits}")
            print(f"  Stability Score: {validation_result.stability_score:.4f}")
            print(f"  Potential Overfitting: {validation_result.is_overfitting()}")
            
            if self.verbose and validation_result.mean_metrics:
                print(f"  Cross-Validation Metrics:")
                for metric, value in validation_result.mean_metrics.items():
                    std_val = validation_result.std_metrics.get(metric, 0)
                    print(f"    {metric}: {value:.4f} ± {std_val:.4f}")
        
        # Robustness results
        robustness_result = self.results.get("robustness")
        if robustness_result:
            print(f"\nRobustness Testing:")
            print(f"  Test Type: {robustness_result.test_type}")
            print(f"  Simulations: {robustness_result.n_simulations}")
            print(f"  Success Rate: {robustness_result.success_rate:.1%}")
            
            if self.verbose:
                print(f"  Worst Case: {robustness_result.worst_case_scenario}")
                print(f"  Best Case: {robustness_result.best_case_scenario}")
        
        # Statistical results
        statistical_result = self.results.get("statistical")
        if statistical_result:
            print(f"\nStatistical Testing:")
            print(f"  Test: {statistical_result.test_name}")
            print(f"  P-value: {statistical_result.p_value:.4f}")
            print(f"  Significant: {statistical_result.is_significant()}")
            
            if self.verbose:
                print(f"  Confidence Interval: {statistical_result.confidence_interval}")
        
        print("="*60)
    
    def create_export_data(self) -> Dict[str, Any]:
        """Create data structure suitable for export."""
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "framework_info": self.results.get("framework_info", {}),
            },
            "optimization": {},
            "validation": {},
            "robustness": {},
            "statistical": {}
        }
        
        # Optimization results
        opt_result = self.results.get("optimization")
        if opt_result:
            export_data["optimization"] = {
                "parameters": opt_result.parameters,
                "objective_value": opt_result.objective_value,
                "metrics": opt_result.metrics,
                "iterations": opt_result.iterations,
                "convergence_achieved": opt_result.convergence_achieved,
                "duration_seconds": opt_result.duration_seconds,
            }
        
        # Validation results
        validation_result = self.results.get("validation")
        if validation_result:
            export_data["validation"] = {
                "method": validation_result.method.value,
                "n_splits": validation_result.n_splits,
                "stability_score": validation_result.stability_score,
                "overfitting_detected": validation_result.is_overfitting(),
                "fold_results": validation_result.fold_results,
                "mean_metrics": validation_result.mean_metrics,
                "std_metrics": validation_result.std_metrics,
                "confidence_intervals": validation_result.confidence_intervals,
            }
        
        # Robustness results
        robustness_result = self.results.get("robustness")
        if robustness_result:
            export_data["robustness"] = {
                "test_type": robustness_result.test_type,
                "n_simulations": robustness_result.n_simulations,
                "success_rate": robustness_result.success_rate,
                "summary_stats": robustness_result.summary_stats,
                "worst_case_scenario": robustness_result.worst_case_scenario,
                "best_case_scenario": robustness_result.best_case_scenario,
                "value_at_risk": robustness_result.value_at_risk,
                "expected_shortfall": robustness_result.expected_shortfall,
            }
        
        # Statistical results
        statistical_result = self.results.get("statistical")
        if statistical_result:
            export_data["statistical"] = {
                "test_name": statistical_result.test_name,
                "statistic": statistical_result.statistic,
                "p_value": statistical_result.p_value,
                "confidence_level": statistical_result.confidence_level,
                "confidence_interval": statistical_result.confidence_interval,
                "is_significant": statistical_result.is_significant(),
                "effect_size": statistical_result.effect_size,
                "power": statistical_result.power,
                "sample_size": statistical_result.sample_size,
            }
        
        return export_data


def parse_parameter_specs(grid_string: str) -> List[ParameterSpec]:
    """
    Parse parameter specifications from grid string.
    
    This function provides backward compatibility with the legacy grid
    parameter format while supporting the new ParameterSpec system.
    
    Args:
        grid_string: Grid parameter string (e.g., "param1=1,2,3;param2=0.1,0.2")
        
    Returns:
        List of ParameterSpec objects
    """
    config = CLIOptimizationConfig.__new__(CLIOptimizationConfig)
    return config._parse_grid_parameters(grid_string)


def run_cli_optimization(
    config_path: Union[str, Path],
    method: str = "bayesian",
    validation: str = "walk-forward",
    robustness: Optional[List[str]] = None,
    statistics: Optional[List[str]] = None,
    metric: str = "total_return",
    max_iterations: Optional[int] = None,
    grid_params: Optional[str] = None,
    verbose: bool = False,
    cash: Optional[float] = None,
    commission: Optional[float] = None,
    slippage: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run optimization from CLI with comprehensive configuration support.
    
    This is the main entry point for CLI-based optimization that handles
    all the complexity of configuration parsing, progress tracking, and
    result formatting.
    
    Args:
        config_path: Path to strategy configuration file
        method: Optimization method
        validation: Validation strategy  
        robustness: List of robustness tests
        statistics: List of statistical tests
        metric: Primary optimization metric
        max_iterations: Maximum optimization iterations (None means use YAML default)
        grid_params: Legacy grid parameter string
        verbose: Show verbose output
        **kwargs: Additional optimization parameters
        
    Returns:
        Complete optimization results dictionary
    """
    # Parse configuration
    cli_config = CLIOptimizationConfig(
        config_path=config_path,
        method=method,
        validation=validation,
        robustness=robustness,
        statistics=statistics,
        metric=metric,
        max_iterations=max_iterations,
        grid_params=grid_params,
        **kwargs
    )
    
    # Use YAML config values for transaction costs if CLI values not provided
    backtest_config = cli_config.strategy_config.get("backtest", {})
    if cash is None:
        cash = backtest_config.get("cash", 10000.0)
    if commission is None:
        commission = backtest_config.get("commission", 0.001)
    if slippage is None:
        slippage = backtest_config.get("slippage", 0.0)
    
    if verbose:
        print(f"Using transaction costs - Cash: {cash}, Commission: {commission}, Slippage: {slippage}")
    
    if verbose:
        print(f"Loaded configuration: {cli_config.strategy_config['name']}")
        print(f"Symbol: {cli_config.strategy_config['symbol']}")
        print(f"Optimization method: {method}")
        print(f"Validation strategy: {validation}")
        print(f"Parameter specifications: {len(cli_config.parameter_specs)}")
        print()
    
    # Adjust expected iterations for genetic algorithm
    # Genetic algorithm runs in generations with population size evaluations each
    if method == "genetic_algorithm":
        # Default population size is 50, so actual iterations will be generations * population_size
        population_size = kwargs.get('population_size', 50)
        expected_generations = cli_config.max_iterations // population_size
        actual_max_iterations = expected_generations * population_size
    else:
        actual_max_iterations = cli_config.max_iterations
    
    # Set up progress tracking
    progress_tracker = CLIProgressTracker(actual_max_iterations, verbose)
    
    try:
        # Run optimization using the integration layer
        opt_config = cli_config.get_optimization_config()
        
        # Override transaction costs with corrected values
        opt_config["cash"] = cash
        opt_config["commission"] = commission  
        opt_config["slippage"] = slippage
        
        # Pass progress tracker to the optimization function
        opt_config["progress_tracker"] = progress_tracker
        
        results = optimize_strategy(**opt_config)
        
        progress_tracker.finish()
        
        return results
        
    except Exception as e:
        progress_tracker.finish()
        raise RuntimeError(f"Optimization failed: {e}") from e


def export_results(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    format_type: Optional[str] = None
) -> None:
    """
    Export optimization results to file.
    
    Args:
        results: Complete optimization results
        output_path: Output file path
        format_type: Export format ('json', 'csv', 'excel') - auto-detected if None
    """
    output_path = Path(output_path)
    
    # Auto-detect format from extension
    if format_type is None:
        format_type = output_path.suffix.lower().lstrip('.')
    
    # Create formatter and export data
    formatter = CLIResultFormatter(results)
    export_data = formatter.create_export_data()
    
    if format_type == "json":
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    elif format_type == "csv":
        # Flatten optimization results for CSV
        opt_data = export_data.get("optimization", {})
        validation_data = export_data.get("validation", {})
        
        csv_data = {}
        csv_data.update(opt_data.get("parameters", {}))
        csv_data.update(opt_data.get("metrics", {}))
        csv_data["objective_value"] = opt_data.get("objective_value", 0)
        csv_data["iterations"] = opt_data.get("iterations", 0)
        csv_data["convergence_achieved"] = opt_data.get("convergence_achieved", False)
        csv_data["stability_score"] = validation_data.get("stability_score", 0)
        csv_data["overfitting_detected"] = validation_data.get("overfitting_detected", False)
        
        df = pd.DataFrame([csv_data])
        df.to_csv(output_path, index=False)
    
    elif format_type in ("xlsx", "excel"):
        # Create multi-sheet Excel file
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Optimization sheet
            opt_data = export_data.get("optimization", {})
            if opt_data:
                opt_df = pd.DataFrame([opt_data])
                opt_df.to_excel(writer, sheet_name="Optimization", index=False)
            
            # Validation sheet
            validation_data = export_data.get("validation", {})
            if validation_data and "fold_results" in validation_data:
                val_df = pd.DataFrame(validation_data["fold_results"])
                val_df.to_excel(writer, sheet_name="Validation", index=False)
            
            # Robustness sheet
            robustness_data = export_data.get("robustness", {})
            if robustness_data and "summary_stats" in robustness_data:
                rob_df = pd.DataFrame(robustness_data["summary_stats"])
                rob_df.to_excel(writer, sheet_name="Robustness", index=False)
    
    else:
        raise ValueError(f"Unsupported export format: {format_type}")
    
    print(f"Results exported to: {output_path}")