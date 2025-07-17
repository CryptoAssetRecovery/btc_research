"""Enhanced CLI for strategy parameter optimization.

This module provides a comprehensive command-line interface for optimizing
trading strategy parameters using advanced algorithms, validation methods,
and robustness testing. It maintains backward compatibility with the legacy
grid search approach while offering state-of-the-art optimization techniques.

Features:
- Multiple optimization algorithms (Grid Search, Random Search, Bayesian, Genetic)
- Advanced validation strategies (Walk-Forward, Time Series Split, Purged CV)
- Robustness testing (Monte Carlo, Bootstrap, Parameter Sensitivity)
- Statistical significance testing
- Comprehensive result reporting and export
- Progress tracking for long-running optimizations
- Backward compatibility with existing configurations
"""

import argparse
import csv
import json
import sys
import time
import warnings
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')

# Import indicators to register them
from btc_research.core.backtester import Backtester, BacktesterError
from btc_research.core.engine import Engine, EngineError

# Import optimization components with fallback
try:
    from btc_research.optimization.cli_integration import (
        CLIOptimizationConfig,
        CLIResultFormatter,
        export_results,
        run_cli_optimization,
    )
    from btc_research.optimization.types import OptimizationMetric
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced optimization not available due to missing dependencies: {e}")
    print("Falling back to legacy grid search only.")
    OPTIMIZATION_AVAILABLE = False
    OptimizationMetric = None


def parse_config_parameters(config: dict[str, Any]) -> dict[str, list[Any]]:
    """
    Parse optimization parameters from config file.
    
    Looks for optimization.parameters section in config and converts
    parameter specifications to grid search format.
    """
    optimization_config = config.get("optimization", {})
    parameters = optimization_config.get("parameters", [])
    
    if not parameters:
        return {}
    
    grid_params = {}
    for param_spec in parameters:
        if not isinstance(param_spec, dict):
            continue
            
        name = param_spec.get("name")
        param_type = param_spec.get("type", "float")
        low = param_spec.get("low")
        high = param_spec.get("high")
        default = param_spec.get("default")
        
        if not name or low is None or high is None:
            continue
            
        # Generate parameter values based on type
        if param_type == "integer":
            # Generate 3-5 integer values across the range
            step = max(1, (high - low) // 4)
            values = list(range(low, high + 1, step))
            if len(values) > 5:
                values = [low, low + step, (low + high) // 2, high - step, high]
        elif param_type == "float":
            # Generate 3-5 float values across the range
            step = (high - low) / 4
            values = [low, low + step, (low + high) / 2, high - step, high]
        else:
            # For other types, use default if available
            values = [default] if default is not None else []
        
        # Remove duplicates and sort
        if param_type == "integer":
            values = sorted(list(set(values)))
        else:
            values = sorted(list(set(round(v, 6) for v in values)))
        
        grid_params[name] = values
    
    return grid_params


def parse_grid_parameters(grid_string: str) -> dict[str, list[Any]]:
    """
    Parse grid search parameter string into parameter ranges.

    Format: param1=1,2,3;param2=0.1,0.2,0.3
    """
    if not grid_string:
        return {}

    params = {}
    for param_spec in grid_string.split(";"):
        if "=" not in param_spec:
            continue

        param_name, values_str = param_spec.split("=", 1)
        param_name = param_name.strip()

        # Parse values (support int, float, string)
        values = []
        for value_str in values_str.split(","):
            value_str = value_str.strip()
            # Try to convert to number
            try:
                if "." in value_str:
                    values.append(float(value_str))
                else:
                    values.append(int(value_str))
            except ValueError:
                # Keep as string
                values.append(value_str)

        params[param_name] = values

    return params


def generate_parameter_combinations(
    grid_params: dict[str, list[Any]]
) -> Iterator[dict[str, Any]]:
    """Generate all combinations of parameters for grid search."""
    if not grid_params:
        yield {}
        return

    param_names = list(grid_params.keys())
    param_values = [grid_params[name] for name in param_names]

    for combination in product(*param_values):
        yield dict(zip(param_names, combination, strict=False))


def update_config_with_params(
    config: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Update configuration with optimization parameters."""
    config_copy = config.copy()

    # Update indicator parameters
    for indicator in config_copy.get("indicators", []):
        indicator_id = indicator["id"]
        for param_name, param_value in params.items():
            # Check if this parameter applies to this indicator
            if param_name.startswith(f"{indicator_id}."):
                # Remove indicator prefix
                actual_param = param_name[len(f"{indicator_id}.") :]
                indicator[actual_param] = param_value
            elif param_name in indicator:
                # Direct parameter name match
                indicator[param_name] = param_value

    return config_copy


def run_single_optimization(
    config: dict[str, Any], params: dict[str, Any], verbose: bool = False
) -> dict[str, Any]:
    """Run a single backtest with given parameters."""
    try:
        # Update config with parameters
        test_config = update_config_with_params(config, params)

        # Run Engine
        engine = Engine(test_config)
        df = engine.run()

        # Run Backtester
        backtester = Backtester(test_config, debug=False)
        backtest_config = test_config.get("backtest", {})

        stats = backtester.run(
            df,
            cash=backtest_config.get("cash", 10000),
            commission=backtest_config.get("commission", 0.001),
            slippage=backtest_config.get("slippage", 0.0),
        )

        # Extract metrics with proper error handling
        metrics = {}
        for metric_name in ["total_return", "max_drawdown", "sharpe_ratio", "num_trades", "profit_factor"]:
            value = stats.get(metric_name, 0)
            # Handle NaN/inf values
            if pd.isna(value) or np.isinf(value):
                if verbose:
                    print(f"  Warning: Invalid {metric_name} value: {value}")
                value = 0 if metric_name != "max_drawdown" else float('-inf')
            metrics[metric_name] = value
        
        # Return optimization result
        return {
            "parameters": params,
            **metrics,
            "success": True,
        }

    except (EngineError, BacktesterError) as e:
        return {"parameters": params, "error": str(e), "success": False}
    except Exception as e:
        return {
            "parameters": params,
            "error": f"Unexpected error: {e}",
            "success": False,
        }


def parse_list_argument(arg_string: Optional[str]) -> List[str]:
    """Parse comma-separated list argument."""
    if not arg_string:
        return []
    return [item.strip() for item in arg_string.split(",") if item.strip()]


def create_argument_parser() -> argparse.ArgumentParser:
    """Create enhanced argument parser with all optimization options."""
    parser = argparse.ArgumentParser(
        description="Advanced strategy parameter optimization with robust validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic grid search (legacy compatible) - manual parameters
  btc-optimise config.yaml --grid "RSI_14.length=10,14,20" --method grid
  
  # Use parameters from config file (no --grid needed)
  btc-optimise config.yaml --method bayesian --validation walk-forward
  
  # Bayesian optimization with manual parameters
  btc-optimise config.yaml --grid "RSI_14.length=10,30" --method bayesian --validation walk-forward
  
  # Comprehensive optimization with robustness testing (using config params)
  btc-optimise config.yaml --method genetic \
    --validation purged-cv --robustness monte-carlo,bootstrap \
    --statistics hypothesis,performance --max-iterations 100
  
  # Export results to Excel
  btc-optimise config.yaml --grid "RSI_14.length=10,30" --output results.xlsx
"""
    )
    
    # Required arguments
    parser.add_argument("config", help="Path to YAML configuration file")
    
    # Parameter specification (legacy and new)
    parser.add_argument(
        "--grid", 
        help="Grid search parameters (e.g., RSI_14.length=10,14,20;EMA_200.length=100,200). "
             "If not specified, will try to use optimization.parameters section from config file."
    )
    
    # Optimization method
    parser.add_argument(
        "--method",
        choices=["grid", "random", "bayesian", "genetic"],
        default="bayesian",
        help="Optimization algorithm (default: bayesian)",
    )
    
    # Validation strategy
    parser.add_argument(
        "--validation",
        choices=["walk-forward", "time-series-split", "purged-cv"],
        default="walk-forward",
        help="Validation strategy (default: walk-forward)",
    )
    
    # Robustness testing
    parser.add_argument(
        "--robustness",
        help="Robustness tests to run (comma-separated): monte-carlo,bootstrap,sensitivity",
    )
    
    # Statistical testing
    parser.add_argument(
        "--statistics",
        help="Statistical tests to run (comma-separated): hypothesis,performance,model-selection",
    )
    
    # Optimization parameters
    # Metric choices depend on whether optimization framework is available
    metric_choices = (
        [m.value for m in OptimizationMetric] if OPTIMIZATION_AVAILABLE 
        else ["total_return", "sharpe_ratio", "profit_factor"]
    )
    parser.add_argument(
        "--metric",
        choices=metric_choices,
        default="total_return",
        help="Primary optimization metric (default: total_return)",
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum optimization iterations (default: 100, or from YAML config)",
    )
    
    # Backtesting parameters
    parser.add_argument(
        "--cash",
        type=float,
        default=argparse.SUPPRESS,
        help="Starting capital for backtesting (default: from YAML config)",
    )
    
    parser.add_argument(
        "--commission",
        type=float,
        default=argparse.SUPPRESS,
        help="Commission rate for backtesting (default: from YAML config)",
    )
    
    parser.add_argument(
        "--slippage",
        type=float,
        default=argparse.SUPPRESS,
        help="Slippage rate for backtesting (default: from YAML config)",
    )
    
    # Output and display options
    parser.add_argument(
        "--output", "-o", 
        help="Output file for results (supports .json, .csv, .xlsx)"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Show detailed progress and intermediate results"
    )
    
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy grid search implementation for comparison",
    )
    
    # Random seed for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible results",
    )
    
    return parser


def main() -> None:
    """Enhanced main entry point for btc-optimise command."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Print header
    print("BTC Research Engine - Advanced Parameter Optimizer")
    print(f"Config: {args.config}")
    print(f"Method: {args.method}")
    print(f"Validation: {args.validation}")
    if args.robustness:
        print(f"Robustness: {args.robustness}")
    if args.statistics:
        print(f"Statistics: {args.statistics}")
    print("=" * 60)

    try:
        # Check if advanced optimization is available
        if not OPTIMIZATION_AVAILABLE or args.legacy or (args.method == "grid" and args.grid):
            if not OPTIMIZATION_AVAILABLE:
                print("\nAdvanced optimization framework not available. Using legacy grid search...")
            else:
                print("\nUsing legacy grid search implementation...")
            return run_legacy_optimization(args)
        
        # Validate input parameters for advanced optimization - support config-based params
        if not args.grid:
            # Check if config file has optimization parameters
            try:
                import yaml
                with open(args.config) as f:
                    config = yaml.safe_load(f)
                if not config.get("optimization", {}).get("parameters"):
                    print("Error: No optimization parameters found. Either specify --grid or add optimization.parameters section to config file.")
                    print("Example: --grid 'RSI_14.length=10,14,20;EMA_200.length=100,200,300'")
                    print("\nFor legacy grid search, use --legacy flag")
                    return 1
            except Exception:
                print("Error: --grid parameter is required for optimization")
                print("Example: --grid 'RSI_14.length=10,14,20;EMA_200.length=100,200,300'")
                print("\nFor legacy grid search, use --legacy flag")
                return 1
        
        # Parse robustness and statistics lists
        robustness_tests = parse_list_argument(args.robustness)
        statistical_tests = parse_list_argument(args.statistics)
        
        # Map CLI method names to internal method names
        method_mapping = {
            "grid": "grid_search",
            "random": "random_search",
            "bayesian": "bayesian",
            "genetic": "genetic_algorithm",
            "genetic_algorithm": "genetic_algorithm"
        }
        internal_method = method_mapping.get(args.method, args.method)
        
        # Map CLI validation names to internal validation names
        validation_mapping = {
            "walk-forward": "walk_forward",
            "time-series-split": "time_series_split",
            "purged-cv": "purged_cv"
        }
        internal_validation = validation_mapping.get(args.validation, args.validation)
        
        # Map CLI robustness test names to internal names
        robustness_mapping = {
            "monte-carlo": "monte_carlo",
            "bootstrap": "bootstrap",
            "sensitivity": "parameter_sensitivity",
            "parameter-sensitivity": "parameter_sensitivity"
        }
        mapped_robustness_tests = [robustness_mapping.get(test, test) for test in robustness_tests]
        
        # Map CLI statistics test names to internal names
        statistics_mapping = {
            "hypothesis": "t_test",
            "performance": "sharpe_ratio",
            "model-selection": "return_distribution",
            "t-test": "t_test",
            "sharpe-ratio": "sharpe_ratio",
            "return-distribution": "return_distribution"
        }
        mapped_statistical_tests = [statistics_mapping.get(test, test) for test in statistical_tests]
        
        # Run advanced optimization
        print("\nRunning advanced optimization...")
        results = run_cli_optimization(
            config_path=args.config,
            method=internal_method,
            validation=internal_validation,
            robustness=mapped_robustness_tests,
            statistics=mapped_statistical_tests,
            metric=args.metric,
            max_iterations=getattr(args, 'max_iterations', None),
            grid_params=args.grid,
            verbose=args.verbose,
            cash=getattr(args, 'cash', None),
            commission=getattr(args, 'commission', None),
            slippage=getattr(args, 'slippage', None),
            random_seed=args.seed,
        )
        
        # Format and display results
        formatter = CLIResultFormatter(results, verbose=args.verbose)
        formatter.print_summary()
        
        # Export results if requested
        if args.output:
            export_results(results, args.output)
        
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except KeyError as e:
        print(f"Error: Missing required configuration field: {e}")
        return 1
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML configuration: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_legacy_optimization(args) -> int:
    """Run legacy grid search optimization for backward compatibility."""
    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {args.config}")
            return 1

        with open(config_path) as f:
            config = yaml.safe_load(f)

        print(f"\nLoaded configuration: {config['name']}")
        print(f"Symbol: {config['symbol']}")

        # Parse grid parameters - support both manual specification and auto-inference
        grid_params = {}
        
        if args.grid:
            # Manual grid specification (legacy format)
            grid_params = parse_grid_parameters(args.grid)
            print("\nUsing manually specified grid parameters:")
        else:
            # Try to infer from config file
            grid_params = parse_config_parameters(config)
            if grid_params:
                print("\nUsing parameters from config file:")
            else:
                print("Error: No grid parameters found. Either specify --grid or add optimization.parameters section to config file.")
                print("\nExample --grid usage:")
                print("  --grid 'EMA_50.length=20,50,100;RSI_14.length=10,14,21'")
                print("\nExample config file optimization section:")
                print("  optimization:")
                print("    parameters:")
                print("      - name: 'EMA_50.length'")
                print("        type: 'integer'")
                print("        low: 20")
                print("        high: 100")
                return 1
        
        if not grid_params:
            print("Error: No valid grid parameters found")
            return 1

        print("\nGrid parameters:")
        for param, values in grid_params.items():
            print(f"  {param}: {values}")

        # Generate parameter combinations
        combinations = list(generate_parameter_combinations(grid_params))
        total_combinations = len(combinations)

        print(f"\nTotal combinations to test: {total_combinations}")
        print()

        # Run optimization with progress tracking
        results = []
        best_result = None
        best_metric_value = float("-inf")

        # Initialize progress bar
        with tqdm(total=total_combinations, desc="Legacy Optimization", unit="combo") as pbar:
            for i, params in enumerate(combinations, 1):
                pbar.set_description(f"Testing combination {i}/{total_combinations}")
                
                start_time = time.time()
                result = run_single_optimization(config, params, args.verbose)
                run_time = time.time() - start_time

                result["run_time"] = run_time
                results.append(result)

                if result["success"]:
                    metric_value = result.get(args.metric, 0)
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = result
                    
                    pbar.set_postfix({
                        "best": f"{best_metric_value:.4f}",
                        "current": f"{metric_value:.4f}"
                    })
                    
                    if args.verbose:
                        print(f"  ✓ {args.metric}: {metric_value:.4f}")
                else:
                    pbar.set_postfix({
                        "best": f"{best_metric_value:.4f}",
                        "status": "failed"
                    })
                    
                    if args.verbose:
                        print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                
                # Update progress bar
                pbar.update(1)

        # Summary
        successful_runs = [r for r in results if r["success"]]
        failed_runs = [r for r in results if not r["success"]]

        print("\nLegacy Optimization Summary:")
        print("-" * 40)
        print(f"Total combinations: {total_combinations}")
        print(f"Successful runs: {len(successful_runs)}")
        print(f"Failed runs: {len(failed_runs)}")

        if best_result:
            print(f"\nBest result ({args.metric}: {best_metric_value:.4f}):")
            print(f"Parameters: {best_result['parameters']}")
            print(f"Total Return: {best_result['total_return']:.2f}%")
            print(f"Max Drawdown: {best_result['max_drawdown']:.2f}%")
            print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.4f}")
            print(f"Num Trades: {best_result['num_trades']}")

        # Save results to file (legacy format)
        if args.output:
            save_legacy_results(results, successful_runs, args.output)

        return 0

    except Exception as e:
        print(f"Legacy optimization error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def save_legacy_results(results, successful_runs, output_path):
    """Save results in legacy format for backward compatibility."""
    output_path = Path(output_path)
    
    if output_path.suffix.lower() == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    elif output_path.suffix.lower() == ".csv":
        if successful_runs:
            csv_rows = []
            for result in successful_runs:
                row = result["parameters"].copy()
                row.update({
                    "total_return": result["total_return"],
                    "max_drawdown": result["max_drawdown"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "num_trades": result["num_trades"],
                    "profit_factor": result.get("profit_factor", 0),
                    "run_time": result["run_time"],
                })
                csv_rows.append(row)
            
            if csv_rows:
                with open(output_path, "w", newline="") as f:
                    fieldnames = list(csv_rows[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_rows)
                print(f"\n✓ Results saved to: {output_path}")
    
    else:
        print(f"\nWarning: Unsupported output format: {output_path.suffix}")


if __name__ == "__main__":
    sys.exit(main())
