"""CLI for strategy parameter optimization."""

import argparse
import csv
import json
import sys
import time
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from typing import Any

import yaml

# Import indicators to register them
from btc_research.core.backtester import Backtester, BacktesterError
from btc_research.core.engine import Engine, EngineError


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

        # Return optimization result
        return {
            "parameters": params,
            "total_return": stats.get("total_return", 0),
            "max_drawdown": stats.get("max_drawdown", 0),
            "sharpe_ratio": stats.get("sharpe_ratio", 0),
            "num_trades": stats.get("num_trades", 0),
            "profit_factor": stats.get("profit_factor", 0),
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


def main() -> None:
    """Main entry point for btc-optimise command."""
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--grid", help="Grid search parameters (e.g., RSI_14.length=10,14,20)"
    )
    parser.add_argument(
        "--method",
        choices=["grid", "bayesian"],
        default="grid",
        help="Optimization method",
    )
    parser.add_argument(
        "--metric",
        choices=["total_return", "sharpe_ratio", "profit_factor"],
        default="total_return",
        help="Optimization metric",
    )
    parser.add_argument("--output", "-o", help="Output file for results (CSV or JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.verbose:
        print("BTC Research Engine - Parameter Optimizer")
        print(f"Config: {args.config}")
        print(f"Method: {args.method}")
        print("=" * 50)

    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {args.config}")
            return 1

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if args.verbose:
            print(f"Loaded configuration: {config['name']}")
            print(f"Symbol: {config['symbol']}")
            print()

        # Parse grid parameters
        if not args.grid:
            print("Error: --grid parameter is required for optimization")
            print("Example: --grid 'RSI_14.length=10,14,20;EMA_200.length=100,200,300'")
            return 1

        grid_params = parse_grid_parameters(args.grid)
        if not grid_params:
            print("Error: No valid grid parameters found")
            return 1

        if args.verbose:
            print("Grid parameters:")
            for param, values in grid_params.items():
                print(f"  {param}: {values}")
            print()

        # Generate parameter combinations
        combinations = list(generate_parameter_combinations(grid_params))
        total_combinations = len(combinations)

        if args.verbose:
            print(f"Total combinations to test: {total_combinations}")
            print()

        if args.method == "bayesian":
            print(
                "Warning: Bayesian optimization not yet implemented, using grid search"
            )

        # Run optimization
        results = []
        best_result = None
        best_metric_value = float("-inf")

        for i, params in enumerate(combinations, 1):
            if not args.verbose:
                print(f"[{i}/{total_combinations}] Testing: {params}")
            else:
                print(f"Testing combination {i}/{total_combinations}: {params}")

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

                if args.verbose:
                    print(
                        f"  ✓ {args.metric}: {metric_value:.4f} (time: {run_time:.2f}s)"
                    )
                else:
                    print(f"  ✓ {args.metric}: {metric_value:.4f}")
            else:
                if args.verbose:
                    print(f"  ✗ Failed: {result['error']}")
                else:
                    print("  ✗ Failed")

            print()

        # Summary
        successful_runs = [r for r in results if r["success"]]
        failed_runs = [r for r in results if not r["success"]]

        print("Optimization Summary:")
        print("-" * 30)
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

        # Save results to file
        if args.output:
            output_path = Path(args.output)

            if output_path.suffix.lower() == ".json":
                # Save as JSON
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\n✓ Results saved to: {output_path}")

            elif output_path.suffix.lower() == ".csv":
                # Save as CSV
                if successful_runs:
                    # Flatten parameters for CSV
                    csv_rows = []
                    for result in successful_runs:
                        row = result["parameters"].copy()
                        row.update(
                            {
                                "total_return": result["total_return"],
                                "max_drawdown": result["max_drawdown"],
                                "sharpe_ratio": result["sharpe_ratio"],
                                "num_trades": result["num_trades"],
                                "profit_factor": result.get("profit_factor", 0),
                                "run_time": result["run_time"],
                            }
                        )
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

        return 0

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


if __name__ == "__main__":
    sys.exit(main())
