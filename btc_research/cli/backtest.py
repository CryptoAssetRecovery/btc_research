"""CLI for running backtests."""

import argparse
import json
import sys
from pathlib import Path

import yaml

# Import indicators to register them
from btc_research.core.backtester import (
    Backtester,
    BacktesterError,
    create_backtest_summary,
)
from btc_research.core.engine import Engine, EngineError


def main() -> None:
    """Main entry point for btc-backtest command."""
    parser = argparse.ArgumentParser(description="Run strategy backtests")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    if not args.json:
        print("BTC Research Engine - Backtester")
        print(f"Config: {args.config}")
        print("=" * 50)

    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {args.config}")
            return 1

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if args.verbose and not args.json:
            print(f"Loaded configuration: {config['name']}")
            print(f"Symbol: {config['symbol']}")
            print(f"Strategy logic: {config['logic']}")
            print()

        # Step 1: Run Engine to get combined DataFrame
        if args.verbose and not args.json:
            print(
                "Step 1: Running Engine to combine multi-timeframe data and indicators..."
            )

        engine = Engine(config)
        df = engine.run()

        if args.verbose and not args.json:
            print(f"✓ Engine completed: {len(df)} rows, {len(df.columns)} columns")
            print()

        # Step 2: Run Backtester
        if args.verbose and not args.json:
            print("Step 2: Running Backtester...")

        backtester = Backtester(config, debug=args.verbose and not args.json)

        # Extract backtest parameters from config
        backtest_config = config.get("backtest", {})
        cash = backtest_config.get("cash", 10000)
        commission = backtest_config.get("commission", 0.001)
        slippage = backtest_config.get("slippage", 0.0)

        stats = backtester.run(df, cash=cash, commission=commission, slippage=slippage)

        if args.verbose and not args.json:
            print("✓ Backtester completed successfully")
            print()

        # Step 3: Output results
        if args.json:
            # Output raw JSON for programmatic use
            print(json.dumps(stats, indent=2))
        else:
            # Human-readable output
            print("Backtest Results:")
            print("-" * 30)
            summary = create_backtest_summary(stats)
            print(summary)

            if args.verbose:
                print()
                print("Additional Details:")
                print(f"  - Strategy: {config['name']}")
                print(
                    f"  - Date Range: {config['backtest']['from']} to {config['backtest']['to']}"
                )
                print(f"  - Data Points: {len(df)}")
                print(f"  - Equity Curve Points: {len(stats['equity_curve'])}")

        # Step 4: Optional plotting
        if args.plot:
            if not args.json:
                print()
                print("Generating equity curve plot...")

            try:
                from datetime import datetime

                import matplotlib.dates as mdates
                import matplotlib.pyplot as plt

                # Extract equity curve data
                equity_data = stats["equity_curve"]
                if not equity_data:
                    print("Warning: No equity curve data available for plotting")
                    return 0

                # Convert to pandas DataFrame for easier plotting
                import pandas as pd

                eq_df = pd.DataFrame(equity_data)
                eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
                eq_df.set_index("timestamp", inplace=True)

                # Create the plot
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
                )

                # Plot equity curve
                ax1.plot(
                    eq_df.index,
                    eq_df["equity"],
                    linewidth=2,
                    color="#2E86AB",
                    label="Portfolio Value",
                )
                ax1.set_title(
                    f"Backtest Results: {config['name']}",
                    fontsize=14,
                    fontweight="bold",
                )
                ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # Format x-axis for dates
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax1.xaxis.set_major_locator(mdates.MonthLocator())

                # Calculate and plot drawdown
                running_max = eq_df["equity"].expanding().max()
                drawdown = (eq_df["equity"] - running_max) / running_max * 100

                ax2.fill_between(
                    eq_df.index, drawdown, 0, alpha=0.7, color="red", label="Drawdown"
                )
                ax2.set_ylabel("Drawdown (%)", fontsize=12)
                ax2.set_xlabel("Date", fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                # Format x-axis for dates
                ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax2.xaxis.set_major_locator(mdates.MonthLocator())

                # Rotate date labels for better readability
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

                # Add summary statistics to plot
                total_return = stats.get("total_return", 0)
                max_dd = stats.get("max_drawdown", 0)
                sharpe = stats.get("sharpe_ratio", 0)
                num_trades = stats.get("num_trades", 0)

                textstr = f"""Total Return: {total_return:.2f}%
Max Drawdown: {max_dd:.2f}%
Sharpe Ratio: {sharpe:.2f}
Trades: {num_trades}"""

                props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
                ax1.text(
                    0.02,
                    0.98,
                    textstr,
                    transform=ax1.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=props,
                )

                # Adjust layout and save
                plt.tight_layout()

                # Save plot to file
                plot_filename = (
                    f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(plot_filename, dpi=150, bbox_inches="tight")

                if not args.json:
                    print(f"✓ Plot saved as: {plot_filename}")

                # Show plot if in interactive environment
                try:
                    plt.show()
                except Exception:
                    # Headless environment, plot already saved
                    pass

            except ImportError:
                if not args.json:
                    print("Warning: matplotlib not available for plotting")
            except Exception as e:
                if not args.json:
                    print(f"Warning: Failed to generate plot: {e}")
                if args.verbose and not args.json:
                    import traceback

                    traceback.print_exc()

        return 0

    except (EngineError, BacktesterError) as e:
        if args.json:
            error_result = {"error": str(e), "success": False}
            print(json.dumps(error_result, indent=2))
        else:
            print(f"Error: {e}")
        return 1

    except Exception as e:
        if args.json:
            error_result = {"error": f"Unexpected error: {e}", "success": False}
            print(json.dumps(error_result, indent=2))
        else:
            print(f"Unexpected error: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
