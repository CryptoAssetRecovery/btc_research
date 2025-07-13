"""CLI for running backtests."""

import argparse
import sys


def main() -> None:
    """Main entry point for btc-backtest command."""
    parser = argparse.ArgumentParser(description="Run strategy backtests")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print(f"BTC Research Engine - Backtester")
    print(f"Config: {args.config}")
    print(f"Plot: {args.plot}")
    print("Note: This is a placeholder implementation for Phase 0 bootstrap.")
    print("Backtest functionality will be implemented in Phase 7.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())