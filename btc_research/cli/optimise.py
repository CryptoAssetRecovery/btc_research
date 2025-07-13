"""CLI for strategy parameter optimization."""

import argparse
import sys


def main() -> None:
    """Main entry point for btc-optimise command."""
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--grid", help="Grid search parameters (e.g., param1=1,2,3)")
    parser.add_argument("--method", choices=["grid", "bayesian"], default="grid", 
                       help="Optimization method")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print(f"BTC Research Engine - Parameter Optimizer")
    print(f"Config: {args.config}")
    print(f"Method: {args.method}")
    if args.grid:
        print(f"Grid parameters: {args.grid}")
    print("Note: This is a placeholder implementation for Phase 0 bootstrap.")
    print("Optimization functionality will be implemented in Phase 10.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())