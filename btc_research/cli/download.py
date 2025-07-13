"""CLI for downloading and caching market data."""

import argparse
import sys


def main() -> None:
    """Main entry point for btc-download command."""
    parser = argparse.ArgumentParser(description="Download and cache market data")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print(f"BTC Research Engine - Data Download")
    print(f"Config: {args.config}")
    print("Note: This is a placeholder implementation for Phase 0 bootstrap.")
    print("Data download functionality will be implemented in Phase 3.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())