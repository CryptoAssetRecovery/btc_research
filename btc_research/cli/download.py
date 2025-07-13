"""CLI for downloading and caching market data."""

import argparse
import sys
import time
from pathlib import Path

import yaml

from btc_research.core.datafeed import DataFeed, DataFeedError


def main() -> None:
    """Main entry point for btc-download command."""
    parser = argparse.ArgumentParser(description="Download and cache market data")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if data is cached",
    )

    args = parser.parse_args()

    if not args.verbose:
        print("BTC Research Engine - Data Download")
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

        if args.verbose:
            print(f"Loaded configuration: {config['name']}")
            print(f"Symbol: {config['symbol']}")
            print(f"Exchange: {config.get('exchange', 'binanceus')}")
            print()

        # Extract data requirements from config
        symbol = config["symbol"]
        exchange = config.get("exchange", "binanceus")
        start_date = config["backtest"]["from"]
        end_date = config["backtest"]["to"]

        # Get unique timeframes from indicators
        timeframes = set()
        for indicator in config["indicators"]:
            timeframes.add(indicator["timeframe"])

        # Add entry timeframe
        entry_tf = config["timeframes"]["entry"]
        timeframes.add(entry_tf)

        if args.verbose:
            print(f"Timeframes to download: {sorted(timeframes)}")
            print(f"Date range: {start_date} to {end_date}")
            print()

        # Initialize DataFeed
        datafeed = DataFeed()

        if args.force:
            # Clear cache for this symbol if forcing re-download
            if args.verbose:
                print("Forcing re-download, clearing cache...")
            datafeed.clear_cache(symbol=symbol, source=exchange)

        # Download data for each timeframe
        total_timeframes = len(timeframes)
        downloaded_data = {}

        for i, timeframe in enumerate(sorted(timeframes), 1):
            if not args.verbose:
                print(
                    f"[{i}/{total_timeframes}] Downloading {symbol} {timeframe} data..."
                )
            else:
                print(f"Downloading {symbol} {timeframe} from {exchange}...")

            start_time = time.time()

            try:
                df = datafeed.get(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date,
                    source=exchange,
                )

                download_time = time.time() - start_time
                downloaded_data[timeframe] = df

                if args.verbose:
                    print(f"  ✓ Downloaded {len(df)} rows in {download_time:.2f}s")
                    if len(df) > 0:
                        print(f"  ✓ Date range: {df.index[0]} to {df.index[-1]}")
                        print(
                            f"  ✓ Cached to: {datafeed._get_cache_path(symbol, timeframe, exchange)}"
                        )
                else:
                    print(f"  ✓ {len(df)} rows downloaded in {download_time:.2f}s")

            except DataFeedError as e:
                print(f"  ✗ Failed to download {timeframe}: {e}")
                return 1
            except Exception as e:
                print(f"  ✗ Unexpected error downloading {timeframe}: {e}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()
                return 1

            print()

        # Summary
        total_rows = sum(len(df) for df in downloaded_data.values())
        cache_stats = datafeed.get_cache_stats()

        print("Download Summary:")
        print("-" * 30)
        print(f"Symbol: {symbol}")
        print(f"Exchange: {exchange}")
        print(f"Timeframes: {len(timeframes)}")
        print(f"Total rows: {total_rows:,}")
        print(f"Cache hits: {cache_stats['hits']}")
        print(f"Cache misses: {cache_stats['misses']}")

        if args.verbose and cache_stats["load_times"]:
            print(f"Avg cache load time: {cache_stats['avg_load_time_ms']:.1f}ms")

        print()
        print("✓ All data successfully downloaded and cached")

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
