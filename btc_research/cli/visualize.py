"""CLI for creating visual tests of indicators."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import yaml

# Import visual testing framework
from btc_research.visual_testing.core.base_visualizer import VisualizationConfig
from btc_research.visual_testing.core.theme_manager import ThemeManager
from btc_research.visual_testing.scenarios.test_scenarios import scenario_registry
from btc_research.visual_testing.overlays.volume_profile_overlay import VolumeProfileVisualizer

# Import indicators to register them
from btc_research.core.engine import Engine, EngineError
from btc_research.indicators import *  # Register all indicators
from tests.fixtures.sample_data import (
    create_btc_sample_data,
    create_trending_market_data,
    create_volatile_market_data,
    create_gap_data
)


def create_visualizer(indicator_type: str):
    """Create appropriate visualizer for indicator type."""
    if indicator_type.lower() == "volumeprofile":
        return VolumeProfileVisualizer()
    else:
        raise ValueError(f"No visualizer available for indicator type: {indicator_type}")


def generate_sample_data(data_type: str, periods: int = 300, **kwargs) -> "pd.DataFrame":
    """Generate sample data of specified type."""
    if data_type == "btc_sample":
        return create_btc_sample_data(periods=periods, **kwargs)
    elif data_type == "trending_bull":
        return create_trending_market_data(periods=periods, trend="bull", **kwargs)
    elif data_type == "trending_bear":
        return create_trending_market_data(periods=periods, trend="bear", **kwargs)
    elif data_type == "sideways":
        return create_trending_market_data(periods=periods, trend="sideways", **kwargs)
    elif data_type == "high_volatility":
        return create_volatile_market_data(periods=periods, volatility_level="high", **kwargs)
    elif data_type == "extreme_volatility":
        return create_volatile_market_data(periods=periods, volatility_level="extreme", **kwargs)
    elif data_type == "gap_data":
        return create_gap_data(periods=periods, **kwargs)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def run_indicator_test(config_path: str, args: argparse.Namespace) -> int:
    """Run indicator test using configuration file."""
    try:
        # Load configuration
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            return 1

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if args.verbose:
            print(f"Loaded configuration: {config['name']}")
            print(f"Symbol: {config['symbol']}")
            print(f"Indicators: {[ind['type'] for ind in config['indicators']]}")
            print()

        # Run Engine to get combined DataFrame
        engine = Engine(config)
        df = engine.run()

        if args.verbose:
            print(f"Processed {len(df)} rows of data")
            print(f"Columns: {list(df.columns)}")
            print()

        # Create visualizations for each indicator
        for indicator_config in config["indicators"]:
            indicator_type = indicator_config["type"]
            indicator_id = indicator_config["id"]
            
            if args.verbose:
                print(f"Creating visualization for {indicator_type} ({indicator_id})...")

            try:
                visualizer = create_visualizer(indicator_type)
                
                # Create visualization config
                viz_config = VisualizationConfig(
                    title=f"{indicator_type} - {config['name']}",
                    width=args.width,
                    height=args.height,
                    show_signals=args.show_signals,
                    theme=args.theme,
                    save_path=f"{indicator_id}_{config['name'].replace(' ', '_')}.{args.format}",
                    format=args.format,
                    dpi=args.dpi
                )

                # Create visualization
                fig = visualizer.create_visualization(df, viz_config)
                
                print(f"✅ Created visualization: {viz_config.save_path}")
                
                if args.show:
                    import matplotlib.pyplot as plt
                    plt.show()
                else:
                    import matplotlib.pyplot as plt
                    plt.close(fig)

            except Exception as e:
                print(f"❌ Error creating visualization for {indicator_type}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

        return 0

    except EngineError as e:
        print(f"Engine error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_scenario_test(scenario_name: str, args: argparse.Namespace) -> int:
    """Run a predefined test scenario."""
    try:
        # Get scenario from registry
        scenario = scenario_registry.get_scenario(scenario_name)
        
        if args.verbose:
            print(f"Running scenario: {scenario.name}")
            print(f"Description: {scenario.description}")
            print(f"Indicator: {scenario.get_indicator_type()}")
            print(f"Tags: {scenario.tags}")
            print()

        # Generate scenario data
        data = scenario.generate_data()
        
        if args.verbose:
            print(f"Generated {len(data)} periods of data")

        # Run indicator
        from btc_research.core.registry import get
        indicator_class = get(scenario.get_indicator_type())
        indicator = indicator_class(**{k: v for k, v in scenario.indicator_config.items() if k != "type" and k != "id"})
        
        indicator_results = indicator.compute(data)
        combined_data = data.join(indicator_results)
        
        if args.verbose:
            print(f"Indicator computed, {len(indicator_results.columns)} columns")

        # Create visualization
        visualizer = create_visualizer(scenario.get_indicator_type())
        
        viz_config = scenario.visualization_config
        viz_config.width = args.width
        viz_config.height = args.height
        viz_config.theme = args.theme
        viz_config.save_path = f"scenario_{scenario.name.replace(' ', '_').replace('-', '_')}.{args.format}"
        viz_config.format = args.format
        viz_config.dpi = args.dpi
        
        fig = visualizer.create_visualization(combined_data, viz_config)
        
        print(f"✅ Scenario visualization created: {viz_config.save_path}")
        
        # Show signals found
        signals_found = []
        for signal_col in ["poc_breakout", "volume_spike", "price_above_poc", "price_below_poc"]:
            if signal_col in indicator_results.columns:
                signal_count = indicator_results[signal_col].sum()
                if signal_count > 0:
                    signals_found.append(f"{signal_col}: {signal_count}")
        
        if signals_found:
            print(f"📈 Signals found: {', '.join(signals_found)}")
        else:
            print("📈 No signals found in this scenario")
        
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            import matplotlib.pyplot as plt
            plt.close(fig)

        return 0

    except KeyError as e:
        print(f"Scenario not found: {e}")
        return 1
    except Exception as e:
        print(f"Error running scenario: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_quick_test(indicator_type: str, args: argparse.Namespace) -> int:
    """Run a quick test with generated data."""
    try:
        if args.verbose:
            print(f"Running quick test for {indicator_type}")

        # Generate sample data
        data_kwargs = {
            "periods": args.periods
        }
        
        # Add parameters based on data type
        if args.data_type in ["btc_sample", "gap_data"]:
            data_kwargs["seed"] = args.seed
            data_kwargs["volatility"] = args.volatility
        elif args.data_type in ["trending_bull", "trending_bear", "sideways"]:
            data_kwargs["volatility"] = args.volatility
        # For high_volatility and extreme_volatility, the volatility level is set by the data type
            
        data = generate_sample_data(args.data_type, **data_kwargs)
        
        if args.verbose:
            print(f"Generated {len(data)} periods of {args.data_type} data")

        # Run indicator directly and manually add prefixes
        from btc_research.core.registry import get
        indicator_class = get(indicator_type)
        
        # Use default parameters or specified ones
        indicator_params = {}
        if args.params:
            indicator_params = json.loads(args.params)
        
        # For quick tests, use a smaller lookback period to ensure we get valid data
        if indicator_type == "VolumeProfile" and "lookback" not in indicator_params:
            # Use 1/3 of the data length or 120, whichever is smaller
            suggested_lookback = min(120, len(data) // 3)
            indicator_params["lookback"] = suggested_lookback
            if args.verbose:
                print(f"Using lookback period: {suggested_lookback} (auto-adjusted for data size)")
        
        indicator = indicator_class(**indicator_params)
        indicator_results = indicator.compute(data)
        
        # Add indicator ID prefix to match what Engine would do
        indicator_id = indicator_type.lower()
        prefixed_results = indicator_results.add_prefix(f"{indicator_id}_")
        
        combined_data = data.join(prefixed_results)
        
        if args.verbose:
            print(f"Indicator computed, {len(indicator_results.columns)} columns")

        # Create visualization
        visualizer = create_visualizer(indicator_type)
        
        viz_config = VisualizationConfig(
            title=f"{indicator_type} - Quick Test ({args.data_type})",
            width=args.width,
            height=args.height,
            show_signals=args.show_signals,
            theme=args.theme,
            save_path=f"quick_{indicator_type.lower()}_{args.data_type}.{args.format}",
            format=args.format,
            dpi=args.dpi
        )
        
        fig = visualizer.create_visualization(combined_data, viz_config)
        
        print(f"✅ Quick test visualization created: {viz_config.save_path}")
        
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            import matplotlib.pyplot as plt
            plt.close(fig)

        return 0

    except Exception as e:
        print(f"Error running quick test: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def list_scenarios() -> int:
    """List available test scenarios."""
    print("Available Test Scenarios:")
    print("=" * 50)
    
    all_scenarios = scenario_registry.list_scenarios()
    
    for scenario_name in sorted(all_scenarios):
        scenario = scenario_registry.get_scenario(scenario_name)
        print(f"📋 {scenario_name}")
        print(f"   Type: {scenario.get_indicator_type()}")
        print(f"   Tags: {', '.join(scenario.tags)}")
        print(f"   Description: {scenario.description}")
        print()
    
    # Show scenario groups
    print("Scenario Groups:")
    print("-" * 30)
    try:
        vp_scenarios = scenario_registry.get_scenarios_by_group("volume_profile")
        print(f"📂 Volume Profile: {len(vp_scenarios)} scenarios")
    except KeyError:
        print("📂 No predefined groups found")
    
    return 0


def list_themes() -> int:
    """List available themes."""
    print("Available Themes:")
    print("=" * 30)
    
    theme_manager = ThemeManager()
    themes = theme_manager.list_themes()
    
    for theme_name in themes:
        theme = theme_manager.get_theme(theme_name)
        print(f"🎨 {theme_name}")
        print(f"   Background: {theme['background_color']}")
        print(f"   Bullish: {theme['bullish_color']}")
        print(f"   Bearish: {theme['bearish_color']}")
        print()
    
    return 0


def main() -> None:
    """Main entry point for btc-visualize command."""
    parser = argparse.ArgumentParser(description="Create visual tests for indicators")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    common_parser.add_argument("--width", type=int, default=14, help="Chart width in inches")
    common_parser.add_argument("--height", type=int, default=10, help="Chart height in inches")
    common_parser.add_argument("--theme", default="default", help="Visualization theme")
    common_parser.add_argument("--format", default="png", help="Output format (png, svg, pdf)")
    common_parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    common_parser.add_argument("--show", action="store_true", help="Show plot interactively")
    common_parser.add_argument("--show-signals", action="store_true", default=True, help="Show trading signals")
    
    # Config command - visualize from config file
    config_parser = subparsers.add_parser("config", parents=[common_parser], help="Visualize from config file")
    config_parser.add_argument("config", help="Path to YAML configuration file")
    
    # Scenario command - run predefined scenario
    scenario_parser = subparsers.add_parser("scenario", parents=[common_parser], help="Run predefined scenario")
    scenario_parser.add_argument("scenario", help="Name of scenario to run")
    
    # Quick command - quick test with generated data
    quick_parser = subparsers.add_parser("quick", parents=[common_parser], help="Quick test with generated data")
    quick_parser.add_argument("indicator", help="Indicator type (e.g., VolumeProfile)")
    quick_parser.add_argument("--data-type", default="btc_sample", 
                             choices=["btc_sample", "trending_bull", "trending_bear", "sideways", 
                                     "high_volatility", "extreme_volatility", "gap_data"],
                             help="Type of data to generate")
    quick_parser.add_argument("--periods", type=int, default=150, help="Number of periods to generate")
    quick_parser.add_argument("--volatility", type=float, default=0.015, help="Volatility level")
    quick_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    quick_parser.add_argument("--params", help="JSON string of indicator parameters")
    
    # List commands
    list_parser = subparsers.add_parser("list", help="List available items")
    list_subparsers = list_parser.add_subparsers(dest="list_type", help="What to list")
    list_subparsers.add_parser("scenarios", help="List available scenarios")
    list_subparsers.add_parser("themes", help="List available themes")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle list commands
    if args.command == "list":
        if args.list_type == "scenarios":
            return list_scenarios()
        elif args.list_type == "themes":
            return list_themes()
        else:
            parser.print_help()
            return 1
    
    # Print header
    if args.verbose:
        print("BTC Research Engine - Visual Testing")
        print("=" * 50)
    
    # Execute command
    if args.command == "config":
        return run_indicator_test(args.config, args)
    elif args.command == "scenario":
        return run_scenario_test(args.scenario, args)
    elif args.command == "quick":
        return run_quick_test(args.indicator, args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())