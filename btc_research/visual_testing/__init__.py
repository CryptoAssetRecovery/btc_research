"""
Visual testing framework for BTC research indicators.

This package provides a comprehensive visual testing framework for validating
technical indicators by rendering them on price charts and comparing results
to known patterns and expectations.

Key components:
- BaseVisualizer: Abstract base class for all visualizers
- ChartRenderer: Core OHLCV candlestick chart rendering
- ThemeManager: Consistent styling and colors
- TestScenarios: Predefined test scenarios for validation
- Indicator Overlays: Pluggable visualizations for each indicator

Usage:
    from btc_research.visual_testing import create_visual_test
    
    # Create a Volume Profile visualization
    visualizer = create_visual_test("VolumeProfile")
    visualizer.run_scenario("trending_bull_market")
    
    # Compare different parameters
    visualizer.compare_parameters({
        "lookback": [48, 96, 144],
        "price_bins": [25, 50, 100]
    })
"""

from btc_research.visual_testing.core.base_visualizer import BaseVisualizer, VisualizationConfig
from btc_research.visual_testing.core.chart_renderer import ChartRenderer
from btc_research.visual_testing.core.theme_manager import ThemeManager
from btc_research.visual_testing.core.test_scenario import TestScenario

__all__ = [
    "BaseVisualizer",
    "VisualizationConfig", 
    "ChartRenderer",
    "ThemeManager",
    "TestScenario",
]