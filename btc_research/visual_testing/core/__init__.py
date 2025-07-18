"""
Core components for the visual testing framework.

This module contains the foundational classes and utilities that power
the visual testing framework.
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