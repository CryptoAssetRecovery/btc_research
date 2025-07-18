"""
Theme manager for consistent styling across visualizations.

This module provides centralized theme management for the visual testing framework,
ensuring consistent colors, fonts, and styling across all visualizations.
"""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns


class ThemeManager:
    """
    Manages themes and styling for the visual testing framework.
    
    This class provides centralized theme management with predefined themes
    and the ability to create custom themes for different visualization needs.
    
    Features:
        - Multiple predefined themes (default, dark, professional, minimal)
        - Consistent color palettes for different chart elements
        - Font and sizing configurations
        - Easy theme switching
        - Custom theme creation
    
    Example:
        ```python
        from btc_research.visual_testing.core.theme_manager import ThemeManager
        
        # Create theme manager
        theme_manager = ThemeManager()
        
        # Get a theme
        dark_theme = theme_manager.get_theme("dark")
        
        # Apply theme to matplotlib
        theme_manager.apply_theme_to_matplotlib("dark")
        
        # Create custom theme
        custom_theme = {
            "background_color": "#F5F5F5",
            "text_color": "#2E2E2E",
            "bullish_color": "#00BCD4",
            "bearish_color": "#FF5722"
        }
        theme_manager.register_theme("custom", custom_theme)
        ```
    """
    
    def __init__(self):
        """Initialize the theme manager with predefined themes."""
        self._themes = self._create_default_themes()
        self._current_theme = "default"
    
    def _create_default_themes(self) -> Dict[str, Dict[str, Any]]:
        """
        Create the default theme configurations.
        
        Returns:
            Dictionary of theme configurations
        """
        return {
            "default": {
                "background_color": "#FFFFFF",
                "axis_background": "#FFFFFF",
                "text_color": "#333333",
                "grid_color": "#E0E0E0",
                "bullish_color": "#26A69A",
                "bearish_color": "#EF5350",
                "volume_bull_color": "#26A69A",
                "volume_bear_color": "#EF5350",
                "indicator_colors": ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#F44336"],
                "support_color": "#4CAF50",
                "resistance_color": "#F44336",
                "neutral_color": "#9E9E9E",
                "highlight_color": "#FFEB3B",
                "warning_color": "#FF9800",
                "font_family": "Arial",
                "font_size": 10,
                "title_font_size": 14,
                "line_width": 1.5,
                "marker_size": 50,
                "alpha": 0.7,
                "grid_alpha": 0.3,
            },
            
            "dark": {
                "background_color": "#1E1E1E",
                "axis_background": "#2D2D2D",
                "text_color": "#FFFFFF",
                "grid_color": "#404040",
                "bullish_color": "#00E676",
                "bearish_color": "#FF5252",
                "volume_bull_color": "#00E676",
                "volume_bear_color": "#FF5252",
                "indicator_colors": ["#03DAC6", "#BB86FC", "#CF6679", "#03DAC6", "#FF0266"],
                "support_color": "#00E676",
                "resistance_color": "#FF5252",
                "neutral_color": "#757575",
                "highlight_color": "#FFC107",
                "warning_color": "#FF9800",
                "font_family": "Arial",
                "font_size": 10,
                "title_font_size": 14,
                "line_width": 1.5,
                "marker_size": 50,
                "alpha": 0.8,
                "grid_alpha": 0.4,
            },
            
            "professional": {
                "background_color": "#F8F9FA",
                "axis_background": "#FFFFFF",
                "text_color": "#212529",
                "grid_color": "#DEE2E6",
                "bullish_color": "#198754",
                "bearish_color": "#DC3545",
                "volume_bull_color": "#198754",
                "volume_bear_color": "#DC3545",
                "indicator_colors": ["#0D6EFD", "#6F42C1", "#D63384", "#20C997", "#FD7E14"],
                "support_color": "#198754",
                "resistance_color": "#DC3545",
                "neutral_color": "#6C757D",
                "highlight_color": "#FFC107",
                "warning_color": "#FD7E14",
                "font_family": "Times New Roman",
                "font_size": 10,
                "title_font_size": 14,
                "line_width": 1.2,
                "marker_size": 40,
                "alpha": 0.75,
                "grid_alpha": 0.3,
            },
            
            "minimal": {
                "background_color": "#FFFFFF",
                "axis_background": "#FFFFFF",
                "text_color": "#000000",
                "grid_color": "#F0F0F0",
                "bullish_color": "#000000",
                "bearish_color": "#666666",
                "volume_bull_color": "#000000",
                "volume_bear_color": "#666666",
                "indicator_colors": ["#000000", "#333333", "#666666", "#999999", "#CCCCCC"],
                "support_color": "#000000",
                "resistance_color": "#666666",
                "neutral_color": "#999999",
                "highlight_color": "#F0F0F0",
                "warning_color": "#666666",
                "font_family": "Arial",
                "font_size": 9,
                "title_font_size": 12,
                "line_width": 1.0,
                "marker_size": 30,
                "alpha": 0.8,
                "grid_alpha": 0.2,
            },
            
            "colorblind": {
                "background_color": "#FFFFFF",
                "axis_background": "#FFFFFF",
                "text_color": "#333333",
                "grid_color": "#E0E0E0",
                "bullish_color": "#1B9E77",  # Teal
                "bearish_color": "#D95F02",  # Orange
                "volume_bull_color": "#1B9E77",
                "volume_bear_color": "#D95F02",
                "indicator_colors": ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E"],
                "support_color": "#1B9E77",
                "resistance_color": "#D95F02",
                "neutral_color": "#999999",
                "highlight_color": "#E6AB02",
                "warning_color": "#A6761D",
                "font_family": "Arial",
                "font_size": 10,
                "title_font_size": 14,
                "line_width": 1.8,  # Thicker lines for accessibility
                "marker_size": 60,   # Larger markers for accessibility
                "alpha": 0.8,
                "grid_alpha": 0.3,
            }
        }
    
    def get_theme(self, theme_name: str) -> Dict[str, Any]:
        """
        Get a theme configuration by name.
        
        Args:
            theme_name: Name of the theme to retrieve
            
        Returns:
            Theme configuration dictionary
            
        Raises:
            ValueError: If theme_name is not found
        """
        if theme_name not in self._themes:
            available_themes = list(self._themes.keys())
            raise ValueError(f"Theme '{theme_name}' not found. Available themes: {available_themes}")
        
        return self._themes[theme_name].copy()
    
    def register_theme(self, theme_name: str, theme_config: Dict[str, Any]) -> None:
        """
        Register a new custom theme.
        
        Args:
            theme_name: Name for the new theme
            theme_config: Theme configuration dictionary
        """
        # Validate theme config has required keys
        required_keys = ["background_color", "text_color", "bullish_color", "bearish_color"]
        missing_keys = [key for key in required_keys if key not in theme_config]
        if missing_keys:
            raise ValueError(f"Theme config missing required keys: {missing_keys}")
        
        # Merge with default theme to fill in missing values
        default_theme = self._themes["default"]
        complete_theme = {**default_theme, **theme_config}
        
        self._themes[theme_name] = complete_theme
    
    def list_themes(self) -> List[str]:
        """
        List all available theme names.
        
        Returns:
            List of theme names
        """
        return list(self._themes.keys())
    
    def apply_theme_to_matplotlib(self, theme_name: str) -> None:
        """
        Apply a theme to matplotlib's global settings.
        
        Args:
            theme_name: Name of theme to apply
        """
        theme = self.get_theme(theme_name)
        
        # Apply theme to matplotlib rcParams
        plt.rcParams.update({
            "figure.facecolor": theme["background_color"],
            "axes.facecolor": theme["axis_background"],
            "axes.edgecolor": theme["text_color"],
            "axes.labelcolor": theme["text_color"],
            "xtick.color": theme["text_color"],
            "ytick.color": theme["text_color"],
            "text.color": theme["text_color"],
            "font.family": theme["font_family"],
            "font.size": theme["font_size"],
            "axes.titlesize": theme["title_font_size"],
            "axes.labelsize": theme["font_size"],
            "xtick.labelsize": theme["font_size"],
            "ytick.labelsize": theme["font_size"],
            "legend.fontsize": theme["font_size"],
            "axes.grid": True,
            "grid.color": theme["grid_color"],
            "grid.alpha": theme["grid_alpha"],
            "lines.linewidth": theme["line_width"],
            "lines.markersize": theme["marker_size"] / 10,  # Convert to matplotlib scale
        })
        
        # Set current theme
        self._current_theme = theme_name
    
    def get_color_palette(self, theme_name: str = None) -> Dict[str, str]:
        """
        Get a color palette for a specific theme.
        
        Args:
            theme_name: Name of theme (uses current theme if None)
            
        Returns:
            Dictionary of color mappings
        """
        if theme_name is None:
            theme_name = self._current_theme
        
        theme = self.get_theme(theme_name)
        
        return {
            "bullish": theme["bullish_color"],
            "bearish": theme["bearish_color"],
            "volume_bull": theme["volume_bull_color"],
            "volume_bear": theme["volume_bear_color"],
            "support": theme["support_color"],
            "resistance": theme["resistance_color"],
            "neutral": theme["neutral_color"],
            "highlight": theme["highlight_color"],
            "warning": theme["warning_color"],
            "background": theme["background_color"],
            "text": theme["text_color"],
            "grid": theme["grid_color"],
        }
    
    def get_indicator_colors(self, theme_name: str = None, n_colors: int = 5) -> List[str]:
        """
        Get a list of colors for indicators.
        
        Args:
            theme_name: Name of theme (uses current theme if None)
            n_colors: Number of colors to return
            
        Returns:
            List of color strings
        """
        if theme_name is None:
            theme_name = self._current_theme
        
        theme = self.get_theme(theme_name)
        indicator_colors = theme["indicator_colors"]
        
        # Cycle through colors if more are needed
        if n_colors > len(indicator_colors):
            colors = []
            for i in range(n_colors):
                colors.append(indicator_colors[i % len(indicator_colors)])
            return colors
        
        return indicator_colors[:n_colors]
    
    def create_seaborn_palette(self, theme_name: str = None) -> sns.color_palette:
        """
        Create a seaborn color palette based on the theme.
        
        Args:
            theme_name: Name of theme (uses current theme if None)
            
        Returns:
            Seaborn color palette
        """
        if theme_name is None:
            theme_name = self._current_theme
        
        theme = self.get_theme(theme_name)
        return sns.color_palette(theme["indicator_colors"])
    
    def get_current_theme(self) -> str:
        """
        Get the name of the currently active theme.
        
        Returns:
            Current theme name
        """
        return self._current_theme
    
    def reset_to_default(self) -> None:
        """Reset matplotlib to default theme."""
        self.apply_theme_to_matplotlib("default")
    
    def create_theme_preview(self, theme_name: str, save_path: str = None) -> plt.Figure:
        """
        Create a preview chart showing the theme colors and styling.
        
        Args:
            theme_name: Name of theme to preview
            save_path: Optional path to save the preview
            
        Returns:
            Matplotlib figure with theme preview
        """
        import numpy as np
        
        # Get theme
        theme = self.get_theme(theme_name)
        colors = self.get_color_palette(theme_name)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Theme Preview: {theme_name}", fontsize=16, fontweight="bold")
        
        # Apply theme
        original_theme = self._current_theme
        self.apply_theme_to_matplotlib(theme_name)
        
        # Color swatches
        ax1 = axes[0, 0]
        color_names = list(colors.keys())
        color_values = list(colors.values())
        y_pos = np.arange(len(color_names))
        
        bars = ax1.barh(y_pos, [1] * len(color_names), color=color_values)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(color_names)
        ax1.set_xlabel("Color Swatches")
        ax1.set_title("Color Palette")
        
        # Sample candlestick-like chart
        ax2 = axes[0, 1]
        x = np.arange(10)
        opens = np.random.uniform(100, 200, 10)
        highs = opens + np.random.uniform(0, 20, 10)
        lows = opens - np.random.uniform(0, 20, 10)
        closes = opens + np.random.uniform(-10, 10, 10)
        
        # Draw simple bars to simulate candlesticks
        bull_mask = closes >= opens
        bear_mask = closes < opens
        
        ax2.bar(x[bull_mask], closes[bull_mask] - opens[bull_mask], 
                bottom=opens[bull_mask], color=colors["bullish"], alpha=0.8)
        ax2.bar(x[bear_mask], opens[bear_mask] - closes[bear_mask], 
                bottom=closes[bear_mask], color=colors["bearish"], alpha=0.8)
        
        ax2.set_title("Sample Price Chart")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        
        # Line chart with indicator colors
        ax3 = axes[1, 0]
        indicator_colors = self.get_indicator_colors(theme_name, 3)
        
        for i, color in enumerate(indicator_colors):
            y = np.sin(x + i) * 10 + 100
            ax3.plot(x, y, color=color, linewidth=theme["line_width"], 
                    label=f"Indicator {i+1}")
        
        ax3.set_title("Sample Indicators")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Value")
        ax3.legend()
        
        # Text and styling example
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.8, f"Font: {theme['font_family']}", ha="center", va="center",
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.5, 0.6, f"Font Size: {theme['font_size']}", ha="center", va="center",
                transform=ax4.transAxes, fontsize=theme["font_size"])
        ax4.text(0.5, 0.4, f"Line Width: {theme['line_width']}", ha="center", va="center",
                transform=ax4.transAxes, fontsize=theme["font_size"])
        ax4.text(0.5, 0.2, f"Alpha: {theme['alpha']}", ha="center", va="center",
                transform=ax4.transAxes, fontsize=theme["font_size"])
        
        ax4.set_title("Typography & Styling")
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_xticks([])
        ax4.set_yticks([])
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        # Restore original theme
        self.apply_theme_to_matplotlib(original_theme)
        
        return fig