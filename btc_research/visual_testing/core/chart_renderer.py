"""
Chart renderer for OHLCV candlestick charts.

This module provides the core chart rendering functionality for the visual testing
framework, including candlestick charts, volume subplots, and common chart elements.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from btc_research.visual_testing.core.base_visualizer import VisualizationConfig
from btc_research.visual_testing.core.theme_manager import ThemeManager


class ChartRenderer:
    """
    Core chart rendering class for OHLCV candlestick charts.
    
    This class handles the creation of professional-looking candlestick charts
    with volume subplots and provides utilities for adding indicator overlays.
    
    Features:
        - OHLCV candlestick charts with proper coloring
        - Volume subplot with color-coded bars
        - Flexible subplot layout for indicators
        - Professional styling and formatting
        - Date/time axis formatting
        - Price and volume scaling
        - Grid and annotation support
    
    Example:
        ```python
        from btc_research.visual_testing.core.chart_renderer import ChartRenderer
        from btc_research.visual_testing.core.base_visualizer import VisualizationConfig
        
        # Create configuration
        config = VisualizationConfig(
            title="BTC Price Analysis",
            width=14,
            height=10,
            theme="default"
        )
        
        # Create renderer and base chart
        renderer = ChartRenderer(config)
        fig, axes = renderer.create_base_chart(df)
        
        # Add custom overlays
        renderer.add_horizontal_line(axes[0], 45000, "Support Level", color="green")
        renderer.add_text_annotation(axes[0], df.index[-1], df["close"].iloc[-1], "Current Price")
        
        # Apply final styling
        renderer.apply_theme(fig, config.theme)
        ```
    """
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize the chart renderer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.theme_manager = ThemeManager()
        
        # Set up matplotlib parameters
        plt.rcParams["figure.figsize"] = (config.width, config.height)
        plt.rcParams["figure.dpi"] = config.dpi
        plt.rcParams["font.size"] = config.annotation_size
        
        # Color definitions
        self.colors = {
            "bullish": "#26A69A",      # Green for bullish candles
            "bearish": "#EF5350",      # Red for bearish candles
            "volume_bull": "#26A69A",  # Green for volume on up days
            "volume_bear": "#EF5350",  # Red for volume on down days
            "grid": "#E0E0E0",         # Light gray for grid
            "text": "#333333",         # Dark gray for text
            "background": "#FFFFFF",   # White background
            "border": "#CCCCCC",       # Light gray for borders
        }
    
    def create_base_chart(self, df: pd.DataFrame, 
                         volume_height_ratio: float = 0.3) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Create base chart with candlesticks and volume subplot.
        
        Args:
            df: DataFrame with OHLCV data
            volume_height_ratio: Ratio of volume subplot to price subplot height
            
        Returns:
            Tuple of (figure, [price_axis, volume_axis])
        """
        # Calculate height ratios
        price_ratio = 1.0 - volume_height_ratio
        height_ratios = [price_ratio]
        
        if self.config.show_volume:
            height_ratios.append(volume_height_ratio)
        
        # Create figure and subplots
        fig, axes = plt.subplots(
            nrows=len(height_ratios),
            ncols=1,
            figsize=(self.config.width, self.config.height),
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.1},
            sharex=True
        )
        
        # Ensure axes is always a list
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        
        # Draw candlesticks on main axis
        self._draw_candlesticks(axes[0], df)
        
        # Draw volume if enabled
        if self.config.show_volume and len(axes) > 1:
            self._draw_volume(axes[1], df)
        
        # Set up axes
        self._setup_price_axis(axes[0], df)
        if self.config.show_volume and len(axes) > 1:
            self._setup_volume_axis(axes[1], df)
        
        # Set title
        fig.suptitle(self.config.title, fontsize=14, fontweight="bold")
        
        return fig, axes
    
    def _draw_candlesticks(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """
        Draw candlestick chart on the given axis.
        
        Args:
            ax: Matplotlib axis to draw on
            df: DataFrame with OHLCV data
        """
        # Convert datetime index to numeric for plotting
        dates = mdates.date2num(df.index)
        
        # Calculate dynamic candle width based on data density
        # For good visualization, we want spacing between candles
        if len(dates) > 1:
            # Calculate average time between candles
            avg_time_diff = np.mean(np.diff(dates))
            # Use 60% of the time difference as candle width, leaving 40% for spacing
            candle_width = avg_time_diff * 0.6
        else:
            candle_width = 0.6  # Default width for single candle
        
        # Calculate candle colors
        colors = np.where(df["close"] >= df["open"], self.colors["bullish"], self.colors["bearish"])
        
        # Draw wicks (high-low lines) with thinner lines
        for i, (date, row) in enumerate(zip(dates, df.itertuples())):
            ax.plot([date, date], [row.low, row.high], color=colors[i], linewidth=0.8, alpha=0.8)
        
        # Draw candle bodies
        for i, (date, row) in enumerate(zip(dates, df.itertuples())):
            body_height = abs(row.close - row.open)
            body_bottom = min(row.open, row.close)
            
            # Create rectangle for candle body
            rect = Rectangle(
                (date - candle_width/2, body_bottom),
                candle_width,
                body_height,
                facecolor=colors[i],
                edgecolor=colors[i],
                alpha=0.9,
                linewidth=0.5
            )
            ax.add_patch(rect)
    
    def _draw_volume(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """
        Draw volume bars on the given axis.
        
        Args:
            ax: Matplotlib axis to draw on
            df: DataFrame with OHLCV data
        """
        # Convert datetime index to numeric for plotting
        dates = mdates.date2num(df.index)
        
        # Calculate dynamic bar width based on data density (same as candlesticks)
        if len(dates) > 1:
            avg_time_diff = np.mean(np.diff(dates))
            bar_width = avg_time_diff * 0.6
        else:
            bar_width = 0.6
        
        # Calculate volume colors based on price direction
        colors = np.where(df["close"] >= df["open"], 
                         self.colors["volume_bull"], 
                         self.colors["volume_bear"])
        
        # Draw volume bars with consistent width
        ax.bar(dates, df["volume"], color=colors, alpha=0.7, width=bar_width, edgecolor='none')
    
    def _setup_price_axis(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """
        Set up the price axis with proper formatting.
        
        Args:
            ax: Price axis to set up
            df: DataFrame with OHLCV data
        """
        # Set up date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df) // 10)))
        
        # Rotate date labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Set price formatting
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        
        # Add grid
        ax.grid(True, alpha=0.3, color=self.colors["grid"])
        
        # Set labels
        ax.set_ylabel("Price ($)", fontweight="bold")
        
        # Set y-axis limits with some padding
        price_min = df[["low", "close", "open"]].min().min()
        price_max = df[["high", "close", "open"]].max().max()
        price_range = price_max - price_min
        padding = price_range * 0.05
        ax.set_ylim(price_min - padding, price_max + padding)
    
    def _setup_volume_axis(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """
        Set up the volume axis with proper formatting.
        
        Args:
            ax: Volume axis to set up
            df: DataFrame with OHLCV data
        """
        # Set up date formatting (shared with price axis)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df) // 10)))
        
        # Rotate date labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Set volume formatting
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
        
        # Add grid
        ax.grid(True, alpha=0.3, color=self.colors["grid"])
        
        # Set labels
        ax.set_ylabel("Volume", fontweight="bold")
        ax.set_xlabel("Date", fontweight="bold")
        
        # Set y-axis limits
        volume_max = df["volume"].max()
        ax.set_ylim(0, volume_max * 1.1)
    
    def add_horizontal_line(self, ax: plt.Axes, y_value: float, label: str, 
                           color: str = "red", linestyle: str = "--", 
                           alpha: float = 0.7) -> None:
        """
        Add a horizontal line to the chart.
        
        Args:
            ax: Axis to add line to
            y_value: Y-coordinate for the line
            label: Label for the line
            color: Line color
            linestyle: Line style
            alpha: Line transparency
        """
        ax.axhline(y=y_value, color=color, linestyle=linestyle, alpha=alpha, label=label)
    
    def add_vertical_line(self, ax: plt.Axes, x_value: pd.Timestamp, label: str,
                         color: str = "blue", linestyle: str = "--",
                         alpha: float = 0.7) -> None:
        """
        Add a vertical line to the chart.
        
        Args:
            ax: Axis to add line to
            x_value: X-coordinate (timestamp) for the line
            label: Label for the line
            color: Line color
            linestyle: Line style
            alpha: Line transparency
        """
        ax.axvline(x=x_value, color=color, linestyle=linestyle, alpha=alpha, label=label)
    
    def add_text_annotation(self, ax: plt.Axes, x: pd.Timestamp, y: float, 
                           text: str, color: str = "black", fontsize: int = 10,
                           ha: str = "center", va: str = "center") -> None:
        """
        Add text annotation to the chart.
        
        Args:
            ax: Axis to add annotation to
            x: X-coordinate (timestamp)
            y: Y-coordinate
            text: Text to display
            color: Text color
            fontsize: Font size
            ha: Horizontal alignment
            va: Vertical alignment
        """
        ax.annotate(text, xy=(x, y), color=color, fontsize=fontsize, ha=ha, va=va,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def add_signal_markers(self, ax: plt.Axes, df: pd.DataFrame, 
                          signal_column: str, signal_type: str = "buy",
                          marker_size: int = 100) -> None:
        """
        Add signal markers to the chart.
        
        Args:
            ax: Axis to add markers to
            df: DataFrame with signal data
            signal_column: Column name containing boolean signals
            signal_type: Type of signal ("buy" or "sell")
            marker_size: Size of markers
        """
        # Filter for signal points
        signal_points = df[df[signal_column] == True]
        
        if len(signal_points) == 0:
            return
        
        # Choose marker style and color based on signal type
        if signal_type.lower() == "buy":
            marker = "^"
            color = self.colors["bullish"]
            y_offset = -0.02
        else:
            marker = "v"
            color = self.colors["bearish"]
            y_offset = 0.02
        
        # Add markers
        for timestamp, row in signal_points.iterrows():
            price = row["close"]
            # Offset markers slightly from the price
            price_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            y_position = price + (price_range * y_offset)
            
            ax.scatter(timestamp, y_position, marker=marker, s=marker_size, 
                      color=color, alpha=0.8, edgecolors="white", linewidth=1)
    
    def add_price_range_highlight(self, ax: plt.Axes, start_date: pd.Timestamp,
                                 end_date: pd.Timestamp, y_min: float, y_max: float,
                                 color: str = "yellow", alpha: float = 0.2,
                                 label: str = "") -> None:
        """
        Add highlighted price range to the chart.
        
        Args:
            ax: Axis to add highlight to
            start_date: Start date for highlight
            end_date: End date for highlight
            y_min: Minimum Y-coordinate
            y_max: Maximum Y-coordinate
            color: Highlight color
            alpha: Highlight transparency
            label: Label for the highlight
        """
        # Convert dates to numeric
        start_num = mdates.date2num(start_date)
        end_num = mdates.date2num(end_date)
        
        # Create rectangle
        rect = Rectangle(
            (start_num, y_min),
            end_num - start_num,
            y_max - y_min,
            facecolor=color,
            alpha=alpha,
            label=label
        )
        ax.add_patch(rect)
    
    def apply_theme(self, fig: plt.Figure, theme_name: str = "default") -> None:
        """
        Apply theme styling to the figure.
        
        Args:
            fig: Figure to apply theme to
            theme_name: Name of theme to apply
        """
        theme = self.theme_manager.get_theme(theme_name)
        
        # Apply theme colors
        fig.patch.set_facecolor(theme.get("background_color", self.colors["background"]))
        
        # Apply to all axes
        for ax in fig.get_axes():
            ax.set_facecolor(theme.get("axis_background", self.colors["background"]))
            ax.tick_params(colors=theme.get("text_color", self.colors["text"]))
            ax.xaxis.label.set_color(theme.get("text_color", self.colors["text"]))
            ax.yaxis.label.set_color(theme.get("text_color", self.colors["text"]))
            
            # Update grid color
            ax.grid(True, alpha=0.3, color=theme.get("grid_color", self.colors["grid"]))
    
    def create_subplot_for_indicator(self, fig: plt.Figure, 
                                   height_ratio: float = 0.3) -> plt.Axes:
        """
        Add a new subplot for an indicator.
        
        Args:
            fig: Figure to add subplot to
            height_ratio: Height ratio for the new subplot
            
        Returns:
            New axes for the indicator
        """
        # This is a simplified version - in practice, you'd need to
        # restructure the existing subplots to accommodate the new one
        # For now, we'll create a new subplot at the bottom
        
        # Get existing axes
        existing_axes = fig.get_axes()
        
        # Create new subplot (this is a simplified approach)
        new_ax = fig.add_subplot(len(existing_axes) + 1, 1, len(existing_axes) + 1)
        
        # Set up the new axis
        new_ax.grid(True, alpha=0.3, color=self.colors["grid"])
        new_ax.tick_params(colors=self.colors["text"])
        
        return new_ax