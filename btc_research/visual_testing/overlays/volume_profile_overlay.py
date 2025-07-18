"""
Volume Profile visualizer for the visual testing framework.

This module provides visualization capabilities for Volume Profile indicators,
including volume distribution histograms, POC/VAH/VAL lines, and signal annotations.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from btc_research.visual_testing.core.base_visualizer import BaseVisualizer, VisualizationConfig
from btc_research.visual_testing.core.chart_renderer import ChartRenderer
from btc_research.visual_testing.core.theme_manager import ThemeManager


class VolumeProfileVisualizer(BaseVisualizer):
    """
    Visualizer for Volume Profile indicators.
    
    This visualizer renders Volume Profile data as histograms on the right side
    of price charts, with POC, VAH, and VAL lines overlaid on the main chart.
    It also highlights value areas and marks significant signals.
    
    Features:
        - Volume distribution histogram on right side of chart
        - Point of Control (POC) line
        - Value Area High (VAH) and Value Area Low (VAL) lines
        - Value Area highlighting
        - Signal markers for POC breakouts and volume spikes
        - Customizable colors and transparency
        - Integration with theme system
    
    Example:
        ```python
        from btc_research.visual_testing.overlays.volume_profile_overlay import VolumeProfileVisualizer
        from btc_research.visual_testing.core.base_visualizer import VisualizationConfig
        
        # Create visualizer
        visualizer = VolumeProfileVisualizer()
        
        # Configure visualization
        config = VisualizationConfig(
            title="Volume Profile Analysis",
            show_signals=True,
            theme="default"
        )
        
        # Render visualization
        fig = visualizer.create_visualization(processed_data, config)
        plt.show()
        ```
    """
    
    def get_required_columns(self) -> List[str]:
        """
        Return the Volume Profile columns required for visualization.
        
        Returns:
            List of required column names (without prefixes)
        """
        return [
            "poc_price",
            "vah_price", 
            "val_price",
            "total_volume",
            "poc_volume",
            "value_area_volume",
            "average_volume",
            "price_above_poc",
            "price_below_poc",
            "price_in_value_area",
            "poc_breakout",
            "volume_spike",
            "is_hvn",
            "is_lvn",
            "dist_to_poc",
            "dist_to_vah",
            "dist_to_val",
            "poc_strength",
            "value_area_width",
            "profile_balance"
        ]
    
    def _find_column(self, df: pd.DataFrame, base_name: str) -> Optional[str]:
        """
        Find a column in the DataFrame, handling possible prefixes.
        
        Args:
            df: DataFrame to search
            base_name: Base column name (without prefix)
            
        Returns:
            Actual column name if found, None otherwise
        """
        # First check if exact column exists
        if base_name in df.columns:
            return base_name
        
        # Check for prefixed columns (e.g., "volumeprofile_poc_price")
        for col in df.columns:
            if col.endswith(f"_{base_name}"):
                return col
                
        return None

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate Volume Profile data before visualization.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If data validation fails
        """
        # Basic OHLCV validation
        required_base_columns = ["open", "high", "low", "close", "volume"]
        missing_base = [col for col in required_base_columns if col not in df.columns]
        if missing_base:
            raise ValueError(f"Missing required OHLCV columns: {missing_base}")
        
        # Basic data structure validation
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        
        # Volume Profile specific validation with prefix handling
        vp_columns = self.get_required_columns()
        
        # Check that we can find the required columns (with or without prefixes)
        missing_columns = []
        for col in vp_columns:
            actual_col = self._find_column(df, col)
            if actual_col is None:
                missing_columns.append(col)
        
        if missing_columns:
            raise ValueError(f"Missing required Volume Profile columns: {missing_columns}")
        
        # Check for at least some non-null values in key columns
        key_columns = ["poc_price", "vah_price", "val_price"]
        for col in key_columns:
            actual_col = self._find_column(df, col)
            if actual_col and df[actual_col].notna().sum() == 0:
                raise ValueError(f"Column '{col}' contains no valid data")
        
        # Validate price relationships where data exists
        poc_col = self._find_column(df, "poc_price")
        vah_col = self._find_column(df, "vah_price")
        val_col = self._find_column(df, "val_price")
        
        if poc_col and vah_col and val_col:
            valid_mask = (df[poc_col].notna() & 
                         df[vah_col].notna() & 
                         df[val_col].notna())
            
            if valid_mask.sum() > 0:
                valid_data = df[valid_mask]
                
                # VAH should be >= POC >= VAL
                if not (valid_data[vah_col] >= valid_data[poc_col] - 1e-6).all():
                    raise ValueError("VAH price should be >= POC price")
                
                if not (valid_data[poc_col] >= valid_data[val_col] - 1e-6).all():
                    raise ValueError("POC price should be >= VAL price")
        
        # POC strength should be between 0 and 1
        strength_col = self._find_column(df, "poc_strength")
        if strength_col:
            strength_values = df[strength_col].dropna()
            if len(strength_values) > 0:
                if not (strength_values >= 0).all() or not (strength_values <= 1).all():
                    raise ValueError("POC strength should be between 0 and 1")
    
    def render(self, df: pd.DataFrame, config: VisualizationConfig) -> plt.Figure:
        """
        Render Volume Profile visualization.
        
        Args:
            df: DataFrame containing OHLCV and Volume Profile data
            config: Visualization configuration
            
        Returns:
            Matplotlib figure with Volume Profile visualization
        """
        # Create base chart
        renderer = ChartRenderer(config)
        fig, axes = renderer.create_base_chart(df)
        
        # Get theme colors
        theme_manager = ThemeManager()
        colors = theme_manager.get_color_palette(config.theme)
        
        # Main price axis
        price_ax = axes[0]
        
        # Add Volume Profile overlays
        self._add_volume_profile_levels(price_ax, df, colors, config)
        self._add_value_area_highlighting(price_ax, df, colors, config)
        
        # Add volume histogram on right side
        self._add_volume_histogram(price_ax, df, colors, config)
        
        # Add signal markers if enabled
        if config.show_signals:
            self._add_signal_markers(price_ax, df, colors, config)
        
        # Add annotations
        self._add_annotations(price_ax, df, colors, config)
        
        # Apply theme
        renderer.apply_theme(fig, config.theme)
        
        # Add legend
        price_ax.legend(loc='upper left', framealpha=0.9)
        
        return fig
    
    def _add_volume_profile_levels(self, ax: plt.Axes, df: pd.DataFrame, 
                                  colors: dict, config: VisualizationConfig) -> None:
        """
        Add POC, VAH, and VAL lines to the chart.
        
        Args:
            ax: Matplotlib axis to draw on
            df: DataFrame with Volume Profile data
            colors: Color palette
            config: Visualization configuration
        """
        # Get column names with prefixes
        poc_col = self._find_column(df, "poc_price")
        vah_col = self._find_column(df, "vah_price")
        val_col = self._find_column(df, "val_price")
        
        if not (poc_col and vah_col and val_col):
            return
        
        # Filter out NaN values
        valid_data = df.dropna(subset=[poc_col, vah_col, val_col])
        
        if len(valid_data) == 0:
            return
        
        # Get latest valid values
        latest_data = valid_data.iloc[-1]
        
        # POC line (Point of Control)
        ax.axhline(y=latest_data[poc_col], 
                  color=colors["warning"], 
                  linestyle="-", 
                  linewidth=config.line_width * 1.5,
                  alpha=config.alpha + 0.1,
                  label=f"POC: ${latest_data[poc_col]:,.0f}")
        
        # VAH line (Value Area High)
        ax.axhline(y=latest_data[vah_col], 
                  color=colors["resistance"], 
                  linestyle="--", 
                  linewidth=config.line_width,
                  alpha=config.alpha,
                  label=f"VAH: ${latest_data[vah_col]:,.0f}")
        
        # VAL line (Value Area Low)
        ax.axhline(y=latest_data[val_col], 
                  color=colors["support"], 
                  linestyle="--", 
                  linewidth=config.line_width,
                  alpha=config.alpha,
                  label=f"VAL: ${latest_data[val_col]:,.0f}")
    
    def _add_value_area_highlighting(self, ax: plt.Axes, df: pd.DataFrame,
                                   colors: dict, config: VisualizationConfig) -> None:
        """
        Add value area highlighting to the chart.
        
        Args:
            ax: Matplotlib axis to draw on
            df: DataFrame with Volume Profile data
            colors: Color palette
            config: Visualization configuration
        """
        # Get column names with prefixes
        vah_col = self._find_column(df, "vah_price")
        val_col = self._find_column(df, "val_price")
        
        if not (vah_col and val_col):
            return
        
        # Filter out NaN values
        valid_data = df.dropna(subset=[vah_col, val_col])
        
        if len(valid_data) == 0:
            return
        
        # Get latest valid values
        latest_data = valid_data.iloc[-1]
        
        # Highlight value area
        ax.axhspan(latest_data[val_col], 
                  latest_data[vah_col],
                  color=colors["highlight"],
                  alpha=0.1,
                  label="Value Area (70%)")
    
    def _add_volume_histogram(self, ax: plt.Axes, df: pd.DataFrame,
                            colors: dict, config: VisualizationConfig) -> None:
        """
        Add volume histogram on the right side of the chart.
        
        Args:
            ax: Matplotlib axis to draw on
            df: DataFrame with Volume Profile data
            colors: Color palette
            config: Visualization configuration
        """
        # This is a simplified version - in a real implementation, you'd need
        # to recreate the volume profile calculation to get the histogram data
        # For now, we'll create a representative histogram
        
        # Get price range
        price_min = df["close"].min()
        price_max = df["close"].max()
        
        # Create price bins
        n_bins = 20
        price_bins = np.linspace(price_min, price_max, n_bins)
        
        # Create histogram data (simplified)
        hist_data = []
        for i in range(len(price_bins) - 1):
            bin_min = price_bins[i]
            bin_max = price_bins[i + 1]
            
            # Find data points in this price range
            mask = (df["close"] >= bin_min) & (df["close"] < bin_max)
            volume_in_bin = df[mask]["volume"].sum()
            
            hist_data.append(volume_in_bin)
        
        # Normalize histogram data
        max_volume = max(hist_data) if hist_data else 1
        hist_data = [v / max_volume for v in hist_data]
        
        # Get current axis limits
        x_min, x_max = ax.get_xlim()
        chart_width = x_max - x_min
        
        # Position histogram on right side (10% of chart width)
        hist_width = chart_width * 0.1
        hist_x_start = x_max - hist_width
        
        # Draw histogram bars
        bin_height = (price_bins[1] - price_bins[0])
        for i, volume_ratio in enumerate(hist_data):
            if volume_ratio > 0:
                bar_width = hist_width * volume_ratio
                bar_x = hist_x_start + bar_width / 2
                bar_y = price_bins[i] + bin_height / 2
                
                # Create rectangle for histogram bar
                rect = Rectangle(
                    (hist_x_start, price_bins[i]),
                    bar_width,
                    bin_height,
                    facecolor=colors["neutral"],
                    edgecolor=colors["text"],
                    alpha=config.alpha * 0.7,
                    linewidth=0.5
                )
                ax.add_patch(rect)
    
    def _add_signal_markers(self, ax: plt.Axes, df: pd.DataFrame,
                           colors: dict, config: VisualizationConfig) -> None:
        """
        Add signal markers to the chart.
        
        Args:
            ax: Matplotlib axis to draw on
            df: DataFrame with Volume Profile data
            colors: Color palette
            config: Visualization configuration
        """
        # Get column names with prefixes
        poc_breakout_col = self._find_column(df, "poc_breakout")
        volume_spike_col = self._find_column(df, "volume_spike")
        price_above_poc_col = self._find_column(df, "price_above_poc")
        price_below_poc_col = self._find_column(df, "price_below_poc")
        
        # POC breakout signals
        if poc_breakout_col:
            breakout_signals = df[df[poc_breakout_col] == True]
            if len(breakout_signals) > 0:
                ax.scatter(breakout_signals.index, 
                          breakout_signals["close"],
                          marker="^",
                          s=100,
                          color=colors["warning"],
                          alpha=config.alpha + 0.2,
                          edgecolors="white",
                          linewidth=1,
                          label="POC Breakout",
                          zorder=5)
        
        # Volume spike signals
        if volume_spike_col:
            volume_spikes = df[df[volume_spike_col] == True]
            if len(volume_spikes) > 0:
                ax.scatter(volume_spikes.index,
                          volume_spikes["close"],
                          marker="*",
                          s=150,
                          color=colors["bullish"],
                          alpha=config.alpha + 0.2,
                          edgecolors="white",
                          linewidth=1,
                          label="Volume Spike",
                          zorder=5)
        
        # Price position relative to POC
        if price_above_poc_col:
            price_above_poc = df[df[price_above_poc_col] == True]
            # Small markers for price position (optional, can be noisy)
            if len(price_above_poc) > 0 and len(price_above_poc) < 100:  # Only show if not too many
                ax.scatter(price_above_poc.index,
                          price_above_poc["close"],
                          marker=".",
                          s=20,
                          color=colors["bullish"],
                          alpha=config.alpha * 0.5,
                          label="Above POC" if len(price_above_poc) < 50 else None)
        
        if price_below_poc_col:
            price_below_poc = df[df[price_below_poc_col] == True]
            if len(price_below_poc) > 0 and len(price_below_poc) < 100:  # Only show if not too many
                ax.scatter(price_below_poc.index,
                          price_below_poc["close"],
                          marker=".",
                          s=20,
                          color=colors["bearish"],
                          alpha=config.alpha * 0.5,
                          label="Below POC" if len(price_below_poc) < 50 else None)
    
    def _add_annotations(self, ax: plt.Axes, df: pd.DataFrame,
                        colors: dict, config: VisualizationConfig) -> None:
        """
        Add text annotations to the chart.
        
        Args:
            ax: Matplotlib axis to draw on
            df: DataFrame with Volume Profile data
            colors: Color palette
            config: Visualization configuration
        """
        # Get column names with prefixes
        poc_price_col = self._find_column(df, "poc_price")
        poc_strength_col = self._find_column(df, "poc_strength")
        value_area_width_col = self._find_column(df, "value_area_width")
        profile_balance_col = self._find_column(df, "profile_balance")
        
        # Build list of columns to check for valid data
        columns_to_check = []
        if poc_price_col:
            columns_to_check.append(poc_price_col)
        if poc_strength_col:
            columns_to_check.append(poc_strength_col)
        if value_area_width_col:
            columns_to_check.append(value_area_width_col)
        
        if not columns_to_check:
            return
        
        # Get latest valid data
        latest_data = df.dropna(subset=columns_to_check).iloc[-1:]
        
        if len(latest_data) == 0:
            return
        
        latest = latest_data.iloc[0]
        
        # Add POC strength annotation
        if poc_strength_col and pd.notna(latest[poc_strength_col]):
            strength_text = f"POC Strength: {latest[poc_strength_col]:.1%}"
            ax.text(0.02, 0.98, strength_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=colors["background"], 
                            alpha=0.8),
                   fontsize=config.annotation_size)
        
        # Add value area width annotation
        if value_area_width_col and pd.notna(latest[value_area_width_col]):
            width_text = f"VA Width: ${latest[value_area_width_col]:,.0f}"
            ax.text(0.02, 0.94, width_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=colors["background"], 
                            alpha=0.8),
                   fontsize=config.annotation_size)
        
        # Add profile balance annotation
        if profile_balance_col and pd.notna(latest[profile_balance_col]):
            balance_text = f"Balance: {latest[profile_balance_col]:.2f}"
            ax.text(0.02, 0.90, balance_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=colors["background"], 
                            alpha=0.8),
                   fontsize=config.annotation_size)
    
    def create_comparison_chart(self, df_list: List[pd.DataFrame], 
                               config_list: List[VisualizationConfig],
                               titles: List[str]) -> plt.Figure:
        """
        Create a comparison chart showing multiple Volume Profile configurations.
        
        Args:
            df_list: List of DataFrames with different VP configurations
            config_list: List of visualization configurations
            titles: List of titles for each subplot
            
        Returns:
            Matplotlib figure with comparison charts
        """
        n_charts = len(df_list)
        
        # Create subplots
        fig, axes = plt.subplots(n_charts, 1, figsize=(14, 6 * n_charts), 
                                sharex=True)
        
        # Ensure axes is always a list
        if n_charts == 1:
            axes = [axes]
        
        # Create each chart
        for i, (df, config, title) in enumerate(zip(df_list, config_list, titles)):
            # Update config title
            config.title = title
            
            # Create individual chart
            individual_fig = self.render(df, config)
            
            # Copy the chart to our subplot (simplified - in practice you'd need
            # to properly transfer the plot elements)
            axes[i].set_title(title)
            axes[i].plot(df.index, df["close"], label="Close Price")
            
            # Add VP levels if available
            poc_col = self._find_column(df, "poc_price")
            if poc_col:
                latest_poc = df[poc_col].dropna().iloc[-1] if df[poc_col].dropna().size > 0 else None
                if latest_poc:
                    axes[i].axhline(y=latest_poc, color="orange", linestyle="-", 
                                   label=f"POC: ${latest_poc:,.0f}")
            
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Close the individual figure
            plt.close(individual_fig)
        
        plt.tight_layout()
        return fig