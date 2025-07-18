"""
Base visualizer abstract class for the visual testing framework.

This module defines the core interface and configuration structures that all
indicator visualizers must implement to integrate with the visual testing system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization rendering.
    
    This dataclass encapsulates all configuration options for controlling
    how visualizations are rendered, styled, and saved.
    
    Attributes:
        title: Chart title
        width: Figure width in inches
        height: Figure height in inches
        save_path: Optional path to save the chart
        show_signals: Whether to show trading signals
        show_indicators: Whether to show indicator overlays
        show_volume: Whether to show volume subplot
        theme: Theme name for styling
        dpi: Resolution for saved images
        format: File format for saved images
        interactive: Whether to create interactive charts
        date_range: Optional date range to display
        price_range: Optional price range to display
        annotation_size: Font size for annotations
        line_width: Default line width for indicators
        alpha: Default alpha/transparency for overlays
    """
    title: str
    width: float = 14
    height: float = 10
    save_path: Optional[Union[str, Path]] = None
    show_signals: bool = True
    show_indicators: bool = True
    show_volume: bool = True
    theme: str = "default"
    dpi: int = 300
    format: str = "png"
    interactive: bool = False
    date_range: Optional[tuple] = None
    price_range: Optional[tuple] = None
    annotation_size: int = 10
    line_width: float = 1.5
    alpha: float = 0.7
    max_data_points: int = 200


class BaseVisualizer(ABC):
    """
    Abstract base class for all indicator visualizers.
    
    This class defines the interface that all indicator visualizers must implement
    to work with the visual testing framework. It provides a consistent API for
    rendering indicators on price charts and ensures proper integration with the
    broader testing system.
    
    The BaseVisualizer follows the same pattern as BaseIndicator, providing
    abstract methods that concrete implementations must override:
    
    1. `get_required_columns()`: Returns the columns needed from the indicator
    2. `render()`: Performs the actual visualization rendering
    3. `validate_data()`: Validates input data before rendering
    
    Example:
        Creating a Volume Profile visualizer:
        
        ```python
        from btc_research.visual_testing.core.base_visualizer import BaseVisualizer
        from btc_research.visual_testing.core.chart_renderer import ChartRenderer
        
        class VolumeProfileVisualizer(BaseVisualizer):
            def get_required_columns(self) -> List[str]:
                return ["poc_price", "vah_price", "val_price", "total_volume"]
            
            def render(self, df: pd.DataFrame, config: VisualizationConfig) -> plt.Figure:
                # Create base chart
                renderer = ChartRenderer(config)
                fig, axes = renderer.create_base_chart(df)
                
                # Add volume profile overlay
                self._add_volume_profile(axes[0], df, config)
                
                return fig
            
            def validate_data(self, df: pd.DataFrame) -> None:
                required_cols = self.get_required_columns()
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
        ```
    
    Integration with indicators:
        The visualizer works with processed indicator data from the engine:
        
        ```python
        # In test scenario
        from btc_research.engine import Engine
        from btc_research.visual_testing.overlays.volume_profile_overlay import VolumeProfileVisualizer
        
        # Process data through engine
        engine = Engine(config)
        processed_data = engine.run()
        
        # Create visualization
        visualizer = VolumeProfileVisualizer()
        config = VisualizationConfig(title="Volume Profile Test")
        fig = visualizer.render(processed_data, config)
        ```
    
    Notes:
        - All visualizers should handle edge cases gracefully
        - Visualizers should be stateless and thread-safe
        - Consider performance for large datasets
        - Follow the project's theming and styling conventions
    """
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Return the column names required from the indicator data.
        
        This method defines which columns from the processed indicator data
        are needed for visualization. The framework will validate that these
        columns exist before calling render().
        
        Returns:
            List[str]: List of required column names
            
        Example:
            ```python
            def get_required_columns(self) -> List[str]:
                return [
                    "RSI_14",           # RSI values
                    "RSI_14_overbought", # Overbought signal
                    "RSI_14_oversold"    # Oversold signal
                ]
            ```
        """
        pass
    
    @abstractmethod
    def render(self, df: pd.DataFrame, config: VisualizationConfig) -> plt.Figure:
        """
        Render the indicator visualization on a price chart.
        
        This is the core method where visualization logic is implemented. It receives
        processed data containing both OHLCV data and indicator columns, and must
        return a matplotlib figure with the rendered visualization.
        
        Args:
            df: DataFrame containing OHLCV data and indicator columns
            config: Configuration object controlling rendering options
            
        Returns:
            plt.Figure: Matplotlib figure with the rendered visualization
            
        Example:
            ```python
            def render(self, df: pd.DataFrame, config: VisualizationConfig) -> plt.Figure:
                # Create base chart with candlesticks
                renderer = ChartRenderer(config)
                fig, axes = renderer.create_base_chart(df)
                
                # Add indicator-specific overlays
                self._add_rsi_subplot(fig, axes, df, config)
                self._add_signal_markers(axes[0], df, config)
                
                # Apply final styling
                renderer.apply_theme(fig, config.theme)
                
                return fig
            ```
            
        Notes:
            - Should handle missing data gracefully
            - Should respect configuration options
            - Should maintain consistent styling
            - Should be performant for large datasets
        """
        pass
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate that the input data contains required columns and is properly formatted.
        
        This method performs data validation before rendering. Override this method
        to add specific validation logic for your indicator.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If data validation fails
            
        Example:
            ```python
            def validate_data(self, df: pd.DataFrame) -> None:
                # Call parent validation
                super().validate_data(df)
                
                # Add specific validation
                if df["RSI_14"].max() > 100 or df["RSI_14"].min() < 0:
                    raise ValueError("RSI values must be between 0 and 100")
            ```
        """
        # Basic OHLCV validation
        required_base_columns = ["open", "high", "low", "close", "volume"]
        missing_base = [col for col in required_base_columns if col not in df.columns]
        if missing_base:
            raise ValueError(f"Missing required OHLCV columns: {missing_base}")
        
        # Indicator-specific validation
        required_indicator_columns = self.get_required_columns()
        missing_indicator = [col for col in required_indicator_columns if col not in df.columns]
        if missing_indicator:
            raise ValueError(f"Missing required indicator columns: {missing_indicator}")
        
        # Basic data structure validation
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
    
    def prepare_data(self, df: pd.DataFrame, config: VisualizationConfig) -> pd.DataFrame:
        """
        Prepare and filter data for visualization.
        
        This method can be overridden to add custom data preparation logic,
        such as filtering date ranges or handling missing values.
        
        Args:
            df: Input DataFrame
            config: Visualization configuration
            
        Returns:
            pd.DataFrame: Prepared DataFrame ready for visualization
        """
        # Apply date range filtering if specified
        if config.date_range:
            start_date, end_date = config.date_range
            df = df.loc[start_date:end_date]
        
        # Apply price range filtering if specified
        if config.price_range:
            min_price, max_price = config.price_range
            mask = (df["close"] >= min_price) & (df["close"] <= max_price)
            df = df[mask]
        
        # Limit data points for better visualization
        # For charts with many data points, show only the most recent data
        max_points = getattr(config, 'max_data_points', 200)
        if len(df) > max_points:
            df = df.tail(max_points)
        
        return df
    
    def save_chart(self, fig: plt.Figure, config: VisualizationConfig) -> None:
        """
        Save the chart to disk if save_path is specified.
        
        Args:
            fig: Matplotlib figure to save
            config: Configuration containing save options
        """
        if config.save_path:
            save_path = Path(config.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            fig.savefig(
                save_path,
                dpi=config.dpi,
                format=config.format,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none"
            )
    
    def create_visualization(self, df: pd.DataFrame, config: VisualizationConfig) -> plt.Figure:
        """
        Main entry point for creating a visualization.
        
        This method orchestrates the full visualization pipeline:
        1. Validates input data
        2. Prepares data for rendering
        3. Renders the visualization
        4. Saves the chart if requested
        
        Args:
            df: DataFrame containing OHLCV and indicator data
            config: Visualization configuration
            
        Returns:
            plt.Figure: Rendered matplotlib figure
        """
        # Validate input data
        self.validate_data(df)
        
        # Prepare data
        prepared_df = self.prepare_data(df, config)
        
        # Render visualization
        fig = self.render(prepared_df, config)
        
        # Save if requested
        self.save_chart(fig, config)
        
        return fig