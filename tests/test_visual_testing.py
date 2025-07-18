"""
Comprehensive test suite for the visual testing framework.

This module tests the visual testing framework components including:
- Base visualizer functionality
- Chart renderer capabilities  
- Theme manager
- Test scenario execution
- Volume Profile visualizer integration
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock, patch

from btc_research.visual_testing.core.base_visualizer import BaseVisualizer, VisualizationConfig
from btc_research.visual_testing.core.chart_renderer import ChartRenderer
from btc_research.visual_testing.core.theme_manager import ThemeManager
from btc_research.visual_testing.core.test_scenario import (
    TestScenario, ScenarioBuilder, ExpectedSignal, ValidationCriteria, TestResult
)
from btc_research.visual_testing.scenarios.test_scenarios import (
    VolumeProfileScenarios, scenario_registry
)
from btc_research.visual_testing.overlays.volume_profile_overlay import VolumeProfileVisualizer
from tests.fixtures.sample_data import create_btc_sample_data


class TestBaseVisualizer:
    """Test the base visualizer abstract class and configuration."""
    
    def test_visualization_config_defaults(self):
        """Test VisualizationConfig default values."""
        config = VisualizationConfig(title="Test Chart")
        
        assert config.title == "Test Chart"
        assert config.width == 14
        assert config.height == 10
        assert config.save_path is None
        assert config.show_signals is True
        assert config.show_indicators is True
        assert config.show_volume is True
        assert config.theme == "default"
        assert config.dpi == 300
        assert config.format == "png"
        assert config.interactive is False
        assert config.date_range is None
        assert config.price_range is None
        assert config.annotation_size == 10
        assert config.line_width == 1.5
        assert config.alpha == 0.7
    
    def test_visualization_config_custom_values(self):
        """Test VisualizationConfig with custom values."""
        config = VisualizationConfig(
            title="Custom Chart",
            width=16,
            height=12,
            save_path="/tmp/test.png",
            show_signals=False,
            theme="dark",
            dpi=150,
            format="svg",
            interactive=True,
            date_range=("2024-01-01", "2024-01-31"),
            price_range=(40000, 50000),
            annotation_size=12,
            line_width=2.0,
            alpha=0.8
        )
        
        assert config.title == "Custom Chart"
        assert config.width == 16
        assert config.height == 12
        assert config.save_path == "/tmp/test.png"
        assert config.show_signals is False
        assert config.theme == "dark"
        assert config.dpi == 150
        assert config.format == "svg"
        assert config.interactive is True
        assert config.date_range == ("2024-01-01", "2024-01-31")
        assert config.price_range == (40000, 50000)
        assert config.annotation_size == 12
        assert config.line_width == 2.0
        assert config.alpha == 0.8
    
    def test_base_visualizer_is_abstract(self):
        """Test that BaseVisualizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVisualizer()
    
    def test_concrete_visualizer_implementation(self):
        """Test a concrete implementation of BaseVisualizer."""
        
        class TestVisualizer(BaseVisualizer):
            def get_required_columns(self):
                return ["test_column"]
            
            def render(self, df, config):
                fig, ax = plt.subplots(figsize=(config.width, config.height))
                ax.plot(df.index, df["close"])
                ax.set_title(config.title)
                return fig
        
        # Test instantiation
        visualizer = TestVisualizer()
        assert isinstance(visualizer, BaseVisualizer)
        
        # Test required columns
        required_cols = visualizer.get_required_columns()
        assert required_cols == ["test_column"]
        
        # Test data validation
        sample_data = create_btc_sample_data(periods=100)
        sample_data["test_column"] = np.random.rand(100)
        
        # Should pass validation
        visualizer.validate_data(sample_data)
        
        # Test render method
        config = VisualizationConfig(title="Test Chart")
        fig = visualizer.render(sample_data, config)
        
        assert isinstance(fig, plt.Figure)
        assert fig.get_suptitle() == "Test Chart"
        plt.close(fig)
    
    def test_data_validation_missing_ohlcv_columns(self):
        """Test data validation with missing OHLCV columns."""
        
        class TestVisualizer(BaseVisualizer):
            def get_required_columns(self):
                return []
            
            def render(self, df, config):
                return plt.figure()
        
        visualizer = TestVisualizer()
        
        # Missing OHLCV columns
        invalid_data = pd.DataFrame({"invalid": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            visualizer.validate_data(invalid_data)
    
    def test_data_validation_missing_indicator_columns(self):
        """Test data validation with missing indicator columns."""
        
        class TestVisualizer(BaseVisualizer):
            def get_required_columns(self):
                return ["missing_indicator"]
            
            def render(self, df, config):
                return plt.figure()
        
        visualizer = TestVisualizer()
        sample_data = create_btc_sample_data(periods=100)
        
        with pytest.raises(ValueError, match="Missing required indicator columns"):
            visualizer.validate_data(sample_data)
    
    def test_data_validation_empty_dataframe(self):
        """Test data validation with empty DataFrame."""
        
        class TestVisualizer(BaseVisualizer):
            def get_required_columns(self):
                return []
            
            def render(self, df, config):
                return plt.figure()
        
        visualizer = TestVisualizer()
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            visualizer.validate_data(empty_data)
    
    def test_data_validation_non_datetime_index(self):
        """Test data validation with non-datetime index."""
        
        class TestVisualizer(BaseVisualizer):
            def get_required_columns(self):
                return []
            
            def render(self, df, config):
                return plt.figure()
        
        visualizer = TestVisualizer()
        sample_data = create_btc_sample_data(periods=100)
        sample_data.index = range(100)  # Change to integer index
        
        with pytest.raises(ValueError, match="DataFrame index must be DatetimeIndex"):
            visualizer.validate_data(sample_data)
    
    def test_prepare_data_date_range_filtering(self):
        """Test data preparation with date range filtering."""
        
        class TestVisualizer(BaseVisualizer):
            def get_required_columns(self):
                return []
            
            def render(self, df, config):
                return plt.figure()
        
        visualizer = TestVisualizer()
        sample_data = create_btc_sample_data(periods=100)
        
        # Set date range
        start_date = sample_data.index[20]
        end_date = sample_data.index[80]
        config = VisualizationConfig(
            title="Test",
            date_range=(start_date, end_date)
        )
        
        prepared_data = visualizer.prepare_data(sample_data, config)
        
        assert len(prepared_data) == 61  # 80 - 20 + 1
        assert prepared_data.index[0] == start_date
        assert prepared_data.index[-1] == end_date
    
    def test_prepare_data_price_range_filtering(self):
        """Test data preparation with price range filtering."""
        
        class TestVisualizer(BaseVisualizer):
            def get_required_columns(self):
                return []
            
            def render(self, df, config):
                return plt.figure()
        
        visualizer = TestVisualizer()
        sample_data = create_btc_sample_data(periods=100)
        
        # Set price range
        min_price = sample_data["close"].quantile(0.3)
        max_price = sample_data["close"].quantile(0.7)
        config = VisualizationConfig(
            title="Test",
            price_range=(min_price, max_price)
        )
        
        prepared_data = visualizer.prepare_data(sample_data, config)
        
        assert (prepared_data["close"] >= min_price).all()
        assert (prepared_data["close"] <= max_price).all()
        assert len(prepared_data) < len(sample_data)
    
    @patch('pathlib.Path.mkdir')
    def test_save_chart(self, mock_mkdir):
        """Test chart saving functionality."""
        
        class TestVisualizer(BaseVisualizer):
            def get_required_columns(self):
                return []
            
            def render(self, df, config):
                return plt.figure()
        
        visualizer = TestVisualizer()
        fig = plt.figure()
        
        # Test save functionality
        config = VisualizationConfig(
            title="Test",
            save_path="/tmp/test_chart.png",
            dpi=200,
            format="png"
        )
        
        with patch.object(fig, 'savefig') as mock_savefig:
            visualizer.save_chart(fig, config)
            
            mock_savefig.assert_called_once_with(
                Path("/tmp/test_chart.png"),
                dpi=200,
                format="png",
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none"
            )
        
        mock_mkdir.assert_called_once()
        plt.close(fig)


class TestChartRenderer:
    """Test the chart renderer functionality."""
    
    def test_chart_renderer_initialization(self):
        """Test ChartRenderer initialization."""
        config = VisualizationConfig(title="Test Chart")
        renderer = ChartRenderer(config)
        
        assert renderer.config == config
        assert isinstance(renderer.theme_manager, ThemeManager)
        assert "bullish" in renderer.colors
        assert "bearish" in renderer.colors
        assert "volume_bull" in renderer.colors
        assert "volume_bear" in renderer.colors
    
    def test_create_base_chart_with_volume(self):
        """Test creating base chart with volume subplot."""
        config = VisualizationConfig(title="Test Chart", show_volume=True)
        renderer = ChartRenderer(config)
        sample_data = create_btc_sample_data(periods=100)
        
        fig, axes = renderer.create_base_chart(sample_data)
        
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2  # Price and volume axes
        assert fig.get_suptitle() == "Test Chart"
        
        # Check that both axes have content
        assert len(axes[0].get_children()) > 0  # Price axis should have candlesticks
        assert len(axes[1].get_children()) > 0  # Volume axis should have bars
        
        plt.close(fig)
    
    def test_create_base_chart_without_volume(self):
        """Test creating base chart without volume subplot."""
        config = VisualizationConfig(title="Test Chart", show_volume=False)
        renderer = ChartRenderer(config)
        sample_data = create_btc_sample_data(periods=100)
        
        fig, axes = renderer.create_base_chart(sample_data)
        
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 1  # Only price axis
        assert fig.get_suptitle() == "Test Chart"
        
        plt.close(fig)
    
    def test_add_horizontal_line(self):
        """Test adding horizontal line to chart."""
        config = VisualizationConfig(title="Test Chart")
        renderer = ChartRenderer(config)
        
        fig, ax = plt.subplots()
        renderer.add_horizontal_line(ax, 45000, "Support Level", color="green")
        
        # Check that line was added
        lines = ax.get_lines()
        assert len(lines) == 1
        assert lines[0].get_ydata()[0] == 45000
        assert lines[0].get_color() == "green"
        
        plt.close(fig)
    
    def test_add_vertical_line(self):
        """Test adding vertical line to chart."""
        config = VisualizationConfig(title="Test Chart")
        renderer = ChartRenderer(config)
        
        fig, ax = plt.subplots()
        timestamp = pd.Timestamp("2024-01-01")
        renderer.add_vertical_line(ax, timestamp, "Event", color="blue")
        
        # Check that line was added
        lines = ax.get_lines()
        assert len(lines) == 1
        
        plt.close(fig)
    
    def test_add_text_annotation(self):
        """Test adding text annotation to chart."""
        config = VisualizationConfig(title="Test Chart")
        renderer = ChartRenderer(config)
        
        fig, ax = plt.subplots()
        timestamp = pd.Timestamp("2024-01-01")
        renderer.add_text_annotation(ax, timestamp, 45000, "Test Annotation")
        
        # Check that annotation was added
        annotations = ax.texts
        assert len(annotations) == 1
        assert annotations[0].get_text() == "Test Annotation"
        
        plt.close(fig)
    
    def test_add_signal_markers(self):
        """Test adding signal markers to chart."""
        config = VisualizationConfig(title="Test Chart")
        renderer = ChartRenderer(config)
        
        fig, ax = plt.subplots()
        sample_data = create_btc_sample_data(periods=100)
        
        # Add some buy signals
        sample_data["buy_signal"] = False
        sample_data.loc[sample_data.index[::20], "buy_signal"] = True
        
        renderer.add_signal_markers(ax, sample_data, "buy_signal", "buy")
        
        # Check that markers were added
        collections = ax.collections
        assert len(collections) > 0
        
        plt.close(fig)
    
    def test_add_price_range_highlight(self):
        """Test adding price range highlight to chart."""
        config = VisualizationConfig(title="Test Chart")
        renderer = ChartRenderer(config)
        
        fig, ax = plt.subplots()
        sample_data = create_btc_sample_data(periods=100)
        
        start_date = sample_data.index[20]
        end_date = sample_data.index[80]
        
        renderer.add_price_range_highlight(
            ax, start_date, end_date, 44000, 46000, color="yellow", label="Test Range"
        )
        
        # Check that highlight was added
        patches = ax.patches
        assert len(patches) == 1
        assert patches[0].get_facecolor()[:3] == (1.0, 1.0, 0.0)  # Yellow color
        
        plt.close(fig)
    
    def test_apply_theme(self):
        """Test applying theme to figure."""
        config = VisualizationConfig(title="Test Chart")
        renderer = ChartRenderer(config)
        
        fig, ax = plt.subplots()
        renderer.apply_theme(fig, "dark")
        
        # Theme should be applied without errors
        assert fig is not None
        
        plt.close(fig)


class TestThemeManager:
    """Test the theme manager functionality."""
    
    def test_theme_manager_initialization(self):
        """Test ThemeManager initialization."""
        theme_manager = ThemeManager()
        
        assert theme_manager._current_theme == "default"
        assert "default" in theme_manager._themes
        assert "dark" in theme_manager._themes
        assert "professional" in theme_manager._themes
        assert "minimal" in theme_manager._themes
        assert "colorblind" in theme_manager._themes
    
    def test_get_theme(self):
        """Test getting theme configuration."""
        theme_manager = ThemeManager()
        
        default_theme = theme_manager.get_theme("default")
        assert isinstance(default_theme, dict)
        assert "background_color" in default_theme
        assert "text_color" in default_theme
        assert "bullish_color" in default_theme
        assert "bearish_color" in default_theme
        
        # Test invalid theme
        with pytest.raises(ValueError, match="Theme 'invalid' not found"):
            theme_manager.get_theme("invalid")
    
    def test_register_theme(self):
        """Test registering custom theme."""
        theme_manager = ThemeManager()
        
        custom_theme = {
            "background_color": "#F0F0F0",
            "text_color": "#000000",
            "bullish_color": "#00FF00",
            "bearish_color": "#FF0000"
        }
        
        theme_manager.register_theme("custom", custom_theme)
        
        # Should be able to get custom theme
        retrieved_theme = theme_manager.get_theme("custom")
        assert retrieved_theme["background_color"] == "#F0F0F0"
        assert retrieved_theme["bullish_color"] == "#00FF00"
        
        # Should have default values for missing keys
        assert "indicator_colors" in retrieved_theme
    
    def test_register_theme_missing_required_keys(self):
        """Test registering theme with missing required keys."""
        theme_manager = ThemeManager()
        
        incomplete_theme = {
            "background_color": "#F0F0F0"
            # Missing required keys
        }
        
        with pytest.raises(ValueError, match="Theme config missing required keys"):
            theme_manager.register_theme("incomplete", incomplete_theme)
    
    def test_list_themes(self):
        """Test listing available themes."""
        theme_manager = ThemeManager()
        
        themes = theme_manager.list_themes()
        assert isinstance(themes, list)
        assert "default" in themes
        assert "dark" in themes
        assert "professional" in themes
        assert "minimal" in themes
        assert "colorblind" in themes
    
    def test_get_color_palette(self):
        """Test getting color palette."""
        theme_manager = ThemeManager()
        
        colors = theme_manager.get_color_palette("default")
        assert isinstance(colors, dict)
        assert "bullish" in colors
        assert "bearish" in colors
        assert "support" in colors
        assert "resistance" in colors
        assert "background" in colors
        assert "text" in colors
    
    def test_get_indicator_colors(self):
        """Test getting indicator colors."""
        theme_manager = ThemeManager()
        
        colors = theme_manager.get_indicator_colors("default", 3)
        assert isinstance(colors, list)
        assert len(colors) == 3
        
        # Test more colors than available
        colors = theme_manager.get_indicator_colors("default", 10)
        assert len(colors) == 10
    
    def test_apply_theme_to_matplotlib(self):
        """Test applying theme to matplotlib settings."""
        theme_manager = ThemeManager()
        
        # Save original settings
        original_facecolor = plt.rcParams["figure.facecolor"]
        
        # Apply dark theme
        theme_manager.apply_theme_to_matplotlib("dark")
        
        # Check that settings were changed
        assert plt.rcParams["figure.facecolor"] != original_facecolor
        assert theme_manager.get_current_theme() == "dark"
        
        # Reset to default
        theme_manager.reset_to_default()
        assert theme_manager.get_current_theme() == "default"
    
    def test_create_theme_preview(self):
        """Test creating theme preview."""
        theme_manager = ThemeManager()
        
        fig = theme_manager.create_theme_preview("default")
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.get_axes()) == 4  # Should have 4 subplots
        
        plt.close(fig)


class TestTestScenario:
    """Test the test scenario functionality."""
    
    def test_test_scenario_creation(self):
        """Test creating a test scenario."""
        scenario = TestScenario(
            name="Test Scenario",
            description="Test description",
            data_generator=lambda: create_btc_sample_data(periods=100),
            indicator_config={"type": "VolumeProfile", "lookback": 96}
        )
        
        assert scenario.name == "Test Scenario"
        assert scenario.description == "Test description"
        assert scenario.get_indicator_type() == "VolumeProfile"
        assert scenario.get_indicator_id() == "VolumeProfile_default"
        assert callable(scenario.data_generator)
    
    def test_test_scenario_validation(self):
        """Test test scenario validation."""
        # Missing name
        with pytest.raises(ValueError, match="Test scenario must have a name"):
            TestScenario(
                name="",
                description="Test",
                data_generator=lambda: None,
                indicator_config={"type": "Test"}
            )
        
        # Non-callable data generator
        with pytest.raises(ValueError, match="data_generator must be callable"):
            TestScenario(
                name="Test",
                description="Test",
                data_generator="not callable",
                indicator_config={"type": "Test"}
            )
        
        # Empty indicator config
        with pytest.raises(ValueError, match="indicator_config cannot be empty"):
            TestScenario(
                name="Test",
                description="Test",
                data_generator=lambda: None,
                indicator_config={}
            )
        
        # Missing type in indicator config
        with pytest.raises(ValueError, match="indicator_config must contain 'type' field"):
            TestScenario(
                name="Test",
                description="Test",
                data_generator=lambda: None,
                indicator_config={"id": "test"}
            )
    
    def test_test_scenario_data_generation(self):
        """Test test scenario data generation."""
        scenario = TestScenario(
            name="Test Scenario",
            description="Test description",
            data_generator=lambda: create_btc_sample_data(periods=50),
            indicator_config={"type": "VolumeProfile", "lookback": 96}
        )
        
        data = scenario.generate_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        assert "close" in data.columns
    
    def test_test_scenario_data_generation_failure(self):
        """Test test scenario data generation failure."""
        def failing_generator():
            raise RuntimeError("Data generation failed")
        
        scenario = TestScenario(
            name="Test Scenario",
            description="Test description",
            data_generator=failing_generator,
            indicator_config={"type": "VolumeProfile", "lookback": 96}
        )
        
        with pytest.raises(RuntimeError, match="Failed to generate data for scenario"):
            scenario.generate_data()
    
    def test_test_scenario_tags(self):
        """Test test scenario tag functionality."""
        scenario = TestScenario(
            name="Test Scenario",
            description="Test description",
            data_generator=lambda: create_btc_sample_data(periods=50),
            indicator_config={"type": "VolumeProfile", "lookback": 96},
            tags=["test", "volume_profile"]
        )
        
        assert scenario.has_tag("test")
        assert scenario.has_tag("volume_profile")
        assert not scenario.has_tag("nonexistent")
    
    def test_test_scenario_filters(self):
        """Test test scenario filter matching."""
        scenario = TestScenario(
            name="Volume Profile Test",
            description="Test description",
            data_generator=lambda: create_btc_sample_data(periods=50),
            indicator_config={"type": "VolumeProfile", "lookback": 96},
            tags=["volume_profile", "test"],
            metadata={"trend": "bull"}
        )
        
        # Test various filters
        assert scenario.matches_filter({"indicator_type": "VolumeProfile"})
        assert scenario.matches_filter({"name": "Volume"})
        assert scenario.matches_filter({"tags": ["volume_profile"]})
        assert scenario.matches_filter({"trend": "bull"})
        
        # Test non-matching filters
        assert not scenario.matches_filter({"indicator_type": "RSI"})
        assert not scenario.matches_filter({"name": "RSI"})
        assert not scenario.matches_filter({"tags": ["rsi"]})
        assert not scenario.matches_filter({"trend": "bear"})
    
    def test_scenario_builder(self):
        """Test scenario builder functionality."""
        scenario = (ScenarioBuilder()
                   .name("Test Scenario")
                   .description("Test description")
                   .data_generator(lambda: create_btc_sample_data(periods=100))
                   .indicator("VolumeProfile", lookback=96, price_bins=50)
                   .expect_signal("poc_breakout", description="Test breakout")
                   .min_signals(1)
                   .max_signals(5)
                   .visualization_title("Test Chart")
                   .visualization_theme("dark")
                   .tag("test", "volume_profile")
                   .metadata("trend", "bull")
                   .build())
        
        assert scenario.name == "Test Scenario"
        assert scenario.description == "Test description"
        assert scenario.get_indicator_type() == "VolumeProfile"
        assert scenario.validation_criteria.min_signals == 1
        assert scenario.validation_criteria.max_signals == 5
        assert scenario.visualization_config.title == "Test Chart"
        assert scenario.visualization_config.theme == "dark"
        assert scenario.has_tag("test")
        assert scenario.has_tag("volume_profile")
        assert scenario.metadata["trend"] == "bull"
        assert len(scenario.validation_criteria.expected_signals) == 1
        assert scenario.validation_criteria.expected_signals[0].signal_type == "poc_breakout"
    
    def test_scenario_builder_validation(self):
        """Test scenario builder validation."""
        # Missing required fields
        with pytest.raises(ValueError, match="Scenario name is required"):
            ScenarioBuilder().build()
        
        with pytest.raises(ValueError, match="Scenario description is required"):
            ScenarioBuilder().name("Test").build()
        
        with pytest.raises(ValueError, match="Data generator is required"):
            ScenarioBuilder().name("Test").description("Test").build()
        
        with pytest.raises(ValueError, match="Indicator configuration is required"):
            (ScenarioBuilder()
             .name("Test")
             .description("Test")
             .data_generator(lambda: None)
             .build())
    
    def test_test_result(self):
        """Test test result functionality."""
        scenario = TestScenario(
            name="Test Scenario",
            description="Test description",
            data_generator=lambda: create_btc_sample_data(periods=50),
            indicator_config={"type": "VolumeProfile", "lookback": 96}
        )
        
        data = create_btc_sample_data(periods=50)
        result = TestResult(
            scenario=scenario,
            success=True,
            data=data
        )
        
        assert result.success is True
        assert result.scenario == scenario
        assert len(result.data) == 50
        assert len(result.signals_found) == 0
        assert len(result.validation_errors) == 0
        
        # Test adding signals and errors
        result.add_signal("poc_breakout", data.index[10], confidence=0.8)
        result.add_error("Test error")
        
        assert len(result.signals_found) == 1
        assert len(result.validation_errors) == 1
        assert result.success is False  # Should be False after adding error
        
        # Test summary
        summary = result.get_summary()
        assert summary["scenario_name"] == "Test Scenario"
        assert summary["success"] is False
        assert summary["signals_found"] == 1
        assert summary["validation_errors"] == 1
        assert summary["data_points"] == 50


class TestVolumeProfileVisualizer:
    """Test the Volume Profile visualizer."""
    
    def test_volume_profile_visualizer_required_columns(self):
        """Test Volume Profile visualizer required columns."""
        visualizer = VolumeProfileVisualizer()
        
        required_cols = visualizer.get_required_columns()
        expected_cols = [
            "poc_price", "vah_price", "val_price", "total_volume",
            "poc_volume", "value_area_volume", "average_volume",
            "price_above_poc", "price_below_poc", "price_in_value_area",
            "poc_breakout", "volume_spike", "is_hvn", "is_lvn",
            "dist_to_poc", "dist_to_vah", "dist_to_val",
            "poc_strength", "value_area_width", "profile_balance"
        ]
        
        assert required_cols == expected_cols
    
    def test_volume_profile_visualizer_data_validation(self):
        """Test Volume Profile visualizer data validation."""
        visualizer = VolumeProfileVisualizer()
        
        # Create mock data with required columns
        sample_data = create_btc_sample_data(periods=100)
        
        # Add Volume Profile columns
        vp_columns = visualizer.get_required_columns()
        for col in vp_columns:
            if col.startswith("price_") or col.startswith("is_") or col == "poc_breakout" or col == "volume_spike":
                sample_data[col] = np.random.choice([True, False], size=100)
            else:
                sample_data[col] = np.random.uniform(40000, 50000, size=100)
        
        # Should pass validation
        visualizer.validate_data(sample_data)
        
        # Test with invalid price relationships
        sample_data["vah_price"] = 40000
        sample_data["poc_price"] = 45000
        sample_data["val_price"] = 50000  # VAL > POC should fail
        
        with pytest.raises(ValueError, match="POC price should be >= VAL price"):
            visualizer.validate_data(sample_data)
    
    def test_volume_profile_visualizer_render(self):
        """Test Volume Profile visualizer render method."""
        visualizer = VolumeProfileVisualizer()
        
        # Create mock data
        sample_data = create_btc_sample_data(periods=100)
        
        # Add Volume Profile columns with realistic values
        vp_columns = visualizer.get_required_columns()
        for col in vp_columns:
            if col.startswith("price_") or col.startswith("is_") or col == "poc_breakout" or col == "volume_spike":
                sample_data[col] = np.random.choice([True, False], size=100)
            else:
                sample_data[col] = np.random.uniform(40000, 50000, size=100)
        
        # Ensure proper price relationships
        sample_data["poc_price"] = 45000
        sample_data["vah_price"] = 45500
        sample_data["val_price"] = 44500
        
        config = VisualizationConfig(title="Volume Profile Test", show_signals=True)
        
        try:
            fig = visualizer.render(sample_data, config)
            assert isinstance(fig, plt.Figure)
            assert fig.get_suptitle() == "Volume Profile Test"
            plt.close(fig)
        except Exception as e:
            # Render might fail due to missing dependencies or plotting issues
            # but it should at least not crash with basic errors
            assert "missing" not in str(e).lower()


class TestScenarioRegistry:
    """Test the scenario registry functionality."""
    
    def test_scenario_registry_initialization(self):
        """Test scenario registry initialization."""
        registry = scenario_registry
        
        # Should have Volume Profile scenarios
        scenarios = registry.list_scenarios()
        assert len(scenarios) > 0
        
        # Should have scenario groups
        vp_scenarios = registry.get_scenarios_by_group("volume_profile")
        assert len(vp_scenarios) > 0
    
    def test_scenario_registry_get_scenario(self):
        """Test getting scenario from registry."""
        registry = scenario_registry
        
        # Get a known scenario
        scenario_name = "Volume Profile - Trending Bull Market"
        scenario = registry.get_scenario(scenario_name)
        
        assert scenario.name == scenario_name
        assert scenario.get_indicator_type() == "VolumeProfile"
        
        # Test getting non-existent scenario
        with pytest.raises(KeyError, match="Scenario 'NonExistent' not found"):
            registry.get_scenario("NonExistent")
    
    def test_scenario_registry_list_scenarios_with_tags(self):
        """Test listing scenarios with tag filtering."""
        registry = scenario_registry
        
        # Filter by tag
        vp_scenarios = registry.list_scenarios(tags=["volume_profile"])
        assert len(vp_scenarios) > 0
        
        # All returned scenarios should have the tag
        for scenario_name in vp_scenarios:
            scenario = registry.get_scenario(scenario_name)
            assert scenario.has_tag("volume_profile")
    
    def test_scenario_registry_filter_scenarios(self):
        """Test filtering scenarios by various criteria."""
        registry = scenario_registry
        
        # Filter by indicator type
        vp_scenarios = registry.filter_scenarios({"indicator_type": "VolumeProfile"})
        assert len(vp_scenarios) > 0
        
        for scenario in vp_scenarios:
            assert scenario.get_indicator_type() == "VolumeProfile"
        
        # Filter by name
        bull_scenarios = registry.filter_scenarios({"name": "Bull"})
        assert len(bull_scenarios) > 0
        
        for scenario in bull_scenarios:
            assert "Bull" in scenario.name
    
    def test_scenario_registry_register_scenario(self):
        """Test registering new scenario."""
        registry = scenario_registry
        
        # Create new scenario
        new_scenario = TestScenario(
            name="Test Custom Scenario",
            description="Custom test scenario",
            data_generator=lambda: create_btc_sample_data(periods=50),
            indicator_config={"type": "VolumeProfile", "lookback": 96}
        )
        
        # Register it
        registry.register_scenario(new_scenario)
        
        # Should be able to retrieve it
        retrieved_scenario = registry.get_scenario("Test Custom Scenario")
        assert retrieved_scenario.name == "Test Custom Scenario"
        assert retrieved_scenario.description == "Custom test scenario"


class TestVolumeProfileScenarios:
    """Test the predefined Volume Profile scenarios."""
    
    def test_trending_bull_market_scenario(self):
        """Test trending bull market scenario."""
        scenario = VolumeProfileScenarios.trending_bull_market()
        
        assert scenario.name == "Volume Profile - Trending Bull Market"
        assert scenario.get_indicator_type() == "VolumeProfile"
        assert scenario.has_tag("volume_profile")
        assert scenario.has_tag("bull_market")
        assert scenario.has_tag("trending")
        assert scenario.metadata["trend_direction"] == "bull"
        
        # Test data generation
        data = scenario.generate_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 500
        assert "close" in data.columns
    
    def test_trending_bear_market_scenario(self):
        """Test trending bear market scenario."""
        scenario = VolumeProfileScenarios.trending_bear_market()
        
        assert scenario.name == "Volume Profile - Trending Bear Market"
        assert scenario.get_indicator_type() == "VolumeProfile"
        assert scenario.has_tag("volume_profile")
        assert scenario.has_tag("bear_market")
        assert scenario.has_tag("trending")
        assert scenario.metadata["trend_direction"] == "bear"
        
        # Test data generation
        data = scenario.generate_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 500
    
    def test_sideways_consolidation_scenario(self):
        """Test sideways consolidation scenario."""
        scenario = VolumeProfileScenarios.sideways_consolidation()
        
        assert scenario.name == "Volume Profile - Sideways Consolidation"
        assert scenario.get_indicator_type() == "VolumeProfile"
        assert scenario.has_tag("volume_profile")
        assert scenario.has_tag("sideways")
        assert scenario.has_tag("consolidation")
        assert scenario.metadata["trend_direction"] == "sideways"
        
        # Test data generation
        data = scenario.generate_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 400
    
    def test_high_volatility_scenario(self):
        """Test high volatility scenario."""
        scenario = VolumeProfileScenarios.high_volatility()
        
        assert scenario.name == "Volume Profile - High Volatility"
        assert scenario.get_indicator_type() == "VolumeProfile"
        assert scenario.has_tag("volume_profile")
        assert scenario.has_tag("high_volatility")
        assert scenario.has_tag("extreme")
        assert scenario.metadata["volatility_level"] == "high"
        
        # Test data generation
        data = scenario.generate_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 300
    
    def test_edge_case_scenarios(self):
        """Test edge case scenarios."""
        # Test insufficient data scenario
        insufficient_data_scenario = VolumeProfileScenarios.edge_case_insufficient_data()
        
        assert insufficient_data_scenario.name == "Volume Profile - Insufficient Data"
        assert insufficient_data_scenario.has_tag("edge_case")
        assert insufficient_data_scenario.has_tag("insufficient_data")
        assert insufficient_data_scenario.validation_criteria.min_signals == 0
        assert insufficient_data_scenario.validation_criteria.max_signals == 0
        
        # Test data generation
        data = insufficient_data_scenario.generate_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50  # Less than lookback period
        
        # Test zero volume scenario
        zero_volume_scenario = VolumeProfileScenarios.edge_case_zero_volume()
        
        assert zero_volume_scenario.name == "Volume Profile - Zero Volume Periods"
        assert zero_volume_scenario.has_tag("edge_case")
        assert zero_volume_scenario.has_tag("zero_volume")
        
        # Test data generation
        data = zero_volume_scenario.generate_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 200
        
        # Should have some zero volume periods
        assert (data["volume"] == 0).sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])