"""
Test scenario data structures for the visual testing framework.

This module defines the data structures and classes used to create, manage,
and execute test scenarios for indicator validation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd

from btc_research.visual_testing.core.base_visualizer import VisualizationConfig


@dataclass
class ExpectedSignal:
    """
    Represents an expected signal in a test scenario.
    
    Attributes:
        signal_type: Type of signal (e.g., "buy", "sell", "poc_breakout")
        timestamp: Expected timestamp for the signal
        tolerance: Time tolerance in periods for signal timing
        description: Human-readable description of the signal
        required: Whether this signal is required for test to pass
    """
    signal_type: str
    timestamp: Optional[pd.Timestamp] = None
    tolerance: int = 1
    description: str = ""
    required: bool = True


@dataclass
class ValidationCriteria:
    """
    Validation criteria for test scenarios.
    
    Attributes:
        min_signals: Minimum number of signals expected
        max_signals: Maximum number of signals expected
        expected_signals: List of specific expected signals
        signal_accuracy_threshold: Minimum accuracy rate for signals
        timing_tolerance: Tolerance for signal timing in periods
        custom_validators: List of custom validation functions
    """
    min_signals: int = 0
    max_signals: int = float('inf')
    expected_signals: List[ExpectedSignal] = field(default_factory=list)
    signal_accuracy_threshold: float = 0.8
    timing_tolerance: int = 2
    custom_validators: List[Callable] = field(default_factory=list)


@dataclass
class TestScenario:
    """
    Represents a complete test scenario for indicator validation.
    
    A test scenario encompasses everything needed to validate an indicator:
    - Market data (generated or historical)
    - Indicator configuration
    - Expected behavior and signals
    - Validation criteria
    - Visualization configuration
    
    Example:
        ```python
        from btc_research.visual_testing.core.test_scenario import TestScenario
        from tests.fixtures.sample_data import create_trending_market_data
        
        # Create a Volume Profile test scenario
        scenario = TestScenario(
            name="Volume Profile - Bull Market",
            description="Test VP behavior in trending bull market",
            data_generator=lambda: create_trending_market_data(trend="bull"),
            indicator_config={
                "id": "VP_96",
                "type": "VolumeProfile",
                "timeframe": "15m",
                "lookback": 96,
                "price_bins": 50
            },
            validation_criteria=ValidationCriteria(
                expected_signals=[
                    ExpectedSignal("poc_breakout", description="POC breakout on trend continuation"),
                    ExpectedSignal("volume_spike", description="Volume spike during breakout")
                ]
            ),
            visualization_config=VisualizationConfig(
                title="Volume Profile - Bull Market Test",
                show_signals=True,
                theme="default"
            )
        )
        ```
    
    Attributes:
        name: Unique name for the test scenario
        description: Human-readable description
        data_generator: Function that generates or loads test data
        indicator_config: Configuration for the indicator being tested
        validation_criteria: Criteria for validating the test results
        visualization_config: Configuration for visualization rendering
        tags: Tags for categorizing and filtering scenarios
        metadata: Additional metadata for the scenario
        expected_outputs: Expected output column names and types
        setup_function: Optional function to run before test
        teardown_function: Optional function to run after test
    """
    name: str
    description: str
    data_generator: Callable[[], pd.DataFrame]
    indicator_config: Dict[str, Any]
    validation_criteria: ValidationCriteria = field(default_factory=ValidationCriteria)
    visualization_config: VisualizationConfig = field(default_factory=lambda: VisualizationConfig(title="Test Scenario"))
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Test scenario must have a name")
        if not callable(self.data_generator):
            raise ValueError("data_generator must be callable")
        if not self.indicator_config:
            raise ValueError("indicator_config cannot be empty")
        if "type" not in self.indicator_config:
            raise ValueError("indicator_config must contain 'type' field")
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate the test data for this scenario.
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            return self.data_generator()
        except Exception as e:
            raise RuntimeError(f"Failed to generate data for scenario '{self.name}': {e}")
    
    def get_indicator_type(self) -> str:
        """
        Get the indicator type for this scenario.
        
        Returns:
            Indicator type string
        """
        return self.indicator_config["type"]
    
    def get_indicator_id(self) -> str:
        """
        Get the indicator ID for this scenario.
        
        Returns:
            Indicator ID string
        """
        return self.indicator_config.get("id", f"{self.get_indicator_type()}_default")
    
    def has_tag(self, tag: str) -> bool:
        """
        Check if scenario has a specific tag.
        
        Args:
            tag: Tag to check for
            
        Returns:
            True if scenario has the tag
        """
        return tag in self.tags
    
    def matches_filter(self, filters: Dict[str, Any]) -> bool:
        """
        Check if scenario matches the given filters.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            True if scenario matches all filters
        """
        for key, value in filters.items():
            if key == "tags":
                if not any(tag in self.tags for tag in value):
                    return False
            elif key == "indicator_type":
                if self.get_indicator_type() != value:
                    return False
            elif key == "name":
                if value not in self.name:
                    return False
            elif key in self.metadata:
                if self.metadata[key] != value:
                    return False
        return True


@dataclass
class TestResult:
    """
    Results from executing a test scenario.
    
    Attributes:
        scenario: The test scenario that was executed
        success: Whether the test passed
        data: The processed data with indicator values
        signals_found: List of signals found during execution
        validation_errors: List of validation errors
        performance_metrics: Performance metrics from execution
        visualization_path: Path to saved visualization (if any)
        execution_time: Time taken to execute the test
        metadata: Additional result metadata
    """
    scenario: TestScenario
    success: bool
    data: pd.DataFrame
    signals_found: List[Dict[str, Any]] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    visualization_path: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str) -> None:
        """
        Add a validation error to the results.
        
        Args:
            error: Error message to add
        """
        self.validation_errors.append(error)
        self.success = False
    
    def add_signal(self, signal_type: str, timestamp: pd.Timestamp, 
                   confidence: float = 1.0, **kwargs) -> None:
        """
        Add a found signal to the results.
        
        Args:
            signal_type: Type of signal found
            timestamp: Timestamp of the signal
            confidence: Confidence level (0.0-1.0)
            **kwargs: Additional signal metadata
        """
        self.signals_found.append({
            "type": signal_type,
            "timestamp": timestamp,
            "confidence": confidence,
            **kwargs
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the test results.
        
        Returns:
            Dictionary containing result summary
        """
        return {
            "scenario_name": self.scenario.name,
            "success": self.success,
            "signals_found": len(self.signals_found),
            "validation_errors": len(self.validation_errors),
            "execution_time": self.execution_time,
            "has_visualization": self.visualization_path is not None,
            "data_points": len(self.data) if self.data is not None else 0,
        }


class ScenarioBuilder:
    """
    Builder class for creating test scenarios with fluent interface.
    
    This class provides a convenient way to build test scenarios step by step
    with a fluent interface pattern.
    
    Example:
        ```python
        from btc_research.visual_testing.core.test_scenario import ScenarioBuilder
        from tests.fixtures.sample_data import create_volatile_market_data
        
        scenario = (ScenarioBuilder()
                   .name("RSI Extreme Volatility")
                   .description("Test RSI behavior during extreme volatility")
                   .data_generator(lambda: create_volatile_market_data(volatility_level="extreme"))
                   .indicator("RSI", id="RSI_14", length=14)
                   .expect_signal("oversold", description="RSI oversold signal")
                   .expect_signal("overbought", description="RSI overbought signal")
                   .visualization_title("RSI - Extreme Volatility Test")
                   .tag("rsi", "volatility", "extreme")
                   .build())
        ```
    """
    
    def __init__(self):
        """Initialize the builder."""
        self._name: Optional[str] = None
        self._description: Optional[str] = None
        self._data_generator: Optional[Callable] = None
        self._indicator_config: Dict[str, Any] = {}
        self._validation_criteria: ValidationCriteria = ValidationCriteria()
        self._visualization_config: VisualizationConfig = VisualizationConfig(title="Test Scenario")
        self._tags: List[str] = []
        self._metadata: Dict[str, Any] = {}
        self._expected_outputs: List[str] = []
        self._setup_function: Optional[Callable] = None
        self._teardown_function: Optional[Callable] = None
    
    def name(self, name: str) -> 'ScenarioBuilder':
        """Set the scenario name."""
        self._name = name
        return self
    
    def description(self, description: str) -> 'ScenarioBuilder':
        """Set the scenario description."""
        self._description = description
        return self
    
    def data_generator(self, generator: Callable[[], pd.DataFrame]) -> 'ScenarioBuilder':
        """Set the data generator function."""
        self._data_generator = generator
        return self
    
    def indicator(self, indicator_type: str, **kwargs) -> 'ScenarioBuilder':
        """Set the indicator configuration."""
        self._indicator_config = {"type": indicator_type, **kwargs}
        return self
    
    def expect_signal(self, signal_type: str, timestamp: Optional[pd.Timestamp] = None,
                     tolerance: int = 1, description: str = "", required: bool = True) -> 'ScenarioBuilder':
        """Add an expected signal to the validation criteria."""
        signal = ExpectedSignal(signal_type, timestamp, tolerance, description, required)
        self._validation_criteria.expected_signals.append(signal)
        return self
    
    def min_signals(self, min_count: int) -> 'ScenarioBuilder':
        """Set minimum number of signals expected."""
        self._validation_criteria.min_signals = min_count
        return self
    
    def max_signals(self, max_count: int) -> 'ScenarioBuilder':
        """Set maximum number of signals expected."""
        self._validation_criteria.max_signals = max_count
        return self
    
    def visualization_title(self, title: str) -> 'ScenarioBuilder':
        """Set the visualization title."""
        self._visualization_config.title = title
        return self
    
    def visualization_theme(self, theme: str) -> 'ScenarioBuilder':
        """Set the visualization theme."""
        self._visualization_config.theme = theme
        return self
    
    def tag(self, *tags: str) -> 'ScenarioBuilder':
        """Add tags to the scenario."""
        self._tags.extend(tags)
        return self
    
    def metadata(self, key: str, value: Any) -> 'ScenarioBuilder':
        """Add metadata to the scenario."""
        self._metadata[key] = value
        return self
    
    def setup(self, setup_function: Callable) -> 'ScenarioBuilder':
        """Set the setup function."""
        self._setup_function = setup_function
        return self
    
    def teardown(self, teardown_function: Callable) -> 'ScenarioBuilder':
        """Set the teardown function."""
        self._teardown_function = teardown_function
        return self
    
    def build(self) -> TestScenario:
        """Build the test scenario."""
        if not self._name:
            raise ValueError("Scenario name is required")
        if not self._description:
            raise ValueError("Scenario description is required")
        if not self._data_generator:
            raise ValueError("Data generator is required")
        if not self._indicator_config:
            raise ValueError("Indicator configuration is required")
        
        return TestScenario(
            name=self._name,
            description=self._description,
            data_generator=self._data_generator,
            indicator_config=self._indicator_config,
            validation_criteria=self._validation_criteria,
            visualization_config=self._visualization_config,
            tags=self._tags,
            metadata=self._metadata,
            expected_outputs=self._expected_outputs,
            setup_function=self._setup_function,
            teardown_function=self._teardown_function,
        )