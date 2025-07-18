"""
Predefined test scenarios for the visual testing framework.

This module contains predefined test scenarios for different indicators
and market conditions, providing a comprehensive suite of validation tests.
"""

from typing import List, Dict, Any
import pandas as pd

from btc_research.visual_testing.core.test_scenario import (
    TestScenario, ScenarioBuilder, ExpectedSignal, ValidationCriteria
)
from btc_research.visual_testing.core.base_visualizer import VisualizationConfig
from tests.fixtures.sample_data import (
    create_btc_sample_data,
    create_trending_market_data,
    create_volatile_market_data,
    create_gap_data,
    create_multi_timeframe_data
)


class VolumeProfileScenarios:
    """
    Predefined test scenarios for Volume Profile indicator.
    
    This class provides a collection of test scenarios specifically designed
    to validate Volume Profile indicator behavior under various market conditions.
    """
    
    @staticmethod
    def trending_bull_market() -> TestScenario:
        """
        Test Volume Profile in a trending bull market.
        
        Expected behavior:
        - POC should migrate upward with trend
        - Volume spikes should occur on breakouts
        - Value area should expand during consolidation
        """
        return (ScenarioBuilder()
                .name("Volume Profile - Trending Bull Market")
                .description("Test VP behavior in sustained bull market with clear trend")
                .data_generator(lambda: create_trending_market_data(
                    periods=500, trend="bull", volatility=0.015
                ))
                .indicator("VolumeProfile", 
                          id="VP_96",
                          lookback=96,
                          price_bins=50,
                          value_area_pct=70,
                          atr_period=14,
                          atr_multiplier=2.0)
                .expect_signal("poc_breakout", 
                              description="POC breakout during trend continuation")
                .expect_signal("volume_spike", 
                              description="Volume spike on strong moves")
                .min_signals(2)
                .max_signals(10)
                .visualization_title("Volume Profile - Bull Market Analysis")
                .visualization_theme("default")
                .tag("volume_profile", "bull_market", "trending")
                .metadata("trend_direction", "bull")
                .metadata("expected_poc_trend", "upward")
                .build())
    
    @staticmethod
    def trending_bear_market() -> TestScenario:
        """
        Test Volume Profile in a trending bear market.
        
        Expected behavior:
        - POC should migrate downward with trend
        - Volume spikes on breakdown moves
        - Support levels should break down
        """
        return (ScenarioBuilder()
                .name("Volume Profile - Trending Bear Market")
                .description("Test VP behavior in sustained bear market with clear downtrend")
                .data_generator(lambda: create_trending_market_data(
                    periods=500, trend="bear", volatility=0.015
                ))
                .indicator("VolumeProfile",
                          id="VP_96", 
                          lookback=96,
                          price_bins=50,
                          value_area_pct=70,
                          atr_period=14,
                          atr_multiplier=2.0)
                .expect_signal("poc_breakout",
                              description="POC breakdown during trend continuation")
                .expect_signal("volume_spike",
                              description="Volume spike on breakdown moves")
                .min_signals(2)
                .max_signals(10)
                .visualization_title("Volume Profile - Bear Market Analysis")
                .visualization_theme("default")
                .tag("volume_profile", "bear_market", "trending")
                .metadata("trend_direction", "bear")
                .metadata("expected_poc_trend", "downward")
                .build())
    
    @staticmethod
    def sideways_consolidation() -> TestScenario:
        """
        Test Volume Profile in sideways consolidation.
        
        Expected behavior:
        - POC should remain relatively stable
        - Value area should be well-defined
        - Volume clusters should form at support/resistance
        """
        return (ScenarioBuilder()
                .name("Volume Profile - Sideways Consolidation")
                .description("Test VP behavior during sideways consolidation phase")
                .data_generator(lambda: create_trending_market_data(
                    periods=400, trend="sideways", volatility=0.012
                ))
                .indicator("VolumeProfile",
                          id="VP_96",
                          lookback=96,
                          price_bins=50,
                          value_area_pct=70,
                          atr_period=14,
                          atr_multiplier=2.0)
                .expect_signal("volume_spike",
                              description="Volume spike at support/resistance test")
                .min_signals(1)
                .max_signals(5)
                .visualization_title("Volume Profile - Sideways Market Analysis")
                .visualization_theme("default")
                .tag("volume_profile", "sideways", "consolidation")
                .metadata("trend_direction", "sideways")
                .metadata("expected_poc_trend", "stable")
                .build())
    
    @staticmethod
    def high_volatility() -> TestScenario:
        """
        Test Volume Profile during high volatility periods.
        
        Expected behavior:
        - Frequent POC breakouts
        - Multiple volume spikes
        - Wider value areas
        """
        return (ScenarioBuilder()
                .name("Volume Profile - High Volatility")
                .description("Test VP behavior during extreme volatility periods")
                .data_generator(lambda: create_volatile_market_data(
                    periods=300, volatility_level="high"
                ))
                .indicator("VolumeProfile",
                          id="VP_96",
                          lookback=96,
                          price_bins=50,
                          value_area_pct=70,
                          atr_period=14,
                          atr_multiplier=2.0)
                .expect_signal("poc_breakout",
                              description="Frequent POC breakouts in volatile market")
                .expect_signal("volume_spike",
                              description="Multiple volume spikes on volatile moves")
                .min_signals(5)
                .max_signals(20)
                .visualization_title("Volume Profile - High Volatility Analysis")
                .visualization_theme("default")
                .tag("volume_profile", "high_volatility", "extreme")
                .metadata("volatility_level", "high")
                .metadata("expected_signals", "frequent")
                .build())
    
    @staticmethod
    def gap_trading() -> TestScenario:
        """
        Test Volume Profile with price gaps.
        
        Expected behavior:
        - POC should adapt to gap moves
        - Volume spikes on gap fills
        - Value area should shift with gaps
        """
        return (ScenarioBuilder()
                .name("Volume Profile - Gap Trading")
                .description("Test VP behavior with price gaps and gap fills")
                .data_generator(lambda: create_gap_data(
                    periods=250, gap_frequency=0.08, gap_size_range=(0.02, 0.06)
                ))
                .indicator("VolumeProfile",
                          id="VP_96",
                          lookback=96,
                          price_bins=50,
                          value_area_pct=70,
                          atr_period=14,
                          atr_multiplier=2.0)
                .expect_signal("poc_breakout",
                              description="POC breakout on gap moves")
                .expect_signal("volume_spike",
                              description="Volume spike on gap fills")
                .min_signals(3)
                .max_signals(15)
                .visualization_title("Volume Profile - Gap Trading Analysis")
                .visualization_theme("default")
                .tag("volume_profile", "gaps", "breakouts")
                .metadata("gap_frequency", 0.08)
                .metadata("expected_behavior", "gap_adaptation")
                .build())
    
    @staticmethod
    def parameter_sensitivity_short_lookback() -> TestScenario:
        """
        Test Volume Profile with short lookback period.
        
        Expected behavior:
        - More responsive to recent price action
        - Frequent POC updates
        - Narrower value areas
        """
        return (ScenarioBuilder()
                .name("Volume Profile - Short Lookback (48)")
                .description("Test VP with short lookback period for responsiveness")
                .data_generator(lambda: create_btc_sample_data(
                    periods=300, freq="15min", seed=42
                ))
                .indicator("VolumeProfile",
                          id="VP_48",
                          lookback=48,
                          price_bins=50,
                          value_area_pct=70,
                          atr_period=14,
                          atr_multiplier=2.0)
                .expect_signal("poc_breakout",
                              description="Frequent POC updates with short lookback")
                .min_signals(3)
                .max_signals(15)
                .visualization_title("Volume Profile - Short Lookback Analysis")
                .visualization_theme("default")
                .tag("volume_profile", "short_lookback", "responsive")
                .metadata("lookback_period", 48)
                .metadata("expected_behavior", "responsive")
                .build())
    
    @staticmethod
    def parameter_sensitivity_long_lookback() -> TestScenario:
        """
        Test Volume Profile with long lookback period.
        
        Expected behavior:
        - Smoother POC transitions
        - More stable signals
        - Wider value areas reflecting longer history
        """
        return (ScenarioBuilder()
                .name("Volume Profile - Long Lookback (144)")
                .description("Test VP with long lookback period for stability")
                .data_generator(lambda: create_btc_sample_data(
                    periods=400, freq="15min", seed=42
                ))
                .indicator("VolumeProfile",
                          id="VP_144",
                          lookback=144,
                          price_bins=50,
                          value_area_pct=70,
                          atr_period=14,
                          atr_multiplier=2.0)
                .expect_signal("poc_breakout",
                              description="Stable POC with long lookback")
                .min_signals(1)
                .max_signals(8)
                .visualization_title("Volume Profile - Long Lookback Analysis")
                .visualization_theme("default")
                .tag("volume_profile", "long_lookback", "stable")
                .metadata("lookback_period", 144)
                .metadata("expected_behavior", "stable")
                .build())
    
    @staticmethod
    def high_resolution_bins() -> TestScenario:
        """
        Test Volume Profile with high resolution (many price bins).
        
        Expected behavior:
        - More precise POC identification
        - Detailed volume distribution
        - Potentially more noise in signals
        """
        return (ScenarioBuilder()
                .name("Volume Profile - High Resolution (100 bins)")
                .description("Test VP with high resolution price binning")
                .data_generator(lambda: create_btc_sample_data(
                    periods=350, freq="15min", seed=42
                ))
                .indicator("VolumeProfile",
                          id="VP_100",
                          lookback=96,
                          price_bins=100,
                          value_area_pct=70,
                          atr_period=14,
                          atr_multiplier=2.0)
                .expect_signal("poc_breakout",
                              description="Precise POC with high resolution")
                .min_signals(2)
                .max_signals(12)
                .visualization_title("Volume Profile - High Resolution Analysis")
                .visualization_theme("default")
                .tag("volume_profile", "high_resolution", "precise")
                .metadata("price_bins", 100)
                .metadata("expected_behavior", "precise")
                .build())
    
    @staticmethod
    def edge_case_insufficient_data() -> TestScenario:
        """
        Test Volume Profile with insufficient data.
        
        Expected behavior:
        - Should handle gracefully without errors
        - No signals should be generated
        - All indicator values should be NaN
        """
        return (ScenarioBuilder()
                .name("Volume Profile - Insufficient Data")
                .description("Test VP behavior with insufficient data")
                .data_generator(lambda: create_btc_sample_data(
                    periods=50, freq="15min", seed=42  # Less than lookback
                ))
                .indicator("VolumeProfile",
                          id="VP_96",
                          lookback=96,
                          price_bins=50,
                          value_area_pct=70,
                          atr_period=14,
                          atr_multiplier=2.0)
                .min_signals(0)
                .max_signals(0)
                .visualization_title("Volume Profile - Insufficient Data Test")
                .visualization_theme("default")
                .tag("volume_profile", "edge_case", "insufficient_data")
                .metadata("expected_behavior", "graceful_handling")
                .build())
    
    @staticmethod
    def edge_case_zero_volume() -> TestScenario:
        """
        Test Volume Profile with zero volume periods.
        
        Expected behavior:
        - Should handle zero volume gracefully
        - POC should adapt to available volume data
        - No divide-by-zero errors
        """
        def zero_volume_data():
            df = create_btc_sample_data(periods=200, freq="15min", seed=42)
            # Set some periods to zero volume
            df.loc[df.index[50:70], "volume"] = 0
            df.loc[df.index[120:130], "volume"] = 0
            return df
        
        return (ScenarioBuilder()
                .name("Volume Profile - Zero Volume Periods")
                .description("Test VP behavior with zero volume periods")
                .data_generator(zero_volume_data)
                .indicator("VolumeProfile",
                          id="VP_96",
                          lookback=96,
                          price_bins=50,
                          value_area_pct=70,
                          atr_period=14,
                          atr_multiplier=2.0)
                .min_signals(0)
                .max_signals(10)
                .visualization_title("Volume Profile - Zero Volume Test")
                .visualization_theme("default")
                .tag("volume_profile", "edge_case", "zero_volume")
                .metadata("expected_behavior", "graceful_handling")
                .build())


class ComparisonScenarios:
    """
    Scenarios for comparing different indicator configurations.
    
    These scenarios are designed to compare the behavior of indicators
    with different parameters or in different market conditions.
    """
    
    @staticmethod
    def volume_profile_lookback_comparison() -> List[TestScenario]:
        """
        Compare Volume Profile with different lookback periods.
        
        Returns:
            List of scenarios for lookback comparison
        """
        base_data_generator = lambda: create_btc_sample_data(
            periods=400, freq="15min", seed=42
        )
        
        scenarios = []
        
        for lookback in [48, 96, 144]:
            scenario = (ScenarioBuilder()
                       .name(f"Volume Profile - Lookback {lookback}")
                       .description(f"VP with {lookback} period lookback")
                       .data_generator(base_data_generator)
                       .indicator("VolumeProfile",
                                 id=f"VP_{lookback}",
                                 lookback=lookback,
                                 price_bins=50,
                                 value_area_pct=70,
                                 atr_period=14,
                                 atr_multiplier=2.0)
                       .visualization_title(f"Volume Profile - Lookback {lookback}")
                       .visualization_theme("default")
                       .tag("volume_profile", "comparison", "lookback")
                       .metadata("lookback_period", lookback)
                       .build())
            scenarios.append(scenario)
        
        return scenarios
    
    @staticmethod
    def volume_profile_resolution_comparison() -> List[TestScenario]:
        """
        Compare Volume Profile with different price bin resolutions.
        
        Returns:
            List of scenarios for resolution comparison
        """
        base_data_generator = lambda: create_btc_sample_data(
            periods=350, freq="15min", seed=42
        )
        
        scenarios = []
        
        for bins in [25, 50, 100]:
            scenario = (ScenarioBuilder()
                       .name(f"Volume Profile - {bins} Bins")
                       .description(f"VP with {bins} price bins")
                       .data_generator(base_data_generator)
                       .indicator("VolumeProfile",
                                 id=f"VP_{bins}",
                                 lookback=96,
                                 price_bins=bins,
                                 value_area_pct=70,
                                 atr_period=14,
                                 atr_multiplier=2.0)
                       .visualization_title(f"Volume Profile - {bins} Bins")
                       .visualization_theme("default")
                       .tag("volume_profile", "comparison", "resolution")
                       .metadata("price_bins", bins)
                       .build())
            scenarios.append(scenario)
        
        return scenarios


class TestScenarioRegistry:
    """
    Registry for managing and accessing test scenarios.
    
    This class provides a centralized way to register, discover, and
    execute test scenarios.
    """
    
    def __init__(self):
        """Initialize the scenario registry."""
        self._scenarios: Dict[str, TestScenario] = {}
        self._scenario_groups: Dict[str, List[str]] = {}
        self._register_default_scenarios()
    
    def _register_default_scenarios(self):
        """Register all default scenarios."""
        # Volume Profile scenarios
        vp_scenarios = [
            VolumeProfileScenarios.trending_bull_market(),
            VolumeProfileScenarios.trending_bear_market(),
            VolumeProfileScenarios.sideways_consolidation(),
            VolumeProfileScenarios.high_volatility(),
            VolumeProfileScenarios.gap_trading(),
            VolumeProfileScenarios.parameter_sensitivity_short_lookback(),
            VolumeProfileScenarios.parameter_sensitivity_long_lookback(),
            VolumeProfileScenarios.high_resolution_bins(),
            VolumeProfileScenarios.edge_case_insufficient_data(),
            VolumeProfileScenarios.edge_case_zero_volume(),
        ]
        
        for scenario in vp_scenarios:
            self.register_scenario(scenario)
        
        # Register scenario groups
        self._scenario_groups["volume_profile"] = [
            s.name for s in vp_scenarios
        ]
        
        self._scenario_groups["trending"] = [
            s.name for s in vp_scenarios if s.has_tag("trending")
        ]
        
        self._scenario_groups["edge_cases"] = [
            s.name for s in vp_scenarios if s.has_tag("edge_case")
        ]
    
    def register_scenario(self, scenario: TestScenario):
        """
        Register a new test scenario.
        
        Args:
            scenario: Test scenario to register
        """
        self._scenarios[scenario.name] = scenario
    
    def get_scenario(self, name: str) -> TestScenario:
        """
        Get a scenario by name.
        
        Args:
            name: Name of the scenario
            
        Returns:
            Test scenario
            
        Raises:
            KeyError: If scenario not found
        """
        if name not in self._scenarios:
            available = list(self._scenarios.keys())
            raise KeyError(f"Scenario '{name}' not found. Available: {available}")
        
        return self._scenarios[name]
    
    def list_scenarios(self, tags: List[str] = None) -> List[str]:
        """
        List available scenario names, optionally filtered by tags.
        
        Args:
            tags: Optional list of tags to filter by
            
        Returns:
            List of scenario names
        """
        if tags is None:
            return list(self._scenarios.keys())
        
        return [
            name for name, scenario in self._scenarios.items()
            if any(scenario.has_tag(tag) for tag in tags)
        ]
    
    def get_scenarios_by_group(self, group_name: str) -> List[TestScenario]:
        """
        Get scenarios by group name.
        
        Args:
            group_name: Name of the scenario group
            
        Returns:
            List of test scenarios
        """
        if group_name not in self._scenario_groups:
            available = list(self._scenario_groups.keys())
            raise KeyError(f"Group '{group_name}' not found. Available: {available}")
        
        scenario_names = self._scenario_groups[group_name]
        return [self._scenarios[name] for name in scenario_names]
    
    def filter_scenarios(self, filters: Dict[str, Any]) -> List[TestScenario]:
        """
        Filter scenarios based on criteria.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            List of matching scenarios
        """
        return [
            scenario for scenario in self._scenarios.values()
            if scenario.matches_filter(filters)
        ]


# Global registry instance
scenario_registry = TestScenarioRegistry()