"""
Stress testing framework for trading strategies.

This module implements comprehensive stress testing scenarios including
extreme market conditions, flash crashes, liquidity crises, and black
swan event modeling to assess strategy robustness.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from btc_research.optimization.base import BaseRobustnessTest
from btc_research.optimization.types import RobustnessResult

__all__ = ["StressTestFramework"]


class StressTestFramework(BaseRobustnessTest):
    """
    Comprehensive stress testing framework for trading strategies.
    
    This framework subjects strategies to extreme market conditions
    and stress scenarios to assess their resilience and robustness.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize stress testing framework.
        
        Args:
            data: Historical data for stress testing
            random_seed: Random seed for reproducibility
        """
        super().__init__(data, random_seed)
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Analyze historical data for stress scenario calibration
        self._calibrate_stress_scenarios()
    
    def run_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 500,
        stress_scenarios: List[str] = None,
        **kwargs: Any,
    ) -> RobustnessResult:
        """
        Run comprehensive stress testing.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of stress simulations to run
            stress_scenarios: List of stress scenarios to test
            **kwargs: Additional test parameters
            
        Returns:
            Robustness test result with stress analysis
        """
        if stress_scenarios is None:
            stress_scenarios = [
                'flash_crash', 'liquidity_crisis', 'high_volatility',
                'trending_market', 'gap_events', 'correlation_breakdown'
            ]
        
        results = []
        
        for scenario in stress_scenarios:
            # Run simulations for each stress scenario
            scenario_results = self._run_stress_simulations(
                parameters, backtest_function, scenario,
                n_simulations // len(stress_scenarios), **kwargs
            )
            results.extend(scenario_results)
        
        # Calculate stress-specific statistics
        summary_stats = self._calculate_stress_statistics(results)
        
        # Calculate stress-adjusted risk metrics
        var_results, es_results = self._calculate_stress_risk_metrics(results)
        
        # Calculate survival rate under stress
        survival_rate = len([r for r in results if 'error' not in r]) / len(results)
        
        # Find best and worst stress scenarios
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_case = max(valid_results, key=lambda x: x.get('total_return', float('-inf')))
            worst_case = min(valid_results, key=lambda x: x.get('total_return', float('-inf')))
        else:
            best_case = {"total_return": 0.0}
            worst_case = {"total_return": 0.0}
        
        return RobustnessResult(
            test_type="stress_testing",
            n_simulations=len(results),
            results=results,
            summary_stats=summary_stats,
            value_at_risk=var_results,
            expected_shortfall=es_results,
            success_rate=survival_rate,
            worst_case_scenario=worst_case,
            best_case_scenario=best_case,
        )
    
    def run_flash_crash_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        crash_magnitudes: List[float] = None,
        recovery_times: List[int] = None,
        n_simulations: int = 100,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run flash crash stress test with various magnitudes and recovery times.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            crash_magnitudes: List of crash magnitudes (as fractions, e.g., 0.2 = 20% crash)
            recovery_times: List of recovery periods in bars
            n_simulations: Number of simulations per scenario
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with flash crash test results
        """
        if crash_magnitudes is None:
            crash_magnitudes = [0.05, 0.10, 0.20, 0.30, 0.50]  # 5% to 50% crashes
        
        if recovery_times is None:
            recovery_times = [1, 5, 20, 100]  # Immediate to slow recovery
        
        crash_results = {}
        
        for magnitude in crash_magnitudes:
            for recovery_time in recovery_times:
                scenario_key = f"crash_{magnitude:.0%}_recovery_{recovery_time}"
                scenario_results = []
                
                for i in range(n_simulations):
                    try:
                        # Generate flash crash data
                        crash_data = self._generate_flash_crash_data(magnitude, recovery_time)
                        
                        # Run backtest
                        metrics = backtest_function(crash_data, parameters)
                        
                        # Add metadata
                        metrics['simulation'] = i
                        metrics['crash_magnitude'] = magnitude
                        metrics['recovery_time'] = recovery_time
                        metrics['stress_scenario'] = 'flash_crash'
                        
                        scenario_results.append(metrics)
                        
                    except Exception as e:
                        failed_metrics = {
                            'simulation': i,
                            'crash_magnitude': magnitude,
                            'recovery_time': recovery_time,
                            'stress_scenario': 'flash_crash',
                            'error': str(e),
                            'total_return': float('-inf'),
                            'sharpe_ratio': float('-inf'),
                        }
                        scenario_results.append(failed_metrics)
                
                # Calculate scenario statistics
                scenario_stats = self._calculate_scenario_statistics(scenario_results)
                
                crash_results[scenario_key] = {
                    'results': scenario_results,
                    'statistics': scenario_stats,
                    'stress_score': self._calculate_stress_score(scenario_results),
                }
        
        # Analyze flash crash resilience
        resilience_analysis = self._analyze_flash_crash_resilience(crash_results)
        
        return {
            'crash_results': crash_results,
            'resilience_analysis': resilience_analysis,
        }
    
    def run_liquidity_crisis_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        crisis_durations: List[int] = None,
        liquidity_reductions: List[float] = None,
        n_simulations: int = 100,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run liquidity crisis stress test.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            crisis_durations: List of crisis durations in bars
            liquidity_reductions: List of liquidity reduction factors
            n_simulations: Number of simulations per scenario
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with liquidity crisis test results
        """
        if crisis_durations is None:
            crisis_durations = [10, 50, 200, 500]  # Short to extended crises
        
        if liquidity_reductions is None:
            liquidity_reductions = [0.5, 0.2, 0.1, 0.05]  # 50% to 95% liquidity reduction
        
        crisis_results = {}
        
        for duration in crisis_durations:
            for reduction in liquidity_reductions:
                scenario_key = f"crisis_{duration}bars_liquidity_{reduction:.0%}"
                scenario_results = []
                
                for i in range(n_simulations):
                    try:
                        # Generate liquidity crisis data
                        crisis_data = self._generate_liquidity_crisis_data(duration, reduction)
                        
                        # Run backtest
                        metrics = backtest_function(crisis_data, parameters)
                        
                        # Add metadata
                        metrics['simulation'] = i
                        metrics['crisis_duration'] = duration
                        metrics['liquidity_reduction'] = reduction
                        metrics['stress_scenario'] = 'liquidity_crisis'
                        
                        scenario_results.append(metrics)
                        
                    except Exception as e:
                        failed_metrics = {
                            'simulation': i,
                            'crisis_duration': duration,
                            'liquidity_reduction': reduction,
                            'stress_scenario': 'liquidity_crisis',
                            'error': str(e),
                            'total_return': float('-inf'),
                            'sharpe_ratio': float('-inf'),
                        }
                        scenario_results.append(failed_metrics)
                
                # Calculate scenario statistics
                scenario_stats = self._calculate_scenario_statistics(scenario_results)
                
                crisis_results[scenario_key] = {
                    'results': scenario_results,
                    'statistics': scenario_stats,
                    'stress_score': self._calculate_stress_score(scenario_results),
                }
        
        return crisis_results
    
    def run_black_swan_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        event_types: List[str] = None,
        n_simulations: int = 200,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run black swan event stress test.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            event_types: List of black swan event types
            n_simulations: Number of simulations per event type
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with black swan test results
        """
        if event_types is None:
            event_types = [
                'market_crash', 'currency_crisis', 'regulatory_shock',
                'technology_failure', 'geopolitical_crisis'
            ]
        
        black_swan_results = {}
        
        for event_type in event_types:
            event_results = []
            
            for i in range(n_simulations):
                try:
                    # Generate black swan event data
                    event_data = self._generate_black_swan_event(event_type)
                    
                    # Run backtest
                    metrics = backtest_function(event_data, parameters)
                    
                    # Add metadata
                    metrics['simulation'] = i
                    metrics['event_type'] = event_type
                    metrics['stress_scenario'] = 'black_swan'
                    
                    event_results.append(metrics)
                    
                except Exception as e:
                    failed_metrics = {
                        'simulation': i,
                        'event_type': event_type,
                        'stress_scenario': 'black_swan',
                        'error': str(e),
                        'total_return': float('-inf'),
                        'sharpe_ratio': float('-inf'),
                    }
                    event_results.append(failed_metrics)
            
            # Calculate event statistics
            event_stats = self._calculate_scenario_statistics(event_results)
            
            black_swan_results[event_type] = {
                'results': event_results,
                'statistics': event_stats,
                'tail_risk_score': self._calculate_tail_risk_score(event_results),
            }
        
        return black_swan_results
    
    def run_correlation_breakdown_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        breakdown_scenarios: List[str] = None,
        n_simulations: int = 150,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run correlation breakdown stress test.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            breakdown_scenarios: List of correlation breakdown scenarios
            n_simulations: Number of simulations per scenario
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with correlation breakdown test results
        """
        if breakdown_scenarios is None:
            breakdown_scenarios = [
                'trend_reversal', 'volatility_spike', 'regime_change',
                'market_decoupling', 'safe_haven_failure'
            ]
        
        breakdown_results = {}
        
        for scenario in breakdown_scenarios:
            scenario_results = []
            
            for i in range(n_simulations):
                try:
                    # Generate correlation breakdown data
                    breakdown_data = self._generate_correlation_breakdown_data(scenario)
                    
                    # Run backtest
                    metrics = backtest_function(breakdown_data, parameters)
                    
                    # Add metadata
                    metrics['simulation'] = i
                    metrics['breakdown_scenario'] = scenario
                    metrics['stress_scenario'] = 'correlation_breakdown'
                    
                    scenario_results.append(metrics)
                    
                except Exception as e:
                    failed_metrics = {
                        'simulation': i,
                        'breakdown_scenario': scenario,
                        'stress_scenario': 'correlation_breakdown',
                        'error': str(e),
                        'total_return': float('-inf'),
                        'sharpe_ratio': float('-inf'),
                    }
                    scenario_results.append(failed_metrics)
            
            # Calculate scenario statistics
            scenario_stats = self._calculate_scenario_statistics(scenario_results)
            
            breakdown_results[scenario] = {
                'results': scenario_results,
                'statistics': scenario_stats,
                'adaptation_score': self._calculate_adaptation_score(scenario_results),
            }
        
        return breakdown_results
    
    def _calibrate_stress_scenarios(self) -> None:
        """
        Calibrate stress scenarios based on historical data characteristics.
        """
        if 'close' in self.data.columns:
            prices = self.data['close']
            returns = prices.pct_change().dropna()
            
            # Calculate historical statistics for calibration
            self.historical_volatility = returns.std()
            self.historical_mean_return = returns.mean()
            self.historical_drawdowns = self._calculate_drawdowns(prices)
            self.max_historical_drawdown = max(self.historical_drawdowns) if self.historical_drawdowns else 0.1
            
            # Identify extreme events in historical data
            self.extreme_events = self._identify_extreme_events(returns)
            
        else:
            # Default calibration values
            self.historical_volatility = 0.02
            self.historical_mean_return = 0.0005
            self.max_historical_drawdown = 0.2
            self.extreme_events = {'crashes': [], 'spikes': []}
        
        # Volume characteristics
        if 'volume' in self.data.columns:
            self.volume_stats = {
                'mean': self.data['volume'].mean(),
                'std': self.data['volume'].std(),
                'median': self.data['volume'].median(),
            }
        else:
            self.volume_stats = {'mean': 1000, 'std': 500, 'median': 1000}
    
    def _run_stress_simulations(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        stress_scenario: str,
        n_simulations: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Run simulations for a specific stress scenario.
        """
        results = []
        
        for i in range(n_simulations):
            try:
                # Generate stress scenario data
                if stress_scenario == 'flash_crash':
                    stress_data = self._generate_flash_crash_data()
                elif stress_scenario == 'liquidity_crisis':
                    stress_data = self._generate_liquidity_crisis_data()
                elif stress_scenario == 'high_volatility':
                    stress_data = self._generate_high_volatility_data()
                elif stress_scenario == 'trending_market':
                    stress_data = self._generate_trending_market_data()
                elif stress_scenario == 'gap_events':
                    stress_data = self._generate_gap_events_data()
                elif stress_scenario == 'correlation_breakdown':
                    stress_data = self._generate_correlation_breakdown_data()
                else:
                    raise ValueError(f"Unknown stress scenario: {stress_scenario}")
                
                # Run backtest
                metrics = backtest_function(stress_data, parameters)
                
                # Add metadata
                metrics['simulation'] = i
                metrics['stress_scenario'] = stress_scenario
                
                results.append(metrics)
                
            except Exception as e:
                failed_metrics = {
                    'simulation': i,
                    'stress_scenario': stress_scenario,
                    'error': str(e),
                    'total_return': float('-inf'),
                    'sharpe_ratio': float('-inf'),
                }
                results.append(failed_metrics)
        
        return results
    
    def _generate_flash_crash_data(
        self, 
        crash_magnitude: float = None,
        recovery_time: int = None
    ) -> pd.DataFrame:
        """
        Generate data with flash crash events.
        """
        if crash_magnitude is None:
            crash_magnitude = np.random.uniform(0.1, 0.4)  # 10-40% crash
        
        if recovery_time is None:
            recovery_time = np.random.randint(1, 50)  # 1-50 bars recovery
        
        crash_data = self.data.copy()
        
        # Randomly place crash event
        crash_start = np.random.randint(len(crash_data) // 4, 3 * len(crash_data) // 4)
        crash_end = min(crash_start + recovery_time, len(crash_data))
        
        if 'close' in crash_data.columns:
            # Create crash pattern
            pre_crash_price = crash_data['close'].iloc[crash_start]
            crash_bottom = pre_crash_price * (1 - crash_magnitude)
            
            # Immediate crash
            crash_data.iloc[crash_start, crash_data.columns.get_loc('close')] = crash_bottom
            
            # Recovery pattern
            recovery_pattern = np.linspace(crash_bottom, pre_crash_price, recovery_time)
            
            for i, recovery_price in enumerate(recovery_pattern):
                if crash_start + i < len(crash_data):
                    for col in ['open', 'high', 'low', 'close']:
                        if col in crash_data.columns:
                            current_price = crash_data.iloc[crash_start + i, crash_data.columns.get_loc(col)]
                            # Scale price based on recovery
                            price_ratio = recovery_price / crash_bottom
                            crash_data.iloc[crash_start + i, crash_data.columns.get_loc(col)] = current_price * price_ratio
            
            # Ensure OHLC consistency
            self._ensure_ohlc_consistency(crash_data)
        
        # Simulate volume spike during crash
        if 'volume' in crash_data.columns:
            volume_spike = np.random.uniform(5.0, 20.0)
            crash_data.iloc[crash_start:crash_start+5, crash_data.columns.get_loc('volume')] *= volume_spike
        
        return crash_data
    
    def _generate_liquidity_crisis_data(
        self,
        crisis_duration: int = None,
        liquidity_reduction: float = None
    ) -> pd.DataFrame:
        """
        Generate data with liquidity crisis.
        """
        if crisis_duration is None:
            crisis_duration = np.random.randint(20, 200)
        
        if liquidity_reduction is None:
            liquidity_reduction = np.random.uniform(0.1, 0.5)  # 50-90% liquidity reduction
        
        crisis_data = self.data.copy()
        
        # Randomly place crisis
        crisis_start = np.random.randint(0, len(crisis_data) - crisis_duration)
        crisis_end = crisis_start + crisis_duration
        
        # Reduce volume (proxy for liquidity)
        if 'volume' in crisis_data.columns:
            crisis_data.iloc[crisis_start:crisis_end, crisis_data.columns.get_loc('volume')] *= liquidity_reduction
        
        # Increase bid-ask spreads (simulate with price volatility)
        spread_multiplier = 1.0 / liquidity_reduction  # Lower liquidity = higher spreads
        
        for col in ['open', 'high', 'low', 'close']:
            if col in crisis_data.columns:
                crisis_noise = np.random.normal(0, self.historical_volatility * spread_multiplier * 0.5, crisis_duration)
                crisis_data.iloc[crisis_start:crisis_end, crisis_data.columns.get_loc(col)] *= (1 + crisis_noise)
        
        # Ensure OHLC consistency
        self._ensure_ohlc_consistency(crisis_data)
        
        return crisis_data
    
    def _generate_high_volatility_data(self) -> pd.DataFrame:
        """
        Generate data with sustained high volatility.
        """
        volatility_data = self.data.copy()
        
        # Increase volatility by 3-10x
        volatility_multiplier = np.random.uniform(3.0, 10.0)
        
        if 'close' in volatility_data.columns:
            # Calculate enhanced returns
            returns = volatility_data['close'].pct_change()
            enhanced_returns = returns * volatility_multiplier
            
            # Reconstruct prices
            initial_price = volatility_data['close'].iloc[0]
            new_prices = [initial_price]
            
            for ret in enhanced_returns.iloc[1:]:
                if not np.isnan(ret):
                    new_price = new_prices[-1] * (1 + ret)
                else:
                    new_price = new_prices[-1]
                new_prices.append(new_price)
            
            # Update all price columns proportionally
            price_ratio = np.array(new_prices) / volatility_data['close'].values
            
            for col in ['open', 'high', 'low', 'close']:
                if col in volatility_data.columns:
                    volatility_data[col] *= price_ratio
        
        # Increase volume variability
        if 'volume' in volatility_data.columns:
            volume_noise = np.random.normal(1.0, 0.5, len(volatility_data))
            volume_noise = np.maximum(volume_noise, 0.1)  # Ensure positive
            volatility_data['volume'] *= volume_noise
        
        return volatility_data
    
    def _generate_trending_market_data(self) -> pd.DataFrame:
        """
        Generate data with strong trending behavior.
        """
        trending_data = self.data.copy()
        
        # Random trend direction and strength
        trend_direction = np.random.choice([-1, 1])
        trend_strength = np.random.uniform(0.001, 0.005)  # Daily trend
        
        # Generate trend
        trend = np.cumsum(np.random.normal(trend_direction * trend_strength, trend_strength * 0.3, len(trending_data)))
        
        if 'close' in trending_data.columns:
            for col in ['open', 'high', 'low', 'close']:
                if col in trending_data.columns:
                    trending_data[col] *= (1 + trend)
        
        return trending_data
    
    def _generate_gap_events_data(self) -> pd.DataFrame:
        """
        Generate data with gap events (overnight gaps).
        """
        gap_data = self.data.copy()
        
        # Create 3-8 gap events
        n_gaps = np.random.randint(3, 9)
        
        for _ in range(n_gaps):
            gap_position = np.random.randint(1, len(gap_data))
            gap_magnitude = np.random.uniform(-0.05, 0.05)  # ±5% gaps
            
            # Apply gap to all price columns
            for col in ['open', 'high', 'low', 'close']:
                if col in gap_data.columns:
                    gap_data.iloc[gap_position:, gap_data.columns.get_loc(col)] *= (1 + gap_magnitude)
        
        return gap_data
    
    def _generate_black_swan_event(self, event_type: str) -> pd.DataFrame:
        """
        Generate black swan event data.
        """
        event_data = self.data.copy()
        
        if event_type == 'market_crash':
            # Severe market crash (30-70% decline)
            crash_magnitude = np.random.uniform(0.3, 0.7)
            return self._generate_flash_crash_data(crash_magnitude, recovery_time=100)
            
        elif event_type == 'currency_crisis':
            # Currency devaluation pattern
            devaluation = np.random.uniform(0.2, 0.6)
            crisis_start = np.random.randint(0, len(event_data) // 2)
            
            # Gradual then sudden devaluation
            gradual_phase = np.random.randint(10, 50)
            sudden_phase = np.random.randint(5, 20)
            
            # Gradual decline
            gradual_decline = np.linspace(0, -devaluation * 0.3, gradual_phase)
            # Sudden crash
            sudden_decline = np.linspace(-devaluation * 0.3, -devaluation, sudden_phase)
            
            combined_decline = np.concatenate([gradual_decline, sudden_decline])
            
            for i, decline in enumerate(combined_decline):
                if crisis_start + i < len(event_data):
                    for col in ['open', 'high', 'low', 'close']:
                        if col in event_data.columns:
                            event_data.iloc[crisis_start + i, event_data.columns.get_loc(col)] *= (1 + decline)
            
        elif event_type == 'regulatory_shock':
            # Sudden regulatory announcement causing sharp movement
            shock_magnitude = np.random.uniform(-0.4, -0.1)  # Negative shock
            shock_position = np.random.randint(0, len(event_data) - 20)
            
            # Immediate price impact
            for col in ['open', 'high', 'low', 'close']:
                if col in event_data.columns:
                    event_data.iloc[shock_position:, event_data.columns.get_loc(col)] *= (1 + shock_magnitude)
            
            # Increased volatility following shock
            volatility_period = 50
            for i in range(volatility_period):
                if shock_position + i < len(event_data):
                    vol_factor = np.random.normal(1.0, 0.1)
                    for col in ['open', 'high', 'low', 'close']:
                        if col in event_data.columns:
                            event_data.iloc[shock_position + i, event_data.columns.get_loc(col)] *= vol_factor
            
        elif event_type == 'technology_failure':
            # Trading halt and extreme volatility
            failure_start = np.random.randint(0, len(event_data) - 30)
            failure_duration = np.random.randint(5, 30)
            
            # Zero volume during failure (trading halt)
            if 'volume' in event_data.columns:
                event_data.iloc[failure_start:failure_start+failure_duration, event_data.columns.get_loc('volume')] = 0
            
            # Extreme volatility when trading resumes
            resume_volatility = 20  # 20x normal volatility
            for i in range(failure_duration, failure_duration + 20):
                if failure_start + i < len(event_data):
                    vol_shock = np.random.normal(0, self.historical_volatility * resume_volatility)
                    for col in ['open', 'high', 'low', 'close']:
                        if col in event_data.columns:
                            event_data.iloc[failure_start + i, event_data.columns.get_loc(col)] *= (1 + vol_shock)
            
        elif event_type == 'geopolitical_crisis':
            # Safe haven flows and correlation breakdown
            crisis_magnitude = np.random.uniform(-0.3, 0.3)  # Can be positive or negative
            crisis_start = np.random.randint(0, len(event_data) - 100)
            crisis_duration = np.random.randint(50, 200)
            
            # Gradual price movement
            crisis_pattern = np.linspace(0, crisis_magnitude, crisis_duration)
            
            for i, movement in enumerate(crisis_pattern):
                if crisis_start + i < len(event_data):
                    for col in ['open', 'high', 'low', 'close']:
                        if col in event_data.columns:
                            event_data.iloc[crisis_start + i, event_data.columns.get_loc(col)] *= (1 + movement)
        
        # Ensure OHLC consistency
        self._ensure_ohlc_consistency(event_data)
        
        return event_data
    
    def _generate_correlation_breakdown_data(self, scenario: str = None) -> pd.DataFrame:
        """
        Generate data with correlation breakdown scenarios.
        """
        if scenario is None:
            scenario = np.random.choice([
                'trend_reversal', 'volatility_spike', 'regime_change',
                'market_decoupling', 'safe_haven_failure'
            ])
        
        breakdown_data = self.data.copy()
        
        if scenario == 'trend_reversal':
            # Sudden trend reversal
            reversal_point = np.random.randint(len(breakdown_data) // 3, 2 * len(breakdown_data) // 3)
            
            # Calculate trend before reversal
            if 'close' in breakdown_data.columns:
                pre_reversal_trend = (breakdown_data['close'].iloc[reversal_point] - 
                                    breakdown_data['close'].iloc[0]) / reversal_point
                
                # Apply opposite trend after reversal
                post_reversal_bars = len(breakdown_data) - reversal_point
                reverse_trend = np.linspace(0, -pre_reversal_trend * post_reversal_bars * 2, post_reversal_bars)
                
                for i, trend_adj in enumerate(reverse_trend):
                    if reversal_point + i < len(breakdown_data):
                        for col in ['open', 'high', 'low', 'close']:
                            if col in breakdown_data.columns:
                                breakdown_data.iloc[reversal_point + i, breakdown_data.columns.get_loc(col)] *= (1 + trend_adj)
        
        elif scenario == 'volatility_spike':
            # Sudden volatility increase
            spike_start = np.random.randint(0, len(breakdown_data) - 50)
            spike_duration = np.random.randint(20, 100)
            spike_magnitude = np.random.uniform(5.0, 15.0)
            
            # Apply volatility spike
            for i in range(spike_duration):
                if spike_start + i < len(breakdown_data):
                    vol_shock = np.random.normal(0, self.historical_volatility * spike_magnitude)
                    for col in ['open', 'high', 'low', 'close']:
                        if col in breakdown_data.columns:
                            breakdown_data.iloc[spike_start + i, breakdown_data.columns.get_loc(col)] *= (1 + vol_shock)
        
        elif scenario == 'regime_change':
            # Market regime change
            change_point = np.random.randint(len(breakdown_data) // 3, 2 * len(breakdown_data) // 3)
            
            # Different return and volatility characteristics after change
            new_mean_return = self.historical_mean_return * np.random.uniform(-3, 3)
            new_volatility = self.historical_volatility * np.random.uniform(0.5, 3.0)
            
            if 'close' in breakdown_data.columns:
                # Generate new price path from change point
                post_change_returns = np.random.normal(new_mean_return, new_volatility, 
                                                     len(breakdown_data) - change_point)
                
                current_price = breakdown_data['close'].iloc[change_point]
                for i, ret in enumerate(post_change_returns):
                    if change_point + i + 1 < len(breakdown_data):
                        new_price = current_price * (1 + ret)
                        price_ratio = new_price / breakdown_data['close'].iloc[change_point + i + 1]
                        
                        for col in ['open', 'high', 'low', 'close']:
                            if col in breakdown_data.columns:
                                breakdown_data.iloc[change_point + i + 1, breakdown_data.columns.get_loc(col)] *= price_ratio
                        
                        current_price = new_price
        
        # Ensure OHLC consistency
        self._ensure_ohlc_consistency(breakdown_data)
        
        return breakdown_data
    
    def _calculate_drawdowns(self, prices: pd.Series) -> List[float]:
        """
        Calculate drawdowns from price series.
        """
        rolling_max = prices.expanding().max()
        drawdowns = (prices - rolling_max) / rolling_max
        return drawdowns.tolist()
    
    def _identify_extreme_events(self, returns: pd.Series) -> Dict[str, List[int]]:
        """
        Identify extreme events in historical returns.
        """
        # Define extreme thresholds (3+ standard deviations)
        threshold = 3 * returns.std()
        
        crashes = returns[returns < -threshold].index.tolist()
        spikes = returns[returns > threshold].index.tolist()
        
        return {'crashes': crashes, 'spikes': spikes}
    
    def _ensure_ohlc_consistency(self, data: pd.DataFrame) -> None:
        """
        Ensure OHLC price consistency after modifications.
        """
        price_columns = ['open', 'high', 'low', 'close']
        
        if all(col in data.columns for col in price_columns):
            # Ensure high >= max(open, close) and low <= min(open, close)
            data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
            data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
            
            # Ensure all prices are positive
            for col in price_columns:
                data[col] = np.maximum(data[col], 0.01)
    
    def _calculate_stress_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate stress-specific statistics.
        """
        summary_stats = {}
        
        # Group results by stress scenario
        stress_groups = {}
        for result in results:
            scenario = result.get('stress_scenario', 'unknown')
            if scenario not in stress_groups:
                stress_groups[scenario] = []
            stress_groups[scenario].append(result)
        
        # Calculate statistics for each stress scenario
        for scenario, group_results in stress_groups.items():
            group_stats = self._calculate_scenario_statistics(group_results)
            
            # Add stress-specific metrics
            valid_results = [r for r in group_results if 'error' not in r]
            if valid_results:
                returns = [r.get('total_return', 0) for r in valid_results]
                
                group_stats['stress_resilience'] = 1.0 / (1.0 + abs(np.min(returns))) if returns else 0
                group_stats['recovery_potential'] = max(0, np.mean(returns)) if returns else 0
                group_stats['tail_risk'] = abs(np.percentile(returns, 5)) if returns else 0
            
            summary_stats[scenario] = group_stats
        
        return summary_stats
    
    def _calculate_stress_risk_metrics(self, results: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate stress-adjusted risk metrics.
        """
        var_results, es_results = self._calculate_risk_metrics(results)
        
        # Add stress-specific risk metrics
        valid_results = [r for r in results if 'error' not in r]
        
        if valid_results:
            returns = [r.get('total_return', 0) for r in valid_results]
            
            # Stress-adjusted VaR (more conservative)
            stress_var_95 = np.percentile(returns, 2.5)  # 97.5% VaR instead of 95%
            stress_var_99 = np.percentile(returns, 0.5)  # 99.5% VaR instead of 99%
            
            var_results['stress_adjusted_var_95'] = stress_var_95
            var_results['stress_adjusted_var_99'] = stress_var_99
            
            # Maximum stress loss
            max_stress_loss = np.min(returns)
            var_results['maximum_stress_loss'] = max_stress_loss
        
        return var_results, es_results
    
    def _calculate_scenario_statistics(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate statistics for a single stress scenario.
        """
        valid_results = [r for r in scenario_results if 'error' not in r]
        
        if not valid_results:
            return {
                'success_rate': 0.0,
                'mean_return': float('-inf'),
                'std_return': 0.0,
                'survival_rate': 0.0
            }
        
        returns = [r.get('total_return', 0) for r in valid_results]
        
        stats = {
            'success_rate': len(valid_results) / len(scenario_results),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'median_return': np.median(returns),
            'survival_rate': sum(1 for r in returns if r > -0.5) / len(returns),  # Survive >50% loss
        }
        
        return stats
    
    def _calculate_stress_score(self, scenario_results: List[Dict[str, Any]]) -> float:
        """
        Calculate stress score for a scenario (0-1, higher is better resilience).
        """
        valid_results = [r for r in scenario_results if 'error' not in r]
        
        if not valid_results:
            return 0.0
        
        returns = [r.get('total_return', 0) for r in valid_results]
        
        # Components of stress score
        survival_rate = sum(1 for r in returns if r > -0.5) / len(returns)
        mean_performance = max(0, np.mean(returns) + 0.5)  # Normalize around -50%
        consistency = 1.0 / (1.0 + np.std(returns))
        
        # Weighted stress score
        stress_score = (0.4 * survival_rate + 0.4 * mean_performance + 0.2 * consistency)
        
        return min(1.0, max(0.0, stress_score))
    
    def _calculate_tail_risk_score(self, event_results: List[Dict[str, Any]]) -> float:
        """
        Calculate tail risk score for black swan events.
        """
        valid_results = [r for r in event_results if 'error' not in r]
        
        if not valid_results:
            return 1.0  # Maximum tail risk
        
        returns = [r.get('total_return', 0) for r in valid_results]
        
        # Tail risk components
        worst_case = np.min(returns)
        tail_mean = np.mean([r for r in returns if r <= np.percentile(returns, 10)])
        extreme_event_frequency = sum(1 for r in returns if r < -0.3) / len(returns)
        
        # Normalize and combine
        tail_risk_score = (
            0.4 * abs(worst_case) +
            0.4 * abs(tail_mean) +
            0.2 * extreme_event_frequency
        )
        
        return min(1.0, tail_risk_score)
    
    def _calculate_adaptation_score(self, scenario_results: List[Dict[str, Any]]) -> float:
        """
        Calculate adaptation score for correlation breakdown scenarios.
        """
        valid_results = [r for r in scenario_results if 'error' not in r]
        
        if not valid_results:
            return 0.0
        
        returns = [r.get('total_return', 0) for r in valid_results]
        
        # Adaptation components
        positive_adaptation = sum(1 for r in returns if r > 0) / len(returns)
        performance_stability = 1.0 / (1.0 + np.std(returns))
        mean_adaptation = max(0, np.mean(returns))
        
        # Weighted adaptation score
        adaptation_score = (0.4 * positive_adaptation + 0.3 * performance_stability + 0.3 * mean_adaptation)
        
        return min(1.0, adaptation_score)
    
    def _analyze_flash_crash_resilience(self, crash_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze flash crash resilience across different scenarios.
        """
        resilience_metrics = {}
        
        # Extract crash magnitudes and recovery times
        magnitudes = []
        recovery_times = []
        stress_scores = []
        
        for scenario_key, scenario_data in crash_results.items():
            # Parse scenario key to extract parameters
            parts = scenario_key.split('_')
            if len(parts) >= 4:
                magnitude = float(parts[1].replace('%', '')) / 100
                recovery_time = int(parts[3])
                
                magnitudes.append(magnitude)
                recovery_times.append(recovery_time)
                stress_scores.append(scenario_data['stress_score'])
        
        if magnitudes and recovery_times and stress_scores:
            # Calculate resilience thresholds
            resilience_metrics['crash_tolerance'] = max([m for m, s in zip(magnitudes, stress_scores) if s > 0.7], default=0.0)
            resilience_metrics['recovery_efficiency'] = np.mean([s / rt for s, rt in zip(stress_scores, recovery_times)])
            resilience_metrics['overall_resilience'] = np.mean(stress_scores)
            
            # Identify weakest scenarios
            min_score_idx = np.argmin(stress_scores)
            resilience_metrics['weakest_scenario'] = {
                'magnitude': magnitudes[min_score_idx],
                'recovery_time': recovery_times[min_score_idx],
                'stress_score': stress_scores[min_score_idx]
            }
        
        return resilience_metrics