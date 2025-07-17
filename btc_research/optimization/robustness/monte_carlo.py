"""
Monte Carlo robustness testing for trading strategies.

Monte Carlo testing assesses strategy robustness by running simulations
with perturbed data, randomized parameters, or varied market conditions
to understand the distribution of possible outcomes.
"""

import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from btc_research.optimization.base import BaseRobustnessTest
from btc_research.optimization.types import RobustnessResult

__all__ = ["MonteCarloRobustnessTest"]


class MonteCarloRobustnessTest(BaseRobustnessTest):
    """
    Monte Carlo robustness testing implementation.
    
    This test runs multiple simulations with different data perturbations
    or parameter variations to assess strategy stability and robustness.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        random_seed: Optional[int] = None,
        noise_level: float = 0.01,
        parameter_variance: float = 0.05,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize Monte Carlo robustness test.
        
        Args:
            data: Historical data for testing
            random_seed: Random seed for reproducibility
            noise_level: Standard deviation of noise to add to price data
            parameter_variance: Relative variance for parameter perturbation
            enable_parallel: Enable parallel execution for simulations
            max_workers: Maximum number of workers for parallel execution
        """
        super().__init__(data, random_seed)
        
        self.noise_level = noise_level
        self.parameter_variance = parameter_variance
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            
        # Pre-calculate trade returns for resampling tests
        self._calculate_trade_statistics()
    
    def run_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 1000,
        test_type: str = "data_noise",
        success_threshold: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> RobustnessResult:
        """
        Run Monte Carlo robustness test.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of simulations to run
            test_type: Type of test ('data_noise', 'parameter_variation', 'bootstrap')
            success_threshold: Minimum acceptable values for each metric
            **kwargs: Additional test parameters
            
        Returns:
            Robustness test result
        """
        if success_threshold is None:
            success_threshold = {"total_return": 0.0, "sharpe_ratio": 0.0}
        
        results = []
        successful_simulations = 0
        
        for i in range(n_simulations):
            try:
                if test_type == "data_noise":
                    # Add noise to price data
                    perturbed_data = self._add_price_noise(self.data)
                    sim_params = parameters.copy()
                    
                elif test_type == "parameter_variation":
                    # Vary parameters slightly
                    perturbed_data = self.data.copy()
                    sim_params = self._perturb_parameters(parameters)
                    
                elif test_type == "bootstrap":
                    # Bootstrap resampling of data
                    perturbed_data = self._bootstrap_resample(self.data)
                    sim_params = parameters.copy()
                    
                else:
                    raise ValueError(f"Unknown test type: {test_type}")
                
                # Run backtest on perturbed data/parameters
                metrics = backtest_function(perturbed_data, sim_params)
                
                # Add simulation info
                metrics['simulation'] = i
                metrics['test_type'] = test_type
                
                results.append(metrics)
                
                # Check if simulation meets success criteria
                meets_criteria = all(
                    metrics.get(metric, float('-inf')) >= threshold
                    for metric, threshold in success_threshold.items()
                )
                
                if meets_criteria:
                    successful_simulations += 1
                
            except Exception as e:
                # Record failed simulation
                failed_metrics = {
                    'simulation': i,
                    'test_type': test_type,
                    'error': str(e),
                    'total_return': float('-inf'),
                    'sharpe_ratio': float('-inf'),
                }
                results.append(failed_metrics)
        
        if not results:
            raise ValueError("All Monte Carlo simulations failed")
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(results)
        
        # Calculate risk metrics
        var_results, es_results = self._calculate_risk_metrics(results)
        
        # Calculate success rate
        success_rate = successful_simulations / n_simulations
        
        # Find best and worst case scenarios
        valid_results = [r for r in results if 'error' not in r]
        
        if valid_results:
            if 'total_return' in valid_results[0]:
                best_case = max(valid_results, key=lambda x: x.get('total_return', float('-inf')))
                worst_case = min(valid_results, key=lambda x: x.get('total_return', float('-inf')))
            else:
                best_case = valid_results[0]
                worst_case = valid_results[0]
        else:
            best_case = {"total_return": 0.0}
            worst_case = {"total_return": 0.0}
        
        return RobustnessResult(
            test_type=test_type,
            n_simulations=n_simulations,
            results=results,
            summary_stats=summary_stats,
            value_at_risk=var_results,
            expected_shortfall=es_results,
            success_rate=success_rate,
            worst_case_scenario=worst_case,
            best_case_scenario=best_case,
        )
    
    def _add_price_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add random noise to OHLC price data.
        
        Args:
            data: Original price data
            
        Returns:
            Data with added noise
        """
        perturbed_data = data.copy()
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col in perturbed_data.columns:
                # Add proportional noise to prices
                prices = perturbed_data[col]
                noise = np.random.normal(0, self.noise_level, len(prices))
                perturbed_prices = prices * (1 + noise)
                
                # Ensure positive prices
                perturbed_prices = np.maximum(perturbed_prices, prices * 0.1)
                perturbed_data[col] = perturbed_prices
        
        # Ensure OHLC consistency (high >= max(open, close), low <= min(open, close))
        if all(col in perturbed_data.columns for col in price_columns):
            perturbed_data['high'] = np.maximum(
                perturbed_data['high'], 
                np.maximum(perturbed_data['open'], perturbed_data['close'])
            )
            perturbed_data['low'] = np.minimum(
                perturbed_data['low'],
                np.minimum(perturbed_data['open'], perturbed_data['close'])
            )
        
        return perturbed_data
    
    def _perturb_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add random variations to parameter values.
        
        Args:
            parameters: Original parameter values
            
        Returns:
            Parameters with added variations
        """
        perturbed_params = {}
        
        for param_name, param_value in parameters.items():
            if isinstance(param_value, (int, float)):
                # Add proportional noise to numeric parameters
                noise = np.random.normal(0, self.parameter_variance)
                new_value = param_value * (1 + noise)
                
                # Keep same type and ensure reasonable bounds
                if isinstance(param_value, int):
                    new_value = max(1, int(round(new_value)))
                else:
                    new_value = max(0.001, new_value)  # Ensure positive for float params
                
                perturbed_params[param_name] = new_value
            else:
                # Keep non-numeric parameters unchanged
                perturbed_params[param_name] = param_value
        
        return perturbed_params
    
    def _bootstrap_resample(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create bootstrap resample of the data.
        
        Args:
            data: Original data
            
        Returns:
            Bootstrap resampled data
        """
        n_samples = len(data)
        
        # Sample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_data = data.iloc[bootstrap_indices].copy()
        
        # Reset index to maintain chronological order for the resampled data
        bootstrap_data = bootstrap_data.reset_index(drop=True)
        
        # If data has datetime index, create new datetime range
        if isinstance(data.index, pd.DatetimeIndex):
            start_date = data.index.min()
            freq = pd.infer_freq(data.index)
            if freq:
                new_index = pd.date_range(start=start_date, periods=len(bootstrap_data), freq=freq)
                bootstrap_data.index = new_index
        
        return bootstrap_data
    
    def _calculate_summary_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for simulation results.
        
        Args:
            results: List of simulation results
            
        Returns:
            Dictionary of summary statistics by metric
        """
        summary_stats = {}
        
        # Get all numeric metrics
        all_metrics = set()
        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ['simulation']:
                    all_metrics.add(key)
        
        for metric in all_metrics:
            values = []
            for result in results:
                if metric in result and isinstance(result[metric], (int, float)):
                    if not (np.isnan(result[metric]) or np.isinf(result[metric])):
                        values.append(result[metric])
            
            if values:
                summary_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'count': len(values),
                }
            else:
                summary_stats[metric] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'median': 0.0, 'q25': 0.0, 'q75': 0.0, 'count': 0,
                }
        
        return summary_stats
    
    def run_stress_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        stress_scenarios: List[Dict[str, Any]],
    ) -> Dict[str, RobustnessResult]:
        """
        Run stress tests under specific scenarios.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            stress_scenarios: List of stress test scenarios
            
        Returns:
            Dictionary of stress test results by scenario name
        """
        stress_results = {}
        
        for scenario in stress_scenarios:
            scenario_name = scenario.get('name', 'unnamed_scenario')
            
            try:
                # Apply stress scenario
                stressed_data = self._apply_stress_scenario(self.data, scenario)
                
                # Run single backtest under stress
                metrics = backtest_function(stressed_data, parameters)
                
                # Wrap in RobustnessResult format
                stress_result = RobustnessResult(
                    test_type=f"stress_test_{scenario_name}",
                    n_simulations=1,
                    results=[metrics],
                    summary_stats={k: {'mean': v, 'std': 0.0} for k, v in metrics.items()},
                    value_at_risk={},
                    expected_shortfall={},
                    success_rate=1.0,
                    worst_case_scenario=metrics,
                    best_case_scenario=metrics,
                )
                
                stress_results[scenario_name] = stress_result
                
            except Exception as e:
                print(f"Stress test '{scenario_name}' failed: {e}")
                continue
        
        return stress_results
    
    def _apply_stress_scenario(self, data: pd.DataFrame, scenario: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a stress scenario to the data.
        
        Args:
            data: Original data
            scenario: Stress scenario specification
            
        Returns:
            Data with stress scenario applied
        """
        stressed_data = data.copy()
        
        # Market crash scenario
        if scenario.get('type') == 'market_crash':
            crash_magnitude = scenario.get('magnitude', 0.2)  # 20% crash
            crash_start = scenario.get('start_pct', 0.5)  # Start at 50% of data
            crash_duration = scenario.get('duration_pct', 0.1)  # Last 10% of data
            
            start_idx = int(len(data) * crash_start)
            end_idx = int(start_idx + len(data) * crash_duration)
            
            # Apply crash to price columns
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in stressed_data.columns:
                    stressed_data.iloc[start_idx:end_idx, stressed_data.columns.get_loc(col)] *= (1 - crash_magnitude)
        
        # High volatility scenario
        elif scenario.get('type') == 'high_volatility':
            volatility_multiplier = scenario.get('multiplier', 3.0)
            
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in stressed_data.columns:
                    prices = stressed_data[col]
                    returns = prices.pct_change()
                    stressed_returns = returns * volatility_multiplier
                    
                    # Reconstruct prices
                    stressed_prices = [prices.iloc[0]]
                    for ret in stressed_returns.iloc[1:]:
                        if not np.isnan(ret):
                            new_price = stressed_prices[-1] * (1 + ret)
                        else:
                            new_price = stressed_prices[-1]
                        stressed_prices.append(new_price)
                    
                    stressed_data[col] = stressed_prices
        
        return stressed_data
    
    def _calculate_trade_statistics(self) -> None:
        """
        Pre-calculate trade statistics for trade sequence resampling.
        """
        # Calculate returns for trade sequence analysis
        if 'close' in self.data.columns:
            self.returns = self.data['close'].pct_change().dropna()
        else:
            self.returns = pd.Series(dtype=float)
            
        # Calculate volatility metrics
        if len(self.returns) > 1:
            self.historical_volatility = self.returns.std()
            self.mean_return = self.returns.mean()
        else:
            self.historical_volatility = 0.0
            self.mean_return = 0.0
    
    def run_trade_sequence_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 1000,
        block_size: Optional[int] = None,
        preserve_correlations: bool = True,
        **kwargs: Any,
    ) -> RobustnessResult:
        """
        Run Monte Carlo test with trade sequence resampling.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of simulations to run
            block_size: Block size for block bootstrap (preserves temporal correlations)
            preserve_correlations: Whether to preserve temporal correlations
            **kwargs: Additional test parameters
            
        Returns:
            Robustness test result with trade sequence analysis
        """
        results = []
        
        if self.enable_parallel and n_simulations > 100:
            results = self._run_parallel_simulations(
                parameters, backtest_function, n_simulations, 
                "trade_sequence", block_size, preserve_correlations
            )
        else:
            results = self._run_sequential_simulations(
                parameters, backtest_function, n_simulations,
                "trade_sequence", block_size, preserve_correlations
            )
        
        # Enhanced statistical analysis
        summary_stats = self._calculate_enhanced_statistics(results)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(results)
        
        # Detect outliers
        outlier_analysis = self._detect_outliers(results)
        
        # Calculate risk metrics
        var_results, es_results = self._calculate_risk_metrics(results)
        
        # Calculate success rate
        success_rate = len([r for r in results if 'error' not in r]) / len(results)
        
        # Find best and worst case scenarios
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_case = max(valid_results, key=lambda x: x.get('total_return', float('-inf')))
            worst_case = min(valid_results, key=lambda x: x.get('total_return', float('-inf')))
        else:
            best_case = {"total_return": 0.0}
            worst_case = {"total_return": 0.0}
        
        # Add enhanced metadata
        enhanced_stats = summary_stats.copy()
        enhanced_stats['confidence_intervals'] = confidence_intervals
        enhanced_stats['outlier_analysis'] = outlier_analysis
        
        return RobustnessResult(
            test_type="monte_carlo_trade_sequence",
            n_simulations=n_simulations,
            results=results,
            summary_stats=enhanced_stats,
            value_at_risk=var_results,
            expected_shortfall=es_results,
            success_rate=success_rate,
            worst_case_scenario=worst_case,
            best_case_scenario=best_case,
        )
    
    def run_statistical_significance_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        benchmark_results: Optional[Dict[str, float]] = None,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo test with statistical significance analysis.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            benchmark_results: Benchmark results for comparison
            n_simulations: Number of simulations to run
            confidence_level: Confidence level for statistical tests
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with statistical significance results
        """
        # Run standard Monte Carlo test
        mc_result = self.run_test(
            parameters, backtest_function, n_simulations, 
            test_type="statistical_significance", **kwargs
        )
        
        # Perform statistical tests
        statistical_tests = {}
        
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            values = [r.get(metric, 0) for r in mc_result.results if 'error' not in r]
            
            if values:
                # Normality test
                _, normality_p = stats.shapiro(values[:5000])  # Limit for shapiro test
                
                # One-sample t-test against zero
                t_stat, t_p = stats.ttest_1samp(values, 0)
                
                # Bootstrap confidence interval
                bootstrap_ci = self._bootstrap_confidence_interval(values, confidence_level)
                
                # Benchmark comparison if provided
                benchmark_test = None
                if benchmark_results and metric in benchmark_results:
                    benchmark_value = benchmark_results[metric]
                    benchmark_t_stat, benchmark_p = stats.ttest_1samp(values, benchmark_value)
                    benchmark_test = {
                        't_statistic': benchmark_t_stat,
                        'p_value': benchmark_p,
                        'significant': benchmark_p < (1 - confidence_level),
                        'benchmark_value': benchmark_value,
                    }
                
                statistical_tests[metric] = {
                    'normality_test': {
                        'p_value': normality_p,
                        'is_normal': normality_p > 0.05,
                    },
                    'zero_test': {
                        't_statistic': t_stat,
                        'p_value': t_p,
                        'significant': t_p < (1 - confidence_level),
                    },
                    'confidence_interval': bootstrap_ci,
                    'benchmark_test': benchmark_test,
                }
        
        return {
            'monte_carlo_result': mc_result,
            'statistical_tests': statistical_tests,
            'confidence_level': confidence_level,
        }
    
    def _run_parallel_simulations(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int,
        test_type: str,
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run simulations in parallel for better performance.
        """
        results = []
        
        # Split simulations into chunks
        chunk_size = max(1, n_simulations // (self.max_workers or 4))
        chunks = [
            (i, min(chunk_size, n_simulations - i)) 
            for i in range(0, n_simulations, chunk_size)
        ]
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for start_idx, chunk_n in chunks:
                    future = executor.submit(
                        self._run_simulation_chunk,
                        parameters, backtest_function, chunk_n, test_type,
                        start_idx, *args, **kwargs
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        chunk_results = future.result()
                        results.extend(chunk_results)
                    except Exception as e:
                        warnings.warn(f"Simulation chunk failed: {e}")
                        
        except Exception as e:
            warnings.warn(f"Parallel execution failed, falling back to sequential: {e}")
            return self._run_sequential_simulations(
                parameters, backtest_function, n_simulations, test_type, *args, **kwargs
            )
        
        return results
    
    def _run_sequential_simulations(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int,
        test_type: str,
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run simulations sequentially.
        """
        return self._run_simulation_chunk(
            parameters, backtest_function, n_simulations, test_type, 0, *args, **kwargs
        )
    
    def _run_simulation_chunk(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int,
        test_type: str,
        start_idx: int = 0,
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run a chunk of simulations (used for both parallel and sequential execution).
        """
        results = []
        
        for i in range(n_simulations):
            try:
                sim_idx = start_idx + i
                
                if test_type == "trade_sequence":
                    block_size = args[0] if args else None
                    preserve_correlations = args[1] if len(args) > 1 else True
                    perturbed_data = self._resample_trade_sequence(block_size, preserve_correlations)
                    sim_params = parameters.copy()
                    
                elif test_type == "data_noise":
                    perturbed_data = self._add_price_noise(self.data)
                    sim_params = parameters.copy()
                    
                elif test_type == "parameter_variation":
                    perturbed_data = self.data.copy()
                    sim_params = self._perturb_parameters(parameters)
                    
                elif test_type == "statistical_significance":
                    perturbed_data = self._add_price_noise(self.data)
                    sim_params = parameters.copy()
                    
                else:
                    raise ValueError(f"Unknown test type: {test_type}")
                
                # Run backtest
                metrics = backtest_function(perturbed_data, sim_params)
                
                # Add simulation metadata
                metrics['simulation'] = sim_idx
                metrics['test_type'] = test_type
                
                results.append(metrics)
                
            except Exception as e:
                # Record failed simulation
                failed_metrics = {
                    'simulation': sim_idx,
                    'test_type': test_type,
                    'error': str(e),
                    'total_return': float('-inf'),
                    'sharpe_ratio': float('-inf'),
                }
                results.append(failed_metrics)
        
        return results
    
    def _resample_trade_sequence(
        self, 
        block_size: Optional[int] = None,
        preserve_correlations: bool = True
    ) -> pd.DataFrame:
        """
        Resample trade sequences while optionally preserving temporal correlations.
        
        Args:
            block_size: Size of blocks for block bootstrap
            preserve_correlations: Whether to preserve temporal correlations
            
        Returns:
            Resampled data
        """
        if preserve_correlations and block_size is not None:
            # Block bootstrap to preserve correlations
            return self._block_bootstrap_resample(block_size)
        else:
            # Standard bootstrap resampling
            return self._bootstrap_resample(self.data)
    
    def _block_bootstrap_resample(self, block_size: int) -> pd.DataFrame:
        """
        Perform block bootstrap resampling to preserve temporal correlations.
        """
        n_samples = len(self.data)
        n_blocks = int(np.ceil(n_samples / block_size))
        
        resampled_data = []
        
        for _ in range(n_blocks):
            # Randomly select block start
            start_idx = np.random.randint(0, max(1, n_samples - block_size + 1))
            end_idx = min(start_idx + block_size, n_samples)
            
            block = self.data.iloc[start_idx:end_idx].copy()
            resampled_data.append(block)
        
        # Concatenate and truncate to original length
        result = pd.concat(resampled_data, ignore_index=True)
        result = result.iloc[:n_samples]
        
        # Restore datetime index
        if isinstance(self.data.index, pd.DatetimeIndex):
            freq = pd.infer_freq(self.data.index)
            if freq:
                new_index = pd.date_range(
                    start=self.data.index.min(), 
                    periods=len(result), 
                    freq=freq
                )
                result.index = new_index
        
        return result
    
    def _calculate_enhanced_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate enhanced statistical measures for simulation results.
        """
        summary_stats = self._calculate_summary_statistics(results)
        
        # Add enhanced metrics
        for metric, stats_dict in summary_stats.items():
            values = [r.get(metric, 0) for r in results if 'error' not in r and isinstance(r.get(metric), (int, float))]
            
            if values:
                # Add skewness and kurtosis
                stats_dict['skewness'] = stats.skew(values)
                stats_dict['kurtosis'] = stats.kurtosis(values)
                
                # Add percentiles
                for p in [1, 5, 10, 90, 95, 99]:
                    stats_dict[f'p{p}'] = np.percentile(values, p)
                
                # Add coefficient of variation
                stats_dict['coefficient_of_variation'] = stats_dict['std'] / abs(stats_dict['mean']) if stats_dict['mean'] != 0 else float('inf')
                
                # Add normality test
                if len(values) >= 3:
                    _, normality_p = stats.shapiro(values[:5000])  # Limit for Shapiro test
                    stats_dict['normality_p_value'] = normality_p
                    stats_dict['is_normal'] = normality_p > 0.05
        
        return summary_stats
    
    def _calculate_confidence_intervals(
        self, 
        results: List[Dict[str, Any]], 
        confidence_levels: List[float] = [0.90, 0.95, 0.99]
    ) -> Dict[str, Dict[float, Tuple[float, float]]]:
        """
        Calculate confidence intervals for key metrics.
        """
        confidence_intervals = {}
        
        # Get all numeric metrics
        all_metrics = set()
        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ['simulation']:
                    all_metrics.add(key)
        
        for metric in all_metrics:
            values = [r.get(metric, 0) for r in results if 'error' not in r and isinstance(r.get(metric), (int, float))]
            
            if values:
                confidence_intervals[metric] = {}
                
                for conf_level in confidence_levels:
                    ci = self._bootstrap_confidence_interval(values, conf_level)
                    confidence_intervals[metric][conf_level] = ci
        
        return confidence_intervals
    
    def _bootstrap_confidence_interval(
        self, 
        values: List[float], 
        confidence_level: float,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        """
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    def _detect_outliers(
        self, 
        results: List[Dict[str, Any]],
        outlier_method: str = "iqr"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers in simulation results.
        
        Args:
            results: Simulation results
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation')
            
        Returns:
            Dictionary with outlier analysis for each metric
        """
        outlier_analysis = {}
        
        # Get all numeric metrics
        all_metrics = set()
        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ['simulation']:
                    all_metrics.add(key)
        
        for metric in all_metrics:
            values = [r.get(metric, 0) for r in results if 'error' not in r and isinstance(r.get(metric), (int, float))]
            
            if len(values) > 10:  # Need sufficient data for outlier detection
                outliers = []
                outlier_indices = []
                
                if outlier_method == "iqr":
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    for i, value in enumerate(values):
                        if value < lower_bound or value > upper_bound:
                            outliers.append(value)
                            outlier_indices.append(i)
                
                elif outlier_method == "zscore":
                    z_scores = np.abs(stats.zscore(values))
                    threshold = 3.0
                    
                    for i, z_score in enumerate(z_scores):
                        if z_score > threshold:
                            outliers.append(values[i])
                            outlier_indices.append(i)
                
                outlier_analysis[metric] = {
                    'outliers': outliers,
                    'outlier_indices': outlier_indices,
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(values) * 100,
                    'method': outlier_method,
                }
            else:
                outlier_analysis[metric] = {
                    'outliers': [],
                    'outlier_indices': [],
                    'outlier_count': 0,
                    'outlier_percentage': 0.0,
                    'method': outlier_method,
                }
        
        return outlier_analysis