"""
Bootstrap robustness testing for trading strategies.

Bootstrap testing uses resampling techniques to assess the stability
and statistical significance of strategy performance by creating
multiple versions of the historical data through sampling with replacement.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from btc_research.optimization.base import BaseRobustnessTest
from btc_research.optimization.types import RobustnessResult

__all__ = ["BootstrapRobustnessTest"]


class BootstrapRobustnessTest(BaseRobustnessTest):
    """
    Bootstrap robustness testing implementation.
    
    This test uses various bootstrap resampling techniques to assess
    the stability of strategy performance across different data samples.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        random_seed: Optional[int] = None,
        block_size: Optional[int] = None,
    ):
        """
        Initialize bootstrap robustness test.
        
        Args:
            data: Historical data for testing
            random_seed: Random seed for reproducibility
            block_size: Block size for block bootstrap (if None, uses single observations)
        """
        super().__init__(data, random_seed)
        
        self.block_size = block_size
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def run_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 1000,
        bootstrap_method: str = "standard",
        success_threshold: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> RobustnessResult:
        """
        Run bootstrap robustness test.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of bootstrap simulations to run
            bootstrap_method: Bootstrap method ('standard', 'block', 'stationary', 'circular')
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
                # Generate bootstrap sample
                if bootstrap_method == "standard":
                    bootstrap_data = self._standard_bootstrap()
                elif bootstrap_method == "block":
                    bootstrap_data = self._block_bootstrap()
                elif bootstrap_method == "stationary":
                    bootstrap_data = self._stationary_bootstrap()
                elif bootstrap_method == "circular":
                    bootstrap_data = self._circular_bootstrap()
                else:
                    raise ValueError(f"Unknown bootstrap method: {bootstrap_method}")
                
                # Run backtest on bootstrap sample
                metrics = backtest_function(bootstrap_data, parameters)
                
                # Add simulation info
                metrics['simulation'] = i
                metrics['bootstrap_method'] = bootstrap_method
                
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
                    'bootstrap_method': bootstrap_method,
                    'error': str(e),
                    'total_return': float('-inf'),
                    'sharpe_ratio': float('-inf'),
                }
                results.append(failed_metrics)
        
        if not results:
            raise ValueError("All bootstrap simulations failed")
        
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
            test_type=f"bootstrap_{bootstrap_method}",
            n_simulations=n_simulations,
            results=results,
            summary_stats=summary_stats,
            value_at_risk=var_results,
            expected_shortfall=es_results,
            success_rate=success_rate,
            worst_case_scenario=worst_case,
            best_case_scenario=best_case,
        )
    
    def _standard_bootstrap(self) -> pd.DataFrame:
        """
        Standard bootstrap resampling (sampling with replacement).
        
        Returns:
            Bootstrap resampled data
        """
        n_samples = len(self.data)
        
        # Sample indices with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_data = self.data.iloc[bootstrap_indices].copy()
        
        # Reset index and create new datetime index if needed
        bootstrap_data = bootstrap_data.reset_index(drop=True)
        
        if isinstance(self.data.index, pd.DatetimeIndex):
            start_date = self.data.index.min()
            freq = pd.infer_freq(self.data.index)
            if freq:
                new_index = pd.date_range(start=start_date, periods=len(bootstrap_data), freq=freq)
                bootstrap_data.index = new_index
        
        return bootstrap_data
    
    def _block_bootstrap(self) -> pd.DataFrame:
        """
        Block bootstrap resampling (maintains temporal structure).
        
        Returns:
            Block bootstrap resampled data
        """
        n_samples = len(self.data)
        
        # Use default block size if not specified
        if self.block_size is None:
            # Rule of thumb: block size ≈ n^(1/3)
            block_size = max(1, int(n_samples ** (1/3)))
        else:
            block_size = self.block_size
        
        # Calculate number of blocks needed
        n_blocks = int(np.ceil(n_samples / block_size))
        
        bootstrap_data_list = []
        
        for _ in range(n_blocks):
            # Randomly select a starting position for the block
            start_pos = np.random.randint(0, max(1, n_samples - block_size + 1))
            end_pos = min(start_pos + block_size, n_samples)
            
            # Extract block
            block = self.data.iloc[start_pos:end_pos].copy()
            bootstrap_data_list.append(block)
        
        # Concatenate blocks
        bootstrap_data = pd.concat(bootstrap_data_list, ignore_index=True)
        
        # Truncate to original length
        bootstrap_data = bootstrap_data.iloc[:n_samples]
        
        # Create new datetime index if needed
        if isinstance(self.data.index, pd.DatetimeIndex):
            start_date = self.data.index.min()
            freq = pd.infer_freq(self.data.index)
            if freq:
                new_index = pd.date_range(start=start_date, periods=len(bootstrap_data), freq=freq)
                bootstrap_data.index = new_index
        
        return bootstrap_data
    
    def _stationary_bootstrap(self) -> pd.DataFrame:
        """
        Stationary bootstrap resampling (random block lengths).
        
        Returns:
            Stationary bootstrap resampled data
        """
        n_samples = len(self.data)
        
        # Average block length
        if self.block_size is None:
            avg_block_length = max(1, int(n_samples ** (1/3)))
        else:
            avg_block_length = self.block_size
        
        bootstrap_data_list = []
        current_length = 0
        
        while current_length < n_samples:
            # Generate block length from geometric distribution
            block_length = np.random.geometric(1.0 / avg_block_length)
            
            # Randomly select starting position
            start_pos = np.random.randint(0, n_samples)
            
            # Extract block (with wraparound if necessary)
            block_indices = []
            for i in range(block_length):
                idx = (start_pos + i) % n_samples
                block_indices.append(idx)
            
            block = self.data.iloc[block_indices].copy()
            bootstrap_data_list.append(block)
            
            current_length += block_length
        
        # Concatenate and truncate to original length
        bootstrap_data = pd.concat(bootstrap_data_list, ignore_index=True)
        bootstrap_data = bootstrap_data.iloc[:n_samples]
        
        # Create new datetime index if needed
        if isinstance(self.data.index, pd.DatetimeIndex):
            start_date = self.data.index.min()
            freq = pd.infer_freq(self.data.index)
            if freq:
                new_index = pd.date_range(start=start_date, periods=len(bootstrap_data), freq=freq)
                bootstrap_data.index = new_index
        
        return bootstrap_data
    
    def _circular_bootstrap(self) -> pd.DataFrame:
        """
        Circular bootstrap resampling (treats data as circular).
        
        Returns:
            Circular bootstrap resampled data
        """
        n_samples = len(self.data)
        
        # Use default block size if not specified
        if self.block_size is None:
            block_size = max(1, int(n_samples ** (1/3)))
        else:
            block_size = self.block_size
        
        # Randomly select starting position
        start_pos = np.random.randint(0, n_samples)
        
        # Create circular sample
        indices = []
        for i in range(n_samples):
            idx = (start_pos + i) % n_samples
            indices.append(idx)
        
        bootstrap_data = self.data.iloc[indices].copy()
        bootstrap_data = bootstrap_data.reset_index(drop=True)
        
        # Create new datetime index if needed
        if isinstance(self.data.index, pd.DatetimeIndex):
            start_date = self.data.index.min()
            freq = pd.infer_freq(self.data.index)
            if freq:
                new_index = pd.date_range(start=start_date, periods=len(bootstrap_data), freq=freq)
                bootstrap_data.index = new_index
        
        return bootstrap_data
    
    def _calculate_summary_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for bootstrap results.
        
        Args:
            results: List of bootstrap simulation results
            
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
                    'bootstrap_bias': np.mean(values) - self._calculate_original_metric(metric),
                    'bootstrap_ci_lower': np.percentile(values, 2.5),
                    'bootstrap_ci_upper': np.percentile(values, 97.5),
                }
            else:
                summary_stats[metric] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'median': 0.0, 'q25': 0.0, 'q75': 0.0, 'count': 0,
                    'bootstrap_bias': 0.0, 'bootstrap_ci_lower': 0.0, 'bootstrap_ci_upper': 0.0,
                }
        
        return summary_stats
    
    def _calculate_original_metric(self, metric: str) -> float:
        """
        Calculate the original metric value for bias calculation.
        
        This is a placeholder - in practice, you would run the backtest
        on the original data to get the baseline metric value.
        
        Args:
            metric: Metric name
            
        Returns:
            Original metric value (placeholder returns 0.0)
        """
        # Placeholder - in real implementation, would calculate from original data
        return 0.0
    
    def get_confidence_intervals(
        self, 
        results: List[Dict[str, Any]], 
        confidence_level: float = 0.95
    ) -> Dict[str, tuple]:
        """
        Calculate bootstrap confidence intervals for metrics.
        
        Args:
            results: Bootstrap simulation results
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Dictionary of confidence intervals by metric
        """
        alpha = 1 - confidence_level
        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100
        
        confidence_intervals = {}
        
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
                lower_bound = np.percentile(values, lower_pct)
                upper_bound = np.percentile(values, upper_pct)
                confidence_intervals[metric] = (lower_bound, upper_bound)
            else:
                confidence_intervals[metric] = (0.0, 0.0)
        
        return confidence_intervals