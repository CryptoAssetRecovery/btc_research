"""
Data perturbation testing for trading strategies.

Data perturbation testing assesses strategy robustness by introducing
controlled variations to market data including price noise, volume
perturbations, market regime changes, and data quality issues.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from btc_research.optimization.base import BaseRobustnessTest
from btc_research.optimization.types import RobustnessResult

__all__ = ["DataPerturbationTest"]


class DataPerturbationTest(BaseRobustnessTest):
    """
    Data perturbation robustness testing implementation.
    
    This test systematically perturbs market data to assess how
    sensitive a strategy is to data quality and market conditions.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize data perturbation test.
        
        Args:
            data: Historical data for testing
            random_seed: Random seed for reproducibility
        """
        super().__init__(data, random_seed)
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Analyze data characteristics for perturbation calibration
        self._analyze_data_characteristics()
    
    def run_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 1000,
        perturbation_types: List[str] = None,
        **kwargs: Any,
    ) -> RobustnessResult:
        """
        Run comprehensive data perturbation test.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of simulations to run
            perturbation_types: Types of perturbations to test
            **kwargs: Additional test parameters
            
        Returns:
            Robustness test result with perturbation analysis
        """
        if perturbation_types is None:
            perturbation_types = [
                'price_noise', 'volume_noise', 'spread_widening',
                'missing_data', 'regime_change'
            ]
        
        results = []
        
        for perturbation_type in perturbation_types:
            # Run simulations for each perturbation type
            type_results = self._run_perturbation_simulations(
                parameters, backtest_function, perturbation_type,
                n_simulations // len(perturbation_types), **kwargs
            )
            results.extend(type_results)
        
        # Calculate summary statistics
        summary_stats = self._calculate_perturbation_statistics(results)
        
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
        
        return RobustnessResult(
            test_type="data_perturbation",
            n_simulations=len(results),
            results=results,
            summary_stats=summary_stats,
            value_at_risk=var_results,
            expected_shortfall=es_results,
            success_rate=success_rate,
            worst_case_scenario=worst_case,
            best_case_scenario=best_case,
        )
    
    def run_price_noise_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        noise_levels: List[float] = None,
        n_simulations_per_level: int = 100,
        noise_type: str = "gaussian",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run systematic price noise injection test.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            noise_levels: List of noise levels to test (as fraction of price volatility)
            n_simulations_per_level: Number of simulations per noise level
            noise_type: Type of noise ('gaussian', 'uniform', 'laplace')
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with detailed noise analysis results
        """
        if noise_levels is None:
            noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        noise_results = {}
        
        for noise_level in noise_levels:
            level_results = []
            
            for i in range(n_simulations_per_level):
                try:
                    # Apply noise to data
                    perturbed_data = self._add_price_noise(
                        self.data, noise_level, noise_type
                    )
                    
                    # Run backtest
                    metrics = backtest_function(perturbed_data, parameters)
                    
                    # Add metadata
                    metrics['simulation'] = i
                    metrics['noise_level'] = noise_level
                    metrics['noise_type'] = noise_type
                    
                    level_results.append(metrics)
                    
                except Exception as e:
                    failed_metrics = {
                        'simulation': i,
                        'noise_level': noise_level,
                        'noise_type': noise_type,
                        'error': str(e),
                        'total_return': float('-inf'),
                        'sharpe_ratio': float('-inf'),
                    }
                    level_results.append(failed_metrics)
            
            # Calculate statistics for this noise level
            level_stats = self._calculate_level_statistics(level_results)
            
            noise_results[noise_level] = {
                'results': level_results,
                'statistics': level_stats,
                'degradation_score': self._calculate_degradation_score(level_results),
            }
        
        # Analyze noise sensitivity
        sensitivity_analysis = self._analyze_noise_sensitivity(noise_results)
        
        return {
            'noise_results': noise_results,
            'sensitivity_analysis': sensitivity_analysis,
            'noise_type': noise_type,
        }
    
    def run_volume_perturbation_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        perturbation_methods: List[str] = None,
        n_simulations: int = 500,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run volume and spread perturbation test.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            perturbation_methods: List of perturbation methods to test
            n_simulations: Number of simulations to run
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with volume perturbation results
        """
        if perturbation_methods is None:
            perturbation_methods = [
                'volume_scaling', 'volume_noise', 'spread_widening', 
                'liquidity_crisis', 'volume_clustering'
            ]
        
        volume_results = {}
        
        for method in perturbation_methods:
            method_results = []
            
            for i in range(n_simulations // len(perturbation_methods)):
                try:
                    # Apply volume perturbation
                    perturbed_data = self._apply_volume_perturbation(
                        self.data, method
                    )
                    
                    # Run backtest
                    metrics = backtest_function(perturbed_data, parameters)
                    
                    # Add metadata
                    metrics['simulation'] = i
                    metrics['perturbation_method'] = method
                    
                    method_results.append(metrics)
                    
                except Exception as e:
                    failed_metrics = {
                        'simulation': i,
                        'perturbation_method': method,
                        'error': str(e),
                        'total_return': float('-inf'),
                        'sharpe_ratio': float('-inf'),
                    }
                    method_results.append(failed_metrics)
            
            volume_results[method] = {
                'results': method_results,
                'statistics': self._calculate_level_statistics(method_results),
            }
        
        return volume_results
    
    def run_market_regime_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        regime_types: List[str] = None,
        n_simulations: int = 300,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run market regime simulation test.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            regime_types: List of market regimes to simulate
            n_simulations: Number of simulations to run
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with market regime results
        """
        if regime_types is None:
            regime_types = ['bull_market', 'bear_market', 'sideways', 'high_volatility', 'trending']
        
        regime_results = {}
        
        for regime in regime_types:
            regime_sims = []
            
            for i in range(n_simulations // len(regime_types)):
                try:
                    # Simulate market regime
                    regime_data = self._simulate_market_regime(self.data, regime)
                    
                    # Run backtest
                    metrics = backtest_function(regime_data, parameters)
                    
                    # Add metadata
                    metrics['simulation'] = i
                    metrics['regime_type'] = regime
                    
                    regime_sims.append(metrics)
                    
                except Exception as e:
                    failed_metrics = {
                        'simulation': i,
                        'regime_type': regime,
                        'error': str(e),
                        'total_return': float('-inf'),
                        'sharpe_ratio': float('-inf'),
                    }
                    regime_sims.append(failed_metrics)
            
            regime_results[regime] = {
                'results': regime_sims,
                'statistics': self._calculate_level_statistics(regime_sims),
            }
        
        return regime_results
    
    def run_data_quality_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        quality_issues: List[str] = None,
        n_simulations: int = 400,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run data quality degradation test.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            quality_issues: List of data quality issues to simulate
            n_simulations: Number of simulations to run
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with data quality test results
        """
        if quality_issues is None:
            quality_issues = [
                'missing_data', 'outliers', 'gaps', 'timestamp_errors', 'duplicate_data'
            ]
        
        quality_results = {}
        
        for issue in quality_issues:
            issue_results = []
            
            for i in range(n_simulations // len(quality_issues)):
                try:
                    # Introduce data quality issue
                    degraded_data = self._introduce_data_quality_issue(self.data, issue)
                    
                    # Run backtest
                    metrics = backtest_function(degraded_data, parameters)
                    
                    # Add metadata
                    metrics['simulation'] = i
                    metrics['quality_issue'] = issue
                    
                    issue_results.append(metrics)
                    
                except Exception as e:
                    failed_metrics = {
                        'simulation': i,
                        'quality_issue': issue,
                        'error': str(e),
                        'total_return': float('-inf'),
                        'sharpe_ratio': float('-inf'),
                    }
                    issue_results.append(failed_metrics)
            
            quality_results[issue] = {
                'results': issue_results,
                'statistics': self._calculate_level_statistics(issue_results),
            }
        
        return quality_results
    
    def _analyze_data_characteristics(self) -> None:
        """
        Analyze data characteristics to calibrate perturbations.
        """
        if 'close' in self.data.columns:
            self.price_volatility = self.data['close'].pct_change().std()
            self.typical_price_range = self.data['close'].max() - self.data['close'].min()
        else:
            self.price_volatility = 0.01
            self.typical_price_range = 1000.0
            
        if 'volume' in self.data.columns:
            self.volume_stats = {
                'mean': self.data['volume'].mean(),
                'std': self.data['volume'].std(),
                'median': self.data['volume'].median(),
            }
        else:
            self.volume_stats = {'mean': 1000, 'std': 500, 'median': 1000}
    
    def _run_perturbation_simulations(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        perturbation_type: str,
        n_simulations: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Run simulations for a specific perturbation type.
        """
        results = []
        
        for i in range(n_simulations):
            try:
                # Apply perturbation based on type
                if perturbation_type == 'price_noise':
                    perturbed_data = self._add_price_noise(self.data)
                elif perturbation_type == 'volume_noise':
                    perturbed_data = self._apply_volume_perturbation(self.data, 'volume_noise')
                elif perturbation_type == 'spread_widening':
                    perturbed_data = self._apply_volume_perturbation(self.data, 'spread_widening')
                elif perturbation_type == 'missing_data':
                    perturbed_data = self._introduce_data_quality_issue(self.data, 'missing_data')
                elif perturbation_type == 'regime_change':
                    perturbed_data = self._simulate_market_regime(self.data, 'high_volatility')
                else:
                    raise ValueError(f"Unknown perturbation type: {perturbation_type}")
                
                # Run backtest
                metrics = backtest_function(perturbed_data, parameters)
                
                # Add metadata
                metrics['simulation'] = i
                metrics['perturbation_type'] = perturbation_type
                
                results.append(metrics)
                
            except Exception as e:
                failed_metrics = {
                    'simulation': i,
                    'perturbation_type': perturbation_type,
                    'error': str(e),
                    'total_return': float('-inf'),
                    'sharpe_ratio': float('-inf'),
                }
                results.append(failed_metrics)
        
        return results
    
    def _add_price_noise(
        self, 
        data: pd.DataFrame, 
        noise_level: float = 0.01,
        noise_type: str = "gaussian"
    ) -> pd.DataFrame:
        """
        Add controlled noise to price data.
        """
        perturbed_data = data.copy()
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col in perturbed_data.columns:
                prices = perturbed_data[col]
                
                if noise_type == "gaussian":
                    noise = np.random.normal(0, noise_level * self.price_volatility, len(prices))
                elif noise_type == "uniform":
                    noise_range = noise_level * self.price_volatility * np.sqrt(3)  # Match variance
                    noise = np.random.uniform(-noise_range, noise_range, len(prices))
                elif noise_type == "laplace":
                    noise_scale = noise_level * self.price_volatility / np.sqrt(2)  # Match variance
                    noise = np.random.laplace(0, noise_scale, len(prices))
                else:
                    raise ValueError(f"Unknown noise type: {noise_type}")
                
                # Apply proportional noise
                perturbed_prices = prices * (1 + noise)
                
                # Ensure positive prices
                perturbed_prices = np.maximum(perturbed_prices, prices * 0.1)
                perturbed_data[col] = perturbed_prices
        
        # Ensure OHLC consistency
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
    
    def _apply_volume_perturbation(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Apply volume and spread perturbations.
        """
        perturbed_data = data.copy()
        
        if method == 'volume_scaling':
            # Scale volume by random factor
            if 'volume' in perturbed_data.columns:
                scale_factor = np.random.uniform(0.5, 2.0)
                perturbed_data['volume'] *= scale_factor
                
        elif method == 'volume_noise':
            # Add noise to volume
            if 'volume' in perturbed_data.columns:
                noise = np.random.normal(0, 0.3, len(perturbed_data))
                perturbed_data['volume'] *= (1 + noise)
                perturbed_data['volume'] = np.maximum(perturbed_data['volume'], 0)
                
        elif method == 'spread_widening':
            # Simulate bid-ask spread widening
            spread_factor = np.random.uniform(1.5, 3.0)
            price_impact = np.random.normal(0, 0.001, len(perturbed_data))
            
            for col in ['open', 'high', 'low', 'close']:
                if col in perturbed_data.columns:
                    perturbed_data[col] *= (1 + price_impact * spread_factor)
                    
        elif method == 'liquidity_crisis':
            # Simulate sudden liquidity drop
            crisis_start = np.random.randint(0, len(perturbed_data) // 2)
            crisis_duration = np.random.randint(10, 50)
            crisis_end = min(crisis_start + crisis_duration, len(perturbed_data))
            
            if 'volume' in perturbed_data.columns:
                perturbed_data.loc[crisis_start:crisis_end, 'volume'] *= 0.1
                
            # Add price volatility during crisis
            for col in ['open', 'high', 'low', 'close']:
                if col in perturbed_data.columns:
                    crisis_noise = np.random.normal(0, 0.02, crisis_end - crisis_start + 1)
                    perturbed_data.iloc[crisis_start:crisis_end + 1, perturbed_data.columns.get_loc(col)] *= (1 + crisis_noise)
                    
        elif method == 'volume_clustering':
            # Simulate volume clustering (high volume periods)
            if 'volume' in perturbed_data.columns:
                # Create random high-volume periods
                n_clusters = np.random.randint(3, 8)
                for _ in range(n_clusters):
                    cluster_start = np.random.randint(0, len(perturbed_data) - 20)
                    cluster_duration = np.random.randint(5, 20)
                    cluster_end = min(cluster_start + cluster_duration, len(perturbed_data))
                    
                    volume_multiplier = np.random.uniform(3.0, 10.0)
                    perturbed_data.iloc[cluster_start:cluster_end, perturbed_data.columns.get_loc('volume')] *= volume_multiplier
        
        return perturbed_data
    
    def _simulate_market_regime(self, data: pd.DataFrame, regime_type: str) -> pd.DataFrame:
        """
        Simulate different market regimes.
        """
        regime_data = data.copy()
        
        if regime_type == 'bull_market':
            # Add upward trend
            trend_factor = np.random.uniform(0.0005, 0.002)  # Daily trend
            trend = np.cumsum(np.random.normal(trend_factor, 0.001, len(regime_data)))
            
            for col in ['open', 'high', 'low', 'close']:
                if col in regime_data.columns:
                    regime_data[col] *= (1 + trend)
                    
        elif regime_type == 'bear_market':
            # Add downward trend
            trend_factor = np.random.uniform(-0.002, -0.0005)  # Daily trend
            trend = np.cumsum(np.random.normal(trend_factor, 0.001, len(regime_data)))
            
            for col in ['open', 'high', 'low', 'close']:
                if col in regime_data.columns:
                    regime_data[col] *= (1 + trend)
                    
        elif regime_type == 'sideways':
            # Add mean reversion
            mean_price = regime_data['close'].iloc[0] if 'close' in regime_data.columns else 50000
            reversion_strength = 0.01
            
            for i in range(1, len(regime_data)):
                if 'close' in regime_data.columns:
                    current_price = regime_data['close'].iloc[i]
                    reversion = (mean_price - current_price) * reversion_strength
                    noise = np.random.normal(0, 0.005)
                    
                    price_change = reversion + noise
                    
                    for col in ['open', 'high', 'low', 'close']:
                        if col in regime_data.columns:
                            regime_data.iloc[i, regime_data.columns.get_loc(col)] *= (1 + price_change)
                            
        elif regime_type == 'high_volatility':
            # Increase volatility
            volatility_multiplier = np.random.uniform(2.0, 5.0)
            
            for col in ['open', 'high', 'low', 'close']:
                if col in regime_data.columns:
                    returns = regime_data[col].pct_change()
                    enhanced_returns = returns * volatility_multiplier
                    
                    # Reconstruct prices
                    new_prices = [regime_data[col].iloc[0]]
                    for ret in enhanced_returns.iloc[1:]:
                        if not np.isnan(ret):
                            new_price = new_prices[-1] * (1 + ret)
                        else:
                            new_price = new_prices[-1]
                        new_prices.append(new_price)
                    
                    regime_data[col] = new_prices
                    
        elif regime_type == 'trending':
            # Add persistent trending behavior
            trend_persistence = 0.7  # AR(1) coefficient
            trend_innovation = np.random.normal(0, 0.001, len(regime_data))
            trend = [0]
            
            for i in range(1, len(regime_data)):
                new_trend = trend_persistence * trend[-1] + trend_innovation[i]
                trend.append(new_trend)
            
            for col in ['open', 'high', 'low', 'close']:
                if col in regime_data.columns:
                    regime_data[col] *= (1 + np.array(trend))
        
        return regime_data
    
    def _introduce_data_quality_issue(self, data: pd.DataFrame, issue_type: str) -> pd.DataFrame:
        """
        Introduce various data quality issues.
        """
        degraded_data = data.copy()
        
        if issue_type == 'missing_data':
            # Randomly remove data points
            missing_rate = np.random.uniform(0.01, 0.05)  # 1-5% missing
            n_missing = int(len(degraded_data) * missing_rate)
            missing_indices = np.random.choice(len(degraded_data), n_missing, replace=False)
            
            # Remove random rows
            degraded_data = degraded_data.drop(degraded_data.index[missing_indices])
            
        elif issue_type == 'outliers':
            # Inject price outliers
            outlier_rate = 0.002  # 0.2% outliers
            n_outliers = max(1, int(len(degraded_data) * outlier_rate))
            outlier_indices = np.random.choice(len(degraded_data), n_outliers, replace=False)
            
            for idx in outlier_indices:
                outlier_factor = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.3)
                for col in ['open', 'high', 'low', 'close']:
                    if col in degraded_data.columns:
                        degraded_data.iloc[idx, degraded_data.columns.get_loc(col)] *= (1 + outlier_factor)
                        
        elif issue_type == 'gaps':
            # Create data gaps
            n_gaps = np.random.randint(1, 5)
            for _ in range(n_gaps):
                gap_start = np.random.randint(0, len(degraded_data) - 10)
                gap_size = np.random.randint(3, 10)
                gap_end = min(gap_start + gap_size, len(degraded_data))
                
                degraded_data = degraded_data.drop(degraded_data.index[gap_start:gap_end])
                
        elif issue_type == 'timestamp_errors':
            # Introduce timestamp inconsistencies
            error_rate = 0.01  # 1% timestamp errors
            n_errors = max(1, int(len(degraded_data) * error_rate))
            error_indices = np.random.choice(len(degraded_data), n_errors, replace=False)
            
            # This is a simplified simulation - in practice would affect datetime index
            # Here we just mark the rows as having timestamp issues
            degraded_data.loc[degraded_data.index[error_indices], 'timestamp_error'] = 1
            
        elif issue_type == 'duplicate_data':
            # Add duplicate rows
            duplicate_rate = 0.005  # 0.5% duplicates
            n_duplicates = max(1, int(len(degraded_data) * duplicate_rate))
            duplicate_indices = np.random.choice(len(degraded_data), n_duplicates, replace=False)
            
            # Duplicate selected rows
            duplicated_rows = degraded_data.iloc[duplicate_indices].copy()
            degraded_data = pd.concat([degraded_data, duplicated_rows]).sort_index()
        
        return degraded_data
    
    def _calculate_perturbation_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics grouped by perturbation type.
        """
        summary_stats = {}
        
        # Group results by perturbation type
        perturbation_groups = {}
        for result in results:
            perturbation_type = result.get('perturbation_type', 'unknown')
            if perturbation_type not in perturbation_groups:
                perturbation_groups[perturbation_type] = []
            perturbation_groups[perturbation_type].append(result)
        
        # Calculate statistics for each group
        for perturbation_type, group_results in perturbation_groups.items():
            group_stats = {}
            
            # Get all numeric metrics
            all_metrics = set()
            for result in group_results:
                for key, value in result.items():
                    if isinstance(value, (int, float)) and key not in ['simulation']:
                        all_metrics.add(key)
            
            for metric in all_metrics:
                values = [r.get(metric, 0) for r in group_results if 'error' not in r and isinstance(r.get(metric), (int, float))]
                
                if values:
                    group_stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'count': len(values),
                        'success_rate': len(values) / len(group_results),
                    }
            
            summary_stats[perturbation_type] = group_stats
        
        return summary_stats
    
    def _calculate_level_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate basic statistics for a set of results.
        """
        stats = {}
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'success_rate': 0.0, 'mean_return': 0.0, 'std_return': 0.0}
        
        returns = [r.get('total_return', 0) for r in valid_results]
        
        stats['success_rate'] = len(valid_results) / len(results)
        stats['mean_return'] = np.mean(returns)
        stats['std_return'] = np.std(returns)
        stats['min_return'] = np.min(returns)
        stats['max_return'] = np.max(returns)
        
        return stats
    
    def _calculate_degradation_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate performance degradation score.
        """
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return 1.0  # Maximum degradation
        
        returns = [r.get('total_return', 0) for r in valid_results]
        
        # Compare to baseline (assumed to be original data performance)
        # This is a simplified metric - in practice would compare to clean data results
        baseline_return = 0.05  # 5% baseline assumption
        
        if baseline_return > 0:
            mean_return = np.mean(returns)
            degradation = max(0, (baseline_return - mean_return) / baseline_return)
            return min(1.0, degradation)
        else:
            return 0.0
    
    def _analyze_noise_sensitivity(self, noise_results: Dict[float, Any]) -> Dict[str, Any]:
        """
        Analyze sensitivity to different noise levels.
        """
        noise_levels = sorted(noise_results.keys())
        degradation_scores = [noise_results[level]['degradation_score'] for level in noise_levels]
        mean_returns = [noise_results[level]['statistics'].get('mean_return', 0) for level in noise_levels]
        
        # Calculate sensitivity metrics
        sensitivity_analysis = {
            'noise_tolerance': self._find_noise_tolerance(noise_levels, degradation_scores),
            'linear_sensitivity': self._calculate_linear_sensitivity(noise_levels, mean_returns),
            'critical_noise_level': self._find_critical_noise_level(noise_levels, degradation_scores),
        }
        
        return sensitivity_analysis
    
    def _find_noise_tolerance(self, noise_levels: List[float], degradation_scores: List[float]) -> float:
        """
        Find maximum noise level with acceptable degradation (< 0.2).
        """
        acceptable_threshold = 0.2
        
        for noise_level, degradation in zip(noise_levels, degradation_scores):
            if degradation > acceptable_threshold:
                return noise_level
        
        return max(noise_levels) if noise_levels else 0.0
    
    def _calculate_linear_sensitivity(self, noise_levels: List[float], mean_returns: List[float]) -> float:
        """
        Calculate linear sensitivity coefficient.
        """
        if len(noise_levels) < 2:
            return 0.0
        
        # Calculate correlation between noise level and performance degradation
        try:
            correlation = np.corrcoef(noise_levels, mean_returns)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _find_critical_noise_level(self, noise_levels: List[float], degradation_scores: List[float]) -> float:
        """
        Find critical noise level where degradation becomes severe (> 0.5).
        """
        critical_threshold = 0.5
        
        for noise_level, degradation in zip(noise_levels, degradation_scores):
            if degradation > critical_threshold:
                return noise_level
        
        return max(noise_levels) if noise_levels else 0.0
    
    def run_comprehensive_perturbation_suite(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 1000,
        validation_method: str = "statistical",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run comprehensive data perturbation test suite with statistical validation.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of simulations to run
            validation_method: Method for validating perturbation effects
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with comprehensive perturbation results and validation
        """
        suite_results = {}
        
        # Run individual perturbation tests
        noise_result = self.run_price_noise_test(
            parameters, backtest_function, n_simulations_per_level=n_simulations//6
        )
        suite_results['noise_analysis'] = noise_result
        
        volume_result = self.run_volume_perturbation_test(
            parameters, backtest_function, n_simulations=n_simulations//4
        )
        suite_results['volume_analysis'] = volume_result
        
        regime_result = self.run_market_regime_test(
            parameters, backtest_function, n_simulations=n_simulations//4
        )
        suite_results['regime_analysis'] = regime_result
        
        quality_result = self.run_data_quality_test(
            parameters, backtest_function, n_simulations=n_simulations//4
        )
        suite_results['quality_analysis'] = quality_result
        
        # Advanced perturbation tests
        microstructure_result = self.run_microstructure_noise_test(
            parameters, backtest_function, n_simulations=n_simulations//6
        )
        suite_results['microstructure_analysis'] = microstructure_result
        
        temporal_result = self.run_temporal_perturbation_test(
            parameters, backtest_function, n_simulations=n_simulations//6
        )
        suite_results['temporal_analysis'] = temporal_result
        
        # Statistical validation
        if validation_method == "statistical":
            validation_results = self._perform_statistical_validation(
                suite_results, parameters, backtest_function
            )
            suite_results['statistical_validation'] = validation_results
        
        # Comprehensive scoring
        perturbation_score = self._calculate_perturbation_robustness_score(suite_results)
        suite_results['overall_perturbation_score'] = perturbation_score
        
        return suite_results
    
    def run_microstructure_noise_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        noise_types: List[str] = None,
        n_simulations: int = 200,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run microstructure noise test to simulate bid-ask bounce and tick effects.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            noise_types: Types of microstructure noise to test
            n_simulations: Number of simulations per noise type
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with microstructure noise test results
        """
        if noise_types is None:
            noise_types = ['bid_ask_bounce', 'tick_clustering', 'price_discretization', 'round_number_bias']
        
        microstructure_results = {}
        
        for noise_type in noise_types:
            type_results = []
            
            for i in range(n_simulations):
                try:
                    # Apply microstructure noise
                    noisy_data = self._apply_microstructure_noise(self.data, noise_type)
                    
                    # Run backtest
                    metrics = backtest_function(noisy_data, parameters)
                    
                    # Add metadata
                    metrics['simulation'] = i
                    metrics['noise_type'] = noise_type
                    
                    type_results.append(metrics)
                    
                except Exception as e:
                    failed_metrics = {
                        'simulation': i,
                        'noise_type': noise_type,
                        'error': str(e),
                        'total_return': float('-inf'),
                        'sharpe_ratio': float('-inf'),
                    }
                    type_results.append(failed_metrics)
            
            # Calculate statistics for this noise type
            type_stats = self._calculate_level_statistics(type_results)
            
            microstructure_results[noise_type] = {
                'results': type_results,
                'statistics': type_stats,
                'robustness_score': self._calculate_microstructure_robustness(type_results),
            }
        
        return microstructure_results
    
    def _apply_microstructure_noise(self, data: pd.DataFrame, noise_type: str) -> pd.DataFrame:
        """
        Apply microstructure noise to simulate market microstructure effects.
        """
        noisy_data = data.copy()
        
        if noise_type == 'bid_ask_bounce':
            # Simulate bid-ask bounce with random direction changes
            spread_size = self.price_volatility * 0.5  # Typical spread
            
            for col in ['open', 'high', 'low', 'close']:
                if col in noisy_data.columns:
                    # Random bounce between bid and ask
                    bounce_direction = np.random.choice([-1, 1], size=len(noisy_data))
                    bounce_magnitude = np.random.uniform(0, spread_size, len(noisy_data))
                    
                    noisy_data[col] += bounce_direction * bounce_magnitude * noisy_data[col]
        
        elif noise_type == 'tick_clustering':
            # Simulate clustering around certain price levels
            tick_size = self.price_volatility * 0.1
            
            for col in ['open', 'high', 'low', 'close']:
                if col in noisy_data.columns:
                    prices = noisy_data[col]
                    
                    # Round to nearest tick with some clustering bias
                    rounded_prices = np.round(prices / tick_size) * tick_size
                    
                    # Add clustering bias toward round numbers
                    cluster_bias = np.random.exponential(0.5, len(prices))
                    clustering_effect = np.random.choice([-1, 1], size=len(prices)) * cluster_bias * tick_size
                    
                    noisy_data[col] = rounded_prices + clustering_effect
        
        elif noise_type == 'price_discretization':
            # Simulate price discretization effects
            min_tick = self.price_volatility * 0.05
            
            for col in ['open', 'high', 'low', 'close']:
                if col in noisy_data.columns:
                    # Discretize prices to minimum tick size
                    noisy_data[col] = np.round(noisy_data[col] / min_tick) * min_tick
        
        elif noise_type == 'round_number_bias':
            # Simulate bias toward round numbers
            for col in ['open', 'high', 'low', 'close']:
                if col in noisy_data.columns:
                    prices = noisy_data[col]
                    
                    # Identify round numbers (multiples of 10, 100, etc.)
                    for round_level in [10, 100, 1000]:
                        round_numbers = np.round(prices / round_level) * round_level
                        distance_to_round = np.abs(prices - round_numbers)
                        
                        # Apply bias toward round numbers (stronger for closer prices)
                        bias_strength = np.exp(-distance_to_round / (round_level * 0.1))
                        bias_direction = np.sign(round_numbers - prices)
                        
                        bias_magnitude = bias_strength * bias_direction * distance_to_round * 0.3
                        noisy_data[col] += bias_magnitude
        
        return noisy_data
    
    def _calculate_microstructure_robustness(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate robustness score for microstructure noise effects.
        """
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return 0.0
        
        returns = [r.get('total_return', 0) for r in valid_results]
        
        # Robustness components
        success_rate = len(valid_results) / len(results)
        performance_stability = 1.0 / (1.0 + np.std(returns)) if returns else 0
        positive_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        
        # Weighted robustness score
        robustness_score = (
            0.4 * success_rate +
            0.4 * performance_stability +
            0.2 * positive_rate
        )
        
        return min(1.0, robustness_score)