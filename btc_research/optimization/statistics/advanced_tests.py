"""
Advanced statistical tests for trading strategy analysis.

This module provides bootstrap, permutation, and other advanced statistical
methods for robust hypothesis testing without distributional assumptions.
"""

from typing import List, Optional, Dict, Any, Union

import numpy as np

from btc_research.optimization.base import BaseStatisticsTest
from btc_research.optimization.types import StatisticsResult

__all__ = [
    "BootstrapTest",
    "PermutationTest", 
    "DieboldMarianoTest",
    "WhiteRealityCheck",
]


class BootstrapTest(BaseStatisticsTest):
    """
    Bootstrap-based hypothesis testing for robust statistical inference.
    
    Implements bootstrap resampling methods for hypothesis testing that don't
    rely on distributional assumptions.
    
    References:
        - Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        test_statistic: str = "mean",
        n_bootstrap: int = 10000,
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Run bootstrap hypothesis test.
        
        Args:
            sample1: First sample
            sample2: Second sample for two-sample test (optional)
            test_statistic: Type of test statistic ('mean', 'median', 'std', 'sharpe')
            n_bootstrap: Number of bootstrap resamples
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        sample1_array = np.array(sample1)
        
        if sample2 is None:
            return self._one_sample_bootstrap(sample1_array, test_statistic, n_bootstrap)
        else:
            sample2_array = np.array(sample2)
            return self._two_sample_bootstrap(
                sample1_array, sample2_array, test_statistic, n_bootstrap
            )
    
    def _one_sample_bootstrap(
        self, 
        sample: np.ndarray, 
        test_statistic: str, 
        n_bootstrap: int
    ) -> StatisticsResult:
        """
        One-sample bootstrap test (test if statistic differs from zero).
        
        Args:
            sample: Sample data
            test_statistic: Type of test statistic
            n_bootstrap: Number of bootstrap resamples
            
        Returns:
            Statistical test result
        """
        n = len(sample)
        
        # Calculate observed test statistic
        observed_stat = self._calculate_statistic(sample, test_statistic)
        
        # Bootstrap resampling under null hypothesis (centered at zero)
        sample_centered = sample - np.mean(sample)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(sample_centered, size=n, replace=True)
            bootstrap_stat = self._calculate_statistic(bootstrap_sample, test_statistic)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
        
        # Confidence interval using percentile method
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        
        # Effect size (standardized)
        if np.std(bootstrap_stats) > 0:
            effect_size = observed_stat / np.std(bootstrap_stats)
        else:
            effect_size = 0.0
        
        return StatisticsResult(
            test_name=f"bootstrap_{test_statistic}_test",
            statistic=observed_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            sample_size=n,
            assumptions_met={
                "exchangeability": True,  # Assumed for bootstrap
                "independence": True,     # Assumed
            },
        )
    
    def _two_sample_bootstrap(
        self, 
        sample1: np.ndarray, 
        sample2: np.ndarray, 
        test_statistic: str, 
        n_bootstrap: int
    ) -> StatisticsResult:
        """
        Two-sample bootstrap test.
        
        Args:
            sample1: First sample
            sample2: Second sample
            test_statistic: Type of test statistic
            n_bootstrap: Number of bootstrap resamples
            
        Returns:
            Statistical test result
        """
        n1, n2 = len(sample1), len(sample2)
        
        # Calculate observed difference
        stat1 = self._calculate_statistic(sample1, test_statistic)
        stat2 = self._calculate_statistic(sample2, test_statistic)
        observed_diff = stat1 - stat2
        
        # Bootstrap resampling under null hypothesis (no difference)
        pooled_sample = np.concatenate([sample1, sample2])
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Resample from pooled data
            bootstrap_pooled = np.random.choice(pooled_sample, size=n1 + n2, replace=True)
            bootstrap_sample1 = bootstrap_pooled[:n1]
            bootstrap_sample2 = bootstrap_pooled[n1:]
            
            bootstrap_stat1 = self._calculate_statistic(bootstrap_sample1, test_statistic)
            bootstrap_stat2 = self._calculate_statistic(bootstrap_sample2, test_statistic)
            bootstrap_diff = bootstrap_stat1 - bootstrap_stat2
            bootstrap_diffs.append(bootstrap_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)
        
        # Effect size
        if np.std(bootstrap_diffs) > 0:
            effect_size = observed_diff / np.std(bootstrap_diffs)
        else:
            effect_size = 0.0
        
        return StatisticsResult(
            test_name=f"bootstrap_{test_statistic}_comparison",
            statistic=observed_diff,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            sample_size=n1 + n2,
        )
    
    def _calculate_statistic(self, sample: np.ndarray, stat_type: str) -> float:
        """Calculate specified test statistic."""
        if stat_type == "mean":
            return np.mean(sample)
        elif stat_type == "median":
            return np.median(sample)
        elif stat_type == "std":
            return np.std(sample, ddof=1)
        elif stat_type == "sharpe":
            if np.std(sample) == 0:
                return 0.0
            return np.mean(sample) / np.std(sample, ddof=1)
        else:
            raise ValueError(f"Unknown test statistic: {stat_type}")


class PermutationTest(BaseStatisticsTest):
    """
    Permutation-based hypothesis testing for exact statistical inference.
    
    Implements permutation tests that provide exact p-values under the null
    hypothesis of exchangeability.
    
    References:
        - Good, P. (2005). Permutation, Parametric and Bootstrap Tests of Hypotheses
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        test_statistic: str = "mean_difference",
        n_permutations: int = 10000,
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Run permutation hypothesis test.
        
        Args:
            sample1: First sample
            sample2: Second sample (required for permutation test)
            test_statistic: Type of test statistic
            n_permutations: Number of permutations
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        if sample2 is None:
            raise ValueError("Permutation test requires two samples")
        
        self._validate_samples(sample1, sample2)
        
        sample1_array = np.array(sample1)
        sample2_array = np.array(sample2)
        
        return self._permutation_test(
            sample1_array, sample2_array, test_statistic, n_permutations
        )
    
    def _permutation_test(
        self, 
        sample1: np.ndarray, 
        sample2: np.ndarray, 
        test_statistic: str, 
        n_permutations: int
    ) -> StatisticsResult:
        """
        Perform permutation test for two samples.
        
        Args:
            sample1: First sample
            sample2: Second sample
            test_statistic: Type of test statistic
            n_permutations: Number of permutations
            
        Returns:
            Statistical test result
        """
        n1, n2 = len(sample1), len(sample2)
        n_total = n1 + n2
        
        # Calculate observed test statistic
        observed_stat = self._calculate_permutation_statistic(
            sample1, sample2, test_statistic
        )
        
        # Combine samples for permutation
        combined_sample = np.concatenate([sample1, sample2])
        
        # Generate permutation distribution
        permutation_stats = []
        
        for _ in range(n_permutations):
            # Random permutation
            permuted_indices = np.random.permutation(n_total)
            permuted_sample1 = combined_sample[permuted_indices[:n1]]
            permuted_sample2 = combined_sample[permuted_indices[n1:]]
            
            permuted_stat = self._calculate_permutation_statistic(
                permuted_sample1, permuted_sample2, test_statistic
            )
            permutation_stats.append(permuted_stat)
        
        permutation_stats = np.array(permutation_stats)
        
        # Calculate exact p-value
        if test_statistic in ["mean_difference", "median_difference"]:
            # Two-tailed test
            p_value = np.mean(np.abs(permutation_stats) >= np.abs(observed_stat))
        else:
            # One-tailed test (assuming larger values are more extreme)
            p_value = np.mean(permutation_stats >= observed_stat)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(permutation_stats, alpha / 2 * 100)
        ci_upper = np.percentile(permutation_stats, (1 - alpha / 2) * 100)
        
        # Effect size
        if np.std(permutation_stats) > 0:
            effect_size = observed_stat / np.std(permutation_stats)
        else:
            effect_size = 0.0
        
        return StatisticsResult(
            test_name=f"permutation_{test_statistic}_test",
            statistic=observed_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            sample_size=n1 + n2,
            assumptions_met={
                "exchangeability": True,  # Required for permutation test
                "independence": True,     # Assumed
            },
        )
    
    def _calculate_permutation_statistic(
        self, 
        sample1: np.ndarray, 
        sample2: np.ndarray, 
        stat_type: str
    ) -> float:
        """Calculate specified permutation test statistic."""
        if stat_type == "mean_difference":
            return np.mean(sample1) - np.mean(sample2)
        elif stat_type == "median_difference":
            return np.median(sample1) - np.median(sample2)
        elif stat_type == "ks_statistic":
            # Kolmogorov-Smirnov statistic
            combined = np.concatenate([sample1, sample2])
            unique_values = np.unique(combined)
            
            cdf1 = np.array([np.mean(sample1 <= x) for x in unique_values])
            cdf2 = np.array([np.mean(sample2 <= x) for x in unique_values])
            
            return np.max(np.abs(cdf1 - cdf2))
        else:
            raise ValueError(f"Unknown permutation test statistic: {stat_type}")


class DieboldMarianoTest(BaseStatisticsTest):
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Tests whether two forecasting methods have equal predictive accuracy
    by comparing their forecast errors.
    
    References:
        - Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        loss_function: str = "squared_error",
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Run Diebold-Mariano test for forecast comparison.
        
        Args:
            sample1: Forecast errors from first method
            sample2: Forecast errors from second method
            loss_function: Loss function type ('squared_error', 'absolute_error')
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        if sample2 is None:
            raise ValueError("Diebold-Mariano test requires two samples")
        
        self._validate_samples(sample1, sample2)
        
        errors1 = np.array(sample1)
        errors2 = np.array(sample2)
        
        if len(errors1) != len(errors2):
            raise ValueError("Error series must have equal length")
        
        return self._diebold_mariano_test(errors1, errors2, loss_function)
    
    def _diebold_mariano_test(
        self, 
        errors1: np.ndarray, 
        errors2: np.ndarray, 
        loss_function: str
    ) -> StatisticsResult:
        """
        Perform Diebold-Mariano test.
        
        Args:
            errors1: Forecast errors from first method
            errors2: Forecast errors from second method
            loss_function: Loss function type
            
        Returns:
            Statistical test result
        """
        n = len(errors1)
        
        # Calculate loss differentials
        if loss_function == "squared_error":
            loss_diff = errors1 ** 2 - errors2 ** 2
        elif loss_function == "absolute_error":
            loss_diff = np.abs(errors1) - np.abs(errors2)
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        # Test statistic
        mean_loss_diff = np.mean(loss_diff)
        
        # Calculate variance with Newey-West correction for autocorrelation
        variance = self._newey_west_variance(loss_diff)
        
        if variance <= 0:
            dm_statistic = 0.0
            p_value = 1.0
        else:
            dm_statistic = mean_loss_diff / np.sqrt(variance / n)
            # Asymptotically normal under null hypothesis
            p_value = 2 * (1 - self._normal_cdf(abs(dm_statistic)))
        
        # Confidence interval for mean loss differential
        alpha = 1 - self.confidence_level
        z_critical = self._normal_critical_value(alpha / 2)
        
        if variance > 0:
            margin_error = z_critical * np.sqrt(variance / n)
            ci_lower = mean_loss_diff - margin_error
            ci_upper = mean_loss_diff + margin_error
        else:
            ci_lower = ci_upper = mean_loss_diff
        
        return StatisticsResult(
            test_name="diebold_mariano_test",
            statistic=dm_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=mean_loss_diff,
            sample_size=n,
            assumptions_met={
                "stationarity": True,     # Assumed
                "ergodicity": True,       # Assumed
            },
        )
    
    def _newey_west_variance(self, series: np.ndarray, max_lags: Optional[int] = None) -> float:
        """
        Calculate Newey-West variance estimator for autocorrelated series.
        
        Args:
            series: Time series data
            max_lags: Maximum number of lags (if None, use automatic selection)
            
        Returns:
            Newey-West variance estimate
        """
        n = len(series)
        
        if max_lags is None:
            # Automatic lag selection (rule of thumb)
            max_lags = min(int(4 * (n / 100) ** (2 / 9)), n - 1)
        
        # Center the series
        centered_series = series - np.mean(series)
        
        # Calculate autocovariances
        variance = np.var(centered_series, ddof=1)
        
        for lag in range(1, max_lags + 1):
            if lag >= n:
                break
            
            # Bartlett kernel weight
            weight = 1 - lag / (max_lags + 1)
            
            # Autocovariance at lag
            autocovariance = np.mean(
                centered_series[:-lag] * centered_series[lag:]
            )
            
            variance += 2 * weight * autocovariance
        
        return max(variance, 1e-10)  # Ensure positive variance
    
    def _normal_cdf(self, x: float) -> float:
        """Simplified normal CDF approximation."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))
    
    def _normal_critical_value(self, alpha: float) -> float:
        """Simplified normal critical value approximation."""
        if alpha <= 0.001:
            return 3.29
        elif alpha <= 0.01:
            return 2.58
        elif alpha <= 0.025:
            return 1.96
        elif alpha <= 0.05:
            return 1.64
        elif alpha <= 0.1:
            return 1.28
        else:
            return 1.0


class WhiteRealityCheck(BaseStatisticsTest):
    """
    White's Reality Check for data snooping bias.
    
    Tests whether the best performing strategy from a universe of strategies
    has genuine predictive ability or is just the result of data snooping.
    
    References:
        - White, H. (2000). A reality check for data snooping
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        benchmark_returns: Optional[List[float]] = None,
        n_bootstrap: int = 10000,
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Run White's Reality Check test.
        
        Args:
            sample1: Returns of best strategy
            sample2: Not used (compatibility with base class)
            benchmark_returns: Returns of benchmark strategy
            n_bootstrap: Number of bootstrap samples
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        if benchmark_returns is None:
            # Use zero returns as benchmark
            benchmark_returns = [0.0] * len(sample1)
        
        self._validate_samples(sample1, benchmark_returns)
        
        strategy_returns = np.array(sample1)
        benchmark_returns = np.array(benchmark_returns)
        
        if len(strategy_returns) != len(benchmark_returns):
            min_len = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
        
        return self._white_reality_check(strategy_returns, benchmark_returns, n_bootstrap)
    
    def _white_reality_check(
        self, 
        strategy_returns: np.ndarray, 
        benchmark_returns: np.ndarray, 
        n_bootstrap: int
    ) -> StatisticsResult:
        """
        Perform White's Reality Check test.
        
        Args:
            strategy_returns: Returns of the strategy being tested
            benchmark_returns: Returns of benchmark strategy
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Statistical test result
        """
        n = len(strategy_returns)
        
        # Calculate performance differential
        performance_diff = strategy_returns - benchmark_returns
        observed_mean_diff = np.mean(performance_diff)
        
        # Bootstrap test under null hypothesis
        centered_diff = performance_diff - observed_mean_diff
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(centered_diff, size=n, replace=True)
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_means.append(bootstrap_mean)
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Reality Check p-value
        # P-value is the probability that the maximum of the bootstrap statistics
        # exceeds the observed statistic
        p_value = np.mean(bootstrap_means >= observed_mean_diff)
        
        # Test statistic (t-statistic)
        std_diff = np.std(performance_diff, ddof=1)
        if std_diff > 0:
            t_statistic = observed_mean_diff / (std_diff / np.sqrt(n))
        else:
            t_statistic = 0.0
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        
        # Effect size (Sharpe ratio of excess returns)
        if std_diff > 0:
            effect_size = observed_mean_diff / std_diff
        else:
            effect_size = 0.0
        
        return StatisticsResult(
            test_name="white_reality_check",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            sample_size=n,
            assumptions_met={
                "stationarity": True,     # Assumed
                "independence": True,     # Assumed for bootstrap
            },
        )