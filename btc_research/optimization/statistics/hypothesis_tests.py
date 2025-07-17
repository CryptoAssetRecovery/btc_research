"""
Hypothesis testing for trading strategy performance.

This module provides statistical hypothesis tests to assess the significance
of strategy performance and compare different strategies, including multiple
comparison corrections and advanced statistical methods.
"""

from typing import List, Optional, Dict, Any, Union, Tuple

import numpy as np

from btc_research.optimization.base import BaseStatisticsTest
from btc_research.optimization.types import StatisticsResult

__all__ = [
    "TTestStatistics",
    "WilcoxonTestStatistics",
    "KolmogorovSmirnovTestStatistics",
    "MultipleComparisonCorrection",
    "PowerAnalysis",
    "BootstrapTest",
    "PermutationTest",
    "DieboldMarianoTest",
    "WhiteRealityCheck",
]


class TTestStatistics(BaseStatisticsTest):
    """
    T-test for comparing strategy returns.
    
    Performs one-sample or two-sample t-tests to assess if strategy
    returns are significantly different from zero or between strategies.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        null_hypothesis_mean: float = 0.0,
        **kwargs,
    ) -> StatisticsResult:
        """
        Run t-test on sample data.
        
        Args:
            sample1: First sample (e.g., strategy returns)
            sample2: Second sample for two-sample test (optional)
            null_hypothesis_mean: Mean under null hypothesis for one-sample test
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        sample1_array = np.array(sample1)
        
        if sample2 is None:
            # One-sample t-test
            return self._one_sample_ttest(sample1_array, null_hypothesis_mean)
        else:
            # Two-sample t-test
            sample2_array = np.array(sample2)
            return self._two_sample_ttest(sample1_array, sample2_array)
    
    def _one_sample_ttest(self, sample: np.ndarray, mu0: float) -> StatisticsResult:
        """
        Perform one-sample t-test.
        
        Args:
            sample: Sample data
            mu0: Null hypothesis mean
            
        Returns:
            Statistical test result
        """
        n = len(sample)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)  # Sample standard deviation
        
        # Calculate t-statistic
        if sample_std == 0:
            t_statistic = float('inf') if sample_mean != mu0 else 0.0
        else:
            t_statistic = (sample_mean - mu0) / (sample_std / np.sqrt(n))
        
        # Degrees of freedom
        df = n - 1
        
        # Calculate p-value (two-tailed test)
        p_value = self._t_distribution_p_value(abs(t_statistic), df) * 2
        
        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        t_critical = self._t_distribution_critical_value(alpha / 2, df)
        margin_error = t_critical * (sample_std / np.sqrt(n))
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        # Effect size (Cohen's d)
        effect_size = (sample_mean - mu0) / sample_std if sample_std > 0 else 0.0
        
        return StatisticsResult(
            test_name="one_sample_t_test",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            critical_value=t_critical,
            sample_size=n,
            assumptions_met={
                "normality": self._check_normality(sample),
                "independence": True,  # Assumed
            },
        )
    
    def _two_sample_ttest(self, sample1: np.ndarray, sample2: np.ndarray) -> StatisticsResult:
        """
        Perform two-sample t-test (Welch's t-test for unequal variances).
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            Statistical test result
        """
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        
        # Welch's t-test (unequal variances)
        pooled_se = np.sqrt(var1 / n1 + var2 / n2)
        
        if pooled_se == 0:
            t_statistic = float('inf') if mean1 != mean2 else 0.0
            df = min(n1, n2) - 1
        else:
            t_statistic = (mean1 - mean2) / pooled_se
            
            # Welch-Satterthwaite degrees of freedom
            if var1 > 0 and var2 > 0:
                df = (var1 / n1 + var2 / n2) ** 2 / (
                    (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
                )
            else:
                df = min(n1, n2) - 1
        
        # Calculate p-value (two-tailed test)
        p_value = self._t_distribution_p_value(abs(t_statistic), df) * 2
        
        # Calculate confidence interval for difference of means
        alpha = 1 - self.confidence_level
        t_critical = self._t_distribution_critical_value(alpha / 2, df)
        margin_error = t_critical * pooled_se
        diff_mean = mean1 - mean2
        ci_lower = diff_mean - margin_error
        ci_upper = diff_mean + margin_error
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        return StatisticsResult(
            test_name="two_sample_t_test",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            critical_value=t_critical,
            sample_size=n1 + n2,
            assumptions_met={
                "normality_sample1": self._check_normality(sample1),
                "normality_sample2": self._check_normality(sample2),
                "independence": True,  # Assumed
            },
        )
    
    def _t_distribution_p_value(self, t_stat: float, df: float) -> float:
        """
        Calculate p-value for t-distribution (simplified approximation).
        
        Args:
            t_stat: T-statistic
            df: Degrees of freedom
            
        Returns:
            P-value
        """
        # Simplified approximation - in practice, use scipy.stats.t
        if df <= 0:
            return 0.5
        
        # For large df, approximate with normal distribution
        if df > 30:
            return self._normal_cdf(-abs(t_stat))
        
        # Simple approximation for smaller df
        # This is not accurate - use proper statistical library in production
        x = t_stat / np.sqrt(df)
        return max(0.001, min(0.999, 0.5 * (1 + np.tanh(x))))
    
    def _t_distribution_critical_value(self, alpha: float, df: float) -> float:
        """
        Calculate critical value for t-distribution (simplified approximation).
        
        Args:
            alpha: Significance level
            df: Degrees of freedom
            
        Returns:
            Critical value
        """
        # Simplified approximation - in practice, use scipy.stats.t
        if df <= 0:
            return 1.96  # Normal approximation
        
        # For large df, approximate with normal distribution
        if df > 30:
            return self._normal_critical_value(alpha)
        
        # Simple approximation for t-distribution
        # This is not accurate - use proper statistical library in production
        z_alpha = self._normal_critical_value(alpha)
        correction = 1 + (z_alpha ** 2 + 1) / (4 * df)
        return z_alpha * correction
    
    def _normal_cdf(self, x: float) -> float:
        """Simplified normal CDF approximation."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))
    
    def _normal_critical_value(self, alpha: float) -> float:
        """Simplified normal critical value approximation."""
        # Approximate inverse normal CDF for common alpha values
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
    
    def _check_normality(self, sample: np.ndarray) -> bool:
        """
        Simple normality check using skewness and kurtosis.
        
        Args:
            sample: Sample data
            
        Returns:
            True if data appears roughly normal
        """
        if len(sample) < 3:
            return False
        
        # Calculate skewness and kurtosis
        mean = np.mean(sample)
        std = np.std(sample)
        
        if std == 0:
            return len(set(sample)) == 1  # All values the same
        
        skewness = np.mean(((sample - mean) / std) ** 3)
        kurtosis = np.mean(((sample - mean) / std) ** 4) - 3
        
        # Simple thresholds for normality
        return abs(skewness) < 2 and abs(kurtosis) < 7


class WilcoxonTestStatistics(BaseStatisticsTest):
    """
    Wilcoxon signed-rank test for non-parametric comparison.
    
    Used when data doesn't meet normality assumptions for t-tests.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        **kwargs,
    ) -> StatisticsResult:
        """
        Run Wilcoxon test on sample data.
        
        Args:
            sample1: First sample
            sample2: Second sample (optional, for paired test)
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        sample1_array = np.array(sample1)
        
        if sample2 is None:
            # One-sample Wilcoxon test (against median = 0)
            return self._one_sample_wilcoxon(sample1_array)
        else:
            # Paired Wilcoxon test
            sample2_array = np.array(sample2)
            if len(sample1_array) != len(sample2_array):
                raise ValueError("Samples must have equal length for paired Wilcoxon test")
            return self._paired_wilcoxon(sample1_array, sample2_array)
    
    def _one_sample_wilcoxon(self, sample: np.ndarray) -> StatisticsResult:
        """
        Perform one-sample Wilcoxon signed-rank test.
        
        Args:
            sample: Sample data
            
        Returns:
            Statistical test result
        """
        # Remove zeros
        nonzero_sample = sample[sample != 0]
        n = len(nonzero_sample)
        
        if n == 0:
            return StatisticsResult(
                test_name="one_sample_wilcoxon",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=len(sample),
            )
        
        # Calculate ranks of absolute values
        abs_values = np.abs(nonzero_sample)
        ranks = self._calculate_ranks(abs_values)
        
        # Sum of positive ranks
        positive_ranks = ranks[nonzero_sample > 0]
        w_statistic = np.sum(positive_ranks)
        
        # Expected value and variance under null hypothesis
        expected_w = n * (n + 1) / 4
        variance_w = n * (n + 1) * (2 * n + 1) / 24
        
        # Z-statistic (normal approximation for large n)
        if variance_w > 0:
            z_statistic = (w_statistic - expected_w) / np.sqrt(variance_w)
        else:
            z_statistic = 0.0
        
        # P-value (two-tailed)
        p_value = 2 * self._normal_cdf(-abs(z_statistic))
        
        return StatisticsResult(
            test_name="one_sample_wilcoxon",
            statistic=w_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(0.0, 0.0),  # Simplified
            sample_size=len(sample),
            assumptions_met={
                "symmetry": True,  # Assumed
                "independence": True,  # Assumed
            },
        )
    
    def _paired_wilcoxon(self, sample1: np.ndarray, sample2: np.ndarray) -> StatisticsResult:
        """
        Perform paired Wilcoxon signed-rank test.
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            Statistical test result
        """
        # Calculate differences
        differences = sample1 - sample2
        
        # Remove zeros
        nonzero_diffs = differences[differences != 0]
        n = len(nonzero_diffs)
        
        if n == 0:
            return StatisticsResult(
                test_name="paired_wilcoxon",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=len(sample1),
            )
        
        # Calculate ranks of absolute differences
        abs_diffs = np.abs(nonzero_diffs)
        ranks = self._calculate_ranks(abs_diffs)
        
        # Sum of positive ranks
        positive_ranks = ranks[nonzero_diffs > 0]
        w_statistic = np.sum(positive_ranks)
        
        # Expected value and variance under null hypothesis
        expected_w = n * (n + 1) / 4
        variance_w = n * (n + 1) * (2 * n + 1) / 24
        
        # Z-statistic (normal approximation for large n)
        if variance_w > 0:
            z_statistic = (w_statistic - expected_w) / np.sqrt(variance_w)
        else:
            z_statistic = 0.0
        
        # P-value (two-tailed)
        p_value = 2 * self._normal_cdf(-abs(z_statistic))
        
        return StatisticsResult(
            test_name="paired_wilcoxon",
            statistic=w_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(0.0, 0.0),  # Simplified
            sample_size=len(sample1),
            assumptions_met={
                "symmetry": True,  # Assumed
                "independence": True,  # Assumed
            },
        )
    
    def _calculate_ranks(self, values: np.ndarray) -> np.ndarray:
        """
        Calculate ranks with ties handled by average ranking.
        
        Args:
            values: Values to rank
            
        Returns:
            Array of ranks
        """
        sorted_indices = np.argsort(values)
        ranks = np.empty_like(sorted_indices, dtype=float)
        
        i = 0
        while i < len(values):
            # Find tied values
            j = i
            while j < len(values) and values[sorted_indices[j]] == values[sorted_indices[i]]:
                j += 1
            
            # Assign average rank to tied values
            avg_rank = (i + j + 1) / 2
            for k in range(i, j):
                ranks[sorted_indices[k]] = avg_rank
            
            i = j
        
        return ranks
    
    def _normal_cdf(self, x: float) -> float:
        """Simplified normal CDF approximation."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))


class KolmogorovSmirnovTestStatistics(BaseStatisticsTest):
    """
    Kolmogorov-Smirnov test for distribution comparison.
    
    Tests whether two samples come from the same distribution or
    whether a sample comes from a specified distribution.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        reference_cdf=None,
        **kwargs,
    ) -> StatisticsResult:
        """
        Run Kolmogorov-Smirnov test.
        
        Args:
            sample1: First sample
            sample2: Second sample for two-sample test (optional)
            reference_cdf: Reference CDF function for one-sample test
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        sample1_array = np.array(sample1)
        
        if sample2 is None:
            if reference_cdf is None:
                # Default: test against normal distribution
                mean1 = np.mean(sample1_array)
                std1 = np.std(sample1_array)
                reference_cdf = lambda x: self._normal_cdf((x - mean1) / std1)
            
            return self._one_sample_ks(sample1_array, reference_cdf)
        else:
            sample2_array = np.array(sample2)
            return self._two_sample_ks(sample1_array, sample2_array)
    
    def _one_sample_ks(self, sample: np.ndarray, reference_cdf) -> StatisticsResult:
        """
        Perform one-sample Kolmogorov-Smirnov test.
        
        Args:
            sample: Sample data
            reference_cdf: Reference cumulative distribution function
            
        Returns:
            Statistical test result
        """
        n = len(sample)
        sorted_sample = np.sort(sample)
        
        # Calculate empirical CDF
        empirical_cdf = np.arange(1, n + 1) / n
        
        # Calculate reference CDF values
        reference_values = np.array([reference_cdf(x) for x in sorted_sample])
        
        # Calculate D statistic (maximum difference)
        d_plus = np.max(empirical_cdf - reference_values)
        d_minus = np.max(reference_values - np.concatenate([[0], empirical_cdf[:-1]]))
        d_statistic = max(d_plus, d_minus)
        
        # Calculate p-value (Kolmogorov distribution approximation)
        sqrt_n = np.sqrt(n)
        p_value = self._kolmogorov_p_value(d_statistic * sqrt_n)
        
        return StatisticsResult(
            test_name="one_sample_ks",
            statistic=d_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(0.0, 0.0),  # Not applicable
            sample_size=n,
            assumptions_met={
                "continuity": True,  # Assumed
                "independence": True,  # Assumed
            },
        )
    
    def _two_sample_ks(self, sample1: np.ndarray, sample2: np.ndarray) -> StatisticsResult:
        """
        Perform two-sample Kolmogorov-Smirnov test.
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            Statistical test result
        """
        n1, n2 = len(sample1), len(sample2)
        
        # Combine and sort all values
        combined = np.concatenate([sample1, sample2])
        unique_values = np.unique(combined)
        
        # Calculate empirical CDFs
        cdf1 = np.array([np.mean(sample1 <= x) for x in unique_values])
        cdf2 = np.array([np.mean(sample2 <= x) for x in unique_values])
        
        # Calculate D statistic
        d_statistic = np.max(np.abs(cdf1 - cdf2))
        
        # Calculate p-value
        effective_n = np.sqrt(n1 * n2 / (n1 + n2))
        p_value = self._kolmogorov_p_value(d_statistic * effective_n)
        
        return StatisticsResult(
            test_name="two_sample_ks",
            statistic=d_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(0.0, 0.0),  # Not applicable
            sample_size=n1 + n2,
            assumptions_met={
                "continuity": True,  # Assumed
                "independence": True,  # Assumed
            },
        )
    
    def _kolmogorov_p_value(self, x: float) -> float:
        """
        Approximate p-value for Kolmogorov distribution.
        
        Args:
            x: Test statistic * sqrt(n)
            
        Returns:
            P-value
        """
        if x <= 0:
            return 1.0
        
        # Asymptotic formula for large x
        if x > 3:
            return 2 * np.exp(-2 * x ** 2)
        
        # Series approximation for moderate x
        p_value = 0.0
        for k in range(1, 100):  # Truncate series
            term = (-1) ** (k - 1) * np.exp(-2 * k ** 2 * x ** 2)
            p_value += term
            if abs(term) < 1e-10:  # Convergence check
                break
        
        return max(0.0, min(1.0, 2 * p_value))
    
    def _normal_cdf(self, x: float) -> float:
        """Simplified normal CDF approximation."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))


class MultipleComparisonCorrection:
    """
    Multiple comparison correction methods for controlling family-wise error rate.
    
    Implements various methods to adjust p-values when performing multiple
    hypothesis tests simultaneously.
    
    References:
        - Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo delle probabilità
        - Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate
    """
    
    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of uncorrected p-values
            alpha: Family-wise error rate
            
        Returns:
            Tuple of (rejection_decisions, corrected_p_values)
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if n_tests == 0:
            return [], []
        
        # Bonferroni correction
        corrected_alpha = alpha / n_tests
        corrected_p_values = np.minimum(p_values * n_tests, 1.0)
        
        # Rejection decisions
        rejections = corrected_p_values < alpha
        
        return rejections.tolist(), corrected_p_values.tolist()
    
    @staticmethod
    def benjamini_hochberg_fdr(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
        """
        Apply Benjamini-Hochberg False Discovery Rate correction.
        
        Args:
            p_values: List of uncorrected p-values
            alpha: False discovery rate
            
        Returns:
            Tuple of (rejection_decisions, corrected_p_values)
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if n_tests == 0:
            return [], []
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # BH procedure
        rejections = np.zeros(n_tests, dtype=bool)
        
        # Find largest k such that P(k) <= (k/m) * alpha
        for k in range(n_tests - 1, -1, -1):
            if sorted_p_values[k] <= (k + 1) / n_tests * alpha:
                # Reject all hypotheses up to and including k
                rejections[sorted_indices[:k + 1]] = True
                break
        
        # Corrected p-values
        corrected_p_values = np.zeros(n_tests)
        for i in range(n_tests):
            corrected_p_values[sorted_indices[i]] = min(
                sorted_p_values[i] * n_tests / (i + 1),
                1.0
            )
        
        # Ensure monotonicity
        for i in range(n_tests - 2, -1, -1):
            corrected_p_values[sorted_indices[i]] = min(
                corrected_p_values[sorted_indices[i]],
                corrected_p_values[sorted_indices[i + 1]]
            )
        
        return rejections.tolist(), corrected_p_values.tolist()
    
    @staticmethod
    def holm_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
        """
        Apply Holm step-down correction for multiple comparisons.
        
        Args:
            p_values: List of uncorrected p-values
            alpha: Family-wise error rate
            
        Returns:
            Tuple of (rejection_decisions, corrected_p_values)
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if n_tests == 0:
            return [], []
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Holm procedure
        rejections = np.zeros(n_tests, dtype=bool)
        corrected_p_values = np.zeros(n_tests)
        
        for i in range(n_tests):
            corrected_alpha = alpha / (n_tests - i)
            corrected_p_values[sorted_indices[i]] = min(
                sorted_p_values[i] * (n_tests - i),
                1.0
            )
            
            if sorted_p_values[i] <= corrected_alpha:
                rejections[sorted_indices[i]] = True
            else:
                # Stop at first non-rejection
                break
        
        return rejections.tolist(), corrected_p_values.tolist()


class PowerAnalysis:
    """
    Statistical power analysis for sample size determination and effect size calculation.
    
    Provides methods to calculate statistical power, required sample sizes,
    and detectable effect sizes for various statistical tests.
    
    References:
        - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
    """
    
    @staticmethod
    def t_test_power(
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05,
        two_sided: bool = True
    ) -> float:
        """
        Calculate statistical power for t-test.
        
        Args:
            effect_size: Standardized effect size (Cohen's d)
            sample_size: Sample size
            alpha: Significance level
            two_sided: Whether test is two-sided
            
        Returns:
            Statistical power (probability of detecting true effect)
        """
        if sample_size <= 1:
            return 0.0
        
        # Degrees of freedom
        df = sample_size - 1
        
        # Critical value
        if two_sided:
            t_critical = PowerAnalysis._t_critical_value(alpha / 2, df)
        else:
            t_critical = PowerAnalysis._t_critical_value(alpha, df)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size)
        
        # Power calculation using non-central t-distribution approximation
        if two_sided:
            power = (
                1 - PowerAnalysis._t_cdf(t_critical, df, ncp) +
                PowerAnalysis._t_cdf(-t_critical, df, ncp)
            )
        else:
            power = 1 - PowerAnalysis._t_cdf(t_critical, df, ncp)
        
        return max(0.0, min(1.0, power))
    
    @staticmethod
    def required_sample_size(
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        two_sided: bool = True
    ) -> int:
        """
        Calculate required sample size for desired power.
        
        Args:
            effect_size: Standardized effect size (Cohen's d)
            power: Desired statistical power
            alpha: Significance level
            two_sided: Whether test is two-sided
            
        Returns:
            Required sample size
        """
        if effect_size == 0 or power <= 0 or power >= 1:
            return float('inf')
        
        # Binary search for sample size
        min_n, max_n = 2, 10000
        
        while max_n - min_n > 1:
            mid_n = (min_n + max_n) // 2
            calculated_power = PowerAnalysis.t_test_power(
                effect_size, mid_n, alpha, two_sided
            )
            
            if calculated_power < power:
                min_n = mid_n
            else:
                max_n = mid_n
        
        return max_n
    
    @staticmethod
    def detectable_effect_size(
        sample_size: int,
        power: float = 0.8,
        alpha: float = 0.05,
        two_sided: bool = True
    ) -> float:
        """
        Calculate minimum detectable effect size for given sample size and power.
        
        Args:
            sample_size: Available sample size
            power: Desired statistical power
            alpha: Significance level
            two_sided: Whether test is two-sided
            
        Returns:
            Minimum detectable effect size
        """
        if sample_size <= 1 or power <= 0 or power >= 1:
            return float('inf')
        
        # Binary search for effect size
        min_effect, max_effect = 0.0, 5.0
        tolerance = 1e-6
        
        while max_effect - min_effect > tolerance:
            mid_effect = (min_effect + max_effect) / 2
            calculated_power = PowerAnalysis.t_test_power(
                mid_effect, sample_size, alpha, two_sided
            )
            
            if calculated_power < power:
                min_effect = mid_effect
            else:
                max_effect = mid_effect
        
        return max_effect
    
    @staticmethod
    def _t_critical_value(alpha: float, df: float) -> float:
        """Calculate critical value for t-distribution."""
        # Simplified approximation
        if df > 30:
            return PowerAnalysis._normal_critical_value(alpha)
        
        z_alpha = PowerAnalysis._normal_critical_value(alpha)
        correction = 1 + (z_alpha ** 2 + 1) / (4 * df)
        return z_alpha * correction
    
    @staticmethod
    def _normal_critical_value(alpha: float) -> float:
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
    
    @staticmethod
    def _t_cdf(x: float, df: float, ncp: float = 0.0) -> float:
        """Approximate CDF for (non-central) t-distribution."""
        # Very simplified approximation - use proper statistical library in production
        if ncp == 0:
            # Central t-distribution
            if df > 30:
                return 0.5 * (1 + np.tanh(x * 0.7978845608))
            else:
                # Crude approximation
                return 0.5 * (1 + np.tanh(x * 0.7978845608 * np.sqrt(df / (df + 2))))
        else:
            # Non-central t-distribution (very crude approximation)
            shifted_x = x - ncp / np.sqrt(df)
            return PowerAnalysis._t_cdf(shifted_x, df, 0.0)