"""
Performance-specific statistical tests for trading strategies.

This module provides statistical tests specifically designed for
trading strategy performance metrics like Sharpe ratio, drawdown,
and return distributions.
"""

from typing import List, Optional

import numpy as np

from btc_research.optimization.base import BaseStatisticsTest
from btc_research.optimization.types import StatisticsResult

__all__ = [
    "SharpeRatioTestStatistics",
    "DrawdownTestStatistics", 
    "ReturnDistributionTestStatistics",
    "BootstrapPerformanceTests",
]


class SharpeRatioTestStatistics(BaseStatisticsTest):
    """
    Statistical tests for Sharpe ratio significance.
    
    Tests whether observed Sharpe ratios are statistically significant
    and compares Sharpe ratios between strategies.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        risk_free_rate: float = 0.0,
        **kwargs,
    ) -> StatisticsResult:
        """
        Test Sharpe ratio significance.
        
        Args:
            sample1: Returns of first strategy
            sample2: Returns of second strategy (for comparison)
            risk_free_rate: Risk-free rate (annual)
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        returns1 = np.array(sample1)
        excess_returns1 = returns1 - risk_free_rate / 252  # Daily risk-free rate
        
        if sample2 is None:
            # Test if Sharpe ratio is significantly different from zero
            return self._test_sharpe_significance(excess_returns1)
        else:
            # Compare Sharpe ratios between two strategies
            returns2 = np.array(sample2)
            excess_returns2 = returns2 - risk_free_rate / 252
            return self._compare_sharpe_ratios(excess_returns1, excess_returns2)
    
    def _test_sharpe_significance(self, excess_returns: np.ndarray) -> StatisticsResult:
        """
        Test if Sharpe ratio is significantly different from zero.
        
        Args:
            excess_returns: Excess returns over risk-free rate
            
        Returns:
            Statistical test result
        """
        n = len(excess_returns)
        
        if n <= 1:
            return StatisticsResult(
                test_name="sharpe_significance",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            sharpe_ratio = float('inf') if mean_excess > 0 else 0.0
            t_statistic = float('inf') if mean_excess > 0 else 0.0
        else:
            sharpe_ratio = mean_excess / std_excess
            # T-statistic for testing if Sharpe ratio is significantly different from 0
            t_statistic = sharpe_ratio * np.sqrt(n)
        
        # P-value (two-tailed)
        df = n - 1
        p_value = self._t_distribution_p_value(abs(t_statistic), df) * 2
        
        # Confidence interval for Sharpe ratio
        alpha = 1 - self.confidence_level
        t_critical = self._t_distribution_critical_value(alpha / 2, df)
        
        if std_excess > 0:
            # Jobson-Korkie confidence interval
            gamma3 = np.mean((excess_returns - mean_excess) ** 3) / (std_excess ** 3)
            gamma4 = np.mean((excess_returns - mean_excess) ** 4) / (std_excess ** 4)
            
            # Variance of Sharpe ratio estimator
            sharpe_var = (1 + 0.5 * sharpe_ratio ** 2 - sharpe_ratio * gamma3 + 
                         (gamma4 - 3) * sharpe_ratio ** 2 / 4) / n
            sharpe_se = np.sqrt(sharpe_var)
            
            ci_lower = sharpe_ratio - t_critical * sharpe_se
            ci_upper = sharpe_ratio + t_critical * sharpe_se
        else:
            ci_lower = ci_upper = sharpe_ratio
        
        # Annualized Sharpe ratio (assuming daily returns)
        annualized_sharpe = sharpe_ratio * np.sqrt(252)
        
        return StatisticsResult(
            test_name="sharpe_significance",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=annualized_sharpe,
            sample_size=n,
            assumptions_met={
                "normality": self._check_normality(excess_returns),
                "independence": True,  # Assumed
                "stationarity": True,  # Assumed
            },
        )
    
    def _compare_sharpe_ratios(self, returns1: np.ndarray, returns2: np.ndarray) -> StatisticsResult:
        """
        Compare Sharpe ratios between two strategies using Jobson-Korkie test.
        
        Args:
            returns1: Excess returns of first strategy
            returns2: Excess returns of second strategy
            
        Returns:
            Statistical test result
        """
        n1, n2 = len(returns1), len(returns2)
        min_n = min(n1, n2)
        
        if min_n <= 1:
            return StatisticsResult(
                test_name="sharpe_comparison",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n1 + n2,
            )
        
        # Calculate Sharpe ratios
        sharpe1 = np.mean(returns1) / np.std(returns1, ddof=1) if np.std(returns1) > 0 else 0
        sharpe2 = np.mean(returns2) / np.std(returns2, ddof=1) if np.std(returns2) > 0 else 0
        
        sharpe_diff = sharpe1 - sharpe2
        
        # Jobson-Korkie test statistic
        # This is a simplified version - full implementation would handle correlation
        if n1 == n2:
            # Paired case (same time periods)
            returns1_trunc = returns1[:min_n]
            returns2_trunc = returns2[:min_n]
            
            # Calculate correlation between strategies
            correlation = np.corrcoef(returns1_trunc, returns2_trunc)[0, 1]
            
            # Variance components
            var1 = np.var(returns1_trunc, ddof=1)
            var2 = np.var(returns2_trunc, ddof=1)
            
            # Simplified variance of Sharpe ratio difference
            if var1 > 0 and var2 > 0:
                sharpe_diff_var = (
                    (1 + 0.5 * sharpe1 ** 2) / min_n +
                    (1 + 0.5 * sharpe2 ** 2) / min_n -
                    2 * correlation * np.sqrt((1 + 0.5 * sharpe1 ** 2) * (1 + 0.5 * sharpe2 ** 2)) / min_n
                )
                
                if sharpe_diff_var > 0:
                    t_statistic = sharpe_diff / np.sqrt(sharpe_diff_var)
                else:
                    t_statistic = 0.0
            else:
                t_statistic = 0.0
        else:
            # Independent samples
            sharpe1_var = (1 + 0.5 * sharpe1 ** 2) / n1 if sharpe1 != 0 else 1 / n1
            sharpe2_var = (1 + 0.5 * sharpe2 ** 2) / n2 if sharpe2 != 0 else 1 / n2
            
            sharpe_diff_var = sharpe1_var + sharpe2_var
            
            if sharpe_diff_var > 0:
                t_statistic = sharpe_diff / np.sqrt(sharpe_diff_var)
            else:
                t_statistic = 0.0
        
        # Degrees of freedom (conservative estimate)
        df = min_n - 1
        
        # P-value (two-tailed)
        p_value = self._t_distribution_p_value(abs(t_statistic), df) * 2
        
        # Confidence interval for difference
        alpha = 1 - self.confidence_level
        t_critical = self._t_distribution_critical_value(alpha / 2, df)
        
        if sharpe_diff_var > 0:
            margin_error = t_critical * np.sqrt(sharpe_diff_var)
            ci_lower = sharpe_diff - margin_error
            ci_upper = sharpe_diff + margin_error
        else:
            ci_lower = ci_upper = sharpe_diff
        
        return StatisticsResult(
            test_name="sharpe_comparison",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=sharpe_diff,
            sample_size=n1 + n2,
            assumptions_met={
                "normality_1": self._check_normality(returns1),
                "normality_2": self._check_normality(returns2),
                "independence": True,  # Assumed
            },
        )
    
    def _t_distribution_p_value(self, t_stat: float, df: float) -> float:
        """Calculate p-value for t-distribution (simplified)."""
        if df <= 0:
            return 0.5
        
        if df > 30:
            return self._normal_cdf(-abs(t_stat))
        
        # Simple approximation
        x = t_stat / np.sqrt(df)
        return max(0.001, min(0.999, 0.5 * (1 + np.tanh(x))))
    
    def _t_distribution_critical_value(self, alpha: float, df: float) -> float:
        """Calculate critical value for t-distribution (simplified)."""
        if df <= 0:
            return 1.96
        
        if df > 30:
            return self._normal_critical_value(alpha)
        
        z_alpha = self._normal_critical_value(alpha)
        correction = 1 + (z_alpha ** 2 + 1) / (4 * df)
        return z_alpha * correction
    
    def _normal_cdf(self, x: float) -> float:
        """Simplified normal CDF."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))
    
    def _normal_critical_value(self, alpha: float) -> float:
        """Simplified normal critical value."""
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
        """Simple normality check."""
        if len(sample) < 3:
            return False
        
        mean = np.mean(sample)
        std = np.std(sample)
        
        if std == 0:
            return len(set(sample)) == 1
        
        skewness = np.mean(((sample - mean) / std) ** 3)
        kurtosis = np.mean(((sample - mean) / std) ** 4) - 3
        
        return abs(skewness) < 2 and abs(kurtosis) < 7


class DrawdownTestStatistics(BaseStatisticsTest):
    """
    Statistical tests for maximum drawdown analysis.
    
    Tests the statistical properties of drawdown metrics and
    compares drawdown characteristics between strategies.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        confidence_threshold: float = 0.95,
        **kwargs,
    ) -> StatisticsResult:
        """
        Test drawdown statistics.
        
        Args:
            sample1: Cumulative returns or equity curve of first strategy
            sample2: Cumulative returns or equity curve of second strategy
            confidence_threshold: Confidence level for drawdown estimates
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        equity1 = np.array(sample1)
        
        if sample2 is None:
            return self._analyze_single_drawdown(equity1, confidence_threshold)
        else:
            equity2 = np.array(sample2)
            return self._compare_drawdowns(equity1, equity2, confidence_threshold)
    
    def _analyze_single_drawdown(self, equity: np.ndarray, confidence_threshold: float) -> StatisticsResult:
        """
        Analyze drawdown characteristics of a single strategy.
        
        Args:
            equity: Equity curve values
            confidence_threshold: Confidence level
            
        Returns:
            Statistical test result
        """
        # Calculate drawdown series
        drawdowns = self._calculate_drawdowns(equity)
        max_drawdown = np.min(drawdowns)
        
        # Calculate drawdown duration
        drawdown_durations = self._calculate_drawdown_durations(drawdowns)
        max_duration = np.max(drawdown_durations) if drawdown_durations else 0
        
        # Calculate underwater periods
        underwater_periods = np.sum(drawdowns < -0.001)  # Periods with >0.1% drawdown
        underwater_ratio = underwater_periods / len(drawdowns) if len(drawdowns) > 0 else 0
        
        # Estimate confidence interval for max drawdown using historical simulation
        n_bootstrap = 1000
        bootstrap_max_dd = []
        
        for _ in range(n_bootstrap):
            # Bootstrap resample returns
            returns = np.diff(equity) / equity[:-1]
            bootstrap_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Reconstruct equity curve
            bootstrap_equity = np.cumprod(1 + bootstrap_returns)
            bootstrap_dd = self._calculate_drawdowns(bootstrap_equity)
            bootstrap_max_dd.append(np.min(bootstrap_dd))
        
        # Confidence interval for max drawdown
        alpha = 1 - confidence_threshold
        ci_lower = np.percentile(bootstrap_max_dd, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_max_dd, (1 - alpha / 2) * 100)
        
        # Test statistic: standardized max drawdown
        if len(bootstrap_max_dd) > 0:
            expected_max_dd = np.mean(bootstrap_max_dd)
            std_max_dd = np.std(bootstrap_max_dd)
            
            if std_max_dd > 0:
                t_statistic = (max_drawdown - expected_max_dd) / std_max_dd
            else:
                t_statistic = 0.0
        else:
            t_statistic = 0.0
        
        # P-value: probability of observing a drawdown this large or larger
        p_value = np.mean(np.array(bootstrap_max_dd) <= max_drawdown)
        
        return StatisticsResult(
            test_name="drawdown_analysis",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=confidence_threshold,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=max_drawdown,
            sample_size=len(equity),
            assumptions_met={
                "independence": False,  # Drawdowns are serially correlated
                "stationarity": True,   # Assumed
            },
        )
    
    def _compare_drawdowns(self, equity1: np.ndarray, equity2: np.ndarray, confidence_threshold: float) -> StatisticsResult:
        """
        Compare drawdown characteristics between two strategies.
        
        Args:
            equity1: Equity curve of first strategy
            equity2: Equity curve of second strategy
            confidence_threshold: Confidence level
            
        Returns:
            Statistical test result
        """
        # Calculate drawdowns for both strategies
        drawdowns1 = self._calculate_drawdowns(equity1)
        drawdowns2 = self._calculate_drawdowns(equity2)
        
        max_dd1 = np.min(drawdowns1)
        max_dd2 = np.min(drawdowns2)
        
        dd_difference = max_dd1 - max_dd2
        
        # Permutation test for drawdown difference
        combined_length = min(len(drawdowns1), len(drawdowns2))
        n_permutations = 1000
        permutation_diffs = []
        
        for _ in range(n_permutations):
            # Randomly assign drawdowns to strategies
            all_drawdowns = np.concatenate([drawdowns1[:combined_length], drawdowns2[:combined_length]])
            np.random.shuffle(all_drawdowns)
            
            perm_dd1 = all_drawdowns[:combined_length]
            perm_dd2 = all_drawdowns[combined_length:2*combined_length]
            
            perm_max_dd1 = np.min(perm_dd1)
            perm_max_dd2 = np.min(perm_dd2)
            
            permutation_diffs.append(perm_max_dd1 - perm_max_dd2)
        
        # P-value: probability of observing a difference this large under null hypothesis
        p_value = np.mean(np.abs(permutation_diffs) >= abs(dd_difference))
        
        # Confidence interval for difference
        alpha = 1 - confidence_threshold
        ci_lower = np.percentile(permutation_diffs, alpha / 2 * 100)
        ci_upper = np.percentile(permutation_diffs, (1 - alpha / 2) * 100)
        
        # Test statistic: standardized difference
        if len(permutation_diffs) > 0:
            std_diff = np.std(permutation_diffs)
            t_statistic = dd_difference / std_diff if std_diff > 0 else 0.0
        else:
            t_statistic = 0.0
        
        return StatisticsResult(
            test_name="drawdown_comparison",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=confidence_threshold,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=dd_difference,
            sample_size=len(equity1) + len(equity2),
        )
    
    def _calculate_drawdowns(self, equity: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown series from equity curve.
        
        Args:
            equity: Equity curve values
            
        Returns:
            Array of drawdown values (negative values)
        """
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        return drawdowns
    
    def _calculate_drawdown_durations(self, drawdowns: np.ndarray) -> List[int]:
        """
        Calculate duration of each drawdown period.
        
        Args:
            drawdowns: Drawdown series
            
        Returns:
            List of drawdown durations
        """
        durations = []
        current_duration = 0
        
        for dd in drawdowns:
            if dd < -0.001:  # In drawdown (>0.1%)
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Add final duration if still in drawdown
        if current_duration > 0:
            durations.append(current_duration)
        
        return durations


class ReturnDistributionTestStatistics(BaseStatisticsTest):
    """
    Statistical tests for return distribution properties.
    
    Tests various characteristics of return distributions including
    normality, skewness, kurtosis, and tail behavior.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        test_type: str = "normality",
        **kwargs,
    ) -> StatisticsResult:
        """
        Test return distribution properties.
        
        Args:
            sample1: Returns of first strategy
            sample2: Returns of second strategy (for comparison)
            test_type: Type of test ('normality', 'skewness', 'kurtosis', 'tail_risk')
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        returns1 = np.array(sample1)
        
        if test_type == "normality":
            return self._test_normality(returns1)
        elif test_type == "skewness":
            return self._test_skewness(returns1)
        elif test_type == "kurtosis":
            return self._test_kurtosis(returns1)
        elif test_type == "tail_risk":
            return self._test_tail_risk(returns1)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def _test_normality(self, returns: np.ndarray) -> StatisticsResult:
        """
        Test if returns follow a normal distribution using Jarque-Bera test.
        
        Args:
            returns: Return series
            
        Returns:
            Statistical test result
        """
        n = len(returns)
        
        if n < 4:
            return StatisticsResult(
                test_name="jarque_bera_normality",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate sample moments
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        if std_ret == 0:
            return StatisticsResult(
                test_name="jarque_bera_normality",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Standardized returns
        standardized = (returns - mean_ret) / std_ret
        
        # Calculate skewness and kurtosis
        skewness = np.mean(standardized ** 3)
        kurtosis = np.mean(standardized ** 4) - 3  # Excess kurtosis
        
        # Jarque-Bera test statistic
        jb_statistic = n * (skewness ** 2 / 6 + kurtosis ** 2 / 24)
        
        # P-value (chi-square distribution with 2 degrees of freedom)
        p_value = self._chi_square_p_value(jb_statistic, 2)
        
        return StatisticsResult(
            test_name="jarque_bera_normality",
            statistic=jb_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(0.0, 0.0),  # Not applicable
            sample_size=n,
            assumptions_met={
                "independence": True,  # Assumed
                "identically_distributed": True,  # Assumed
            },
        )
    
    def _test_skewness(self, returns: np.ndarray) -> StatisticsResult:
        """
        Test if skewness is significantly different from zero.
        
        Args:
            returns: Return series
            
        Returns:
            Statistical test result
        """
        n = len(returns)
        
        if n < 3:
            return StatisticsResult(
                test_name="skewness_test",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate skewness
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        if std_ret == 0:
            skewness = 0.0
            t_statistic = 0.0
        else:
            standardized = (returns - mean_ret) / std_ret
            skewness = np.mean(standardized ** 3)
            
            # Test statistic for skewness
            se_skewness = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
            t_statistic = skewness / se_skewness
        
        # P-value (two-tailed test)
        p_value = 2 * self._normal_cdf(-abs(t_statistic))
        
        # Confidence interval for skewness
        alpha = 1 - self.confidence_level
        z_critical = self._normal_critical_value(alpha / 2)
        
        if std_ret > 0:
            se_skewness = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
            ci_lower = skewness - z_critical * se_skewness
            ci_upper = skewness + z_critical * se_skewness
        else:
            ci_lower = ci_upper = skewness
        
        return StatisticsResult(
            test_name="skewness_test",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=skewness,
            sample_size=n,
        )
    
    def _test_kurtosis(self, returns: np.ndarray) -> StatisticsResult:
        """
        Test if excess kurtosis is significantly different from zero.
        
        Args:
            returns: Return series
            
        Returns:
            Statistical test result
        """
        n = len(returns)
        
        if n < 4:
            return StatisticsResult(
                test_name="kurtosis_test",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate excess kurtosis
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        if std_ret == 0:
            excess_kurtosis = 0.0
            t_statistic = 0.0
        else:
            standardized = (returns - mean_ret) / std_ret
            excess_kurtosis = np.mean(standardized ** 4) - 3
            
            # Test statistic for excess kurtosis
            se_kurtosis = np.sqrt(24 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5)))
            t_statistic = excess_kurtosis / se_kurtosis
        
        # P-value (two-tailed test)
        p_value = 2 * self._normal_cdf(-abs(t_statistic))
        
        # Confidence interval for excess kurtosis
        alpha = 1 - self.confidence_level
        z_critical = self._normal_critical_value(alpha / 2)
        
        if std_ret > 0:
            se_kurtosis = np.sqrt(24 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5)))
            ci_lower = excess_kurtosis - z_critical * se_kurtosis
            ci_upper = excess_kurtosis + z_critical * se_kurtosis
        else:
            ci_lower = ci_upper = excess_kurtosis
        
        return StatisticsResult(
            test_name="kurtosis_test",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=excess_kurtosis,
            sample_size=n,
        )
    
    def _test_tail_risk(self, returns: np.ndarray) -> StatisticsResult:
        """
        Test tail risk characteristics using Value at Risk.
        
        Args:
            returns: Return series
            
        Returns:
            Statistical test result
        """
        n = len(returns)
        
        if n < 10:
            return StatisticsResult(
                test_name="tail_risk_test",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate VaR at 5% level
        var_5 = np.percentile(returns, 5)
        
        # Expected VaR under normal distribution
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        if std_ret > 0:
            # Normal VaR at 5% level
            normal_var_5 = mean_ret - 1.645 * std_ret
            
            # Test if observed VaR is significantly different from normal VaR
            # Using bootstrap to estimate distribution of VaR
            n_bootstrap = 1000
            bootstrap_vars = []
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(returns, size=n, replace=True)
                bootstrap_var = np.percentile(bootstrap_sample, 5)
                bootstrap_vars.append(bootstrap_var)
            
            bootstrap_vars = np.array(bootstrap_vars)
            expected_var = np.mean(bootstrap_vars)
            std_var = np.std(bootstrap_vars)
            
            if std_var > 0:
                t_statistic = (var_5 - expected_var) / std_var
            else:
                t_statistic = 0.0
            
            # P-value
            p_value = 2 * np.mean(np.abs(bootstrap_vars - expected_var) >= abs(var_5 - expected_var))
            
            # Confidence interval
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(bootstrap_vars, alpha / 2 * 100)
            ci_upper = np.percentile(bootstrap_vars, (1 - alpha / 2) * 100)
        else:
            t_statistic = 0.0
            p_value = 1.0
            ci_lower = ci_upper = var_5
        
        return StatisticsResult(
            test_name="tail_risk_test",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=var_5,
            sample_size=n,
        )
    
    def _chi_square_p_value(self, chi_stat: float, df: int) -> float:
        """Approximate p-value for chi-square distribution."""
        if chi_stat <= 0 or df <= 0:
            return 1.0
        
        # Simple approximation for chi-square p-value
        # For production use, implement proper chi-square CDF
        if df == 1:
            return 2 * self._normal_cdf(-np.sqrt(chi_stat))
        elif df == 2:
            return np.exp(-chi_stat / 2)
        else:
            # Normal approximation for large df
            z = (chi_stat - df) / np.sqrt(2 * df)
            return self._normal_cdf(-abs(z))
    
    def _normal_cdf(self, x: float) -> float:
        """Simplified normal CDF."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))
    
    def _normal_critical_value(self, alpha: float) -> float:
        """Simplified normal critical value."""
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