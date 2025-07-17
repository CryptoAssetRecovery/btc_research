"""
Bootstrap-enhanced performance tests for robust statistical inference.

This module provides bootstrap methods for performance metrics that don't rely
on distributional assumptions and offer robust confidence intervals.
"""

from typing import List, Optional, Any
import numpy as np

from btc_research.optimization.base import BaseStatisticsTest
from btc_research.optimization.types import StatisticsResult

__all__ = [
    "BootstrapPerformanceTests",
]


class BootstrapPerformanceTests(BaseStatisticsTest):
    """
    Bootstrap-enhanced performance tests for robust statistical inference.
    
    Provides bootstrap methods for performance metrics that don't rely
    on distributional assumptions and offer robust confidence intervals.
    
    References:
        - Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap
        - Ledoit, O. & Wolf, M. (2008). Robust performance hypothesis testing
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        test_type: str = "bootstrap_sharpe",
        n_bootstrap: int = 10000,
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Run bootstrap performance test.
        
        Args:
            sample1: Returns of first strategy
            sample2: Returns of second strategy (for comparison)
            test_type: Type of bootstrap test ('bootstrap_sharpe', 'bootstrap_sortino', 'bootstrap_var')
            n_bootstrap: Number of bootstrap samples
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        returns1 = np.array(sample1)
        
        if test_type == "bootstrap_sharpe":
            return self._bootstrap_sharpe_test(returns1, sample2, n_bootstrap)
        elif test_type == "bootstrap_sortino":
            return self._bootstrap_sortino_test(returns1, sample2, n_bootstrap)
        elif test_type == "bootstrap_calmar":
            return self._bootstrap_calmar_test(returns1, sample2, n_bootstrap)
        elif test_type == "bootstrap_var":
            confidence_level = kwargs.get("var_confidence", 0.05)
            return self._bootstrap_var_test(returns1, sample2, n_bootstrap, confidence_level)
        elif test_type == "bootstrap_max_drawdown":
            return self._bootstrap_max_drawdown_test(returns1, sample2, n_bootstrap)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def _bootstrap_sharpe_test(
        self, 
        returns1: np.ndarray, 
        returns2: Optional[List[float]], 
        n_bootstrap: int
    ) -> StatisticsResult:
        """Bootstrap test for Sharpe ratio with bias correction."""
        if returns2 is None:
            return self._single_bootstrap_sharpe(returns1, n_bootstrap)
        else:
            returns2_array = np.array(returns2)
            return self._paired_bootstrap_sharpe(returns1, returns2_array, n_bootstrap)
    
    def _single_bootstrap_sharpe(self, returns: np.ndarray, n_bootstrap: int) -> StatisticsResult:
        """Single-sample bootstrap test for Sharpe ratio."""
        n = len(returns)
        
        if n <= 1:
            return StatisticsResult(
                test_name="bootstrap_sharpe_single",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate observed Sharpe ratio
        observed_sharpe = self._calculate_sharpe_ratio(returns)
        
        # Bootstrap resampling with bias correction
        bootstrap_sharpes = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_sharpe = self._calculate_sharpe_ratio(bootstrap_sample)
            bootstrap_sharpes.append(bootstrap_sharpe)
        
        bootstrap_sharpes = np.array(bootstrap_sharpes)
        
        # Bias correction
        bias = np.mean(bootstrap_sharpes) - observed_sharpe
        bias_corrected_sharpe = observed_sharpe - bias
        
        # Bootstrap confidence interval (percentile method)
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)
        
        # P-value for H0: Sharpe ratio = 0
        p_value = np.mean(np.abs(bootstrap_sharpes) >= np.abs(observed_sharpe))
        
        # Annualized Sharpe ratio
        annualized_sharpe = observed_sharpe * np.sqrt(252)
        
        return StatisticsResult(
            test_name="bootstrap_sharpe_single",
            statistic=bias_corrected_sharpe,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=annualized_sharpe,
            sample_size=n,
            assumptions_met={
                "exchangeability": True,
                "independence": True,
            },
        )
    
    def _paired_bootstrap_sharpe(
        self, 
        returns1: np.ndarray, 
        returns2: np.ndarray, 
        n_bootstrap: int
    ) -> StatisticsResult:
        """Paired bootstrap test for comparing Sharpe ratios."""
        n1, n2 = len(returns1), len(returns2)
        
        # Calculate observed Sharpe ratios
        sharpe1 = self._calculate_sharpe_ratio(returns1)
        sharpe2 = self._calculate_sharpe_ratio(returns2)
        observed_diff = sharpe1 - sharpe2
        
        # Bootstrap resampling
        bootstrap_diffs = []
        
        if n1 == n2:
            # Paired bootstrap (same time periods)
            for _ in range(n_bootstrap):
                indices = np.random.choice(n1, size=n1, replace=True)
                boot_returns1 = returns1[indices]
                boot_returns2 = returns2[indices]
                
                boot_sharpe1 = self._calculate_sharpe_ratio(boot_returns1)
                boot_sharpe2 = self._calculate_sharpe_ratio(boot_returns2)
                bootstrap_diffs.append(boot_sharpe1 - boot_sharpe2)
        else:
            # Independent bootstrap
            for _ in range(n_bootstrap):
                boot_returns1 = np.random.choice(returns1, size=n1, replace=True)
                boot_returns2 = np.random.choice(returns2, size=n2, replace=True)
                
                boot_sharpe1 = self._calculate_sharpe_ratio(boot_returns1)
                boot_sharpe2 = self._calculate_sharpe_ratio(boot_returns2)
                bootstrap_diffs.append(boot_sharpe1 - boot_sharpe2)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Bias correction
        bias = np.mean(bootstrap_diffs) - observed_diff
        bias_corrected_diff = observed_diff - bias
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)
        
        # P-value
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        return StatisticsResult(
            test_name="bootstrap_sharpe_comparison",
            statistic=bias_corrected_diff,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=observed_diff,
            sample_size=n1 + n2,
        )
    
    def _bootstrap_sortino_test(
        self, 
        returns1: np.ndarray, 
        returns2: Optional[List[float]], 
        n_bootstrap: int
    ) -> StatisticsResult:
        """Bootstrap test for Sortino ratio."""
        if returns2 is None:
            return self._single_bootstrap_sortino(returns1, n_bootstrap)
        else:
            returns2_array = np.array(returns2)
            return self._paired_bootstrap_sortino(returns1, returns2_array, n_bootstrap)
    
    def _single_bootstrap_sortino(self, returns: np.ndarray, n_bootstrap: int) -> StatisticsResult:
        """Single-sample bootstrap test for Sortino ratio."""
        n = len(returns)
        
        if n <= 1:
            return StatisticsResult(
                test_name="bootstrap_sortino_single",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate observed Sortino ratio
        observed_sortino = self._calculate_sortino_ratio(returns)
        
        # Bootstrap resampling
        bootstrap_sortinos = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_sortino = self._calculate_sortino_ratio(bootstrap_sample)
            if np.isfinite(bootstrap_sortino):
                bootstrap_sortinos.append(bootstrap_sortino)
        
        if not bootstrap_sortinos:
            return StatisticsResult(
                test_name="bootstrap_sortino_single",
                statistic=observed_sortino,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(observed_sortino, observed_sortino),
                sample_size=n,
            )
        
        bootstrap_sortinos = np.array(bootstrap_sortinos)
        
        # Bias correction
        bias = np.mean(bootstrap_sortinos) - observed_sortino
        bias_corrected_sortino = observed_sortino - bias
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_sortinos, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_sortinos, (1 - alpha / 2) * 100)
        
        # P-value
        p_value = np.mean(np.abs(bootstrap_sortinos) >= np.abs(observed_sortino))
        
        return StatisticsResult(
            test_name="bootstrap_sortino_single",
            statistic=bias_corrected_sortino,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=observed_sortino * np.sqrt(252),
            sample_size=n,
        )
    
    def _paired_bootstrap_sortino(
        self, 
        returns1: np.ndarray, 
        returns2: np.ndarray, 
        n_bootstrap: int
    ) -> StatisticsResult:
        """Paired bootstrap test for comparing Sortino ratios."""
        n1, n2 = len(returns1), len(returns2)
        
        # Calculate observed Sortino ratios
        sortino1 = self._calculate_sortino_ratio(returns1)
        sortino2 = self._calculate_sortino_ratio(returns2)
        observed_diff = sortino1 - sortino2
        
        # Bootstrap resampling
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            if n1 == n2:
                indices = np.random.choice(n1, size=n1, replace=True)
                boot_returns1 = returns1[indices]
                boot_returns2 = returns2[indices]
            else:
                boot_returns1 = np.random.choice(returns1, size=n1, replace=True)
                boot_returns2 = np.random.choice(returns2, size=n2, replace=True)
            
            boot_sortino1 = self._calculate_sortino_ratio(boot_returns1)
            boot_sortino2 = self._calculate_sortino_ratio(boot_returns2)
            
            if np.isfinite(boot_sortino1) and np.isfinite(boot_sortino2):
                bootstrap_diffs.append(boot_sortino1 - boot_sortino2)
        
        if not bootstrap_diffs:
            return StatisticsResult(
                test_name="bootstrap_sortino_comparison",
                statistic=observed_diff,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(observed_diff, observed_diff),
                sample_size=n1 + n2,
            )
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)
        
        # P-value
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        return StatisticsResult(
            test_name="bootstrap_sortino_comparison",
            statistic=observed_diff,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=observed_diff,
            sample_size=n1 + n2,
        )
    
    def _bootstrap_calmar_test(
        self, 
        returns1: np.ndarray, 
        returns2: Optional[List[float]], 
        n_bootstrap: int
    ) -> StatisticsResult:
        """Bootstrap test for Calmar ratio."""
        if returns2 is None:
            return self._single_bootstrap_calmar(returns1, n_bootstrap)
        else:
            returns2_array = np.array(returns2)
            return self._paired_bootstrap_calmar(returns1, returns2_array, n_bootstrap)
    
    def _single_bootstrap_calmar(self, returns: np.ndarray, n_bootstrap: int) -> StatisticsResult:
        """Single-sample bootstrap test for Calmar ratio."""
        n = len(returns)
        
        if n <= 1:
            return StatisticsResult(
                test_name="bootstrap_calmar_single",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate observed Calmar ratio
        observed_calmar = self._calculate_calmar_ratio(returns)
        
        # Bootstrap resampling
        bootstrap_calmars = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_calmar = self._calculate_calmar_ratio(bootstrap_sample)
            
            if np.isfinite(bootstrap_calmar):
                bootstrap_calmars.append(bootstrap_calmar)
        
        if not bootstrap_calmars:
            return StatisticsResult(
                test_name="bootstrap_calmar_single",
                statistic=observed_calmar,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(observed_calmar, observed_calmar),
                sample_size=n,
            )
        
        bootstrap_calmars = np.array(bootstrap_calmars)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_calmars, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_calmars, (1 - alpha / 2) * 100)
        
        # P-value
        p_value = np.mean(np.abs(bootstrap_calmars) >= np.abs(observed_calmar))
        
        return StatisticsResult(
            test_name="bootstrap_calmar_single",
            statistic=observed_calmar,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=observed_calmar,
            sample_size=n,
        )
    
    def _paired_bootstrap_calmar(
        self, 
        returns1: np.ndarray, 
        returns2: np.ndarray, 
        n_bootstrap: int
    ) -> StatisticsResult:
        """Paired bootstrap test for comparing Calmar ratios."""
        n1, n2 = len(returns1), len(returns2)
        
        # Calculate observed Calmar ratios
        calmar1 = self._calculate_calmar_ratio(returns1)
        calmar2 = self._calculate_calmar_ratio(returns2)
        observed_diff = calmar1 - calmar2
        
        # Bootstrap resampling
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            if n1 == n2:
                indices = np.random.choice(n1, size=n1, replace=True)
                boot_returns1 = returns1[indices]
                boot_returns2 = returns2[indices]
            else:
                boot_returns1 = np.random.choice(returns1, size=n1, replace=True)
                boot_returns2 = np.random.choice(returns2, size=n2, replace=True)
            
            boot_calmar1 = self._calculate_calmar_ratio(boot_returns1)
            boot_calmar2 = self._calculate_calmar_ratio(boot_returns2)
            
            if np.isfinite(boot_calmar1) and np.isfinite(boot_calmar2):
                bootstrap_diffs.append(boot_calmar1 - boot_calmar2)
        
        if not bootstrap_diffs:
            return StatisticsResult(
                test_name="bootstrap_calmar_comparison",
                statistic=observed_diff,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(observed_diff, observed_diff),
                sample_size=n1 + n2,
            )
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)
        
        # P-value
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        return StatisticsResult(
            test_name="bootstrap_calmar_comparison",
            statistic=observed_diff,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=observed_diff,
            sample_size=n1 + n2,
        )
    
    def _bootstrap_var_test(
        self, 
        returns1: np.ndarray, 
        returns2: Optional[List[float]], 
        n_bootstrap: int,
        var_confidence: float
    ) -> StatisticsResult:
        """Bootstrap test for Value at Risk (VaR)."""
        if returns2 is None:
            return self._single_bootstrap_var(returns1, n_bootstrap, var_confidence)
        else:
            returns2_array = np.array(returns2)
            return self._paired_bootstrap_var(returns1, returns2_array, n_bootstrap, var_confidence)
    
    def _single_bootstrap_var(
        self, 
        returns: np.ndarray, 
        n_bootstrap: int, 
        var_confidence: float
    ) -> StatisticsResult:
        """Single-sample bootstrap test for VaR."""
        n = len(returns)
        
        if n <= 1:
            return StatisticsResult(
                test_name="bootstrap_var_single",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate observed VaR
        observed_var = np.percentile(returns, var_confidence * 100)
        
        # Bootstrap resampling
        bootstrap_vars = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_var = np.percentile(bootstrap_sample, var_confidence * 100)
            bootstrap_vars.append(bootstrap_var)
        
        bootstrap_vars = np.array(bootstrap_vars)
        
        # Confidence interval for VaR
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_vars, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_vars, (1 - alpha / 2) * 100)
        
        # P-value
        p_value = np.mean(np.abs(bootstrap_vars) >= np.abs(observed_var))
        
        return StatisticsResult(
            test_name="bootstrap_var_single",
            statistic=observed_var,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=observed_var,
            sample_size=n,
        )
    
    def _paired_bootstrap_var(
        self, 
        returns1: np.ndarray, 
        returns2: np.ndarray, 
        n_bootstrap: int,
        var_confidence: float
    ) -> StatisticsResult:
        """Paired bootstrap test for comparing VaR."""
        n1, n2 = len(returns1), len(returns2)
        
        # Calculate observed VaRs
        var1 = np.percentile(returns1, var_confidence * 100)
        var2 = np.percentile(returns2, var_confidence * 100)
        observed_diff = var1 - var2
        
        # Bootstrap resampling
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            boot_returns1 = np.random.choice(returns1, size=n1, replace=True)
            boot_returns2 = np.random.choice(returns2, size=n2, replace=True)
            
            boot_var1 = np.percentile(boot_returns1, var_confidence * 100)
            boot_var2 = np.percentile(boot_returns2, var_confidence * 100)
            bootstrap_diffs.append(boot_var1 - boot_var2)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)
        
        # P-value
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        return StatisticsResult(
            test_name="bootstrap_var_comparison",
            statistic=observed_diff,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=observed_diff,
            sample_size=n1 + n2,
        )
    
    def _bootstrap_max_drawdown_test(
        self, 
        returns1: np.ndarray, 
        returns2: Optional[List[float]], 
        n_bootstrap: int
    ) -> StatisticsResult:
        """Bootstrap test for maximum drawdown."""
        if returns2 is None:
            return self._single_bootstrap_max_drawdown(returns1, n_bootstrap)
        else:
            returns2_array = np.array(returns2)
            return self._paired_bootstrap_max_drawdown(returns1, returns2_array, n_bootstrap)
    
    def _single_bootstrap_max_drawdown(self, returns: np.ndarray, n_bootstrap: int) -> StatisticsResult:
        """Single-sample bootstrap test for maximum drawdown."""
        n = len(returns)
        
        if n <= 1:
            return StatisticsResult(
                test_name="bootstrap_max_drawdown_single",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate observed maximum drawdown
        observed_max_dd = self._calculate_max_drawdown(returns)
        
        # Bootstrap resampling
        bootstrap_max_dds = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_max_dd = self._calculate_max_drawdown(bootstrap_sample)
            bootstrap_max_dds.append(bootstrap_max_dd)
        
        bootstrap_max_dds = np.array(bootstrap_max_dds)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_max_dds, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_max_dds, (1 - alpha / 2) * 100)
        
        # P-value (probability of observing a drawdown this large or larger)
        p_value = np.mean(bootstrap_max_dds <= observed_max_dd)
        
        return StatisticsResult(
            test_name="bootstrap_max_drawdown_single",
            statistic=observed_max_dd,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=observed_max_dd,
            sample_size=n,
        )
    
    def _paired_bootstrap_max_drawdown(
        self, 
        returns1: np.ndarray, 
        returns2: np.ndarray, 
        n_bootstrap: int
    ) -> StatisticsResult:
        """Paired bootstrap test for comparing maximum drawdowns."""
        n1, n2 = len(returns1), len(returns2)
        
        # Calculate observed maximum drawdowns
        max_dd1 = self._calculate_max_drawdown(returns1)
        max_dd2 = self._calculate_max_drawdown(returns2)
        observed_diff = max_dd1 - max_dd2
        
        # Bootstrap resampling
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            boot_returns1 = np.random.choice(returns1, size=n1, replace=True)
            boot_returns2 = np.random.choice(returns2, size=n2, replace=True)
            
            boot_max_dd1 = self._calculate_max_drawdown(boot_returns1)
            boot_max_dd2 = self._calculate_max_drawdown(boot_returns2)
            bootstrap_diffs.append(boot_max_dd1 - boot_max_dd2)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)
        
        # P-value
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        return StatisticsResult(
            test_name="bootstrap_max_drawdown_comparison",
            statistic=observed_diff,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=observed_diff,
            sample_size=n1 + n2,
        )
    
    # Helper methods
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        std_returns = np.std(excess_returns, ddof=1)
        
        if std_returns == 0:
            return 0.0
        
        return np.mean(excess_returns) / std_returns
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - target_return / 252  # Daily target return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        return np.mean(excess_returns) / downside_std
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        if len(returns) <= 1:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown(returns)
        
        if max_drawdown == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        # Annualized return
        total_return = cumulative_returns[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        return annualized_return / abs(max_drawdown)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) <= 1:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum
        peak = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - peak) / peak
        
        # Return maximum drawdown (most negative value)
        return np.min(drawdown)