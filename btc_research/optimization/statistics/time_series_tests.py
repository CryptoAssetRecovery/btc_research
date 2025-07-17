"""
Time series statistical tests for trading strategy analysis.

This module provides statistical tests specifically designed for time series data,
including stationarity tests, autocorrelation analysis, and regime detection.
"""

from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np

from btc_research.optimization.base import BaseStatisticsTest
from btc_research.optimization.types import StatisticsResult

__all__ = [
    "StationarityTests",
    "AutocorrelationAnalysis",
    "RegimeChangeDetection",
    "VolatilityClusteringTests",
    "MarketEfficiencyTests",
]


class StationarityTests(BaseStatisticsTest):
    """
    Statistical tests for time series stationarity.
    
    Implements various tests to assess whether a time series is stationary,
    which is crucial for many statistical analyses.
    
    References:
        - Dickey, D. A. & Fuller, W. A. (1979). Distribution of the estimators
        - Kwiatkowski, D. et al. (1992). Testing the null hypothesis of stationarity
        - Phillips, P. C. B. & Perron, P. (1988). Testing for a unit root
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        test_type: str = "adf",
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Run stationarity test on time series data.
        
        Args:
            sample1: Time series data
            sample2: Not used (compatibility with base class)
            test_type: Type of test ('adf', 'kpss', 'phillips_perron')
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        time_series = np.array(sample1)
        
        if test_type == "adf":
            return self._augmented_dickey_fuller_test(time_series, **kwargs)
        elif test_type == "kpss":
            return self._kpss_test(time_series, **kwargs)
        elif test_type == "phillips_perron":
            return self._phillips_perron_test(time_series, **kwargs)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def _augmented_dickey_fuller_test(
        self, 
        time_series: np.ndarray, 
        max_lags: Optional[int] = None,
        include_trend: bool = False
    ) -> StatisticsResult:
        """
        Augmented Dickey-Fuller test for unit root (non-stationarity).
        
        Args:
            time_series: Time series data
            max_lags: Maximum number of lags to include
            include_trend: Whether to include deterministic trend
            
        Returns:
            Statistical test result
        """
        n = len(time_series)
        
        if n < 3:
            return StatisticsResult(
                test_name="augmented_dickey_fuller",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate first differences
        y = time_series
        dy = np.diff(y)
        y_lag = y[:-1]
        
        # Select number of lags
        if max_lags is None:
            max_lags = min(int(12 * (n / 100) ** (1/4)), n // 3)
        
        max_lags = min(max_lags, len(dy) - 2)
        
        # Construct regression matrices
        if max_lags > 0:
            # Add lagged differences
            dy_lags = np.array([dy[max_lags-i-1:-i-1] for i in range(max_lags)]).T
            X = np.column_stack([y_lag[max_lags:], dy_lags])
            Y = dy[max_lags:]
        else:
            X = y_lag[:-1].reshape(-1, 1)
            Y = dy[1:]
        
        # Add constant and trend if specified
        n_reg = len(Y)
        if include_trend:
            trend = np.arange(1, n_reg + 1)
            X = np.column_stack([X, np.ones(n_reg), trend])
        else:
            X = np.column_stack([X, np.ones(n_reg)])
        
        # OLS regression
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ Y)
            residuals = Y - X @ beta
            mse = np.sum(residuals ** 2) / (n_reg - X.shape[1])
            
            # Standard error of coefficient on y_lag
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_beta0 = np.sqrt(var_beta[0, 0])
            
            # ADF test statistic
            adf_stat = beta[0] / se_beta0
            
        except np.linalg.LinAlgError:
            return StatisticsResult(
                test_name="augmented_dickey_fuller",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Critical values (approximate)
        critical_values = self._adf_critical_values(n, include_trend)
        
        # P-value (approximate)
        p_value = self._adf_p_value(adf_stat, n, include_trend)
        
        return StatisticsResult(
            test_name="augmented_dickey_fuller",
            statistic=adf_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(critical_values["1%"], critical_values["10%"]),
            critical_value=critical_values["5%"],
            sample_size=n,
            assumptions_met={
                "no_structural_breaks": True,  # Assumed
                "homoscedasticity": True,      # Assumed
            },
        )
    
    def _kpss_test(
        self, 
        time_series: np.ndarray, 
        include_trend: bool = False
    ) -> StatisticsResult:
        """
        KPSS test for stationarity (null hypothesis: stationary).
        
        Args:
            time_series: Time series data
            include_trend: Whether to include deterministic trend
            
        Returns:
            Statistical test result
        """
        n = len(time_series)
        
        if n < 3:
            return StatisticsResult(
                test_name="kpss",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        y = time_series
        
        # Detrend the series
        if include_trend:
            # Linear detrending
            t = np.arange(n)
            X = np.column_stack([np.ones(n), t])
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            residuals = y - X @ beta
        else:
            # Demean only
            residuals = y - np.mean(y)
        
        # Calculate partial sums
        S = np.cumsum(residuals)
        
        # Calculate LM statistic
        sigma2 = self._long_run_variance(residuals)
        
        if sigma2 <= 0:
            return StatisticsResult(
                test_name="kpss",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        kpss_stat = np.sum(S ** 2) / (n ** 2 * sigma2)
        
        # Critical values
        critical_values = self._kpss_critical_values(include_trend)
        
        # P-value (approximate)
        p_value = self._kpss_p_value(kpss_stat, include_trend)
        
        return StatisticsResult(
            test_name="kpss",
            statistic=kpss_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(critical_values["1%"], critical_values["10%"]),
            critical_value=critical_values["5%"],
            sample_size=n,
        )
    
    def _phillips_perron_test(
        self, 
        time_series: np.ndarray,
        include_trend: bool = False
    ) -> StatisticsResult:
        """
        Phillips-Perron test for unit root.
        
        Args:
            time_series: Time series data
            include_trend: Whether to include deterministic trend
            
        Returns:
            Statistical test result
        """
        n = len(time_series)
        
        if n < 3:
            return StatisticsResult(
                test_name="phillips_perron",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        y = time_series
        dy = np.diff(y)
        y_lag = y[:-1]
        
        # Construct regression
        if include_trend:
            t = np.arange(1, n)
            X = np.column_stack([y_lag, np.ones(n - 1), t])
        else:
            X = np.column_stack([y_lag, np.ones(n - 1)])
        
        # OLS regression
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ dy)
            residuals = dy - X @ beta
            
            # Long-run variance
            sigma2 = self._long_run_variance(residuals)
            
            # Short-run variance
            gamma0 = np.var(residuals, ddof=1)
            
            if gamma0 <= 0 or sigma2 <= 0:
                pp_stat = 0.0
            else:
                # Phillips-Perron adjustment
                var_beta = np.linalg.inv(X.T @ X)
                se_beta0 = np.sqrt(gamma0 * var_beta[0, 0])
                
                # PP test statistic
                t_stat = beta[0] / se_beta0
                adjustment = 0.5 * (sigma2 - gamma0) / (se_beta0 * gamma0) * var_beta[0, 0]
                pp_stat = t_stat - adjustment
            
        except np.linalg.LinAlgError:
            pp_stat = 0.0
        
        # Use ADF critical values (asymptotically equivalent)
        critical_values = self._adf_critical_values(n, include_trend)
        p_value = self._adf_p_value(pp_stat, n, include_trend)
        
        return StatisticsResult(
            test_name="phillips_perron",
            statistic=pp_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(critical_values["1%"], critical_values["10%"]),
            critical_value=critical_values["5%"],
            sample_size=n,
        )
    
    def _long_run_variance(
        self, 
        residuals: np.ndarray, 
        max_lags: Optional[int] = None
    ) -> float:
        """
        Calculate long-run variance using Newey-West estimator.
        
        Args:
            residuals: Residual series
            max_lags: Maximum number of lags
            
        Returns:
            Long-run variance estimate
        """
        n = len(residuals)
        
        if max_lags is None:
            max_lags = min(int(4 * (n / 100) ** (2/9)), n - 1)
        
        # Calculate autocovariances
        gamma0 = np.var(residuals, ddof=1)
        sigma2 = gamma0
        
        for lag in range(1, max_lags + 1):
            if lag >= n:
                break
            
            # Bartlett kernel weight
            weight = 1 - lag / (max_lags + 1)
            
            # Autocovariance
            gamma_lag = np.mean(residuals[:-lag] * residuals[lag:])
            sigma2 += 2 * weight * gamma_lag
        
        return max(sigma2, 1e-10)
    
    def _adf_critical_values(self, n: int, include_trend: bool) -> Dict[str, float]:
        """Get approximate ADF critical values."""
        # Simplified critical values (constant and trend)
        if include_trend:
            return {
                "1%": -4.0,
                "5%": -3.4,
                "10%": -3.1,
            }
        else:
            return {
                "1%": -3.5,
                "5%": -2.9,
                "10%": -2.6,
            }
    
    def _adf_p_value(self, stat: float, n: int, include_trend: bool) -> float:
        """Approximate p-value for ADF test."""
        # Very simplified p-value calculation
        critical_5 = -3.4 if include_trend else -2.9
        
        if stat < critical_5:
            return 0.01  # Reject null
        else:
            return 0.1   # Fail to reject
    
    def _kpss_critical_values(self, include_trend: bool) -> Dict[str, float]:
        """Get KPSS critical values."""
        if include_trend:
            return {
                "1%": 0.216,
                "5%": 0.146,
                "10%": 0.119,
            }
        else:
            return {
                "1%": 0.739,
                "5%": 0.463,
                "10%": 0.347,
            }
    
    def _kpss_p_value(self, stat: float, include_trend: bool) -> float:
        """Approximate p-value for KPSS test."""
        critical_5 = 0.146 if include_trend else 0.463
        
        if stat > critical_5:
            return 0.01  # Reject null (non-stationary)
        else:
            return 0.1   # Fail to reject (stationary)


class AutocorrelationAnalysis(BaseStatisticsTest):
    """
    Analysis of autocorrelation in strategy returns.
    
    Provides tests for serial correlation, which can indicate
    predictability in returns or model inadequacy.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        max_lags: int = 20,
        test_type: str = "ljung_box",
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Run autocorrelation test.
        
        Args:
            sample1: Time series data (returns)
            sample2: Not used
            max_lags: Maximum number of lags to test
            test_type: Type of test ('ljung_box', 'durbin_watson')
            **kwargs: Additional parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        returns = np.array(sample1)
        
        if test_type == "ljung_box":
            return self._ljung_box_test(returns, max_lags)
        elif test_type == "durbin_watson":
            return self._durbin_watson_test(returns)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def _ljung_box_test(self, returns: np.ndarray, max_lags: int) -> StatisticsResult:
        """
        Ljung-Box test for autocorrelation in returns.
        
        Args:
            returns: Return series
            max_lags: Maximum number of lags
            
        Returns:
            Statistical test result
        """
        n = len(returns)
        
        if n < max_lags + 2:
            return StatisticsResult(
                test_name="ljung_box",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate autocorrelations
        autocorrelations = self._calculate_autocorrelations(returns, max_lags)
        
        # Ljung-Box test statistic
        lb_stat = 0.0
        for k in range(1, max_lags + 1):
            if k < len(autocorrelations):
                rho_k = autocorrelations[k]
                lb_stat += (rho_k ** 2) / (n - k)
        
        lb_stat *= n * (n + 2)
        
        # P-value (chi-square distribution with max_lags degrees of freedom)
        p_value = self._chi_square_p_value(lb_stat, max_lags)
        
        return StatisticsResult(
            test_name="ljung_box",
            statistic=lb_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(0.0, 0.0),  # Not applicable
            sample_size=n,
            assumptions_met={
                "independence": False,  # Testing for dependence
                "stationarity": True,   # Assumed
            },
        )
    
    def _durbin_watson_test(self, returns: np.ndarray) -> StatisticsResult:
        """
        Durbin-Watson test for first-order autocorrelation.
        
        Args:
            returns: Return series
            
        Returns:
            Statistical test result
        """
        n = len(returns)
        
        if n < 3:
            return StatisticsResult(
                test_name="durbin_watson",
                statistic=2.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate differences
        diff_returns = np.diff(returns)
        sum_squared_diff = np.sum(diff_returns ** 2)
        sum_squared_returns = np.sum(returns ** 2)
        
        if sum_squared_returns == 0:
            dw_stat = 2.0
        else:
            dw_stat = sum_squared_diff / sum_squared_returns
        
        # Approximate p-value
        # DW ≈ 2 indicates no autocorrelation
        # DW < 2 indicates positive autocorrelation
        # DW > 2 indicates negative autocorrelation
        
        # Convert to correlation
        rho_approx = 1 - dw_stat / 2
        
        # Test if significantly different from 0
        se_rho = 1 / np.sqrt(n - 1)
        t_stat = rho_approx / se_rho
        
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        
        return StatisticsResult(
            test_name="durbin_watson",
            statistic=dw_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(1.5, 2.5),  # Rough bounds for no autocorrelation
            effect_size=rho_approx,
            sample_size=n,
        )
    
    def _calculate_autocorrelations(
        self, 
        series: np.ndarray, 
        max_lags: int
    ) -> np.ndarray:
        """
        Calculate sample autocorrelations.
        
        Args:
            series: Time series
            max_lags: Maximum number of lags
            
        Returns:
            Array of autocorrelations
        """
        n = len(series)
        autocorr = np.zeros(max_lags + 1)
        
        # Center the series
        centered = series - np.mean(series)
        
        # Calculate autocorrelations
        for lag in range(max_lags + 1):
            if lag >= n:
                autocorr[lag] = 0.0
            else:
                if lag == 0:
                    autocorr[lag] = 1.0
                else:
                    numerator = np.sum(centered[:-lag] * centered[lag:])
                    denominator = np.sum(centered ** 2)
                    
                    if denominator > 0:
                        autocorr[lag] = numerator / denominator
                    else:
                        autocorr[lag] = 0.0
        
        return autocorr
    
    def _chi_square_p_value(self, chi_stat: float, df: int) -> float:
        """Approximate p-value for chi-square distribution."""
        if chi_stat <= 0 or df <= 0:
            return 1.0
        
        # Simple approximation
        if df == 1:
            return 2 * (1 - self._normal_cdf(np.sqrt(chi_stat)))
        elif df == 2:
            return np.exp(-chi_stat / 2)
        else:
            # Normal approximation for large df
            z = (chi_stat - df) / np.sqrt(2 * df)
            return 1 - self._normal_cdf(z)
    
    def _normal_cdf(self, x: float) -> float:
        """Simplified normal CDF."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))


class RegimeChangeDetection(BaseStatisticsTest):
    """
    Detection of structural breaks and regime changes in time series.
    
    Implements tests to detect changes in the underlying data generating
    process, which is important for strategy robustness.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        method: str = "chow_test",
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Run regime change detection test.
        
        Args:
            sample1: Time series data
            sample2: Not used
            method: Detection method ('chow_test', 'cusum', 'bai_perron')
            **kwargs: Additional parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        time_series = np.array(sample1)
        
        if method == "chow_test":
            break_point = kwargs.get("break_point", len(time_series) // 2)
            return self._chow_test(time_series, break_point)
        elif method == "cusum":
            return self._cusum_test(time_series)
        elif method == "bai_perron":
            return self._bai_perron_test(time_series, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _chow_test(self, time_series: np.ndarray, break_point: int) -> StatisticsResult:
        """
        Chow test for structural break at known break point.
        
        Args:
            time_series: Time series data
            break_point: Index of suspected break point
            
        Returns:
            Statistical test result
        """
        n = len(time_series)
        
        if break_point <= 1 or break_point >= n - 1:
            return StatisticsResult(
                test_name="chow_test",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Split series
        y1 = time_series[:break_point]
        y2 = time_series[break_point:]
        
        # Fit models
        # Simple model: y_t = α + β*t + ε_t
        
        # Full sample regression
        t_full = np.arange(n)
        X_full = np.column_stack([np.ones(n), t_full])
        
        try:
            beta_full = np.linalg.solve(X_full.T @ X_full, X_full.T @ time_series)
            residuals_full = time_series - X_full @ beta_full
            ssr_full = np.sum(residuals_full ** 2)
            
            # Sub-sample regressions
            t1 = np.arange(break_point)
            X1 = np.column_stack([np.ones(break_point), t1])
            beta1 = np.linalg.solve(X1.T @ X1, X1.T @ y1)
            residuals1 = y1 - X1 @ beta1
            ssr1 = np.sum(residuals1 ** 2)
            
            t2 = np.arange(len(y2))
            X2 = np.column_stack([np.ones(len(y2)), t2])
            beta2 = np.linalg.solve(X2.T @ X2, X2.T @ y2)
            residuals2 = y2 - X2 @ beta2
            ssr2 = np.sum(residuals2 ** 2)
            
            # Chow test statistic
            k = 2  # Number of parameters
            numerator = (ssr_full - ssr1 - ssr2) / k
            denominator = (ssr1 + ssr2) / (n - 2 * k)
            
            if denominator > 0:
                chow_stat = numerator / denominator
            else:
                chow_stat = 0.0
            
            # P-value (F-distribution)
            p_value = self._f_distribution_p_value(chow_stat, k, n - 2 * k)
            
        except np.linalg.LinAlgError:
            chow_stat = 0.0
            p_value = 1.0
        
        return StatisticsResult(
            test_name="chow_test",
            statistic=chow_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(0.0, 0.0),
            sample_size=n,
        )
    
    def _cusum_test(self, time_series: np.ndarray) -> StatisticsResult:
        """
        CUSUM test for structural stability.
        
        Args:
            time_series: Time series data
            
        Returns:
            Statistical test result
        """
        n = len(time_series)
        
        if n < 10:
            return StatisticsResult(
                test_name="cusum",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Fit model to full sample
        t = np.arange(n)
        X = np.column_stack([np.ones(n), t])
        
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ time_series)
            residuals = time_series - X @ beta
            sigma = np.std(residuals, ddof=2)
            
            if sigma == 0:
                return StatisticsResult(
                    test_name="cusum",
                    statistic=0.0,
                    p_value=1.0,
                    confidence_level=self.confidence_level,
                    confidence_interval=(0.0, 0.0),
                    sample_size=n,
                )
            
            # Calculate CUSUM statistics
            cusum = np.cumsum(residuals) / (sigma * np.sqrt(n))
            
            # Test statistic (maximum absolute CUSUM)
            cusum_stat = np.max(np.abs(cusum))
            
            # Critical value (approximate)
            critical_value = 0.948  # 5% critical value
            
            # P-value (approximate)
            p_value = 0.05 if cusum_stat > critical_value else 0.1
            
        except np.linalg.LinAlgError:
            cusum_stat = 0.0
            p_value = 1.0
        
        return StatisticsResult(
            test_name="cusum",
            statistic=cusum_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(0.0, critical_value),
            sample_size=n,
        )
    
    def _bai_perron_test(
        self, 
        time_series: np.ndarray, 
        max_breaks: int = 5
    ) -> StatisticsResult:
        """
        Bai-Perron test for multiple structural breaks.
        
        Args:
            time_series: Time series data
            max_breaks: Maximum number of breaks to consider
            
        Returns:
            Statistical test result
        """
        n = len(time_series)
        
        if n < 20:
            return StatisticsResult(
                test_name="bai_perron",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Simplified version: test for one break at optimal location
        min_segment = max(5, n // 10)  # Minimum segment size
        
        best_break = None
        best_stat = 0.0
        
        for break_point in range(min_segment, n - min_segment):
            # Calculate LR statistic for break at this point
            lr_stat = self._likelihood_ratio_break(time_series, break_point)
            
            if lr_stat > best_stat:
                best_stat = lr_stat
                best_break = break_point
        
        # Approximate p-value
        # This is highly simplified - actual Bai-Perron test is much more complex
        if best_stat > 10:  # Arbitrary threshold
            p_value = 0.01
        elif best_stat > 5:
            p_value = 0.05
        else:
            p_value = 0.1
        
        return StatisticsResult(
            test_name="bai_perron",
            statistic=best_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(0.0, 0.0),
            effect_size=best_break / n if best_break else 0.0,
            sample_size=n,
        )
    
    def _likelihood_ratio_break(self, time_series: np.ndarray, break_point: int) -> float:
        """
        Calculate likelihood ratio statistic for structural break.
        
        Args:
            time_series: Time series data
            break_point: Proposed break point
            
        Returns:
            Likelihood ratio statistic
        """
        n = len(time_series)
        
        # Full sample variance
        var_full = np.var(time_series, ddof=1)
        
        # Sub-sample variances
        y1 = time_series[:break_point]
        y2 = time_series[break_point:]
        
        var1 = np.var(y1, ddof=1) if len(y1) > 1 else var_full
        var2 = np.var(y2, ddof=1) if len(y2) > 1 else var_full
        
        if var_full <= 0 or var1 <= 0 or var2 <= 0:
            return 0.0
        
        # Log-likelihood ratio
        ll_full = -0.5 * n * np.log(var_full)
        ll_break = -0.5 * len(y1) * np.log(var1) - 0.5 * len(y2) * np.log(var2)
        
        return 2 * (ll_break - ll_full)
    
    def _f_distribution_p_value(self, f_stat: float, df1: int, df2: int) -> float:
        """Approximate p-value for F-distribution."""
        if f_stat <= 1:
            return 0.5
        
        # Crude approximation
        if f_stat > 5:
            return 0.01
        elif f_stat > 3:
            return 0.05
        else:
            return 0.1


class VolatilityClusteringTests(BaseStatisticsTest):
    """
    Tests for volatility clustering (ARCH effects) in returns.
    
    Volatility clustering is a common feature of financial time series
    where periods of high volatility tend to cluster together.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        lags: int = 5,
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Test for volatility clustering in returns.
        
        Args:
            sample1: Return series
            sample2: Not used
            lags: Number of lags for ARCH test
            **kwargs: Additional parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        returns = np.array(sample1)
        
        return self._arch_lm_test(returns, lags)
    
    def _arch_lm_test(self, returns: np.ndarray, lags: int) -> StatisticsResult:
        """
        ARCH-LM test for volatility clustering.
        
        Args:
            returns: Return series
            lags: Number of lags
            
        Returns:
            Statistical test result
        """
        n = len(returns)
        
        if n < lags + 5:
            return StatisticsResult(
                test_name="arch_lm",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate squared residuals (assuming mean = 0)
        squared_returns = returns ** 2
        
        # Regression of squared returns on lagged squared returns
        Y = squared_returns[lags:]
        X = np.ones((len(Y), 1))  # Constant
        
        # Add lagged squared returns
        for lag in range(1, lags + 1):
            X = np.column_stack([X, squared_returns[lags - lag:-lag]])
        
        try:
            # OLS regression
            beta = np.linalg.solve(X.T @ X, X.T @ Y)
            residuals = Y - X @ beta
            
            # Sum of squared residuals
            ssr = np.sum(residuals ** 2)
            
            # Total sum of squares
            tss = np.sum((Y - np.mean(Y)) ** 2)
            
            # R-squared
            if tss > 0:
                r_squared = 1 - ssr / tss
            else:
                r_squared = 0.0
            
            # LM test statistic
            lm_stat = len(Y) * r_squared
            
            # P-value (chi-square with 'lags' degrees of freedom)
            p_value = self._chi_square_p_value(lm_stat, lags)
            
        except np.linalg.LinAlgError:
            lm_stat = 0.0
            p_value = 1.0
        
        return StatisticsResult(
            test_name="arch_lm",
            statistic=lm_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(0.0, 0.0),
            effect_size=r_squared,
            sample_size=n,
            assumptions_met={
                "normality": True,        # Assumed for residuals
                "homoscedasticity": False,  # Testing for heteroscedasticity
            },
        )
    
    def _chi_square_p_value(self, chi_stat: float, df: int) -> float:
        """Approximate p-value for chi-square distribution."""
        if chi_stat <= 0 or df <= 0:
            return 1.0
        
        # Simple approximation
        if df == 1:
            return 2 * (1 - self._normal_cdf(np.sqrt(chi_stat)))
        elif df == 2:
            return np.exp(-chi_stat / 2)
        else:
            # Normal approximation for large df
            z = (chi_stat - df) / np.sqrt(2 * df)
            return 1 - self._normal_cdf(z)
    
    def _normal_cdf(self, x: float) -> float:
        """Simplified normal CDF."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))


class MarketEfficiencyTests(BaseStatisticsTest):
    """
    Tests for market efficiency using return predictability.
    
    Implements various tests to assess whether returns are predictable,
    which would violate the efficient market hypothesis.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        test_type: str = "variance_ratio",
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Run market efficiency test.
        
        Args:
            sample1: Return series
            sample2: Not used
            test_type: Type of test ('variance_ratio', 'runs_test')
            **kwargs: Additional parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        returns = np.array(sample1)
        
        if test_type == "variance_ratio":
            period = kwargs.get("period", 2)
            return self._variance_ratio_test(returns, period)
        elif test_type == "runs_test":
            return self._runs_test(returns)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def _variance_ratio_test(self, returns: np.ndarray, period: int) -> StatisticsResult:
        """
        Variance ratio test for random walk hypothesis.
        
        Args:
            returns: Return series
            period: Aggregation period
            
        Returns:
            Statistical test result
        """
        n = len(returns)
        
        if n < period * 5:
            return StatisticsResult(
                test_name="variance_ratio",
                statistic=1.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Calculate variances
        var_1 = np.var(returns, ddof=1)
        
        # Aggregate returns over 'period' periods
        n_agg = n // period
        agg_returns = []
        
        for i in range(n_agg):
            start_idx = i * period
            end_idx = start_idx + period
            agg_return = np.sum(returns[start_idx:end_idx])
            agg_returns.append(agg_return)
        
        agg_returns = np.array(agg_returns)
        var_k = np.var(agg_returns, ddof=1)
        
        # Variance ratio
        if var_1 > 0:
            vr = var_k / (period * var_1)
        else:
            vr = 1.0
        
        # Test statistic (under random walk, VR should equal 1)
        vr_stat = (vr - 1) * np.sqrt(n_agg)
        
        # P-value (two-tailed test)
        p_value = 2 * (1 - self._normal_cdf(abs(vr_stat)))
        
        return StatisticsResult(
            test_name="variance_ratio",
            statistic=vr_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(0.8, 1.2),  # Rough bounds for random walk
            effect_size=vr,
            sample_size=n,
            assumptions_met={
                "independence": False,  # Testing for dependence
                "homoscedasticity": True,  # Assumed
            },
        )
    
    def _runs_test(self, returns: np.ndarray) -> StatisticsResult:
        """
        Runs test for randomness in return signs.
        
        Args:
            returns: Return series
            
        Returns:
            Statistical test result
        """
        n = len(returns)
        
        if n < 3:
            return StatisticsResult(
                test_name="runs_test",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Convert to signs (+1 for positive, -1 for negative, 0 for zero)
        signs = np.sign(returns)
        
        # Remove zeros
        signs = signs[signs != 0]
        n_nonzero = len(signs)
        
        if n_nonzero < 3:
            return StatisticsResult(
                test_name="runs_test",
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=n,
            )
        
        # Count runs
        runs = 1
        for i in range(1, n_nonzero):
            if signs[i] != signs[i-1]:
                runs += 1
        
        # Count positive and negative observations
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)
        
        # Expected number of runs under randomness
        expected_runs = (2 * n_pos * n_neg) / n_nonzero + 1
        
        # Variance of runs under randomness
        if n_nonzero > 1:
            var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_nonzero)) / (
                n_nonzero ** 2 * (n_nonzero - 1)
            )
        else:
            var_runs = 1.0
        
        # Test statistic
        if var_runs > 0:
            z_stat = (runs - expected_runs) / np.sqrt(var_runs)
        else:
            z_stat = 0.0
        
        # P-value (two-tailed test)
        p_value = 2 * (1 - self._normal_cdf(abs(z_stat)))
        
        return StatisticsResult(
            test_name="runs_test",
            statistic=z_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(expected_runs - 1.96 * np.sqrt(var_runs), 
                               expected_runs + 1.96 * np.sqrt(var_runs)),
            effect_size=runs / expected_runs if expected_runs > 0 else 1.0,
            sample_size=n,
        )
    
    def _normal_cdf(self, x: float) -> float:
        """Simplified normal CDF."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))