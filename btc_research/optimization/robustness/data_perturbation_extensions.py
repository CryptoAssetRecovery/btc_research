"""
Extended data perturbation methods for comprehensive robustness testing.

This module contains additional methods that complement the main
DataPerturbationTest class with advanced temporal and statistical
validation capabilities.
"""

from typing import Any, Callable, Dict, List
import numpy as np
import pandas as pd
from scipy import stats


def run_temporal_perturbation_test(
    self,
    parameters: Dict[str, Any],
    backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
    perturbation_methods: List[str] = None,
    n_simulations: int = 200,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run temporal perturbation test to assess time-dependent effects.
    
    Args:
        parameters: Parameter values to test
        backtest_function: Function that runs backtest and returns metrics
        perturbation_methods: Types of temporal perturbations
        n_simulations: Number of simulations per method
        **kwargs: Additional test parameters
        
    Returns:
        Dictionary with temporal perturbation results
    """
    if perturbation_methods is None:
        perturbation_methods = ['time_shifts', 'frequency_changes', 'seasonal_effects', 'calendar_anomalies']
    
    temporal_results = {}
    
    for method in perturbation_methods:
        method_results = []
        
        for i in range(n_simulations):
            try:
                # Apply temporal perturbation
                perturbed_data = _apply_temporal_perturbation(self.data, method)
                
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
        
        # Calculate statistics
        method_stats = self._calculate_level_statistics(method_results)
        
        temporal_results[method] = {
            'results': method_results,
            'statistics': method_stats,
            'temporal_robustness': _calculate_temporal_robustness(method_results),
        }
    
    return temporal_results


def _apply_temporal_perturbation(data: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Apply temporal perturbations to assess time-dependent robustness.
    
    Args:
        data: Original data
        method: Type of temporal perturbation
        
    Returns:
        Data with temporal perturbation applied
    """
    perturbed_data = data.copy()
    
    if method == 'time_shifts':
        # Random time shifts to break temporal dependencies
        max_shift = len(data) // 20  # Maximum 5% shift
        shift_amount = np.random.randint(-max_shift, max_shift + 1)
        
        if shift_amount > 0:
            # Shift forward (remove from beginning)
            perturbed_data = perturbed_data.iloc[shift_amount:].copy()
        elif shift_amount < 0:
            # Shift backward (remove from end)
            perturbed_data = perturbed_data.iloc[:shift_amount].copy()
        
        # Reset index
        perturbed_data = perturbed_data.reset_index(drop=True)
    
    elif method == 'frequency_changes':
        # Simulate changes in data frequency (sub-sampling or interpolation)
        frequency_factors = [0.5, 2.0]  # Half or double frequency
        factor = np.random.choice(frequency_factors)
        
        if factor < 1.0:
            # Sub-sample (reduce frequency)
            step = int(1 / factor)
            perturbed_data = perturbed_data.iloc[::step].copy()
        else:
            # Interpolate (increase frequency)
            # Simple linear interpolation
            new_length = int(len(data) * factor)
            perturbed_data = perturbed_data.reindex(
                np.linspace(0, len(data) - 1, new_length)
            ).interpolate(method='linear')
        
        perturbed_data = perturbed_data.reset_index(drop=True)
    
    elif method == 'seasonal_effects':
        # Add artificial seasonal patterns
        price_volatility = data['close'].pct_change().std() if 'close' in data.columns else 0.02
        seasonal_amplitude = price_volatility * 0.5
        
        # Create seasonal pattern (daily, weekly, monthly cycles)
        t = np.arange(len(data))
        
        daily_cycle = seasonal_amplitude * np.sin(2 * np.pi * t / 24)  # 24-period cycle
        weekly_cycle = seasonal_amplitude * np.sin(2 * np.pi * t / (24 * 7))  # Weekly cycle
        
        seasonal_effect = daily_cycle + weekly_cycle
        
        for col in ['open', 'high', 'low', 'close']:
            if col in perturbed_data.columns:
                perturbed_data[col] *= (1 + seasonal_effect)
    
    elif method == 'calendar_anomalies':
        # Simulate calendar anomalies (end-of-month, day-of-week effects)
        price_volatility = data['close'].pct_change().std() if 'close' in data.columns else 0.02
        anomaly_strength = price_volatility * 0.3
        
        # Random calendar effects
        calendar_effects = np.zeros(len(data))
        
        # End-of-period effects (every 20-30 periods)
        period_length = np.random.randint(20, 31)
        for i in range(0, len(data), period_length):
            if i < len(data):
                calendar_effects[i] = np.random.normal(0, anomaly_strength)
        
        # Day-of-week effects (every 5 periods)
        for i in range(0, len(data), 5):
            day_effect = np.random.normal(0, anomaly_strength * 0.5)
            for j in range(min(5, len(data) - i)):
                if i + j < len(data):
                    calendar_effects[i + j] += day_effect
        
        for col in ['open', 'high', 'low', 'close']:
            if col in perturbed_data.columns:
                perturbed_data[col] *= (1 + calendar_effects)
    
    return perturbed_data


def _calculate_temporal_robustness(results: List[Dict[str, Any]]) -> float:
    """
    Calculate robustness score for temporal perturbation effects.
    
    Args:
        results: List of simulation results
        
    Returns:
        Temporal robustness score (0-1, higher is better)
    """
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return 0.0
    
    returns = [r.get('total_return', 0) for r in valid_results]
    
    # Temporal robustness components
    consistency = 1.0 / (1.0 + np.std(returns)) if returns else 0
    mean_performance = max(0, np.mean(returns)) if returns else 0
    adaptation_rate = sum(1 for r in returns if r > -0.1) / len(returns) if returns else 0  # Tolerance for small losses
    
    # Weighted temporal robustness score
    temporal_score = (
        0.4 * consistency +
        0.3 * mean_performance +
        0.3 * adaptation_rate
    )
    
    return min(1.0, temporal_score)


def _perform_statistical_validation(
    suite_results: Dict[str, Any],
    parameters: Dict[str, Any],
    backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
    data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Perform statistical validation of perturbation effects.
    
    Args:
        suite_results: Results from perturbation suite
        parameters: Parameter values used
        backtest_function: Backtest function
        data: Original data
        
    Returns:
        Dictionary with statistical validation results
    """
    validation_results = {}
    
    # Get baseline result
    baseline_result = backtest_function(data, parameters)
    baseline_return = baseline_result.get('total_return', 0)
    
    # Collect all perturbation results
    all_perturbed_returns = []
    perturbation_types = []
    
    for test_name, test_results in suite_results.items():
        if isinstance(test_results, dict) and 'results' in test_results:
            for result in test_results['results']:
                if 'error' not in result:
                    all_perturbed_returns.append(result.get('total_return', 0))
                    perturbation_types.append(test_name)
    
    if len(all_perturbed_returns) < 10:
        validation_results['error'] = 'Insufficient data for statistical validation'
        return validation_results
    
    # Statistical tests
    validation_results['baseline_vs_perturbed'] = _test_baseline_vs_perturbed(
        baseline_return, all_perturbed_returns
    )
    
    validation_results['distribution_analysis'] = _analyze_perturbation_distribution(
        all_perturbed_returns
    )
    
    validation_results['effect_size_analysis'] = _calculate_perturbation_effect_sizes(
        suite_results, baseline_return
    )
    
    validation_results['robustness_classification'] = _classify_robustness_level(
        baseline_return, all_perturbed_returns
    )
    
    return validation_results


def _test_baseline_vs_perturbed(
    baseline_return: float, 
    perturbed_returns: List[float]
) -> Dict[str, Any]:
    """
    Test statistical significance of difference between baseline and perturbed results.
    """
    try:
        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(perturbed_returns, baseline_return)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(perturbed_returns) - baseline_return
        pooled_std = np.std(perturbed_returns)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                [r - baseline_return for r in perturbed_returns]
            )
        except:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        
        return {
            't_statistic': t_stat,
            't_test_p_value': p_value,
            'cohens_d': cohens_d,
            'wilcoxon_statistic': wilcoxon_stat,
            'wilcoxon_p_value': wilcoxon_p,
            'significant_difference': p_value < 0.05,
            'mean_degradation': mean_diff,
            'degradation_percentage': (mean_diff / abs(baseline_return) * 100) if baseline_return != 0 else 0
        }
    except Exception as e:
        return {'error': str(e)}


def _analyze_perturbation_distribution(perturbed_returns: List[float]) -> Dict[str, Any]:
    """
    Analyze the distribution of perturbed returns.
    """
    try:
        # Basic statistics
        mean_return = np.mean(perturbed_returns)
        std_return = np.std(perturbed_returns)
        skewness = stats.skew(perturbed_returns)
        kurtosis = stats.kurtosis(perturbed_returns)
        
        # Normality test
        _, normality_p = stats.jarque_bera(perturbed_returns)
        
        # Percentiles
        percentiles = {
            'p5': np.percentile(perturbed_returns, 5),
            'p25': np.percentile(perturbed_returns, 25),
            'p50': np.percentile(perturbed_returns, 50),
            'p75': np.percentile(perturbed_returns, 75),
            'p95': np.percentile(perturbed_returns, 95)
        }
        
        return {
            'mean': mean_return,
            'std': std_return,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': normality_p > 0.05,
            'normality_p_value': normality_p,
            'percentiles': percentiles,
            'negative_return_rate': sum(1 for r in perturbed_returns if r < 0) / len(perturbed_returns)
        }
    except Exception as e:
        return {'error': str(e)}


def _calculate_perturbation_effect_sizes(
    suite_results: Dict[str, Any], 
    baseline_return: float
) -> Dict[str, float]:
    """
    Calculate effect sizes for different perturbation types.
    """
    effect_sizes = {}
    
    for test_name, test_results in suite_results.items():
        if isinstance(test_results, dict) and 'results' in test_results:
            returns = [
                r.get('total_return', 0) for r in test_results['results'] 
                if 'error' not in r
            ]
            
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                
                # Cohen's d effect size
                if std_return > 0:
                    cohens_d = abs(mean_return - baseline_return) / std_return
                else:
                    cohens_d = 0
                
                effect_sizes[test_name] = cohens_d
    
    return effect_sizes


def _classify_robustness_level(
    baseline_return: float, 
    perturbed_returns: List[float]
) -> Dict[str, Any]:
    """
    Classify the overall robustness level based on perturbation results.
    """
    if not perturbed_returns:
        return {'classification': 'unknown', 'confidence': 0.0}
    
    mean_perturbed = np.mean(perturbed_returns)
    std_perturbed = np.std(perturbed_returns)
    
    # Performance degradation
    if baseline_return != 0:
        degradation = (baseline_return - mean_perturbed) / abs(baseline_return)
    else:
        degradation = baseline_return - mean_perturbed
    
    # Stability score
    stability = 1.0 / (1.0 + std_perturbed)
    
    # Success rate (positive returns)
    success_rate = sum(1 for r in perturbed_returns if r > 0) / len(perturbed_returns)
    
    # Combined robustness score
    robustness_score = (
        0.4 * max(0, 1 - degradation) +
        0.3 * stability +
        0.3 * success_rate
    )
    
    # Classification
    if robustness_score >= 0.8:
        classification = 'highly_robust'
        confidence = robustness_score
    elif robustness_score >= 0.6:
        classification = 'moderately_robust'
        confidence = robustness_score
    elif robustness_score >= 0.4:
        classification = 'somewhat_robust'
        confidence = robustness_score
    else:
        classification = 'fragile'
        confidence = 1.0 - robustness_score
    
    return {
        'classification': classification,
        'confidence': confidence,
        'robustness_score': robustness_score,
        'degradation': degradation,
        'stability': stability,
        'success_rate': success_rate
    }