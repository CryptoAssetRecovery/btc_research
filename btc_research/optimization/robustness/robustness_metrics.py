"""
Comprehensive robustness metrics and scoring system.

This module provides advanced metrics for evaluating trading strategy
robustness including tail risk analysis, stability scoring, and
comprehensive robustness assessment frameworks.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from btc_research.optimization.types import RobustnessResult

__all__ = ["RobustnessMetrics", "RobustnessScoring"]


class RobustnessMetrics:
    """
    Comprehensive robustness metrics calculator.
    
    Provides advanced metrics for assessing strategy robustness
    including tail risk, stability, and degradation measures.
    """
    
    @staticmethod
    def calculate_tail_risk_metrics(
        results: List[Dict[str, Any]],
        metrics: List[str] = None,
        confidence_levels: List[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive tail risk metrics.
        
        Args:
            results: List of simulation results
            metrics: List of metrics to analyze
            confidence_levels: List of confidence levels for VaR/CVaR
            
        Returns:
            Dictionary with tail risk metrics
        """
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        tail_metrics = {}
        
        for metric in metrics:
            values = [r.get(metric, 0) for r in results if 'error' not in r and isinstance(r.get(metric), (int, float))]
            
            if not values:
                continue
            
            metric_tail_metrics = {}
            
            # Value at Risk (VaR)
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                var_value = np.percentile(values, alpha * 100)
                metric_tail_metrics[f'var_{conf_level:.0%}'] = var_value
            
            # Conditional Value at Risk (CVaR / Expected Shortfall)
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                var_threshold = np.percentile(values, alpha * 100)
                tail_values = [v for v in values if v <= var_threshold]
                cvar_value = np.mean(tail_values) if tail_values else var_threshold
                metric_tail_metrics[f'cvar_{conf_level:.0%}'] = cvar_value
            
            # Expected Tail Loss (worst 5% average)
            worst_5_percent = sorted(values)[:max(1, len(values) // 20)]
            metric_tail_metrics['expected_tail_loss'] = np.mean(worst_5_percent)
            
            # Maximum loss
            metric_tail_metrics['maximum_loss'] = np.min(values)
            
            # Tail ratio (95th percentile / 5th percentile)
            p95 = np.percentile(values, 95)
            p5 = np.percentile(values, 5)
            metric_tail_metrics['tail_ratio'] = p95 / p5 if p5 != 0 else float('inf')
            
            # Lower partial moments
            mean_value = np.mean(values)
            for moment in [1, 2, 3]:
                downside_values = [v for v in values if v < mean_value]
                if downside_values:
                    lpm = np.mean([(mean_value - v) ** moment for v in downside_values])
                else:
                    lpm = 0.0
                metric_tail_metrics[f'lower_partial_moment_{moment}'] = lpm
            
            tail_metrics[metric] = metric_tail_metrics
        
        return tail_metrics
    
    @staticmethod
    def calculate_stability_metrics(
        results: List[Dict[str, Any]],
        baseline_results: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate stability and consistency metrics.
        
        Args:
            results: List of simulation results
            baseline_results: Baseline results for comparison
            
        Returns:
            Dictionary with stability metrics
        """
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'stability_score': 0.0, 'consistency_score': 0.0}
        
        stability_metrics = {}
        
        # Success rate
        stability_metrics['success_rate'] = len(valid_results) / len(results)
        
        # Performance consistency
        returns = [r.get('total_return', 0) for r in valid_results]
        
        if returns:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Coefficient of variation
            stability_metrics['coefficient_of_variation'] = abs(std_return / mean_return) if mean_return != 0 else float('inf')
            
            # Stability ratio (mean / std)
            stability_metrics['stability_ratio'] = mean_return / std_return if std_return > 0 else 0
            
            # Consistency score (percentage of results within 1 std of mean)
            within_one_std = sum(1 for r in returns if abs(r - mean_return) <= std_return)
            stability_metrics['consistency_score'] = within_one_std / len(returns)
            
            # Robustness score (percentage of positive results)
            stability_metrics['positive_outcome_rate'] = sum(1 for r in returns if r > 0) / len(returns)
            
            # Performance degradation vs baseline
            if baseline_results:
                baseline_return = baseline_results.get('total_return', 0)
                if baseline_return != 0:
                    degradation = (baseline_return - mean_return) / abs(baseline_return)
                    stability_metrics['performance_degradation'] = max(0, degradation)
                else:
                    stability_metrics['performance_degradation'] = 0.0
            
            # Drawdown consistency
            drawdowns = [r.get('max_drawdown', 0) for r in valid_results if 'max_drawdown' in r]
            if drawdowns:
                stability_metrics['drawdown_consistency'] = 1.0 / (1.0 + np.std(drawdowns))
            
            # Sharpe ratio stability
            sharpe_ratios = [r.get('sharpe_ratio', 0) for r in valid_results if 'sharpe_ratio' in r]
            if sharpe_ratios:
                stability_metrics['sharpe_stability'] = 1.0 / (1.0 + np.std(sharpe_ratios))
        
        # Overall stability score (weighted combination)
        weights = {
            'success_rate': 0.3,
            'consistency_score': 0.25,
            'positive_outcome_rate': 0.2,
            'stability_ratio': 0.15,
            'drawdown_consistency': 0.1
        }
        
        stability_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in stability_metrics:
                value = stability_metrics[metric]
                if not (np.isnan(value) or np.isinf(value)):
                    # Normalize metrics to 0-1 range
                    if metric == 'coefficient_of_variation':
                        normalized_value = 1.0 / (1.0 + value)
                    elif metric == 'stability_ratio':
                        normalized_value = min(1.0, max(0.0, (value + 1) / 2))  # Normalize around 0
                    else:
                        normalized_value = min(1.0, max(0.0, value))
                    
                    stability_score += weight * normalized_value
                    total_weight += weight
        
        if total_weight > 0:
            stability_metrics['overall_stability_score'] = stability_score / total_weight
        else:
            stability_metrics['overall_stability_score'] = 0.0
        
        return stability_metrics
    
    @staticmethod
    def calculate_sensitivity_metrics(
        parameter_results: Dict[str, List[Dict[str, Any]]],
        base_parameters: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate parameter sensitivity metrics.
        
        Args:
            parameter_results: Results grouped by parameter variations
            base_parameters: Base parameter values
            
        Returns:
            Dictionary with sensitivity metrics for each parameter
        """
        sensitivity_metrics = {}
        
        for param_name, param_results in parameter_results.items():
            if param_name not in base_parameters:
                continue
            
            valid_results = [r for r in param_results if 'error' not in r]
            
            if not valid_results:
                sensitivity_metrics[param_name] = {'sensitivity_score': float('inf')}
                continue
            
            # Extract parameter values and performance metrics
            param_values = [r.get('parameter_value', base_parameters[param_name]) for r in valid_results]
            returns = [r.get('total_return', 0) for r in valid_results]
            
            param_metrics = {}
            
            if len(param_values) > 1 and len(returns) > 1:
                # Sensitivity coefficient (correlation between parameter change and performance)
                try:
                    correlation = np.corrcoef(param_values, returns)[0, 1]
                    param_metrics['sensitivity_coefficient'] = abs(correlation) if not np.isnan(correlation) else 0
                except:
                    param_metrics['sensitivity_coefficient'] = 0
                
                # Performance range across parameter values
                param_metrics['performance_range'] = max(returns) - min(returns)
                
                # Relative sensitivity (coefficient of variation of performance)
                mean_performance = np.mean(returns)
                std_performance = np.std(returns)
                param_metrics['relative_sensitivity'] = abs(std_performance / mean_performance) if mean_performance != 0 else float('inf')
                
                # Optimal parameter distance (distance from base parameter to best performance)
                best_idx = np.argmax(returns)
                best_param_value = param_values[best_idx]
                base_value = base_parameters[param_name]
                
                if isinstance(base_value, (int, float)):
                    param_metrics['optimal_distance'] = abs(best_param_value - base_value) / abs(base_value) if base_value != 0 else 0
                else:
                    param_metrics['optimal_distance'] = 0
                
                # Stability around base parameter
                base_tolerance = 0.1  # 10% tolerance around base parameter
                if isinstance(base_value, (int, float)):
                    tolerance_range = abs(base_value) * base_tolerance
                    stable_results = [
                        r for r, p in zip(returns, param_values) 
                        if abs(p - base_value) <= tolerance_range
                    ]
                    
                    if stable_results:
                        param_metrics['local_stability'] = 1.0 / (1.0 + np.std(stable_results))
                    else:
                        param_metrics['local_stability'] = 0.0
                else:
                    param_metrics['local_stability'] = 1.0
            
            # Overall sensitivity score (lower is better)
            sensitivity_components = []
            
            if 'sensitivity_coefficient' in param_metrics:
                sensitivity_components.append(param_metrics['sensitivity_coefficient'])
            
            if 'relative_sensitivity' in param_metrics:
                normalized_rel_sens = min(1.0, param_metrics['relative_sensitivity'])
                sensitivity_components.append(normalized_rel_sens)
            
            if sensitivity_components:
                param_metrics['sensitivity_score'] = np.mean(sensitivity_components)
            else:
                param_metrics['sensitivity_score'] = 0.0
            
            sensitivity_metrics[param_name] = param_metrics
        
        return sensitivity_metrics
    
    @staticmethod
    def calculate_regime_robustness(
        regime_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        Calculate robustness across different market regimes.
        
        Args:
            regime_results: Results grouped by market regime
            
        Returns:
            Dictionary with regime robustness metrics
        """
        regime_metrics = {}
        
        # Calculate performance in each regime
        regime_performances = {}
        regime_stabilities = {}
        
        for regime, results in regime_results.items():
            valid_results = [r for r in results if 'error' not in r]
            
            if valid_results:
                returns = [r.get('total_return', 0) for r in valid_results]
                regime_performances[regime] = np.mean(returns)
                regime_stabilities[regime] = 1.0 / (1.0 + np.std(returns))
            else:
                regime_performances[regime] = float('-inf')
                regime_stabilities[regime] = 0.0
        
        if regime_performances:
            # Overall regime consistency
            performance_values = list(regime_performances.values())
            stability_values = list(regime_stabilities.values())
            
            # Filter out invalid values
            valid_performances = [p for p in performance_values if not (np.isnan(p) or np.isinf(p))]
            valid_stabilities = [s for s in stability_values if not (np.isnan(s) or np.isinf(s))]
            
            if valid_performances:
                regime_metrics['regime_consistency'] = 1.0 / (1.0 + np.std(valid_performances))
                regime_metrics['average_regime_performance'] = np.mean(valid_performances)
                regime_metrics['worst_regime_performance'] = np.min(valid_performances)
                regime_metrics['best_regime_performance'] = np.max(valid_performances)
            
            if valid_stabilities:
                regime_metrics['average_regime_stability'] = np.mean(valid_stabilities)
            
            # Regime adaptation score
            positive_regimes = sum(1 for p in valid_performances if p > 0)
            regime_metrics['regime_adaptation_rate'] = positive_regimes / len(valid_performances) if valid_performances else 0
            
            # Regime robustness score (weighted combination)
            robustness_components = []
            
            if 'regime_consistency' in regime_metrics:
                robustness_components.append(regime_metrics['regime_consistency'])
            
            if 'regime_adaptation_rate' in regime_metrics:
                robustness_components.append(regime_metrics['regime_adaptation_rate'])
            
            if 'average_regime_stability' in regime_metrics:
                robustness_components.append(regime_metrics['average_regime_stability'])
            
            if robustness_components:
                regime_metrics['overall_regime_robustness'] = np.mean(robustness_components)
            else:
                regime_metrics['overall_regime_robustness'] = 0.0
        
        return regime_metrics
    
    @staticmethod
    def calculate_overfitting_metrics(
        original_results: Dict[str, float],
        oos_results: List[Dict[str, Any]],
        synthetic_results: List[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate overfitting detection metrics.
        
        Args:
            original_results: Results on original data
            oos_results: Out-of-sample results
            synthetic_results: Results on synthetic data
            
        Returns:
            Dictionary with overfitting metrics
        """
        overfitting_metrics = {}
        
        # Original vs out-of-sample comparison
        if oos_results:
            valid_oos = [r for r in oos_results if 'error' not in r]
            
            if valid_oos:
                oos_returns = [r.get('total_return', 0) for r in valid_oos]
                original_return = original_results.get('total_return', 0)
                oos_mean = np.mean(oos_returns)
                
                # Performance degradation
                if original_return != 0:
                    degradation = (original_return - oos_mean) / abs(original_return)
                    overfitting_metrics['oos_degradation'] = max(0, degradation)
                else:
                    overfitting_metrics['oos_degradation'] = 0.0
                
                # Statistical significance of degradation
                if len(oos_returns) > 1:
                    t_stat, p_value = stats.ttest_1samp(oos_returns, original_return)
                    overfitting_metrics['degradation_p_value'] = p_value
                    overfitting_metrics['significant_degradation'] = p_value < 0.05 and oos_mean < original_return
                
                # Consistency ratio
                oos_positive_rate = sum(1 for r in oos_returns if r > 0) / len(oos_returns)
                original_positive = original_return > 0
                
                if original_positive:
                    overfitting_metrics['consistency_ratio'] = oos_positive_rate
                else:
                    overfitting_metrics['consistency_ratio'] = 1.0 - oos_positive_rate
        
        # Synthetic data analysis
        if synthetic_results:
            valid_synthetic = [r for r in synthetic_results if 'error' not in r]
            
            if valid_synthetic:
                synthetic_returns = [r.get('total_return', 0) for r in valid_synthetic]
                original_return = original_results.get('total_return', 0)
                
                # Permutation p-value
                if synthetic_returns:
                    p_value = sum(1 for r in synthetic_returns if r >= original_return) / len(synthetic_returns)
                    overfitting_metrics['permutation_p_value'] = p_value
                    overfitting_metrics['overfitting_detected'] = p_value > 0.95  # Strategy performs too well
        
        # Overall overfitting score
        overfitting_signals = []
        
        if 'oos_degradation' in overfitting_metrics:
            overfitting_signals.append(min(1.0, overfitting_metrics['oos_degradation']))
        
        if 'significant_degradation' in overfitting_metrics and overfitting_metrics['significant_degradation']:
            overfitting_signals.append(0.8)
        
        if 'overfitting_detected' in overfitting_metrics and overfitting_metrics['overfitting_detected']:
            overfitting_signals.append(0.9)
        
        if 'permutation_p_value' in overfitting_metrics:
            p_val = overfitting_metrics['permutation_p_value']
            if p_val > 0.9:
                overfitting_signals.append(0.8)
            elif p_val > 0.8:
                overfitting_signals.append(0.6)
        
        if overfitting_signals:
            overfitting_metrics['overfitting_score'] = np.mean(overfitting_signals)
        else:
            overfitting_metrics['overfitting_score'] = 0.5  # Neutral
        
        return overfitting_metrics


class RobustnessScoring:
    """
    Comprehensive robustness scoring system.
    
    Provides unified scoring framework for strategy robustness assessment.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize robustness scoring system.
        
        Args:
            weights: Custom weights for different robustness components
        """
        # Default weights for robustness components
        self.weights = weights or {
            'stability': 0.25,
            'tail_risk': 0.20,
            'sensitivity': 0.15,
            'regime_robustness': 0.15,
            'stress_resilience': 0.15,
            'overfitting': 0.10
        }
        
        # Ensure weights sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def calculate_comprehensive_score(
        self,
        robustness_results: Dict[str, Any],
        normalization_method: str = "minmax"
    ) -> Dict[str, float]:
        """
        Calculate comprehensive robustness score.
        
        Args:
            robustness_results: Dictionary containing all robustness test results
            normalization_method: Method for normalizing component scores
            
        Returns:
            Dictionary with component scores and overall robustness score
        """
        component_scores = {}
        
        # Calculate stability score
        if 'stability_metrics' in robustness_results:
            stability_data = robustness_results['stability_metrics']
            component_scores['stability'] = self._calculate_stability_score(stability_data)
        
        # Calculate tail risk score
        if 'tail_risk_metrics' in robustness_results:
            tail_risk_data = robustness_results['tail_risk_metrics']
            component_scores['tail_risk'] = self._calculate_tail_risk_score(tail_risk_data)
        
        # Calculate sensitivity score
        if 'sensitivity_metrics' in robustness_results:
            sensitivity_data = robustness_results['sensitivity_metrics']
            component_scores['sensitivity'] = self._calculate_sensitivity_score(sensitivity_data)
        
        # Calculate regime robustness score
        if 'regime_metrics' in robustness_results:
            regime_data = robustness_results['regime_metrics']
            component_scores['regime_robustness'] = self._calculate_regime_score(regime_data)
        
        # Calculate stress resilience score
        if 'stress_results' in robustness_results:
            stress_data = robustness_results['stress_results']
            component_scores['stress_resilience'] = self._calculate_stress_score(stress_data)
        
        # Calculate overfitting score
        if 'overfitting_metrics' in robustness_results:
            overfitting_data = robustness_results['overfitting_metrics']
            component_scores['overfitting'] = self._calculate_overfitting_score(overfitting_data)
        
        # Normalize component scores
        if normalization_method == "minmax":
            component_scores = self._minmax_normalize(component_scores)
        elif normalization_method == "zscore":
            component_scores = self._zscore_normalize(component_scores)
        
        # Calculate weighted overall score
        overall_score = 0.0
        total_weight = 0.0
        
        for component, score in component_scores.items():
            if component in self.weights and not (np.isnan(score) or np.isinf(score)):
                overall_score += self.weights[component] * score
                total_weight += self.weights[component]
        
        if total_weight > 0:
            overall_score /= total_weight
        
        # Add overall score to results
        result = component_scores.copy()
        result['overall_robustness_score'] = overall_score
        result['robustness_grade'] = self._assign_robustness_grade(overall_score)
        
        return result
    
    def _calculate_stability_score(self, stability_data: Dict[str, float]) -> float:
        """Calculate normalized stability score."""
        if 'overall_stability_score' in stability_data:
            return stability_data['overall_stability_score']
        
        # Fallback calculation
        components = []
        
        if 'success_rate' in stability_data:
            components.append(stability_data['success_rate'])
        
        if 'consistency_score' in stability_data:
            components.append(stability_data['consistency_score'])
        
        if 'positive_outcome_rate' in stability_data:
            components.append(stability_data['positive_outcome_rate'])
        
        return np.mean(components) if components else 0.0
    
    def _calculate_tail_risk_score(self, tail_risk_data: Dict[str, Any]) -> float:
        """Calculate normalized tail risk score (higher = lower risk)."""
        if not tail_risk_data:
            return 0.0
        
        # Focus on total_return tail risk
        total_return_metrics = tail_risk_data.get('total_return', {})
        
        if not total_return_metrics:
            return 0.0
        
        risk_components = []
        
        # VaR component (less negative is better)
        if 'var_95%' in total_return_metrics:
            var_95 = total_return_metrics['var_95%']
            # Normalize VaR (assume -50% is worst case)
            risk_components.append(max(0, 1 + var_95 / 0.5))
        
        # Expected tail loss component
        if 'expected_tail_loss' in total_return_metrics:
            etl = total_return_metrics['expected_tail_loss']
            risk_components.append(max(0, 1 + etl / 0.5))
        
        # Maximum loss component
        if 'maximum_loss' in total_return_metrics:
            max_loss = total_return_metrics['maximum_loss']
            risk_components.append(max(0, 1 + max_loss / 0.5))
        
        return np.mean(risk_components) if risk_components else 0.0
    
    def _calculate_sensitivity_score(self, sensitivity_data: Dict[str, Dict[str, float]]) -> float:
        """Calculate normalized sensitivity score (higher = less sensitive)."""
        if not sensitivity_data:
            return 1.0
        
        sensitivity_scores = []
        
        for param_name, param_metrics in sensitivity_data.items():
            if 'sensitivity_score' in param_metrics:
                # Invert sensitivity score (lower sensitivity = higher robustness)
                param_robustness = 1.0 / (1.0 + param_metrics['sensitivity_score'])
                sensitivity_scores.append(param_robustness)
        
        return np.mean(sensitivity_scores) if sensitivity_scores else 1.0
    
    def _calculate_regime_score(self, regime_data: Dict[str, float]) -> float:
        """Calculate normalized regime robustness score."""
        if 'overall_regime_robustness' in regime_data:
            return regime_data['overall_regime_robustness']
        
        # Fallback calculation
        components = []
        
        if 'regime_adaptation_rate' in regime_data:
            components.append(regime_data['regime_adaptation_rate'])
        
        if 'regime_consistency' in regime_data:
            components.append(regime_data['regime_consistency'])
        
        if 'average_regime_stability' in regime_data:
            components.append(regime_data['average_regime_stability'])
        
        return np.mean(components) if components else 0.0
    
    def _calculate_stress_score(self, stress_data: Any) -> float:
        """Calculate normalized stress resilience score."""
        if hasattr(stress_data, 'success_rate'):
            # RobustnessResult object
            return stress_data.success_rate
        elif isinstance(stress_data, dict):
            if 'success_rate' in stress_data:
                return stress_data['success_rate']
            elif 'survival_rate' in stress_data:
                return stress_data['survival_rate']
        
        return 0.0
    
    def _calculate_overfitting_score(self, overfitting_data: Dict[str, float]) -> float:
        """Calculate normalized overfitting score (higher = less overfitting)."""
        if 'overfitting_score' in overfitting_data:
            # Invert overfitting score
            return 1.0 - overfitting_data['overfitting_score']
        
        return 0.5  # Neutral score
    
    def _minmax_normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply min-max normalization to scores."""
        # For robustness scores, we assume they are already in reasonable ranges [0, 1]
        # This method ensures they are properly bounded
        normalized = {}
        
        for component, score in scores.items():
            if np.isnan(score) or np.isinf(score):
                normalized[component] = 0.0
            else:
                normalized[component] = max(0.0, min(1.0, score))
        
        return normalized
    
    def _zscore_normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply z-score normalization to scores."""
        values = [v for v in scores.values() if not (np.isnan(v) or np.isinf(v))]
        
        if len(values) < 2:
            return self._minmax_normalize(scores)
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return self._minmax_normalize(scores)
        
        normalized = {}
        
        for component, score in scores.items():
            if np.isnan(score) or np.isinf(score):
                normalized[component] = 0.0
            else:
                z_score = (score - mean_val) / std_val
                # Convert z-score to 0-1 range using sigmoid
                normalized[component] = 1.0 / (1.0 + np.exp(-z_score))
        
        return normalized
    
    def _assign_robustness_grade(self, overall_score: float) -> str:
        """Assign letter grade based on overall robustness score."""
        if overall_score >= 0.9:
            return "A+"
        elif overall_score >= 0.8:
            return "A"
        elif overall_score >= 0.7:
            return "B+"
        elif overall_score >= 0.6:
            return "B"
        elif overall_score >= 0.5:
            return "C+"
        elif overall_score >= 0.4:
            return "C"
        elif overall_score >= 0.3:
            return "D+"
        elif overall_score >= 0.2:
            return "D"
        else:
            return "F"
    
    def generate_robustness_report(
        self,
        component_scores: Dict[str, float],
        detailed_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive robustness report.
        
        Args:
            component_scores: Component and overall robustness scores
            detailed_results: Detailed robustness test results
            
        Returns:
            Formatted robustness report string
        """
        report_lines = []
        
        # Header
        overall_score = component_scores.get('overall_robustness_score', 0.0)
        grade = component_scores.get('robustness_grade', 'F')
        
        report_lines.append("=" * 60)
        report_lines.append("STRATEGY ROBUSTNESS ASSESSMENT REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Overall Robustness Score: {overall_score:.3f} ({grade})")
        report_lines.append("")
        
        # Component breakdown
        report_lines.append("Component Scores:")
        report_lines.append("-" * 30)
        
        component_order = ['stability', 'tail_risk', 'sensitivity', 'regime_robustness', 'stress_resilience', 'overfitting']
        
        for component in component_order:
            if component in component_scores:
                score = component_scores[component]
                weight = self.weights.get(component, 0.0)
                report_lines.append(f"{component.replace('_', ' ').title():<20}: {score:.3f} (weight: {weight:.1%})")
        
        report_lines.append("")
        
        # Recommendations
        report_lines.append("Recommendations:")
        report_lines.append("-" * 20)
        
        recommendations = self._generate_recommendations(component_scores)
        for rec in recommendations:
            report_lines.append(f"• {rec}")
        
        report_lines.append("")
        
        # Detailed analysis if available
        if detailed_results:
            report_lines.append("Detailed Analysis:")
            report_lines.append("-" * 20)
            
            # Add key insights from detailed results
            insights = self._extract_key_insights(detailed_results)
            for insight in insights:
                report_lines.append(f"• {insight}")
        
        return "\n".join(report_lines)
    
    def _generate_recommendations(self, component_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on component scores."""
        recommendations = []
        
        # Check each component and suggest improvements
        if component_scores.get('stability', 1.0) < 0.6:
            recommendations.append("Improve strategy stability - consider more conservative parameters")
        
        if component_scores.get('tail_risk', 1.0) < 0.5:
            recommendations.append("High tail risk detected - implement stronger risk management")
        
        if component_scores.get('sensitivity', 1.0) < 0.6:
            recommendations.append("High parameter sensitivity - consider parameter smoothing or averaging")
        
        if component_scores.get('regime_robustness', 1.0) < 0.5:
            recommendations.append("Poor regime adaptation - consider regime-aware logic or multiple strategies")
        
        if component_scores.get('stress_resilience', 1.0) < 0.4:
            recommendations.append("Low stress resilience - implement crisis detection and position sizing")
        
        if component_scores.get('overfitting', 1.0) < 0.6:
            recommendations.append("Possible overfitting detected - validate on additional out-of-sample data")
        
        # Overall score recommendations
        overall_score = component_scores.get('overall_robustness_score', 0.0)
        
        if overall_score < 0.3:
            recommendations.append("CRITICAL: Strategy shows low robustness - major revisions recommended")
        elif overall_score < 0.5:
            recommendations.append("Strategy robustness below acceptable threshold - significant improvements needed")
        elif overall_score < 0.7:
            recommendations.append("Moderate robustness - consider targeted improvements in weak areas")
        elif overall_score >= 0.8:
            recommendations.append("Strong robustness profile - suitable for live trading with proper risk management")
        
        return recommendations
    
    def _extract_key_insights(self, detailed_results: Dict[str, Any]) -> List[str]:
        """Extract key insights from detailed robustness results."""
        insights = []
        
        # Add specific insights based on available data
        if 'monte_carlo_results' in detailed_results:
            mc_results = detailed_results['monte_carlo_results']
            if hasattr(mc_results, 'success_rate'):
                success_rate = mc_results.success_rate
                insights.append(f"Monte Carlo success rate: {success_rate:.1%}")
        
        if 'stress_test_results' in detailed_results:
            stress_results = detailed_results['stress_test_results']
            insights.append("Strategy tested under extreme market conditions")
        
        if 'synthetic_data_results' in detailed_results:
            insights.append("Overfitting assessment completed using synthetic data")
        
        return insights