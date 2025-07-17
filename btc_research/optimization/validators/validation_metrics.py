"""
Comprehensive validation metrics and stability scoring for optimization frameworks.

This module provides advanced metrics for assessing the robustness and 
generalizability of trading strategies across different validation schemes.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from btc_research.optimization.types import ValidationResult

# Set up logging
logger = logging.getLogger(__name__)

__all__ = [
    "ValidationMetricsCalculator",
    "StabilityAnalyzer", 
    "PerformanceDegradationDetector",
    "OverfittingDetector",
    "ValidationSummaryGenerator",
]


class ValidationMetricsCalculator:
    """
    Calculator for comprehensive validation metrics.
    
    Provides statistical analysis of performance across validation folds
    including consistency, stability, and robustness measures.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize validation metrics calculator.
        
        Args:
            confidence_level: Confidence level for statistical tests and intervals
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_comprehensive_metrics(
        self, 
        validation_result: ValidationResult
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive validation metrics from validation result.
        
        Args:
            validation_result: Result from any validation method
            
        Returns:
            Dictionary with comprehensive metrics
        """
        metrics = {
            "basic_statistics": self._calculate_basic_statistics(validation_result),
            "stability_metrics": self._calculate_stability_metrics(validation_result),
            "consistency_metrics": self._calculate_consistency_metrics(validation_result),
            "robustness_metrics": self._calculate_robustness_metrics(validation_result),
            "statistical_tests": self._perform_statistical_tests(validation_result),
        }
        
        # Overall assessment
        metrics["overall_assessment"] = self._generate_overall_assessment(metrics)
        
        return metrics
    
    def _calculate_basic_statistics(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Calculate basic statistical measures."""
        fold_results = validation_result.fold_results
        if not fold_results:
            return {}
        
        # Extract all numeric metrics
        numeric_metrics = self._extract_numeric_metrics(fold_results)
        
        basic_stats = {}
        for metric_name, values in numeric_metrics.items():
            if len(values) < 2:
                continue
            
            values_array = np.array(values)
            
            basic_stats[metric_name] = {
                "count": len(values),
                "mean": float(np.mean(values_array)),
                "median": float(np.median(values_array)),
                "std": float(np.std(values_array, ddof=1)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "range": float(np.max(values_array) - np.min(values_array)),
                "q25": float(np.percentile(values_array, 25)),
                "q75": float(np.percentile(values_array, 75)),
                "iqr": float(np.percentile(values_array, 75) - np.percentile(values_array, 25)),
                "skewness": float(stats.skew(values_array)),
                "kurtosis": float(stats.kurtosis(values_array)),
            }
        
        return basic_stats
    
    def _calculate_stability_metrics(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Calculate stability and consistency metrics."""
        fold_results = validation_result.fold_results
        if not fold_results:
            return {}
        
        numeric_metrics = self._extract_numeric_metrics(fold_results)
        stability_metrics = {}
        
        for metric_name, values in numeric_metrics.items():
            if len(values) < 2:
                continue
            
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array, ddof=1)
            
            # Coefficient of variation (lower is more stable)
            cv = abs(std_val / mean_val) if mean_val != 0 else float('inf')
            
            # Relative standard deviation
            rsd = (std_val / abs(mean_val)) * 100 if mean_val != 0 else float('inf')
            
            # Stability ratio (values within 1 std of mean)
            within_1std = np.sum(np.abs(values_array - mean_val) <= std_val) / len(values)
            within_2std = np.sum(np.abs(values_array - mean_val) <= 2 * std_val) / len(values)
            
            # Outlier detection using IQR method
            q1, q3 = np.percentile(values_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = np.sum((values_array < lower_bound) | (values_array > upper_bound))
            outlier_ratio = outliers / len(values)
            
            stability_metrics[metric_name] = {
                "coefficient_of_variation": cv,
                "relative_std_deviation": rsd,
                "within_1std_ratio": within_1std,
                "within_2std_ratio": within_2std,
                "outlier_count": int(outliers),
                "outlier_ratio": outlier_ratio,
                "stability_score": 1 / (1 + cv) if not np.isinf(cv) else 0,  # Normalized stability
            }
        
        return stability_metrics
    
    def _calculate_consistency_metrics(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Calculate consistency metrics across folds."""
        fold_results = validation_result.fold_results
        if not fold_results:
            return {}
        
        numeric_metrics = self._extract_numeric_metrics(fold_results)
        consistency_metrics = {}
        
        for metric_name, values in numeric_metrics.items():
            if len(values) < 3:
                continue
            
            values_array = np.array(values)
            
            # Sequential consistency (correlation between adjacent folds)
            adjacent_corr = []
            for i in range(len(values) - 1):
                if len(values) > 2:
                    corr, _ = stats.pearsonr(values[:-1], values[1:])
                    adjacent_corr.append(corr)
            
            avg_adjacent_corr = np.mean(adjacent_corr) if adjacent_corr else 0
            
            # Monotonicity test (trend consistency)
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values_array)
            
            # Sign consistency (how often performance is above/below median)
            median_val = np.median(values_array)
            above_median = np.sum(values_array > median_val)
            below_median = np.sum(values_array < median_val)
            sign_consistency = max(above_median, below_median) / len(values)
            
            consistency_metrics[metric_name] = {
                "adjacent_correlation": avg_adjacent_corr,
                "trend_slope": slope,
                "trend_r_squared": r_value ** 2,
                "trend_p_value": p_value,
                "sign_consistency": sign_consistency,
                "is_trending": abs(slope) > 2 * std_err,  # Significant trend
            }
        
        return consistency_metrics
    
    def _calculate_robustness_metrics(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Calculate robustness and reliability metrics."""
        fold_results = validation_result.fold_results
        if not fold_results:
            return {}
        
        numeric_metrics = self._extract_numeric_metrics(fold_results)
        robustness_metrics = {}
        
        for metric_name, values in numeric_metrics.items():
            if len(values) < 2:
                continue
            
            values_array = np.array(values)
            
            # Worst-case analysis
            worst_case = np.min(values_array)
            best_case = np.max(values_array)
            worst_case_ratio = worst_case / np.mean(values_array) if np.mean(values_array) != 0 else 0
            
            # Downside risk metrics
            mean_val = np.mean(values_array)
            downside_values = values_array[values_array < mean_val]
            downside_std = np.std(downside_values) if len(downside_values) > 1 else 0
            downside_ratio = len(downside_values) / len(values_array)
            
            # Value at Risk (VaR) and Expected Shortfall (ES)
            var_95 = np.percentile(values_array, 5)  # 5th percentile (95% VaR)
            var_99 = np.percentile(values_array, 1)  # 1st percentile (99% VaR)
            
            # Expected Shortfall (mean of worst 5% outcomes)
            worst_5pct_count = max(1, int(0.05 * len(values_array)))
            sorted_values = np.sort(values_array)
            es_95 = np.mean(sorted_values[:worst_5pct_count])
            
            robustness_metrics[metric_name] = {
                "worst_case": worst_case,
                "best_case": best_case,
                "worst_case_ratio": worst_case_ratio,
                "downside_std": downside_std,
                "downside_ratio": downside_ratio,
                "var_95": var_95,
                "var_99": var_99,
                "expected_shortfall_95": es_95,
                "robust_score": worst_case_ratio * (1 - downside_ratio),  # Combined robustness measure
            }
        
        return robustness_metrics
    
    def _perform_statistical_tests(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        fold_results = validation_result.fold_results
        if not fold_results:
            return {}
        
        numeric_metrics = self._extract_numeric_metrics(fold_results)
        statistical_tests = {}
        
        for metric_name, values in numeric_metrics.items():
            if len(values) < 3:
                continue
            
            values_array = np.array(values)
            
            # Normality test (Shapiro-Wilk)
            try:
                shapiro_stat, shapiro_p = stats.shapiro(values_array)
                is_normal = shapiro_p > self.alpha
            except:
                shapiro_stat, shapiro_p, is_normal = 0, 1, False
            
            # Test if mean is significantly different from zero
            try:
                t_stat, t_p = stats.ttest_1samp(values_array, 0)
                significantly_positive = t_p < self.alpha and t_stat > 0
                significantly_negative = t_p < self.alpha and t_stat < 0
            except:
                t_stat, t_p, significantly_positive, significantly_negative = 0, 1, False, False
            
            # Confidence interval for the mean
            sem = stats.sem(values_array)
            confidence_interval = stats.t.interval(
                self.confidence_level, 
                len(values_array) - 1, 
                loc=np.mean(values_array), 
                scale=sem
            )
            
            statistical_tests[metric_name] = {
                "normality_test": {
                    "statistic": shapiro_stat,
                    "p_value": shapiro_p,
                    "is_normal": is_normal,
                },
                "t_test_vs_zero": {
                    "statistic": t_stat,
                    "p_value": t_p,
                    "significantly_positive": significantly_positive,
                    "significantly_negative": significantly_negative,
                },
                "confidence_interval": {
                    "lower": confidence_interval[0],
                    "upper": confidence_interval[1],
                    "width": confidence_interval[1] - confidence_interval[0],
                },
            }
        
        return statistical_tests
    
    def _generate_overall_assessment(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an overall assessment from all metrics."""
        # Key metrics for assessment
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        assessment_scores = []
        issues = []
        recommendations = []
        
        stability_metrics = metrics.get('stability_metrics', {})
        robustness_metrics = metrics.get('robustness_metrics', {})
        consistency_metrics = metrics.get('consistency_metrics', {})
        
        for metric in key_metrics:
            if metric not in stability_metrics:
                continue
            
            # Stability assessment
            cv = stability_metrics[metric].get('coefficient_of_variation', float('inf'))
            if cv > 0.5:
                issues.append(f"High variability in {metric} (CV={cv:.2f})")
                assessment_scores.append(1)  # Poor
            elif cv > 0.25:
                assessment_scores.append(2)  # Fair
            else:
                assessment_scores.append(3)  # Good
            
            # Robustness assessment
            if metric in robustness_metrics:
                worst_case_ratio = robustness_metrics[metric].get('worst_case_ratio', 0)
                if worst_case_ratio < 0.5:
                    issues.append(f"Poor worst-case performance in {metric}")
                    recommendations.append(f"Review parameter robustness for {metric}")
        
        # Overall score (1-3 scale)
        if assessment_scores:
            overall_score = np.mean(assessment_scores)
            if overall_score >= 2.5:
                overall_rating = "Good"
            elif overall_score >= 2.0:
                overall_rating = "Fair"
            else:
                overall_rating = "Poor"
        else:
            overall_score = 0
            overall_rating = "Insufficient Data"
        
        # Generate recommendations
        if not recommendations:
            if overall_score >= 2.5:
                recommendations.append("Parameters show good stability and robustness")
            elif overall_score >= 2.0:
                recommendations.append("Consider minor parameter adjustments for improved stability")
            else:
                recommendations.extend([
                    "Significant parameter optimization needed",
                    "Consider expanding validation period",
                    "Review strategy logic for robustness",
                ])
        
        return {
            "overall_score": overall_score,
            "overall_rating": overall_rating,
            "issues_identified": issues,
            "recommendations": recommendations,
            "metrics_analyzed": len(assessment_scores),
        }
    
    def _extract_numeric_metrics(self, fold_results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract numeric metrics from fold results."""
        numeric_metrics = {}
        
        for result in fold_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and not key.startswith(('fold', 'train_', 'test_', 'val_')):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(float(value))
        
        return numeric_metrics


class StabilityAnalyzer:
    """Analyzer for performance stability across validation periods."""
    
    def __init__(self):
        self.metrics_calculator = ValidationMetricsCalculator()
    
    def analyze_stability(
        self, 
        validation_result: ValidationResult,
        primary_metric: str = 'total_return'
    ) -> Dict[str, Any]:
        """
        Analyze stability of performance across validation periods.
        
        Args:
            validation_result: Validation result to analyze
            primary_metric: Primary metric to focus analysis on
            
        Returns:
            Dictionary with stability analysis
        """
        fold_results = validation_result.fold_results
        if not fold_results:
            return {"error": "No fold results available"}
        
        # Extract primary metric values
        metric_values = []
        for result in fold_results:
            if primary_metric in result:
                metric_values.append(result[primary_metric])
        
        if len(metric_values) < 2:
            return {"error": f"Insufficient data for {primary_metric}"}
        
        values_array = np.array(metric_values)
        
        # Calculate stability metrics
        mean_val = np.mean(values_array)
        std_val = np.std(values_array, ddof=1)
        cv = abs(std_val / mean_val) if mean_val != 0 else float('inf')
        
        # Rolling stability analysis
        window_size = min(3, len(metric_values) // 2)
        rolling_stds = []
        rolling_means = []
        
        for i in range(window_size, len(metric_values) + 1):
            window_values = values_array[i-window_size:i]
            rolling_means.append(np.mean(window_values))
            rolling_stds.append(np.std(window_values, ddof=1))
        
        # Stability trend
        if len(rolling_stds) > 1:
            x = np.arange(len(rolling_stds))
            trend_slope, _, _, _, _ = stats.linregress(x, rolling_stds)
            stability_trend = "improving" if trend_slope < 0 else "deteriorating" if trend_slope > 0 else "stable"
        else:
            trend_slope = 0
            stability_trend = "stable"
        
        # Stability classification
        if cv < 0.1:
            stability_class = "highly_stable"
        elif cv < 0.25:
            stability_class = "stable"
        elif cv < 0.5:
            stability_class = "moderately_unstable"
        else:
            stability_class = "highly_unstable"
        
        return {
            "primary_metric": primary_metric,
            "coefficient_of_variation": cv,
            "stability_class": stability_class,
            "stability_trend": stability_trend,
            "trend_slope": trend_slope,
            "rolling_analysis": {
                "window_size": window_size,
                "rolling_means": rolling_means,
                "rolling_stds": rolling_stds,
                "mean_rolling_std": np.mean(rolling_stds) if rolling_stds else 0,
            },
            "recommendations": self._get_stability_recommendations(stability_class, stability_trend),
        }
    
    def _get_stability_recommendations(self, stability_class: str, stability_trend: str) -> List[str]:
        """Generate recommendations based on stability analysis."""
        recommendations = []
        
        if stability_class == "highly_unstable":
            recommendations.extend([
                "Consider more conservative parameter values",
                "Increase training data or validation periods",
                "Review strategy logic for market regime sensitivity",
            ])
        elif stability_class == "moderately_unstable":
            recommendations.extend([
                "Monitor performance closely in live trading",
                "Consider ensemble methods for improved stability",
            ])
        
        if stability_trend == "deteriorating":
            recommendations.extend([
                "Investigate causes of increasing instability",
                "Consider adaptive parameter mechanisms",
            ])
        elif stability_trend == "improving":
            recommendations.append("Stability is improving - good sign for robustness")
        
        return recommendations


class PerformanceDegradationDetector:
    """Detector for performance degradation patterns."""
    
    def detect_degradation(
        self, 
        validation_result: ValidationResult,
        metrics_to_check: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect performance degradation across validation periods.
        
        Args:
            validation_result: Validation result to analyze
            metrics_to_check: List of metrics to check for degradation
            
        Returns:
            Dictionary with degradation analysis
        """
        if metrics_to_check is None:
            metrics_to_check = ['total_return', 'sharpe_ratio', 'win_rate']
        
        fold_results = validation_result.fold_results
        if not fold_results:
            return {"error": "No fold results available"}
        
        degradation_analysis = {}
        
        for metric in metrics_to_check:
            metric_values = []
            for result in fold_results:
                if metric in result:
                    metric_values.append(result[metric])
            
            if len(metric_values) < 3:
                continue
            
            values_array = np.array(metric_values)
            x = np.arange(len(metric_values))
            
            # Linear trend analysis
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values_array)
            
            # Significant degradation test
            is_declining = slope < 0 and p_value < 0.05
            decline_rate = slope / np.mean(values_array) if np.mean(values_array) != 0 else 0
            
            # Change point detection (simple method)
            changes = []
            for i in range(1, len(metric_values)):
                pct_change = (metric_values[i] - metric_values[i-1]) / abs(metric_values[i-1]) if metric_values[i-1] != 0 else 0
                if pct_change < -0.1:  # 10% drop
                    changes.append({
                        "period": i,
                        "drop_percentage": pct_change * 100,
                        "from_value": metric_values[i-1],
                        "to_value": metric_values[i],
                    })
            
            # Performance phases analysis
            mid_point = len(metric_values) // 2
            early_performance = np.mean(values_array[:mid_point])
            late_performance = np.mean(values_array[mid_point:])
            performance_ratio = late_performance / early_performance if early_performance != 0 else 1
            
            degradation_analysis[metric] = {
                "trend_slope": slope,
                "trend_p_value": p_value,
                "is_significantly_declining": is_declining,
                "decline_rate_per_period": decline_rate,
                "r_squared": r_value ** 2,
                "significant_drops": changes,
                "num_significant_drops": len(changes),
                "early_vs_late_ratio": performance_ratio,
                "degradation_severity": self._classify_degradation(is_declining, decline_rate, len(changes)),
            }
        
        # Overall degradation assessment
        overall_degradation = self._assess_overall_degradation(degradation_analysis)
        
        return {
            "metric_analysis": degradation_analysis,
            "overall_assessment": overall_degradation,
        }
    
    def _classify_degradation(self, is_declining: bool, decline_rate: float, num_drops: int) -> str:
        """Classify the severity of degradation."""
        if not is_declining and num_drops == 0:
            return "none"
        elif is_declining and abs(decline_rate) > 0.05:
            return "severe"
        elif is_declining or num_drops > 2:
            return "moderate"
        else:
            return "mild"
    
    def _assess_overall_degradation(self, degradation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall degradation across all metrics."""
        severity_scores = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
        
        severities = []
        declining_metrics = []
        
        for metric, analysis in degradation_analysis.items():
            severity = analysis.get("degradation_severity", "none")
            severities.append(severity_scores[severity])
            
            if analysis.get("is_significantly_declining", False):
                declining_metrics.append(metric)
        
        if not severities:
            return {"overall_severity": "unknown", "recommendations": []}
        
        avg_severity = np.mean(severities)
        
        if avg_severity >= 2.5:
            overall_severity = "severe"
        elif avg_severity >= 1.5:
            overall_severity = "moderate"
        elif avg_severity >= 0.5:
            overall_severity = "mild"
        else:
            overall_severity = "none"
        
        recommendations = []
        if overall_severity != "none":
            recommendations.extend([
                "Investigate causes of performance degradation",
                "Consider re-optimization with more recent data",
                "Monitor strategy performance more closely",
            ])
            
            if declining_metrics:
                recommendations.append(f"Focus on improving: {', '.join(declining_metrics)}")
        
        return {
            "overall_severity": overall_severity,
            "declining_metrics": declining_metrics,
            "recommendations": recommendations,
            "metrics_analyzed": len(degradation_analysis),
        }


class OverfittingDetector:
    """Detector for overfitting in validation results."""
    
    def detect_overfitting(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """
        Detect signs of overfitting in validation results.
        
        Args:
            validation_result: Validation result to analyze
            
        Returns:
            Dictionary with overfitting analysis
        """
        overfitting_indicators = []
        risk_score = 0
        
        # Check stability score
        stability_score = validation_result.stability_score
        if stability_score > 0.5:
            overfitting_indicators.append("High performance variability across folds")
            risk_score += 3
        elif stability_score > 0.3:
            overfitting_indicators.append("Moderate performance variability")
            risk_score += 1
        
        # Check number of folds
        if validation_result.n_splits < 3:
            overfitting_indicators.append("Insufficient number of validation folds")
            risk_score += 2
        
        # Check confidence intervals
        wide_intervals = []
        for metric, (lower, upper) in validation_result.confidence_intervals.items():
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                width = upper - lower
                mean_val = validation_result.mean_metrics.get(metric, 0)
                relative_width = abs(width / mean_val) if mean_val != 0 else float('inf')
                
                if relative_width > 1.0:  # Confidence interval wider than mean
                    wide_intervals.append(metric)
        
        if wide_intervals:
            overfitting_indicators.append(f"Wide confidence intervals for: {', '.join(wide_intervals)}")
            risk_score += len(wide_intervals)
        
        # Determine risk level
        if risk_score >= 5:
            risk_level = "high"
        elif risk_score >= 3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        recommendations = self._get_overfitting_recommendations(risk_level, overfitting_indicators)
        
        return {
            "overfitting_risk": risk_level,
            "risk_score": risk_score,
            "indicators": overfitting_indicators,
            "recommendations": recommendations,
            "stability_score": stability_score,
            "num_folds": validation_result.n_splits,
        }
    
    def _get_overfitting_recommendations(self, risk_level: str, indicators: List[str]) -> List[str]:
        """Generate recommendations for overfitting risk."""
        recommendations = []
        
        if risk_level == "high":
            recommendations.extend([
                "Reduce model complexity or parameter count",
                "Increase validation data size",
                "Use regularization techniques",
                "Consider ensemble methods",
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Monitor out-of-sample performance carefully",
                "Consider additional validation methods",
                "Use cross-validation with more folds",
            ])
        
        if any("variability" in indicator for indicator in indicators):
            recommendations.append("Focus on parameter stability")
        
        if any("Insufficient" in indicator for indicator in indicators):
            recommendations.append("Increase number of validation periods")
        
        return recommendations


class ValidationSummaryGenerator:
    """Generator for comprehensive validation summaries."""
    
    def __init__(self):
        self.metrics_calculator = ValidationMetricsCalculator()
        self.stability_analyzer = StabilityAnalyzer()
        self.degradation_detector = PerformanceDegradationDetector()
        self.overfitting_detector = OverfittingDetector()
    
    def generate_comprehensive_summary(
        self, 
        validation_result: ValidationResult,
        primary_metric: str = 'total_return'
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive validation summary report.
        
        Args:
            validation_result: Validation result to summarize
            primary_metric: Primary metric for focused analysis
            
        Returns:
            Comprehensive summary dictionary
        """
        summary = {
            "validation_overview": {
                "method": validation_result.method.value,
                "num_folds": validation_result.n_splits,
                "primary_metric": primary_metric,
                "overall_stability": validation_result.stability_score,
            },
            "comprehensive_metrics": self.metrics_calculator.calculate_comprehensive_metrics(validation_result),
            "stability_analysis": self.stability_analyzer.analyze_stability(validation_result, primary_metric),
            "degradation_analysis": self.degradation_detector.detect_degradation(validation_result),
            "overfitting_analysis": self.overfitting_detector.detect_overfitting(validation_result),
        }
        
        # Generate final recommendations
        summary["final_recommendations"] = self._generate_final_recommendations(summary)
        
        # Overall validation score
        summary["overall_validation_score"] = self._calculate_overall_score(summary)
        
        return summary
    
    def _generate_final_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate final recommendations based on all analyses."""
        recommendations = set()  # Use set to avoid duplicates
        
        # Collect recommendations from all analyses
        for analysis_key in ["comprehensive_metrics", "stability_analysis", "degradation_analysis", "overfitting_analysis"]:
            analysis = summary.get(analysis_key, {})
            if isinstance(analysis, dict):
                # Look for recommendations in various places
                for key, value in analysis.items():
                    if isinstance(value, dict) and "recommendations" in value:
                        recommendations.update(value["recommendations"])
                    elif key == "recommendations" and isinstance(value, list):
                        recommendations.update(value)
        
        return sorted(list(recommendations))
    
    def _calculate_overall_score(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate an overall validation score."""
        scores = []
        
        # Stability score (0-1, higher is better)
        stability_score = 1 / (1 + summary["validation_overview"]["overall_stability"])
        scores.append(("stability", stability_score, 0.3))
        
        # Overfitting risk score (0-1, higher is better)
        overfitting_risk = summary.get("overfitting_analysis", {}).get("overfitting_risk", "medium")
        overfitting_score = {"low": 1.0, "medium": 0.6, "high": 0.2}.get(overfitting_risk, 0.6)
        scores.append(("overfitting", overfitting_score, 0.3))
        
        # Degradation score (0-1, higher is better)
        degradation_severity = (summary.get("degradation_analysis", {})
                               .get("overall_assessment", {})
                               .get("overall_severity", "moderate"))
        degradation_score = {"none": 1.0, "mild": 0.8, "moderate": 0.5, "severe": 0.2}.get(degradation_severity, 0.5)
        scores.append(("degradation", degradation_score, 0.2))
        
        # Robustness score from comprehensive metrics
        overall_assessment = (summary.get("comprehensive_metrics", {})
                             .get("overall_assessment", {}))
        robustness_score = overall_assessment.get("overall_score", 0) / 3.0  # Normalize to 0-1
        scores.append(("robustness", robustness_score, 0.2))
        
        # Calculate weighted average
        weighted_score = sum(score * weight for _, score, weight in scores)
        
        # Classify overall score
        if weighted_score >= 0.8:
            classification = "excellent"
        elif weighted_score >= 0.6:
            classification = "good"
        elif weighted_score >= 0.4:
            classification = "fair"
        else:
            classification = "poor"
        
        return {
            "overall_score": weighted_score,
            "classification": classification,
            "component_scores": {name: score for name, score, _ in scores},
            "interpretation": self._interpret_overall_score(classification, weighted_score),
        }
    
    def _interpret_overall_score(self, classification: str, score: float) -> str:
        """Provide interpretation of the overall validation score."""
        interpretations = {
            "excellent": "The strategy shows excellent validation characteristics with high stability, low overfitting risk, and robust performance.",
            "good": "The strategy demonstrates good validation results with acceptable stability and manageable risks.",
            "fair": "The strategy shows fair validation results but has some concerns that should be addressed before deployment.",
            "poor": "The strategy shows poor validation characteristics and requires significant improvement before use.",
        }
        
        base_interpretation = interpretations.get(classification, "Unable to classify validation quality.")
        return f"{base_interpretation} (Score: {score:.2f})"