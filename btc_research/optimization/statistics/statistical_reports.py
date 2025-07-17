"""
Comprehensive statistical reporting for trading strategy analysis.

This module provides tools to generate detailed statistical reports that
combine multiple test results with proper interpretation and visualization.
"""

from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
from datetime import datetime

from btc_research.optimization.base import BaseStatisticsTest
from btc_research.optimization.types import StatisticsResult, OptimizationResult

__all__ = [
    "StatisticalReportGenerator",
    "MultipleTestingCorrector",
    "EffectSizeCalculator",
    "PowerAnalysisReporter",
    "VisualizationHelper",
]


class StatisticalReportGenerator:
    """
    Comprehensive statistical report generator.
    
    Combines multiple statistical test results into cohesive reports
    with proper interpretation and methodological considerations.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize report generator.
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def generate_full_report(
        self,
        optimization_results: List[OptimizationResult],
        statistical_tests: List[StatisticsResult],
        strategy_names: Optional[List[str]] = None,
        include_methodology: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistical report.
        
        Args:
            optimization_results: Results from optimization runs
            statistical_tests: Results from statistical tests
            strategy_names: Names of strategies (optional)
            include_methodology: Whether to include methodology section
            
        Returns:
            Dictionary containing full statistical report
        """
        if strategy_names is None:
            strategy_names = [f"Strategy_{i+1}" for i in range(len(optimization_results))]
        
        report = {
            "report_metadata": self._generate_metadata(),
            "executive_summary": self._generate_executive_summary(
                optimization_results, statistical_tests, strategy_names
            ),
            "performance_analysis": self._generate_performance_analysis(optimization_results),
            "statistical_significance": self._generate_significance_analysis(statistical_tests),
            "multiple_testing_correction": self._apply_multiple_testing_correction(statistical_tests),
            "effect_size_analysis": self._generate_effect_size_analysis(statistical_tests),
            "power_analysis": self._generate_power_analysis(statistical_tests),
            "robustness_assessment": self._generate_robustness_assessment(statistical_tests),
            "recommendations": self._generate_recommendations(
                optimization_results, statistical_tests
            ),
        }
        
        if include_methodology:
            report["methodology"] = self._generate_methodology_section(statistical_tests)
        
        return report
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            "generated_at": datetime.now().isoformat(),
            "confidence_level": self.confidence_level,
            "alpha_level": self.alpha,
            "framework_version": "1.0.0",
            "statistical_methods": [
                "Hypothesis Testing",
                "Multiple Comparison Correction", 
                "Effect Size Analysis",
                "Bootstrap Methods",
                "Bayesian Analysis",
                "Time Series Analysis"
            ]
        }
    
    def _generate_executive_summary(
        self,
        optimization_results: List[OptimizationResult],
        statistical_tests: List[StatisticsResult],
        strategy_names: List[str]
    ) -> Dict[str, Any]:
        """Generate executive summary."""
        if not optimization_results:
            return {"message": "No optimization results provided"}
        
        # Best performing strategy
        best_idx = np.argmax([result.objective_value for result in optimization_results])
        best_strategy = strategy_names[best_idx] if best_idx < len(strategy_names) else f"Strategy_{best_idx+1}"
        best_performance = optimization_results[best_idx].objective_value
        
        # Statistical significance count
        significant_tests = [test for test in statistical_tests if test.is_significant(self.alpha)]
        significance_rate = len(significant_tests) / len(statistical_tests) if statistical_tests else 0.0
        
        # Overall assessment
        if significance_rate > 0.7:
            overall_assessment = "Strong statistical evidence"
        elif significance_rate > 0.3:
            overall_assessment = "Moderate statistical evidence"
        else:
            overall_assessment = "Limited statistical evidence"
        
        return {
            "best_strategy": best_strategy,
            "best_performance": best_performance,
            "total_strategies_tested": len(optimization_results),
            "total_statistical_tests": len(statistical_tests),
            "significant_tests": len(significant_tests),
            "significance_rate": significance_rate,
            "overall_assessment": overall_assessment,
            "key_findings": self._extract_key_findings(statistical_tests),
        }
    
    def _generate_performance_analysis(
        self, 
        optimization_results: List[OptimizationResult]
    ) -> Dict[str, Any]:
        """Generate performance analysis section."""
        if not optimization_results:
            return {}
        
        performances = [result.objective_value for result in optimization_results]
        performances = np.array([p for p in performances if not np.isnan(p) and not np.isinf(p)])
        
        if len(performances) == 0:
            return {"message": "No valid performance data"}
        
        # Descriptive statistics
        stats = {
            "mean": np.mean(performances),
            "median": np.median(performances),
            "std": np.std(performances, ddof=1) if len(performances) > 1 else 0.0,
            "min": np.min(performances),
            "max": np.max(performances),
            "range": np.max(performances) - np.min(performances),
            "skewness": self._calculate_skewness(performances),
            "kurtosis": self._calculate_kurtosis(performances),
        }
        
        # Percentiles
        percentiles = {
            "5th": np.percentile(performances, 5),
            "25th": np.percentile(performances, 25),
            "75th": np.percentile(performances, 75),
            "95th": np.percentile(performances, 95),
        }
        
        # Performance distribution analysis
        distribution_analysis = {
            "is_normal": self._test_normality(performances),
            "outliers": self._detect_outliers(performances),
            "stability": self._calculate_stability(performances),
        }
        
        return {
            "descriptive_statistics": stats,
            "percentiles": percentiles,
            "distribution_analysis": distribution_analysis,
            "performance_ranking": self._rank_strategies(optimization_results),
        }
    
    def _generate_significance_analysis(
        self, 
        statistical_tests: List[StatisticsResult]
    ) -> Dict[str, Any]:
        """Generate statistical significance analysis."""
        if not statistical_tests:
            return {}
        
        # Categorize tests by p-value
        highly_significant = [t for t in statistical_tests if t.p_value < 0.01]
        significant = [t for t in statistical_tests if 0.01 <= t.p_value < 0.05]
        marginally_significant = [t for t in statistical_tests if 0.05 <= t.p_value < 0.1]
        not_significant = [t for t in statistical_tests if t.p_value >= 0.1]
        
        # P-value distribution analysis
        p_values = [t.p_value for t in statistical_tests]
        p_value_analysis = {
            "mean_p_value": np.mean(p_values),
            "median_p_value": np.median(p_values),
            "p_value_distribution": {
                "< 0.01": len(highly_significant),
                "0.01 - 0.05": len(significant),
                "0.05 - 0.1": len(marginally_significant),
                ">= 0.1": len(not_significant),
            }
        }
        
        # Test-specific analysis
        test_analysis = {}
        for test in statistical_tests:
            test_name = test.test_name
            if test_name not in test_analysis:
                test_analysis[test_name] = {
                    "count": 0,
                    "significant_count": 0,
                    "p_values": [],
                    "effect_sizes": [],
                }
            
            test_analysis[test_name]["count"] += 1
            if test.is_significant(self.alpha):
                test_analysis[test_name]["significant_count"] += 1
            test_analysis[test_name]["p_values"].append(test.p_value)
            if test.effect_size is not None:
                test_analysis[test_name]["effect_sizes"].append(test.effect_size)
        
        # Calculate success rates for each test type
        for test_name, analysis in test_analysis.items():
            analysis["success_rate"] = analysis["significant_count"] / analysis["count"]
            if analysis["effect_sizes"]:
                analysis["mean_effect_size"] = np.mean(analysis["effect_sizes"])
                analysis["median_effect_size"] = np.median(analysis["effect_sizes"])
        
        return {
            "p_value_analysis": p_value_analysis,
            "significance_categories": {
                "highly_significant": len(highly_significant),
                "significant": len(significant),
                "marginally_significant": len(marginally_significant),
                "not_significant": len(not_significant),
            },
            "test_analysis": test_analysis,
        }
    
    def _apply_multiple_testing_correction(
        self, 
        statistical_tests: List[StatisticsResult]
    ) -> Dict[str, Any]:
        """Apply multiple testing corrections."""
        if not statistical_tests:
            return {}
        
        p_values = [test.p_value for test in statistical_tests]
        test_names = [test.test_name for test in statistical_tests]
        
        # Apply different corrections
        corrections = {}
        
        # Bonferroni correction
        bonferroni_corrected = np.minimum(np.array(p_values) * len(p_values), 1.0)
        bonferroni_significant = bonferroni_corrected < self.alpha
        
        corrections["bonferroni"] = {
            "corrected_p_values": bonferroni_corrected.tolist(),
            "significant_tests": np.sum(bonferroni_significant),
            "rejection_rate": np.mean(bonferroni_significant),
        }
        
        # Benjamini-Hochberg FDR correction
        bh_significant, bh_corrected = self._benjamini_hochberg_correction(p_values)
        
        corrections["benjamini_hochberg"] = {
            "corrected_p_values": bh_corrected,
            "significant_tests": np.sum(bh_significant),
            "rejection_rate": np.mean(bh_significant),
        }
        
        # Holm correction
        holm_significant, holm_corrected = self._holm_correction(p_values)
        
        corrections["holm"] = {
            "corrected_p_values": holm_corrected,
            "significant_tests": np.sum(holm_significant),
            "rejection_rate": np.mean(holm_significant),
        }
        
        # Summary
        summary = {
            "original_significant": np.sum(np.array(p_values) < self.alpha),
            "bonferroni_significant": corrections["bonferroni"]["significant_tests"],
            "bh_significant": corrections["benjamini_hochberg"]["significant_tests"],
            "holm_significant": corrections["holm"]["significant_tests"],
            "recommendation": self._recommend_correction_method(corrections),
        }
        
        return {
            "corrections": corrections,
            "summary": summary,
            "test_names": test_names,
        }
    
    def _generate_effect_size_analysis(
        self, 
        statistical_tests: List[StatisticsResult]
    ) -> Dict[str, Any]:
        """Generate effect size analysis."""
        tests_with_effects = [t for t in statistical_tests if t.effect_size is not None]
        
        if not tests_with_effects:
            return {"message": "No effect sizes available"}
        
        effect_sizes = [t.effect_size for t in tests_with_effects]
        effect_sizes = np.array(effect_sizes)
        
        # Effect size categories (Cohen's conventions)
        small_effects = np.sum(np.abs(effect_sizes) < 0.2)
        medium_effects = np.sum((np.abs(effect_sizes) >= 0.2) & (np.abs(effect_sizes) < 0.8))
        large_effects = np.sum(np.abs(effect_sizes) >= 0.8)
        
        # Effect size statistics
        stats = {
            "mean_effect_size": np.mean(effect_sizes),
            "median_effect_size": np.median(effect_sizes),
            "std_effect_size": np.std(effect_sizes, ddof=1) if len(effect_sizes) > 1 else 0.0,
            "min_effect_size": np.min(effect_sizes),
            "max_effect_size": np.max(effect_sizes),
        }
        
        # Practical significance assessment
        practical_significance = {
            "small_effects": small_effects,
            "medium_effects": medium_effects,
            "large_effects": large_effects,
            "total_tests": len(tests_with_effects),
            "interpretation": self._interpret_effect_sizes(effect_sizes),
        }
        
        return {
            "effect_size_statistics": stats,
            "practical_significance": practical_significance,
            "effect_size_by_test": {
                t.test_name: t.effect_size for t in tests_with_effects
            },
        }
    
    def _generate_power_analysis(
        self, 
        statistical_tests: List[StatisticsResult]
    ) -> Dict[str, Any]:
        """Generate statistical power analysis."""
        tests_with_power = [t for t in statistical_tests if t.power is not None]
        
        if not tests_with_power:
            # Estimate power based on effect sizes and sample sizes
            power_estimates = []
            for test in statistical_tests:
                if test.effect_size is not None and test.sample_size > 0:
                    estimated_power = self._estimate_power(
                        test.effect_size, test.sample_size, self.alpha
                    )
                    power_estimates.append(estimated_power)
            
            if power_estimates:
                power_analysis = {
                    "estimated_power": {
                        "mean": np.mean(power_estimates),
                        "median": np.median(power_estimates),
                        "min": np.min(power_estimates),
                        "max": np.max(power_estimates),
                    },
                    "power_categories": {
                        "low_power": np.sum(np.array(power_estimates) < 0.6),
                        "adequate_power": np.sum((np.array(power_estimates) >= 0.6) & 
                                               (np.array(power_estimates) < 0.8)),
                        "high_power": np.sum(np.array(power_estimates) >= 0.8),
                    },
                    "sample_size_recommendations": self._recommend_sample_sizes(statistical_tests),
                }
            else:
                power_analysis = {"message": "Insufficient data for power analysis"}
        else:
            powers = [t.power for t in tests_with_power]
            power_analysis = {
                "power_statistics": {
                    "mean": np.mean(powers),
                    "median": np.median(powers),
                    "min": np.min(powers),
                    "max": np.max(powers),
                },
                "power_by_test": {t.test_name: t.power for t in tests_with_power},
            }
        
        return power_analysis
    
    def _generate_robustness_assessment(
        self, 
        statistical_tests: List[StatisticsResult]
    ) -> Dict[str, Any]:
        """Generate robustness assessment."""
        # Assess assumption violations
        assumption_violations = {}
        for test in statistical_tests:
            test_name = test.test_name
            if test_name not in assumption_violations:
                assumption_violations[test_name] = {
                    "total_tests": 0,
                    "violations": {},
                }
            
            assumption_violations[test_name]["total_tests"] += 1
            
            for assumption, met in test.assumptions_met.items():
                if assumption not in assumption_violations[test_name]["violations"]:
                    assumption_violations[test_name]["violations"][assumption] = 0
                if not met:
                    assumption_violations[test_name]["violations"][assumption] += 1
        
        # Calculate violation rates
        for test_name, data in assumption_violations.items():
            total = data["total_tests"]
            for assumption in data["violations"]:
                data["violations"][assumption] = {
                    "count": data["violations"][assumption],
                    "rate": data["violations"][assumption] / total if total > 0 else 0.0,
                }
        
        # Overall robustness score
        robustness_scores = []
        for test in statistical_tests:
            if test.assumptions_met:
                met_assumptions = sum(test.assumptions_met.values())
                total_assumptions = len(test.assumptions_met)
                score = met_assumptions / total_assumptions if total_assumptions > 0 else 1.0
                robustness_scores.append(score)
        
        overall_robustness = np.mean(robustness_scores) if robustness_scores else 1.0
        
        return {
            "assumption_violations": assumption_violations,
            "overall_robustness_score": overall_robustness,
            "robustness_interpretation": self._interpret_robustness(overall_robustness),
            "recommendations": self._robustness_recommendations(assumption_violations),
        }
    
    def _generate_recommendations(
        self,
        optimization_results: List[OptimizationResult],
        statistical_tests: List[StatisticsResult]
    ) -> Dict[str, List[str]]:
        """Generate actionable recommendations."""
        recommendations = {
            "statistical_methodology": [],
            "strategy_selection": [],
            "further_testing": [],
            "risk_management": [],
        }
        
        # Statistical methodology recommendations
        if len(statistical_tests) > 10:
            recommendations["statistical_methodology"].append(
                "Apply multiple testing corrections due to large number of tests"
            )
        
        significant_tests = [t for t in statistical_tests if t.is_significant(self.alpha)]
        if len(significant_tests) < len(statistical_tests) * 0.3:
            recommendations["statistical_methodology"].append(
                "Consider increasing sample size or effect size for better statistical power"
            )
        
        # Strategy selection recommendations
        if optimization_results:
            performances = [r.objective_value for r in optimization_results]
            if np.std(performances) / np.mean(performances) > 0.5:
                recommendations["strategy_selection"].append(
                    "High performance variability detected - consider robustness in selection"
                )
        
        # Further testing recommendations
        bootstrap_tests = [t for t in statistical_tests if "bootstrap" in t.test_name.lower()]
        if not bootstrap_tests:
            recommendations["further_testing"].append(
                "Consider bootstrap methods for more robust statistical inference"
            )
        
        time_series_tests = [t for t in statistical_tests if any(
            ts_test in t.test_name.lower() for ts_test in ["adf", "kpss", "arch", "ljung"]
        )]
        if not time_series_tests:
            recommendations["further_testing"].append(
                "Conduct time series specific tests for return autocorrelation and stationarity"
            )
        
        # Risk management recommendations
        effect_sizes = [t.effect_size for t in statistical_tests if t.effect_size is not None]
        if effect_sizes and np.mean(np.abs(effect_sizes)) < 0.2:
            recommendations["risk_management"].append(
                "Small effect sizes detected - ensure economic significance alongside statistical significance"
            )
        
        return recommendations
    
    def _generate_methodology_section(
        self, 
        statistical_tests: List[StatisticsResult]
    ) -> Dict[str, Any]:
        """Generate methodology section."""
        test_types = list(set(test.test_name for test in statistical_tests))
        
        methodology = {
            "confidence_level": self.confidence_level,
            "alpha_level": self.alpha,
            "tests_conducted": test_types,
            "sample_sizes": [test.sample_size for test in statistical_tests],
            "multiple_testing_considerations": {
                "correction_applied": len(statistical_tests) > 1,
                "recommended_methods": ["Benjamini-Hochberg FDR", "Bonferroni"],
            },
            "assumptions": self._summarize_assumptions(statistical_tests),
            "limitations": self._identify_limitations(statistical_tests),
        }
        
        return methodology
    
    # Helper methods
    
    def _extract_key_findings(self, statistical_tests: List[StatisticsResult]) -> List[str]:
        """Extract key findings from statistical tests."""
        findings = []
        
        significant_tests = [t for t in statistical_tests if t.is_significant(self.alpha)]
        if significant_tests:
            findings.append(f"{len(significant_tests)} tests showed statistical significance")
        
        large_effects = [t for t in statistical_tests 
                        if t.effect_size is not None and abs(t.effect_size) >= 0.8]
        if large_effects:
            findings.append(f"{len(large_effects)} tests showed large effect sizes")
        
        return findings
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _test_normality(self, data: np.ndarray) -> bool:
        """Simple normality test."""
        if len(data) < 3:
            return True
        
        skewness = abs(self._calculate_skewness(data))
        kurtosis = abs(self._calculate_kurtosis(data))
        
        return skewness < 2 and kurtosis < 7
    
    def _detect_outliers(self, data: np.ndarray) -> List[int]:
        """Detect outliers using IQR method."""
        if len(data) < 4:
            return []
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        return outliers.tolist()
    
    def _calculate_stability(self, data: np.ndarray) -> float:
        """Calculate stability score."""
        if len(data) <= 1:
            return 1.0
        
        cv = np.std(data, ddof=1) / abs(np.mean(data)) if np.mean(data) != 0 else float('inf')
        return 1.0 / (1.0 + cv) if cv < float('inf') else 0.0
    
    def _rank_strategies(self, optimization_results: List[OptimizationResult]) -> List[Dict[str, Any]]:
        """Rank strategies by performance."""
        performances = [(i, result.objective_value) for i, result in enumerate(optimization_results)]
        performances.sort(key=lambda x: x[1], reverse=True)
        
        rankings = []
        for rank, (idx, performance) in enumerate(performances, 1):
            rankings.append({
                "rank": rank,
                "strategy_index": idx,
                "performance": performance,
            })
        
        return rankings
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> Tuple[List[bool], List[float]]:
        """Apply Benjamini-Hochberg FDR correction."""
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if n_tests == 0:
            return [], []
        
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        rejections = np.zeros(n_tests, dtype=bool)
        
        for k in range(n_tests - 1, -1, -1):
            if sorted_p_values[k] <= (k + 1) / n_tests * self.alpha:
                rejections[sorted_indices[:k + 1]] = True
                break
        
        corrected_p_values = np.zeros(n_tests)
        for i in range(n_tests):
            corrected_p_values[sorted_indices[i]] = min(
                sorted_p_values[i] * n_tests / (i + 1), 1.0
            )
        
        return rejections.tolist(), corrected_p_values.tolist()
    
    def _holm_correction(self, p_values: List[float]) -> Tuple[List[bool], List[float]]:
        """Apply Holm step-down correction."""
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if n_tests == 0:
            return [], []
        
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        rejections = np.zeros(n_tests, dtype=bool)
        corrected_p_values = np.zeros(n_tests)
        
        for i in range(n_tests):
            corrected_alpha = self.alpha / (n_tests - i)
            corrected_p_values[sorted_indices[i]] = min(
                sorted_p_values[i] * (n_tests - i), 1.0
            )
            
            if sorted_p_values[i] <= corrected_alpha:
                rejections[sorted_indices[i]] = True
            else:
                break
        
        return rejections.tolist(), corrected_p_values.tolist()
    
    def _recommend_correction_method(self, corrections: Dict[str, Any]) -> str:
        """Recommend best correction method."""
        bh_rate = corrections["benjamini_hochberg"]["rejection_rate"]
        bonf_rate = corrections["bonferroni"]["rejection_rate"]
        
        if bonf_rate > 0 and bh_rate > bonf_rate * 1.5:
            return "Benjamini-Hochberg (better power while controlling FDR)"
        elif bonf_rate > 0:
            return "Bonferroni (conservative, controls FWER)"
        else:
            return "Consider increasing sample size or effect sizes"
    
    def _interpret_effect_sizes(self, effect_sizes: np.ndarray) -> str:
        """Interpret effect sizes."""
        abs_effects = np.abs(effect_sizes)
        large_prop = np.mean(abs_effects >= 0.8)
        medium_prop = np.mean((abs_effects >= 0.2) & (abs_effects < 0.8))
        small_prop = np.mean(abs_effects < 0.2)
        
        if large_prop > 0.5:
            return "Predominantly large effect sizes - strong practical significance"
        elif medium_prop > 0.5:
            return "Predominantly medium effect sizes - moderate practical significance"
        else:
            return "Predominantly small effect sizes - limited practical significance"
    
    def _estimate_power(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """Estimate statistical power."""
        # Simplified power calculation for t-test
        if sample_size <= 1:
            return 0.0
        
        import math
        
        # Non-centrality parameter
        ncp = abs(effect_size) * math.sqrt(sample_size)
        
        # Critical value
        z_critical = 1.96 if alpha == 0.05 else 2.58 if alpha == 0.01 else 1.64
        
        # Power approximation
        power = 1 - self._normal_cdf(z_critical - ncp)
        
        return max(0.0, min(1.0, power))
    
    def _normal_cdf(self, x: float) -> float:
        """Normal CDF approximation."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))
    
    def _recommend_sample_sizes(self, statistical_tests: List[StatisticsResult]) -> Dict[str, int]:
        """Recommend sample sizes for adequate power."""
        recommendations = {}
        
        for test in statistical_tests:
            if test.effect_size is not None:
                # Target power = 0.8
                recommended_n = self._calculate_required_sample_size(
                    abs(test.effect_size), 0.8, self.alpha
                )
                recommendations[test.test_name] = recommended_n
        
        return recommendations
    
    def _calculate_required_sample_size(
        self, 
        effect_size: float, 
        power: float, 
        alpha: float
    ) -> int:
        """Calculate required sample size for desired power."""
        if effect_size == 0:
            return float('inf')
        
        # Simplified calculation
        z_alpha = 1.96 if alpha == 0.05 else 2.58
        z_beta = 0.84 if power == 0.8 else 1.28  # For power = 0.8 or 0.9
        
        n = ((z_alpha + z_beta) / effect_size) ** 2
        return max(10, int(np.ceil(n)))
    
    def _interpret_robustness(self, robustness_score: float) -> str:
        """Interpret robustness score."""
        if robustness_score >= 0.9:
            return "Excellent - assumptions well met"
        elif robustness_score >= 0.7:
            return "Good - minor assumption violations"
        elif robustness_score >= 0.5:
            return "Moderate - some assumption violations"
        else:
            return "Poor - major assumption violations, consider robust alternatives"
    
    def _robustness_recommendations(self, assumption_violations: Dict[str, Any]) -> List[str]:
        """Generate robustness recommendations."""
        recommendations = []
        
        # Check for common violations
        for test_name, data in assumption_violations.items():
            for assumption, violation_data in data["violations"].items():
                if violation_data["rate"] > 0.5:
                    if assumption == "normality":
                        recommendations.append(
                            f"Consider non-parametric alternatives for {test_name} due to normality violations"
                        )
                    elif assumption == "independence":
                        recommendations.append(
                            f"Address autocorrelation issues in {test_name}"
                        )
                    elif assumption == "homoscedasticity":
                        recommendations.append(
                            f"Consider robust standard errors for {test_name} due to heteroscedasticity"
                        )
        
        return recommendations
    
    def _summarize_assumptions(self, statistical_tests: List[StatisticsResult]) -> Dict[str, List[str]]:
        """Summarize assumptions across all tests."""
        all_assumptions = set()
        for test in statistical_tests:
            all_assumptions.update(test.assumptions_met.keys())
        
        assumption_summary = {}
        for assumption in all_assumptions:
            tests_with_assumption = [
                test.test_name for test in statistical_tests 
                if assumption in test.assumptions_met
            ]
            assumption_summary[assumption] = tests_with_assumption
        
        return assumption_summary
    
    def _identify_limitations(self, statistical_tests: List[StatisticsResult]) -> List[str]:
        """Identify limitations in the analysis."""
        limitations = []
        
        if len(statistical_tests) > 20:
            limitations.append("Large number of tests increases multiple testing concerns")
        
        small_samples = [t for t in statistical_tests if t.sample_size < 30]
        if small_samples:
            limitations.append("Some tests have small sample sizes, limiting power")
        
        missing_effects = [t for t in statistical_tests if t.effect_size is None]
        if missing_effects:
            limitations.append("Effect sizes not available for all tests")
        
        return limitations


class MultipleTestingCorrector:
    """
    Specialized class for multiple testing corrections.
    
    Provides various methods to control family-wise error rate
    and false discovery rate in multiple hypothesis testing.
    """
    
    @staticmethod
    def correct_p_values(
        p_values: List[float], 
        method: str = "benjamini_hochberg",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Apply multiple testing correction to p-values.
        
        Args:
            p_values: List of p-values to correct
            method: Correction method ('bonferroni', 'benjamini_hochberg', 'holm')
            alpha: Significance level
            
        Returns:
            Dictionary with correction results
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if method == "bonferroni":
            corrected = np.minimum(p_values * n_tests, 1.0)
            significant = corrected < alpha
            
        elif method == "benjamini_hochberg":
            significant, corrected = MultipleTestingCorrector._bh_correction(p_values, alpha)
            
        elif method == "holm":
            significant, corrected = MultipleTestingCorrector._holm_correction(p_values, alpha)
            
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        return {
            "method": method,
            "original_p_values": p_values.tolist(),
            "corrected_p_values": corrected if isinstance(corrected, list) else corrected.tolist(),
            "significant": significant if isinstance(significant, list) else significant.tolist(),
            "n_significant": np.sum(significant),
            "rejection_rate": np.mean(significant),
        }
    
    @staticmethod
    def _bh_correction(p_values: np.ndarray, alpha: float) -> Tuple[List[bool], List[float]]:
        """Benjamini-Hochberg correction."""
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        rejections = np.zeros(n_tests, dtype=bool)
        
        for k in range(n_tests - 1, -1, -1):
            if sorted_p_values[k] <= (k + 1) / n_tests * alpha:
                rejections[sorted_indices[:k + 1]] = True
                break
        
        corrected_p_values = np.zeros(n_tests)
        for i in range(n_tests):
            corrected_p_values[sorted_indices[i]] = min(
                sorted_p_values[i] * n_tests / (i + 1), 1.0
            )
        
        return rejections.tolist(), corrected_p_values.tolist()
    
    @staticmethod
    def _holm_correction(p_values: np.ndarray, alpha: float) -> Tuple[List[bool], List[float]]:
        """Holm step-down correction."""
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        rejections = np.zeros(n_tests, dtype=bool)
        corrected_p_values = np.zeros(n_tests)
        
        for i in range(n_tests):
            corrected_alpha = alpha / (n_tests - i)
            corrected_p_values[sorted_indices[i]] = min(
                sorted_p_values[i] * (n_tests - i), 1.0
            )
            
            if sorted_p_values[i] <= corrected_alpha:
                rejections[sorted_indices[i]] = True
            else:
                break
        
        return rejections.tolist(), corrected_p_values.tolist()


class EffectSizeCalculator:
    """
    Calculator for various effect size measures.
    
    Provides standardized effect size calculations for
    different types of statistical tests.
    """
    
    @staticmethod
    def cohens_d(sample1: List[float], sample2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        x1, x2 = np.array(sample1), np.array(sample2)
        n1, n2 = len(x1), len(x2)
        
        if n1 <= 1 or n2 <= 1:
            return 0.0
        
        # Pooled standard deviation
        s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(x1) - np.mean(x2)) / pooled_std
    
    @staticmethod
    def glass_delta(sample1: List[float], sample2: List[float]) -> float:
        """Calculate Glass's delta effect size."""
        x1, x2 = np.array(sample1), np.array(sample2)
        
        if len(x2) <= 1:
            return 0.0
        
        s2 = np.std(x2, ddof=1)
        if s2 == 0:
            return 0.0
        
        return (np.mean(x1) - np.mean(x2)) / s2
    
    @staticmethod
    def eta_squared(f_statistic: float, df1: int, df2: int) -> float:
        """Calculate eta-squared effect size for F-test."""
        if df1 <= 0 or df2 <= 0:
            return 0.0
        
        return (f_statistic * df1) / (f_statistic * df1 + df2)
    
    @staticmethod
    def cramers_v(chi_squared: float, n: int, df: int) -> float:
        """Calculate Cramer's V effect size for chi-square test."""
        if n <= 0 or df <= 0:
            return 0.0
        
        return np.sqrt(chi_squared / (n * df))


class PowerAnalysisReporter:
    """
    Specialized reporter for statistical power analysis.
    
    Provides detailed power analysis reports and recommendations
    for experimental design.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize power analysis reporter.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def analyze_power(
        self,
        statistical_tests: List[StatisticsResult],
        target_power: float = 0.8
    ) -> Dict[str, Any]:
        """
        Analyze statistical power across tests.
        
        Args:
            statistical_tests: List of statistical test results
            target_power: Target power level
            
        Returns:
            Power analysis report
        """
        power_analysis = {
            "target_power": target_power,
            "alpha": self.alpha,
            "test_analysis": [],
            "summary": {},
            "recommendations": [],
        }
        
        adequate_power_count = 0
        total_tests = 0
        
        for test in statistical_tests:
            if test.effect_size is not None and test.sample_size > 0:
                estimated_power = self._estimate_power(
                    test.effect_size, test.sample_size, self.alpha
                )
                
                required_n = self._calculate_required_n(
                    test.effect_size, target_power, self.alpha
                )
                
                test_analysis = {
                    "test_name": test.test_name,
                    "effect_size": test.effect_size,
                    "sample_size": test.sample_size,
                    "estimated_power": estimated_power,
                    "adequate_power": estimated_power >= target_power,
                    "required_sample_size": required_n,
                    "power_deficit": max(0, target_power - estimated_power),
                }
                
                power_analysis["test_analysis"].append(test_analysis)
                
                if estimated_power >= target_power:
                    adequate_power_count += 1
                total_tests += 1
        
        # Summary statistics
        if total_tests > 0:
            power_analysis["summary"] = {
                "total_tests": total_tests,
                "adequate_power_count": adequate_power_count,
                "adequate_power_rate": adequate_power_count / total_tests,
                "mean_power": np.mean([t["estimated_power"] for t in power_analysis["test_analysis"]]),
                "min_power": np.min([t["estimated_power"] for t in power_analysis["test_analysis"]]),
                "max_power": np.max([t["estimated_power"] for t in power_analysis["test_analysis"]]),
            }
            
            # Generate recommendations
            power_analysis["recommendations"] = self._generate_power_recommendations(
                power_analysis["test_analysis"], target_power
            )
        
        return power_analysis
    
    def _estimate_power(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """Estimate statistical power for t-test."""
        if sample_size <= 1:
            return 0.0
        
        # Non-centrality parameter
        ncp = abs(effect_size) * np.sqrt(sample_size)
        
        # Critical value
        z_critical = 1.96 if alpha == 0.05 else 2.58 if alpha == 0.01 else 1.64
        
        # Power calculation
        power = 1 - self._normal_cdf(z_critical - ncp)
        
        return max(0.0, min(1.0, power))
    
    def _calculate_required_n(self, effect_size: float, power: float, alpha: float) -> int:
        """Calculate required sample size for desired power."""
        if effect_size == 0:
            return float('inf')
        
        z_alpha = 1.96 if alpha == 0.05 else 2.58
        z_beta = 0.84 if power == 0.8 else 1.28
        
        n = ((z_alpha + z_beta) / abs(effect_size)) ** 2
        return max(10, int(np.ceil(n)))
    
    def _normal_cdf(self, x: float) -> float:
        """Normal CDF approximation."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))
    
    def _generate_power_recommendations(
        self, 
        test_analysis: List[Dict[str, Any]], 
        target_power: float
    ) -> List[str]:
        """Generate power analysis recommendations."""
        recommendations = []
        
        low_power_tests = [t for t in test_analysis if t["estimated_power"] < target_power]
        
        if low_power_tests:
            avg_required_increase = np.mean([
                t["required_sample_size"] / t["sample_size"] 
                for t in low_power_tests if t["sample_size"] > 0
            ])
            
            recommendations.append(
                f"Increase sample size by approximately {avg_required_increase:.1f}x "
                f"for {len(low_power_tests)} underpowered tests"
            )
        
        small_effect_tests = [t for t in test_analysis if abs(t["effect_size"]) < 0.2]
        if small_effect_tests:
            recommendations.append(
                f"{len(small_effect_tests)} tests have small effect sizes - "
                "consider practical significance alongside statistical significance"
            )
        
        return recommendations


class VisualizationHelper:
    """
    Helper class for creating statistical visualizations.
    
    Provides data preparation for common statistical plots
    and visualization recommendations.
    """
    
    @staticmethod
    def prepare_p_value_histogram(statistical_tests: List[StatisticsResult]) -> Dict[str, Any]:
        """Prepare data for p-value histogram."""
        p_values = [test.p_value for test in statistical_tests]
        
        return {
            "p_values": p_values,
            "bins": np.linspace(0, 1, 21),
            "expected_uniform": len(p_values) / 20,  # Expected count per bin under null
            "interpretation": VisualizationHelper._interpret_p_value_distribution(p_values),
        }
    
    @staticmethod
    def prepare_effect_size_plot(statistical_tests: List[StatisticsResult]) -> Dict[str, Any]:
        """Prepare data for effect size visualization."""
        tests_with_effects = [t for t in statistical_tests if t.effect_size is not None]
        
        if not tests_with_effects:
            return {"message": "No effect sizes available"}
        
        effect_sizes = [t.effect_size for t in tests_with_effects]
        test_names = [t.test_name for t in tests_with_effects]
        p_values = [t.p_value for t in tests_with_effects]
        
        return {
            "effect_sizes": effect_sizes,
            "test_names": test_names,
            "p_values": p_values,
            "cohen_categories": {
                "small": 0.2,
                "medium": 0.5,
                "large": 0.8,
            },
        }
    
    @staticmethod
    def prepare_power_analysis_plot(power_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for power analysis visualization."""
        if "test_analysis" not in power_analysis:
            return {"message": "No power analysis data available"}
        
        test_data = power_analysis["test_analysis"]
        
        return {
            "test_names": [t["test_name"] for t in test_data],
            "estimated_power": [t["estimated_power"] for t in test_data],
            "effect_sizes": [t["effect_size"] for t in test_data],
            "sample_sizes": [t["sample_size"] for t in test_data],
            "target_power": power_analysis.get("target_power", 0.8),
        }
    
    @staticmethod
    def _interpret_p_value_distribution(p_values: List[float]) -> str:
        """Interpret p-value distribution."""
        p_array = np.array(p_values)
        
        # Check for uniform distribution (expected under null hypothesis)
        low_p_prop = np.mean(p_array < 0.05)
        
        if low_p_prop > 0.2:
            return "Many significant results - possible true effects or multiple testing issues"
        elif low_p_prop < 0.02:
            return "Few significant results - possible low power or no true effects"
        else:
            return "P-value distribution appears reasonable"