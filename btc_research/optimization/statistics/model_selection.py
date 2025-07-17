"""
Model selection statistics for trading strategy comparison.

This module provides information criteria, cross-validation statistics,
and model averaging techniques for robust strategy selection.
"""

from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np

from btc_research.optimization.base import BaseStatisticsTest
from btc_research.optimization.types import StatisticsResult

__all__ = [
    "InformationCriteria",
    "CrossValidationStatistics", 
    "ModelAveraging",
    "OverfittingDetection",
    "StrategyRankingStatistics",
]


class InformationCriteria:
    """
    Information criteria for model selection and comparison.
    
    Implements various information criteria (AIC, BIC, etc.) to assess
    model quality while penalizing complexity.
    
    References:
        - Akaike, H. (1974). A new look at the statistical model identification
        - Schwarz, G. (1978). Estimating the dimension of a model
    """
    
    @staticmethod
    def calculate_aic(log_likelihood: float, n_parameters: int) -> float:
        """
        Calculate Akaike Information Criterion.
        
        Args:
            log_likelihood: Log-likelihood of the model
            n_parameters: Number of model parameters
            
        Returns:
            AIC value (lower is better)
        """
        return 2 * n_parameters - 2 * log_likelihood
    
    @staticmethod
    def calculate_bic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
        """
        Calculate Bayesian Information Criterion.
        
        Args:
            log_likelihood: Log-likelihood of the model
            n_parameters: Number of model parameters
            n_observations: Number of observations
            
        Returns:
            BIC value (lower is better)
        """
        return np.log(n_observations) * n_parameters - 2 * log_likelihood
    
    @staticmethod
    def calculate_aicc(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
        """
        Calculate corrected Akaike Information Criterion.
        
        Args:
            log_likelihood: Log-likelihood of the model
            n_parameters: Number of model parameters
            n_observations: Number of observations
            
        Returns:
            AICc value (lower is better)
        """
        aic = InformationCriteria.calculate_aic(log_likelihood, n_parameters)
        
        if n_observations - n_parameters - 1 <= 0:
            return float('inf')  # Undefined for small samples
        
        correction = (2 * n_parameters * (n_parameters + 1)) / (n_observations - n_parameters - 1)
        return aic + correction
    
    @staticmethod
    def calculate_hqic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
        """
        Calculate Hannan-Quinn Information Criterion.
        
        Args:
            log_likelihood: Log-likelihood of the model
            n_parameters: Number of model parameters
            n_observations: Number of observations
            
        Returns:
            HQIC value (lower is better)
        """
        if n_observations <= 2:
            return float('inf')
        
        return 2 * n_parameters * np.log(np.log(n_observations)) - 2 * log_likelihood
    
    @staticmethod
    def compare_models(
        log_likelihoods: List[float],
        n_parameters_list: List[int],
        n_observations: int,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models using information criteria.
        
        Args:
            log_likelihoods: Log-likelihoods for each model
            n_parameters_list: Number of parameters for each model
            n_observations: Number of observations
            model_names: Names for each model (optional)
            
        Returns:
            Dictionary with model comparison results
        """
        n_models = len(log_likelihoods)
        
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(n_models)]
        
        results = {}
        
        for i, (ll, n_params, name) in enumerate(zip(log_likelihoods, n_parameters_list, model_names)):
            aic = InformationCriteria.calculate_aic(ll, n_params)
            bic = InformationCriteria.calculate_bic(ll, n_params, n_observations)
            aicc = InformationCriteria.calculate_aicc(ll, n_params, n_observations)
            hqic = InformationCriteria.calculate_hqic(ll, n_params, n_observations)
            
            results[name] = {
                "log_likelihood": ll,
                "n_parameters": n_params,
                "AIC": aic,
                "BIC": bic,
                "AICc": aicc,
                "HQIC": hqic,
            }
        
        return results
    
    @staticmethod
    def model_weights(criterion_values: List[float]) -> List[float]:
        """
        Calculate Akaike weights for model averaging.
        
        Args:
            criterion_values: Information criterion values (AIC, BIC, etc.)
            
        Returns:
            Model weights (probabilities) summing to 1
        """
        if not criterion_values:
            return []
        
        criterion_values = np.array(criterion_values)
        
        # Handle infinite values
        finite_mask = np.isfinite(criterion_values)
        if not np.any(finite_mask):
            # All values are infinite, return equal weights
            return [1.0 / len(criterion_values)] * len(criterion_values)
        
        # Calculate relative likelihoods
        min_criterion = np.min(criterion_values[finite_mask])
        delta_criterion = criterion_values - min_criterion
        
        # Set infinite values to very large number
        delta_criterion[~finite_mask] = 1000
        
        # Calculate weights
        weights = np.exp(-0.5 * delta_criterion)
        weights = weights / np.sum(weights)
        
        return weights.tolist()


class CrossValidationStatistics(BaseStatisticsTest):
    """
    Statistical significance testing for cross-validation results.
    
    Provides methods to test whether cross-validation performance
    differences are statistically significant.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        cv_method: str = "paired_cv",
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Test statistical significance of cross-validation results.
        
        Args:
            sample1: CV scores from first model
            sample2: CV scores from second model (for comparison)
            cv_method: Type of CV test ('paired_cv', 'corrected_cv')
            **kwargs: Additional test parameters
            
        Returns:
            Statistical test result
        """
        self._validate_samples(sample1, sample2)
        
        cv_scores1 = np.array(sample1)
        
        if sample2 is None:
            return self._single_cv_test(cv_scores1)
        else:
            cv_scores2 = np.array(sample2)
            if cv_method == "paired_cv":
                return self._paired_cv_test(cv_scores1, cv_scores2)
            elif cv_method == "corrected_cv":
                return self._corrected_cv_test(cv_scores1, cv_scores2)
            else:
                raise ValueError(f"Unknown CV method: {cv_method}")
    
    def _single_cv_test(self, cv_scores: np.ndarray) -> StatisticsResult:
        """
        Test if CV performance is significantly different from zero.
        
        Args:
            cv_scores: Cross-validation scores
            
        Returns:
            Statistical test result
        """
        n_folds = len(cv_scores)
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores, ddof=1)
        
        if std_score == 0 or n_folds <= 1:
            t_statistic = float('inf') if mean_score > 0 else 0.0
            p_value = 0.0 if mean_score > 0 else 1.0
        else:
            # Standard t-test
            t_statistic = mean_score / (std_score / np.sqrt(n_folds))
            # Two-tailed p-value
            p_value = 2 * self._t_distribution_p_value(abs(t_statistic), n_folds - 1)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        t_critical = self._t_distribution_critical_value(alpha / 2, n_folds - 1)
        
        if std_score > 0:
            margin_error = t_critical * (std_score / np.sqrt(n_folds))
            ci_lower = mean_score - margin_error
            ci_upper = mean_score + margin_error
        else:
            ci_lower = ci_upper = mean_score
        
        return StatisticsResult(
            test_name="single_cv_test",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=mean_score / std_score if std_score > 0 else 0.0,
            sample_size=n_folds,
            assumptions_met={
                "normality": self._check_normality(cv_scores),
                "independence": False,  # CV scores are not independent
            },
        )
    
    def _paired_cv_test(self, cv_scores1: np.ndarray, cv_scores2: np.ndarray) -> StatisticsResult:
        """
        Paired t-test for comparing CV results from two models.
        
        Args:
            cv_scores1: CV scores from first model
            cv_scores2: CV scores from second model
            
        Returns:
            Statistical test result
        """
        if len(cv_scores1) != len(cv_scores2):
            raise ValueError("CV score arrays must have same length for paired test")
        
        differences = cv_scores1 - cv_scores2
        n_folds = len(differences)
        
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        if std_diff == 0 or n_folds <= 1:
            t_statistic = float('inf') if mean_diff > 0 else 0.0
            p_value = 0.0 if mean_diff > 0 else 1.0
        else:
            # Paired t-test
            t_statistic = mean_diff / (std_diff / np.sqrt(n_folds))
            p_value = 2 * self._t_distribution_p_value(abs(t_statistic), n_folds - 1)
        
        # Confidence interval for difference
        alpha = 1 - self.confidence_level
        t_critical = self._t_distribution_critical_value(alpha / 2, n_folds - 1)
        
        if std_diff > 0:
            margin_error = t_critical * (std_diff / np.sqrt(n_folds))
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error
        else:
            ci_lower = ci_upper = mean_diff
        
        return StatisticsResult(
            test_name="paired_cv_test",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=mean_diff / std_diff if std_diff > 0 else 0.0,
            sample_size=n_folds,
            assumptions_met={
                "normality": self._check_normality(differences),
                "independence": False,  # CV scores are not independent
            },
        )
    
    def _corrected_cv_test(self, cv_scores1: np.ndarray, cv_scores2: np.ndarray) -> StatisticsResult:
        """
        Corrected t-test for CV comparison (Nadeau & Bengio correction).
        
        Args:
            cv_scores1: CV scores from first model
            cv_scores2: CV scores from second model
            
        Returns:
            Statistical test result
        """
        if len(cv_scores1) != len(cv_scores2):
            raise ValueError("CV score arrays must have same length")
        
        differences = cv_scores1 - cv_scores2
        n_folds = len(differences)
        
        mean_diff = np.mean(differences)
        var_diff = np.var(differences, ddof=1)
        
        if var_diff == 0 or n_folds <= 1:
            t_statistic = float('inf') if mean_diff > 0 else 0.0
            p_value = 0.0 if mean_diff > 0 else 1.0
        else:
            # Nadeau & Bengio correction for CV
            # Assumes test set size is roughly equal to training set size
            test_train_ratio = 0.5  # Simplified assumption
            
            # Corrected variance
            corrected_var = var_diff * (1 / n_folds + test_train_ratio)
            
            if corrected_var > 0:
                t_statistic = mean_diff / np.sqrt(corrected_var)
                p_value = 2 * self._t_distribution_p_value(abs(t_statistic), n_folds - 1)
            else:
                t_statistic = 0.0
                p_value = 1.0
        
        # Confidence interval with correction
        alpha = 1 - self.confidence_level
        t_critical = self._t_distribution_critical_value(alpha / 2, n_folds - 1)
        
        if corrected_var > 0:
            margin_error = t_critical * np.sqrt(corrected_var)
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error
        else:
            ci_lower = ci_upper = mean_diff
        
        return StatisticsResult(
            test_name="corrected_cv_test",
            statistic=t_statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=mean_diff / np.sqrt(var_diff) if var_diff > 0 else 0.0,
            sample_size=n_folds,
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


class ModelAveraging:
    """
    Model averaging techniques for robust parameter estimation.
    
    Implements various model averaging methods to combine predictions
    from multiple models based on their relative performance.
    """
    
    @staticmethod
    def weight_predictions(
        predictions: List[np.ndarray], 
        weights: List[float]
    ) -> np.ndarray:
        """
        Calculate weighted average of model predictions.
        
        Args:
            predictions: List of prediction arrays from different models
            weights: Weights for each model
            
        Returns:
            Weighted average predictions
        """
        if len(predictions) != len(weights):
            raise ValueError("Number of predictions and weights must match")
        
        if not predictions:
            return np.array([])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_pred += weight * pred
        
        return weighted_pred
    
    @staticmethod
    def bayesian_model_averaging(
        models_performance: List[float],
        prior_weights: Optional[List[float]] = None
    ) -> List[float]:
        """
        Calculate Bayesian model averaging weights.
        
        Args:
            models_performance: Performance scores for each model
            prior_weights: Prior weights for each model (uniform if None)
            
        Returns:
            Posterior model weights
        """
        n_models = len(models_performance)
        
        if prior_weights is None:
            prior_weights = [1.0 / n_models] * n_models
        
        if len(prior_weights) != n_models:
            raise ValueError("Prior weights must match number of models")
        
        # Convert performance to likelihoods (assuming higher is better)
        performance = np.array(models_performance)
        
        # Handle negative performance
        if np.any(performance < 0):
            performance = performance - np.min(performance)
        
        # Calculate posterior weights
        likelihoods = np.exp(performance - np.max(performance))  # Numerical stability
        posterior_numerator = np.array(prior_weights) * likelihoods
        posterior_weights = posterior_numerator / np.sum(posterior_numerator)
        
        return posterior_weights.tolist()
    
    @staticmethod
    def stacked_averaging(
        base_predictions: List[np.ndarray],
        true_values: np.ndarray,
        meta_learner: str = "linear"
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Stacked model averaging using a meta-learner.
        
        Args:
            base_predictions: Predictions from base models
            true_values: True target values
            meta_learner: Type of meta-learner ('linear', 'ridge')
            
        Returns:
            Tuple of (stacked_predictions, meta_weights)
        """
        if not base_predictions:
            return np.array([]), []
        
        # Stack predictions as features
        X = np.column_stack(base_predictions)
        y = true_values
        
        if meta_learner == "linear":
            # Simple linear regression weights
            weights = ModelAveraging._linear_regression_weights(X, y)
        elif meta_learner == "ridge":
            # Ridge regression with regularization
            weights = ModelAveraging._ridge_regression_weights(X, y, alpha=0.01)
        else:
            raise ValueError(f"Unknown meta-learner: {meta_learner}")
        
        # Ensure non-negative weights and normalization
        weights = np.maximum(weights, 0)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        # Calculate stacked predictions
        stacked_pred = X @ weights
        
        return stacked_pred, weights.tolist()
    
    @staticmethod
    def _linear_regression_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate linear regression weights."""
        try:
            # Normal equation: weights = (X'X)^(-1) X'y
            XtX = X.T @ X
            Xty = X.T @ y
            
            # Add small regularization for numerical stability
            reg = 1e-8 * np.eye(XtX.shape[0])
            weights = np.linalg.solve(XtX + reg, Xty)
            
            return weights
        except np.linalg.LinAlgError:
            # Fallback to equal weights
            return np.ones(X.shape[1]) / X.shape[1]
    
    @staticmethod
    def _ridge_regression_weights(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        """Calculate ridge regression weights."""
        try:
            # Ridge regression: weights = (X'X + αI)^(-1) X'y
            XtX = X.T @ X
            Xty = X.T @ y
            
            ridge_matrix = XtX + alpha * np.eye(XtX.shape[0])
            weights = np.linalg.solve(ridge_matrix, Xty)
            
            return weights
        except np.linalg.LinAlgError:
            # Fallback to equal weights
            return np.ones(X.shape[1]) / X.shape[1]


class OverfittingDetection:
    """
    Statistical methods for detecting overfitting in strategy optimization.
    
    Provides various statistical tests and metrics to identify when
    a strategy has been overfit to historical data.
    """
    
    @staticmethod
    def in_sample_out_sample_degradation(
        in_sample_performance: float,
        out_sample_performance: float,
        n_trials: int = 1
    ) -> Dict[str, float]:
        """
        Calculate performance degradation metrics.
        
        Args:
            in_sample_performance: In-sample strategy performance
            out_sample_performance: Out-of-sample strategy performance
            n_trials: Number of optimization trials (for correction)
            
        Returns:
            Dictionary with overfitting metrics
        """
        # Performance degradation
        absolute_degradation = in_sample_performance - out_sample_performance
        
        if in_sample_performance != 0:
            relative_degradation = absolute_degradation / abs(in_sample_performance)
        else:
            relative_degradation = 0.0
        
        # Expected degradation due to multiple testing
        # Based on extreme value theory
        expected_degradation = OverfittingDetection._expected_degradation(n_trials)
        
        # Excess degradation (beyond what's expected from optimization)
        excess_degradation = absolute_degradation - expected_degradation
        
        return {
            "absolute_degradation": absolute_degradation,
            "relative_degradation": relative_degradation,
            "expected_degradation": expected_degradation,
            "excess_degradation": excess_degradation,
            "overfitting_ratio": excess_degradation / expected_degradation if expected_degradation > 0 else 0.0,
        }
    
    @staticmethod
    def _expected_degradation(n_trials: int) -> float:
        """
        Calculate expected performance degradation from multiple testing.
        
        Args:
            n_trials: Number of optimization trials
            
        Returns:
            Expected degradation in performance units
        """
        if n_trials <= 1:
            return 0.0
        
        # Simplified model based on extreme value theory
        # This is a rough approximation - actual values depend on the distribution
        return 0.1 * np.log(n_trials)  # Arbitrary scaling factor
    
    @staticmethod
    def parameter_stability_test(
        optimal_parameters: List[Dict[str, float]],
        parameter_names: List[str]
    ) -> Dict[str, float]:
        """
        Test stability of optimal parameters across different samples.
        
        Args:
            optimal_parameters: List of optimal parameter sets
            parameter_names: Names of parameters to analyze
            
        Returns:
            Dictionary with stability metrics for each parameter
        """
        stability_metrics = {}
        
        for param_name in parameter_names:
            # Extract parameter values
            param_values = []
            for param_set in optimal_parameters:
                if param_name in param_set:
                    param_values.append(param_set[param_name])
            
            if not param_values:
                stability_metrics[param_name] = {
                    "coefficient_of_variation": float('inf'),
                    "stability_score": 0.0,
                }
                continue
            
            param_values = np.array(param_values)
            
            # Coefficient of variation
            if np.mean(param_values) != 0:
                cv = np.std(param_values) / abs(np.mean(param_values))
            else:
                cv = float('inf') if np.std(param_values) > 0 else 0.0
            
            # Stability score (lower CV = higher stability)
            stability_score = 1.0 / (1.0 + cv) if cv < float('inf') else 0.0
            
            stability_metrics[param_name] = {
                "coefficient_of_variation": cv,
                "stability_score": stability_score,
            }
        
        return stability_metrics
    
    @staticmethod
    def complexity_penalty(
        n_parameters: int,
        n_observations: int,
        performance: float
    ) -> Dict[str, float]:
        """
        Calculate complexity-adjusted performance metrics.
        
        Args:
            n_parameters: Number of strategy parameters
            n_observations: Number of data observations
            performance: Raw strategy performance
            
        Returns:
            Dictionary with complexity-adjusted metrics
        """
        if n_observations <= n_parameters:
            return {
                "adjusted_performance": float('-inf'),
                "complexity_ratio": float('inf'),
                "effective_sample_size": 0.0,
            }
        
        # Complexity ratio
        complexity_ratio = n_parameters / n_observations
        
        # Effective sample size (accounting for parameter estimation)
        effective_sample_size = n_observations - n_parameters
        
        # Adjusted performance using AIC-like penalty
        aic_penalty = 2 * n_parameters / n_observations
        adjusted_performance = performance - aic_penalty
        
        return {
            "adjusted_performance": adjusted_performance,
            "complexity_ratio": complexity_ratio,
            "effective_sample_size": effective_sample_size,
            "aic_penalty": aic_penalty,
        }


class StrategyRankingStatistics(BaseStatisticsTest):
    """
    Statistical methods for ranking strategies with confidence intervals.
    
    Provides robust ranking methods that account for statistical uncertainty
    in performance estimates.
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        all_strategy_performances: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Rank strategies with statistical confidence intervals.
        
        Args:
            sample1: Performance scores of primary strategy
            sample2: Not used (compatibility)
            all_strategy_performances: List of performance scores for all strategies
            **kwargs: Additional parameters
            
        Returns:
            Statistical test result with ranking information
        """
        if all_strategy_performances is None:
            all_strategy_performances = [sample1]
        
        ranking_result = self._bootstrap_ranking(all_strategy_performances)
        
        # Extract primary strategy results
        primary_idx = 0  # Assume first strategy is primary
        mean_rank = ranking_result["mean_ranks"][primary_idx]
        rank_std = ranking_result["rank_std"][primary_idx]
        
        return StatisticsResult(
            test_name="strategy_ranking",
            statistic=mean_rank,
            p_value=ranking_result["ranking_p_values"][primary_idx],
            confidence_level=self.confidence_level,
            confidence_interval=ranking_result["rank_confidence_intervals"][primary_idx],
            effect_size=1.0 / mean_rank if mean_rank > 0 else 0.0,
            sample_size=len(sample1),
        )
    
    def _bootstrap_ranking(
        self, 
        strategy_performances: List[List[float]], 
        n_bootstrap: int = 10000
    ) -> Dict[str, List[float]]:
        """
        Bootstrap-based strategy ranking with confidence intervals.
        
        Args:
            strategy_performances: List of performance lists for each strategy
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with ranking statistics
        """
        n_strategies = len(strategy_performances)
        n_samples = [len(perf) for perf in strategy_performances]
        
        # Bootstrap rankings
        bootstrap_ranks = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample for each strategy
            bootstrap_means = []
            for i, performances in enumerate(strategy_performances):
                if n_samples[i] > 0:
                    bootstrap_sample = np.random.choice(performances, size=n_samples[i], replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                else:
                    bootstrap_means.append(0.0)
            
            # Rank strategies (1 = best, higher = worse)
            ranks = len(bootstrap_means) + 1 - np.argsort(np.argsort(bootstrap_means)) - 1
            bootstrap_ranks.append(ranks)
        
        bootstrap_ranks = np.array(bootstrap_ranks)
        
        # Calculate statistics
        mean_ranks = np.mean(bootstrap_ranks, axis=0)
        rank_std = np.std(bootstrap_ranks, axis=0)
        
        # Confidence intervals for ranks
        alpha = 1 - self.confidence_level
        rank_confidence_intervals = []
        for i in range(n_strategies):
            ci_lower = np.percentile(bootstrap_ranks[:, i], alpha / 2 * 100)
            ci_upper = np.percentile(bootstrap_ranks[:, i], (1 - alpha / 2) * 100)
            rank_confidence_intervals.append((ci_lower, ci_upper))
        
        # P-values for being ranked #1
        ranking_p_values = []
        for i in range(n_strategies):
            prob_rank_1 = np.mean(bootstrap_ranks[:, i] == 1)
            # Two-tailed p-value for being significantly different from random ranking
            expected_prob = 1.0 / n_strategies
            ranking_p_values.append(2 * min(prob_rank_1, 1 - prob_rank_1))
        
        return {
            "mean_ranks": mean_ranks.tolist(),
            "rank_std": rank_std.tolist(),
            "rank_confidence_intervals": rank_confidence_intervals,
            "ranking_p_values": ranking_p_values,
            "prob_rank_1": [np.mean(bootstrap_ranks[:, i] == 1) for i in range(n_strategies)],
        }