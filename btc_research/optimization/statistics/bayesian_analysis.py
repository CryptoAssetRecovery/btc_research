"""
Bayesian statistical analysis for trading strategy evaluation.

This module provides Bayesian methods for model comparison, parameter estimation,
and uncertainty quantification in strategy optimization.
"""

from typing import List, Optional, Dict, Any, Tuple, Callable, Union
import numpy as np

from btc_research.optimization.base import BaseStatisticsTest
from btc_research.optimization.types import StatisticsResult

__all__ = [
    "BayesianModelComparison",
    "BayesianParameterEstimation",
    "BayesianOptimizationUncertainty", 
    "PriorSensitivityAnalysis",
]


class BayesianModelComparison(BaseStatisticsTest):
    """
    Bayesian model comparison using Bayes factors and model evidence.
    
    Implements Bayesian methods for comparing trading strategies and
    calculating the probability that one model is better than another.
    
    References:
        - Kass, R. E. & Raftery, A. E. (1995). Bayes factors
        - Gelman, A. et al. (2013). Bayesian Data Analysis
    """
    
    def run_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        prior_odds: float = 1.0,
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Compare models using Bayesian methods.
        
        Args:
            sample1: Performance data from first model
            sample2: Performance data from second model
            prior_odds: Prior odds ratio (model1/model2)
            **kwargs: Additional parameters
            
        Returns:
            Statistical test result with Bayes factor
        """
        if sample2 is None:
            return self._single_model_analysis(sample1)
        else:
            return self._two_model_comparison(sample1, sample2, prior_odds)
    
    def _single_model_analysis(self, data: List[float]) -> StatisticsResult:
        """
        Bayesian analysis of a single model.
        
        Args:
            data: Performance data
            
        Returns:
            Statistical test result
        """
        data_array = np.array(data)
        n = len(data_array)
        
        if n == 0:
            return StatisticsResult(
                test_name="bayesian_single_model",
                statistic=0.0,
                p_value=0.5,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                sample_size=0,
            )
        
        # Bayesian estimation assuming normal likelihood with unknown mean and variance
        posterior_params = self._normal_inverse_gamma_posterior(data_array)
        
        # Posterior probability that mean > 0
        prob_positive = self._posterior_probability_positive(posterior_params, n)
        
        # Credible interval for the mean
        alpha = 1 - self.confidence_level
        credible_interval = self._credible_interval_mean(posterior_params, alpha, n)
        
        # Bayes factor vs null hypothesis (mean = 0)
        bayes_factor = self._bayes_factor_vs_null(data_array)
        
        return StatisticsResult(
            test_name="bayesian_single_model",
            statistic=bayes_factor,
            p_value=1 - prob_positive,  # P(mean <= 0)
            confidence_level=self.confidence_level,
            confidence_interval=credible_interval,
            effect_size=posterior_params["mean"],
            sample_size=n,
            assumptions_met={
                "normality": True,  # Assumed for Bayesian normal model
                "exchangeability": True,
            },
        )
    
    def _two_model_comparison(
        self, 
        data1: List[float], 
        data2: List[float], 
        prior_odds: float
    ) -> StatisticsResult:
        """
        Bayesian comparison of two models.
        
        Args:
            data1: Performance data from first model
            data2: Performance data from second model  
            prior_odds: Prior odds ratio
            
        Returns:
            Statistical test result
        """
        data1_array = np.array(data1)
        data2_array = np.array(data2)
        
        # Calculate Bayes factor
        bayes_factor = self._calculate_bayes_factor(data1_array, data2_array)
        
        # Posterior odds = prior odds × Bayes factor
        posterior_odds = prior_odds * bayes_factor
        
        # Posterior probability that model 1 is better
        prob_model1_better = posterior_odds / (1 + posterior_odds)
        
        # Effect size (difference in means)
        mean1 = np.mean(data1_array) if len(data1_array) > 0 else 0.0
        mean2 = np.mean(data2_array) if len(data2_array) > 0 else 0.0
        effect_size = mean1 - mean2
        
        # Credible interval for difference
        credible_interval = self._credible_interval_difference(data1_array, data2_array)
        
        return StatisticsResult(
            test_name="bayesian_model_comparison",
            statistic=bayes_factor,
            p_value=1 - prob_model1_better,
            confidence_level=self.confidence_level,
            confidence_interval=credible_interval,
            effect_size=effect_size,
            sample_size=len(data1_array) + len(data2_array),
        )
    
    def _normal_inverse_gamma_posterior(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate posterior parameters for normal-inverse-gamma model.
        
        Args:
            data: Observed data
            
        Returns:
            Dictionary with posterior parameters
        """
        n = len(data)
        
        if n == 0:
            return {"mean": 0.0, "precision": 1.0, "shape": 1.0, "scale": 1.0}
        
        # Uninformative prior parameters
        mu_0 = 0.0  # Prior mean
        lambda_0 = 1e-6  # Prior precision for mean
        alpha_0 = 1e-6  # Prior shape for variance
        beta_0 = 1e-6  # Prior scale for variance
        
        # Sample statistics
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1) if n > 1 else 1.0
        
        # Posterior parameters
        lambda_n = lambda_0 + n
        mu_n = (lambda_0 * mu_0 + n * sample_mean) / lambda_n
        alpha_n = alpha_0 + n / 2
        
        if n > 1:
            beta_n = beta_0 + 0.5 * np.sum((data - sample_mean) ** 2) + \
                     (lambda_0 * n * (sample_mean - mu_0) ** 2) / (2 * lambda_n)
        else:
            beta_n = beta_0 + 0.5
        
        return {
            "mean": mu_n,
            "precision": lambda_n,
            "shape": alpha_n,
            "scale": beta_n,
        }
    
    def _posterior_probability_positive(self, posterior_params: Dict[str, float], n: int) -> float:
        """
        Calculate posterior probability that mean is positive.
        
        Args:
            posterior_params: Posterior distribution parameters
            n: Sample size
            
        Returns:
            Probability that mean > 0
        """
        # Marginal distribution of mean is t-distribution
        df = 2 * posterior_params["shape"]
        
        if df <= 0:
            return 0.5
        
        # Location and scale parameters
        loc = posterior_params["mean"]
        scale = np.sqrt(posterior_params["scale"] / (posterior_params["precision"] * posterior_params["shape"]))
        
        if scale <= 0:
            return 1.0 if loc > 0 else 0.0
        
        # Standardized value
        t_value = loc / scale
        
        # P(T > -t_value) where T ~ t(df)
        return 1 - self._t_cdf(t_value, df)
    
    def _credible_interval_mean(
        self, 
        posterior_params: Dict[str, float], 
        alpha: float, 
        n: int
    ) -> Tuple[float, float]:
        """
        Calculate credible interval for the mean.
        
        Args:
            posterior_params: Posterior distribution parameters
            alpha: Significance level
            n: Sample size
            
        Returns:
            Credible interval (lower, upper)
        """
        # Marginal distribution of mean is t-distribution
        df = 2 * posterior_params["shape"]
        
        if df <= 0:
            return (posterior_params["mean"], posterior_params["mean"])
        
        # Location and scale parameters
        loc = posterior_params["mean"]
        scale = np.sqrt(posterior_params["scale"] / (posterior_params["precision"] * posterior_params["shape"]))
        
        if scale <= 0:
            return (loc, loc)
        
        # Critical value
        t_critical = self._t_critical_value(alpha / 2, df)
        
        # Credible interval
        margin = t_critical * scale
        return (loc - margin, loc + margin)
    
    def _bayes_factor_vs_null(self, data: np.ndarray) -> float:
        """
        Calculate Bayes factor vs null hypothesis (mean = 0).
        
        Args:
            data: Observed data
            
        Returns:
            Bayes factor (alternative / null)
        """
        n = len(data)
        
        if n == 0:
            return 1.0
        
        # Sample statistics
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1) if n > 1 else 1.0
        
        if sample_var <= 0:
            return float('inf') if sample_mean != 0 else 1.0
        
        # Simplified Bayes factor calculation
        # Using default g-prior with g = n
        g = n
        
        # t-statistic
        t_stat = sample_mean / np.sqrt(sample_var / n)
        
        # Bayes factor approximation
        bf_log = -0.5 * np.log(1 + g) + 0.5 * (g / (1 + g)) * t_stat ** 2
        
        return np.exp(bf_log)
    
    def _calculate_bayes_factor(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate Bayes factor for comparing two models.
        
        Args:
            data1: Data from first model
            data2: Data from second model
            
        Returns:
            Bayes factor (model1 / model2)
        """
        # Calculate marginal likelihoods for each model
        log_ml1 = self._log_marginal_likelihood(data1)
        log_ml2 = self._log_marginal_likelihood(data2)
        
        # Bayes factor = exp(log ML1 - log ML2)
        log_bf = log_ml1 - log_ml2
        
        # Prevent overflow
        if log_bf > 100:
            return float('inf')
        elif log_bf < -100:
            return 0.0
        else:
            return np.exp(log_bf)
    
    def _log_marginal_likelihood(self, data: np.ndarray) -> float:
        """
        Calculate log marginal likelihood for normal model with uninformative prior.
        
        Args:
            data: Observed data
            
        Returns:
            Log marginal likelihood
        """
        n = len(data)
        
        if n == 0:
            return 0.0
        
        if n == 1:
            # Special case for single observation
            return -0.5 * np.log(2 * np.pi)
        
        # Sample statistics
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)
        
        if sample_var <= 0:
            return -float('inf')
        
        # Log marginal likelihood for normal model with uninformative prior
        log_ml = (
            -0.5 * n * np.log(2 * np.pi) +
            -0.5 * (n - 1) * np.log(sample_var) +
            -0.5 * np.log(n)
        )
        
        return log_ml
    
    def _credible_interval_difference(
        self, 
        data1: np.ndarray, 
        data2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate credible interval for difference in means.
        
        Args:
            data1: Data from first model
            data2: Data from second model
            
        Returns:
            Credible interval for difference
        """
        # Simplified approach using normal approximation
        n1, n2 = len(data1), len(data2)
        
        if n1 == 0 or n2 == 0:
            return (0.0, 0.0)
        
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        var1 = np.var(data1, ddof=1) if n1 > 1 else 1.0
        var2 = np.var(data2, ddof=1) if n2 > 1 else 1.0
        
        # Difference and its variance
        diff_mean = mean1 - mean2
        diff_var = var1 / n1 + var2 / n2
        diff_std = np.sqrt(diff_var)
        
        if diff_std <= 0:
            return (diff_mean, diff_mean)
        
        # Approximate degrees of freedom (Welch-Satterthwaite)
        if var1 > 0 and var2 > 0:
            df = (var1 / n1 + var2 / n2) ** 2 / (
                (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
            )
        else:
            df = min(n1, n2) - 1
        
        # Critical value
        alpha = 1 - self.confidence_level
        t_critical = self._t_critical_value(alpha / 2, df)
        
        # Credible interval
        margin = t_critical * diff_std
        return (diff_mean - margin, diff_mean + margin)
    
    def _t_cdf(self, x: float, df: float) -> float:
        """Approximate CDF for t-distribution."""
        if df <= 0:
            return 0.5
        
        if df > 30:
            return 0.5 * (1 + np.tanh(x * 0.7978845608))
        
        # Crude approximation for small df
        return 0.5 * (1 + np.tanh(x * 0.7978845608 * np.sqrt(df / (df + 2))))
    
    def _t_critical_value(self, alpha: float, df: float) -> float:
        """Calculate critical value for t-distribution."""
        if df <= 0:
            return 1.96
        
        if df > 30:
            return self._normal_critical_value(alpha)
        
        z_alpha = self._normal_critical_value(alpha)
        correction = 1 + (z_alpha ** 2 + 1) / (4 * df)
        return z_alpha * correction
    
    def _normal_critical_value(self, alpha: float) -> float:
        """Normal critical value approximation."""
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


class BayesianParameterEstimation:
    """
    Bayesian parameter estimation for trading strategies.
    
    Provides methods for estimating strategy parameters with
    full posterior distributions and uncertainty quantification.
    """
    
    def __init__(self, prior_type: str = "normal", **prior_params):
        """
        Initialize Bayesian parameter estimator.
        
        Args:
            prior_type: Type of prior distribution
            **prior_params: Parameters for the prior distribution
        """
        self.prior_type = prior_type
        self.prior_params = prior_params
    
    def estimate_parameters(
        self,
        performance_data: List[float],
        parameter_values: List[Dict[str, float]],
        parameter_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate posterior distributions for strategy parameters.
        
        Args:
            performance_data: Strategy performance for each parameter set
            parameter_values: List of parameter value dictionaries
            parameter_names: Names of parameters to estimate
            
        Returns:
            Dictionary with posterior statistics for each parameter
        """
        if len(performance_data) != len(parameter_values):
            raise ValueError("Performance data and parameter values must have same length")
        
        results = {}
        
        for param_name in parameter_names:
            # Extract parameter values and corresponding performances
            param_vals = []
            performances = []
            
            for i, param_dict in enumerate(parameter_values):
                if param_name in param_dict:
                    param_vals.append(param_dict[param_name])
                    performances.append(performance_data[i])
            
            if not param_vals:
                continue
            
            # Estimate posterior for this parameter
            posterior_stats = self._estimate_parameter_posterior(
                np.array(param_vals), 
                np.array(performances)
            )
            
            results[param_name] = posterior_stats
        
        return results
    
    def _estimate_parameter_posterior(
        self, 
        param_values: np.ndarray, 
        performances: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate posterior distribution for a single parameter.
        
        Args:
            param_values: Parameter values
            performances: Corresponding performance values
            
        Returns:
            Dictionary with posterior statistics
        """
        # Simple approach: weight parameter values by their performance
        # This is a heuristic approximation to Bayesian inference
        
        # Convert performances to weights (higher performance = higher weight)
        # Shift to make all positive if needed
        shifted_perf = performances - np.min(performances)
        
        if np.max(shifted_perf) == 0:
            # All performances are equal
            weights = np.ones(len(performances)) / len(performances)
        else:
            # Exponential weighting
            weights = np.exp(shifted_perf / np.std(shifted_perf))
            weights = weights / np.sum(weights)
        
        # Posterior statistics
        posterior_mean = np.average(param_values, weights=weights)
        posterior_var = np.average((param_values - posterior_mean) ** 2, weights=weights)
        posterior_std = np.sqrt(posterior_var)
        
        # Credible intervals (assuming normal posterior)
        alpha = 0.05  # 95% credible interval
        z_critical = 1.96
        ci_lower = posterior_mean - z_critical * posterior_std
        ci_upper = posterior_mean + z_critical * posterior_std
        
        # Mode (most likely value)
        best_idx = np.argmax(performances)
        posterior_mode = param_values[best_idx]
        
        return {
            "mean": posterior_mean,
            "std": posterior_std,
            "var": posterior_var,
            "mode": posterior_mode,
            "credible_interval_lower": ci_lower,
            "credible_interval_upper": ci_upper,
            "effective_sample_size": 1 / np.sum(weights ** 2),  # Approximation
        }
    
    def predict_performance(
        self,
        parameter_posterior: Dict[str, Dict[str, float]],
        new_parameters: Dict[str, float],
        n_samples: int = 10000
    ) -> Dict[str, float]:
        """
        Predict performance for new parameter values using posterior.
        
        Args:
            parameter_posterior: Posterior distributions for parameters
            new_parameters: New parameter values to predict
            n_samples: Number of posterior samples
            
        Returns:
            Dictionary with prediction statistics
        """
        # This is a simplified prediction method
        # In practice, you'd need a proper likelihood model
        
        prediction_samples = []
        
        for _ in range(n_samples):
            # Sample from parameter posteriors
            sampled_params = {}
            for param_name, posterior in parameter_posterior.items():
                if param_name in new_parameters:
                    # Simple normal sampling
                    sampled_value = np.random.normal(
                        posterior["mean"], 
                        posterior["std"]
                    )
                    sampled_params[param_name] = sampled_value
            
            # Predict performance (simplified linear model)
            predicted_perf = self._predict_performance_sample(
                sampled_params, 
                new_parameters
            )
            prediction_samples.append(predicted_perf)
        
        prediction_samples = np.array(prediction_samples)
        
        return {
            "mean": np.mean(prediction_samples),
            "std": np.std(prediction_samples),
            "median": np.median(prediction_samples),
            "percentile_5": np.percentile(prediction_samples, 5),
            "percentile_95": np.percentile(prediction_samples, 95),
        }
    
    def _predict_performance_sample(
        self, 
        sampled_params: Dict[str, float], 
        new_parameters: Dict[str, float]
    ) -> float:
        """
        Predict performance for a single posterior sample.
        
        Args:
            sampled_params: Sampled parameter values from posterior
            new_parameters: Target parameter values
            
        Returns:
            Predicted performance
        """
        # Simplified prediction based on parameter similarity
        # In practice, you'd use a proper predictive model
        
        if not sampled_params or not new_parameters:
            return 0.0
        
        # Calculate similarity between sampled and new parameters
        similarity = 0.0
        n_params = 0
        
        for param_name in new_parameters:
            if param_name in sampled_params:
                # Normalized difference
                diff = abs(sampled_params[param_name] - new_parameters[param_name])
                similarity += np.exp(-diff)  # Exponential decay with difference
                n_params += 1
        
        if n_params > 0:
            similarity /= n_params
        
        # Convert similarity to performance (this is very simplified)
        return similarity
    
    def parameter_importance(
        self,
        parameter_posterior: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate parameter importance based on posterior uncertainty.
        
        Args:
            parameter_posterior: Posterior distributions for parameters
            
        Returns:
            Dictionary with importance scores (higher = more important)
        """
        importance_scores = {}
        
        for param_name, posterior in parameter_posterior.items():
            # Importance based on inverse of posterior uncertainty
            # Parameters with lower uncertainty are more important
            if posterior["std"] > 0:
                importance = 1.0 / posterior["std"]
            else:
                importance = float('inf')
            
            # Normalize by credible interval width
            ci_width = posterior["credible_interval_upper"] - posterior["credible_interval_lower"]
            if ci_width > 0:
                importance = importance / ci_width
            
            importance_scores[param_name] = importance
        
        # Normalize to sum to 1
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            for param_name in importance_scores:
                importance_scores[param_name] /= total_importance
        
        return importance_scores


class BayesianOptimizationUncertainty:
    """
    Uncertainty quantification for Bayesian optimization results.
    
    Provides methods to quantify and analyze uncertainty in
    optimization results using Bayesian approaches.
    """
    
    def __init__(self, acquisition_function: str = "expected_improvement"):
        """
        Initialize uncertainty quantification.
        
        Args:
            acquisition_function: Type of acquisition function used
        """
        self.acquisition_function = acquisition_function
    
    def quantify_uncertainty(
        self,
        optimization_results: List[Dict[str, Any]],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Quantify uncertainty in optimization results.
        
        Args:
            optimization_results: List of optimization result dictionaries
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with uncertainty metrics
        """
        if not optimization_results:
            return {}
        
        # Extract performance values
        performances = [result.get("objective_value", 0.0) for result in optimization_results]
        performances = np.array(performances)
        
        # Extract parameter values
        all_params = {}
        for result in optimization_results:
            params = result.get("parameters", {})
            for param_name, param_value in params.items():
                if param_name not in all_params:
                    all_params[param_name] = []
                all_params[param_name].append(param_value)
        
        # Convert to arrays
        for param_name in all_params:
            all_params[param_name] = np.array(all_params[param_name])
        
        # Calculate uncertainty metrics
        uncertainty_metrics = {
            "performance_uncertainty": self._performance_uncertainty(performances, confidence_level),
            "parameter_uncertainty": self._parameter_uncertainty(all_params, confidence_level),
            "optimization_confidence": self._optimization_confidence(performances),
            "convergence_uncertainty": self._convergence_uncertainty(performances),
        }
        
        return uncertainty_metrics
    
    def _performance_uncertainty(
        self, 
        performances: np.ndarray, 
        confidence_level: float
    ) -> Dict[str, float]:
        """
        Calculate uncertainty in performance estimates.
        
        Args:
            performances: Array of performance values
            confidence_level: Confidence level
            
        Returns:
            Dictionary with performance uncertainty metrics
        """
        if len(performances) == 0:
            return {}
        
        # Basic statistics
        mean_perf = np.mean(performances)
        std_perf = np.std(performances, ddof=1) if len(performances) > 1 else 0.0
        
        # Credible interval
        alpha = 1 - confidence_level
        if len(performances) > 1:
            ci_lower = np.percentile(performances, alpha / 2 * 100)
            ci_upper = np.percentile(performances, (1 - alpha / 2) * 100)
        else:
            ci_lower = ci_upper = mean_perf
        
        # Coefficient of variation
        cv = std_perf / abs(mean_perf) if mean_perf != 0 else float('inf')
        
        return {
            "mean": mean_perf,
            "std": std_perf,
            "coefficient_of_variation": cv,
            "credible_interval_lower": ci_lower,
            "credible_interval_upper": ci_upper,
            "range": np.max(performances) - np.min(performances),
        }
    
    def _parameter_uncertainty(
        self, 
        all_params: Dict[str, np.ndarray], 
        confidence_level: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate uncertainty in parameter estimates.
        
        Args:
            all_params: Dictionary of parameter arrays
            confidence_level: Confidence level
            
        Returns:
            Dictionary with parameter uncertainty metrics
        """
        param_uncertainty = {}
        alpha = 1 - confidence_level
        
        for param_name, param_values in all_params.items():
            if len(param_values) == 0:
                continue
            
            mean_val = np.mean(param_values)
            std_val = np.std(param_values, ddof=1) if len(param_values) > 1 else 0.0
            
            # Credible interval
            if len(param_values) > 1:
                ci_lower = np.percentile(param_values, alpha / 2 * 100)
                ci_upper = np.percentile(param_values, (1 - alpha / 2) * 100)
            else:
                ci_lower = ci_upper = mean_val
            
            param_uncertainty[param_name] = {
                "mean": mean_val,
                "std": std_val,
                "credible_interval_lower": ci_lower,
                "credible_interval_upper": ci_upper,
                "stability": 1.0 / (1.0 + std_val / abs(mean_val)) if mean_val != 0 else 0.0,
            }
        
        return param_uncertainty
    
    def _optimization_confidence(self, performances: np.ndarray) -> float:
        """
        Calculate confidence in optimization result.
        
        Args:
            performances: Array of performance values
            
        Returns:
            Confidence score (0-1)
        """
        if len(performances) <= 1:
            return 0.5
        
        # Based on how much better the best result is compared to others
        best_perf = np.max(performances)
        other_perfs = performances[performances != best_perf]
        
        if len(other_perfs) == 0:
            return 1.0
        
        mean_others = np.mean(other_perfs)
        std_others = np.std(other_perfs, ddof=1)
        
        if std_others == 0:
            confidence = 1.0 if best_perf > mean_others else 0.5
        else:
            # Z-score of best performance relative to others
            z_score = (best_perf - mean_others) / std_others
            # Convert to probability using normal CDF
            confidence = 0.5 * (1 + np.tanh(z_score * 0.7978845608))
        
        return confidence
    
    def _convergence_uncertainty(self, performances: np.ndarray) -> Dict[str, float]:
        """
        Calculate uncertainty about optimization convergence.
        
        Args:
            performances: Array of performance values (in order)
            
        Returns:
            Dictionary with convergence uncertainty metrics
        """
        if len(performances) < 3:
            return {"converged": False, "convergence_confidence": 0.0}
        
        # Look at trend in best performance over time
        best_so_far = np.maximum.accumulate(performances)
        improvements = np.diff(best_so_far)
        
        # Recent improvements
        recent_window = min(10, len(improvements) // 2)
        if recent_window > 0:
            recent_improvements = improvements[-recent_window:]
            avg_recent_improvement = np.mean(recent_improvements)
        else:
            avg_recent_improvement = 0.0
        
        # Convergence indicators
        has_converged = avg_recent_improvement < 1e-6
        convergence_confidence = np.exp(-abs(avg_recent_improvement) * 1000)
        
        # Stability of recent results
        if len(performances) >= 5:
            recent_perfs = performances[-5:]
            stability = 1.0 / (1.0 + np.std(recent_perfs))
        else:
            stability = 0.5
        
        return {
            "converged": has_converged,
            "convergence_confidence": convergence_confidence,
            "recent_improvement_rate": avg_recent_improvement,
            "stability": stability,
        }


class PriorSensitivityAnalysis:
    """
    Analysis of sensitivity to prior assumptions in Bayesian methods.
    
    Provides methods to test how sensitive results are to different
    prior specifications.
    """
    
    def analyze_prior_sensitivity(
        self,
        data: List[float],
        prior_specifications: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity to different prior specifications.
        
        Args:
            data: Observed data
            prior_specifications: List of prior specification dictionaries
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        data_array = np.array(data)
        results = {}
        
        for i, prior_spec in enumerate(prior_specifications):
            prior_name = prior_spec.get("name", f"prior_{i}")
            
            # Calculate posterior under this prior
            posterior = self._calculate_posterior(data_array, prior_spec)
            
            results[prior_name] = {
                "posterior_mean": posterior["mean"],
                "posterior_std": posterior["std"],
                "credible_interval": posterior["credible_interval"],
                "bayes_factor": posterior.get("bayes_factor", 1.0),
            }
        
        # Calculate sensitivity metrics
        sensitivity_metrics = self._calculate_sensitivity_metrics(results)
        
        return {
            "prior_results": results,
            "sensitivity_metrics": sensitivity_metrics,
        }
    
    def _calculate_posterior(
        self, 
        data: np.ndarray, 
        prior_spec: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate posterior under specified prior.
        
        Args:
            data: Observed data
            prior_spec: Prior specification
            
        Returns:
            Dictionary with posterior statistics
        """
        n = len(data)
        
        if n == 0:
            return {"mean": 0.0, "std": 1.0, "credible_interval": (0.0, 0.0)}
        
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1) if n > 1 else 1.0
        
        prior_type = prior_spec.get("type", "uninformative")
        
        if prior_type == "normal":
            # Normal prior on mean
            prior_mean = prior_spec.get("mean", 0.0)
            prior_var = prior_spec.get("variance", 1.0)
            
            # Posterior for normal-normal model
            precision_prior = 1.0 / prior_var
            precision_data = n / sample_var if sample_var > 0 else n
            
            posterior_precision = precision_prior + precision_data
            posterior_var = 1.0 / posterior_precision
            posterior_mean = (
                precision_prior * prior_mean + precision_data * sample_mean
            ) / posterior_precision
            posterior_std = np.sqrt(posterior_var)
            
        elif prior_type == "uninformative":
            # Uninformative (flat) prior
            posterior_mean = sample_mean
            posterior_std = np.sqrt(sample_var / n) if sample_var > 0 else 0.0
            
        else:
            # Default to uninformative
            posterior_mean = sample_mean
            posterior_std = np.sqrt(sample_var / n) if sample_var > 0 else 0.0
        
        # Credible interval (assuming normality)
        z_critical = 1.96  # 95% interval
        ci_lower = posterior_mean - z_critical * posterior_std
        ci_upper = posterior_mean + z_critical * posterior_std
        
        return {
            "mean": posterior_mean,
            "std": posterior_std,
            "credible_interval": (ci_lower, ci_upper),
        }
    
    def _calculate_sensitivity_metrics(
        self, 
        prior_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate metrics for prior sensitivity.
        
        Args:
            prior_results: Results for different priors
            
        Returns:
            Dictionary with sensitivity metrics
        """
        if len(prior_results) < 2:
            return {"max_difference": 0.0, "coefficient_of_variation": 0.0}
        
        # Extract posterior means and stds
        posterior_means = [result["posterior_mean"] for result in prior_results.values()]
        posterior_stds = [result["posterior_std"] for result in prior_results.values()]
        
        # Maximum difference in posterior means
        max_diff_mean = np.max(posterior_means) - np.min(posterior_means)
        
        # Coefficient of variation across priors
        mean_of_means = np.mean(posterior_means)
        std_of_means = np.std(posterior_means, ddof=1) if len(posterior_means) > 1 else 0.0
        
        if mean_of_means != 0:
            cv_means = std_of_means / abs(mean_of_means)
        else:
            cv_means = float('inf') if std_of_means > 0 else 0.0
        
        # Average overlap of credible intervals
        overlaps = []
        prior_names = list(prior_results.keys())
        
        for i in range(len(prior_names)):
            for j in range(i + 1, len(prior_names)):
                ci1 = prior_results[prior_names[i]]["credible_interval"]
                ci2 = prior_results[prior_names[j]]["credible_interval"]
                
                # Calculate overlap
                overlap_start = max(ci1[0], ci2[0])
                overlap_end = min(ci1[1], ci2[1])
                overlap_length = max(0, overlap_end - overlap_start)
                
                # Normalize by average interval length
                avg_length = ((ci1[1] - ci1[0]) + (ci2[1] - ci2[0])) / 2
                if avg_length > 0:
                    normalized_overlap = overlap_length / avg_length
                else:
                    normalized_overlap = 1.0 if overlap_length == 0 else 0.0
                
                overlaps.append(normalized_overlap)
        
        avg_overlap = np.mean(overlaps) if overlaps else 1.0
        
        return {
            "max_difference_mean": max_diff_mean,
            "coefficient_of_variation_mean": cv_means,
            "average_credible_interval_overlap": avg_overlap,
            "sensitivity_score": 1.0 - avg_overlap,  # Higher = more sensitive
        }