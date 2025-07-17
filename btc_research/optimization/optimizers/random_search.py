"""
Random Search optimization for efficient baseline parameter exploration.

This module implements random search with various sampling strategies and
early stopping criteria, providing an efficient baseline for comparison
with more sophisticated optimization algorithms.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from btc_research.optimization.base import BaseOptimizer, ObjectiveFunction
from btc_research.optimization.types import (
    OptimizationMethod,
    OptimizationMetric,
    OptimizationResult,
    ParameterSpec,
    ParameterType,
)

__all__ = ["RandomSearchOptimizer", "SamplingStrategy"]


class SamplingStrategy:
    """Enumeration of supported sampling strategies."""
    
    UNIFORM = "uniform"
    LATIN_HYPERCUBE = "latin_hypercube"
    SOBOL = "sobol"
    HALTON = "halton"
    LOG_UNIFORM = "log_uniform"


class RandomSearchOptimizer(BaseOptimizer):
    """
    Random Search optimization implementation.
    
    This optimizer randomly samples parameter combinations from the search space,
    providing an efficient baseline that often outperforms more complex algorithms
    when the budget is limited or the parameter space is high-dimensional.
    """
    
    def __init__(
        self,
        parameter_specs: List[ParameterSpec],
        objective_function: ObjectiveFunction,
        metric: OptimizationMetric,
        maximize: bool = True,
        random_seed: Optional[int] = None,
        sampling_strategy: str = SamplingStrategy.UNIFORM,
        early_stopping_patience: int = 20,
        improvement_threshold: float = 1e-6,
        statistical_testing: bool = True,
        confidence_level: float = 0.95,
        min_samples_for_stats: int = 10,
    ):
        """
        Initialize Random Search optimizer.
        
        Args:
            parameter_specs: List of parameter specifications defining search space
            objective_function: Function to optimize (takes params dict, returns score)
            metric: Primary optimization metric
            maximize: Whether to maximize (True) or minimize (False) the objective
            random_seed: Random seed for reproducibility
            sampling_strategy: Strategy for parameter sampling
            early_stopping_patience: Stop if no improvement for this many evaluations
            improvement_threshold: Minimum improvement to reset early stopping counter
            statistical_testing: Whether to perform statistical significance testing
            confidence_level: Confidence level for statistical tests
            min_samples_for_stats: Minimum samples required for statistical testing
        """
        super().__init__(parameter_specs, objective_function, metric, maximize, random_seed)
        
        self.sampling_strategy = sampling_strategy
        self.early_stopping_patience = early_stopping_patience
        self.improvement_threshold = improvement_threshold
        self.statistical_testing = statistical_testing
        self.confidence_level = confidence_level
        self.min_samples_for_stats = min_samples_for_stats
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_seed)
        
        # Parameter space setup
        self._setup_parameter_space()
        
        # Sampling state
        self._sample_count = 0
        self._samples_evaluated = []
        self._scores_observed = []
        
        # Convergence tracking
        self._last_improvement = 0
        self._no_improvement_count = 0
        self._best_score = float('-inf') if maximize else float('inf')
        
        # Statistical analysis
        self._running_mean = 0.0
        self._running_var = 0.0
        self._convergence_history = []
        
        # Quasi-random sequence state (for Sobol, Halton)
        self._quasi_random_state = {}
        
        # Validation
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")
        
        if min_samples_for_stats < 2:
            raise ValueError("Minimum samples for statistics must be at least 2")
    
    def _setup_parameter_space(self) -> None:
        """Setup parameter space bounds and transformations."""
        self.bounds = []
        self.param_names = []
        self.param_types = []
        self.categorical_mappings = {}
        
        for spec in self.parameter_specs:
            self.param_names.append(spec.name)
            self.param_types.append(spec.param_type)
            
            if spec.param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
                self.bounds.append((spec.low, spec.high))
            elif spec.param_type == ParameterType.CATEGORICAL:
                # Map categorical values to integers
                self.categorical_mappings[spec.name] = {
                    val: idx for idx, val in enumerate(spec.choices)
                }
                self.bounds.append((0, len(spec.choices) - 1))
            elif spec.param_type == ParameterType.BOOLEAN:
                self.bounds.append((0, 1))
        
        self.bounds = np.array(self.bounds)
        self.n_dims = len(self.bounds)
        
        # Initialize quasi-random sequences
        if self.sampling_strategy in (SamplingStrategy.SOBOL, SamplingStrategy.HALTON):
            self._initialize_quasi_random()
    
    def _initialize_quasi_random(self) -> None:
        """Initialize quasi-random sequence generators."""
        if self.sampling_strategy == SamplingStrategy.SOBOL:
            try:
                from scipy.stats import qmc
                self._quasi_random_state['sobol'] = qmc.Sobol(
                    d=self.n_dims,
                    scramble=True,
                    seed=self.random_seed
                )
            except ImportError:
                warnings.warn("SciPy QMC not available. Falling back to uniform sampling.")
                self.sampling_strategy = SamplingStrategy.UNIFORM
        
        elif self.sampling_strategy == SamplingStrategy.HALTON:
            try:
                from scipy.stats import qmc
                self._quasi_random_state['halton'] = qmc.Halton(
                    d=self.n_dims,
                    scramble=True,
                    seed=self.random_seed
                )
            except ImportError:
                warnings.warn("SciPy QMC not available. Falling back to uniform sampling.")
                self.sampling_strategy = SamplingStrategy.UNIFORM
    
    def _encode_parameters(self, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Encode parameter dictionary to numerical vector.
        
        Args:
            parameters: Parameter dictionary
            
        Returns:
            Numerical vector representation
        """
        x = np.zeros(self.n_dims)
        
        for i, (spec, param_name) in enumerate(zip(self.parameter_specs, self.param_names)):
            value = parameters[param_name]
            
            if spec.param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
                x[i] = value
            elif spec.param_type == ParameterType.CATEGORICAL:
                x[i] = self.categorical_mappings[param_name][value]
            elif spec.param_type == ParameterType.BOOLEAN:
                x[i] = 1 if value else 0
        
        return x
    
    def _decode_parameters(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Decode numerical vector to parameter dictionary.
        
        Args:
            x: Numerical vector
            
        Returns:
            Parameter dictionary
        """
        parameters = {}
        
        for i, (spec, param_name) in enumerate(zip(self.parameter_specs, self.param_names)):
            value = x[i]
            
            if spec.param_type == ParameterType.INTEGER:
                parameters[param_name] = int(np.round(np.clip(value, spec.low, spec.high)))
            elif spec.param_type == ParameterType.FLOAT:
                parameters[param_name] = float(np.clip(value, spec.low, spec.high))
            elif spec.param_type == ParameterType.CATEGORICAL:
                idx = int(np.round(np.clip(value, 0, len(spec.choices) - 1)))
                parameters[param_name] = spec.choices[idx]
            elif spec.param_type == ParameterType.BOOLEAN:
                parameters[param_name] = bool(np.round(value))
        
        return parameters
    
    def _uniform_sample(self) -> np.ndarray:
        """Generate uniform random sample within parameter bounds."""
        x = np.zeros(self.n_dims)
        
        for i, (low, high) in enumerate(self.bounds):
            # Handle log-scale parameters
            spec = self.parameter_specs[i]
            if hasattr(spec, 'log_scale') and spec.log_scale:
                # Log-uniform sampling
                log_low = np.log(max(low, 1e-10))
                log_high = np.log(max(high, 1e-10))
                x[i] = np.exp(self.rng.uniform(log_low, log_high))
            else:
                # Standard uniform sampling
                x[i] = self.rng.uniform(low, high)
        
        return x
    
    def _latin_hypercube_sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=self.n_dims, seed=self.random_seed)
            samples = sampler.random(n=n_samples)
            
            # Scale to bounds
            scaled_samples = np.zeros_like(samples)
            for i, (low, high) in enumerate(self.bounds):
                spec = self.parameter_specs[i]
                if hasattr(spec, 'log_scale') and spec.log_scale:
                    # Log-uniform scaling
                    log_low = np.log(max(low, 1e-10))
                    log_high = np.log(max(high, 1e-10))
                    scaled_samples[:, i] = np.exp(
                        samples[:, i] * (log_high - log_low) + log_low
                    )
                else:
                    # Linear scaling
                    scaled_samples[:, i] = samples[:, i] * (high - low) + low
            
            return scaled_samples[0] if n_samples == 1 else scaled_samples
            
        except ImportError:
            warnings.warn("SciPy QMC not available. Falling back to uniform sampling.")
            return self._uniform_sample()
    
    def _sobol_sample(self) -> np.ndarray:
        """Generate Sobol sequence sample."""
        if 'sobol' not in self._quasi_random_state:
            return self._uniform_sample()
        
        try:
            sampler = self._quasi_random_state['sobol']
            sample = sampler.random(1)[0]
            
            # Scale to bounds
            scaled_sample = np.zeros(self.n_dims)
            for i, (low, high) in enumerate(self.bounds):
                spec = self.parameter_specs[i]
                if hasattr(spec, 'log_scale') and spec.log_scale:
                    # Log-uniform scaling
                    log_low = np.log(max(low, 1e-10))
                    log_high = np.log(max(high, 1e-10))
                    scaled_sample[i] = np.exp(
                        sample[i] * (log_high - log_low) + log_low
                    )
                else:
                    # Linear scaling
                    scaled_sample[i] = sample[i] * (high - low) + low
            
            return scaled_sample
            
        except Exception:
            return self._uniform_sample()
    
    def _halton_sample(self) -> np.ndarray:
        """Generate Halton sequence sample."""
        if 'halton' not in self._quasi_random_state:
            return self._uniform_sample()
        
        try:
            sampler = self._quasi_random_state['halton']
            sample = sampler.random(1)[0]
            
            # Scale to bounds
            scaled_sample = np.zeros(self.n_dims)
            for i, (low, high) in enumerate(self.bounds):
                spec = self.parameter_specs[i]
                if hasattr(spec, 'log_scale') and spec.log_scale:
                    # Log-uniform scaling
                    log_low = np.log(max(low, 1e-10))
                    log_high = np.log(max(high, 1e-10))
                    scaled_sample[i] = np.exp(
                        sample[i] * (log_high - log_low) + log_low
                    )
                else:
                    # Linear scaling
                    scaled_sample[i] = sample[i] * (high - low) + low
            
            return scaled_sample
            
        except Exception:
            return self._uniform_sample()
    
    def _generate_sample(self) -> np.ndarray:
        """
        Generate sample using configured sampling strategy.
        
        Returns:
            Parameter vector sample
        """
        if self.sampling_strategy == SamplingStrategy.UNIFORM:
            return self._uniform_sample()
        elif self.sampling_strategy == SamplingStrategy.LATIN_HYPERCUBE:
            return self._latin_hypercube_sample()
        elif self.sampling_strategy == SamplingStrategy.SOBOL:
            return self._sobol_sample()
        elif self.sampling_strategy == SamplingStrategy.HALTON:
            return self._halton_sample()
        elif self.sampling_strategy == SamplingStrategy.LOG_UNIFORM:
            # Force log-uniform for all parameters
            x = np.zeros(self.n_dims)
            for i, (low, high) in enumerate(self.bounds):
                log_low = np.log(max(low, 1e-10))
                log_high = np.log(max(high, 1e-10))
                x[i] = np.exp(self.rng.uniform(log_low, log_high))
            return x
        else:
            # Default to uniform
            return self._uniform_sample()
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """
        Suggest next parameter combination to evaluate.
        
        Returns:
            Dictionary of parameter names and suggested values
        """
        x_next = self._generate_sample()
        return self._decode_parameters(x_next)
    
    def _update_running_statistics(self, new_score: float) -> None:
        """
        Update running statistics for convergence analysis.
        
        Args:
            new_score: New objective function value
        """
        n = len(self._scores_observed)
        
        if n == 0:
            self._running_mean = new_score
            self._running_var = 0.0
        else:
            # Welford's online algorithm for running mean and variance
            delta = new_score - self._running_mean
            self._running_mean += delta / (n + 1)
            delta2 = new_score - self._running_mean
            self._running_var += delta * delta2
    
    def _calculate_convergence_metrics(self) -> Dict[str, float]:
        """
        Calculate convergence metrics.
        
        Returns:
            Dictionary with convergence statistics
        """
        n = len(self._scores_observed)
        
        if n < 2:
            return {
                'mean': self._running_mean,
                'std': 0.0,
                'cv': 0.0,
                'improvement_rate': 0.0,
                'stagnation_ratio': 0.0,
            }
        
        # Standard deviation
        std = np.sqrt(self._running_var / (n - 1)) if n > 1 else 0.0
        
        # Coefficient of variation
        cv = std / abs(self._running_mean) if self._running_mean != 0 else 0.0
        
        # Recent improvement rate
        window_size = min(10, n // 2)
        if window_size >= 2:
            recent_scores = self._scores_observed[-window_size:]
            if self.maximize:
                improvement_rate = (max(recent_scores) - recent_scores[0]) / window_size
            else:
                improvement_rate = (recent_scores[0] - min(recent_scores)) / window_size
        else:
            improvement_rate = 0.0
        
        # Stagnation ratio (fraction of recent evaluations without improvement)
        stagnation_ratio = self._no_improvement_count / max(1, self.early_stopping_patience)
        
        return {
            'mean': self._running_mean,
            'std': std,
            'cv': cv,
            'improvement_rate': improvement_rate,
            'stagnation_ratio': stagnation_ratio,
        }
    
    def _statistical_significance_test(self, recent_window: int = 20) -> Dict[str, float]:
        """
        Perform statistical significance test on recent improvements.
        
        Args:
            recent_window: Number of recent samples to analyze
            
        Returns:
            Dictionary with statistical test results
        """
        n = len(self._scores_observed)
        
        if n < self.min_samples_for_stats:
            return {'p_value': 1.0, 'significant': False}
        
        # Split into two halves for comparison
        split_point = max(self.min_samples_for_stats // 2, n - recent_window)
        early_scores = self._scores_observed[:split_point]
        recent_scores = self._scores_observed[split_point:]
        
        if len(early_scores) < 2 or len(recent_scores) < 2:
            return {'p_value': 1.0, 'significant': False}
        
        try:
            # Welch's t-test for unequal variances
            early_mean = np.mean(early_scores)
            recent_mean = np.mean(recent_scores)
            early_var = np.var(early_scores, ddof=1)
            recent_var = np.var(recent_scores, ddof=1)
            
            # Test statistic
            pooled_se = np.sqrt(early_var / len(early_scores) + recent_var / len(recent_scores))
            
            if pooled_se == 0:
                return {'p_value': 1.0, 'significant': False}
            
            t_stat = (recent_mean - early_mean) / pooled_se
            
            # Degrees of freedom (Welch-Satterthwaite equation)
            num = (early_var / len(early_scores) + recent_var / len(recent_scores)) ** 2
            denom = (
                (early_var / len(early_scores)) ** 2 / (len(early_scores) - 1) +
                (recent_var / len(recent_scores)) ** 2 / (len(recent_scores) - 1)
            )
            df = num / denom if denom > 0 else 1
            
            # Calculate p-value (two-tailed test)
            from scipy.stats import t
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
            
            significant = p_value < (1 - self.confidence_level)
            
            return {
                'p_value': p_value,
                'significant': significant,
                't_statistic': t_stat,
                'degrees_of_freedom': df,
                'early_mean': early_mean,
                'recent_mean': recent_mean,
            }
            
        except Exception:
            return {'p_value': 1.0, 'significant': False}
    
    def update_with_result(self, parameters: Dict[str, Any], score: float) -> None:
        """
        Update optimizer state with evaluation result.
        
        Args:
            parameters: Parameter values that were evaluated
            score: Objective function result
        """
        super().update_with_result(parameters, score)
        
        # Store evaluation
        x = self._encode_parameters(parameters)
        self._samples_evaluated.append(x)
        self._scores_observed.append(score)
        self._sample_count += 1
        
        # Update running statistics
        self._update_running_statistics(score)
        
        # Check for improvement
        improved = False
        if self.maximize:
            if score > self._best_score + self.improvement_threshold:
                self._best_score = score
                self._last_improvement = self._sample_count
                self._no_improvement_count = 0
                improved = True
            else:
                self._no_improvement_count += 1
        else:
            if score < self._best_score - self.improvement_threshold:
                self._best_score = score
                self._last_improvement = self._sample_count
                self._no_improvement_count = 0
                improved = True
            else:
                self._no_improvement_count += 1
        
        # Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics()
        self._convergence_history.append(convergence_metrics)
    
    def optimize(
        self,
        max_iterations: int = 100,
        timeout_seconds: Optional[float] = None,
        convergence_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Run random search optimization.
        
        Args:
            max_iterations: Maximum number of iterations/evaluations
            timeout_seconds: Maximum time to run optimization
            convergence_threshold: Stop if improvement falls below this threshold
            **kwargs: Additional parameters
            
        Returns:
            Best optimization result found
        """
        start_time = datetime.now()
        
        # Reset state
        self._sample_count = 0
        self._samples_evaluated.clear()
        self._scores_observed.clear()
        self._last_improvement = 0
        self._no_improvement_count = 0
        self._best_score = float('-inf') if self.maximize else float('inf')
        self._running_mean = 0.0
        self._running_var = 0.0
        self._convergence_history.clear()
        
        best_score = self._best_score
        best_params = None
        
        # Optimization loop
        for iteration in range(max_iterations):
            # Check timeout
            if timeout_seconds:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    print(f"Random search stopped due to timeout after {iteration} evaluations")
                    break
            
            # Check early stopping
            if self._no_improvement_count >= self.early_stopping_patience:
                print(f"Random search stopped due to early stopping after {iteration} evaluations")
                break
            
            # Check convergence
            if convergence_threshold is not None and len(self._scores_observed) >= 2:
                recent_improvement = abs(self._scores_observed[-1] - self._scores_observed[-2])
                if recent_improvement < convergence_threshold:
                    print(f"Random search converged after {iteration} evaluations")
                    break
            
            try:
                # Get next parameter combination
                params = self.suggest_parameters()
                
                # Evaluate parameters
                score = self.evaluate_parameters(params)
                
                # Update optimizer state
                self.update_with_result(params, score)
                
                # Update best result
                if self.maximize and score > best_score:
                    best_score = score
                    best_params = params.copy()
                elif not self.maximize and score < best_score:
                    best_score = score
                    best_params = params.copy()
                
            except Exception as e:
                print(f"Warning: Evaluation failed for parameters {params}: {e}")
                continue
        
        end_time = datetime.now()
        
        if best_params is None:
            raise ValueError("No valid parameter combinations found")
        
        # Statistical significance testing
        statistical_results = {}
        if self.statistical_testing and len(self._scores_observed) >= self.min_samples_for_stats:
            statistical_results = self._statistical_significance_test()
        
        # Final convergence metrics
        final_convergence = self._calculate_convergence_metrics()
        
        # Create optimization result
        result = OptimizationResult(
            parameters=best_params,
            metrics={"objective_value": best_score},
            objective_value=best_score,
            in_sample_metrics={"objective_value": best_score},
            method=OptimizationMethod.RANDOM_SEARCH,
            metric=self.metric,
            start_time=start_time,
            end_time=end_time,
            iterations=self._sample_count,
            convergence_achieved=self._no_improvement_count < self.early_stopping_patience,
            diagnostics={
                "sampling_strategy": self.sampling_strategy,
                "early_stopping_patience": self.early_stopping_patience,
                "improvement_threshold": self.improvement_threshold,
                "last_improvement_iteration": self._last_improvement,
                "no_improvement_count": self._no_improvement_count,
                "convergence_metrics": final_convergence,
                "statistical_results": statistical_results,
                "convergence_history": self._convergence_history,
                "all_scores": self._scores_observed,
            }
        )
        
        self._best_result = result
        return result
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get current search statistics.
        
        Returns:
            Dictionary with search statistics
        """
        if not self._scores_observed:
            return {"samples_evaluated": 0}
        
        convergence_metrics = self._calculate_convergence_metrics()
        
        return {
            "samples_evaluated": self._sample_count,
            "best_score": self._best_score,
            "last_improvement": self._last_improvement,
            "no_improvement_count": self._no_improvement_count,
            "sampling_strategy": self.sampling_strategy,
            **convergence_metrics,
        }
    
    def get_sample_distribution(self) -> Dict[str, np.ndarray]:
        """
        Get distribution of evaluated samples.
        
        Returns:
            Dictionary with sample distributions per parameter
        """
        if not self._samples_evaluated:
            return {}
        
        samples = np.array(self._samples_evaluated)
        
        distributions = {}
        for i, param_name in enumerate(self.param_names):
            distributions[param_name] = samples[:, i]
        
        return distributions
    
    def estimate_parameter_importance(self) -> Dict[str, float]:
        """
        Estimate parameter importance using correlation with objective.
        
        Returns:
            Dictionary with importance scores per parameter
        """
        if len(self._samples_evaluated) < 10:
            return {name: 0.0 for name in self.param_names}
        
        samples = np.array(self._samples_evaluated)
        scores = np.array(self._scores_observed)
        
        importance = {}
        for i, param_name in enumerate(self.param_names):
            try:
                correlation = np.corrcoef(samples[:, i], scores)[0, 1]
                # Use absolute correlation as importance
                importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            except Exception:
                importance[param_name] = 0.0
        
        return importance