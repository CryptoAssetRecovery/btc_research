"""
Bayesian optimization using Gaussian Process regression for surrogate modeling.

This module implements Bayesian optimization with various acquisition functions
for efficient exploration of parameter spaces, particularly suited for expensive
objective functions like trading strategy optimization.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
    from scipy.optimize import minimize
    from scipy.stats import norm
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. BayesianOptimizer will use simplified implementation.")

from btc_research.optimization.base import BaseOptimizer, ObjectiveFunction
from btc_research.optimization.types import (
    OptimizationMethod,
    OptimizationMetric,
    OptimizationResult,
    ParameterSpec,
    ParameterType,
)

__all__ = ["BayesianOptimizer", "AcquisitionFunction"]


class AcquisitionFunction:
    """Enumeration of supported acquisition functions."""
    
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    ENTROPY_SEARCH = "entropy_search"


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimization implementation using Gaussian Process regression.
    
    This optimizer uses a Gaussian Process as a surrogate model to predict
    the objective function and employs acquisition functions to balance
    exploration and exploitation in parameter space.
    """
    
    def __init__(
        self,
        parameter_specs: List[ParameterSpec],
        objective_function: ObjectiveFunction,
        metric: OptimizationMetric,
        maximize: bool = True,
        random_seed: Optional[int] = None,
        acquisition_function: str = AcquisitionFunction.EXPECTED_IMPROVEMENT,
        acquisition_kappa: float = 2.576,  # 99% confidence for UCB
        acquisition_xi: float = 0.01,  # exploration parameter for EI
        n_initial_points: int = 5,
        kernel_type: str = "matern",
        gp_noise_alpha: float = 1e-6,
        normalize_y: bool = True,
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            parameter_specs: List of parameter specifications defining search space
            objective_function: Function to optimize (takes params dict, returns score)
            metric: Primary optimization metric
            maximize: Whether to maximize (True) or minimize (False) the objective
            random_seed: Random seed for reproducibility
            acquisition_function: Acquisition function to use for next point selection
            acquisition_kappa: Confidence multiplier for UCB acquisition
            acquisition_xi: Exploration parameter for EI acquisition
            n_initial_points: Number of random initial points to evaluate
            kernel_type: Gaussian Process kernel type ('matern', 'rbf')
            gp_noise_alpha: Noise regularization parameter for GP
            normalize_y: Whether to normalize target values for GP
        """
        super().__init__(parameter_specs, objective_function, metric, maximize, random_seed)
        
        self.acquisition_function = acquisition_function
        self.acquisition_kappa = acquisition_kappa
        self.acquisition_xi = acquisition_xi
        self.n_initial_points = n_initial_points
        self.kernel_type = kernel_type
        self.gp_noise_alpha = gp_noise_alpha
        self.normalize_y = normalize_y
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_seed)
        
        # Parameter space bounds and transformations
        self._setup_parameter_space()
        
        # Gaussian Process model
        self.gp_model = None
        self._X_observed = []  # Observed parameter vectors
        self._y_observed = []  # Observed objective values
        
        # Convergence tracking
        self._last_improvement = 0
        self._no_improvement_count = 0
        
        # Validation
        if not SKLEARN_AVAILABLE:
            self._use_simplified_implementation = True
        else:
            self._use_simplified_implementation = False
            self._initialize_gp_model()
    
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
    
    def _initialize_gp_model(self) -> None:
        """Initialize Gaussian Process model."""
        if not SKLEARN_AVAILABLE:
            return
        
        # Choose kernel
        if self.kernel_type == "matern":
            kernel = Matern(
                length_scale=[1.0] * self.n_dims,
                length_scale_bounds=[(1e-3, 1e3)] * self.n_dims,
                nu=2.5
            )
        elif self.kernel_type == "rbf":
            kernel = RBF(
                length_scale=[1.0] * self.n_dims,
                length_scale_bounds=[(1e-3, 1e3)] * self.n_dims
            )
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
        
        # Add noise kernel for robustness
        kernel = kernel + WhiteKernel(
            noise_level=self.gp_noise_alpha,
            noise_level_bounds=(1e-10, 1e2)  # Wider bounds to prevent convergence warnings
        )
        
        # Initialize GP
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.gp_noise_alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=3,
            random_state=self.random_seed,
        )
    
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
    
    def _generate_random_sample(self) -> np.ndarray:
        """Generate random sample within parameter bounds."""
        x = np.zeros(self.n_dims)
        
        for i, (low, high) in enumerate(self.bounds):
            x[i] = self.rng.uniform(low, high)
        
        return x
    
    def _expected_improvement(self, x: np.ndarray) -> float:
        """
        Calculate Expected Improvement acquisition function.
        
        Args:
            x: Parameter vector
            
        Returns:
            Expected improvement value
        """
        if self._use_simplified_implementation or len(self._y_observed) == 0:
            return self.rng.random()
        
        x = x.reshape(1, -1)
        
        try:
            mu, sigma = self.gp_model.predict(x, return_std=True)
            mu = mu[0]
            sigma = sigma[0]
            
            if sigma < 1e-9:  # Avoid division by zero
                return 0.0
            
            # Current best value
            f_best = max(self._y_observed) if self.maximize else min(self._y_observed)
            
            # Calculate improvement
            if self.maximize:
                improvement = mu - f_best - self.acquisition_xi
            else:
                improvement = f_best - mu - self.acquisition_xi
            
            # Expected improvement
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            
            return ei
            
        except Exception:
            # Fallback to random value
            return self.rng.random()
    
    def _upper_confidence_bound(self, x: np.ndarray) -> float:
        """
        Calculate Upper Confidence Bound acquisition function.
        
        Args:
            x: Parameter vector
            
        Returns:
            UCB value
        """
        if self._use_simplified_implementation or len(self._y_observed) == 0:
            return self.rng.random()
        
        x = x.reshape(1, -1)
        
        try:
            mu, sigma = self.gp_model.predict(x, return_std=True)
            mu = mu[0]
            sigma = sigma[0]
            
            if self.maximize:
                ucb = mu + self.acquisition_kappa * sigma
            else:
                ucb = mu - self.acquisition_kappa * sigma
            
            return ucb
            
        except Exception:
            # Fallback to random value
            return self.rng.random()
    
    def _probability_of_improvement(self, x: np.ndarray) -> float:
        """
        Calculate Probability of Improvement acquisition function.
        
        Args:
            x: Parameter vector
            
        Returns:
            Probability of improvement value
        """
        if self._use_simplified_implementation or len(self._y_observed) == 0:
            return self.rng.random()
        
        x = x.reshape(1, -1)
        
        try:
            mu, sigma = self.gp_model.predict(x, return_std=True)
            mu = mu[0]
            sigma = sigma[0]
            
            if sigma < 1e-9:  # Avoid division by zero
                return 0.0
            
            # Current best value
            f_best = max(self._y_observed) if self.maximize else min(self._y_observed)
            
            # Calculate improvement probability
            if self.maximize:
                improvement = mu - f_best - self.acquisition_xi
            else:
                improvement = f_best - mu - self.acquisition_xi
            
            z = improvement / sigma
            pi = norm.cdf(z)
            
            return pi
            
        except Exception:
            # Fallback to random value
            return self.rng.random()
    
    def _acquisition_function(self, x: np.ndarray) -> float:
        """
        Evaluate acquisition function at point x.
        
        Args:
            x: Parameter vector
            
        Returns:
            Acquisition function value
        """
        if self.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT:
            return self._expected_improvement(x)
        elif self.acquisition_function == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
            return self._upper_confidence_bound(x)
        elif self.acquisition_function == AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT:
            return self._probability_of_improvement(x)
        else:
            # Default to expected improvement
            return self._expected_improvement(x)
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """
        Suggest next parameter combination to evaluate.
        
        Returns:
            Dictionary of parameter names and suggested values
        """
        # Initial random exploration
        if len(self._X_observed) < self.n_initial_points:
            x_next = self._generate_random_sample()
        else:
            # Bayesian optimization
            x_next = self._optimize_acquisition()
        
        return self._decode_parameters(x_next)
    
    def _optimize_acquisition(self) -> np.ndarray:
        """
        Optimize acquisition function to find next evaluation point.
        
        Returns:
            Next parameter vector to evaluate
        """
        if self._use_simplified_implementation:
            # Fallback to random sampling
            return self._generate_random_sample()
        
        # Multiple random starts for global optimization
        n_restarts = 10
        best_x = None
        best_acq = float('-inf')
        
        for _ in range(n_restarts):
            # Random starting point
            x0 = self._generate_random_sample()
            
            # Minimize negative acquisition function
            def objective(x):
                return -self._acquisition_function(x)
            
            try:
                result = minimize(
                    objective,
                    x0,
                    bounds=self.bounds,
                    method='L-BFGS-B',
                )
                
                if result.success and -result.fun > best_acq:
                    best_acq = -result.fun
                    best_x = result.x
                    
            except Exception:
                continue
        
        # Fallback to random if optimization failed
        if best_x is None:
            best_x = self._generate_random_sample()
        
        return best_x
    
    def update_with_result(self, parameters: Dict[str, Any], score: float) -> None:
        """
        Update optimizer state with evaluation result.
        
        Args:
            parameters: Parameter values that were evaluated
            score: Objective function result
        """
        super().update_with_result(parameters, score)
        
        # Store observation
        x = self._encode_parameters(parameters)
        self._X_observed.append(x)
        self._y_observed.append(score)
        
        # Update GP model
        if not self._use_simplified_implementation and len(self._X_observed) >= 2:
            try:
                X = np.array(self._X_observed)
                y = np.array(self._y_observed)
                self.gp_model.fit(X, y)
            except Exception as e:
                warnings.warn(f"GP model update failed: {e}")
        
        # Track convergence
        if self._best_result is None or (
            self.maximize and score > self._best_result.objective_value
        ) or (
            not self.maximize and score < self._best_result.objective_value
        ):
            self._last_improvement = self._iteration_count
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
    
    def optimize(
        self,
        max_iterations: int = 100,
        timeout_seconds: Optional[float] = None,
        convergence_threshold: Optional[float] = None,
        early_stopping_patience: int = 20,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            max_iterations: Maximum number of iterations/evaluations
            timeout_seconds: Maximum time to run optimization
            convergence_threshold: Stop if improvement falls below this threshold
            early_stopping_patience: Stop if no improvement for this many iterations
            **kwargs: Additional parameters
            
        Returns:
            Best optimization result found
        """
        start_time = datetime.now()
        
        # Reset state
        self._X_observed.clear()
        self._y_observed.clear()
        self._all_results.clear()
        self._last_improvement = 0
        self._no_improvement_count = 0
        
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None
        
        # Optimization loop
        for iteration in range(max_iterations):
            # Check timeout
            if timeout_seconds:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    print(f"Bayesian optimization stopped due to timeout after {iteration} evaluations")
                    break
            
            # Check early stopping
            if self._no_improvement_count >= early_stopping_patience:
                print(f"Bayesian optimization stopped due to early stopping after {iteration} evaluations")
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
                
                # Check convergence
                if convergence_threshold is not None:
                    if len(self._y_observed) >= 2:
                        recent_improvement = abs(self._y_observed[-1] - self._y_observed[-2])
                        if recent_improvement < convergence_threshold:
                            print(f"Bayesian optimization converged after {iteration + 1} evaluations")
                            break
                
            except Exception as e:
                print(f"Warning: Evaluation failed for parameters {params}: {e}")
                continue
        
        end_time = datetime.now()
        
        if best_params is None:
            raise ValueError("No valid parameter combinations found")
        
        # Create optimization result
        result = OptimizationResult(
            parameters=best_params,
            metrics={"objective_value": best_score},
            objective_value=best_score,
            in_sample_metrics={"objective_value": best_score},
            method=OptimizationMethod.BAYESIAN,
            metric=self.metric,
            start_time=start_time,
            end_time=end_time,
            iterations=len(self._X_observed),
            convergence_achieved=self._no_improvement_count < early_stopping_patience,
            diagnostics={
                "acquisition_function": self.acquisition_function,
                "n_initial_points": self.n_initial_points,
                "early_stopping_patience": early_stopping_patience,
                "last_improvement_iteration": self._last_improvement,
                "no_improvement_count": self._no_improvement_count,
                "gp_kernel": self.kernel_type,
                "use_simplified": self._use_simplified_implementation,
            }
        )
        
        self._best_result = result
        return result
    
    def get_acquisition_surface(
        self,
        param1: str,
        param2: str,
        resolution: int = 50,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate acquisition function surface for visualization.
        
        Args:
            param1: First parameter name
            param2: Second parameter name
            resolution: Grid resolution for each dimension
            fixed_params: Fixed values for other parameters
            
        Returns:
            Tuple of (X, Y, Z) arrays for surface plotting
        """
        if param1 not in self.param_names or param2 not in self.param_names:
            raise ValueError("Parameter names must be in parameter specifications")
        
        if len(self._X_observed) < self.n_initial_points:
            raise ValueError("Not enough observations to generate acquisition surface")
        
        # Get parameter indices
        idx1 = self.param_names.index(param1)
        idx2 = self.param_names.index(param2)
        
        # Get bounds
        bounds1 = self.bounds[idx1]
        bounds2 = self.bounds[idx2]
        
        # Create grid
        x1 = np.linspace(bounds1[0], bounds1[1], resolution)
        x2 = np.linspace(bounds2[0], bounds2[1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Fixed parameter values
        if fixed_params is None:
            fixed_params = {}
        
        # Generate acquisition values
        Z = np.zeros_like(X1)
        
        for i in range(resolution):
            for j in range(resolution):
                # Create parameter vector
                x = np.zeros(self.n_dims)
                
                # Set grid values
                x[idx1] = X1[i, j]
                x[idx2] = X2[i, j]
                
                # Set fixed values or use defaults
                for k, spec in enumerate(self.parameter_specs):
                    if k not in (idx1, idx2):
                        if spec.name in fixed_params:
                            if spec.param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
                                x[k] = fixed_params[spec.name]
                            elif spec.param_type == ParameterType.CATEGORICAL:
                                x[k] = self.categorical_mappings[spec.name][fixed_params[spec.name]]
                            elif spec.param_type == ParameterType.BOOLEAN:
                                x[k] = 1 if fixed_params[spec.name] else 0
                        else:
                            # Use midpoint as default
                            x[k] = (self.bounds[k][0] + self.bounds[k][1]) / 2
                
                # Calculate acquisition value
                Z[i, j] = self._acquisition_function(x)
        
        return X1, X2, Z
    
    def get_gp_predictions(
        self,
        test_points: Optional[List[Dict[str, Any]]] = None,
        return_std: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get Gaussian Process predictions for test points.
        
        Args:
            test_points: List of parameter dictionaries to predict
            return_std: Whether to return prediction uncertainty
            
        Returns:
            Tuple of (predictions, std) arrays
        """
        if self._use_simplified_implementation or self.gp_model is None:
            raise ValueError("GP model not available")
        
        if test_points is None:
            # Use observed points
            X_test = np.array(self._X_observed)
        else:
            X_test = np.array([self._encode_parameters(params) for params in test_points])
        
        if return_std:
            predictions, std = self.gp_model.predict(X_test, return_std=True)
            return predictions, std
        else:
            predictions = self.gp_model.predict(X_test)
            return predictions, None