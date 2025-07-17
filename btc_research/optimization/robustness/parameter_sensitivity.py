"""
Parameter sensitivity analysis for trading strategies.

Parameter sensitivity analysis assesses how sensitive strategy performance
is to changes in individual parameters, helping identify which parameters
are most critical and which ranges are stable.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from btc_research.optimization.base import BaseRobustnessTest
from btc_research.optimization.types import RobustnessResult

__all__ = ["ParameterSensitivityTest"]


class ParameterSensitivityTest(BaseRobustnessTest):
    """
    Parameter sensitivity analysis implementation.
    
    This test varies individual parameters while keeping others constant
    to understand the sensitivity of strategy performance to parameter changes.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize parameter sensitivity test.
        
        Args:
            data: Historical data for testing
            random_seed: Random seed for reproducibility
        """
        super().__init__(data, random_seed)
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def run_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 100,
        sensitivity_range: float = 0.2,
        parameter_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None,
        **kwargs: Any,
    ) -> RobustnessResult:
        """
        Run parameter sensitivity analysis.
        
        Args:
            parameters: Base parameter values
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of sensitivity tests per parameter
            sensitivity_range: Relative range to vary parameters (±20% by default)
            parameter_ranges: Specific ranges for each parameter (overrides sensitivity_range)
            **kwargs: Additional test parameters
            
        Returns:
            Robustness test result with sensitivity analysis
        """
        results = []
        
        # Test sensitivity for each parameter
        for param_name, base_value in parameters.items():
            if not isinstance(base_value, (int, float)):
                # Skip non-numeric parameters
                continue
            
            # Determine parameter range
            if parameter_ranges and param_name in parameter_ranges:
                param_min, param_max = parameter_ranges[param_name]
            else:
                # Use relative range around base value
                if base_value > 0:
                    param_min = base_value * (1 - sensitivity_range)
                    param_max = base_value * (1 + sensitivity_range)
                else:
                    # For negative or zero values, use absolute range
                    abs_range = abs(base_value) * sensitivity_range
                    param_min = base_value - abs_range
                    param_max = base_value + abs_range
            
            # Generate test values for this parameter
            if isinstance(base_value, int):
                test_values = np.linspace(param_min, param_max, n_simulations)
                test_values = [int(round(val)) for val in test_values]
                # Remove duplicates while preserving order
                seen = set()
                test_values = [x for x in test_values if not (x in seen or seen.add(x))]
            else:
                test_values = np.linspace(param_min, param_max, n_simulations)
            
            # Test each value
            for i, test_value in enumerate(test_values):
                try:
                    # Create modified parameters
                    test_params = parameters.copy()
                    test_params[param_name] = test_value
                    
                    # Run backtest
                    metrics = backtest_function(self.data, test_params)
                    
                    # Add sensitivity analysis info
                    metrics['parameter_tested'] = param_name
                    metrics['parameter_value'] = test_value
                    metrics['parameter_baseline'] = base_value
                    metrics['parameter_change_pct'] = (test_value - base_value) / base_value * 100 if base_value != 0 else 0
                    metrics['simulation'] = i
                    
                    results.append(metrics)
                    
                except Exception as e:
                    # Record failed simulation
                    failed_metrics = {
                        'parameter_tested': param_name,
                        'parameter_value': test_value,
                        'parameter_baseline': base_value,
                        'parameter_change_pct': (test_value - base_value) / base_value * 100 if base_value != 0 else 0,
                        'simulation': i,
                        'error': str(e),
                        'total_return': float('-inf'),
                        'sharpe_ratio': float('-inf'),
                    }
                    results.append(failed_metrics)
        
        if not results:
            raise ValueError("All sensitivity tests failed")
        
        # Calculate summary statistics
        summary_stats = self._calculate_sensitivity_statistics(results)
        
        # Calculate risk metrics
        var_results, es_results = self._calculate_risk_metrics(results)
        
        # Find best and worst case scenarios
        valid_results = [r for r in results if 'error' not in r]
        
        if valid_results:
            if 'total_return' in valid_results[0]:
                best_case = max(valid_results, key=lambda x: x.get('total_return', float('-inf')))
                worst_case = min(valid_results, key=lambda x: x.get('total_return', float('-inf')))
            else:
                best_case = valid_results[0]
                worst_case = valid_results[0]
        else:
            best_case = {"total_return": 0.0}
            worst_case = {"total_return": 0.0}
        
        # Calculate success rate (placeholder)
        success_rate = len(valid_results) / len(results) if results else 0.0
        
        return RobustnessResult(
            test_type="parameter_sensitivity",
            n_simulations=len(results),
            results=results,
            summary_stats=summary_stats,
            value_at_risk=var_results,
            expected_shortfall=es_results,
            success_rate=success_rate,
            worst_case_scenario=worst_case,
            best_case_scenario=best_case,
        )
    
    def run_one_at_a_time_analysis(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        parameter_ranges: Dict[str, Tuple[Any, Any]],
        n_points_per_param: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run one-at-a-time sensitivity analysis.
        
        Args:
            parameters: Base parameter values
            backtest_function: Function that runs backtest and returns metrics
            parameter_ranges: Ranges for each parameter to test
            n_points_per_param: Number of test points per parameter
            
        Returns:
            Dictionary with sensitivity results for each parameter
        """
        sensitivity_results = {}
        
        for param_name, (param_min, param_max) in parameter_ranges.items():
            if param_name not in parameters:
                continue
            
            base_value = parameters[param_name]
            
            # Generate test values
            if isinstance(base_value, int):
                test_values = np.linspace(param_min, param_max, n_points_per_param)
                test_values = [int(round(val)) for val in test_values]
            else:
                test_values = np.linspace(param_min, param_max, n_points_per_param)
            
            param_results = []
            
            for test_value in test_values:
                try:
                    # Create modified parameters
                    test_params = parameters.copy()
                    test_params[param_name] = test_value
                    
                    # Run backtest
                    metrics = backtest_function(self.data, test_params)
                    
                    param_results.append({
                        'parameter_value': test_value,
                        'metrics': metrics,
                    })
                    
                except Exception as e:
                    param_results.append({
                        'parameter_value': test_value,
                        'error': str(e),
                        'metrics': None,
                    })
            
            # Calculate sensitivity metrics for this parameter
            sensitivity_metrics = self._calculate_parameter_sensitivity(param_results, param_name)
            
            sensitivity_results[param_name] = {
                'base_value': base_value,
                'test_range': (param_min, param_max),
                'results': param_results,
                'sensitivity_metrics': sensitivity_metrics,
            }
        
        return sensitivity_results
    
    def run_interaction_analysis(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        parameter_pairs: List[Tuple[str, str]],
        parameter_ranges: Dict[str, Tuple[Any, Any]],
        n_points_per_param: int = 5,
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Run parameter interaction analysis.
        
        Args:
            parameters: Base parameter values
            backtest_function: Function that runs backtest and returns metrics
            parameter_pairs: List of parameter pairs to test for interactions
            parameter_ranges: Ranges for each parameter
            n_points_per_param: Number of test points per parameter dimension
            
        Returns:
            Dictionary with interaction results for each parameter pair
        """
        interaction_results = {}
        
        for param1, param2 in parameter_pairs:
            if param1 not in parameters or param2 not in parameters:
                continue
            
            if param1 not in parameter_ranges or param2 not in parameter_ranges:
                continue
            
            # Generate test grids
            param1_min, param1_max = parameter_ranges[param1]
            param2_min, param2_max = parameter_ranges[param2]
            
            base_value1 = parameters[param1]
            base_value2 = parameters[param2]
            
            if isinstance(base_value1, int):
                param1_values = np.linspace(param1_min, param1_max, n_points_per_param)
                param1_values = [int(round(val)) for val in param1_values]
            else:
                param1_values = np.linspace(param1_min, param1_max, n_points_per_param)
            
            if isinstance(base_value2, int):
                param2_values = np.linspace(param2_min, param2_max, n_points_per_param)
                param2_values = [int(round(val)) for val in param2_values]
            else:
                param2_values = np.linspace(param2_min, param2_max, n_points_per_param)
            
            # Test all combinations
            interaction_data = []
            
            for val1 in param1_values:
                for val2 in param2_values:
                    try:
                        # Create modified parameters
                        test_params = parameters.copy()
                        test_params[param1] = val1
                        test_params[param2] = val2
                        
                        # Run backtest
                        metrics = backtest_function(self.data, test_params)
                        
                        interaction_data.append({
                            param1: val1,
                            param2: val2,
                            'metrics': metrics,
                        })
                        
                    except Exception as e:
                        interaction_data.append({
                            param1: val1,
                            param2: val2,
                            'error': str(e),
                            'metrics': None,
                        })
            
            # Calculate interaction metrics
            interaction_metrics = self._calculate_interaction_metrics(interaction_data, param1, param2)
            
            interaction_results[(param1, param2)] = {
                'parameter_ranges': {param1: (param1_min, param1_max), param2: (param2_min, param2_max)},
                'base_values': {param1: base_value1, param2: base_value2},
                'interaction_data': interaction_data,
                'interaction_metrics': interaction_metrics,
            }
        
        return interaction_results
    
    def _calculate_sensitivity_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate sensitivity-specific statistics.
        
        Args:
            results: List of sensitivity test results
            
        Returns:
            Dictionary of sensitivity statistics by metric and parameter
        """
        summary_stats = {}
        
        # Group results by parameter
        param_groups = {}
        for result in results:
            param_name = result.get('parameter_tested')
            if param_name:
                if param_name not in param_groups:
                    param_groups[param_name] = []
                param_groups[param_name].append(result)
        
        # Calculate statistics for each parameter
        for param_name, param_results in param_groups.items():
            param_stats = {}
            
            # Get all numeric metrics
            all_metrics = set()
            for result in param_results:
                for key, value in result.items():
                    if isinstance(value, (int, float)) and key not in ['simulation', 'parameter_value', 'parameter_baseline', 'parameter_change_pct']:
                        all_metrics.add(key)
            
            for metric in all_metrics:
                values = []
                param_changes = []
                
                for result in param_results:
                    if metric in result and isinstance(result[metric], (int, float)):
                        if not (np.isnan(result[metric]) or np.isinf(result[metric])):
                            values.append(result[metric])
                            param_changes.append(result.get('parameter_change_pct', 0))
                
                if values and param_changes:
                    # Calculate sensitivity coefficient (dMetric/dParameter)
                    if len(values) > 1:
                        sensitivity_coeff = np.corrcoef(param_changes, values)[0, 1] if np.std(param_changes) > 0 else 0
                    else:
                        sensitivity_coeff = 0
                    
                    param_stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'range': np.max(values) - np.min(values),
                        'sensitivity_coefficient': sensitivity_coeff,
                        'relative_sensitivity': np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else float('inf'),
                    }
            
            summary_stats[param_name] = param_stats
        
        return summary_stats
    
    def _calculate_parameter_sensitivity(self, param_results: List[Dict[str, Any]], param_name: str) -> Dict[str, float]:
        """
        Calculate sensitivity metrics for a single parameter.
        
        Args:
            param_results: Results for different values of the parameter
            param_name: Name of the parameter
            
        Returns:
            Dictionary with sensitivity metrics
        """
        # Extract successful results
        valid_results = [r for r in param_results if r.get('metrics') is not None]
        
        if len(valid_results) < 2:
            return {'sensitivity_score': 0.0, 'stability_score': 0.0}
        
        # Get parameter values and primary metric (total_return)
        param_values = [r['parameter_value'] for r in valid_results]
        metric_values = [r['metrics'].get('total_return', 0) for r in valid_results]
        
        # Calculate sensitivity score (coefficient of variation)
        if len(metric_values) > 1:
            mean_metric = np.mean(metric_values)
            std_metric = np.std(metric_values)
            sensitivity_score = std_metric / abs(mean_metric) if mean_metric != 0 else float('inf')
        else:
            sensitivity_score = 0.0
        
        # Calculate stability score (inverse of range)
        metric_range = max(metric_values) - min(metric_values)
        stability_score = 1.0 / (1.0 + metric_range)  # Higher score = more stable
        
        return {
            'sensitivity_score': sensitivity_score,
            'stability_score': stability_score,
            'metric_range': metric_range,
            'n_valid_tests': len(valid_results),
        }
    
    def _calculate_interaction_metrics(self, interaction_data: List[Dict[str, Any]], param1: str, param2: str) -> Dict[str, float]:
        """
        Calculate interaction metrics for a parameter pair.
        
        Args:
            interaction_data: Interaction test results
            param1: First parameter name
            param2: Second parameter name
            
        Returns:
            Dictionary with interaction metrics
        """
        # Extract successful results
        valid_results = [r for r in interaction_data if r.get('metrics') is not None]
        
        if len(valid_results) < 4:
            return {'interaction_strength': 0.0}
        
        # Get parameter values and metric values
        param1_values = [r[param1] for r in valid_results]
        param2_values = [r[param2] for r in valid_results]
        metric_values = [r['metrics'].get('total_return', 0) for r in valid_results]
        
        # Calculate interaction strength using variance decomposition
        # This is a simplified measure - proper interaction analysis would use ANOVA
        total_variance = np.var(metric_values)
        
        # Calculate main effects
        param1_effect = np.var([np.mean([m for i, m in enumerate(metric_values) if param1_values[i] == val]) 
                               for val in set(param1_values)])
        param2_effect = np.var([np.mean([m for i, m in enumerate(metric_values) if param2_values[i] == val]) 
                               for val in set(param2_values)])
        
        # Interaction effect (simplified)
        main_effects = param1_effect + param2_effect
        interaction_effect = max(0, total_variance - main_effects)
        
        interaction_strength = interaction_effect / total_variance if total_variance > 0 else 0
        
        return {
            'interaction_strength': interaction_strength,
            'total_variance': total_variance,
            'param1_effect': param1_effect,
            'param2_effect': param2_effect,
            'interaction_effect': interaction_effect,
        }