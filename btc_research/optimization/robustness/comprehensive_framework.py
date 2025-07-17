"""
Comprehensive robustness testing framework integration.

This module provides a unified interface for running comprehensive
robustness testing including Monte Carlo simulations, stress testing,
synthetic data generation, and comprehensive scoring.
"""

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from btc_research.optimization.base import BaseRobustnessTest
from btc_research.optimization.types import RobustnessResult
from btc_research.optimization.robustness.monte_carlo import MonteCarloRobustnessTest
from btc_research.optimization.robustness.data_perturbation import DataPerturbationTest
from btc_research.optimization.robustness.synthetic_data import SyntheticDataTest
from btc_research.optimization.robustness.stress_tests import StressTestFramework
from btc_research.optimization.robustness.parameter_sensitivity import ParameterSensitivityTest
from btc_research.optimization.robustness.robustness_metrics import RobustnessMetrics, RobustnessScoring

__all__ = ["ComprehensiveRobustnessFramework"]

logger = logging.getLogger(__name__)


class ComprehensiveRobustnessFramework:
    """
    Unified framework for comprehensive strategy robustness testing.
    
    This framework orchestrates multiple robustness testing methods to provide
    a complete assessment of trading strategy robustness and reliability.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        random_seed: Optional[int] = None,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        logging_level: int = logging.INFO,
    ):
        """
        Initialize comprehensive robustness framework.
        
        Args:
            data: Historical data for testing
            random_seed: Random seed for reproducibility
            enable_parallel: Enable parallel execution
            max_workers: Maximum number of workers for parallel execution
            logging_level: Logging level for framework output
        """
        self.data = data
        self.random_seed = random_seed
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        
        # Set up logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        
        # Initialize robustness test components
        self._initialize_test_components()
        
        # Initialize scoring system
        self.scorer = RobustnessScoring()
        
        # Store results
        self.results = {}
        
    def _initialize_test_components(self) -> None:
        """Initialize all robustness test components."""
        self.monte_carlo_test = MonteCarloRobustnessTest(
            data=self.data,
            random_seed=self.random_seed,
            enable_parallel=self.enable_parallel,
            max_workers=self.max_workers
        )
        
        self.data_perturbation_test = DataPerturbationTest(
            data=self.data,
            random_seed=self.random_seed
        )
        
        self.synthetic_data_test = SyntheticDataTest(
            data=self.data,
            random_seed=self.random_seed
        )
        
        self.stress_test = StressTestFramework(
            data=self.data,
            random_seed=self.random_seed
        )
        
        self.parameter_sensitivity_test = ParameterSensitivityTest(
            data=self.data,
            random_seed=self.random_seed
        )
    
    def run_comprehensive_assessment(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        test_config: Optional[Dict[str, Any]] = None,
        baseline_results: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive robustness assessment.
        
        Args:
            parameters: Strategy parameters to test
            backtest_function: Function that runs backtest and returns metrics
            test_config: Configuration for robustness tests
            baseline_results: Baseline results for comparison
            
        Returns:
            Dictionary with comprehensive robustness assessment results
        """
        if test_config is None:
            test_config = self._get_default_test_config()
        
        self.logger.info("Starting comprehensive robustness assessment...")
        
        # Run baseline backtest if not provided
        if baseline_results is None:
            self.logger.info("Running baseline backtest...")
            baseline_results = backtest_function(self.data, parameters)
        
        # Store baseline results
        self.results['baseline_results'] = baseline_results
        
        # Run all robustness tests
        if self.enable_parallel:
            self._run_tests_parallel(parameters, backtest_function, test_config)
        else:
            self._run_tests_sequential(parameters, backtest_function, test_config)
        
        # Calculate comprehensive metrics
        self.logger.info("Calculating comprehensive robustness metrics...")
        comprehensive_metrics = self._calculate_comprehensive_metrics(baseline_results)
        
        # Calculate comprehensive score
        self.logger.info("Calculating comprehensive robustness score...")
        robustness_score = self.scorer.calculate_comprehensive_score(comprehensive_metrics)
        
        # Generate comprehensive report
        self.logger.info("Generating comprehensive robustness report...")
        report = self.scorer.generate_robustness_report(robustness_score, self.results)
        
        # Compile final results
        final_results = {
            'baseline_results': baseline_results,
            'robustness_tests': self.results,
            'comprehensive_metrics': comprehensive_metrics,
            'robustness_score': robustness_score,
            'robustness_report': report,
            'framework_config': test_config,
        }
        
        self.logger.info("Comprehensive robustness assessment completed.")
        
        return final_results
    
    def run_targeted_assessment(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        test_types: List[str],
        test_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run targeted robustness assessment with specific test types.
        
        Args:
            parameters: Strategy parameters to test
            backtest_function: Function that runs backtest and returns metrics
            test_types: List of test types to run
            test_config: Configuration for robustness tests
            
        Returns:
            Dictionary with targeted robustness assessment results
        """
        if test_config is None:
            test_config = self._get_default_test_config()
        
        self.logger.info(f"Running targeted robustness assessment: {test_types}")
        
        targeted_results = {}
        
        for test_type in test_types:
            try:
                if test_type == 'monte_carlo':
                    result = self._run_monte_carlo_test(parameters, backtest_function, test_config)
                    targeted_results['monte_carlo'] = result
                    
                elif test_type == 'data_perturbation':
                    result = self._run_data_perturbation_test(parameters, backtest_function, test_config)
                    targeted_results['data_perturbation'] = result
                    
                elif test_type == 'synthetic_data':
                    result = self._run_synthetic_data_test(parameters, backtest_function, test_config)
                    targeted_results['synthetic_data'] = result
                    
                elif test_type == 'stress_testing':
                    result = self._run_stress_test(parameters, backtest_function, test_config)
                    targeted_results['stress_testing'] = result
                    
                elif test_type == 'parameter_sensitivity':
                    result = self._run_parameter_sensitivity_test(parameters, backtest_function, test_config)
                    targeted_results['parameter_sensitivity'] = result
                    
                else:
                    self.logger.warning(f"Unknown test type: {test_type}")
                    
            except Exception as e:
                self.logger.error(f"Test {test_type} failed: {e}")
                targeted_results[test_type] = {'error': str(e)}
        
        return targeted_results
    
    def run_quick_assessment(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Run quick robustness assessment with reduced simulation counts.
        
        Args:
            parameters: Strategy parameters to test
            backtest_function: Function that runs backtest and returns metrics
            
        Returns:
            Dictionary with quick robustness assessment results
        """
        self.logger.info("Running quick robustness assessment...")
        
        quick_config = {
            'monte_carlo': {'n_simulations': 100},
            'data_perturbation': {'n_simulations': 50},
            'synthetic_data': {'n_simulations': 100},
            'stress_testing': {'n_simulations': 50},
            'parameter_sensitivity': {'n_simulations': 50},
        }
        
        return self.run_targeted_assessment(
            parameters, backtest_function, 
            ['monte_carlo', 'data_perturbation', 'stress_testing'],
            quick_config
        )
    
    def _get_default_test_config(self) -> Dict[str, Any]:
        """Get default configuration for robustness tests."""
        return {
            'monte_carlo': {
                'n_simulations': 1000,
                'test_types': ['data_noise', 'parameter_variation', 'trade_sequence'],
                'noise_levels': [0.005, 0.01, 0.02],
                'block_sizes': [10, 20, 50],
            },
            'data_perturbation': {
                'n_simulations': 500,
                'perturbation_types': ['price_noise', 'volume_noise', 'missing_data', 'regime_change'],
                'noise_levels': [0.001, 0.005, 0.01, 0.02, 0.05],
            },
            'synthetic_data': {
                'n_simulations': 500,
                'methods': ['permutation', 'garch', 'regime_switching'],
                'overfitting_detection': True,
            },
            'stress_testing': {
                'n_simulations': 300,
                'scenarios': ['flash_crash', 'liquidity_crisis', 'high_volatility', 'black_swan'],
                'crash_magnitudes': [0.1, 0.2, 0.3, 0.5],
                'recovery_times': [5, 20, 50, 100],
            },
            'parameter_sensitivity': {
                'n_simulations': 200,
                'sensitivity_range': 0.2,
                'interaction_analysis': True,
            },
        }
    
    def _run_tests_parallel(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        test_config: Dict[str, Any],
    ) -> None:
        """Run robustness tests in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tests
            futures = {}
            
            futures['monte_carlo'] = executor.submit(
                self._run_monte_carlo_test, parameters, backtest_function, test_config
            )
            
            futures['data_perturbation'] = executor.submit(
                self._run_data_perturbation_test, parameters, backtest_function, test_config
            )
            
            futures['synthetic_data'] = executor.submit(
                self._run_synthetic_data_test, parameters, backtest_function, test_config
            )
            
            futures['stress_testing'] = executor.submit(
                self._run_stress_test, parameters, backtest_function, test_config
            )
            
            futures['parameter_sensitivity'] = executor.submit(
                self._run_parameter_sensitivity_test, parameters, backtest_function, test_config
            )
            
            # Collect results
            for test_name, future in futures.items():
                try:
                    self.results[test_name] = future.result()
                    self.logger.info(f"Completed {test_name} test")
                except Exception as e:
                    self.logger.error(f"Test {test_name} failed: {e}")
                    self.results[test_name] = {'error': str(e)}
    
    def _run_tests_sequential(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        test_config: Dict[str, Any],
    ) -> None:
        """Run robustness tests sequentially."""
        test_methods = [
            ('monte_carlo', self._run_monte_carlo_test),
            ('data_perturbation', self._run_data_perturbation_test),
            ('synthetic_data', self._run_synthetic_data_test),
            ('stress_testing', self._run_stress_test),
            ('parameter_sensitivity', self._run_parameter_sensitivity_test),
        ]
        
        for test_name, test_method in test_methods:
            try:
                self.logger.info(f"Running {test_name} test...")
                self.results[test_name] = test_method(parameters, backtest_function, test_config)
                self.logger.info(f"Completed {test_name} test")
            except Exception as e:
                self.logger.error(f"Test {test_name} failed: {e}")
                self.results[test_name] = {'error': str(e)}
    
    def _run_monte_carlo_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        test_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Monte Carlo robustness test."""
        mc_config = test_config.get('monte_carlo', {})
        
        results = {}
        
        # Standard Monte Carlo test
        results['standard'] = self.monte_carlo_test.run_test(
            parameters=parameters,
            backtest_function=backtest_function,
            n_simulations=mc_config.get('n_simulations', 1000),
            test_type="data_noise"
        )
        
        # Trade sequence resampling
        results['trade_sequence'] = self.monte_carlo_test.run_trade_sequence_test(
            parameters=parameters,
            backtest_function=backtest_function,
            n_simulations=mc_config.get('n_simulations', 1000) // 2,
            block_size=20,
            preserve_correlations=True
        )
        
        # Statistical significance test
        results['statistical_significance'] = self.monte_carlo_test.run_statistical_significance_test(
            parameters=parameters,
            backtest_function=backtest_function,
            n_simulations=mc_config.get('n_simulations', 1000),
            confidence_level=0.95
        )
        
        return results
    
    def _run_data_perturbation_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        test_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run data perturbation robustness test."""
        dp_config = test_config.get('data_perturbation', {})
        
        results = {}
        
        # Comprehensive perturbation test
        results['comprehensive'] = self.data_perturbation_test.run_test(
            parameters=parameters,
            backtest_function=backtest_function,
            n_simulations=dp_config.get('n_simulations', 500),
            perturbation_types=dp_config.get('perturbation_types', ['price_noise', 'volume_noise'])
        )
        
        # Systematic noise testing
        results['noise_analysis'] = self.data_perturbation_test.run_price_noise_test(
            parameters=parameters,
            backtest_function=backtest_function,
            noise_levels=dp_config.get('noise_levels', [0.005, 0.01, 0.02]),
            n_simulations_per_level=50,
            noise_type="gaussian"
        )
        
        # Market regime testing
        results['regime_testing'] = self.data_perturbation_test.run_market_regime_test(
            parameters=parameters,
            backtest_function=backtest_function,
            regime_types=['bull_market', 'bear_market', 'high_volatility'],
            n_simulations=dp_config.get('n_simulations', 500) // 3
        )
        
        return results
    
    def _run_synthetic_data_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        test_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run synthetic data robustness test."""
        sd_config = test_config.get('synthetic_data', {})
        
        results = {}
        
        # Comprehensive synthetic data test
        results['comprehensive'] = self.synthetic_data_test.run_test(
            parameters=parameters,
            backtest_function=backtest_function,
            n_simulations=sd_config.get('n_simulations', 500),
            synthetic_methods=sd_config.get('methods', ['permutation', 'bootstrap_block'])
        )
        
        # Permutation test for overfitting
        results['permutation'] = self.synthetic_data_test.run_permutation_test(
            parameters=parameters,
            backtest_function=backtest_function,
            n_simulations=sd_config.get('n_simulations', 500),
            preserve_properties=['returns_distribution']
        )
        
        # Overfitting detection
        if sd_config.get('overfitting_detection', True):
            results['overfitting_detection'] = self.synthetic_data_test.run_overfitting_detection(
                parameters=parameters,
                backtest_function=backtest_function,
                detection_methods=['permutation', 'data_mining_bias'],
                n_simulations=sd_config.get('n_simulations', 500)
            )
        
        return results
    
    def _run_stress_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        test_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run stress testing."""
        st_config = test_config.get('stress_testing', {})
        
        results = {}
        
        # Comprehensive stress test
        results['comprehensive'] = self.stress_test.run_test(
            parameters=parameters,
            backtest_function=backtest_function,
            n_simulations=st_config.get('n_simulations', 300),
            stress_scenarios=st_config.get('scenarios', ['flash_crash', 'high_volatility'])
        )
        
        # Flash crash test
        results['flash_crash'] = self.stress_test.run_flash_crash_test(
            parameters=parameters,
            backtest_function=backtest_function,
            crash_magnitudes=st_config.get('crash_magnitudes', [0.1, 0.2, 0.3]),
            recovery_times=st_config.get('recovery_times', [5, 20, 50]),
            n_simulations=50
        )
        
        # Black swan events
        results['black_swan'] = self.stress_test.run_black_swan_test(
            parameters=parameters,
            backtest_function=backtest_function,
            event_types=['market_crash', 'regulatory_shock'],
            n_simulations=st_config.get('n_simulations', 300) // 3
        )
        
        return results
    
    def _run_parameter_sensitivity_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        test_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run parameter sensitivity test."""
        ps_config = test_config.get('parameter_sensitivity', {})
        
        results = {}
        
        # Standard sensitivity test
        results['standard'] = self.parameter_sensitivity_test.run_test(
            parameters=parameters,
            backtest_function=backtest_function,
            n_simulations=ps_config.get('n_simulations', 200),
            sensitivity_range=ps_config.get('sensitivity_range', 0.2)
        )
        
        # One-at-a-time analysis if parameter ranges provided
        if 'parameter_ranges' in ps_config:
            results['one_at_a_time'] = self.parameter_sensitivity_test.run_one_at_a_time_analysis(
                parameters=parameters,
                backtest_function=backtest_function,
                parameter_ranges=ps_config['parameter_ranges'],
                n_points_per_param=10
            )
        
        return results
    
    def _calculate_comprehensive_metrics(self, baseline_results: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive robustness metrics."""
        comprehensive_metrics = {}
        
        # Extract results for metrics calculation
        all_simulation_results = []
        
        # Collect Monte Carlo results
        if 'monte_carlo' in self.results and 'standard' in self.results['monte_carlo']:
            mc_results = self.results['monte_carlo']['standard']
            if hasattr(mc_results, 'results'):
                all_simulation_results.extend(mc_results.results)
        
        # Calculate stability metrics
        if all_simulation_results:
            stability_metrics = RobustnessMetrics.calculate_stability_metrics(
                all_simulation_results, baseline_results
            )
            comprehensive_metrics['stability_metrics'] = stability_metrics
        
        # Calculate tail risk metrics
        if all_simulation_results:
            tail_risk_metrics = RobustnessMetrics.calculate_tail_risk_metrics(
                all_simulation_results,
                metrics=['total_return', 'sharpe_ratio', 'max_drawdown'],
                confidence_levels=[0.90, 0.95, 0.99]
            )
            comprehensive_metrics['tail_risk_metrics'] = tail_risk_metrics
        
        # Calculate overfitting metrics
        if 'synthetic_data' in self.results:
            overfitting_data = {}
            
            if 'overfitting_detection' in self.results['synthetic_data']:
                overfitting_result = self.results['synthetic_data']['overfitting_detection']
                overfitting_data['overfitting_score'] = overfitting_result.get('overfitting_score', 0.5)
            
            if overfitting_data:
                overfitting_metrics = RobustnessMetrics.calculate_overfitting_metrics(
                    baseline_results, all_simulation_results, all_simulation_results
                )
                comprehensive_metrics['overfitting_metrics'] = overfitting_metrics
        
        # Add stress test results
        if 'stress_testing' in self.results and 'comprehensive' in self.results['stress_testing']:
            stress_result = self.results['stress_testing']['comprehensive']
            comprehensive_metrics['stress_results'] = stress_result
        
        # Add parameter sensitivity results
        if 'parameter_sensitivity' in self.results and 'standard' in self.results['parameter_sensitivity']:
            sensitivity_result = self.results['parameter_sensitivity']['standard']
            if hasattr(sensitivity_result, 'summary_stats'):
                comprehensive_metrics['sensitivity_metrics'] = sensitivity_result.summary_stats
        
        return comprehensive_metrics
    
    def export_results(self, filepath: str, format: str = 'json') -> None:
        """
        Export robustness test results.
        
        Args:
            filepath: Path to save results
            format: Export format ('json', 'pickle')
        """
        if not self.results:
            raise ValueError("No results to export. Run assessment first.")
        
        if format == 'json':
            import json
            # Convert results to JSON-serializable format
            serializable_results = self._make_json_serializable(self.results)
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
        elif format == 'pickle':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.results, f)
                
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Results exported to {filepath}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting to dict
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj