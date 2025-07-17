"""
Synthetic data generation for trading strategy robustness testing.

This module implements various methods for generating synthetic market data
including GARCH models, permutation testing, regime-switching models, and
other statistical approaches for overfitting detection.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    warnings.warn("arch package not available. GARCH models will be disabled.")

from btc_research.optimization.base import BaseRobustnessTest
from btc_research.optimization.types import RobustnessResult

__all__ = ["SyntheticDataTest"]


class SyntheticDataTest(BaseRobustnessTest):
    """
    Synthetic data generation for robustness testing.
    
    This test generates synthetic market data using various statistical
    models to assess strategy robustness and detect overfitting.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize synthetic data test.
        
        Args:
            data: Historical data for model calibration
            random_seed: Random seed for reproducibility
        """
        super().__init__(data, random_seed)
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Analyze data for model calibration
        self._calibrate_models()
    
    def run_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 1000,
        synthetic_methods: List[str] = None,
        **kwargs: Any,
    ) -> RobustnessResult:
        """
        Run comprehensive synthetic data robustness test.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of simulations to run
            synthetic_methods: List of synthetic data methods to use
            **kwargs: Additional test parameters
            
        Returns:
            Robustness test result with synthetic data analysis
        """
        if synthetic_methods is None:
            synthetic_methods = ['permutation', 'garch', 'regime_switching', 'bootstrap_block']
            
        # Filter out methods that aren't available
        if not HAS_ARCH and 'garch' in synthetic_methods:
            synthetic_methods.remove('garch')
            warnings.warn("GARCH method disabled due to missing arch package")
        
        results = []
        
        for method in synthetic_methods:
            # Run simulations for each synthetic method
            method_results = self._run_synthetic_simulations(
                parameters, backtest_function, method,
                n_simulations // len(synthetic_methods), **kwargs
            )
            results.extend(method_results)
        
        # Calculate summary statistics
        summary_stats = self._calculate_synthetic_statistics(results)
        
        # Calculate risk metrics
        var_results, es_results = self._calculate_risk_metrics(results)
        
        # Calculate success rate
        success_rate = len([r for r in results if 'error' not in r]) / len(results)
        
        # Find best and worst case scenarios
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_case = max(valid_results, key=lambda x: x.get('total_return', float('-inf')))
            worst_case = min(valid_results, key=lambda x: x.get('total_return', float('-inf')))
        else:
            best_case = {"total_return": 0.0}
            worst_case = {"total_return": 0.0}
        
        return RobustnessResult(
            test_type="synthetic_data",
            n_simulations=len(results),
            results=results,
            summary_stats=summary_stats,
            value_at_risk=var_results,
            expected_shortfall=es_results,
            success_rate=success_rate,
            worst_case_scenario=worst_case,
            best_case_scenario=best_case,
        )
    
    def run_permutation_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 1000,
        preserve_properties: List[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run permutation test with statistical property preservation.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of permutation simulations
            preserve_properties: List of statistical properties to preserve
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with permutation test results and p-values
        """
        if preserve_properties is None:
            preserve_properties = ['returns_distribution', 'volatility_clustering', 'autocorrelation']
        
        # Get baseline performance
        baseline_result = backtest_function(self.data, parameters)
        baseline_return = baseline_result.get('total_return', 0)
        
        # Run permutation simulations
        permutation_results = []
        
        for i in range(n_simulations):
            try:
                # Generate permuted data
                permuted_data = self._generate_permuted_data(preserve_properties)
                
                # Run backtest
                metrics = backtest_function(permuted_data, parameters)
                
                # Add metadata
                metrics['simulation'] = i
                metrics['method'] = 'permutation'
                metrics['preserved_properties'] = preserve_properties
                
                permutation_results.append(metrics)
                
            except Exception as e:
                failed_metrics = {
                    'simulation': i,
                    'method': 'permutation',
                    'error': str(e),
                    'total_return': float('-inf'),
                    'sharpe_ratio': float('-inf'),
                }
                permutation_results.append(failed_metrics)
        
        # Calculate p-value
        valid_results = [r for r in permutation_results if 'error' not in r]
        if valid_results:
            permuted_returns = [r.get('total_return', 0) for r in valid_results]
            p_value = sum(1 for r in permuted_returns if r >= baseline_return) / len(permuted_returns)
        else:
            p_value = 1.0
        
        # Statistical significance test
        significance_test = {
            'baseline_return': baseline_return,
            'permutation_mean': np.mean([r.get('total_return', 0) for r in valid_results]) if valid_results else 0,
            'p_value': p_value,
            'significant_at_05': p_value < 0.05,
            'significant_at_01': p_value < 0.01,
            'overfitting_detected': p_value > 0.95,  # Strategy performs too well on original data
        }
        
        return {
            'permutation_results': permutation_results,
            'significance_test': significance_test,
            'baseline_result': baseline_result,
        }
    
    def run_garch_simulation(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 500,
        garch_params: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run GARCH model simulation for volatility modeling.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of GARCH simulations
            garch_params: GARCH model parameters
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with GARCH simulation results
        """
        if not HAS_ARCH:
            raise ImportError("arch package required for GARCH simulations. Install with: pip install arch")
        
        if garch_params is None:
            garch_params = {'p': 1, 'q': 1, 'vol': 'GARCH'}
        
        garch_results = []
        
        for i in range(n_simulations):
            try:
                # Generate GARCH data
                garch_data = self._generate_garch_data(garch_params)
                
                # Run backtest
                metrics = backtest_function(garch_data, parameters)
                
                # Add metadata
                metrics['simulation'] = i
                metrics['method'] = 'garch'
                metrics['garch_params'] = garch_params
                
                garch_results.append(metrics)
                
            except Exception as e:
                failed_metrics = {
                    'simulation': i,
                    'method': 'garch',
                    'error': str(e),
                    'total_return': float('-inf'),
                    'sharpe_ratio': float('-inf'),
                }
                garch_results.append(failed_metrics)
        
        # Analyze GARCH results
        garch_analysis = self._analyze_garch_results(garch_results)
        
        return {
            'garch_results': garch_results,
            'garch_analysis': garch_analysis,
            'model_params': garch_params,
        }
    
    def run_regime_switching_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int = 300,
        n_regimes: int = 2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run regime-switching model simulation.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            n_simulations: Number of regime-switching simulations
            n_regimes: Number of market regimes
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with regime-switching results
        """
        regime_results = []
        
        for i in range(n_simulations):
            try:
                # Generate regime-switching data
                regime_data = self._generate_regime_switching_data(n_regimes)
                
                # Run backtest
                metrics = backtest_function(regime_data, parameters)
                
                # Add metadata
                metrics['simulation'] = i
                metrics['method'] = 'regime_switching'
                metrics['n_regimes'] = n_regimes
                
                regime_results.append(metrics)
                
            except Exception as e:
                failed_metrics = {
                    'simulation': i,
                    'method': 'regime_switching',
                    'error': str(e),
                    'total_return': float('-inf'),
                    'sharpe_ratio': float('-inf'),
                }
                regime_results.append(failed_metrics)
        
        # Analyze regime results
        regime_analysis = self._analyze_regime_results(regime_results)
        
        return {
            'regime_results': regime_results,
            'regime_analysis': regime_analysis,
            'n_regimes': n_regimes,
        }
    
    def run_overfitting_detection(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        detection_methods: List[str] = None,
        n_simulations: int = 1000,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run comprehensive overfitting detection using multiple methods.
        
        Args:
            parameters: Parameter values to test
            backtest_function: Function that runs backtest and returns metrics
            detection_methods: List of detection methods to use
            n_simulations: Number of simulations per method
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with overfitting detection results
        """
        if detection_methods is None:
            detection_methods = ['permutation', 'data_mining_bias', 'multiple_testing']
        
        detection_results = {}
        
        # Get baseline performance
        baseline_result = backtest_function(self.data, parameters)
        
        for method in detection_methods:
            if method == 'permutation':
                # Permutation test
                perm_result = self.run_permutation_test(
                    parameters, backtest_function, 
                    n_simulations // len(detection_methods)
                )
                detection_results['permutation'] = perm_result
                
            elif method == 'data_mining_bias':
                # Data mining bias test
                dmb_result = self._run_data_mining_bias_test(
                    parameters, backtest_function, n_simulations // len(detection_methods)
                )
                detection_results['data_mining_bias'] = dmb_result
                
            elif method == 'multiple_testing':
                # Multiple testing correction
                mt_result = self._run_multiple_testing_analysis(
                    parameters, backtest_function, n_simulations // len(detection_methods)
                )
                detection_results['multiple_testing'] = mt_result
        
        # Aggregate overfitting signals
        overfitting_score = self._calculate_overfitting_score(detection_results)
        
        return {
            'detection_results': detection_results,
            'overfitting_score': overfitting_score,
            'baseline_result': baseline_result,
            'overfitting_detected': overfitting_score > 0.7,
        }
    
    def _calibrate_models(self) -> None:
        """
        Calibrate models based on historical data characteristics.
        """
        if 'close' in self.data.columns:
            prices = self.data['close']
            self.returns = prices.pct_change().dropna()
            
            # Basic statistics
            self.return_mean = self.returns.mean()
            self.return_std = self.returns.std()
            self.return_skew = self.returns.skew()
            self.return_kurt = self.returns.kurtosis()
            
            # Autocorrelation
            self.autocorr_lags = [self.returns.autocorr(lag) for lag in range(1, 11)]
            
            # Volatility clustering (ARCH effects)
            squared_returns = self.returns ** 2
            self.arch_effects = [squared_returns.autocorr(lag) for lag in range(1, 6)]
            
        else:
            # Default values if no price data
            self.return_mean = 0.0005
            self.return_std = 0.02
            self.return_skew = -0.5
            self.return_kurt = 3.0
            self.autocorr_lags = [0.0] * 10
            self.arch_effects = [0.0] * 5
    
    def _run_synthetic_simulations(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        method: str,
        n_simulations: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Run simulations for a specific synthetic data method.
        """
        results = []
        
        for i in range(n_simulations):
            try:
                # Generate synthetic data based on method
                if method == 'permutation':
                    synthetic_data = self._generate_permuted_data()
                elif method == 'garch':
                    synthetic_data = self._generate_garch_data()
                elif method == 'regime_switching':
                    synthetic_data = self._generate_regime_switching_data()
                elif method == 'bootstrap_block':
                    synthetic_data = self._generate_block_bootstrap_data()
                else:
                    raise ValueError(f"Unknown synthetic method: {method}")
                
                # Run backtest
                metrics = backtest_function(synthetic_data, parameters)
                
                # Add metadata
                metrics['simulation'] = i
                metrics['synthetic_method'] = method
                
                results.append(metrics)
                
            except Exception as e:
                failed_metrics = {
                    'simulation': i,
                    'synthetic_method': method,
                    'error': str(e),
                    'total_return': float('-inf'),
                    'sharpe_ratio': float('-inf'),
                }
                results.append(failed_metrics)
        
        return results
    
    def _generate_permuted_data(self, preserve_properties: List[str] = None) -> pd.DataFrame:
        """
        Generate permuted data while preserving statistical properties.
        """
        if preserve_properties is None:
            preserve_properties = ['returns_distribution']
        
        permuted_data = self.data.copy()
        
        if 'close' in self.data.columns:
            if 'returns_distribution' in preserve_properties:
                # Shuffle returns but preserve distribution
                shuffled_returns = np.random.permutation(self.returns.values)
                
                # Reconstruct prices
                initial_price = self.data['close'].iloc[0]
                new_prices = [initial_price]
                
                for ret in shuffled_returns:
                    new_price = new_prices[-1] * (1 + ret)
                    new_prices.append(new_price)
                
                # Update all price columns proportionally
                price_ratio = np.array(new_prices[1:]) / self.data['close'].iloc[1:].values
                
                for col in ['open', 'high', 'low', 'close']:
                    if col in permuted_data.columns:
                        permuted_data.iloc[1:, permuted_data.columns.get_loc(col)] = (
                            permuted_data.iloc[1:, permuted_data.columns.get_loc(col)] * price_ratio
                        )
            
            elif 'volatility_clustering' in preserve_properties:
                # Preserve volatility clustering using bootstrap blocks
                block_size = 20  # Preserve short-term dependencies
                permuted_data = self._generate_block_bootstrap_data(block_size)
        
        return permuted_data
    
    def _generate_garch_data(self, garch_params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Generate synthetic data using GARCH model.
        """
        if not HAS_ARCH:
            # Fallback to simple volatility model
            return self._generate_simple_volatility_data()
        
        if garch_params is None:
            garch_params = {'p': 1, 'q': 1}
        
        # Fit GARCH model to returns
        try:
            model = arch_model(self.returns * 100, vol='GARCH', p=garch_params['p'], q=garch_params['q'])
            fitted_model = model.fit(disp='off')
            
            # Simulate new returns
            simulated = fitted_model.simulate(nobs=len(self.data))
            simulated_returns = simulated['data'].values / 100  # Convert back from percentage
            
        except Exception:
            # Fallback to calibrated normal model
            simulated_returns = np.random.normal(self.return_mean, self.return_std, len(self.data))
        
        # Reconstruct price data
        garch_data = self.data.copy()
        
        if 'close' in garch_data.columns:
            initial_price = garch_data['close'].iloc[0]
            new_prices = [initial_price]
            
            for ret in simulated_returns:
                new_price = new_prices[-1] * (1 + ret)
                new_prices.append(new_price)
            
            # Update all price columns proportionally
            price_ratio = np.array(new_prices[1:]) / garch_data['close'].iloc[1:].values
            
            for col in ['open', 'high', 'low', 'close']:
                if col in garch_data.columns:
                    garch_data.iloc[1:, garch_data.columns.get_loc(col)] = (
                        garch_data.iloc[1:, garch_data.columns.get_loc(col)] * price_ratio
                    )
        
        return garch_data
    
    def _generate_simple_volatility_data(self) -> pd.DataFrame:
        """
        Generate synthetic data using simple volatility model (fallback).
        """
        # Generate returns with time-varying volatility
        vol_persistence = 0.9
        vol_innovation = 0.1
        
        volatilities = [self.return_std]
        returns = []
        
        for i in range(len(self.data)):
            # Update volatility (simple GARCH(1,1) approximation)
            new_vol = (vol_persistence * volatilities[-1] + 
                      vol_innovation * abs(np.random.normal(0, self.return_std)))
            volatilities.append(new_vol)
            
            # Generate return with current volatility
            ret = np.random.normal(self.return_mean, new_vol)
            returns.append(ret)
        
        # Reconstruct prices
        synthetic_data = self.data.copy()
        
        if 'close' in synthetic_data.columns:
            initial_price = synthetic_data['close'].iloc[0]
            new_prices = [initial_price]
            
            for ret in returns:
                new_price = new_prices[-1] * (1 + ret)
                new_prices.append(new_price)
            
            # Update price columns
            price_ratio = np.array(new_prices[1:]) / synthetic_data['close'].iloc[1:].values
            
            for col in ['open', 'high', 'low', 'close']:
                if col in synthetic_data.columns:
                    synthetic_data.iloc[1:, synthetic_data.columns.get_loc(col)] = (
                        synthetic_data.iloc[1:, synthetic_data.columns.get_loc(col)] * price_ratio
                    )
        
        return synthetic_data
    
    def _generate_regime_switching_data(self, n_regimes: int = 2) -> pd.DataFrame:
        """
        Generate synthetic data using regime-switching model.
        """
        # Define regime parameters
        if n_regimes == 2:
            # Bull and bear regimes
            regime_params = [
                {'mean': self.return_mean * 2, 'std': self.return_std * 0.8},  # Bull
                {'mean': self.return_mean * -1, 'std': self.return_std * 1.5}  # Bear
            ]
            transition_matrix = np.array([[0.95, 0.05], [0.1, 0.9]])
        else:
            # Multiple regimes
            regime_params = []
            for i in range(n_regimes):
                mean_mult = np.random.uniform(-2, 3)
                std_mult = np.random.uniform(0.5, 2.0)
                regime_params.append({
                    'mean': self.return_mean * mean_mult,
                    'std': self.return_std * std_mult
                })
            
            # Random transition matrix
            transition_matrix = np.random.dirichlet([1] * n_regimes, n_regimes)
        
        # Simulate regime sequence
        current_regime = 0
        regimes = [current_regime]
        
        for _ in range(len(self.data) - 1):
            # Transition to next regime
            transition_probs = transition_matrix[current_regime]
            current_regime = np.random.choice(n_regimes, p=transition_probs)
            regimes.append(current_regime)
        
        # Generate returns based on regimes
        returns = []
        for regime in regimes:
            params = regime_params[regime]
            ret = np.random.normal(params['mean'], params['std'])
            returns.append(ret)
        
        # Reconstruct prices
        regime_data = self.data.copy()
        
        if 'close' in regime_data.columns:
            initial_price = regime_data['close'].iloc[0]
            new_prices = [initial_price]
            
            for ret in returns:
                new_price = new_prices[-1] * (1 + ret)
                new_prices.append(new_price)
            
            # Update price columns
            price_ratio = np.array(new_prices[1:]) / regime_data['close'].iloc[1:].values
            
            for col in ['open', 'high', 'low', 'close']:
                if col in regime_data.columns:
                    regime_data.iloc[1:, regime_data.columns.get_loc(col)] = (
                        regime_data.iloc[1:, regime_data.columns.get_loc(col)] * price_ratio
                    )
        
        return regime_data
    
    def _generate_block_bootstrap_data(self, block_size: int = 20) -> pd.DataFrame:
        """
        Generate synthetic data using block bootstrap.
        """
        n_samples = len(self.data)
        n_blocks = int(np.ceil(n_samples / block_size))
        
        bootstrap_data = []
        
        for _ in range(n_blocks):
            # Randomly select block start
            start_idx = np.random.randint(0, max(1, n_samples - block_size + 1))
            end_idx = min(start_idx + block_size, n_samples)
            
            block = self.data.iloc[start_idx:end_idx].copy()
            bootstrap_data.append(block)
        
        # Concatenate and truncate
        result = pd.concat(bootstrap_data, ignore_index=True)
        result = result.iloc[:n_samples]
        
        # Restore datetime index
        if isinstance(self.data.index, pd.DatetimeIndex):
            freq = pd.infer_freq(self.data.index)
            if freq:
                new_index = pd.date_range(
                    start=self.data.index.min(), 
                    periods=len(result), 
                    freq=freq
                )
                result.index = new_index
        
        return result
    
    def _run_data_mining_bias_test(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_simulations: int,
    ) -> Dict[str, Any]:
        """
        Test for data mining bias using random parameter variations.
        """
        # Generate random parameter variations
        random_results = []
        
        for i in range(n_simulations):
            try:
                # Generate random parameters within reasonable ranges
                random_params = self._generate_random_parameters(parameters)
                
                # Run backtest on original data
                metrics = backtest_function(self.data, random_params)
                
                # Add metadata
                metrics['simulation'] = i
                metrics['parameter_set'] = random_params
                
                random_results.append(metrics)
                
            except Exception as e:
                failed_metrics = {
                    'simulation': i,
                    'error': str(e),
                    'total_return': float('-inf'),
                }
                random_results.append(failed_metrics)
        
        # Analyze results for data mining bias
        valid_results = [r for r in random_results if 'error' not in r]
        
        if valid_results:
            random_returns = [r.get('total_return', 0) for r in valid_results]
            
            # Get original strategy performance
            original_result = backtest_function(self.data, parameters)
            original_return = original_result.get('total_return', 0)
            
            # Calculate percentile rank
            percentile_rank = (sum(1 for r in random_returns if r < original_return) / 
                              len(random_returns) * 100)
            
            bias_analysis = {
                'original_return': original_return,
                'random_mean': np.mean(random_returns),
                'random_std': np.std(random_returns),
                'percentile_rank': percentile_rank,
                'data_mining_bias_detected': percentile_rank > 95,  # Top 5%
                'n_random_strategies': len(valid_results),
            }
        else:
            bias_analysis = {
                'data_mining_bias_detected': False,
                'error': 'No valid random strategies',
            }
        
        return {
            'random_results': random_results,
            'bias_analysis': bias_analysis,
        }
    
    def _run_multiple_testing_analysis(
        self,
        parameters: Dict[str, Any],
        backtest_function: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        n_tests: int,
    ) -> Dict[str, Any]:
        """
        Analyze multiple testing effects on strategy performance.
        """
        # Simulate multiple strategy testing
        strategy_results = []
        
        for i in range(n_tests):
            try:
                # Generate slightly different parameters (simulating strategy variations)
                varied_params = self._vary_parameters_slightly(parameters)
                
                # Test on permuted data (null hypothesis)
                permuted_data = self._generate_permuted_data()
                metrics = backtest_function(permuted_data, varied_params)
                
                # Add metadata
                metrics['test_number'] = i
                metrics['significant'] = metrics.get('total_return', 0) > 0.05  # 5% threshold
                
                strategy_results.append(metrics)
                
            except Exception as e:
                failed_metrics = {
                    'test_number': i,
                    'error': str(e),
                    'significant': False,
                }
                strategy_results.append(failed_metrics)
        
        # Calculate multiple testing corrections
        valid_results = [r for r in strategy_results if 'error' not in r]
        
        if valid_results:
            significant_count = sum(1 for r in valid_results if r.get('significant', False))
            false_discovery_rate = significant_count / len(valid_results)
            
            # Bonferroni correction
            bonferroni_threshold = 0.05 / n_tests
            bonferroni_significant = sum(
                1 for r in valid_results 
                if r.get('total_return', 0) > bonferroni_threshold
            )
            
            multiple_testing_analysis = {
                'n_tests': len(valid_results),
                'significant_count': significant_count,
                'false_discovery_rate': false_discovery_rate,
                'bonferroni_threshold': bonferroni_threshold,
                'bonferroni_significant': bonferroni_significant,
                'multiple_testing_bias_detected': false_discovery_rate > 0.1,
            }
        else:
            multiple_testing_analysis = {
                'multiple_testing_bias_detected': False,
                'error': 'No valid strategy tests',
            }
        
        return {
            'strategy_results': strategy_results,
            'multiple_testing_analysis': multiple_testing_analysis,
        }
    
    def _generate_random_parameters(self, base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate random parameter variations for data mining bias test.
        """
        random_params = {}
        
        for param_name, param_value in base_parameters.items():
            if isinstance(param_value, int):
                # Vary integer parameters
                min_val = max(1, param_value // 2)
                max_val = param_value * 2
                random_params[param_name] = np.random.randint(min_val, max_val + 1)
            elif isinstance(param_value, float):
                # Vary float parameters
                min_val = param_value * 0.5
                max_val = param_value * 2.0
                random_params[param_name] = np.random.uniform(min_val, max_val)
            else:
                # Keep non-numeric parameters unchanged
                random_params[param_name] = param_value
        
        return random_params
    
    def _vary_parameters_slightly(self, base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create slight parameter variations for multiple testing analysis.
        """
        varied_params = {}
        
        for param_name, param_value in base_parameters.items():
            if isinstance(param_value, int):
                # Small integer variations
                variation = np.random.randint(-2, 3)
                varied_params[param_name] = max(1, param_value + variation)
            elif isinstance(param_value, float):
                # Small float variations (±10%)
                variation = np.random.uniform(-0.1, 0.1)
                varied_params[param_name] = param_value * (1 + variation)
            else:
                # Keep non-numeric parameters unchanged
                varied_params[param_name] = param_value
        
        return varied_params
    
    def _calculate_synthetic_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics grouped by synthetic method.
        """
        summary_stats = {}
        
        # Group results by synthetic method
        method_groups = {}
        for result in results:
            method = result.get('synthetic_method', 'unknown')
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(result)
        
        # Calculate statistics for each group
        for method, group_results in method_groups.items():
            group_stats = {}
            
            # Get all numeric metrics
            all_metrics = set()
            for result in group_results:
                for key, value in result.items():
                    if isinstance(value, (int, float)) and key not in ['simulation']:
                        all_metrics.add(key)
            
            for metric in all_metrics:
                values = [r.get(metric, 0) for r in group_results if 'error' not in r and isinstance(r.get(metric), (int, float))]
                
                if values:
                    group_stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'count': len(values),
                        'success_rate': len(values) / len(group_results),
                    }
            
            summary_stats[method] = group_stats
        
        return summary_stats
    
    def _analyze_garch_results(self, garch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze GARCH simulation results.
        """
        valid_results = [r for r in garch_results if 'error' not in r]
        
        if not valid_results:
            return {'analysis_failed': True, 'reason': 'No valid GARCH results'}
        
        returns = [r.get('total_return', 0) for r in valid_results]
        
        analysis = {
            'mean_return': np.mean(returns),
            'volatility_of_returns': np.std(returns),
            'garch_stability_score': 1.0 / (1.0 + np.std(returns)),  # Higher = more stable
            'performance_consistency': len(valid_results) / len(garch_results),
        }
        
        return analysis
    
    def _analyze_regime_results(self, regime_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze regime-switching simulation results.
        """
        valid_results = [r for r in regime_results if 'error' not in r]
        
        if not valid_results:
            return {'analysis_failed': True, 'reason': 'No valid regime results'}
        
        returns = [r.get('total_return', 0) for r in valid_results]
        
        analysis = {
            'mean_return': np.mean(returns),
            'regime_robustness_score': 1.0 / (1.0 + np.std(returns)),  # Higher = more robust
            'tail_performance': {
                'worst_5_percent': np.percentile(returns, 5),
                'best_5_percent': np.percentile(returns, 95),
            },
        }
        
        return analysis
    
    def _calculate_overfitting_score(self, detection_results: Dict[str, Any]) -> float:
        """
        Calculate aggregate overfitting score from multiple detection methods.
        """
        overfitting_signals = []
        
        # Permutation test signal
        if 'permutation' in detection_results:
            perm_result = detection_results['permutation']
            sig_test = perm_result.get('significance_test', {})
            if sig_test.get('overfitting_detected', False):
                overfitting_signals.append(0.8)
            elif sig_test.get('p_value', 0.5) > 0.8:
                overfitting_signals.append(0.6)
            else:
                overfitting_signals.append(0.2)
        
        # Data mining bias signal
        if 'data_mining_bias' in detection_results:
            dmb_result = detection_results['data_mining_bias']
            bias_analysis = dmb_result.get('bias_analysis', {})
            if bias_analysis.get('data_mining_bias_detected', False):
                overfitting_signals.append(0.9)
            elif bias_analysis.get('percentile_rank', 50) > 90:
                overfitting_signals.append(0.7)
            else:
                overfitting_signals.append(0.3)
        
        # Multiple testing signal
        if 'multiple_testing' in detection_results:
            mt_result = detection_results['multiple_testing']
            mt_analysis = mt_result.get('multiple_testing_analysis', {})
            if mt_analysis.get('multiple_testing_bias_detected', False):
                overfitting_signals.append(0.8)
            else:
                overfitting_signals.append(0.2)
        
        # Aggregate signals
        if overfitting_signals:
            return np.mean(overfitting_signals)
        else:
            return 0.5  # Neutral score if no signals available