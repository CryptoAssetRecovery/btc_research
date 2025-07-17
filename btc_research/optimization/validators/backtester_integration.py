"""
Integration utilities for connecting validation frameworks with backtester.

This module provides seamless integration between the validation framework
and the existing backtester, engine, and datafeed components.
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from btc_research.core.backtester import Backtester
from btc_research.core.engine import Engine
from btc_research.core.datafeed import DataFeed
from btc_research.optimization.base import BaseValidator
from btc_research.optimization.types import ValidationResult
from btc_research.optimization.validators.validation_metrics import ValidationSummaryGenerator

# Set up logging
logger = logging.getLogger(__name__)

__all__ = [
    "BacktesterIntegrator",
    "ValidationPipeline",
    "OptimizationValidationWrapper",
    "ValidationBacktestFunction",
]


class ValidationBacktestFunction:
    """
    Wrapper to create backtest functions compatible with validation frameworks.
    
    This class creates standardized backtest functions that can be used
    across different validation methods while maintaining compatibility
    with the existing backtester infrastructure.
    """
    
    def __init__(
        self,
        strategy_config: Dict[str, Any],
        backtester_config: Optional[Dict[str, Any]] = None,
        engine: Optional[Engine] = None,
        use_existing_indicators: bool = True,
    ):
        """
        Initialize validation backtest function.
        
        Args:
            strategy_config: Base strategy configuration
            backtester_config: Backtester configuration
            engine: Pre-configured engine instance
            use_existing_indicators: Whether to reuse pre-calculated indicators
        """
        self.strategy_config = strategy_config.copy()
        self.backtester_config = backtester_config or {}
        self.engine = engine
        self.use_existing_indicators = use_existing_indicators
        
        # Thread safety for parallel validation
        self._lock = threading.RLock()
        
        # Cache for indicator calculations
        self._indicator_cache: Dict[str, pd.DataFrame] = {}
    
    def __call__(
        self, 
        data: pd.DataFrame, 
        parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Execute backtest with given data and parameters.
        
        Args:
            data: Time series data for backtesting
            parameters: Strategy parameters to test
            
        Returns:
            Dictionary of performance metrics
        """
        with self._lock:
            try:
                # Create updated strategy config with new parameters
                config = self._create_strategy_config(parameters)
                
                # Prepare data with indicators if needed
                processed_data = self._prepare_data(data, config)
                
                # Run backtest
                backtester = Backtester(config)
                result = backtester.run(processed_data)
                
                # Extract standardized metrics
                metrics = self._extract_metrics(result)
                
                return metrics
                
            except Exception as e:
                logger.error(f"Backtest failed: {e}")
                # Return default poor performance metrics
                return self._get_default_metrics()
    
    def _create_strategy_config(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create strategy configuration with updated parameters."""
        config = self.strategy_config.copy()
        
        # Update strategy parameters
        if "strategy" not in config:
            config["strategy"] = {}
        
        # Merge parameters into strategy config
        for key, value in parameters.items():
            # Handle nested parameter paths (e.g., "rsi.period")
            if "." in key:
                parts = key.split(".")
                current = config["strategy"]
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config["strategy"][key] = value
        
        return config
    
    def _prepare_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Prepare data with required indicators."""
        if self.engine and self.use_existing_indicators:
            # Use existing engine to calculate indicators
            try:
                # Create a copy of the engine with new config
                temp_engine = Engine(config)
                processed_data = temp_engine.run(data)
                return processed_data
            except Exception as e:
                logger.warning(f"Failed to use engine for indicator calculation: {e}")
                # Fall back to original data
                return data
        else:
            # Return data as-is if no engine or indicators disabled
            return data
    
    def _extract_metrics(self, backtest_result) -> Dict[str, float]:
        """Extract standardized metrics from backtest result."""
        # This assumes the backtester returns some result object
        # Adapt based on actual backtester implementation
        try:
            if hasattr(backtest_result, 'stats'):
                stats = backtest_result.stats
            elif isinstance(backtest_result, dict):
                stats = backtest_result
            else:
                # Try to extract metrics from the result
                stats = {}
            
            # Map to standardized metric names
            metrics = {
                "total_return": self._safe_get_metric(stats, ["Total Return", "total_return", "return"], 0.0),
                "sharpe_ratio": self._safe_get_metric(stats, ["Sharpe Ratio", "sharpe_ratio", "sharpe"], 0.0),
                "max_drawdown": self._safe_get_metric(stats, ["Max Drawdown", "max_drawdown", "mdd"], 0.0),
                "win_rate": self._safe_get_metric(stats, ["Win Rate", "win_rate", "win_pct"], 0.0),
                "profit_factor": self._safe_get_metric(stats, ["Profit Factor", "profit_factor"], 1.0),
                "num_trades": self._safe_get_metric(stats, ["# Trades", "num_trades", "trades"], 0),
                "avg_trade": self._safe_get_metric(stats, ["Avg Trade", "avg_trade"], 0.0),
                "volatility": self._safe_get_metric(stats, ["Volatility", "volatility", "vol"], 0.0),
            }
            
            # Calculate additional derived metrics
            if metrics["volatility"] > 0:
                metrics["calmar_ratio"] = abs(metrics["total_return"] / metrics["max_drawdown"]) if metrics["max_drawdown"] != 0 else 0
                metrics["sortino_ratio"] = metrics["total_return"] / metrics["volatility"]  # Simplified
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to extract metrics: {e}")
            return self._get_default_metrics()
    
    def _safe_get_metric(self, stats: Dict, keys: List[str], default: float) -> float:
        """Safely get metric value from stats dictionary."""
        for key in keys:
            if key in stats:
                value = stats[key]
                if isinstance(value, (int, float)):
                    return float(value)
                elif hasattr(value, '__float__'):
                    try:
                        return float(value)
                    except:
                        continue
        return default
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics for failed backtests."""
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": -1.0,  # Very poor
            "win_rate": 0.0,
            "profit_factor": 1.0,
            "num_trades": 0,
            "avg_trade": 0.0,
            "volatility": 0.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
        }


class BacktesterIntegrator:
    """
    Main integration class for connecting validators with backtester.
    
    Provides high-level interface for running validation with existing
    backtester infrastructure.
    """
    
    def __init__(
        self,
        datafeed: Optional[DataFeed] = None,
        engine: Optional[Engine] = None,
        default_backtester_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize backtester integrator.
        
        Args:
            datafeed: DataFeed instance for data retrieval
            engine: Engine instance for indicator calculation
            default_backtester_config: Default backtester configuration
        """
        self.datafeed = datafeed
        self.engine = engine
        self.default_backtester_config = default_backtester_config or {}
        
        # Summary generator for comprehensive reports
        self.summary_generator = ValidationSummaryGenerator()
    
    def create_backtest_function(
        self, 
        strategy_config: Dict[str, Any],
        backtester_config: Optional[Dict[str, Any]] = None,
    ) -> ValidationBacktestFunction:
        """
        Create a backtest function compatible with validators.
        
        Args:
            strategy_config: Strategy configuration
            backtester_config: Backtester configuration override
            
        Returns:
            Configured backtest function
        """
        config = self.default_backtester_config.copy()
        if backtester_config:
            config.update(backtester_config)
        
        return ValidationBacktestFunction(
            strategy_config=strategy_config,
            backtester_config=config,
            engine=self.engine,
        )
    
    def run_validation(
        self,
        validator: BaseValidator,
        strategy_config: Dict[str, Any],
        parameters: Dict[str, Any],
        backtester_config: Optional[Dict[str, Any]] = None,
        generate_summary: bool = True,
    ) -> Tuple[ValidationResult, Optional[Dict[str, Any]]]:
        """
        Run validation using specified validator and configuration.
        
        Args:
            validator: Validation strategy to use
            strategy_config: Base strategy configuration
            parameters: Parameters to validate
            backtester_config: Backtester configuration
            generate_summary: Whether to generate comprehensive summary
            
        Returns:
            Tuple of (ValidationResult, Optional[Summary])
        """
        # Create backtest function
        backtest_function = self.create_backtest_function(
            strategy_config, backtester_config
        )
        
        # Run validation
        logger.info(f"Running validation with {validator.__class__.__name__}")
        validation_result = validator.validate(parameters, backtest_function)
        
        # Generate summary if requested
        summary = None
        if generate_summary:
            logger.info("Generating comprehensive validation summary")
            summary = self.summary_generator.generate_comprehensive_summary(validation_result)
        
        return validation_result, summary
    
    def get_data_for_validation(
        self,
        symbol: str,
        timeframe: str,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        source: str = "binanceus",
    ) -> pd.DataFrame:
        """
        Get data for validation using the datafeed.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start: Start date/time
            end: End date/time
            source: Data source
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.datafeed is None:
            raise ValueError("DataFeed not configured")
        
        logger.info(f"Fetching data for {symbol} {timeframe} from {start} to {end}")
        data = self.datafeed.get(symbol, timeframe, start, end, source)
        
        if data.empty:
            raise ValueError(f"No data retrieved for {symbol}")
        
        logger.info(f"Retrieved {len(data)} samples for validation")
        return data


class ValidationPipeline:
    """
    Comprehensive validation pipeline for strategy optimization.
    
    Orchestrates the entire validation process from data retrieval
    to final reporting.
    """
    
    def __init__(
        self,
        integrator: BacktesterIntegrator,
        validators: Optional[List[BaseValidator]] = None,
    ):
        """
        Initialize validation pipeline.
        
        Args:
            integrator: Backtester integrator instance
            validators: List of validators to use in pipeline
        """
        self.integrator = integrator
        self.validators = validators or []
    
    def add_validator(self, validator: BaseValidator) -> None:
        """Add validator to the pipeline."""
        self.validators.append(validator)
    
    def run_comprehensive_validation(
        self,
        strategy_config: Dict[str, Any],
        parameters: Dict[str, Any],
        data: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        source: str = "binanceus",
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation using all configured validators.
        
        Args:
            strategy_config: Strategy configuration
            parameters: Parameters to validate
            data: Pre-loaded data (optional)
            symbol: Symbol for data retrieval (if data not provided)
            timeframe: Timeframe for data retrieval
            start: Start date for data retrieval
            end: End date for data retrieval
            source: Data source
            
        Returns:
            Comprehensive validation report
        """
        # Get data if not provided
        if data is None:
            if not all([symbol, timeframe, start, end]):
                raise ValueError("Either data or symbol/timeframe/start/end must be provided")
            data = self.integrator.get_data_for_validation(symbol, timeframe, start, end, source)
        
        # Initialize each validator with the data
        validation_results = {}
        summaries = {}
        
        for validator in self.validators:
            validator_name = validator.__class__.__name__
            logger.info(f"Running {validator_name}")
            
            try:
                # Update validator data
                validator.data = data
                
                # Run validation
                result, summary = self.integrator.run_validation(
                    validator, strategy_config, parameters
                )
                
                validation_results[validator_name] = result
                summaries[validator_name] = summary
                
            except Exception as e:
                logger.error(f"Validation failed for {validator_name}: {e}")
                continue
        
        # Generate combined report
        combined_report = self._generate_combined_report(
            validation_results, summaries, strategy_config, parameters
        )
        
        return combined_report
    
    def _generate_combined_report(
        self,
        validation_results: Dict[str, ValidationResult],
        summaries: Dict[str, Optional[Dict[str, Any]]],
        strategy_config: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate combined validation report."""
        if not validation_results:
            return {"error": "No successful validations"}
        
        # Aggregate metrics across validators
        aggregated_metrics = self._aggregate_metrics(validation_results)
        
        # Determine overall consensus
        consensus = self._determine_consensus(validation_results, summaries)
        
        # Generate final recommendations
        final_recommendations = self._generate_final_recommendations(summaries, consensus)
        
        combined_report = {
            "strategy_config": strategy_config,
            "parameters_tested": parameters,
            "validation_methods": list(validation_results.keys()),
            "aggregated_metrics": aggregated_metrics,
            "individual_results": {
                name: {
                    "validation_result": result,
                    "summary": summaries.get(name),
                }
                for name, result in validation_results.items()
            },
            "consensus_analysis": consensus,
            "final_recommendations": final_recommendations,
            "overall_assessment": self._generate_overall_assessment(consensus, aggregated_metrics),
        }
        
        return combined_report
    
    def _aggregate_metrics(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Aggregate metrics across validation methods."""
        all_metrics = {}
        
        # Collect all metrics
        for name, result in validation_results.items():
            for metric, value in result.mean_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate aggregate statistics
        aggregated = {}
        for metric, values in all_metrics.items():
            if values:
                aggregated[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "consensus": len(values),  # Number of validators that measured this metric
                }
        
        return aggregated
    
    def _determine_consensus(
        self, 
        validation_results: Dict[str, ValidationResult],
        summaries: Dict[str, Optional[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Determine consensus across validation methods."""
        # Collect overall scores and assessments
        scores = []
        risk_levels = []
        stability_scores = []
        
        for name, result in validation_results.items():
            stability_scores.append(result.stability_score)
            
            summary = summaries.get(name)
            if summary:
                # Extract overall score if available
                overall_score = (summary.get("overall_validation_score", {})
                               .get("overall_score", 0))
                scores.append(overall_score)
                
                # Extract overfitting risk
                overfitting_risk = (summary.get("overfitting_analysis", {})
                                  .get("overfitting_risk", "medium"))
                risk_levels.append(overfitting_risk)
        
        # Calculate consensus metrics
        consensus = {
            "num_validators": len(validation_results),
            "mean_stability_score": float(np.mean(stability_scores)) if stability_scores else 0,
            "agreement_level": self._calculate_agreement_level(stability_scores),
        }
        
        if scores:
            consensus["mean_overall_score"] = float(np.mean(scores))
            consensus["score_consistency"] = 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) != 0 else 0
        
        if risk_levels:
            consensus["predominant_risk_level"] = max(set(risk_levels), key=risk_levels.count)
            consensus["risk_agreement"] = risk_levels.count(consensus["predominant_risk_level"]) / len(risk_levels)
        
        return consensus
    
    def _calculate_agreement_level(self, scores: List[float]) -> str:
        """Calculate agreement level based on score consistency."""
        if len(scores) < 2:
            return "insufficient_data"
        
        cv = np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else float('inf')
        
        if cv < 0.1:
            return "high_agreement"
        elif cv < 0.25:
            return "moderate_agreement"
        else:
            return "low_agreement"
    
    def _generate_final_recommendations(
        self, 
        summaries: Dict[str, Optional[Dict[str, Any]]],
        consensus: Dict[str, Any],
    ) -> List[str]:
        """Generate final recommendations based on all validation results."""
        recommendations = set()
        
        # Collect recommendations from all summaries
        for summary in summaries.values():
            if summary and "final_recommendations" in summary:
                recommendations.update(summary["final_recommendations"])
        
        # Add consensus-based recommendations
        agreement_level = consensus.get("agreement_level", "moderate_agreement")
        mean_stability = consensus.get("mean_stability_score", 0)
        
        if agreement_level == "low_agreement":
            recommendations.add("Validation methods show low agreement - investigate parameter sensitivity")
        
        if mean_stability > 0.5:
            recommendations.add("High instability across validation methods - consider parameter re-optimization")
        
        return sorted(list(recommendations))
    
    def _generate_overall_assessment(
        self, 
        consensus: Dict[str, Any], 
        aggregated_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall assessment of validation results."""
        # Score components
        stability_component = 1 / (1 + consensus.get("mean_stability_score", 1))
        agreement_component = {"high_agreement": 1.0, "moderate_agreement": 0.7, "low_agreement": 0.3}.get(
            consensus.get("agreement_level", "moderate_agreement"), 0.7
        )
        
        # Performance component from aggregated metrics
        performance_component = 0.5  # Default neutral
        if "total_return" in aggregated_metrics:
            total_return = aggregated_metrics["total_return"]["mean"]
            if total_return > 0.1:
                performance_component = 0.8
            elif total_return > 0:
                performance_component = 0.6
            else:
                performance_component = 0.2
        
        # Overall score
        overall_score = (stability_component * 0.4 + 
                        agreement_component * 0.3 + 
                        performance_component * 0.3)
        
        # Classification
        if overall_score >= 0.8:
            classification = "excellent"
            recommendation = "Parameters are well-validated and ready for deployment"
        elif overall_score >= 0.6:
            classification = "good"
            recommendation = "Parameters show good validation characteristics with minor concerns"
        elif overall_score >= 0.4:
            classification = "fair"
            recommendation = "Parameters have validation concerns that should be addressed"
        else:
            classification = "poor"
            recommendation = "Parameters require significant improvement before deployment"
        
        return {
            "overall_score": overall_score,
            "classification": classification,
            "recommendation": recommendation,
            "component_scores": {
                "stability": stability_component,
                "agreement": agreement_component,
                "performance": performance_component,
            },
            "num_validators_used": consensus.get("num_validators", 0),
        }


class OptimizationValidationWrapper:
    """
    Wrapper for integrating validation into optimization loops.
    
    Provides a standardized interface for optimization algorithms
    to incorporate validation feedback.
    """
    
    def __init__(
        self,
        validation_pipeline: ValidationPipeline,
        validation_weight: float = 0.3,
        stability_threshold: float = 0.5,
        min_validation_score: float = 0.4,
    ):
        """
        Initialize optimization validation wrapper.
        
        Args:
            validation_pipeline: Validation pipeline to use
            validation_weight: Weight of validation score in optimization
            stability_threshold: Minimum stability score required
            min_validation_score: Minimum validation score required
        """
        self.validation_pipeline = validation_pipeline
        self.validation_weight = validation_weight
        self.stability_threshold = stability_threshold
        self.min_validation_score = min_validation_score
    
    def evaluate_with_validation(
        self,
        strategy_config: Dict[str, Any],
        parameters: Dict[str, Any],
        data: pd.DataFrame,
        primary_objective: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate parameters with validation feedback.
        
        Args:
            strategy_config: Strategy configuration
            parameters: Parameters to evaluate
            data: Validation data
            primary_objective: Primary objective function value
            
        Returns:
            Tuple of (adjusted_objective, validation_info)
        """
        try:
            # Run validation
            validation_report = self.validation_pipeline.run_comprehensive_validation(
                strategy_config, parameters, data
            )
            
            # Extract validation score
            overall_assessment = validation_report.get("overall_assessment", {})
            validation_score = overall_assessment.get("overall_score", 0)
            
            # Apply validation penalty/bonus
            if validation_score < self.min_validation_score:
                # Heavy penalty for poor validation
                adjusted_objective = primary_objective * 0.1
            else:
                # Weighted combination of primary objective and validation
                adjusted_objective = (
                    primary_objective * (1 - self.validation_weight) +
                    validation_score * self.validation_weight * primary_objective
                )
            
            validation_info = {
                "validation_score": validation_score,
                "adjustment_applied": True,
                "original_objective": primary_objective,
                "adjusted_objective": adjusted_objective,
                "validation_report": validation_report,
            }
            
        except Exception as e:
            logger.warning(f"Validation failed during optimization: {e}")
            # Fall back to primary objective without validation
            adjusted_objective = primary_objective
            validation_info = {
                "validation_score": 0,
                "adjustment_applied": False,
                "error": str(e),
                "original_objective": primary_objective,
                "adjusted_objective": adjusted_objective,
            }
        
        return adjusted_objective, validation_info