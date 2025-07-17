# BTC Research Optimization Framework - API Reference

## Overview

The BTC Research Optimization Framework provides a comprehensive set of tools for optimizing trading strategy parameters using modern optimization techniques. This API reference covers all public classes, methods, and functions available in the framework.

## Quick Start

```python
from btc_research.optimization import (
    optimize_strategy,
    BayesianOptimizer,
    WalkForwardValidator,
    MonteCarloRobustnessTest,
    ParameterSpec,
    ParameterType,
    OptimizationMetric
)

# Define parameters to optimize
parameter_specs = [
    ParameterSpec("rsi_period", ParameterType.INTEGER, low=10, high=30),
    ParameterSpec("rsi_oversold", ParameterType.FLOAT, low=20.0, high=40.0)
]

# Run optimization
result = optimize_strategy(
    config_path="strategy_config.yaml",
    parameter_specs=parameter_specs,
    optimizer=BayesianOptimizer,
    validator=WalkForwardValidator,
    max_iterations=100
)

print(f"Best parameters: {result.best_parameters}")
print(f"Best score: {result.best_score}")
```

## Core Components

### 1. Optimization Algorithms

#### BaseOptimizer

Abstract base class for all optimization algorithms.

```python
class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    def __init__(
        self,
        parameter_specs: List[ParameterSpec],
        objective_function: ObjectiveFunction,
        metric: OptimizationMetric,
        maximize: bool = True,
        random_seed: Optional[int] = None
    )
```

**Parameters:**
- `parameter_specs`: List of parameter specifications defining search space
- `objective_function`: Function to optimize (takes params dict, returns score)
- `metric`: Primary optimization metric
- `maximize`: Whether to maximize (True) or minimize (False) the objective
- `random_seed`: Random seed for reproducibility

**Key Methods:**
- `optimize(max_iterations, timeout_seconds, convergence_threshold, **kwargs) -> OptimizationResult`
- `suggest_parameters() -> Dict[str, Any]`
- `evaluate_parameters(parameters) -> float`

#### GridSearchOptimizer

Exhaustive grid search optimization.

```python
from btc_research.optimization.optimizers import GridSearchOptimizer

optimizer = GridSearchOptimizer(
    parameter_specs=parameter_specs,
    objective_function=objective_function,
    metric=OptimizationMetric.SHARPE_RATIO
)

result = optimizer.optimize()
```

**Features:**
- Exhaustive evaluation of all parameter combinations
- Guaranteed to find global optimum within search space
- Best for small parameter spaces (< 10,000 combinations)

#### RandomSearchOptimizer

Random search with various sampling strategies.

```python
from btc_research.optimization.optimizers import (
    RandomSearchOptimizer,
    SamplingStrategy
)

optimizer = RandomSearchOptimizer(
    parameter_specs=parameter_specs,
    objective_function=objective_function,
    metric=OptimizationMetric.SHARPE_RATIO,
    sampling_strategy=SamplingStrategy.LATIN_HYPERCUBE,
    random_seed=42
)

result = optimizer.optimize(max_iterations=100)
```

**Sampling Strategies:**
- `UNIFORM`: Standard uniform random sampling
- `LATIN_HYPERCUBE`: Latin Hypercube sampling for better space coverage
- `SOBOL`: Sobol sequence for quasi-random sampling
- `HALTON`: Halton sequence for low-discrepancy sampling

#### BayesianOptimizer

Bayesian optimization using Gaussian processes.

```python
from btc_research.optimization.optimizers import (
    BayesianOptimizer,
    AcquisitionFunction
)

optimizer = BayesianOptimizer(
    parameter_specs=parameter_specs,
    objective_function=objective_function,
    metric=OptimizationMetric.SHARPE_RATIO,
    acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
    n_initial_points=10,
    xi=0.01,  # Exploration parameter
    kappa=2.576,  # Confidence parameter for UCB
    random_seed=42
)

result = optimizer.optimize(max_iterations=50)
```

**Acquisition Functions:**
- `EXPECTED_IMPROVEMENT`: Expected improvement over current best
- `UPPER_CONFIDENCE_BOUND`: Upper confidence bound
- `PROBABILITY_OF_IMPROVEMENT`: Probability of improvement

#### GeneticAlgorithmOptimizer

Genetic algorithm optimization with customizable operators.

```python
from btc_research.optimization.optimizers import (
    GeneticAlgorithmOptimizer,
    SelectionMethod,
    CrossoverMethod,
    MutationMethod
)

optimizer = GeneticAlgorithmOptimizer(
    parameter_specs=parameter_specs,
    objective_function=objective_function,
    metric=OptimizationMetric.SHARPE_RATIO,
    population_size=50,
    selection_method=SelectionMethod.TOURNAMENT,
    crossover_method=CrossoverMethod.UNIFORM,
    mutation_method=MutationMethod.GAUSSIAN,
    crossover_probability=0.8,
    mutation_probability=0.1,
    tournament_size=3,
    elite_size=2,
    random_seed=42
)

result = optimizer.optimize(max_iterations=30)  # 30 generations
```

### 2. Validation Strategies

#### BaseValidator

Abstract base class for validation strategies.

```python
class BaseValidator(ABC):
    """Abstract base class for validation strategies."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        date_column: str = "timestamp",
        random_seed: Optional[int] = None
    )
```

#### WalkForwardValidator

Walk-forward validation for time series data.

```python
from btc_research.optimization.validators import WalkForwardValidator

validator = WalkForwardValidator(
    data=market_data,
    window_size=252,  # Training window size (1 year)
    step_size=21,     # Step size (1 month)
    min_train_size=100,
    max_train_size=500,
    random_seed=42
)

validation_result = validator.validate(
    parameters={"rsi_period": 14},
    backtest_function=backtest_function
)
```

#### TimeSeriesSplitValidator

Time series cross-validation with configurable splits.

```python
from btc_research.optimization.validators import TimeSeriesSplitValidator

validator = TimeSeriesSplitValidator(
    data=market_data,
    n_splits=5,
    test_size=0.2,  # 20% for testing
    gap=24,         # 24-hour gap between train and test
    expanding_window=True,
    random_seed=42
)
```

#### PurgedCrossValidator

Purged cross-validation to prevent data leakage.

```python
from btc_research.optimization.validators import PurgedCrossValidator

validator = PurgedCrossValidator(
    data=market_data,
    n_splits=5,
    purge_length=24,    # Remove 24 periods around test set
    embargo_length=12,  # 12-period embargo after training
    shuffle=False,
    random_seed=42
)
```

### 3. Robustness Testing

#### MonteCarloRobustnessTest

Monte Carlo robustness testing with data perturbation.

```python
from btc_research.optimization.robustness import MonteCarloRobustnessTest

robustness_test = MonteCarloRobustnessTest(
    data=market_data,
    perturbation_methods=["price_noise", "bootstrap", "volume_noise"],
    noise_level=0.01,  # 1% noise level
    random_seed=42
)

robustness_result = robustness_test.run_test(
    parameters={"rsi_period": 14},
    backtest_function=backtest_function,
    n_simulations=1000
)

print(f"Mean return: {robustness_result.mean_metrics['total_return']}")
print(f"VaR 95%: {robustness_result.var_metrics['total_return_0.95']}")
```

#### BootstrapRobustnessTest

Bootstrap-based robustness testing.

```python
from btc_research.optimization.robustness import BootstrapRobustnessTest

robustness_test = BootstrapRobustnessTest(
    data=market_data,
    n_bootstrap_samples=1000,
    sample_size_ratio=0.8,
    random_seed=42
)
```

#### ParameterSensitivityTest

Parameter sensitivity analysis.

```python
from btc_research.optimization.robustness import ParameterSensitivityTest

sensitivity_test = ParameterSensitivityTest(
    data=market_data,
    sensitivity_levels=[0.01, 0.05, 0.1],  # 1%, 5%, 10% perturbation
    random_seed=42
)
```

### 4. Statistical Testing

#### Statistical Test Classes

```python
from btc_research.optimization.statistics import (
    TTestStatistics,
    WilcoxonTestStatistics,
    SharpeRatioTestStatistics,
    DrawdownTestStatistics
)

# T-test for mean comparison
t_test = TTestStatistics(confidence_level=0.95)
result = t_test.run_test(
    sample1=strategy_returns,
    sample2=benchmark_returns
)

# Sharpe ratio significance test
sharpe_test = SharpeRatioTestStatistics(confidence_level=0.95)
result = sharpe_test.run_test(sample1=strategy_returns)
```

### 5. Data Types

#### ParameterSpec

Specification for optimization parameters.

```python
from btc_research.optimization.types import ParameterSpec, ParameterType

# Integer parameter
int_param = ParameterSpec(
    name="rsi_period",
    param_type=ParameterType.INTEGER,
    low=10,
    high=30,
    step=1
)

# Float parameter
float_param = ParameterSpec(
    name="threshold",
    param_type=ParameterType.FLOAT,
    low=0.0,
    high=1.0,
    step=0.01
)

# Categorical parameter
cat_param = ParameterSpec(
    name="method",
    param_type=ParameterType.CATEGORICAL,
    choices=["sma", "ema", "wma"]
)

# Boolean parameter
bool_param = ParameterSpec(
    name="enable_feature",
    param_type=ParameterType.BOOLEAN
)
```

#### OptimizationResult

Result object returned by optimization algorithms.

```python
@dataclass
class OptimizationResult:
    best_parameters: Dict[str, Any]
    best_score: float
    total_evaluations: int
    optimization_time: float
    convergence_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
```

#### ValidationResult

Result object returned by validation strategies.

```python
@dataclass
class ValidationResult:
    parameters: Dict[str, Any]
    fold_results: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    metadata: Dict[str, Any]
```

#### RobustnessResult

Result object returned by robustness tests.

```python
@dataclass
class RobustnessResult:
    parameters: Dict[str, Any]
    simulation_results: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    var_metrics: Dict[str, float]
    es_metrics: Dict[str, float]
    metadata: Dict[str, Any]
```

### 6. Integration Functions

#### optimize_strategy

High-level function for strategy optimization.

```python
def optimize_strategy(
    config_path: str,
    parameter_specs: List[ParameterSpec],
    optimizer: Type[BaseOptimizer] = BayesianOptimizer,
    validator: Optional[Type[BaseValidator]] = None,
    robustness_test: Optional[Type[BaseRobustnessTest]] = None,
    max_iterations: int = 100,
    metric: OptimizationMetric = OptimizationMetric.SHARPE_RATIO,
    **kwargs
) -> OptimizationResult
```

#### OptimizationFramework

Complete optimization framework with all components.

```python
from btc_research.optimization import OptimizationFramework

framework = OptimizationFramework(
    data=market_data,
    optimizer_class=BayesianOptimizer,
    validator_class=WalkForwardValidator,
    robustness_test_class=MonteCarloRobustnessTest
)

# Run complete optimization workflow
optimization_result = framework.optimize(
    parameter_specs=parameter_specs,
    max_iterations=50
)

validation_result = framework.validate(
    parameters=optimization_result.best_parameters,
    backtest_function=backtest_function
)

robustness_result = framework.test_robustness(
    parameters=optimization_result.best_parameters,
    backtest_function=backtest_function,
    n_simulations=100
)
```

## Enumerations

### OptimizationMetric

```python
class OptimizationMetric(Enum):
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    ANNUAL_RETURN = "annual_return"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
```

### ParameterType

```python
class ParameterType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
```

### ValidationMethod

```python
class ValidationMethod(Enum):
    WALK_FORWARD = "walk_forward"
    TIME_SERIES_SPLIT = "time_series_split"
    PURGED_CV = "purged_cv"
```

## Error Handling

The framework provides comprehensive error handling:

```python
from btc_research.optimization.exceptions import (
    OptimizationError,
    ValidationError,
    ParameterError,
    ConvergenceError
)

try:
    result = optimize_strategy(
        config_path="config.yaml",
        parameter_specs=parameter_specs,
        max_iterations=100
    )
except ParameterError as e:
    print(f"Parameter specification error: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except OptimizationError as e:
    print(f"Optimization error: {e}")
```

## Performance Considerations

### Memory Usage

- Use `max_train_size` in validators to limit memory usage
- Consider using `step_size` to reduce number of validation folds
- Use smaller `n_simulations` for robustness tests during development

### Computation Time

- Bayesian optimization is most efficient for expensive objective functions
- Grid search is best for small parameter spaces
- Use `timeout_seconds` to prevent long-running optimizations
- Enable `convergence_threshold` for early stopping

### Parallelization

```python
# Future support for parallel evaluation
optimizer = BayesianOptimizer(
    parameter_specs=parameter_specs,
    objective_function=objective_function,
    metric=OptimizationMetric.SHARPE_RATIO,
    n_jobs=4  # Use 4 parallel workers
)
```

## Best Practices

1. **Always use validation** to assess out-of-sample performance
2. **Use robustness testing** to ensure strategy stability
3. **Set random seeds** for reproducible results
4. **Monitor convergence** to avoid overfitting
5. **Use appropriate metrics** for your strategy type
6. **Validate parameter bounds** before optimization
7. **Keep optimization logs** for analysis and debugging

## CLI Integration

The optimization framework integrates seamlessly with the existing CLI:

```bash
# Run Bayesian optimization
btc-research optimize \
    --config strategy.yaml \
    --method bayesian \
    --max-iterations 100 \
    --metric sharpe_ratio \
    --validation walk_forward \
    --robustness-test

# Legacy grid search (backward compatible)
btc-research optimize \
    --config strategy.yaml \
    --method grid_search
```

## Configuration Format

```yaml
# strategy.yaml
version: "1.0"
name: "RSI Strategy"
symbol: "BTC/USD"
timeframes:
  entry: "1h"

indicators:
  - id: "RSI_14"
    type: "RSI"
    timeframe: "1h"
    length: 14

logic:
  entry_long: ["RSI_14 < 30"]
  exit_long: ["RSI_14 > 70"]

backtest:
  cash: 10000
  commission: 0.001
  from: "2024-01-01"
  to: "2024-12-31"

optimization:
  method: "bayesian"
  max_iterations: 100
  metric: "sharpe_ratio"
  
  parameters:
    - name: "rsi_period"
      type: "integer"
      low: 10
      high: 30
    
    - name: "rsi_oversold"
      type: "float"
      low: 20.0
      high: 40.0
  
  validation:
    method: "walk_forward"
    window_size: 252
    step_size: 21
  
  robustness_test:
    enabled: true
    n_simulations: 1000
    perturbation_methods: ["price_noise", "bootstrap"]
```

For more detailed examples and tutorials, see the [User Guide](../user_guide/README.md) and [Examples](../examples/README.md) sections.