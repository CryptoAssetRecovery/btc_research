# BTC Research Engine

A modular research harness for Bitcoin trading strategies with multi-timeframe indicator support.

## Features

- **Modular Architecture**: Clean separation between framework code and strategy-specific implementations
- **Plugin System**: Register indicators once and reuse them everywhere
- **Multi-Timeframe Support**: Mix any timeframes in a single YAML configuration
- **YAML-Driven Experiments**: Configure strategies without writing code
- **Comprehensive Testing**: Pre-commit hooks, CI/CD, and coverage reporting
- **Production Ready**: Type hints, linting, formatting, and security checks

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd btc_research
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up pre-commit hooks:
```bash
poetry run pre-commit install
```

### Usage

The BTC Research Engine provides three main CLI commands:

```bash
# Download and cache market data
poetry run btc-download config/strategy.yaml

# Run a backtest
poetry run btc-backtest config/strategy.yaml --plot

# Optimize strategy parameters
poetry run btc-optimise config/strategy.yaml --grid rsi_length=10,14,21
```

## Development

### Project Structure

```
btc_research/
├── pyproject.toml          # Poetry configuration and tool settings
├── btc_research/
│   ├── core/               # Framework code (rarely changes)
│   │   ├── registry.py     # Indicator plugin registry
│   │   ├── datafeed.py     # Data fetching and caching
│   │   ├── engine.py       # Multi-timeframe confluence engine
│   │   ├── backtester.py   # Backtrader wrapper
│   │   └── schema.py       # YAML configuration validation
│   ├── indicators/         # Strategy-specific indicators
│   ├── cli/                # Command-line interfaces
│   └── config/             # YAML strategy templates
└── tests/                  # Test suite
```

### Code Quality

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **Pytest**: Testing framework with coverage
- **Pre-commit**: Git hooks for quality gates
- **Bandit**: Security linting
- **Safety**: Dependency vulnerability scanning

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=btc_research --cov-report=html

# Run linting
poetry run ruff check .
poetry run black --check .
poetry run mypy .
```

## Architecture

The BTC Research Engine follows a plugin-based architecture where:

1. **Core Framework** (`btc_research.core`) provides stable, reusable components
2. **Indicators** (`btc_research.indicators`) are pluggable strategy components
3. **Configuration** drives behavior through YAML files
4. **CLI Tools** provide simple interfaces for common tasks

### Adding New Indicators

Create a new indicator by implementing the `Indicator` interface:

```python
from btc_research.core.registry import register
from btc_research.core.base_indicator import Indicator

@register("MyIndicator")
class MyIndicator(Indicator):
    @classmethod
    def params(cls):
        return {"length": 14}
    
    def __init__(self, length=14):
        self.length = length
    
    def compute(self, df):
        # Implement your indicator logic
        pass
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

All contributions must pass the CI pipeline including:
- Code formatting (Black)
- Linting (Ruff)
- Type checking (MyPy)
- Tests (Pytest with 80%+ coverage)
- Security checks (Bandit, Safety)