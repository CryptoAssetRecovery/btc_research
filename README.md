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
poetry run btc-download btc_research/config/fvg-strategy.yaml

# Run a backtest
poetry run btc-backtest btc_research/config/fvg-strategy.yaml --plot

# Optimize strategy parameters
poetry run btc-optimise btc_research/config/fvg-strategy.yaml --grid rsi_length=10,14,21
```

## Indicators

### Fair Value Gap (FVG) Indicator

The FVG indicator implements ICT (Inner Circle Trader) Fair Value Gap methodology for identifying market imbalances and potential trading opportunities.

#### Overview

Fair Value Gaps are price imbalances that occur when there's a gap between three consecutive candles, creating zones that often act as future support or resistance levels. These gaps represent areas where institutional trading has created inefficiencies that the market tends to revisit.

#### Features

- **3-candle gap pattern detection**: Identifies bullish and bearish gaps automatically
- **Gap filling detection**: Tracks when gaps are filled by subsequent price action
- **Support/resistance calculation**: Provides nearest gap levels for trade management
- **Configurable filtering**: Adjustable minimum gap size to filter noise
- **Historical tracking**: Maintains active gaps with configurable lookback period

#### Gap Detection Logic

**Bullish FVG (Support Zone):**
- Occurs when: `low[i+2] > high[i]` (gap between candle 1 and candle 3)
- Creates a support zone between `high[i]` and `low[i+2]`
- Filled when price trades back down into the gap

**Bearish FVG (Resistance Zone):**
- Occurs when: `high[i+2] < low[i]` (gap between candle 1 and candle 3)  
- Creates a resistance zone between `high[i+2]` and `low[i]`
- Filled when price trades back up into the gap

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_gap_pips` | float | 1.0 | Minimum gap size in price units to filter small gaps |
| `max_lookback` | int | 500 | Maximum number of historical gaps to track |

#### Available Columns

The FVG indicator provides the following output columns:

| Column | Type | Description |
|--------|------|-------------|
| `FVG_bullish_signal` | bool | True when price is touching a bullish FVG zone |
| `FVG_bearish_signal` | bool | True when price is touching a bearish FVG zone |
| `FVG_gap_filled` | bool | True when a gap was filled in this period |
| `FVG_nearest_support` | float | Price level of nearest bullish FVG below current price |
| `FVG_nearest_resistance` | float | Price level of nearest bearish FVG above current price |
| `FVG_active_bullish_gaps` | int | Count of unfilled bullish gaps |
| `FVG_active_bearish_gaps` | int | Count of unfilled bearish gaps |

#### Example Configuration

```yaml
# Optimized FVG Strategy Configuration
indicators:
  - id: "FVG_15m"
    type: "FVG"
    timeframe: "15m"
    min_gap_pips: 50.0     # Filter for significant gaps only
    max_lookback: 200      # Track last 200 candles

  - id: "EMA_50"
    type: "EMA"
    timeframe: "1h"
    length: 50

  - id: "RSI_14"
    type: "RSI"
    timeframe: "15m"
    length: 14

logic:
  entry_long:
    - "EMA_50_trend == 'bull'"                         # Trend filter
    - "FVG_15m_FVG_bullish_signal == True"            # Price at FVG zone
    - "FVG_15m_FVG_active_bullish_gaps >= 2"          # Multiple support levels
    - "RSI_14 < 40"                                   # Oversold condition
  
  exit_long:
    - "FVG_15m_FVG_gap_filled == True"                # Natural target
    - "RSI_14 > 75"                                   # Overbought exit
```

#### ICT Methodology

The FVG indicator is based on Inner Circle Trader (ICT) concepts:

1. **Market Structure**: FVGs form during strong directional moves, leaving inefficiencies
2. **Institutional Footprint**: Large orders create gaps that retail traders can exploit
3. **Retracement Targets**: Gaps act as magnets for price, providing high-probability reversal zones
4. **Time-based Fills**: Recent gaps have higher probability of being filled than older ones

#### Trading Applications

**Entry Signals:**
- Look for price returning to unfilled gaps in the direction of the higher timeframe trend
- Use multiple gap confluence for stronger signals
- Combine with momentum indicators (RSI) for entry timing

**Exit Strategies:**
- Gap filling provides natural profit targets
- Monitor gap strength (size and age) for stop placement
- Use gap count as position sizing indicator

**Risk Management:**
- Larger gaps (higher `min_gap_pips`) tend to be more reliable
- Multiple active gaps provide stronger support/resistance
- Consider gap age - newer gaps have higher fill probability

#### Performance Optimization

Based on systematic testing, optimal parameters for BTC/USDC:

- **Timeframe**: 15m entry with 1h structure bias
- **Min Gap Size**: 50 USD for BTC (filters noise effectively)
- **Gap Count Filter**: Require >= 2 active gaps for entries
- **RSI Thresholds**: 40/75 for long entries/exits, 60/25 for shorts

This configuration achieved:
- **58% Total Return** vs 7.5% unoptimized
- **0.74 Sharpe Ratio** vs 0.25 unoptimized  
- **31% Max Drawdown** vs 51% unoptimized
- **64% Win Rate** with 78 total trades

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