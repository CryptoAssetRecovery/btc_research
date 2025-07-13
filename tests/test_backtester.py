"""
Unit tests for the Backtester module.

Tests the Backtester class that wraps Backtrader functionality,
including DataFrame conversion, dynamic strategy generation,
and performance statistics calculation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from btc_research.core.backtester import (
    Backtester,
    BacktesterError,
    PandasData,
    StrategyLogic,
    create_backtest_summary,
)


class TestBacktester:
    """Test the Backtester class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "version": "1.0",
            "name": "Test Strategy",
            "symbol": "BTC/USDC",
            "timeframes": {"entry": "5m"},
            "indicators": [],
            "logic": {
                "entry_long": ["close > open"],
                "exit_long": ["close < open"],
                "entry_short": [],
                "exit_short": [],
            },
            "backtest": {"cash": 10000, "commission": 0.001, "slippage": 0.0},
        }

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame with OHLCV data and indicators."""
        dates = pd.date_range(
            start="2024-01-01 00:00:00", end="2024-01-01 01:00:00", freq="5min"
        )

        # Create synthetic price data with some volatility
        np.random.seed(42)  # For reproducible tests
        prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 100)

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * (1 + np.abs(np.random.randn(len(dates)) * 0.002)),
                "low": prices * (1 - np.abs(np.random.randn(len(dates)) * 0.002)),
                "close": prices + np.random.randn(len(dates)) * 50,
                "volume": np.random.randint(100, 1000, len(dates)),
                "RSI_14": 30 + 40 * np.random.rand(len(dates)),  # Sample indicator
                "EMA_200_trend": np.random.choice(["bull", "bear"], len(dates)),
            },
            index=dates,
        )

        # Ensure high >= low and other OHLCV constraints
        df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
        df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))

        return df

    def test_init_valid_config(self, sample_config):
        """Test Backtester initialization with valid config."""
        backtester = Backtester(sample_config)
        assert backtester.config == sample_config
        assert backtester.debug is False

    def test_init_invalid_config(self):
        """Test Backtester initialization with invalid config."""
        # Missing logic section
        with pytest.raises(
            BacktesterError, match="Config must contain 'logic' section"
        ):
            Backtester({})

        # Logic not a dictionary
        with pytest.raises(BacktesterError, match="Logic section must be a dictionary"):
            Backtester({"logic": "invalid"})

        # Invalid logic expressions
        with pytest.raises(BacktesterError, match="must be a list of expressions"):
            Backtester({"logic": {"entry_long": "not_a_list"}})

    def test_create_data_feed_valid(self, sample_dataframe):
        """Test DataFrame to Backtrader feed conversion with valid data."""
        config = {"logic": {"entry_long": []}}
        backtester = Backtester(config)

        feed = backtester._create_data_feed(sample_dataframe)
        assert isinstance(feed, PandasData)

    def test_create_data_feed_invalid(self):
        """Test DataFrame conversion with invalid data."""
        config = {"logic": {"entry_long": []}}
        backtester = Backtester(config)

        # Empty DataFrame
        with pytest.raises(BacktesterError, match="DataFrame cannot be None or empty"):
            backtester._create_data_feed(pd.DataFrame())

        # Missing required columns
        df = pd.DataFrame({"price": [1, 2, 3]})
        with pytest.raises(BacktesterError, match="DataFrame missing required columns"):
            backtester._create_data_feed(df)

        # Non-datetime index
        df = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [1, 2, 3],
                "low": [1, 2, 3],
                "close": [1, 2, 3],
                "volume": [100, 200, 300],
            }
        )
        with pytest.raises(BacktesterError, match="DataFrame must have DatetimeIndex"):
            backtester._create_data_feed(df)

    def test_run_basic_backtest(self, sample_config, sample_dataframe):
        """Test basic backtest execution."""
        backtester = Backtester(sample_config)

        stats = backtester.run(sample_dataframe, cash=10000, commission=0.001)

        # Check that all required statistics are present
        required_keys = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "num_trades",
            "win_rate",
            "final_value",
            "initial_cash",
            "profit_factor",
            "avg_trade",
            "equity_curve",
            "trades",
        ]

        for key in required_keys:
            assert key in stats

        # Check data types and ranges
        assert isinstance(stats["total_return"], (int, float))
        assert isinstance(stats["sharpe_ratio"], (int, float))
        assert isinstance(stats["max_drawdown"], (int, float))
        assert isinstance(stats["num_trades"], int)
        assert isinstance(stats["win_rate"], (int, float))
        assert 0 <= stats["win_rate"] <= 1
        assert stats["final_value"] > 0
        assert stats["initial_cash"] == 10000

    def test_run_with_no_trades(self, sample_dataframe):
        """Test backtest with configuration that generates no trades."""
        config = {
            "logic": {
                "entry_long": ["close > 100000"],  # Impossible condition
                "exit_long": [],
                "entry_short": [],
                "exit_short": [],
            }
        }

        backtester = Backtester(config)
        stats = backtester.run(sample_dataframe, cash=10000)

        assert stats["num_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["total_return"] == 0.0  # No trades, no return

    def test_run_with_indicator_conditions(self, sample_dataframe):
        """Test backtest with indicator-based conditions."""
        config = {
            "logic": {
                "entry_long": ["RSI_14 < 40"],
                "exit_long": ["RSI_14 > 60"],
                "entry_short": [],
                "exit_short": [],
            }
        }

        backtester = Backtester(config)
        stats = backtester.run(sample_dataframe, cash=10000)

        # Should be able to execute without errors
        assert "total_return" in stats
        assert isinstance(stats["num_trades"], int)

    def test_run_with_complex_logic(self, sample_dataframe):
        """Test backtest with multiple conditions."""
        config = {
            "logic": {
                "entry_long": ["RSI_14 < 35", "close > open"],
                "exit_long": ["RSI_14 > 65"],
                "entry_short": [],
                "exit_short": [],
            }
        }

        backtester = Backtester(config)
        stats = backtester.run(sample_dataframe, cash=10000)

        assert "total_return" in stats
        assert isinstance(stats["num_trades"], int)

    def test_run_with_invalid_dataframe(self, sample_config):
        """Test backtest with invalid DataFrame."""
        backtester = Backtester(sample_config)

        # Test with None
        with pytest.raises(BacktesterError):
            backtester.run(None)

        # Test with empty DataFrame
        with pytest.raises(BacktesterError):
            backtester.run(pd.DataFrame())


class TestStrategyLogic:
    """Test the StrategyLogic class."""

    @pytest.fixture
    def sample_strategy_config(self):
        """Sample configuration for strategy testing."""
        return {
            "logic": {
                "entry_long": ["close > open"],
                "exit_long": ["close < open"],
                "entry_short": [],
                "exit_short": [],
            }
        }

    @pytest.fixture
    def sample_strategy_dataframe(self):
        """Sample DataFrame for strategy testing."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1H")
        return pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                "close": [
                    100.5,
                    101.5,
                    102.5,
                    103.5,
                    104.5,
                    105.5,
                    106.5,
                    107.5,
                    108.5,
                    109.5,
                ],
                "volume": [1000] * 10,
            },
            index=dates,
        )

    def test_evaluate_rules_simple(
        self, sample_strategy_config, sample_strategy_dataframe
    ):
        """Test rule evaluation with simple expressions."""
        # Create strategy logic instance
        strategy_logic = StrategyLogic(
            sample_strategy_config, sample_strategy_dataframe, debug=False
        )

        # Test evaluation context
        context = {"close": 101.5, "open": 101.0}
        result = strategy_logic.evaluate_rules(["close > open"], context)
        assert result is True

        context = {"close": 100.5, "open": 101.0}
        result = strategy_logic.evaluate_rules(["close > open"], context)
        assert result is False

    def test_evaluate_rules_multiple_conditions(
        self, sample_strategy_config, sample_strategy_dataframe
    ):
        """Test rule evaluation with multiple conditions."""
        strategy_logic = StrategyLogic(
            sample_strategy_config, sample_strategy_dataframe, debug=False
        )

        # All conditions must be true
        context = {"close": 102, "open": 100, "volume": 1500}
        rules = ["close > open", "volume > 1000"]
        result = strategy_logic.evaluate_rules(rules, context)
        assert result is True

        # One condition false
        rules = ["close > open", "volume > 2000"]
        result = strategy_logic.evaluate_rules(rules, context)
        assert result is False

    def test_evaluate_rules_with_nan(
        self, sample_strategy_config, sample_strategy_dataframe
    ):
        """Test rule evaluation with NaN values."""
        strategy_logic = StrategyLogic(
            sample_strategy_config, sample_strategy_dataframe, debug=False
        )

        # Context with NaN should return False
        context = {"close": None, "open": 100}
        result = strategy_logic.evaluate_rules(["close > open"], context)
        assert result is False

    def test_evaluate_rules_invalid_expression(
        self, sample_strategy_config, sample_strategy_dataframe
    ):
        """Test rule evaluation with invalid expressions."""
        strategy_logic = StrategyLogic(
            sample_strategy_config, sample_strategy_dataframe, debug=False
        )

        # Invalid syntax should return False
        context = {"close": 101, "open": 100}
        result = strategy_logic.evaluate_rules(["invalid syntax"], context)
        assert result is False


class TestBacktesterIntegration:
    """Integration tests using demo.yaml configuration."""

    def test_demo_config_structure(self):
        """Test that demo.yaml has the expected structure."""
        demo_path = (
            Path(__file__).parent.parent / "btc_research" / "config" / "demo.yaml"
        )

        if demo_path.exists():
            with open(demo_path) as f:
                config = yaml.safe_load(f)

            # Check required sections
            assert "logic" in config
            assert "backtest" in config

            # Check logic structure
            logic = config["logic"]
            assert isinstance(logic.get("entry_long", []), list)
            assert isinstance(logic.get("exit_long", []), list)
            assert isinstance(logic.get("entry_short", []), list)
            assert isinstance(logic.get("exit_short", []), list)

    def test_backtester_with_demo_config(self):
        """Test Backtester initialization with demo.yaml config."""
        demo_path = (
            Path(__file__).parent.parent / "btc_research" / "config" / "demo.yaml"
        )

        if demo_path.exists():
            with open(demo_path) as f:
                config = yaml.safe_load(f)

            # Should not raise an exception
            backtester = Backtester(config)
            assert backtester.config == config

    def test_backtest_summary_formatting(self):
        """Test backtest summary formatting."""
        stats = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
            "num_trades": 25,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "avg_trade": 125.50,
            "final_value": 11500.0,
            "initial_cash": 10000.0,
        }

        summary = create_backtest_summary(stats)

        # Check that all key metrics are in the summary
        assert "Total Return:     15.00%" in summary
        assert "Sharpe Ratio:     1.20" in summary
        assert "Max Drawdown:     8.00%" in summary
        assert "Number of Trades: 25" in summary
        assert "Win Rate:         60.00%" in summary
        assert "Profit Factor:    1.80" in summary
        assert "$125.50" in summary
        assert "$11500.00" in summary
        assert "$10000.00" in summary


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_backtester_with_malformed_config(self):
        """Test Backtester with various malformed configurations."""
        # Non-dictionary config
        with pytest.raises(BacktesterError, match="Config must be a dictionary"):
            Backtester("not a dict")

        # Missing logic section
        with pytest.raises(
            BacktesterError, match="Config must contain 'logic' section"
        ):
            Backtester({})

        # Logic not a dictionary
        with pytest.raises(BacktesterError, match="Logic section must be a dictionary"):
            Backtester({"logic": "not a dict"})

    def test_strategy_logic_initialization(self):
        """Test StrategyLogic initialization."""
        # Valid initialization
        config = {"logic": {"entry_long": ["close > open"]}}
        df = pd.DataFrame({"close": [1, 2], "open": [0.5, 1.5]})

        strategy_logic = StrategyLogic(config, df, debug=False)
        assert strategy_logic.config == config
        assert len(strategy_logic.entry_long_rules) == 1
        assert strategy_logic.entry_long_rules[0] == "close > open"

    def test_data_feed_edge_cases(self):
        """Test edge cases in data feed creation."""
        config = {"logic": {"entry_long": []}}
        backtester = Backtester(config)

        # DataFrame with NaN values in OHLCV
        dates = pd.date_range("2024-01-01", periods=5, freq="1H")
        df_with_nan = pd.DataFrame(
            {
                "open": [100, np.nan, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )

        # Should handle NaN values by dropping rows
        with pytest.warns(UserWarning, match="Removed .* rows with NaN OHLCV data"):
            feed = backtester._create_data_feed(df_with_nan)
            assert isinstance(feed, PandasData)

        # DataFrame with all NaN should raise error
        df_all_nan = pd.DataFrame(
            {
                "open": [np.nan] * 5,
                "high": [np.nan] * 5,
                "low": [np.nan] * 5,
                "close": [np.nan] * 5,
                "volume": [np.nan] * 5,
            },
            index=dates,
        )

        with pytest.raises(BacktesterError, match="No valid OHLCV data after cleaning"):
            backtester._create_data_feed(df_all_nan)


if __name__ == "__main__":
    pytest.main([__file__])
