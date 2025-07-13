"""
Unit tests for CLI tools.

This module tests all three CLI commands (download, backtest, optimise) including
argument parsing, integration with core components, error handling, and output formats.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import yaml

# Import CLI modules
from btc_research.cli import backtest, download, optimise


class TestDownloadCLI:
    """Test the download CLI command."""

    def create_test_config(self):
        """Create a test configuration for download testing."""
        return {
            "name": "Test Strategy",
            "symbol": "BTC/USDC",
            "exchange": "binanceus",
            "timeframes": {"entry": "5m"},
            "indicators": [
                {"id": "RSI_14", "type": "RSI", "timeframe": "5m", "length": 14}
            ],
            "backtest": {"from": "2024-01-01", "to": "2024-01-02", "cash": 10000},
        }

    def test_download_main_missing_config(self, capsys):
        """Test download command with missing config file."""
        with patch("sys.argv", ["btc-download", "nonexistent.yaml"]):
            result = download.main()

        captured = capsys.readouterr()
        assert result == 1
        assert "Configuration file not found" in captured.out

    @patch("btc_research.cli.download.DataFeed")
    def test_download_main_success(self, mock_datafeed_class, capsys):
        """Test successful download execution."""
        # Create mock datafeed instance
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed

        # Mock the get method to return test data
        test_data = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [1000, 1100],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"),
        )

        mock_datafeed.get.return_value = test_data
        mock_datafeed.get_cache_stats.return_value = {
            "hits": 0,
            "misses": 1,
            "load_times": [50.0],
        }
        mock_datafeed._get_cache_path.return_value = Path("/test/cache/path")

        # Create temporary config file
        config = self.create_test_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch("sys.argv", ["btc-download", config_path]):
                result = download.main()

            captured = capsys.readouterr()
            assert result == 0
            assert "All data successfully downloaded and cached" in captured.out
            assert "2 rows downloaded" in captured.out

            # Verify DataFeed was called correctly
            mock_datafeed.get.assert_called_once_with(
                symbol="BTC/USDC",
                timeframe="5m",
                start="2024-01-01",
                end="2024-01-02",
                source="binanceus",
            )

        finally:
            os.unlink(config_path)

    @patch("btc_research.cli.download.DataFeed")
    def test_download_main_verbose(self, mock_datafeed_class, capsys):
        """Test download command with verbose output."""
        mock_datafeed = Mock()
        mock_datafeed_class.return_value = mock_datafeed

        test_data = pd.DataFrame(
            {
                "open": [100],
                "high": [102],
                "low": [99],
                "close": [101],
                "volume": [1000],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="5min", tz="UTC"),
        )

        mock_datafeed.get.return_value = test_data
        mock_datafeed.get_cache_stats.return_value = {
            "hits": 0,
            "misses": 1,
            "load_times": [],
        }
        mock_datafeed._get_cache_path.return_value = Path("/test/cache/path")

        config = self.create_test_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch("sys.argv", ["btc-download", config_path, "--verbose"]):
                result = download.main()

            captured = capsys.readouterr()
            assert result == 0
            assert "Loaded configuration: Test Strategy" in captured.out
            assert "Symbol: BTC/USDC" in captured.out
            assert "Cached to:" in captured.out

        finally:
            os.unlink(config_path)

    def test_download_invalid_yaml(self, capsys):
        """Test download command with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with patch("sys.argv", ["btc-download", config_path]):
                result = download.main()

            captured = capsys.readouterr()
            assert result == 1
            assert "Invalid YAML configuration" in captured.out

        finally:
            os.unlink(config_path)


class TestBacktestCLI:
    """Test the backtest CLI command."""

    def create_test_config(self):
        """Create a test configuration for backtest testing."""
        return {
            "name": "Test Strategy",
            "symbol": "BTC/USDC",
            "exchange": "binanceus",
            "timeframes": {"entry": "5m"},
            "indicators": [
                {"id": "RSI_14", "type": "RSI", "timeframe": "5m", "length": 14}
            ],
            "logic": {
                "entry_long": ["RSI_14 < 30"],
                "exit_long": ["RSI_14 > 70"],
                "entry_short": [],
                "exit_short": [],
            },
            "backtest": {
                "from": "2024-01-01",
                "to": "2024-01-02",
                "cash": 10000,
                "commission": 0.001,
            },
        }

    def test_backtest_main_missing_config(self, capsys):
        """Test backtest command with missing config file."""
        with patch("sys.argv", ["btc-backtest", "nonexistent.yaml"]):
            result = backtest.main()

        captured = capsys.readouterr()
        assert result == 1
        assert "Configuration file not found" in captured.out

    @patch("btc_research.cli.backtest.Backtester")
    @patch("btc_research.cli.backtest.Engine")
    def test_backtest_main_success(
        self, mock_engine_class, mock_backtester_class, capsys
    ):
        """Test successful backtest execution."""
        # Mock Engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        test_data = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [1000, 1100],
                "RSI_14": [25, 75],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"),
        )

        mock_engine.run.return_value = test_data

        # Mock Backtester
        mock_backtester = Mock()
        mock_backtester_class.return_value = mock_backtester

        mock_stats = {
            "total_return": 0.055,  # 5.5%
            "max_drawdown": -0.021,  # -2.1%
            "sharpe_ratio": 1.2,
            "num_trades": 3,
            "win_rate": 0.67,  # 67%
            "profit_factor": 1.5,
            "avg_trade": 183.33,
            "final_value": 10550.0,
            "initial_cash": 10000.0,
            "equity_curve": [
                {"timestamp": "2024-01-01T00:00:00Z", "equity": 10000},
                {"timestamp": "2024-01-01T12:00:00Z", "equity": 10550},
            ],
        }
        mock_backtester.run.return_value = mock_stats

        config = self.create_test_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch("sys.argv", ["btc-backtest", config_path]):
                result = backtest.main()

            captured = capsys.readouterr()
            assert result == 0
            assert "Backtest Results" in captured.out
            assert "Total Return:" in captured.out

            # Verify components were called
            mock_engine_class.assert_called_once()
            mock_engine.run.assert_called_once()
            mock_backtester_class.assert_called_once()
            mock_backtester.run.assert_called_once()

        finally:
            os.unlink(config_path)

    @patch("btc_research.cli.backtest.Backtester")
    @patch("btc_research.cli.backtest.Engine")
    def test_backtest_json_output(
        self, mock_engine_class, mock_backtester_class, capsys
    ):
        """Test backtest command with JSON output."""
        # Setup mocks
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.return_value = pd.DataFrame({"close": [100]})

        mock_backtester = Mock()
        mock_backtester_class.return_value = mock_backtester
        mock_stats = {"total_return": 0.055, "num_trades": 2}
        mock_backtester.run.return_value = mock_stats

        config = self.create_test_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch("sys.argv", ["btc-backtest", config_path, "--json"]):
                result = backtest.main()

            captured = capsys.readouterr()
            assert result == 0

            # Parse JSON output
            output_data = json.loads(captured.out)
            assert output_data["total_return"] == 0.055
            assert output_data["num_trades"] == 2

        finally:
            os.unlink(config_path)

    @patch("matplotlib.pyplot")
    @patch("btc_research.cli.backtest.Backtester")
    @patch("btc_research.cli.backtest.Engine")
    def test_backtest_with_plot(
        self, mock_engine_class, mock_backtester_class, mock_plt, capsys
    ):
        """Test backtest command with plotting."""
        # Setup mocks
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.return_value = pd.DataFrame({"close": [100]})

        mock_backtester = Mock()
        mock_backtester_class.return_value = mock_backtester
        mock_stats = {
            "total_return": 0.055,  # 5.5%
            "max_drawdown": -0.021,  # -2.1%
            "sharpe_ratio": 1.2,
            "num_trades": 3,
            "win_rate": 0.67,  # 67%
            "profit_factor": 1.5,
            "avg_trade": 183.33,
            "final_value": 10550.0,
            "initial_cash": 10000.0,
            "equity_curve": [
                {"timestamp": "2024-01-01T00:00:00Z", "equity": 10000},
                {"timestamp": "2024-01-01T12:00:00Z", "equity": 10550},
            ],
        }
        mock_backtester.run.return_value = mock_stats

        # Mock matplotlib
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        config = self.create_test_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch("sys.argv", ["btc-backtest", config_path, "--plot"]):
                result = backtest.main()

            captured = capsys.readouterr()
            assert result == 0
            assert "Generating equity curve plot" in captured.out
            assert "Plot saved as:" in captured.out

            # Verify plotting was called
            mock_plt.subplots.assert_called_once()
            mock_plt.savefig.assert_called_once()

        finally:
            os.unlink(config_path)


class TestOptimiseCLI:
    """Test the optimise CLI command."""

    def create_test_config(self):
        """Create a test configuration for optimization testing."""
        return {
            "name": "Test Strategy",
            "symbol": "BTC/USDC",
            "exchange": "binanceus",
            "timeframes": {"entry": "5m"},
            "indicators": [
                {"id": "RSI_14", "type": "RSI", "timeframe": "5m", "length": 14}
            ],
            "logic": {
                "entry_long": ["RSI_14 < 30"],
                "exit_long": ["RSI_14 > 70"],
                "entry_short": [],
                "exit_short": [],
            },
            "backtest": {"from": "2024-01-01", "to": "2024-01-02", "cash": 10000},
        }

    def test_optimise_main_missing_config(self, capsys):
        """Test optimise command with missing config file."""
        with patch(
            "sys.argv", ["btc-optimise", "nonexistent.yaml", "--grid", "length=10,14"]
        ):
            result = optimise.main()

        captured = capsys.readouterr()
        assert result == 1
        assert "Configuration file not found" in captured.out

    def test_optimise_main_no_grid(self, capsys):
        """Test optimise command without grid parameters."""
        config = self.create_test_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch("sys.argv", ["btc-optimise", config_path]):
                result = optimise.main()

            captured = capsys.readouterr()
            assert result == 1
            assert "--grid parameter is required" in captured.out

        finally:
            os.unlink(config_path)

    def test_parse_grid_parameters(self):
        """Test grid parameter parsing function."""
        # Test basic parsing
        result = optimise.parse_grid_parameters("length=10,14,20")
        assert result == {"length": [10, 14, 20]}

        # Test multiple parameters
        result = optimise.parse_grid_parameters("length=10,14;period=0.1,0.2")
        assert result == {"length": [10, 14], "period": [0.1, 0.2]}

        # Test string values
        result = optimise.parse_grid_parameters("method=sma,ema")
        assert result == {"method": ["sma", "ema"]}

    def test_generate_parameter_combinations(self):
        """Test parameter combination generation."""
        grid_params = {"length": [10, 14], "period": [0.1, 0.2]}
        combinations = list(optimise.generate_parameter_combinations(grid_params))

        expected = [
            {"length": 10, "period": 0.1},
            {"length": 10, "period": 0.2},
            {"length": 14, "period": 0.1},
            {"length": 14, "period": 0.2},
        ]
        assert combinations == expected

    def test_update_config_with_params(self):
        """Test configuration updating with parameters."""
        config = self.create_test_config()
        params = {"RSI_14.length": 20}

        updated_config = optimise.update_config_with_params(config, params)

        # Check that RSI indicator length was updated
        rsi_indicator = updated_config["indicators"][0]
        assert rsi_indicator["length"] == 20
        assert rsi_indicator["id"] == "RSI_14"

    @patch("btc_research.cli.optimise.Backtester")
    @patch("btc_research.cli.optimise.Engine")
    def test_run_single_optimization(self, mock_engine_class, mock_backtester_class):
        """Test single optimization run."""
        # Setup mocks
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.return_value = pd.DataFrame({"close": [100]})

        mock_backtester = Mock()
        mock_backtester_class.return_value = mock_backtester
        mock_stats = {
            "total_return": 5.5,
            "max_drawdown": -2.1,
            "sharpe_ratio": 1.2,
            "num_trades": 3,
            "profit_factor": 1.5,
        }
        mock_backtester.run.return_value = mock_stats

        config = self.create_test_config()
        params = {"RSI_14.length": 20}

        result = optimise.run_single_optimization(config, params)

        assert result["success"] is True
        assert result["parameters"] == params
        assert result["total_return"] == 5.5
        assert result["sharpe_ratio"] == 1.2

    @patch("btc_research.cli.optimise.run_single_optimization")
    def test_optimise_main_success(self, mock_run_optimization, capsys):
        """Test successful optimization execution."""
        # Mock optimization results
        mock_run_optimization.side_effect = [
            {
                "success": True,
                "parameters": {"RSI_14.length": 10},
                "total_return": 3.0,
                "max_drawdown": -1.5,
                "sharpe_ratio": 0.8,
                "num_trades": 2,
                "profit_factor": 1.2,
            },
            {
                "success": True,
                "parameters": {"RSI_14.length": 14},
                "total_return": 5.5,
                "max_drawdown": -2.1,
                "sharpe_ratio": 1.2,
                "num_trades": 3,
                "profit_factor": 1.5,
            },
        ]

        config = self.create_test_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch(
                "sys.argv",
                ["btc-optimise", config_path, "--grid", "RSI_14.length=10,14"],
            ):
                result = optimise.main()

            captured = capsys.readouterr()
            assert result == 0
            assert "Best result (total_return: 5.5000)" in captured.out
            assert "Total combinations: 2" in captured.out
            assert "Successful runs: 2" in captured.out

        finally:
            os.unlink(config_path)

    @patch("btc_research.cli.optimise.run_single_optimization")
    def test_optimise_csv_output(self, mock_run_optimization):
        """Test optimization with CSV output."""
        # Mock optimization results
        mock_run_optimization.return_value = {
            "success": True,
            "parameters": {"RSI_14.length": 10},
            "total_return": 3.0,
            "max_drawdown": -1.5,
            "sharpe_ratio": 0.8,
            "num_trades": 2,
            "profit_factor": 1.2,
            "run_time": 1.5,
        }

        config = self.create_test_config()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as config_file:
            yaml.dump(config, config_file)
            config_path = config_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as output_file:
            output_path = output_file.name

        try:
            with patch(
                "sys.argv",
                [
                    "btc-optimise",
                    config_path,
                    "--grid",
                    "RSI_14.length=10",
                    "--output",
                    output_path,
                ],
            ):
                result = optimise.main()

            assert result == 0

            # Check CSV file was created and has correct content
            assert os.path.exists(output_path)
            with open(output_path) as f:
                content = f.read()
                assert "total_return" in content
                assert "RSI_14.length" in content
                assert "3.0" in content

        finally:
            os.unlink(config_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_help_commands(self, capsys):
        """Test that all CLI commands support --help."""
        commands = [
            (download, ["btc-download", "--help"]),
            (backtest, ["btc-backtest", "--help"]),
            (optimise, ["btc-optimise", "--help"]),
        ]

        for module, argv in commands:
            with patch("sys.argv", argv):
                with pytest.raises(SystemExit) as exc_info:
                    module.main()

                # Help should exit with code 0
                assert exc_info.value.code == 0

                captured = capsys.readouterr()
                assert (
                    "usage:" in captured.out.lower() or "usage:" in captured.err.lower()
                )

    def test_error_handling_consistency(self):
        """Test that all CLI commands handle errors consistently."""
        # Test with non-existent config file
        commands = [
            (download, ["btc-download", "nonexistent.yaml"]),
            (backtest, ["btc-backtest", "nonexistent.yaml"]),
            (optimise, ["btc-optimise", "nonexistent.yaml", "--grid", "param=1"]),
        ]

        for module, argv in commands:
            with patch("sys.argv", argv):
                result = module.main()
                assert result == 1  # All should return error code 1
