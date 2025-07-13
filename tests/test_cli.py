"""Tests for CLI modules."""

import subprocess
import sys
from pathlib import Path


def test_cli_download_help():
    """Test that btc-download shows help."""
    result = subprocess.run([
        sys.executable, "-m", "btc_research.cli.download", "--help"
    ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    assert result.returncode == 0
    assert "Download and cache market data" in result.stdout


def test_cli_backtest_help():
    """Test that btc-backtest shows help."""
    result = subprocess.run([
        sys.executable, "-m", "btc_research.cli.backtest", "--help"
    ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    assert result.returncode == 0
    assert "Run strategy backtests" in result.stdout


def test_cli_optimise_help():
    """Test that btc-optimise shows help."""
    result = subprocess.run([
        sys.executable, "-m", "btc_research.cli.optimise", "--help"
    ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    assert result.returncode == 0
    assert "Optimize strategy parameters" in result.stdout