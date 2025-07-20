#!/bin/bash

set -e

echo "🔧 BTC Research CLI Installation Script"
echo "======================================"

# Get the directory of this script and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Changing to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Verify we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ pyproject.toml not found. Please run this script from the project root or scripts directory."
    exit 1
fi

# Check if Python 3.11+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.11 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "❌ Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Check if poetry is now available
    if ! command -v poetry &> /dev/null; then
        echo "❌ Poetry installation failed. Please install Poetry manually:"
        echo "   curl -sSL https://install.python-poetry.org | python3 -"
        echo "   Then add ~/.local/bin to your PATH"
        exit 1
    fi
fi

echo "✅ Poetry is available"

# Install dependencies
echo "📦 Installing project dependencies..."
poetry install

# Verify CLI tools are working
echo "🔍 Verifying CLI installation..."

if poetry run btc-download --help > /dev/null 2>&1; then
    echo "✅ btc-download CLI is working"
else
    echo "❌ btc-download CLI installation failed"
    exit 1
fi

if poetry run btc-backtest --help > /dev/null 2>&1; then
    echo "✅ btc-backtest CLI is working"
else
    echo "❌ btc-backtest CLI installation failed"
    exit 1
fi

if poetry run btc-optimise --help > /dev/null 2>&1; then
    echo "✅ btc-optimise CLI is working"
else
    echo "❌ btc-optimise CLI installation failed"
    exit 1
fi

if poetry run btc-visualize --help > /dev/null 2>&1; then
    echo "✅ btc-visualize CLI is working"
else
    echo "❌ btc-visualize CLI installation failed"
    exit 1
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "Usage:"
echo "  poetry run btc-download --help    # Download market data"
echo "  poetry run btc-backtest --help    # Run backtests"
echo "  poetry run btc-optimise --help    # Optimize strategies"
echo "  poetry run btc-visualize --help   # Visualize results"
echo ""
echo "Or activate the virtual environment:"
echo "  poetry shell"
echo "  btc-download --help"
echo ""
echo "Example commands:"
echo "  poetry run btc-download --exchange binanceus --symbol BTC-USDT --timeframe 1h --days 30"
echo "  poetry run btc-backtest --config btc_research/config/volume-profile-breakout.yaml"