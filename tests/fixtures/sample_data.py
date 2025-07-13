"""
Sample data fixtures for testing.

This module provides realistic market data fixtures that can be used
across multiple test files for consistent testing.
"""


import numpy as np
import pandas as pd


def create_btc_sample_data(start_date="2024-01-01", periods=1000, freq="1h", seed=42):
    """
    Create realistic BTC price data for testing.

    Args:
        start_date (str): Start date for the data
        periods (int): Number of periods to generate
        freq (str): Frequency of data ('1h', '4h', '1d', etc.)
        seed (int): Random seed for reproducible data

    Returns:
        pd.DataFrame: OHLCV data with realistic price movements
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, periods=periods, freq=freq, tz="UTC")

    # Start with a base price around BTC levels
    base_price = 45000.0
    prices = []

    # Generate realistic price movements using geometric brownian motion
    for i in range(periods):
        # Add some trend and volatility patterns
        trend = 0.0001 * np.sin(i * 0.02)  # Slight cyclical trend
        volatility = 0.015 + 0.005 * np.sin(i * 0.05)  # Variable volatility

        # Random walk with drift
        change = np.random.normal(trend, volatility)
        base_price *= 1 + change
        prices.append(base_price)

    # Create OHLCV data with realistic relationships
    data = pd.DataFrame(index=dates)
    data["close"] = prices

    # Generate open prices (close of previous period + gap)
    data["open"] = data["close"].shift(1).fillna(data["close"].iloc[0])
    gap_factor = np.random.normal(1.0, 0.002, periods)  # Small gaps
    data["open"] *= gap_factor

    # Generate high and low with realistic spreads
    spread_factor = np.random.uniform(0.005, 0.02, periods)  # 0.5% to 2% spread
    data["high"] = np.maximum(data["open"], data["close"]) * (1 + spread_factor)
    data["low"] = np.minimum(data["open"], data["close"]) * (1 - spread_factor)

    # Ensure OHLC relationships are valid
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    # Generate volume data correlated with price volatility
    price_changes = np.abs(data["close"].pct_change().fillna(0))
    base_volume = 500
    volume_multiplier = 1 + price_changes * 50  # Higher volume on big moves
    data["volume"] = (
        np.random.uniform(0.5, 1.5, periods) * base_volume * volume_multiplier
    )

    return data


def create_trending_market_data(periods=500, trend="bull", volatility=0.015):
    """
    Create market data with specific trend characteristics.

    Args:
        periods (int): Number of periods
        trend (str): 'bull', 'bear', or 'sideways'
        volatility (float): Volatility level

    Returns:
        pd.DataFrame: OHLCV data with specified trend
    """
    dates = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    base_price = 45000.0

    # Set trend parameters
    if trend == "bull":
        drift = 0.0003  # Positive drift
    elif trend == "bear":
        drift = -0.0003  # Negative drift
    else:  # sideways
        drift = 0.0  # No drift

    prices = []
    price = base_price

    for _ in range(periods):
        change = np.random.normal(drift, volatility)
        price *= 1 + change
        prices.append(price)

    data = pd.DataFrame(index=dates)
    data["close"] = prices
    data["open"] = data["close"].shift(1).fillna(data["close"].iloc[0])

    # Generate OHLC with realistic spreads
    spread = volatility * 0.5
    data["high"] = np.maximum(data["open"], data["close"]) * (
        1 + np.random.uniform(0, spread, periods)
    )
    data["low"] = np.minimum(data["open"], data["close"]) * (
        1 - np.random.uniform(0, spread, periods)
    )

    # Fix OHLC relationships
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    data["volume"] = np.random.uniform(300, 800, periods)

    return data


def create_volatile_market_data(periods=300, volatility_level="high"):
    """
    Create market data with specific volatility characteristics.

    Args:
        periods (int): Number of periods
        volatility_level (str): 'low', 'medium', 'high', or 'extreme'

    Returns:
        pd.DataFrame: OHLCV data with specified volatility
    """
    volatility_map = {"low": 0.008, "medium": 0.015, "high": 0.025, "extreme": 0.040}

    volatility = volatility_map.get(volatility_level, 0.015)

    dates = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    base_price = 45000.0
    prices = [base_price]

    for _ in range(periods - 1):
        change = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    data = pd.DataFrame(index=dates)
    data["close"] = prices
    data["open"] = data["close"].shift(1).fillna(data["close"].iloc[0])

    # Higher volatility = wider OHLC spreads
    spread_factor = volatility * 2
    data["high"] = np.maximum(data["open"], data["close"]) * (
        1 + np.random.uniform(0, spread_factor, periods)
    )
    data["low"] = np.minimum(data["open"], data["close"]) * (
        1 - np.random.uniform(0, spread_factor, periods)
    )

    # Fix OHLC relationships
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    # Higher volatility = higher volume
    base_volume = 400
    volume_multiplier = 1 + volatility * 20
    data["volume"] = (
        np.random.uniform(0.5, 1.5, periods) * base_volume * volume_multiplier
    )

    return data


def create_multi_timeframe_data(symbol="BTC/USD", start_date="2024-01-01", days=30):
    """
    Create consistent multi-timeframe data for testing.

    Args:
        symbol (str): Symbol name
        start_date (str): Start date
        days (int): Number of days of data

    Returns:
        dict: Dictionary with timeframe keys and DataFrame values
    """
    # Create base 1-minute data
    periods_1m = days * 24 * 60
    data_1m = create_btc_sample_data(start_date, periods_1m, "1min")

    # Resample to different timeframes
    timeframes = {
        "1m": data_1m,
        "5m": data_1m.resample("5min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(),
        "15m": data_1m.resample("15min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(),
        "1h": data_1m.resample("1h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(),
        "4h": data_1m.resample("4h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(),
        "1d": data_1m.resample("1d")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(),
    }

    return timeframes


def create_gap_data(periods=200, gap_frequency=0.05, gap_size_range=(0.01, 0.05)):
    """
    Create data with price gaps for testing edge cases.

    Args:
        periods (int): Number of periods
        gap_frequency (float): Probability of a gap occurring
        gap_size_range (tuple): Range of gap sizes as percentage

    Returns:
        pd.DataFrame: OHLCV data with gaps
    """
    dates = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    base_price = 45000.0
    prices = [base_price]

    for i in range(periods - 1):
        # Normal price movement
        normal_change = np.random.normal(0, 0.015)
        new_price = prices[-1] * (1 + normal_change)

        # Occasionally add gaps
        if np.random.random() < gap_frequency:
            gap_size = np.random.uniform(*gap_size_range)
            gap_direction = np.random.choice([-1, 1])
            new_price *= 1 + gap_direction * gap_size

        prices.append(new_price)

    data = pd.DataFrame(index=dates)
    data["close"] = prices
    data["open"] = data["close"].shift(1).fillna(data["close"].iloc[0])

    # Generate OHLC
    spread = 0.01
    data["high"] = np.maximum(data["open"], data["close"]) * (
        1 + np.random.uniform(0, spread, periods)
    )
    data["low"] = np.minimum(data["open"], data["close"]) * (
        1 - np.random.uniform(0, spread, periods)
    )

    # Fix OHLC relationships
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    data["volume"] = np.random.uniform(200, 1000, periods)

    return data


# Configuration examples for testing
SAMPLE_CONFIGS = {
    "simple_ema": {
        "version": "1.0",
        "name": "Simple EMA Strategy",
        "symbol": "BTC/USD",
        "timeframes": {"entry": "1h"},
        "indicators": [
            {"id": "EMA_20", "type": "EMA", "timeframe": "1h", "length": 20}
        ],
        "logic": {
            "entry_long": ["EMA_20_trend == 'bull'"],
            "exit_long": ["EMA_20_trend == 'bear'"],
            "entry_short": [],
            "exit_short": [],
        },
        "backtest": {
            "cash": 10000,
            "commission": 0.001,
            "from": "2024-01-01",
            "to": "2024-01-31",
        },
    },
    "rsi_mean_reversion": {
        "version": "1.0",
        "name": "RSI Mean Reversion",
        "symbol": "BTC/USD",
        "timeframes": {"entry": "1h"},
        "indicators": [
            {"id": "RSI_14", "type": "RSI", "timeframe": "1h", "length": 14}
        ],
        "logic": {
            "entry_long": ["RSI_14 < 30"],
            "exit_long": ["RSI_14 > 70"],
            "entry_short": ["RSI_14 > 70"],
            "exit_short": ["RSI_14 < 30"],
        },
        "backtest": {
            "cash": 10000,
            "commission": 0.001,
            "from": "2024-01-01",
            "to": "2024-01-31",
        },
    },
    "multi_timeframe": {
        "version": "1.0",
        "name": "Multi-Timeframe Strategy",
        "symbol": "BTC/USD",
        "timeframes": {"bias": "4h", "entry": "1h"},
        "indicators": [
            {"id": "BIAS_EMA", "type": "EMA", "timeframe": "4h", "length": 50},
            {"id": "ENTRY_RSI", "type": "RSI", "timeframe": "1h", "length": 14},
        ],
        "logic": {
            "entry_long": ["BIAS_EMA_trend == 'bull'", "ENTRY_RSI < 30"],
            "exit_long": ["ENTRY_RSI > 70"],
            "entry_short": [],
            "exit_short": [],
        },
        "backtest": {
            "cash": 10000,
            "commission": 0.001,
            "from": "2024-01-01",
            "to": "2024-01-31",
        },
    },
}
