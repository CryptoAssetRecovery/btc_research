"""Indicator implementations for trading strategies."""

# Import indicators to register them automatically
from btc_research.indicators.ema import EMA
from btc_research.indicators.rsi import RSI

__all__ = ["EMA", "RSI"]
