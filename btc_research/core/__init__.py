"""Core framework components for the BTC research engine."""

from .backtester import Backtester, BacktesterError
from .base_indicator import BaseIndicator
from .registry import RegistrationError, get, register

__all__ = [
    "BaseIndicator",
    "register",
    "get",
    "RegistrationError",
    "Backtester",
    "BacktesterError",
]
