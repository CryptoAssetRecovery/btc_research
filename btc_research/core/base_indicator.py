"""
Abstract base class for all technical indicators in the BTC research engine.

This module defines the contract that all indicators must implement to integrate
with the research framework. It provides type safety, clear interface definitions,
and comprehensive documentation for indicator developers.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

__all__ = ["BaseIndicator"]


class BaseIndicator(ABC):
    """
    Abstract base class defining the contract for all technical indicators.

    This class establishes the interface that all indicators must implement to work
    with the BTC research engine. It enforces a consistent pattern for indicator
    development and ensures proper integration with the registry system.

    The BaseIndicator class defines two abstract methods that concrete implementations
    must provide:

    1. `params()`: A class method returning default parameters for the indicator
    2. `compute()`: An instance method that performs the actual calculation

    Example:
        Creating a simple RSI indicator:

        ```python
        from btc_research.core.base_indicator import BaseIndicator
        from btc_research.core.registry import register
        import pandas as pd
        import talib

        @register("RSI")
        class RSI(BaseIndicator):
            @classmethod
            def params(cls) -> dict:
                return {"length": 14, "overbought": 70, "oversold": 30}

            def __init__(self, length: int = 14, overbought: float = 70, oversold: float = 30):
                self.length = length
                self.overbought = overbought
                self.oversold = oversold

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                rsi_values = talib.RSI(df["close"], timeperiod=self.length)

                result = pd.DataFrame(index=df.index)
                result[f"RSI_{self.length}"] = rsi_values
                result[f"RSI_{self.length}_overbought"] = rsi_values > self.overbought
                result[f"RSI_{self.length}_oversold"] = rsi_values < self.oversold

                return result
        ```

    Integration with registry:
        The BaseIndicator class is designed to work seamlessly with the registry
        system. Use the @register decorator to make indicators discoverable:

        ```python
        @register("MyIndicator")
        class MyIndicator(BaseIndicator):
            # Implementation here...
        ```

    Multi-timeframe support:
        Indicators receive OHLCV data for their specific timeframe. The engine
        handles timeframe alignment and forward-filling automatically:

        ```python
        # In YAML configuration:
        indicators:
          - id: "RSI_14"
            type: "RSI"
            timeframe: "5m"
            length: 14
          - id: "EMA_200"
            type: "EMA"
            timeframe: "1h"
            length: 200
        ```

    Notes:
        - All column names should be prefixed with the indicator's identifier
        - Return DataFrames must have the same index as the input DataFrame
        - Use descriptive column names that include parameter values
        - Handle edge cases like insufficient data gracefully
        - Consider performance for large datasets
    """

    @classmethod
    @abstractmethod
    def params(cls) -> dict[str, Any]:
        """
        Return default hyper-parameters for YAML auto-documentation.

        This class method defines the default parameters that the indicator accepts.
        It serves multiple purposes:

        1. **Documentation**: Parameters are automatically documented in generated docs
        2. **YAML generation**: Tools can auto-generate YAML templates
        3. **Validation**: The engine can validate user-provided parameters
        4. **Defaults**: Provides fallback values when parameters are not specified

        Returns:
            Dict[str, Any]: A dictionary mapping parameter names to their default values.
                          Keys should be valid Python identifiers that match the
                          constructor parameters.

        Example:
            ```python
            @classmethod
            def params(cls) -> dict:
                return {
                    "length": 14,           # Moving average period
                    "source": "close",      # Price source (open/high/low/close)
                    "smoothing": 2.0,       # Smoothing factor
                    "min_periods": None     # Minimum periods for calculation
                }
            ```

        Best practices:
            - Use descriptive parameter names
            - Include sensible defaults for all parameters
            - Document complex parameters with comments
            - Keep parameter types JSON-serializable when possible
            - Consider parameter validation in the constructor

        Note:
            This method should be pure (no side effects) and deterministic.
            It should not depend on instance state since it's a class method.
        """
        pass

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given OHLCV DataFrame in this indicator's timeframe, return computed columns.

        This is the core method where indicator calculations are performed. It receives
        OHLCV data for the indicator's specific timeframe and must return one or more
        calculated columns indexed exactly like the input DataFrame.

        Args:
            df (pd.DataFrame): Input OHLCV DataFrame with columns:
                             - 'open': Opening prices
                             - 'high': High prices
                             - 'low': Low prices
                             - 'close': Closing prices
                             - 'volume': Trading volumes

                             The DataFrame index should be datetime-based and sorted.
                             Additional columns may be present from other indicators.

        Returns:
            pd.DataFrame: DataFrame with computed indicator columns. Must satisfy:
                        - Same index as input DataFrame (df.index)
                        - Column names prefixed with indicator identifier
                        - All numeric data should be float64 when possible
                        - Boolean signals should use bool dtype
                        - Missing values handled appropriately (NaN for numeric)

        Example:
            ```python
            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                # Calculate 20-period simple moving average
                sma_20 = df["close"].rolling(window=20).mean()

                # Calculate crossover signals
                price_above_sma = df["close"] > sma_20

                # Create result DataFrame with same index
                result = pd.DataFrame(index=df.index)
                result["SMA_20"] = sma_20
                result["SMA_20_signal"] = price_above_sma
                result["SMA_20_strength"] = (df["close"] / sma_20 - 1) * 100

                return result
            ```

        Error handling:
            ```python
            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                if len(df) < self.min_periods:
                    # Return empty DataFrame with correct index and columns
                    result = pd.DataFrame(index=df.index)
                    result[f"RSI_{self.length}"] = float('nan')
                    return result

                # Proceed with calculation...
            ```

        Performance considerations:
            - Use vectorized pandas operations when possible
            - Avoid loops over DataFrame rows
            - Consider memory usage for large datasets
            - Cache expensive intermediate calculations
            - Use appropriate dtypes to minimize memory

        Raises:
            ValueError: If input DataFrame is invalid or missing required columns
            RuntimeError: If calculation fails due to insufficient data or other issues

        Note:
            - This method should be pure (no side effects beyond logging)
            - Should handle edge cases like empty DataFrames gracefully
            - Consider timezone-aware datetime indexes
            - Ensure output is deterministic for same inputs
        """
        pass
