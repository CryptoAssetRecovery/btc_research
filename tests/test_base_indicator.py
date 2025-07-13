"""
Unit tests for the BaseIndicator abstract base class.

This module tests the BaseIndicator contract, abstract method enforcement,
integration with the registry system, and proper type validation.
"""

from abc import ABC
from typing import Any

import numpy as np
import pandas as pd
import pytest

from btc_research.core.base_indicator import BaseIndicator
from btc_research.core.registry import _clear_registry, get, register


class TestBaseIndicatorContract:
    """Test the abstract base class contract and method enforcement."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseIndicator cannot be instantiated directly due to abstract methods."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseIndicator()

    def test_is_abstract_base_class(self):
        """BaseIndicator should inherit from ABC."""
        assert issubclass(BaseIndicator, ABC)
        assert hasattr(BaseIndicator, "__abstractmethods__")

        # Check that both required methods are abstract
        abstract_methods = BaseIndicator.__abstractmethods__
        assert "params" in abstract_methods
        assert "compute" in abstract_methods
        assert len(abstract_methods) == 2

    def test_params_method_signature(self):
        """Verify params method has correct signature."""
        # Check method exists and is a classmethod
        assert hasattr(BaseIndicator, "params")
        assert isinstance(BaseIndicator.__dict__["params"], classmethod)

        # Check it's abstract
        assert BaseIndicator.params.__isabstractmethod__

    def test_compute_method_signature(self):
        """Verify compute method has correct signature."""
        # Check method exists
        assert hasattr(BaseIndicator, "compute")

        # Check it's abstract
        assert BaseIndicator.compute.__isabstractmethod__


class TestConcreteImplementations:
    """Test concrete implementations of BaseIndicator."""

    def setup_method(self):
        """Clear registry before each test."""
        _clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        _clear_registry()

    def test_missing_params_method_fails(self):
        """Class without params method cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class IncompleteIndicator(BaseIndicator):
                def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                    return pd.DataFrame(index=df.index)

            IncompleteIndicator()

    def test_missing_compute_method_fails(self):
        """Class without compute method cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class IncompleteIndicator(BaseIndicator):
                @classmethod
                def params(cls) -> dict:
                    return {}

            IncompleteIndicator()

    def test_valid_implementation_works(self):
        """Complete implementation can be instantiated."""

        class ValidIndicator(BaseIndicator):
            @classmethod
            def params(cls) -> dict:
                return {"length": 14}

            def __init__(self, length: int = 14):
                self.length = length

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                result = pd.DataFrame(index=df.index)
                result[f"TEST_{self.length}"] = df["close"].rolling(self.length).mean()
                return result

        # Should not raise any exception
        indicator = ValidIndicator()
        assert indicator.length == 14

        # Test with custom parameters
        indicator_custom = ValidIndicator(length=20)
        assert indicator_custom.length == 20

    def test_params_return_type(self):
        """params method should return a dictionary."""

        class TestIndicator(BaseIndicator):
            @classmethod
            def params(cls) -> dict[str, Any]:
                return {"length": 14, "source": "close", "smoothing": 2.0}

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(index=df.index)

        params = TestIndicator.params()
        assert isinstance(params, dict)
        assert params == {"length": 14, "source": "close", "smoothing": 2.0}

    def test_compute_with_valid_dataframe(self):
        """compute method should work with valid OHLCV DataFrame."""

        class TestIndicator(BaseIndicator):
            @classmethod
            def params(cls) -> dict:
                return {"length": 5}

            def __init__(self, length: int = 5):
                self.length = length

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                result = pd.DataFrame(index=df.index)
                result[f"SMA_{self.length}"] = df["close"].rolling(self.length).mean()
                result[f"SMA_{self.length}_signal"] = (
                    df["close"] > result[f"SMA_{self.length}"]
                )
                return result

        # Create test OHLCV data
        dates = pd.date_range("2024-01-01", periods=20, freq="1H")
        df = pd.DataFrame(
            {
                "open": np.random.rand(20) * 100,
                "high": np.random.rand(20) * 100 + 1,
                "low": np.random.rand(20) * 100 - 1,
                "close": np.random.rand(20) * 100,
                "volume": np.random.rand(20) * 1000,
            },
            index=dates,
        )

        indicator = TestIndicator(length=5)
        result = indicator.compute(df)

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert result.index.equals(df.index)
        assert "SMA_5" in result.columns
        assert "SMA_5_signal" in result.columns

        # Verify calculation correctness for SMA
        expected_sma = df["close"].rolling(5).mean()
        pd.testing.assert_series_equal(result["SMA_5"], expected_sma, check_names=False)


class TestRegistryIntegration:
    """Test integration with the registry system."""

    def setup_method(self):
        """Clear registry before each test."""
        _clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        _clear_registry()

    def test_register_decorator_with_base_indicator(self):
        """BaseIndicator subclasses can be registered and retrieved."""

        @register("TestRSI")
        class RSIIndicator(BaseIndicator):
            @classmethod
            def params(cls) -> dict:
                return {"length": 14, "overbought": 70, "oversold": 30}

            def __init__(
                self, length: int = 14, overbought: float = 70, oversold: float = 30
            ):
                self.length = length
                self.overbought = overbought
                self.oversold = oversold

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                # Simplified RSI calculation for testing
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.length).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.length).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                result = pd.DataFrame(index=df.index)
                result[f"RSI_{self.length}"] = rsi
                result[f"RSI_{self.length}_overbought"] = rsi > self.overbought
                result[f"RSI_{self.length}_oversold"] = rsi < self.oversold
                return result

        # Retrieve from registry
        retrieved_class = get("TestRSI")
        assert retrieved_class is RSIIndicator

        # Create instance
        indicator = retrieved_class(length=21)
        assert indicator.length == 21
        assert indicator.overbought == 70
        assert indicator.oversold == 30

    def test_multiple_indicators_registration(self):
        """Multiple BaseIndicator subclasses can be registered."""

        @register("SMA")
        class SMAIndicator(BaseIndicator):
            @classmethod
            def params(cls) -> dict:
                return {"length": 20}

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(index=df.index)

        @register("EMA")
        class EMAIndicator(BaseIndicator):
            @classmethod
            def params(cls) -> dict:
                return {"length": 20, "alpha": 0.1}

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(index=df.index)

        # Both should be retrievable
        sma_class = get("SMA")
        ema_class = get("EMA")

        assert sma_class is SMAIndicator
        assert ema_class is EMAIndicator

        # Check params
        assert sma_class.params() == {"length": 20}
        assert ema_class.params() == {"length": 20, "alpha": 0.1}


class TestTypeHints:
    """Test type hint compliance and validation."""

    def test_params_return_type_annotation(self):
        """params method should have proper return type annotation."""
        import inspect

        from btc_research.core.base_indicator import BaseIndicator

        signature = inspect.signature(BaseIndicator.params)
        return_annotation = signature.return_annotation

        # Should be Dict[str, Any] or dict
        assert return_annotation in [dict[str, Any], dict, "Dict[str, Any]"]

    def test_compute_type_annotations(self):
        """compute method should have proper type annotations."""
        import inspect

        from btc_research.core.base_indicator import BaseIndicator

        signature = inspect.signature(BaseIndicator.compute)

        # Check parameter annotation
        df_param = signature.parameters["df"]
        assert df_param.annotation == pd.DataFrame

        # Check return annotation
        return_annotation = signature.return_annotation
        assert return_annotation == pd.DataFrame


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_compute_with_empty_dataframe(self):
        """compute should handle empty DataFrames gracefully."""

        class RobustIndicator(BaseIndicator):
            @classmethod
            def params(cls) -> dict:
                return {"length": 5}

            def __init__(self, length: int = 5):
                self.length = length

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                result = pd.DataFrame(index=df.index)
                if len(df) < self.length:
                    result[f"SMA_{self.length}"] = float("nan")
                else:
                    result[f"SMA_{self.length}"] = (
                        df["close"].rolling(self.length).mean()
                    )
                return result

        # Empty DataFrame
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        indicator = RobustIndicator()
        result = indicator.compute(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert result.index.equals(empty_df.index)

    def test_compute_with_insufficient_data(self):
        """compute should handle insufficient data for calculation."""

        class TestIndicator(BaseIndicator):
            @classmethod
            def params(cls) -> dict:
                return {"length": 10}

            def __init__(self, length: int = 10):
                self.length = length

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                result = pd.DataFrame(index=df.index)
                sma = df["close"].rolling(self.length).mean()
                result[f"SMA_{self.length}"] = sma
                return result

        # DataFrame with only 5 rows, but indicator needs 10
        dates = pd.date_range("2024-01-01", periods=5, freq="1H")
        df = pd.DataFrame(
            {
                "open": [1, 2, 3, 4, 5],
                "high": [1.1, 2.1, 3.1, 4.1, 5.1],
                "low": [0.9, 1.9, 2.9, 3.9, 4.9],
                "close": [1, 2, 3, 4, 5],
                "volume": [100, 200, 300, 400, 500],
            },
            index=dates,
        )

        indicator = TestIndicator(length=10)
        result = indicator.compute(df)

        # Should return DataFrame with NaN values for SMA
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert result.index.equals(df.index)
        assert result["SMA_10"].isna().all()  # All values should be NaN


class TestDocumentationExamples:
    """Test that documentation examples work correctly."""

    def setup_method(self):
        """Clear registry before each test."""
        _clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        _clear_registry()

    def test_rsi_example_from_docstring(self):
        """Test that the RSI example from the docstring works."""

        @register("RSI")
        class RSI(BaseIndicator):
            @classmethod
            def params(cls) -> dict:
                return {"length": 14, "overbought": 70, "oversold": 30}

            def __init__(
                self, length: int = 14, overbought: float = 70, oversold: float = 30
            ):
                self.length = length
                self.overbought = overbought
                self.oversold = oversold

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                # Simplified RSI calculation (normally would use talib)
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.length).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.length).mean()
                rs = gain / loss
                rsi_values = 100 - (100 / (1 + rs))

                result = pd.DataFrame(index=df.index)
                result[f"RSI_{self.length}"] = rsi_values
                result[f"RSI_{self.length}_overbought"] = rsi_values > self.overbought
                result[f"RSI_{self.length}_oversold"] = rsi_values < self.oversold

                return result

        # Test the example
        dates = pd.date_range("2024-01-01", periods=30, freq="1H")
        prices = [50 + i + np.sin(i / 3) * 5 for i in range(30)]  # Trending prices

        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1000] * 30,
            },
            index=dates,
        )

        rsi = RSI(length=14, overbought=70, oversold=30)
        result = rsi.compute(df)

        # Verify structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 30
        assert result.index.equals(df.index)
        assert "RSI_14" in result.columns
        assert "RSI_14_overbought" in result.columns
        assert "RSI_14_oversold" in result.columns

        # Verify data types
        assert result["RSI_14"].dtype == float
        assert result["RSI_14_overbought"].dtype == bool
        assert result["RSI_14_oversold"].dtype == bool
