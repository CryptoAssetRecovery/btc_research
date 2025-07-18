"""
Test suite for position sizing module.

This test suite verifies that the position sizing module correctly implements
risk-per-trade position sizing and integrates properly with the existing
risk management system.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock

from btc_research.core.position_sizing import (
    PositionSizer,
    BacktraderPositionSizer,
    PositionSizingError,
    calculate_atr_stop_loss,
    validate_position_sizing_config,
)


class TestPositionSizer:
    """Test cases for the PositionSizer class."""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        sizer = PositionSizer(
            default_risk_pct=0.02,
            max_position_pct=0.25,
            min_position_size=0.001,
            max_position_size=10.0,
        )
        
        assert sizer.default_risk_pct == 0.02
        assert sizer.max_position_pct == 0.25
        assert sizer.min_position_size == 0.001
        assert sizer.max_position_size == 10.0

    def test_init_invalid_risk_pct(self):
        """Test initialization with invalid risk percentage."""
        with pytest.raises(PositionSizingError):
            PositionSizer(default_risk_pct=1.5)  # > 100%
        
        with pytest.raises(PositionSizingError):
            PositionSizer(default_risk_pct=0.0)  # = 0%

    def test_init_invalid_max_position_pct(self):
        """Test initialization with invalid max position percentage."""
        with pytest.raises(PositionSizingError):
            PositionSizer(max_position_pct=1.5)  # > 100%
        
        with pytest.raises(PositionSizingError):
            PositionSizer(max_position_pct=0.0)  # = 0%

    def test_calculate_risk_amount(self):
        """Test risk amount calculation."""
        sizer = PositionSizer()
        
        # Test 1% risk on $10,000 equity
        risk_amount = sizer.calculate_risk_amount(10000, 0.01)
        assert risk_amount == 100.0
        
        # Test 2% risk on $5,000 equity
        risk_amount = sizer.calculate_risk_amount(5000, 0.02)
        assert risk_amount == 100.0

    def test_calculate_risk_amount_invalid_inputs(self):
        """Test risk amount calculation with invalid inputs."""
        sizer = PositionSizer()
        
        # Invalid equity
        with pytest.raises(PositionSizingError):
            sizer.calculate_risk_amount(-1000, 0.01)
        
        # Invalid risk percentage
        with pytest.raises(PositionSizingError):
            sizer.calculate_risk_amount(10000, 1.5)

    def test_calculate_position_size_long(self):
        """Test position size calculation for long positions."""
        sizer = PositionSizer(default_risk_pct=0.01)
        
        # BTC long: $10,000 equity, $50,000 entry, $49,000 stop
        position_size = sizer.calculate_position_size(
            equity=10000,
            entry_price=50000,
            stop_price=49000,
            is_long=True
        )
        
        # Expected: $100 risk / $1000 stop distance = 0.1 BTC
        assert abs(position_size - 0.1) < 0.0001

    def test_calculate_position_size_short(self):
        """Test position size calculation for short positions."""
        sizer = PositionSizer(default_risk_pct=0.01)
        
        # BTC short: $10,000 equity, $50,000 entry, $51,000 stop
        position_size = sizer.calculate_position_size(
            equity=10000,
            entry_price=50000,
            stop_price=51000,
            is_long=False
        )
        
        # Expected: $100 risk / $1000 stop distance = 0.1 BTC
        assert abs(position_size - 0.1) < 0.0001

    def test_calculate_position_size_auto_detect_direction(self):
        """Test automatic direction detection."""
        sizer = PositionSizer(default_risk_pct=0.01)
        
        # Stop below entry = long position
        position_size = sizer.calculate_position_size(
            equity=10000,
            entry_price=50000,
            stop_price=49000
        )
        assert position_size > 0
        
        # Stop above entry = short position
        position_size = sizer.calculate_position_size(
            equity=10000,
            entry_price=50000,
            stop_price=51000
        )
        assert position_size > 0

    def test_calculate_position_size_invalid_stops(self):
        """Test position size calculation with invalid stop prices."""
        sizer = PositionSizer()
        
        # Long position with stop above entry
        with pytest.raises(PositionSizingError):
            sizer.calculate_position_size(
                equity=10000,
                entry_price=50000,
                stop_price=51000,
                is_long=True
            )
        
        # Short position with stop below entry
        with pytest.raises(PositionSizingError):
            sizer.calculate_position_size(
                equity=10000,
                entry_price=50000,
                stop_price=49000,
                is_long=False
            )

    def test_calculate_position_size_stop_too_close(self):
        """Test position size calculation with stop too close to entry."""
        sizer = PositionSizer(min_stop_distance_pct=0.01)  # 1% minimum
        
        # Stop only 0.1% away (too close)
        with pytest.raises(PositionSizingError):
            sizer.calculate_position_size(
                equity=10000,
                entry_price=50000,
                stop_price=49950,  # Only 0.1% away
                is_long=True
            )

    def test_calculate_position_size_stop_too_far(self):
        """Test position size calculation with stop too far from entry."""
        sizer = PositionSizer(max_stop_distance_pct=0.05)  # 5% maximum
        
        # Stop 10% away (too far)
        with pytest.raises(PositionSizingError):
            sizer.calculate_position_size(
                equity=10000,
                entry_price=50000,
                stop_price=45000,  # 10% away
                is_long=True
            )

    def test_validate_position_size(self):
        """Test position size validation."""
        sizer = PositionSizer(
            min_position_size=0.001,
            max_position_size=1.0,
            max_position_pct=0.2
        )
        
        # Valid position size
        validated = sizer.validate_position_size(0.1, 10000, 50000)
        assert validated == 0.1
        
        # Position too small (gets increased to minimum)
        validated = sizer.validate_position_size(0.0001, 10000, 50000)
        assert validated == 0.001
        
        # Position too large (gets capped at maximum)
        validated = sizer.validate_position_size(2.0, 10000, 50000)
        assert validated == 1.0

    def test_validate_position_size_max_equity_pct(self):
        """Test position size validation with max equity percentage."""
        sizer = PositionSizer(max_position_pct=0.1)  # 10% max
        
        # Position value = 0.1 BTC * $50,000 = $5,000
        # Max allowed = $10,000 * 10% = $1,000
        # So position should be capped at $1,000 / $50,000 = 0.02 BTC
        validated = sizer.validate_position_size(0.1, 10000, 50000)
        assert abs(validated - 0.02) < 0.0001

    def test_calculate_r_multiple_size(self):
        """Test R-multiple position sizing."""
        sizer = PositionSizer(default_risk_pct=0.01)
        
        position_size, target_price = sizer.calculate_r_multiple_size(
            equity=10000,
            entry_price=50000,
            stop_price=49000,
            target_r=2.0
        )
        
        # Position size should be 0.1 BTC (as calculated above)
        assert abs(position_size - 0.1) < 0.0001
        
        # Target price should be entry + 2 * stop_distance
        # $50,000 + 2 * $1,000 = $52,000
        assert abs(target_price - 52000) < 0.01

    def test_calculate_position_metrics(self):
        """Test position metrics calculation."""
        sizer = PositionSizer()
        
        metrics = sizer.calculate_position_metrics(
            equity=10000,
            entry_price=50000,
            stop_price=49000,
            position_size=0.1
        )
        
        assert metrics['position_size'] == 0.1
        assert metrics['position_value'] == 5000  # 0.1 * 50000
        assert metrics['position_pct'] == 0.5  # 5000 / 10000
        assert metrics['risk_amount'] == 100  # 0.1 * 1000
        assert metrics['risk_pct'] == 0.01  # 100 / 10000
        assert metrics['stop_distance'] == 1000
        assert metrics['stop_distance_pct'] == 0.02  # 1000 / 50000
        assert metrics['is_long'] == True
        assert metrics['entry_price'] == 50000
        assert metrics['stop_price'] == 49000
        assert metrics['equity'] == 10000


class TestBacktraderPositionSizer:
    """Test cases for Backtrader integration."""

    def test_init(self):
        """Test BacktraderPositionSizer initialization."""
        bt_sizer = BacktraderPositionSizer(risk_pct=0.02, max_position_pct=0.25)
        
        assert bt_sizer.position_sizer.default_risk_pct == 0.02
        assert bt_sizer.position_sizer.max_position_pct == 0.25

    def test_calculate_bt_position_size(self):
        """Test Backtrader position size calculation."""
        bt_sizer = BacktraderPositionSizer(risk_pct=0.01)
        
        # Mock strategy and broker
        mock_strategy = Mock()
        mock_broker = Mock()
        mock_broker.getvalue.return_value = 10000
        mock_strategy.broker = mock_broker
        
        mock_data = Mock()
        
        # Calculate position size
        position_size = bt_sizer.calculate_bt_position_size(
            strategy=mock_strategy,
            data=mock_data,
            entry_price=50000,
            stop_price=49000,
            is_long=True
        )
        
        # Should be 0.1 BTC (same as direct calculation)
        assert abs(position_size - 0.1) < 0.0001


class TestATRStopLoss:
    """Test cases for ATR stop loss calculation."""

    def create_sample_data(self, length=50):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=length, freq='1H')
        
        # Generate realistic price data
        base_price = 50000
        prices = [base_price]
        
        for i in range(1, length):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            close_price = price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)

    def test_calculate_atr_stop_loss(self):
        """Test ATR stop loss calculation."""
        df = self.create_sample_data(30)
        
        # Test long stop
        long_stop = calculate_atr_stop_loss(df, period=14, multiplier=2.0, is_long=True)
        current_price = df['close'].iloc[-1]
        
        assert long_stop > 0
        assert long_stop < current_price  # Stop should be below current price for long
        
        # Test short stop
        short_stop = calculate_atr_stop_loss(df, period=14, multiplier=2.0, is_long=False)
        
        assert short_stop > 0
        assert short_stop > current_price  # Stop should be above current price for short

    def test_calculate_atr_stop_loss_insufficient_data(self):
        """Test ATR stop loss with insufficient data."""
        df = self.create_sample_data(10)  # Less than 14 periods
        
        with pytest.raises(PositionSizingError):
            calculate_atr_stop_loss(df, period=14, multiplier=2.0, is_long=True)

    def test_calculate_atr_stop_loss_different_multipliers(self):
        """Test ATR stop loss with different multipliers."""
        df = self.create_sample_data(30)
        current_price = df['close'].iloc[-1]
        
        # Test different multipliers
        stop_1x = calculate_atr_stop_loss(df, period=14, multiplier=1.0, is_long=True)
        stop_2x = calculate_atr_stop_loss(df, period=14, multiplier=2.0, is_long=True)
        stop_3x = calculate_atr_stop_loss(df, period=14, multiplier=3.0, is_long=True)
        
        # Higher multiplier should give wider stop (further from price)
        assert stop_1x > stop_2x > stop_3x
        assert all(stop < current_price for stop in [stop_1x, stop_2x, stop_3x])


class TestConfigValidation:
    """Test cases for configuration validation."""

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            "risk_pct": 0.01,
            "max_position_pct": 0.2,
            "atr_period": 14,
            "atr_multiplier": 2.0
        }
        
        validated = validate_position_sizing_config(config)
        
        assert validated["risk_pct"] == 0.01
        assert validated["max_position_pct"] == 0.2
        assert validated["atr_period"] == 14
        assert validated["atr_multiplier"] == 2.0

    def test_validate_empty_config(self):
        """Test validation with empty configuration (uses defaults)."""
        config = {}
        
        validated = validate_position_sizing_config(config)
        
        # Should have default values
        assert validated["risk_pct"] == 0.01
        assert validated["max_position_pct"] == 0.2
        assert validated["atr_period"] == 14
        assert validated["atr_multiplier"] == 2.0

    def test_validate_invalid_risk_pct(self):
        """Test validation with invalid risk percentage."""
        config = {"risk_pct": 1.5}
        
        with pytest.raises(PositionSizingError):
            validate_position_sizing_config(config)

    def test_validate_invalid_atr_period(self):
        """Test validation with invalid ATR period."""
        config = {"atr_period": 0}
        
        with pytest.raises(PositionSizingError):
            validate_position_sizing_config(config)

    def test_validate_invalid_atr_multiplier(self):
        """Test validation with invalid ATR multiplier."""
        config = {"atr_multiplier": -1.0}
        
        with pytest.raises(PositionSizingError):
            validate_position_sizing_config(config)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_crypto_position_sizing_scenarios(self):
        """Test position sizing for realistic crypto scenarios."""
        sizer = PositionSizer(default_risk_pct=0.01)
        
        scenarios = [
            # Conservative BTC trade
            {
                "equity": 10000,
                "entry": 50000,
                "stop": 49000,
                "expected_size": 0.1,
                "expected_risk": 100
            },
            # Volatile altcoin trade
            {
                "equity": 10000,
                "entry": 100,
                "stop": 90,
                "expected_size": 10,
                "expected_risk": 100
            },
            # Small account
            {
                "equity": 1000,
                "entry": 50000,
                "stop": 49000,
                "expected_size": 0.01,
                "expected_risk": 10
            }
        ]
        
        for scenario in scenarios:
            position_size = sizer.calculate_position_size(
                equity=scenario["equity"],
                entry_price=scenario["entry"],
                stop_price=scenario["stop"],
                is_long=True
            )
            
            # Check position size
            assert abs(position_size - scenario["expected_size"]) < 0.0001
            
            # Check risk amount
            risk_amount = position_size * (scenario["entry"] - scenario["stop"])
            assert abs(risk_amount - scenario["expected_risk"]) < 0.01

    def test_extreme_market_conditions(self):
        """Test position sizing under extreme market conditions."""
        sizer = PositionSizer(default_risk_pct=0.01)
        
        # Very high volatility (wide stop)
        position_size = sizer.calculate_position_size(
            equity=10000,
            entry_price=50000,
            stop_price=45000,  # 10% stop
            is_long=True
        )
        
        # Should still risk exactly 1%
        risk_amount = position_size * 5000
        assert abs(risk_amount - 100) < 0.01
        
        # Very low volatility (tight stop)
        position_size = sizer.calculate_position_size(
            equity=10000,
            entry_price=50000,
            stop_price=49750,  # 0.5% stop
            is_long=True
        )
        
        # Should still risk exactly 1%
        risk_amount = position_size * 250
        assert abs(risk_amount - 100) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])