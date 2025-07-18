"""
Position Sizing Module for Risk-Per-Trade Management.

This module provides ATR-based position sizing to replace the current 95% capital
allocation approach. It implements proper risk management where position size is
calculated based on the distance to stop loss and desired risk percentage.

Key Features:
- ATR-based position sizing with configurable risk percentage
- Support for both long and short positions
- Comprehensive validation and error handling
- Integration with Backtrader position sizing
- R-multiple risk management integration
"""

import numpy as np
from typing import Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    position_size: float
    risk_amount: float
    stop_distance: float
    risk_percentage: float
    is_valid: bool
    warnings: list[str]


class PositionSizingError(Exception):
    """Raised when position sizing calculation fails."""
    pass


class PositionSizer:
    """
    ATR-based position sizing calculator.
    
    This class implements true risk-per-trade position sizing where the position
    size is calculated based on the desired risk percentage and stop loss distance.
    
    Formula: Position Size = (Equity × Risk%) / |Entry Price - Stop Price|
    
    Attributes:
        default_risk_pct (float): Default risk percentage per trade (1%)
        max_position_pct (float): Maximum position size as % of equity (20%)
        min_position_value (float): Minimum position value in base currency ($100)
        max_risk_pct (float): Maximum risk percentage allowed (5%)
    """
    
    def __init__(self, 
                 default_risk_pct: float = 0.01,
                 max_position_pct: float = 0.20,
                 min_position_value: float = 100.0,
                 max_risk_pct: float = 0.05):
        """
        Initialize Position Sizer.
        
        Args:
            default_risk_pct (float): Default risk percentage per trade (1%)
            max_position_pct (float): Maximum position size as % of equity (20%)
            min_position_value (float): Minimum position value ($100)
            max_risk_pct (float): Maximum risk percentage allowed (5%)
        """
        self.default_risk_pct = default_risk_pct
        self.max_position_pct = max_position_pct
        self.min_position_value = min_position_value
        self.max_risk_pct = max_risk_pct
    
    def calculate_position_size(self,
                              equity: float,
                              entry_price: float,
                              stop_price: float,
                              risk_pct: Optional[float] = None,
                              position_type: str = "long") -> PositionSizeResult:
        """
        Calculate position size based on risk percentage and stop distance.
        
        Args:
            equity (float): Current portfolio equity
            entry_price (float): Entry price for the position
            stop_price (float): Stop loss price
            risk_pct (float): Risk percentage (defaults to default_risk_pct)
            position_type (str): "long" or "short"
            
        Returns:
            PositionSizeResult: Result with position size and validation info
        """
        warnings = []
        
        # Use default risk percentage if not provided
        if risk_pct is None:
            risk_pct = self.default_risk_pct
        
        # Validate inputs
        try:
            self._validate_inputs(equity, entry_price, stop_price, risk_pct, position_type)
        except PositionSizingError as e:
            return PositionSizeResult(
                position_size=0.0,
                risk_amount=0.0,
                stop_distance=0.0,
                risk_percentage=risk_pct,
                is_valid=False,
                warnings=[str(e)]
            )
        
        # Calculate risk amount
        risk_amount = self.calculate_risk_amount(equity, risk_pct)
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_price)
        
        # Check for minimum stop distance
        min_stop_distance = entry_price * 0.001  # 0.1% minimum stop distance
        if stop_distance < min_stop_distance:
            warnings.append(f"Stop distance {stop_distance:.4f} is very small (< 0.1% of entry price)")
        
        # Calculate position size
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            return PositionSizeResult(
                position_size=0.0,
                risk_amount=risk_amount,
                stop_distance=stop_distance,
                risk_percentage=risk_pct,
                is_valid=False,
                warnings=["Stop distance is zero - cannot calculate position size"]
            )
        
        # Validate position size
        validated_size, size_warnings = self._validate_position_size(
            position_size, equity, entry_price
        )
        warnings.extend(size_warnings)
        
        return PositionSizeResult(
            position_size=validated_size,
            risk_amount=risk_amount,
            stop_distance=stop_distance,
            risk_percentage=risk_pct,
            is_valid=len([w for w in warnings if "Error" in w]) == 0,
            warnings=warnings
        )
    
    def calculate_risk_amount(self, equity: float, risk_pct: float) -> float:
        """
        Calculate risk amount based on equity and risk percentage.
        
        Args:
            equity (float): Current portfolio equity
            risk_pct (float): Risk percentage (e.g., 0.01 for 1%)
            
        Returns:
            float: Risk amount in base currency
        """
        return equity * risk_pct
    
    def calculate_units_from_notional(self, notional_value: float, entry_price: float) -> float:
        """
        Calculate number of units from notional value.
        
        Args:
            notional_value (float): Notional value to invest
            entry_price (float): Entry price per unit
            
        Returns:
            float: Number of units to purchase
        """
        return notional_value / entry_price
    
    def calculate_backtrader_size(self, 
                                equity: float,
                                entry_price: float,
                                stop_price: float,
                                risk_pct: Optional[float] = None) -> float:
        """
        Calculate position size in Backtrader format (number of units).
        
        Args:
            equity (float): Current portfolio equity
            entry_price (float): Entry price
            stop_price (float): Stop price
            risk_pct (float): Risk percentage
            
        Returns:
            float: Number of units for Backtrader
        """
        result = self.calculate_position_size(equity, entry_price, stop_price, risk_pct)
        if result.is_valid:
            return self.calculate_units_from_notional(result.position_size, entry_price)
        else:
            return 0.0
    
    def _validate_inputs(self,
                        equity: float,
                        entry_price: float,
                        stop_price: float,
                        risk_pct: float,
                        position_type: str) -> None:
        """Validate input parameters."""
        if equity <= 0:
            raise PositionSizingError(f"Equity must be positive, got {equity}")
        
        if entry_price <= 0:
            raise PositionSizingError(f"Entry price must be positive, got {entry_price}")
        
        if stop_price <= 0:
            raise PositionSizingError(f"Stop price must be positive, got {stop_price}")
        
        if risk_pct <= 0 or risk_pct > self.max_risk_pct:
            raise PositionSizingError(
                f"Risk percentage must be between 0 and {self.max_risk_pct}, got {risk_pct}"
            )
        
        if position_type not in ["long", "short"]:
            raise PositionSizingError(f"Position type must be 'long' or 'short', got {position_type}")
        
        # Validate stop price makes sense for position type
        if position_type == "long" and stop_price >= entry_price:
            raise PositionSizingError(
                f"Long position stop price {stop_price} must be below entry price {entry_price}"
            )
        
        if position_type == "short" and stop_price <= entry_price:
            raise PositionSizingError(
                f"Short position stop price {stop_price} must be above entry price {entry_price}"
            )
    
    def _validate_position_size(self,
                               position_size: float,
                               equity: float,
                               entry_price: float) -> Tuple[float, list[str]]:
        """
        Validate and adjust position size if needed.
        
        Args:
            position_size (float): Calculated position size
            equity (float): Current equity
            entry_price (float): Entry price
            
        Returns:
            Tuple[float, list[str]]: (validated_size, warnings)
        """
        warnings = []
        validated_size = position_size
        
        # Check minimum position value
        if validated_size < self.min_position_value:
            warnings.append(f"Position size {validated_size:.2f} below minimum {self.min_position_value:.2f}")
            validated_size = self.min_position_value
        
        # Check maximum position percentage
        max_position_value = equity * self.max_position_pct
        if validated_size > max_position_value:
            warnings.append(
                f"Position size {validated_size:.2f} exceeds maximum {max_position_value:.2f} "
                f"({self.max_position_pct:.1%} of equity)"
            )
            validated_size = max_position_value
        
        # Check if position size exceeds equity
        if validated_size > equity:
            warnings.append(f"Position size {validated_size:.2f} exceeds equity {equity:.2f}")
            validated_size = equity * 0.95  # Use 95% of equity as maximum
        
        return validated_size, warnings


class RiskManagementIntegration:
    """Integration utilities for risk management systems."""
    
    @staticmethod
    def calculate_r_multiple_position_size(equity: float,
                                         entry_price: float,
                                         stop_price: float,
                                         target_r_multiple: float = 1.0,
                                         risk_pct: float = 0.01) -> float:
        """
        Calculate position size for R-multiple target.
        
        Args:
            equity (float): Current equity
            entry_price (float): Entry price
            stop_price (float): Stop price
            target_r_multiple (float): Target R-multiple
            risk_pct (float): Risk percentage
            
        Returns:
            float: Position size for target R-multiple
        """
        sizer = PositionSizer()
        result = sizer.calculate_position_size(equity, entry_price, stop_price, risk_pct)
        
        if result.is_valid:
            return result.position_size
        else:
            return 0.0
    
    @staticmethod
    def calculate_kelly_position_size(equity: float,
                                    win_rate: float,
                                    avg_win: float,
                                    avg_loss: float,
                                    max_kelly_pct: float = 0.25) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            equity (float): Current equity
            win_rate (float): Historical win rate (0-1)
            avg_win (float): Average win amount
            avg_loss (float): Average loss amount (positive)
            max_kelly_pct (float): Maximum Kelly percentage to use
            
        Returns:
            float: Kelly-optimal position size
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at maximum Kelly percentage
        kelly_fraction = min(kelly_fraction, max_kelly_pct)
        kelly_fraction = max(kelly_fraction, 0.0)  # No negative positions
        
        return equity * kelly_fraction


def create_position_size_column(df, 
                               equity_column: str = "equity",
                               entry_price_column: str = "close",
                               stop_price_column: str = "stop_price",
                               risk_pct: float = 0.01) -> np.ndarray:
    """
    Create position size column for DataFrame.
    
    Args:
        df: DataFrame with required columns
        equity_column (str): Column name for equity values
        entry_price_column (str): Column name for entry prices
        stop_price_column (str): Column name for stop prices
        risk_pct (float): Risk percentage
        
    Returns:
        np.ndarray: Position sizes for each row
    """
    sizer = PositionSizer()
    position_sizes = []
    
    for _, row in df.iterrows():
        if (equity_column in row and 
            entry_price_column in row and 
            stop_price_column in row):
            
            equity = row[equity_column]
            entry_price = row[entry_price_column]
            stop_price = row[stop_price_column]
            
            if not (np.isnan(equity) or np.isnan(entry_price) or np.isnan(stop_price)):
                result = sizer.calculate_position_size(equity, entry_price, stop_price, risk_pct)
                position_sizes.append(result.position_size if result.is_valid else 0.0)
            else:
                position_sizes.append(0.0)
        else:
            position_sizes.append(0.0)
    
    return np.array(position_sizes)


# Convenience functions for common use cases
def quick_position_size(equity: float, 
                       entry_price: float, 
                       stop_price: float,
                       risk_pct: float = 0.01) -> float:
    """Quick position size calculation."""
    sizer = PositionSizer()
    result = sizer.calculate_position_size(equity, entry_price, stop_price, risk_pct)
    return result.position_size if result.is_valid else 0.0


def validate_position_parameters(equity: float,
                               entry_price: float,
                               stop_price: float,
                               risk_pct: float = 0.01) -> bool:
    """Validate position sizing parameters."""
    sizer = PositionSizer()
    try:
        sizer._validate_inputs(equity, entry_price, stop_price, risk_pct, "long")
        return True
    except PositionSizingError:
        return False