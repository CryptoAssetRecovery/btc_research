"""
PaperTrader for simulating live trading with realistic market conditions.

This module implements paper trading functionality that closely mimics real trading
environments, including realistic slippage, commissions, latency simulation, and
comprehensive position tracking. It integrates with StreamManager for real-time
market data and provides comprehensive trade recording.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd

__all__ = ["PaperTrader", "Position", "Order", "Trade", "OrderSide", "OrderStatus", "OrderType", "PaperTraderError"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperTraderError(Exception):
    """Base exception for PaperTrader related errors."""
    pass


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"  # Future enhancement


@dataclass
class Position:
    """
    Represents a position in a trading symbol.
    
    Tracks net position size, average entry price, unrealized P&L,
    and provides methods for position management calculations.
    """
    symbol: str
    size: float = 0.0  # Net position size (positive = long, negative = short)
    average_price: float = 0.0  # Average entry price
    unrealized_pnl: float = 0.0  # Unrealized P&L based on current market price
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 0
    
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < 0
    
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return abs(self.size) < 1e-8  # Account for floating point precision
    
    def update_price(self, current_price: float) -> None:
        """Update unrealized P&L based on current market price."""
        if self.is_flat():
            self.unrealized_pnl = 0.0
        else:
            self.unrealized_pnl = (current_price - self.average_price) * self.size
        self.updated_at = datetime.now(UTC)
    
    def add_to_position(self, size: float, price: float) -> None:
        """
        Add to existing position and update average price.
        
        Args:
            size: Size to add (positive for buy, negative for sell)
            price: Fill price
        """
        if self.is_flat():
            # Starting new position
            self.size = size
            self.average_price = price
        else:
            # Adding to existing position
            old_notional = self.size * self.average_price
            new_notional = size * price
            new_size = self.size + size
            
            if abs(new_size) < 1e-8:
                # Position closed
                self.size = 0.0
                self.average_price = 0.0
            else:
                # Update average price
                self.size = new_size
                self.average_price = (old_notional + new_notional) / new_size
        
        self.updated_at = datetime.now(UTC)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'size': self.size,
            'average_price': self.average_price,
            'unrealized_pnl': self.unrealized_pnl,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class Order:
    """
    Represents a trading order with execution tracking.
    
    Supports market orders with realistic fill simulation including
    slippage, partial fills, and latency simulation.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    size: float = 0.0
    price: Optional[float] = None  # For limit orders (future enhancement)
    filled_size: float = 0.0
    remaining_size: float = 0.0
    average_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    commission: float = 0.0
    slippage: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize remaining size after object creation."""
        if self.remaining_size == 0.0:
            self.remaining_size = self.size
    
    def is_complete(self) -> bool:
        """Check if order is completely filled."""
        return abs(self.remaining_size) < 1e-8
    
    def fill(self, size: float, price: float, commission: float = 0.0) -> None:
        """
        Fill order partially or completely.
        
        Args:
            size: Size to fill
            price: Fill price
            commission: Commission charged on this fill
        """
        if size > self.remaining_size + 1e-8:  # Allow small floating point errors
            raise PaperTraderError(f"Cannot fill {size} when only {self.remaining_size} remaining")
        
        # Update fill tracking
        old_filled_notional = self.filled_size * self.average_fill_price if self.filled_size > 0 else 0.0
        new_filled_notional = size * price
        self.filled_size += size
        self.remaining_size -= size
        self.commission += commission
        
        # Update average fill price
        if self.filled_size > 0:
            self.average_fill_price = (old_filled_notional + new_filled_notional) / self.filled_size
        
        # Update status
        if self.is_complete():
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now(UTC)
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def cancel(self) -> None:
        """Cancel the order."""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise PaperTraderError(f"Cannot cancel order in status {self.status}")
        
        self.status = OrderStatus.CANCELLED
        self.remaining_size = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for serialization."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'size': self.size,
            'price': self.price,
            'filled_size': self.filled_size,
            'remaining_size': self.remaining_size,
            'average_fill_price': self.average_fill_price,
            'status': self.status.value,
            'commission': self.commission,
            'slippage': self.slippage,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None
        }


@dataclass
class Trade:
    """
    Represents a completed trade transaction.
    
    Records the details of executed trades for performance analysis
    and trade history tracking.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    size: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    order_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of the trade."""
        return abs(self.size * self.price)
    
    @property
    def net_proceeds(self) -> float:
        """Calculate net proceeds after commission."""
        proceeds = self.size * self.price
        return proceeds - self.commission if self.side == OrderSide.SELL else proceeds + self.commission
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for serialization."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'price': self.price,
            'commission': self.commission,
            'slippage': self.slippage,
            'order_id': self.order_id,
            'timestamp': self.timestamp.isoformat(),
            'notional_value': self.notional_value,
            'net_proceeds': self.net_proceeds
        }


class PaperTrader:
    """
    Paper trading engine with realistic market simulation.
    
    Provides realistic order execution simulation including:
    - Market orders with bid/ask spread simulation
    - Realistic slippage (0.05% configurable)
    - Commission model (0.1% per trade, Binance standard)
    - Net position tracking with average entry price calculation
    - Separate available/total balance tracking
    - Partial fills on large orders
    - 50-100ms latency simulation for order acknowledgment
    - Position size validation and balance checks
    - Trade history recording with timestamps
    - Unrealized P&L calculations
    
    Example:
        >>> trader = PaperTrader(initial_balance=10000.0)
        >>> order = await trader.submit_order('BTC/USDT', OrderSide.BUY, 0.1)
        >>> positions = trader.get_positions()
        >>> trades = trader.get_trade_history()
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1% per trade (Binance standard)
        slippage_rate: float = 0.0005,   # 0.05% slippage
        min_latency_ms: int = 50,        # Minimum order acknowledgment latency
        max_latency_ms: int = 100,       # Maximum order acknowledgment latency
        max_position_pct: float = 0.95,  # Maximum position as percentage of balance
        min_order_value: float = 10.0,   # Minimum order value in base currency
    ):
        """
        Initialize the PaperTrader.
        
        Args:
            initial_balance: Starting balance in base currency
            commission_rate: Commission rate per trade (0.001 = 0.1%)
            slippage_rate: Slippage rate for market orders (0.0005 = 0.05%)
            min_latency_ms: Minimum order processing latency in milliseconds
            max_latency_ms: Maximum order processing latency in milliseconds
            max_position_pct: Maximum position size as percentage of balance
            min_order_value: Minimum order value in base currency
        """
        self.initial_balance = initial_balance
        self.total_balance = initial_balance
        self.available_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.max_position_pct = max_position_pct
        self.min_order_value = min_order_value
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        
        # Market data interface (will be set by caller)
        self.stream_manager = None
        
        # Performance tracking
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.orders_processed = 0
        
        logger.info(f"PaperTrader initialized with ${initial_balance:,.2f} balance")
    
    def set_stream_manager(self, stream_manager) -> None:
        """Set the StreamManager for market data access."""
        self.stream_manager = stream_manager
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if not available
        """
        if self.stream_manager:
            return self.stream_manager.get_latest_price(symbol)
        return None
    
    def _simulate_bid_ask_spread(self, price: float, side: OrderSide) -> float:
        """
        Simulate bid/ask spread for realistic fill prices.
        
        Args:
            price: Mid price
            side: Order side
            
        Returns:
            Adjusted price accounting for spread
        """
        # Simulate 0.02% spread (1 bps each side)
        spread_pct = 0.0001
        
        if side == OrderSide.BUY:
            # Buy at ask (higher price)
            return price * (1 + spread_pct)
        else:
            # Sell at bid (lower price)  
            return price * (1 - spread_pct)
    
    def _apply_slippage(self, price: float, size: float, side: OrderSide) -> float:
        """
        Apply slippage based on order size and market conditions.
        
        Args:
            price: Base price
            size: Order size
            side: Order side
            
        Returns:
            Price with slippage applied
        """
        # Increased slippage for larger orders (simplified model)
        size_factor = min(1.0 + (size * 0.1), 2.0)  # Up to 2x slippage for large orders
        slippage = self.slippage_rate * size_factor
        
        if side == OrderSide.BUY:
            # Worse fill price for buys (higher)
            return price * (1 + slippage)
        else:
            # Worse fill price for sells (lower)
            return price * (1 - slippage)
    
    def _calculate_commission(self, notional_value: float) -> float:
        """Calculate commission for a trade."""
        return notional_value * self.commission_rate
    
    def _validate_order(self, symbol: str, side: OrderSide, size: float) -> None:
        """
        Validate order parameters before submission.
        
        Args:
            symbol: Trading symbol
            side: Order side
            size: Order size
            
        Raises:
            PaperTraderError: If order validation fails
        """
        if size <= 0:
            raise PaperTraderError("Order size must be positive")
        
        # Get current price for validation
        current_price = self.get_current_price(symbol)
        if current_price is None:
            raise PaperTraderError(f"No market data available for {symbol}")
        
        # Check minimum order value
        notional_value = size * current_price
        if notional_value < self.min_order_value:
            raise PaperTraderError(f"Order value ${notional_value:.2f} below minimum ${self.min_order_value:.2f}")
        
        # Check position size limits first (before balance check)
        current_position = self.positions.get(symbol, Position(symbol=symbol))
        new_size = current_position.size + (size if side == OrderSide.BUY else -size)
        position_value = abs(new_size * current_price)
        
        if position_value > self.total_balance * self.max_position_pct:
            max_position_value = self.total_balance * self.max_position_pct
            raise PaperTraderError(
                f"Position size limit exceeded: ${position_value:.2f} > ${max_position_value:.2f}"
            )
        
        # Check available balance for buy orders
        if side == OrderSide.BUY:
            # Estimate total cost including commission and slippage
            estimated_price = self._apply_slippage(
                self._simulate_bid_ask_spread(current_price, side), size, side
            )
            estimated_cost = size * estimated_price
            estimated_commission = self._calculate_commission(estimated_cost)
            total_cost = estimated_cost + estimated_commission
            
            if total_cost > self.available_balance:
                raise PaperTraderError(
                    f"Insufficient balance: ${total_cost:.2f} required, ${self.available_balance:.2f} available"
                )
    
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET
    ) -> Order:
        """
        Submit a trading order for execution.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            side: Order side (BUY or SELL)
            size: Order size in base currency units
            order_type: Order type (currently only MARKET supported)
            
        Returns:
            Order object with execution details
            
        Raises:
            PaperTraderError: If order validation or execution fails
        """
        # Validate order
        self._validate_order(symbol, side, size)
        
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size
        )
        
        # Store order
        self.orders[order.id] = order
        
        # Simulate latency
        import random
        latency_ms = random.randint(self.min_latency_ms, self.max_latency_ms)
        await asyncio.sleep(latency_ms / 1000.0)
        
        # Update order status
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now(UTC)
        
        # Execute market order immediately
        if order_type == OrderType.MARKET:
            await self._execute_market_order(order)
        
        self.orders_processed += 1
        
        logger.info(f"Order {order.id[:8]} submitted: {side.value} {size} {symbol}")
        
        return order
    
    async def _execute_market_order(self, order: Order) -> None:
        """
        Execute a market order with realistic market simulation.
        
        Args:
            order: Order to execute
        """
        current_price = self.get_current_price(order.symbol)
        if current_price is None:
            order.status = OrderStatus.REJECTED
            logger.error(f"Order {order.id[:8]} rejected: No market data for {order.symbol}")
            return
        
        # Simulate partial fills for large orders
        remaining_size = order.size
        fill_price_base = self._simulate_bid_ask_spread(current_price, order.side)
        
        while remaining_size > 1e-8:
            # Determine fill size (may be partial for large orders)
            fill_size = min(remaining_size, order.size * 0.8)  # Max 80% fill at once for large orders
            if remaining_size <= order.size * 0.2:  # Fill remainder if small
                fill_size = remaining_size
            
            # Apply slippage
            fill_price = self._apply_slippage(fill_price_base, fill_size, order.side)
            
            # Calculate commission
            notional_value = fill_size * fill_price
            commission = self._calculate_commission(notional_value)
            
            # Fill the order
            order.fill(fill_size, fill_price, commission)
            
            # Update balances and positions
            self._process_fill(order, fill_size, fill_price, commission)
            
            # Create trade record
            trade = Trade(
                symbol=order.symbol,
                side=order.side,
                size=fill_size,
                price=fill_price,
                commission=commission,
                slippage=abs(fill_price - fill_price_base),
                order_id=order.id
            )
            self.trades.append(trade)
            
            remaining_size -= fill_size
            
            # Simulate small delay between partial fills
            if remaining_size > 1e-8:
                await asyncio.sleep(0.01)  # 10ms delay
            
            logger.info(f"Fill: {fill_size} {order.symbol} at ${fill_price:.2f} (commission: ${commission:.2f})")
    
    def _process_fill(self, order: Order, fill_size: float, fill_price: float, commission: float) -> None:
        """
        Process order fill and update positions and balances.
        
        Args:
            order: Order being filled
            fill_size: Size of this fill
            fill_price: Price of this fill
            commission: Commission for this fill
        """
        symbol = order.symbol
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        
        # Calculate position change
        position_change = fill_size if order.side == OrderSide.BUY else -fill_size
        
        # Update position
        position.add_to_position(position_change, fill_price)
        
        # Update balances
        notional_value = fill_size * fill_price
        
        if order.side == OrderSide.BUY:
            # Buying: reduce available balance
            total_cost = notional_value + commission
            self.available_balance -= total_cost
            self.total_balance -= commission  # Only commission reduces total balance
        else:
            # Selling: increase available balance
            net_proceeds = notional_value - commission
            self.available_balance += net_proceeds
            self.total_balance -= commission  # Only commission reduces total balance
        
        # Track costs
        self.total_commission_paid += commission
        self.total_slippage_cost += abs(fill_price - self.get_current_price(symbol) or fill_price)
        
        # Update unrealized P&L for all positions
        self._update_unrealized_pnl()
        
        # Clean up empty positions
        if position.is_flat():
            del self.positions[symbol]
    
    def _update_unrealized_pnl(self) -> None:
        """Update unrealized P&L for all positions based on current market prices."""
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price is not None:
                position.update_price(current_price)
    
    def get_positions(self) -> Dict[str, Position]:
        """
        Get current positions.
        
        Returns:
            Dictionary mapping symbols to Position objects
        """
        self._update_unrealized_pnl()
        return self.positions.copy()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position object or None if no position
        """
        position = self.positions.get(symbol)
        if position:
            current_price = self.get_current_price(symbol)
            if current_price is not None:
                position.update_price(current_price)
        return position
    
    def get_orders(self) -> Dict[str, Order]:
        """
        Get all orders.
        
        Returns:
            Dictionary mapping order IDs to Order objects
        """
        return self.orders.copy()
    
    def get_active_orders(self) -> Dict[str, Order]:
        """
        Get active (non-terminal) orders.
        
        Returns:
            Dictionary mapping order IDs to active Order objects
        """
        return {
            oid: order for oid, order in self.orders.items()
            if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
        }
    
    def get_trade_history(self, symbol: Optional[str] = None, limit: Optional[int] = None) -> List[Trade]:
        """
        Get trade history.
        
        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of trades to return (optional)
            
        Returns:
            List of Trade objects sorted by timestamp (newest first)
        """
        trades = self.trades
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        # Sort by timestamp (newest first)
        trades = sorted(trades, key=lambda t: t.timestamp, reverse=True)
        
        if limit:
            trades = trades[:limit]
        
        return trades
    
    def get_balance_info(self) -> Dict[str, float]:
        """
        Get comprehensive balance information.
        
        Returns:
            Dictionary with balance details
        """
        self._update_unrealized_pnl()
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_equity = self.total_balance + total_unrealized_pnl
        
        return {
            'initial_balance': self.initial_balance,
            'total_balance': self.total_balance,
            'available_balance': self.available_balance,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_equity': total_equity,
            'total_commission_paid': self.total_commission_paid,
            'total_slippage_cost': self.total_slippage_cost,
            'return_pct': (total_equity - self.initial_balance) / self.initial_balance,
            'orders_processed': self.orders_processed
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        balance_info = self.get_balance_info()
        trades = self.get_trade_history()
        
        if not trades:
            return {
                **balance_info,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'average_trade': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0
            }
        
        # Calculate trade statistics
        realized_pnl = []
        winning_trades = 0
        losing_trades = 0
        
        # Simple P&L calculation (needs position tracking for accurate P&L)
        for trade in trades:
            # This is simplified - real P&L calculation would require position tracking
            trade_pnl = -trade.commission  # At minimum, we lose commission
            realized_pnl.append(trade_pnl)
            
            if trade_pnl > 0:
                winning_trades += 1
            elif trade_pnl < 0:
                losing_trades += 1
        
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        average_trade = sum(realized_pnl) / len(realized_pnl) if realized_pnl else 0.0
        largest_win = max(realized_pnl) if realized_pnl else 0.0
        largest_loss = min(realized_pnl) if realized_pnl else 0.0
        
        # Profit factor
        gross_profit = sum(pnl for pnl in realized_pnl if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in realized_pnl if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        return {
            **balance_info,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'average_trade': average_trade,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor
        }
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if order was cancelled, False if not found or already terminal
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        try:
            order.cancel()
            logger.info(f"Order {order_id[:8]} cancelled")
            return True
        except PaperTraderError:
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all active orders.
        
        Args:
            symbol: Symbol to filter by (optional)
            
        Returns:
            Number of orders cancelled
        """
        active_orders = self.get_active_orders()
        
        if symbol:
            active_orders = {oid: order for oid, order in active_orders.items() if order.symbol == symbol}
        
        cancelled_count = 0
        for order_id in active_orders:
            if await self.cancel_order(order_id):
                cancelled_count += 1
        
        return cancelled_count
    
    def reset(self, initial_balance: Optional[float] = None) -> None:
        """
        Reset the paper trader to initial state.
        
        Args:
            initial_balance: New initial balance (optional, uses current if not specified)
        """
        if initial_balance is not None:
            self.initial_balance = initial_balance
        
        self.total_balance = self.initial_balance
        self.available_balance = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.orders_processed = 0
        
        logger.info(f"PaperTrader reset with ${self.initial_balance:,.2f} balance")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PaperTrader state to dictionary for serialization.
        
        Returns:
            Dictionary representation of the PaperTrader state
        """
        return {
            'balance_info': self.get_balance_info(),
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'orders': {oid: order.to_dict() for oid, order in self.orders.items()},
            'recent_trades': [trade.to_dict() for trade in self.get_trade_history(limit=100)],
            'performance_stats': self.get_performance_stats()
        }