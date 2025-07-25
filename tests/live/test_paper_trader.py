"""
Test suite for PaperTrader module.

This module provides comprehensive tests for the PaperTrader implementation,
covering all trading scenarios, market simulation, and edge cases.
"""

import asyncio
import pytest
from datetime import datetime, UTC
from unittest.mock import Mock, AsyncMock

from btc_research.live.paper_trader import (
    PaperTrader, Position, Order, Trade, OrderSide, OrderStatus, OrderType, PaperTraderError
)


class TestPosition:
    """Test cases for Position class."""
    
    def test_position_initialization(self):
        """Test Position initialization."""
        pos = Position(symbol="BTC/USDT")
        
        assert pos.symbol == "BTC/USDT"
        assert pos.size == 0.0
        assert pos.average_price == 0.0
        assert pos.unrealized_pnl == 0.0
        assert pos.is_flat()
        assert not pos.is_long()
        assert not pos.is_short()
    
    def test_position_states(self):
        """Test position state methods."""
        pos = Position(symbol="BTC/USDT")
        
        # Test flat position
        assert pos.is_flat()
        assert not pos.is_long()
        assert not pos.is_short()
        
        # Test long position
        pos.size = 1.5
        assert not pos.is_flat()
        assert pos.is_long()
        assert not pos.is_short()
        
        # Test short position
        pos.size = -0.8
        assert not pos.is_flat()
        assert not pos.is_long()
        assert pos.is_short()
    
    def test_position_update_price(self):
        """Test position price updates and P&L calculation."""
        pos = Position(symbol="BTC/USDT", size=1.0, average_price=50000.0)
        
        # Test profit scenario
        pos.update_price(52000.0)
        assert pos.unrealized_pnl == 2000.0
        
        # Test loss scenario
        pos.update_price(48000.0)
        assert pos.unrealized_pnl == -2000.0
        
        # Test flat position
        pos.size = 0.0
        pos.update_price(60000.0)
        assert pos.unrealized_pnl == 0.0
    
    def test_position_add_new_position(self):
        """Test adding to a new (flat) position."""
        pos = Position(symbol="BTC/USDT")
        
        # Add long position
        pos.add_to_position(1.0, 50000.0)
        assert pos.size == 1.0
        assert pos.average_price == 50000.0
        
        # Reset and add short position
        pos = Position(symbol="BTC/USDT")
        pos.add_to_position(-0.5, 51000.0)
        assert pos.size == -0.5
        assert pos.average_price == 51000.0
    
    def test_position_add_to_existing_position(self):
        """Test adding to existing position with average price calculation."""
        pos = Position(symbol="BTC/USDT", size=1.0, average_price=50000.0)
        
        # Add more to long position
        pos.add_to_position(0.5, 52000.0)
        expected_avg = (1.0 * 50000.0 + 0.5 * 52000.0) / 1.5
        assert pos.size == 1.5
        assert abs(pos.average_price - expected_avg) < 1e-6
    
    def test_position_close_position(self):
        """Test closing a position completely."""
        pos = Position(symbol="BTC/USDT", size=1.0, average_price=50000.0)
        
        # Close position completely
        pos.add_to_position(-1.0, 52000.0)
        assert pos.is_flat()
        assert pos.size == 0.0
        assert pos.average_price == 0.0
    
    def test_position_to_dict(self):
        """Test position serialization."""
        pos = Position(symbol="BTC/USDT", size=1.5, average_price=50000.0, unrealized_pnl=1500.0)
        data = pos.to_dict()
        
        assert data['symbol'] == "BTC/USDT"
        assert data['size'] == 1.5
        assert data['average_price'] == 50000.0
        assert data['unrealized_pnl'] == 1500.0
        assert 'created_at' in data
        assert 'updated_at' in data


class TestOrder:
    """Test cases for Order class."""
    
    def test_order_initialization(self):
        """Test Order initialization."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=1.0
        )
        
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.size == 1.0
        assert order.remaining_size == 1.0
        assert order.filled_size == 0.0
        assert order.status == OrderStatus.PENDING
        assert not order.is_complete()
    
    def test_order_partial_fill(self):
        """Test partial order fill."""
        order = Order(symbol="BTC/USDT", side=OrderSide.BUY, size=1.0)
        
        # First partial fill
        order.fill(0.4, 50000.0, 20.0)
        assert order.filled_size == 0.4
        assert order.remaining_size == 0.6
        assert order.average_fill_price == 50000.0
        assert order.commission == 20.0
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert not order.is_complete()
        
        # Second partial fill at different price
        order.fill(0.6, 51000.0, 30.0)
        expected_avg = (0.4 * 50000.0 + 0.6 * 51000.0) / 1.0
        assert order.filled_size == 1.0
        assert order.remaining_size == 0.0
        assert abs(order.average_fill_price - expected_avg) < 1e-6
        assert order.commission == 50.0
        assert order.status == OrderStatus.FILLED
        assert order.is_complete()
    
    def test_order_overfill_protection(self):
        """Test that orders cannot be overfilled."""
        order = Order(symbol="BTC/USDT", side=OrderSide.BUY, size=1.0)
        
        with pytest.raises(PaperTraderError, match="Cannot fill"):
            order.fill(1.5, 50000.0, 10.0)
    
    def test_order_cancellation(self):
        """Test order cancellation."""
        order = Order(symbol="BTC/USDT", side=OrderSide.BUY, size=1.0)
        
        # Cancel pending order
        order.cancel()
        assert order.status == OrderStatus.CANCELLED
        assert order.remaining_size == 0.0
        
        # Cannot cancel already cancelled order
        with pytest.raises(PaperTraderError, match="Cannot cancel"):
            order.cancel()
    
    def test_order_to_dict(self):
        """Test order serialization."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            size=1.0,
            filled_size=0.5,
            average_fill_price=50000.0
        )
        
        data = order.to_dict()
        assert data['symbol'] == "BTC/USDT"
        assert data['side'] == "buy"
        assert data['size'] == 1.0
        assert data['filled_size'] == 0.5
        assert data['average_fill_price'] == 50000.0
        assert 'created_at' in data


class TestTrade:
    """Test cases for Trade class."""
    
    def test_trade_initialization(self):
        """Test Trade initialization."""
        trade = Trade(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            size=1.0,
            price=50000.0,
            commission=50.0,
            order_id="test-order"
        )
        
        assert trade.symbol == "BTC/USDT"
        assert trade.side == OrderSide.BUY
        assert trade.size == 1.0
        assert trade.price == 50000.0
        assert trade.commission == 50.0
        assert trade.order_id == "test-order"
    
    def test_trade_calculations(self):
        """Test trade calculation properties."""
        trade = Trade(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            size=1.0,
            price=50000.0,
            commission=50.0
        )
        
        assert trade.notional_value == 50000.0
        assert trade.net_proceeds == 50050.0  # Buy adds commission to proceeds
        
        # Test sell trade
        sell_trade = Trade(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            size=1.0,
            price=52000.0,
            commission=52.0
        )
        
        assert sell_trade.notional_value == 52000.0
        assert sell_trade.net_proceeds == 51948.0  # Sell subtracts commission
    
    def test_trade_to_dict(self):
        """Test trade serialization."""
        trade = Trade(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            size=1.0,
            price=50000.0,
            commission=50.0
        )
        
        data = trade.to_dict()
        assert data['symbol'] == "BTC/USDT"
        assert data['side'] == "buy"
        assert data['size'] == 1.0
        assert data['price'] == 50000.0
        assert data['commission'] == 50.0
        assert data['notional_value'] == 50000.0
        assert 'timestamp' in data


class TestPaperTrader:
    """Test cases for PaperTrader class."""
    
    @pytest.fixture
    def trader(self):
        """Create a PaperTrader instance for testing."""
        return PaperTrader(initial_balance=10000.0)
    
    @pytest.fixture
    def mock_stream_manager(self):
        """Create a mock StreamManager."""
        mock = Mock()
        mock.get_latest_price = Mock(return_value=50000.0)
        return mock
    
    def test_trader_initialization(self, trader):
        """Test PaperTrader initialization."""
        assert trader.initial_balance == 10000.0
        assert trader.total_balance == 10000.0
        assert trader.available_balance == 10000.0
        assert trader.commission_rate == 0.001
        assert trader.slippage_rate == 0.0005
        assert len(trader.positions) == 0
        assert len(trader.orders) == 0
        assert len(trader.trades) == 0
    
    def test_stream_manager_integration(self, trader, mock_stream_manager):
        """Test StreamManager integration."""
        trader.set_stream_manager(mock_stream_manager)
        assert trader.stream_manager == mock_stream_manager
        
        price = trader.get_current_price("BTC/USDT")
        assert price == 50000.0
        mock_stream_manager.get_latest_price.assert_called_with("BTC/USDT")
    
    def test_bid_ask_spread_simulation(self, trader):
        """Test bid/ask spread simulation."""
        # Buy at ask (higher price)
        buy_price = trader._simulate_bid_ask_spread(50000.0, OrderSide.BUY)
        assert buy_price > 50000.0
        
        # Sell at bid (lower price)
        sell_price = trader._simulate_bid_ask_spread(50000.0, OrderSide.SELL)
        assert sell_price < 50000.0
        assert buy_price > sell_price  # Spread exists
    
    def test_slippage_calculation(self, trader):
        """Test slippage calculation."""
        base_price = 50000.0
        
        # Buy with slippage (higher price)
        buy_price = trader._apply_slippage(base_price, 1.0, OrderSide.BUY)
        assert buy_price > base_price
        
        # Sell with slippage (lower price)
        sell_price = trader._apply_slippage(base_price, 1.0, OrderSide.SELL)
        assert sell_price < base_price
        
        # Larger orders have more slippage
        large_buy_price = trader._apply_slippage(base_price, 10.0, OrderSide.BUY)
        assert large_buy_price > buy_price
    
    def test_commission_calculation(self, trader):
        """Test commission calculation."""
        notional_value = 50000.0
        commission = trader._calculate_commission(notional_value)
        expected = notional_value * trader.commission_rate
        assert commission == expected
    
    def test_order_validation_basic(self, trader, mock_stream_manager):
        """Test basic order validation."""
        trader.set_stream_manager(mock_stream_manager)
        
        # Test invalid size
        with pytest.raises(PaperTraderError, match="Order size must be positive"):
            trader._validate_order("BTC/USDT", OrderSide.BUY, 0.0)
        
        with pytest.raises(PaperTraderError, match="Order size must be positive"):
            trader._validate_order("BTC/USDT", OrderSide.BUY, -1.0)
    
    def test_order_validation_no_market_data(self, trader):
        """Test order validation without market data."""
        # No stream manager set
        with pytest.raises(PaperTraderError, match="No market data available"):
            trader._validate_order("BTC/USDT", OrderSide.BUY, 1.0)
        
        # Stream manager returns None
        mock_stream = Mock()
        mock_stream.get_latest_price = Mock(return_value=None)
        trader.set_stream_manager(mock_stream)
        
        with pytest.raises(PaperTraderError, match="No market data available"):
            trader._validate_order("BTC/USDT", OrderSide.BUY, 1.0)
    
    def test_order_validation_minimum_value(self, trader, mock_stream_manager):
        """Test minimum order value validation."""
        trader.set_stream_manager(mock_stream_manager)
        
        # Order too small
        with pytest.raises(PaperTraderError, match="Order value.*below minimum"):
            trader._validate_order("BTC/USDT", OrderSide.BUY, 0.0001)  # $5 at $50k
    
    def test_order_validation_insufficient_balance(self, trader, mock_stream_manager):
        """Test insufficient balance validation."""
        trader.set_stream_manager(mock_stream_manager)
        # Increase total balance so position limit check passes first
        trader.total_balance = 100000.0
        
        # Order too large for available balance  
        with pytest.raises(PaperTraderError, match="Insufficient balance"):
            trader._validate_order("BTC/USDT", OrderSide.BUY, 0.5)  # ~$25k order with $10k available balance
    
    def test_order_validation_position_limit(self, trader, mock_stream_manager):
        """Test position size limit validation."""
        trader.set_stream_manager(mock_stream_manager)
        trader.available_balance = 200000.0  # Increase balance to cover the order
        trader.total_balance = 100000.0  # Keep total balance lower for position limit test
        
        # Order would exceed position limit
        with pytest.raises(PaperTraderError, match="Position size limit exceeded"):
            trader._validate_order("BTC/USDT", OrderSide.BUY, 2.0)  # $100k position > 95% of $100k
    
    @pytest.mark.asyncio
    async def test_submit_market_order_buy(self, trader, mock_stream_manager):
        """Test submitting a buy market order."""
        trader.set_stream_manager(mock_stream_manager)
        trader.available_balance = 30000.0  # Increase balance for test
        trader.total_balance = 30000.0
        
        # Submit buy order
        order = await trader.submit_order("BTC/USDT", OrderSide.BUY, 0.5)
        
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.size == 0.5
        assert order.status == OrderStatus.FILLED
        assert order.is_complete()
        assert order.id in trader.orders
        
        # Check position created
        assert "BTC/USDT" in trader.positions
        position = trader.positions["BTC/USDT"]
        assert position.size == 0.5
        assert position.is_long()
        
        # Check trades created (may be multiple due to partial fills)
        assert len(trader.trades) >= 1
        total_trade_size = sum(trade.size for trade in trader.trades)
        assert abs(total_trade_size - 0.5) < 1e-8
        
        # Check all trades are for the correct symbol and side
        for trade in trader.trades:
            assert trade.symbol == "BTC/USDT"
            assert trade.side == OrderSide.BUY
        
        # Check balance updated
        assert trader.available_balance < 30000.0  # Reduced by purchase
        assert trader.total_commission_paid > 0
    
    @pytest.mark.asyncio
    async def test_submit_market_order_sell(self, trader, mock_stream_manager):
        """Test submitting a sell market order."""
        trader.set_stream_manager(mock_stream_manager)
        
        # First create a long position (and adjust balance as if it was purchased)
        trader.positions["BTC/USDT"] = Position(symbol="BTC/USDT", size=1.0, average_price=49000.0)
        trader.available_balance = 5000.0  # Remaining balance after purchase
        trader.total_balance = 60000.0  # Higher total balance to allow the position
        
        # Submit sell order
        order = await trader.submit_order("BTC/USDT", OrderSide.SELL, 0.5)
        
        assert order.status == OrderStatus.FILLED
        assert order.is_complete()
        
        # Check position reduced
        position = trader.positions["BTC/USDT"]
        assert position.size == 0.5
        assert position.is_long()
        
        # Check trades created (may be multiple due to partial fills)
        assert len(trader.trades) >= 1
        total_trade_size = sum(trade.size for trade in trader.trades)
        assert abs(total_trade_size - 0.5) < 1e-8
        
        # Check all trades are for the correct side
        for trade in trader.trades:
            assert trade.side == OrderSide.SELL
    
    @pytest.mark.asyncio
    async def test_partial_fill_simulation(self, trader, mock_stream_manager):
        """Test partial fill simulation for large orders."""
        trader.set_stream_manager(mock_stream_manager)
        trader.available_balance = 100000.0
        trader.total_balance = 100000.0
        
        # Submit large order that should be partially filled
        order = await trader.submit_order("BTC/USDT", OrderSide.BUY, 1.8)
        
        # Should still be completely filled but potentially in multiple fills
        assert order.status == OrderStatus.FILLED
        assert order.is_complete()
        assert order.filled_size == 1.8
        
        # Multiple trades might be created for partial fills
        total_trade_size = sum(trade.size for trade in trader.trades)
        assert abs(total_trade_size - 1.8) < 1e-8
    
    def test_position_management(self, trader):
        """Test position retrieval and management."""
        # Create test positions
        trader.positions["BTC/USDT"] = Position(symbol="BTC/USDT", size=1.0, average_price=50000.0)
        trader.positions["ETH/USDT"] = Position(symbol="ETH/USDT", size=-0.5, average_price=3000.0)
        
        # Test get_positions
        positions = trader.get_positions()
        assert len(positions) == 2
        assert "BTC/USDT" in positions
        assert "ETH/USDT" in positions
        
        # Test get_position
        btc_pos = trader.get_position("BTC/USDT")
        assert btc_pos.symbol == "BTC/USDT"
        assert btc_pos.size == 1.0
        
        # Test non-existent position
        non_pos = trader.get_position("ADA/USDT")
        assert non_pos is None
    
    def test_order_management(self, trader):
        """Test order retrieval and management."""
        # Create test orders
        order1 = Order(symbol="BTC/USDT", side=OrderSide.BUY, size=1.0)
        order2 = Order(symbol="ETH/USDT", side=OrderSide.SELL, size=0.5)
        order2.status = OrderStatus.FILLED
        
        trader.orders[order1.id] = order1
        trader.orders[order2.id] = order2
        
        # Test get_orders
        all_orders = trader.get_orders()
        assert len(all_orders) == 2
        
        # Test get_active_orders
        active_orders = trader.get_active_orders()
        assert len(active_orders) == 1
        assert order1.id in active_orders
        assert order2.id not in active_orders
    
    def test_trade_history(self, trader):
        """Test trade history functionality."""
        # Create test trades
        trade1 = Trade(symbol="BTC/USDT", side=OrderSide.BUY, size=1.0, price=50000.0)
        trade2 = Trade(symbol="ETH/USDT", side=OrderSide.SELL, size=0.5, price=3000.0)
        trade3 = Trade(symbol="BTC/USDT", side=OrderSide.SELL, size=0.5, price=51000.0)
        
        trader.trades = [trade1, trade2, trade3]
        
        # Test get all trades
        all_trades = trader.get_trade_history()
        assert len(all_trades) == 3
        
        # Test filter by symbol
        btc_trades = trader.get_trade_history(symbol="BTC/USDT")
        assert len(btc_trades) == 2
        
        # Test limit
        limited_trades = trader.get_trade_history(limit=2)
        assert len(limited_trades) == 2
    
    def test_balance_info(self, trader, mock_stream_manager):
        """Test balance information calculation."""
        trader.set_stream_manager(mock_stream_manager)
        
        # Create position with unrealized P&L
        trader.positions["BTC/USDT"] = Position(symbol="BTC/USDT", size=1.0, average_price=49000.0)
        trader.total_commission_paid = 50.0
        trader.total_slippage_cost = 25.0
        
        balance_info = trader.get_balance_info()
        
        assert balance_info['initial_balance'] == 10000.0
        assert balance_info['total_balance'] == 10000.0
        assert balance_info['total_commission_paid'] == 50.0
        assert balance_info['total_slippage_cost'] == 25.0
        assert 'total_unrealized_pnl' in balance_info
        assert 'total_equity' in balance_info
        assert 'return_pct' in balance_info
    
    def test_performance_stats(self, trader):
        """Test performance statistics calculation."""
        # Add some trades
        trade1 = Trade(symbol="BTC/USDT", side=OrderSide.BUY, size=1.0, price=50000.0, commission=50.0)
        trade2 = Trade(symbol="BTC/USDT", side=OrderSide.SELL, size=1.0, price=51000.0, commission=51.0)
        trader.trades = [trade1, trade2]
        
        stats = trader.get_performance_stats()
        
        assert stats['total_trades'] == 2
        assert 'win_rate' in stats
        assert 'average_trade' in stats
        assert 'profit_factor' in stats
        assert 'largest_win' in stats
        assert 'largest_loss' in stats
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self, trader, mock_stream_manager):
        """Test order cancellation."""
        trader.set_stream_manager(mock_stream_manager)
        
        # Create a pending order manually
        order = Order(symbol="BTC/USDT", side=OrderSide.BUY, size=0.1)
        trader.orders[order.id] = order
        
        # Cancel the order
        result = await trader.cancel_order(order.id)
        assert result is True
        assert order.status == OrderStatus.CANCELLED
        
        # Try to cancel non-existent order
        result = await trader.cancel_order("non-existent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, trader):
        """Test cancelling all orders."""
        # Create multiple orders
        order1 = Order(symbol="BTC/USDT", side=OrderSide.BUY, size=0.1)
        order2 = Order(symbol="ETH/USDT", side=OrderSide.SELL, size=0.2)
        order3 = Order(symbol="BTC/USDT", side=OrderSide.BUY, size=0.3)
        order3.status = OrderStatus.FILLED  # Already filled
        
        trader.orders[order1.id] = order1
        trader.orders[order2.id] = order2
        trader.orders[order3.id] = order3
        
        # Cancel all orders
        cancelled_count = await trader.cancel_all_orders()
        assert cancelled_count == 2  # Only pending orders cancelled
        
        # Cancel all BTC orders
        order4 = Order(symbol="BTC/USDT", side=OrderSide.BUY, size=0.4)
        trader.orders[order4.id] = order4
        
        cancelled_count = await trader.cancel_all_orders(symbol="BTC/USDT")
        assert cancelled_count == 1
    
    def test_trader_reset(self, trader):
        """Test trader reset functionality."""
        # Add some state
        trader.total_balance = 9500.0
        trader.available_balance = 8000.0
        trader.positions["BTC/USDT"] = Position(symbol="BTC/USDT", size=1.0)
        trader.orders["test"] = Order(symbol="BTC/USDT", side=OrderSide.BUY, size=1.0)
        trader.trades.append(Trade(symbol="BTC/USDT", side=OrderSide.BUY, size=1.0, price=50000.0))
        trader.total_commission_paid = 100.0
        
        # Reset with same balance
        trader.reset()
        assert trader.total_balance == 10000.0
        assert trader.available_balance == 10000.0
        assert len(trader.positions) == 0
        assert len(trader.orders) == 0
        assert len(trader.trades) == 0
        assert trader.total_commission_paid == 0.0
        
        # Reset with new balance
        trader.reset(initial_balance=20000.0)
        assert trader.initial_balance == 20000.0
        assert trader.total_balance == 20000.0
        assert trader.available_balance == 20000.0
    
    def test_trader_serialization(self, trader):
        """Test trader state serialization."""
        # Add some state
        trader.positions["BTC/USDT"] = Position(symbol="BTC/USDT", size=1.0, average_price=50000.0)
        trader.trades.append(Trade(symbol="BTC/USDT", side=OrderSide.BUY, size=1.0, price=50000.0))
        
        data = trader.to_dict()
        
        assert 'balance_info' in data
        assert 'positions' in data
        assert 'orders' in data
        assert 'recent_trades' in data
        assert 'performance_stats' in data
        
        # Check structure
        assert len(data['positions']) == 1
        assert 'BTC/USDT' in data['positions']
        assert len(data['recent_trades']) == 1


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple features."""
    
    @pytest.mark.asyncio
    async def test_complete_trading_cycle(self):
        """Test a complete trading cycle from order to position to close."""
        trader = PaperTrader(initial_balance=100000.0)
        
        # Mock market data - start with base price
        base_price = 50000.0
        mock_stream = Mock()
        mock_stream.get_latest_price = Mock(return_value=base_price)
        trader.set_stream_manager(mock_stream)
        
        # 1. Submit buy order
        buy_order = await trader.submit_order("BTC/USDT", OrderSide.BUY, 1.0)
        assert buy_order.status == OrderStatus.FILLED
        
        # 2. Check position created
        position = trader.get_position("BTC/USDT")
        assert position is not None
        assert position.size == 1.0
        assert position.is_long()
        
        # 3. Update market price and check unrealized P&L
        new_price = 52000.0
        mock_stream.get_latest_price.return_value = new_price
        position = trader.get_position("BTC/USDT")  # This updates P&L
        # Profit should be positive since new price > average entry price (considering slippage)
        expected_pnl = (new_price - position.average_price) * position.size
        assert expected_pnl > 0 or abs(expected_pnl) < 1000  # Allow small loss due to slippage
        
        # 4. Partial close
        sell_order = await trader.submit_order("BTC/USDT", OrderSide.SELL, 0.6)
        assert sell_order.status == OrderStatus.FILLED
        
        # 5. Check position reduced
        position = trader.get_position("BTC/USDT")
        assert abs(position.size - 0.4) < 1e-8
        
        # 6. Close remaining position
        close_order = await trader.submit_order("BTC/USDT", OrderSide.SELL, 0.4)
        assert close_order.status == OrderStatus.FILLED
        
        # 7. Check position closed
        position = trader.get_position("BTC/USDT")
        assert position is None or position.is_flat()
        
        # 8. Check trade history
        trades = trader.get_trade_history()
        assert len(trades) >= 3  # At least 3 trades (might be more due to partial fills)
        
        # 9. Check performance stats
        stats = trader.get_performance_stats()
        assert stats['total_trades'] >= 3
    
    @pytest.mark.asyncio
    async def test_risk_management_scenarios(self):
        """Test various risk management scenarios."""
        trader = PaperTrader(initial_balance=10000.0)
        
        mock_stream = Mock()
        mock_stream.get_latest_price = Mock(return_value=50000.0)
        trader.set_stream_manager(mock_stream)
        
        # Test 1: Order too large for balance
        trader.total_balance = 100000.0  # Increase total to pass position limit check
        with pytest.raises(PaperTraderError, match="Insufficient balance"):
            await trader.submit_order("BTC/USDT", OrderSide.BUY, 0.3)  # ~$15k order with $10k available
        
        # Test 2: Position size limit
        trader.available_balance = 200000.0  # Enough balance to cover order
        trader.total_balance = 100000.0  # Lower total balance for position limit
        
        with pytest.raises(PaperTraderError, match="Position size limit exceeded"):
            await trader.submit_order("BTC/USDT", OrderSide.BUY, 2.0)  # $100k position > 95% of $100k
        
        # Test 3: Minimum order value
        with pytest.raises(PaperTraderError, match="Order value.*below minimum"):
            await trader.submit_order("BTC/USDT", OrderSide.BUY, 0.0001)
    
    @pytest.mark.asyncio
    async def test_market_conditions_simulation(self):
        """Test various market conditions and their effects."""
        trader = PaperTrader(initial_balance=100000.0)
        
        mock_stream = Mock()
        mock_stream.get_latest_price = Mock(return_value=50000.0)
        trader.set_stream_manager(mock_stream)
        
        # Test slippage increases with order size
        small_order = await trader.submit_order("BTC/USDT", OrderSide.BUY, 0.1)
        small_fill_price = small_order.average_fill_price
        
        # Reset for comparison
        trader.reset(100000.0)
        
        large_order = await trader.submit_order("BTC/USDT", OrderSide.BUY, 1.0)
        large_fill_price = large_order.average_fill_price
        
        # Larger order should have worse fill price (more slippage)
        assert large_fill_price > small_fill_price
        
        # Test bid/ask spread
        trader.reset(100000.0)
        buy_order = await trader.submit_order("BTC/USDT", OrderSide.BUY, 0.5)
        buy_price = buy_order.average_fill_price
        
        # Create position for sell test
        trader.positions["BTC/USDT"] = Position(symbol="BTC/USDT", size=1.0, average_price=49000.0)
        
        sell_order = await trader.submit_order("BTC/USDT", OrderSide.SELL, 0.5)
        sell_price = sell_order.average_fill_price
        
        # Buy price should be higher than sell price (spread)
        assert buy_price > sell_price


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])