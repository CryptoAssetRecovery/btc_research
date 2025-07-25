"""
Tests for the StreamManager class.

This module contains unit tests for the StreamManager class to verify
WebSocket connection handling, data buffering, and integration compatibility.
"""

import asyncio
import json
import threading
import time
import unittest
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from btc_research.live.stream_manager import StreamManager, StreamManagerError


class TestStreamManager(unittest.TestCase):
    """Test suite for StreamManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ['BTC/USDT', 'ETH/USDT']
        self.timeframes = ['1m', '5m', '15m']
        
        # Mock Redis to avoid requiring Redis server for tests
        with patch('btc_research.live.stream_manager.redis.Redis') as mock_redis:
            mock_redis.return_value.ping.return_value = True
            self.stream_manager = StreamManager(
                symbols=self.symbols,
                timeframes=self.timeframes,
                buffer_size=100
            )

    def test_initialization(self):
        """Test StreamManager initialization."""
        self.assertEqual(self.stream_manager.symbols, self.symbols)
        self.assertEqual(self.stream_manager.timeframes, self.timeframes)
        self.assertEqual(self.stream_manager.buffer_size, 100)
        self.assertFalse(self.stream_manager.is_running)
        
        # Check buffer structure
        for symbol in self.symbols:
            self.assertIn(symbol, self.stream_manager.buffers)
            for tf in self.timeframes:
                self.assertIn(tf, self.stream_manager.buffers[symbol])
                self.assertEqual(len(self.stream_manager.buffers[symbol][tf]), 0)

    def test_invalid_initialization(self):
        """Test StreamManager initialization with invalid parameters."""
        # Test empty symbols
        with self.assertRaises(StreamManagerError):
            StreamManager([], self.timeframes)
        
        # Test empty timeframes
        with self.assertRaises(StreamManagerError):
            StreamManager(self.symbols, [])
        
        # Test invalid timeframe
        with self.assertRaises(StreamManagerError):
            StreamManager(self.symbols, ['1m', '2m'])  # 2m not supported
        
        # Test invalid symbol format
        with self.assertRaises(StreamManagerError):
            StreamManager(['BTCUSDT'], self.timeframes)  # Missing '/'

    def test_symbol_to_binance_format(self):
        """Test symbol format conversion."""
        self.assertEqual(
            self.stream_manager._symbol_to_binance_format('BTC/USDT'),
            'btcusdt'
        )
        self.assertEqual(
            self.stream_manager._symbol_to_binance_format('ETH/BTC'),
            'ethbtc'
        )

    def test_websocket_url_creation(self):
        """Test WebSocket URL creation."""
        expected_url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
        actual_url = self.stream_manager._create_websocket_url('BTC/USDT')
        self.assertEqual(actual_url, expected_url)

    @patch('btc_research.live.stream_manager.websocket.WebSocketApp')
    def test_connect_symbol(self, mock_ws_app):
        """Test WebSocket connection for a symbol."""
        mock_ws = Mock()
        mock_ws_app.return_value = mock_ws
        
        # Run async test
        async def run_test():
            await self.stream_manager._connect_symbol('BTC/USDT')
            
            # Check WebSocket app was created
            mock_ws_app.assert_called_once()
            self.assertIn('BTC/USDT', self.stream_manager.ws_connections)
        
        asyncio.run(run_test())

    def test_process_tick(self):
        """Test tick processing and candle updates."""
        symbol = 'BTC/USDT'
        
        # Simulate Binance trade message
        tick_data = {
            'e': 'trade',
            'p': '50000.00',  # price
            'q': '0.001',     # quantity
            'T': int(time.time() * 1000)  # timestamp
        }
        
        async def run_test():
            await self.stream_manager._process_tick(symbol, tick_data)
            
            # Check that current tick data was updated
            current = self.stream_manager.current_ticks[symbol]
            self.assertEqual(current['price'], 50000.00)
            self.assertEqual(current['close'], 50000.00)
            
            # Check that candles were created for each timeframe
            for tf in self.timeframes:
                buffer = self.stream_manager.buffers[symbol][tf]
                self.assertGreater(len(buffer), 0)
                
                candle = buffer[-1]  # Most recent candle
                self.assertEqual(candle['close'], 50000.00)
        
        asyncio.run(run_test())

    def test_get_data_empty(self):
        """Test getting data when no data is available."""
        df = self.stream_manager.get_data('BTC/USDT', '1m')
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
        self.assertListEqual(list(df.columns), self.stream_manager.REQUIRED_COLUMNS)
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_get_data_invalid_symbol(self):
        """Test getting data for invalid symbol."""
        with self.assertRaises(StreamManagerError):
            self.stream_manager.get_data('INVALID/SYMBOL', '1m')

    def test_get_data_invalid_timeframe(self):
        """Test getting data for invalid timeframe."""
        with self.assertRaises(StreamManagerError):
            self.stream_manager.get_data('BTC/USDT', '2m')

    def test_get_latest_price_no_data(self):
        """Test getting latest price when no data is available."""
        price = self.stream_manager.get_latest_price('BTC/USDT')
        self.assertIsNone(price)

    def test_get_statistics(self):
        """Test getting performance statistics."""
        stats = self.stream_manager.get_statistics()
        
        self.assertIn('is_running', stats)
        self.assertIn('symbols', stats)
        self.assertIn('timeframes', stats)
        self.assertIn('tick_count', stats)
        self.assertIn('avg_processing_time_ms', stats)
        self.assertIn('buffer_sizes', stats)
        
        self.assertEqual(stats['symbols'], self.symbols)
        self.assertEqual(stats['timeframes'], self.timeframes)
        self.assertFalse(stats['is_running'])

    def test_dataframe_compatibility(self):
        """Test that returned DataFrames are compatible with Engine class."""
        # Add some mock data to buffer
        symbol = 'BTC/USDT'
        timeframe = '1m'
        
        from datetime import datetime, UTC
        mock_candle = {
            'timestamp': datetime.now(UTC),
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 1.5
        }
        
        self.stream_manager.buffers[symbol][timeframe].append(mock_candle)
        
        df = self.stream_manager.get_data(symbol, timeframe)
        
        # Check DataFrame structure matches Engine expectations
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(str(df.index.tz), 'UTC')
        self.assertIsNone(df.index.freq)  # Frequency should be None for compatibility
        
        # Check required columns
        for col in self.stream_manager.REQUIRED_COLUMNS:
            self.assertIn(col, df.columns)
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))

    def test_stop(self):
        """Test stopping the StreamManager."""
        # Mock some WebSocket connections
        mock_ws1 = Mock()
        mock_ws2 = Mock()
        self.stream_manager.ws_connections = {
            'BTC/USDT': mock_ws1,
            'ETH/USDT': mock_ws2
        }
        self.stream_manager.is_running = True
        
        async def run_test():
            await self.stream_manager.stop()
            
            # Check that connections were closed
            mock_ws1.close.assert_called_once()
            mock_ws2.close.assert_called_once()
            self.assertFalse(self.stream_manager.is_running)
            self.assertEqual(len(self.stream_manager.ws_connections), 0)
        
        asyncio.run(run_test())


class TestStreamManagerIntegration(unittest.TestCase):
    """Integration tests for StreamManager with mocked external dependencies."""

    @patch('btc_research.live.stream_manager.redis.Redis')
    @patch('btc_research.live.stream_manager.websocket.WebSocketApp')
    def test_full_workflow_mock(self, mock_ws_app, mock_redis):
        """Test complete workflow with mocked dependencies."""
        # Setup mocks
        mock_redis.return_value.ping.return_value = True
        mock_ws = Mock()
        mock_ws_app.return_value = mock_ws
        
        # Create StreamManager
        stream_manager = StreamManager(['BTC/USDT'], ['1m'])
        
        # Test full workflow
        async def run_test():
            await stream_manager.start()
            self.assertTrue(stream_manager.is_running)
            
            # Simulate some tick data
            tick_data = {
                'e': 'trade',
                'p': '50000.00',
                'q': '0.001',
                'T': int(time.time() * 1000)
            }
            
            await stream_manager._process_tick('BTC/USDT', tick_data)
            
            # Get data
            df = stream_manager.get_data('BTC/USDT', '1m')
            self.assertGreater(len(df), 0)
            
            # Get statistics
            stats = stream_manager.get_statistics()
            self.assertGreater(stats['tick_count'], 0)
            
            await stream_manager.stop()
            self.assertFalse(stream_manager.is_running)
        
        # Run async test
        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()