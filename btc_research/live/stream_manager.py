"""
StreamManager for real-time data streaming via WebSocket connections.

This module provides the StreamManager class that connects to Binance WebSocket streams
for real-time market data, maintains circular buffers for multiple timeframes,
and integrates with the existing Engine class for paper trading.
"""

import asyncio
import json
import logging
import time
import warnings
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import redis
import websocket

__all__ = ["StreamManager", "StreamManagerError"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamManagerError(Exception):
    """Base exception for StreamManager related errors."""
    pass


class StreamManager:
    """
    Real-time data streaming manager with WebSocket connections and Redis caching.
    
    This class manages WebSocket connections to Binance for real-time data streaming,
    maintains circular buffers for multiple timeframes, and provides DataFrame
    compatibility with the existing Engine class.
    
    Key features:
    - WebSocket connections to Binance for real-time tick data
    - Support for multiple symbols and timeframes
    - Circular buffers (deque) for each timeframe with configurable size
    - Redis integration for cross-process data sharing
    - Exponential backoff reconnection strategy
    - DataFrame compatibility with existing Engine class
    - Forward-fill logic for timeframe alignment
    - Performance optimized for <10ms tick processing
    
    Example:
        >>> symbols = ['BTC/USDT', 'BTC/USDC']
        >>> timeframes = ['1m', '5m', '15m', '30m', '1h']
        >>> stream_manager = StreamManager(symbols, timeframes)
        >>> await stream_manager.start()
        >>> df = stream_manager.get_data('BTC/USDT', '1m')
    """
    
    # Supported timeframes and their minute equivalents
    TIMEFRAMES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
    }
    
    # Required OHLCV columns
    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
    
    # Binance WebSocket base URL
    BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws/"
    
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        buffer_size: int = 1000,
        redis_host: str = "redis",
        redis_port: int = 6379,
        redis_db: int = 0,
    ):
        """
        Initialize the StreamManager.
        
        Args:
            symbols: List of trading symbols (e.g., ['BTC/USDT', 'BTC/USDC'])
            timeframes: List of timeframes to maintain (e.g., ['1m', '5m', '15m'])
            buffer_size: Maximum number of candles to keep in memory per timeframe
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            
        Raises:
            StreamManagerError: If initialization fails
        """
        self._validate_inputs(symbols, timeframes)
        
        self.symbols = symbols
        self.timeframes = timeframes
        self.buffer_size = buffer_size
        
        # Initialize circular buffers for each symbol and timeframe
        self.buffers: Dict[str, Dict[str, deque]] = {}
        for symbol in symbols:
            self.buffers[symbol] = {}
            for tf in timeframes:
                self.buffers[symbol][tf] = deque(maxlen=buffer_size)
        
        # WebSocket connections
        self.ws_connections: Dict[str, websocket.WebSocketApp] = {}
        self.is_running = False
        
        # Redis client for cross-process data sharing
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False  # Keep as bytes for pandas serialization
            )
            # Test connection
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Continuing without Redis caching.")
            self.redis_client = None
        
        # Reconnection parameters
        self.reconnect_delay = 1  # Start with 1 second
        self.max_reconnect_delay = 60  # Max 60 seconds
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Performance tracking
        self.tick_count = 0
        self.last_tick_time = time.time()
        self.processing_times = deque(maxlen=1000)
        
        # Current tick data for aggregation
        self.current_ticks: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            self.current_ticks[symbol] = {
                'price': 0.0,
                'volume': 0.0,
                'timestamp': 0,
                'open': 0.0,
                'high': 0.0,
                'low': 0.0,
                'close': 0.0,
                'candle_start': {}  # Track candle start times for each timeframe
            }
    
    def _validate_inputs(self, symbols: List[str], timeframes: List[str]) -> None:
        """Validate input parameters."""
        if not symbols:
            raise StreamManagerError("At least one symbol must be specified")
        
        if not timeframes:
            raise StreamManagerError("At least one timeframe must be specified")
        
        # Validate timeframes
        for tf in timeframes:
            if tf not in self.TIMEFRAMES:
                available = list(self.TIMEFRAMES.keys())
                raise StreamManagerError(f"Unsupported timeframe '{tf}'. Available: {available}")
        
        # Validate symbols format
        for symbol in symbols:
            if not isinstance(symbol, str) or '/' not in symbol:
                raise StreamManagerError(f"Invalid symbol format: {symbol}. Expected format: 'BASE/QUOTE'")
    
    def _symbol_to_binance_format(self, symbol: str) -> str:
        """Convert symbol from 'BTC/USDT' format to Binance 'btcusdt' format."""
        return symbol.replace('/', '').lower()
    
    def _create_websocket_url(self, symbol: str) -> str:
        """Create Binance WebSocket URL for a symbol."""
        binance_symbol = self._symbol_to_binance_format(symbol)
        return f"{self.BINANCE_WS_BASE}{binance_symbol}@trade"
    
    async def start(self) -> None:
        """
        Start WebSocket connections for all symbols.
        
        Raises:
            StreamManagerError: If startup fails
        """
        if self.is_running:
            logger.warning("StreamManager is already running")
            return
        
        logger.info(f"Starting StreamManager for symbols {self.symbols} and timeframes {self.timeframes}")
        
        try:
            # Create WebSocket connections for each symbol
            for symbol in self.symbols:
                await self._connect_symbol(symbol)
            
            self.is_running = True
            logger.info("StreamManager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start StreamManager: {e}")
            await self.stop()
            raise StreamManagerError(f"Failed to start StreamManager: {e}") from e
    
    async def _connect_symbol(self, symbol: str) -> None:
        """Create WebSocket connection for a specific symbol."""
        url = self._create_websocket_url(symbol)
        logger.info(f"Connecting to WebSocket for {symbol}: {url}")
        
        def on_message(ws, message):
            asyncio.create_task(self._handle_message(symbol, message))
        
        def on_error(ws, error):
            logger.error(f"WebSocket error for {symbol}: {error}")
            asyncio.create_task(self._handle_reconnect(symbol))
        
        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"WebSocket closed for {symbol}: {close_status_code} - {close_msg}")
            if self.is_running:
                asyncio.create_task(self._handle_reconnect(symbol))
        
        def on_open(ws):
            logger.info(f"WebSocket connected for {symbol}")
            self.reconnect_attempts = 0
            self.reconnect_delay = 1
        
        ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        self.ws_connections[symbol] = ws
        
        # Start WebSocket in a separate thread
        import threading
        threading.Thread(target=ws.run_forever, daemon=True).start()
    
    async def _handle_message(self, symbol: str, message: str) -> None:
        """Handle incoming WebSocket message."""
        start_time = time.time()
        
        try:
            data = json.loads(message)
            
            # Binance trade stream format
            if 'e' in data and data['e'] == 'trade':
                await self._process_tick(symbol, data)
            
            # Track processing performance
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.processing_times.append(processing_time)
            
            if processing_time > 10:  # Log if processing takes more than 10ms
                logger.warning(f"Slow tick processing: {processing_time:.2f}ms for {symbol}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error processing message for {symbol}: {e}")
    
    async def _process_tick(self, symbol: str, tick_data: dict) -> None:
        """
        Process individual tick and update candle data.
        
        Args:
            symbol: Trading symbol
            tick_data: Binance trade data
        """
        try:
            price = float(tick_data['p'])
            quantity = float(tick_data['q'])
            timestamp = int(tick_data['T'])  # Trade time
            
            # Update current tick data
            current = self.current_ticks[symbol]
            current['price'] = price
            current['volume'] += quantity
            current['timestamp'] = timestamp
            current['close'] = price
            
            # Initialize OHLC if first tick
            if current['open'] == 0.0:
                current['open'] = price
                current['high'] = price
                current['low'] = price
            else:
                current['high'] = max(current['high'], price)
                current['low'] = min(current['low'], price)
            
            # Update candles for each timeframe
            for tf in self.timeframes:
                await self._update_candle(symbol, tf, timestamp, current.copy())
            
            # Persist to Redis if available
            if self.redis_client:
                await self._save_to_redis(symbol)
            
            self.tick_count += 1
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error processing tick for {symbol}: {e}")
    
    async def _update_candle(self, symbol: str, timeframe: str, timestamp: int, tick_data: dict) -> None:
        """Update candle for a specific timeframe."""
        try:
            # Calculate candle start time
            tf_minutes = self.TIMEFRAMES[timeframe]
            dt = datetime.fromtimestamp(timestamp / 1000, tz=UTC)
            
            # Round down to timeframe boundary
            minutes_since_hour = dt.minute
            candle_start_minute = (minutes_since_hour // tf_minutes) * tf_minutes
            candle_start = dt.replace(minute=candle_start_minute, second=0, microsecond=0)
            
            buffer = self.buffers[symbol][timeframe]
            candle_key = f"{symbol}_{timeframe}"
            
            # Check if we need to start a new candle
            if (candle_key not in tick_data['candle_start'] or 
                tick_data['candle_start'][candle_key] != candle_start):
                
                # Finalize previous candle if exists
                if buffer and candle_key in tick_data['candle_start']:
                    # Previous candle is complete, ensure it's properly stored
                    pass
                
                # Start new candle
                new_candle = {
                    'timestamp': candle_start,
                    'open': tick_data['price'],
                    'high': tick_data['price'],
                    'low': tick_data['price'],
                    'close': tick_data['price'],
                    'volume': tick_data['volume']
                }
                
                buffer.append(new_candle)
                tick_data['candle_start'][candle_key] = candle_start
                
            else:
                # Update existing candle
                if buffer:
                    current_candle = buffer[-1]
                    current_candle['high'] = max(current_candle['high'], tick_data['price'])
                    current_candle['low'] = min(current_candle['low'], tick_data['price'])
                    current_candle['close'] = tick_data['price']
                    current_candle['volume'] += tick_data['volume']
        
        except Exception as e:
            logger.error(f"Error updating candle for {symbol} {timeframe}: {e}")
    
    async def _save_to_redis(self, symbol: str) -> None:
        """Save current buffer state to Redis."""
        if not self.redis_client:
            return
        
        try:
            for tf in self.timeframes:
                buffer = self.buffers[symbol][tf]
                if buffer:
                    # Convert deque to list for JSON serialization
                    data = list(buffer)
                    key = f"stream:{symbol}:{tf}"
                    
                    # Serialize to JSON bytes
                    serialized = json.dumps(data, default=str).encode('utf-8')
                    self.redis_client.set(key, serialized, ex=3600)  # Expire in 1 hour
        
        except Exception as e:
            logger.error(f"Error saving to Redis for {symbol}: {e}")
    
    async def _handle_reconnect(self, symbol: str) -> None:
        """Handle WebSocket reconnection with exponential backoff."""
        if not self.is_running:
            return
        
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts > self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts reached for {symbol}")
            return
        
        logger.info(f"Reconnecting to {symbol} in {self.reconnect_delay} seconds (attempt {self.reconnect_attempts})")
        
        await asyncio.sleep(self.reconnect_delay)
        
        try:
            await self._connect_symbol(symbol)
            logger.info(f"Successfully reconnected to {symbol}")
            
        except Exception as e:
            logger.error(f"Reconnection failed for {symbol}: {e}")
            
            # Exponential backoff
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
            await self._handle_reconnect(symbol)
    
    def get_data(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get DataFrame for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Maximum number of candles to return (default: all)
            
        Returns:
            pd.DataFrame: OHLCV data with DatetimeIndex compatible with Engine class
            
        Raises:
            StreamManagerError: If symbol/timeframe not found or invalid
        """
        if symbol not in self.buffers:
            raise StreamManagerError(f"Symbol {symbol} not found")
        
        if timeframe not in self.buffers[symbol]:
            raise StreamManagerError(f"Timeframe {timeframe} not found for {symbol}")
        
        buffer = self.buffers[symbol][timeframe]
        
        if not buffer:
            # Return empty DataFrame with proper structure
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS).set_index(
                pd.DatetimeIndex([], name="timestamp", tz="UTC")
            )
        
        # Convert deque to list and create DataFrame
        data = list(buffer)
        if limit:
            data = data[-limit:]
        
        df = pd.DataFrame(data)
        
        # Set datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Ensure required columns and proper types
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clear frequency for consistency with Engine class
        df.index.freq = None
        
        return df[self.REQUIRED_COLUMNS]
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        if symbol not in self.current_ticks:
            return None
        
        return self.current_ticks[symbol]['price'] if self.current_ticks[symbol]['price'] > 0 else None
    
    def get_statistics(self) -> dict:
        """Get performance and connection statistics."""
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )
        
        return {
            'is_running': self.is_running,
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'tick_count': self.tick_count,
            'avg_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max(self.processing_times) if self.processing_times else 0,
            'reconnect_attempts': self.reconnect_attempts,
            'buffer_sizes': {
                symbol: {tf: len(self.buffers[symbol][tf]) for tf in self.timeframes}
                for symbol in self.symbols
            },
            'redis_connected': self.redis_client is not None
        }
    
    async def stop(self) -> None:
        """Stop all WebSocket connections and cleanup resources."""
        logger.info("Stopping StreamManager...")
        
        self.is_running = False
        
        # Close WebSocket connections
        for symbol, ws in self.ws_connections.items():
            try:
                ws.close()
                logger.info(f"Closed WebSocket connection for {symbol}")
            except Exception as e:
                logger.error(f"Error closing WebSocket for {symbol}: {e}")
        
        self.ws_connections.clear()
        
        # Close Redis connection
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
        
        logger.info("StreamManager stopped")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.create_task(self.stop())
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()