"""
StrategyRunner for executing trading strategies in real-time paper trading environment.

This module implements the StrategyRunner class that executes YAML-configured trading
strategies in real-time using the existing Engine class extended for live mode.
It provides event-driven execution, strategy state persistence, performance tracking,
and integration with StreamManager and PaperTrader.
"""

import asyncio
import json
import logging
import time
import uuid
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import yaml

from btc_research.core.engine import Engine, EngineError
from btc_research.core.backtester import StrategyLogic
from btc_research.live.paper_trader import OrderSide, PaperTrader, PaperTraderError
from btc_research.live.stream_manager import StreamManager, StreamManagerError

__all__ = ["StrategyRunner", "StrategyRunnerError", "StrategyState", "StrategyStatistics"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyRunnerError(Exception):
    """Base exception for StrategyRunner related errors."""
    pass


class StrategyState:
    """
    Manages persistent state for a trading strategy.
    
    Handles saving and loading strategy state to JSON files to support
    graceful restarts and recovery from system failures.
    """
    
    def __init__(self, strategy_id: str, data_dir: Path = None):
        """
        Initialize strategy state management.
        
        Args:
            strategy_id: Unique identifier for the strategy
            data_dir: Directory to store state files (default: ./data/strategies)
        """
        self.strategy_id = strategy_id
        self.data_dir = data_dir or Path("data/strategies")
        self.state_file = self.data_dir / f"{strategy_id}_state.json"
        
        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # State data
        self.state = {
            'strategy_id': strategy_id,
            'created_at': datetime.now(UTC).isoformat(),
            'last_updated': None,
            'last_signal_time': None,
            'last_processed_timestamp': None,
            'position_entries': {},  # Track entry prices and times
            'signal_history': [],    # Keep recent signals for analysis
            'performance_metrics': {
                'signals_generated': 0,
                'orders_submitted': 0,
                'successful_orders': 0,
                'failed_orders': 0,
                'total_processing_time_ms': 0.0,
                'avg_processing_time_ms': 0.0,
                'max_processing_time_ms': 0.0
            },
            'error_count': 0,
            'last_error': None,
            'restart_count': 0
        }
        
        # Load existing state if available
        self.load()
    
    def save(self) -> None:
        """Save current state to JSON file."""
        try:
            self.state['last_updated'] = datetime.now(UTC).isoformat()
            
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save strategy state for {self.strategy_id}: {e}")
    
    def load(self) -> None:
        """Load state from JSON file if it exists."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    loaded_state = json.load(f)
                
                # Merge loaded state with defaults
                self.state.update(loaded_state)
                self.state['restart_count'] = self.state.get('restart_count', 0) + 1
                
                logger.info(f"Restored strategy state for {self.strategy_id} (restart #{self.state['restart_count']})")
        
        except Exception as e:
            logger.warning(f"Failed to load strategy state for {self.strategy_id}: {e}")
    
    def record_signal(self, signal_type: str, conditions: Dict[str, Any]) -> None:
        """Record a generated signal.""" 
        signal_record = {
            'timestamp': datetime.now(UTC).isoformat(),
            'type': signal_type,
            'conditions': conditions
        }
        
        # Keep only last 100 signals
        self.state['signal_history'].append(signal_record)
        if len(self.state['signal_history']) > 100:
            self.state['signal_history'] = self.state['signal_history'][-100:]
        
        self.state['last_signal_time'] = signal_record['timestamp']
        self.state['performance_metrics']['signals_generated'] += 1
    
    def record_order(self, success: bool, error: str = None) -> None:
        """Record order submission result."""
        self.state['performance_metrics']['orders_submitted'] += 1
        
        if success:
            self.state['performance_metrics']['successful_orders'] += 1
        else:
            self.state['performance_metrics']['failed_orders'] += 1
            if error:
                self.state['last_error'] = {
                    'timestamp': datetime.now(UTC).isoformat(),
                    'error': str(error)
                }
    
    def record_processing_time(self, processing_time_ms: float) -> None:
        """Record strategy processing time."""
        metrics = self.state['performance_metrics']
        
        # Update running averages
        total_time = metrics['total_processing_time_ms'] + processing_time_ms
        signal_count = metrics['signals_generated']
        
        if signal_count > 0:
            metrics['avg_processing_time_ms'] = total_time / signal_count
        
        metrics['total_processing_time_ms'] = total_time
        metrics['max_processing_time_ms'] = max(
            metrics['max_processing_time_ms'], processing_time_ms
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics."""
        return {
            'strategy_id': self.strategy_id,
            'created_at': self.state['created_at'],
            'last_updated': self.state['last_updated'],
            'restart_count': self.state['restart_count'],
            'performance_metrics': self.state['performance_metrics'].copy(),
            'error_count': self.state['error_count'],
            'last_error': self.state['last_error'],
            'recent_signals': self.state['signal_history'][-10:] if self.state['signal_history'] else []
        }


class StrategyStatistics:
    """
    Real-time performance statistics tracking for strategies.
    
    Provides comprehensive metrics including execution times, signal generation
    rates, success rates, and performance analytics.
    """
    
    def __init__(self, strategy_id: str):
        """Initialize statistics tracking."""
        self.strategy_id = strategy_id
        self.start_time = datetime.now(UTC)
        
        # Performance counters
        self.data_updates_processed = 0
        self.signals_generated = 0
        self.orders_submitted = 0
        self.successful_orders = 0
        self.errors_encountered = 0
        
        # Timing statistics
        self.processing_times = []
        self.last_update_time = None
        
        # Rate calculations
        self.signal_rate_1min = 0.0
        self.signal_rate_5min = 0.0
        self.signal_rate_1hour = 0.0
        
        # Recent activity tracking
        self.recent_signals = []
        self.recent_errors = []
    
    def record_data_update(self, processing_time_ms: float) -> None:
        """Record a data update processing event."""
        self.data_updates_processed += 1
        self.processing_times.append(processing_time_ms)
        self.last_update_time = datetime.now(UTC)
        
        # Keep only last 1000 processing times for memory efficiency
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
    
    def record_signal(self, signal_type: str) -> None:
        """Record signal generation."""
        self.signals_generated += 1
        signal_record = {
            'timestamp': datetime.now(UTC),
            'type': signal_type
        }
        self.recent_signals.append(signal_record)
        
        # Keep only last 50 signals
        if len(self.recent_signals) > 50:
            self.recent_signals = self.recent_signals[-50:]
        
        # Update signal rates
        self._update_signal_rates()
    
    def record_order_result(self, success: bool, error: str = None) -> None:
        """Record order execution result."""
        self.orders_submitted += 1
        
        if success:
            self.successful_orders += 1
        else:
            self.errors_encountered += 1
            if error:
                error_record = {
                    'timestamp': datetime.now(UTC),
                    'error': str(error)
                }
                self.recent_errors.append(error_record)
                
                # Keep only last 20 errors
                if len(self.recent_errors) > 20:
                    self.recent_errors = self.recent_errors[-20:]
    
    def _update_signal_rates(self) -> None:
        """Update signal generation rates."""
        now = datetime.now(UTC)
        
        # Calculate rates for different time windows
        for minutes, attr in [(1, 'signal_rate_1min'), (5, 'signal_rate_5min'), (60, 'signal_rate_1hour')]:
            cutoff = now - timedelta(minutes=minutes)
            recent_count = sum(1 for s in self.recent_signals if s['timestamp'] >= cutoff)
            setattr(self, attr, recent_count / minutes)  # signals per minute
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        now = datetime.now(UTC)
        uptime_seconds = (now - self.start_time).total_seconds()
        
        # Calculate averages
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        max_processing_time = max(self.processing_times) if self.processing_times else 0.0
        
        # Success rates
        order_success_rate = (
            self.successful_orders / self.orders_submitted
            if self.orders_submitted > 0 else 0.0
        )
        
        return {
            'strategy_id': self.strategy_id,
            'uptime_seconds': uptime_seconds,
            'start_time': self.start_time.isoformat(),
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'data_updates_processed': self.data_updates_processed,
            'signals_generated': self.signals_generated,
            'orders_submitted': self.orders_submitted,
            'successful_orders': self.successful_orders,
            'errors_encountered': self.errors_encountered,
            'order_success_rate': order_success_rate,
            'avg_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max_processing_time,
            'signal_rates': {
                '1min': self.signal_rate_1min,
                '5min': self.signal_rate_5min,
                '1hour': self.signal_rate_1hour
            },
            'recent_signals': [
                {'timestamp': s['timestamp'].isoformat(), 'type': s['type']}
                for s in self.recent_signals[-10:]
            ],
            'recent_errors': [
                {'timestamp': e['timestamp'].isoformat(), 'error': e['error']}
                for e in self.recent_errors[-5:]
            ]
        }


class StrategyRunner:
    """
    Real-time strategy execution engine for paper trading.
    
    This class executes YAML-configured trading strategies in real-time using
    the existing Engine class extended for live mode. It provides:
    
    - Event-driven execution model with async/await
    - Direct YAML configuration loading (reuse existing configs)
    - Strategy-level error isolation (one failure doesn't stop others)
    - Persistent strategy state across restarts
    - Integration with existing Engine class for indicator calculations
    - Real-time strategy signal generation and order submission
    - Performance monitoring and statistics tracking
    
    Key Features:
    1. YAML Configuration Loading: Uses existing volume-profile-breakout.yaml
    2. Engine Integration: Extends existing Engine class for live_mode operation
    3. Real-time Event Loop: Processes market data updates and generates signals
    4. Order Management: Submits orders via PaperTrader based on strategy signals
    5. State Persistence: Saves/restores strategy state to handle restarts
    6. Error Handling: Robust error recovery without stopping execution
    7. Performance Tracking: Real-time statistics and monitoring
    
    Example:
        >>> config_path = "config/volume-profile-breakout.yaml"
        >>> runner = StrategyRunner(config_path, paper_trader, stream_manager)
        >>> await runner.start()
        >>> stats = runner.get_statistics()
    """
    
    def __init__(
        self,
        config_path: str,
        paper_trader: PaperTrader,
        stream_manager: StreamManager,
        max_processing_time_ms: float = 200.0,
        state_save_interval: int = 10,  # Save state every 10 seconds
        data_dir: Path = None
    ):
        """
        Initialize the StrategyRunner.
        
        Args:
            config_path: Path to YAML strategy configuration file
            paper_trader: PaperTrader instance for order execution
            stream_manager: StreamManager instance for real-time data
            max_processing_time_ms: Maximum allowed processing time per update
            state_save_interval: Interval in seconds to save strategy state
            data_dir: Directory for state persistence (default: ./data/strategies)
        """
        self.config_path = Path(config_path)
        self.paper_trader = paper_trader
        self.stream_manager = stream_manager
        self.max_processing_time_ms = max_processing_time_ms
        self.state_save_interval = state_save_interval
        self.data_dir = data_dir or Path("data/strategies")
        
        # Strategy identification
        self.strategy_id = str(uuid.uuid4())
        self.strategy_name = self.config_path.stem
        
        # Configuration and state
        self.config = None
        self.engine = None
        self.strategy_logic = None
        self.strategy_state = None
        self.statistics = None
        
        # Runtime state
        self.is_running = False
        self.last_processed_timestamp = None
        self.current_position_size = 0.0
        self.entry_price = None
        self.entry_time = None
        
        # Error handling
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.error_backoff_delay = 1.0  # Start with 1 second delay
        self.max_error_delay = 60.0     # Max 60 seconds delay
        
        # Event loop control
        self._stop_event = asyncio.Event()
        self._update_task = None
        self._state_save_task = None
        
        logger.info(f"StrategyRunner initialized: {self.strategy_name} (ID: {self.strategy_id[:8]})")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate YAML configuration."""
        try:
            if not self.config_path.exists():
                raise StrategyRunnerError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = ["symbol", "timeframes", "indicators", "logic"]
            for field in required_fields:
                if field not in config:
                    raise StrategyRunnerError(f"Missing required field in config: {field}")
            
            # Add backtest section if missing (required by Engine)
            if "backtest" not in config:
                # Use recent data for initial indicator calculation
                end_date = datetime.now(UTC).strftime("%Y-%m-%d")
                start_date = (datetime.now(UTC) - timedelta(days=30)).strftime("%Y-%m-%d")
                
                config["backtest"] = {
                    "from": start_date,
                    "to": end_date,
                    "cash": 10000
                }
            
            logger.info(f"Loaded configuration for {config.get('name', 'Unnamed Strategy')}")
            return config
            
        except Exception as e:
            raise StrategyRunnerError(f"Failed to load configuration: {e}") from e
    
    def _create_live_engine(self) -> Engine:
        """Create Engine instance configured for live mode."""
        try:
            # Modify config for live mode - we need minimal historical data for indicators
            live_config = self.config.copy()
            
            # Override backtest period to fetch minimal data for indicator initialization
            # We only need enough data to calculate indicators (typically 200-500 bars)
            from datetime import datetime, timedelta
            
            # Set a reasonable period for live mode - 6 months for volume profile and long-term indicators
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            live_config['backtest'] = {
                'cash': live_config.get('backtest', {}).get('cash', 10000),
                'commission': live_config.get('backtest', {}).get('commission', 0.001),
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            logger.info(f"Modified config for live mode: fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Engine expects a standard config structure
            engine = Engine(live_config)
            
            # Run engine once to initialize indicators with historical data
            initial_df = engine.run()
            
            if len(initial_df) == 0:
                raise StrategyRunnerError("No historical data available for indicator initialization")
            
            logger.info(f"Engine initialized with {len(initial_df)} historical data points")
            return engine
            
        except Exception as e:
            raise StrategyRunnerError(f"Failed to create live engine: {e}") from e
    
    def _create_strategy_logic(self, initial_df: pd.DataFrame) -> StrategyLogic:
        """Create StrategyLogic instance for signal evaluation."""
        try:
            return StrategyLogic(self.config, initial_df, debug=True)
        except Exception as e:
            raise StrategyRunnerError(f"Failed to create strategy logic: {e}") from e
    
    async def start(self) -> None:
        """
        Start the strategy runner.
        
        Initializes all components and begins the real-time event loop.
        """
        if self.is_running:
            logger.warning(f"Strategy {self.strategy_name} is already running")
            return
        
        try:
            logger.info(f"Starting strategy: {self.strategy_name}")
            
            # Load configuration
            self.config = self._load_config()
            
            # Initialize state management
            self.strategy_state = StrategyState(self.strategy_id, self.data_dir)
            
            # Initialize statistics tracking
            self.statistics = StrategyStatistics(self.strategy_id)
            
            # Create engine with historical data for indicator initialization
            self.engine = self._create_live_engine()
            initial_df = self.engine.run()
            
            # Create strategy logic
            self.strategy_logic = self._create_strategy_logic(initial_df)
            
            # Validate that we have the required symbols in stream manager
            required_symbol = self.config["symbol"]
            if required_symbol not in self.stream_manager.symbols:
                raise StrategyRunnerError(f"Required symbol {required_symbol} not available in StreamManager")
            
            # Set the stream manager reference in paper trader
            self.paper_trader.set_stream_manager(self.stream_manager)
            
            # Start the update loop
            self.is_running = True
            self._stop_event.clear()
            
            # Start background tasks
            self._update_task = asyncio.create_task(self._update_loop())
            self._state_save_task = asyncio.create_task(self._state_save_loop())
            
            logger.info(f"Strategy {self.strategy_name} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start strategy {self.strategy_name}: {e}")
            await self.stop()
            raise StrategyRunnerError(f"Failed to start strategy: {e}") from e
    
    async def stop(self) -> None:
        """Stop the strategy runner and cleanup resources."""
        if not self.is_running:
            return
        
        logger.info(f"Stopping strategy: {self.strategy_name}")
        
        self.is_running = False
        self._stop_event.set()
        
        # Cancel background tasks
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        if self._state_save_task:
            self._state_save_task.cancel()
            try:
                await self._state_save_task
            except asyncio.CancelledError:
                pass
        
        # Save final state
        if self.strategy_state:
            self.strategy_state.save()
        
        logger.info(f"Strategy {self.strategy_name} stopped")
    
    async def _update_loop(self) -> None:
        """
        Main event loop for processing market data updates.
        
        This loop continuously monitors for new market data and processes
        strategy signals when updates are available.
        """
        logger.info(f"Starting update loop for strategy {self.strategy_name}")
        
        # Get entry timeframe from config
        entry_timeframe = self.config["timeframes"]["entry"]
        symbol = self.config["symbol"]
        
        while not self._stop_event.is_set():
            try:
                # Check if we have new data
                current_data = self.stream_manager.get_data(symbol, entry_timeframe, limit=500)
                
                if len(current_data) == 0:
                    # No data available yet, wait and continue
                    await asyncio.sleep(1.0)
                    continue
                
                # Get the latest timestamp
                latest_timestamp = current_data.index[-1]
                
                # Skip if we've already processed this timestamp
                if (self.last_processed_timestamp is not None and 
                    latest_timestamp <= self.last_processed_timestamp):
                    await asyncio.sleep(0.1)  # Short sleep to avoid busy waiting
                    continue
                
                # Process the update
                start_time = time.time()
                await self._process_data_update(symbol, current_data)
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Update statistics
                self.statistics.record_data_update(processing_time_ms)
                self.strategy_state.record_processing_time(processing_time_ms)
                
                # Check processing time performance requirement
                if processing_time_ms > self.max_processing_time_ms:
                    logger.warning(
                        f"Processing took {processing_time_ms:.2f}ms, "
                        f"exceeds limit of {self.max_processing_time_ms}ms"
                    )
                
                # Update last processed timestamp
                self.last_processed_timestamp = latest_timestamp
                
                # Reset error counters on successful processing
                self.consecutive_errors = 0
                self.error_backoff_delay = 1.0
                
            except Exception as e:
                await self._handle_update_error(e)
    
    async def _process_data_update(self, symbol: str, current_data: pd.DataFrame) -> None:
        """
        Process a market data update and evaluate strategy conditions.
        
        Args:
            symbol: Trading symbol
            current_data: Latest market data DataFrame
        """
        try:
            # Get all required timeframes
            timeframes_needed = set(ind["timeframe"] for ind in self.config["indicators"])
            timeframes_needed.add(self.config["timeframes"]["entry"])
            
            # Collect data for all timeframes
            data_by_tf = {}
            for tf in timeframes_needed:
                tf_data = self.stream_manager.get_data(symbol, tf, limit=500)
                if len(tf_data) == 0:
                    logger.debug(f"No data available for {symbol} {tf}")
                    return  # Skip processing if any required timeframe has no data
                data_by_tf[tf] = tf_data
            
            # Update indicators with live data using Engine pattern
            updated_data = await self._update_indicators_live(data_by_tf)
            
            if len(updated_data) == 0:
                logger.debug("No indicator data available for signal evaluation")
                return
            
            # Get the latest row for evaluation
            latest_row = updated_data.iloc[-1]
            eval_context = latest_row.to_dict()
            
            # Handle NaN values
            for key, value in eval_context.items():
                if pd.isna(value):
                    eval_context[key] = None
            
            # Evaluate strategy conditions
            await self._evaluate_strategy_conditions(eval_context, latest_row.name)
            
        except Exception as e:
            logger.error(f"Error processing data update: {e}")
            raise
    
    async def _update_indicators_live(self, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Update indicators with live data using Engine pattern.
        
        Args:
            data_by_tf: Dictionary mapping timeframes to DataFrames
            
        Returns:
            Combined DataFrame with updated indicators
        """
        try:
            # Use similar logic to Engine._instantiate_indicators but for live data
            entry_tf = self.config["timeframes"]["entry"]
            base_data = data_by_tf[entry_tf].copy()
            
            # Update each indicator
            for spec in self.config["indicators"]:
                # Get the corresponding indicator object from the engine
                indicator_class = None
                for indicator_obj in self.engine.indicator_objects:
                    if hasattr(indicator_obj, '__class__'):
                        # Match by indicator type and parameters
                        expected_params = {
                            k: v for k, v in spec.items() 
                            if k not in ("id", "type", "timeframe")
                        }
                        
                        # Simple matching - in production, this could be more sophisticated
                        if indicator_obj.__class__.__name__.lower() == spec["type"].lower():
                            indicator_class = indicator_obj
                            break
                
                if indicator_class is None:
                    logger.warning(f"No indicator found for {spec['id']}")
                    continue
                
                # Get timeframe data
                timeframe = spec["timeframe"]
                if timeframe not in data_by_tf:
                    continue
                
                tf_data = data_by_tf[timeframe]
                
                # Compute indicator with latest data
                try:
                    indicator_output = indicator_class.compute(tf_data)
                    
                    if not isinstance(indicator_output, pd.DataFrame) or len(indicator_output) == 0:
                        continue
                    
                    # Rename columns to include indicator ID prefix (same as Engine logic)
                    renamed_columns = {}
                    for col in indicator_output.columns:
                        if col == spec["id"]:
                            renamed_columns[col] = spec["id"]
                        elif col.startswith(spec["id"] + "_"):
                            renamed_columns[col] = col
                        else:
                            renamed_columns[col] = f"{spec['id']}_{col}"
                    
                    indicator_output = indicator_output.rename(columns=renamed_columns)
                    
                    # Align with entry timeframe if different
                    if timeframe != entry_tf:
                        # Forward-fill higher timeframe data onto entry timeframe
                        aligned_output = indicator_output.reindex(base_data.index, method="ffill")
                        
                        # Only join non-OHLCV columns
                        ohlcv_cols = {"open", "high", "low", "close", "volume"}
                        cols_to_join = [
                            col for col in aligned_output.columns 
                            if col not in ohlcv_cols
                        ]
                        if cols_to_join:
                            base_data = base_data.join(aligned_output[cols_to_join], how="left")
                    else:
                        # Same timeframe, direct join
                        ohlcv_cols = {"open", "high", "low", "close", "volume"}
                        cols_to_join = [
                            col for col in indicator_output.columns 
                            if col not in ohlcv_cols
                        ]
                        if cols_to_join:
                            base_data = base_data.join(indicator_output[cols_to_join], how="left")
                
                except Exception as e:
                    logger.error(f"Failed to update indicator {spec['id']}: {e}")
                    continue
            
            return base_data
            
        except Exception as e:
            logger.error(f"Failed to update indicators: {e}")
            return pd.DataFrame()
    
    async def _evaluate_strategy_conditions(self, eval_context: Dict[str, Any], timestamp: pd.Timestamp) -> None:
        """
        Evaluate strategy entry and exit conditions.
        
        Args:
            eval_context: Dictionary of indicator values for evaluation
            timestamp: Current timestamp for signal recording
        """
        try:
            current_position = self.paper_trader.get_position(self.config["symbol"])
            position_size = current_position.size if current_position else 0.0
            
            # Track position changes
            if position_size != self.current_position_size:
                self.current_position_size = position_size
                if position_size != 0:
                    self.entry_price = current_position.average_price
                    self.entry_time = datetime.now(UTC)
                else:
                    self.entry_price = None
                    self.entry_time = None
            
            # Check entry conditions if no position
            if position_size == 0:
                await self._check_entry_conditions(eval_context, timestamp)
            else:
                # Check exit conditions if we have a position
                await self._check_exit_conditions(eval_context, timestamp, position_size > 0)
                
        except Exception as e:
            logger.error(f"Error evaluating strategy conditions: {e}")
            raise
    
    async def _check_entry_conditions(self, eval_context: Dict[str, Any], timestamp: pd.Timestamp) -> None:
        """Check and execute entry conditions."""
        symbol = self.config["symbol"]
        
        # Check long entry conditions
        if len(self.strategy_logic.entry_long_rules) > 0:
            if self.strategy_logic.evaluate_rules(self.strategy_logic.entry_long_rules, eval_context):
                logger.info(f"Long entry signal detected for {symbol}")
                
                # Record signal
                self.strategy_state.record_signal("entry_long", eval_context)
                self.statistics.record_signal("entry_long")
                
                # Calculate position size (simplified for now)
                position_size = self._calculate_position_size(eval_context, True)
                
                # Submit buy order
                try:
                    order = await self.paper_trader.submit_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        size=position_size
                    )
                    
                    logger.info(f"Long entry order submitted: {order.id[:8]} for {position_size} {symbol}")
                    self.strategy_state.record_order(True)
                    self.statistics.record_order_result(True)
                    
                except Exception as e:
                    logger.error(f"Failed to submit long entry order: {e}")
                    self.strategy_state.record_order(False, str(e))
                    self.statistics.record_order_result(False, str(e))
        
        # Check short entry conditions
        if len(self.strategy_logic.entry_short_rules) > 0:
            if self.strategy_logic.evaluate_rules(self.strategy_logic.entry_short_rules, eval_context):
                logger.info(f"Short entry signal detected for {symbol}")
                
                # Record signal
                self.strategy_state.record_signal("entry_short", eval_context)
                self.statistics.record_signal("entry_short")
                
                # Calculate position size
                position_size = self._calculate_position_size(eval_context, False)
                
                # Submit sell order
                try:
                    order = await self.paper_trader.submit_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        size=position_size
                    )
                    
                    logger.info(f"Short entry order submitted: {order.id[:8]} for {position_size} {symbol}")
                    self.strategy_state.record_order(True)
                    self.statistics.record_order_result(True)
                    
                except Exception as e:
                    logger.error(f"Failed to submit short entry order: {e}")
                    self.strategy_state.record_order(False, str(e))
                    self.statistics.record_order_result(False, str(e))
    
    async def _check_exit_conditions(self, eval_context: Dict[str, Any], timestamp: pd.Timestamp, is_long: bool) -> None:
        """Check and execute exit conditions."""
        symbol = self.config["symbol"]
        current_position = self.paper_trader.get_position(symbol)
        
        if not current_position or current_position.is_flat():
            return
        
        exit_rules = self.strategy_logic.exit_long_rules if is_long else self.strategy_logic.exit_short_rules
        signal_type = "exit_long" if is_long else "exit_short"
        
        if len(exit_rules) > 0:
            if self.strategy_logic.evaluate_rules(exit_rules, eval_context):
                logger.info(f"{signal_type.title()} signal detected for {symbol}")
                
                # Record signal
                self.strategy_state.record_signal(signal_type, eval_context)
                self.statistics.record_signal(signal_type)
                
                # Submit exit order (opposite side)
                exit_side = OrderSide.SELL if is_long else OrderSide.BUY
                position_size = abs(current_position.size)
                
                try:
                    order = await self.paper_trader.submit_order(
                        symbol=symbol,
                        side=exit_side,
                        size=position_size
                    )
                    
                    logger.info(f"{signal_type.title()} order submitted: {order.id[:8]} for {position_size} {symbol}")
                    self.strategy_state.record_order(True)
                    self.statistics.record_order_result(True)
                    
                except Exception as e:
                    logger.error(f"Failed to submit {signal_type} order: {e}")
                    self.strategy_state.record_order(False, str(e))
                    self.statistics.record_order_result(False, str(e))
    
    def _calculate_position_size(self, eval_context: Dict[str, Any], is_long: bool) -> float:
        """
        Calculate position size for new trades.
        
        Args:
            eval_context: Current market data and indicators
            is_long: True for long positions, False for short
            
        Returns:
            Position size in base currency units
        """
        # Get balance info
        balance_info = self.paper_trader.get_balance_info()
        available_balance = balance_info['available_balance']
        
        # Simple position sizing: use 10% of available balance
        risk_pct = 0.1
        max_position_value = available_balance * risk_pct
        
        # Get current price
        current_price = self.paper_trader.get_current_price(self.config["symbol"])
        if current_price is None:
            logger.warning("No current price available for position sizing")
            return 0.0
        
        # Calculate position size
        position_size = max_position_value / current_price
        
        # Minimum position size
        min_size = 10.0 / current_price  # $10 minimum
        position_size = max(position_size, min_size)
        
        return position_size
    
    async def _handle_update_error(self, error: Exception) -> None:
        """Handle errors in the update loop with exponential backoff."""
        self.consecutive_errors += 1
        self.statistics.record_order_result(False, str(error))
        
        if self.strategy_state:
            self.strategy_state.state['error_count'] += 1
            self.strategy_state.state['last_error'] = {
                'timestamp': datetime.now(UTC).isoformat(),
                'error': str(error)
            }
        
        logger.error(f"Update loop error #{self.consecutive_errors}: {error}")
        
        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.error(
                f"Too many consecutive errors ({self.consecutive_errors}), "
                f"stopping strategy {self.strategy_name}"
            )
            await self.stop()
            return
        
        # Exponential backoff
        await asyncio.sleep(self.error_backoff_delay)
        self.error_backoff_delay = min(self.error_backoff_delay * 2, self.max_error_delay)
    
    async def _state_save_loop(self) -> None:
        """Periodically save strategy state."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.state_save_interval)
                if self.strategy_state:
                    self.strategy_state.save()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error saving strategy state: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics."""
        stats = {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'config_path': str(self.config_path),
            'is_running': self.is_running,
            'symbol': self.config["symbol"] if self.config else None,
            'last_processed_timestamp': self.last_processed_timestamp.isoformat() if self.last_processed_timestamp else None,
            'current_position_size': self.current_position_size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'consecutive_errors': self.consecutive_errors
        }
        
        # Add statistics if available
        if self.statistics:
            stats['performance'] = self.statistics.get_statistics()
        
        # Add state statistics if available
        if self.strategy_state:
            stats['state'] = self.strategy_state.get_statistics()
        
        # Add paper trader statistics
        if self.paper_trader:
            stats['trading'] = self.paper_trader.get_performance_stats()
        
        return stats
    
    def get_current_position(self) -> Optional[Dict[str, Any]]:
        """Get current position information."""
        if not self.config:
            return None
        
        position = self.paper_trader.get_position(self.config["symbol"])
        if position:
            return position.to_dict()
        return None
    
    def get_recent_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent order history."""
        orders = self.paper_trader.get_orders()
        
        # Sort by creation time and limit
        sorted_orders = sorted(
            orders.values(),
            key=lambda o: o.created_at,
            reverse=True
        )
        
        return [order.to_dict() for order in sorted_orders[:limit]]
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trade history."""
        if not self.config:
            return []
        
        trades = self.paper_trader.get_trade_history(
            symbol=self.config["symbol"],
            limit=limit
        )
        
        return [trade.to_dict() for trade in trades]
    
    def __repr__(self) -> str:
        """String representation of the StrategyRunner."""
        status = "running" if self.is_running else "stopped"
        return f"StrategyRunner(name={self.strategy_name}, id={self.strategy_id[:8]}, status={status})"