"""
Test suite for StrategyRunner component.

This module provides comprehensive tests for the StrategyRunner class,
covering configuration loading, signal generation, order execution,
state persistence, error handling, and performance monitoring.
"""

import asyncio
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import yaml

from btc_research.live.paper_trader import OrderSide, PaperTrader, Position
from btc_research.live.stream_manager import StreamManager
from btc_research.live.strategy_runner import (
    StrategyRunner,
    StrategyRunnerError,
    StrategyState,
    StrategyStatistics,
)


class TestStrategyState:
    """Test cases for StrategyState class."""
    
    def test_init_creates_directory(self):
        """Test that StrategyState creates the data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "test_strategies"
            strategy_id = str(uuid.uuid4())
            
            state = StrategyState(strategy_id, data_dir)
            
            assert data_dir.exists()
            assert state.strategy_id == strategy_id
            assert state.state_file == data_dir / f"{strategy_id}_state.json"
    
    def test_save_and_load_state(self):
        """Test saving and loading strategy state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            strategy_id = str(uuid.uuid4())
            
            # Create and save state
            state1 = StrategyState(strategy_id, data_dir)
            state1.record_signal("entry_long", {"close": 50000})
            state1.record_order(True)
            state1.save()
            
            # Load state in new instance
            state2 = StrategyState(strategy_id, data_dir)
            
            assert state2.state['performance_metrics']['signals_generated'] == 1
            assert state2.state['performance_metrics']['successful_orders'] == 1
            assert len(state2.state['signal_history']) == 1
    
    def test_record_signal(self):
        """Test signal recording functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = StrategyState(str(uuid.uuid4()), Path(tmpdir))
            
            conditions = {"close": 50000, "volume": 1000}
            state.record_signal("entry_long", conditions)
            
            assert state.state['performance_metrics']['signals_generated'] == 1
            assert len(state.state['signal_history']) == 1
            assert state.state['signal_history'][0]['type'] == "entry_long"
            assert state.state['signal_history'][0]['conditions'] == conditions
    
    def test_record_processing_time(self):
        """Test processing time recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = StrategyState(str(uuid.uuid4()), Path(tmpdir))
            
            # Record some processing times
            state.record_processing_time(10.0)
            state.record_processing_time(20.0)
            
            metrics = state.state['performance_metrics']
            assert metrics['total_processing_time_ms'] == 30.0
            assert metrics['max_processing_time_ms'] == 20.0


class TestStrategyStatistics:
    """Test cases for StrategyStatistics class."""
    
    def test_init(self):
        """Test StrategyStatistics initialization."""
        strategy_id = str(uuid.uuid4())
        stats = StrategyStatistics(strategy_id)
        
        assert stats.strategy_id == strategy_id
        assert stats.data_updates_processed == 0
        assert stats.signals_generated == 0
        assert isinstance(stats.start_time, datetime)
    
    def test_record_data_update(self):
        """Test data update recording."""
        stats = StrategyStatistics(str(uuid.uuid4()))
        
        stats.record_data_update(15.5)
        
        assert stats.data_updates_processed == 1
        assert stats.processing_times == [15.5]
        assert stats.last_update_time is not None
    
    def test_record_signal(self):
        """Test signal recording."""
        stats = StrategyStatistics(str(uuid.uuid4()))
        
        stats.record_signal("entry_long")
        stats.record_signal("exit_long")
        
        assert stats.signals_generated == 2
        assert len(stats.recent_signals) == 2
        assert stats.recent_signals[0]['type'] == "entry_long"
    
    def test_get_statistics(self):
        """Test statistics generation."""
        stats = StrategyStatistics(str(uuid.uuid4()))
        
        # Record some activity
        stats.record_data_update(10.0)
        stats.record_data_update(20.0)
        stats.record_signal("entry_long")
        stats.record_order_result(True)
        
        result = stats.get_statistics()
        
        assert result['data_updates_processed'] == 2
        assert result['signals_generated'] == 1
        assert result['successful_orders'] == 1
        assert result['avg_processing_time_ms'] == 15.0
        assert result['max_processing_time_ms'] == 20.0


class TestStrategyRunner:
    """Test cases for StrategyRunner class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock strategy configuration."""
        return {
            "name": "Test Strategy",
            "symbol": "BTC/USDT",
            "exchange": "binanceus",
            "timeframes": {
                "entry": "15m",
                "analysis": "1h"
            },
            "indicators": [
                {
                    "id": "EMA_20",
                    "type": "EMA",
                    "timeframe": "15m",
                    "length": 20
                },
                {
                    "id": "RSI_14",
                    "type": "RSI",
                    "timeframe": "15m",
                    "length": 14
                }
            ],
            "logic": {
                "entry_long": [
                    "close > EMA_20",
                    "RSI_14 < 70"
                ],
                "exit_long": [
                    "RSI_14 > 80"
                ],
                "entry_short": [
                    "close < EMA_20",
                    "RSI_14 > 30"
                ],
                "exit_short": [
                    "RSI_14 < 20"
                ]
            }
        }
    
    @pytest.fixture
    def mock_config_file(self, mock_config):
        """Create a temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(mock_config, f)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def mock_paper_trader(self):
        """Create a mock PaperTrader."""
        trader = MagicMock(spec=PaperTrader)
        trader.get_position.return_value = None
        trader.get_current_price.return_value = 50000.0
        trader.submit_order = AsyncMock()
        trader.get_balance_info.return_value = {
            'available_balance': 10000.0,
            'total_balance': 10000.0
        }
        trader.get_performance_stats.return_value = {
            'total_trades': 0,
            'win_rate': 0.0
        }
        trader.set_stream_manager = MagicMock()
        return trader
    
    @pytest.fixture
    def mock_stream_manager(self):
        """Create a mock StreamManager."""
        manager = MagicMock(spec=StreamManager)
        manager.symbols = ["BTC/USDT"]
        
        # Create sample data
        timestamps = pd.date_range(
            start=datetime.now(UTC) - timedelta(hours=24),
            end=datetime.now(UTC),
            freq='15min'
        )
        
        sample_data = pd.DataFrame({
            'open': [50000.0] * len(timestamps),
            'high': [50100.0] * len(timestamps),
            'low': [49900.0] * len(timestamps),
            'close': [50000.0] * len(timestamps),
            'volume': [1000.0] * len(timestamps)
        }, index=timestamps)
        
        manager.get_data.return_value = sample_data
        return manager
    
    def test_init(self, mock_config_file, mock_paper_trader, mock_stream_manager):
        """Test StrategyRunner initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = StrategyRunner(
                config_path=mock_config_file,
                paper_trader=mock_paper_trader,
                stream_manager=mock_stream_manager,
                data_dir=Path(tmpdir)
            )
            
            assert runner.config_path == Path(mock_config_file)
            assert runner.paper_trader == mock_paper_trader
            assert runner.stream_manager == mock_stream_manager
            assert not runner.is_running
            assert runner.strategy_name == Path(mock_config_file).stem
    
    def test_load_config_success(self, mock_config_file, mock_paper_trader, mock_stream_manager):
        """Test successful configuration loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = StrategyRunner(
                config_path=mock_config_file,
                paper_trader=mock_paper_trader,
                stream_manager=mock_stream_manager,
                data_dir=Path(tmpdir)
            )
            
            config = runner._load_config()
            
            assert config["symbol"] == "BTC/USDT"
            assert config["timeframes"]["entry"] == "15m"
            assert "backtest" in config  # Should be added automatically
    
    def test_load_config_missing_file(self, mock_paper_trader, mock_stream_manager):
        """Test error handling for missing config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = StrategyRunner(
                config_path="nonexistent.yaml",
                paper_trader=mock_paper_trader,
                stream_manager=mock_stream_manager,
                data_dir=Path(tmpdir)
            )
            
            with pytest.raises(StrategyRunnerError, match="Configuration file not found"):
                runner._load_config()
    
    def test_load_config_invalid_yaml(self, mock_paper_trader, mock_stream_manager):
        """Test error handling for invalid YAML configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("invalid: yaml: content: [")
                invalid_config_file = f.name
            
            try:
                runner = StrategyRunner(
                    config_path=invalid_config_file,
                    paper_trader=mock_paper_trader,
                    stream_manager=mock_stream_manager,
                    data_dir=Path(tmpdir)
                )
                
                with pytest.raises(StrategyRunnerError, match="Failed to load configuration"):
                    runner._load_config()
            finally:
                os.unlink(invalid_config_file)
    
    @patch('btc_research.live.strategy_runner.Engine')
    def test_create_live_engine(self, mock_engine_class, mock_config_file, mock_paper_trader, mock_stream_manager):
        """Test live engine creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mock engine
            mock_engine = MagicMock()
            mock_engine.run.return_value = pd.DataFrame({
                'close': [50000.0, 50100.0],
                'volume': [1000.0, 1100.0]
            })
            mock_engine_class.return_value = mock_engine
            
            runner = StrategyRunner(
                config_path=mock_config_file,
                paper_trader=mock_paper_trader,
                stream_manager=mock_stream_manager,
                data_dir=Path(tmpdir)
            )
            runner.config = runner._load_config()
            
            engine = runner._create_live_engine()
            
            assert engine == mock_engine
            mock_engine.run.assert_called_once()
    
    def test_calculate_position_size(self, mock_config_file, mock_paper_trader, mock_stream_manager):
        """Test position size calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = StrategyRunner(
                config_path=mock_config_file,
                paper_trader=mock_paper_trader,
                stream_manager=mock_stream_manager,
                data_dir=Path(tmpdir)
            )
            runner.config = {"symbol": "BTC/USDT"}
            
            eval_context = {"close": 50000.0}
            position_size = runner._calculate_position_size(eval_context, True)
            
            # With 10% risk and $10,000 balance, max position value = $1,000
            # At $50,000 price, position size = $1,000 / $50,000 = 0.02
            assert position_size > 0
            assert position_size <= 1.0  # Reasonable upper bound
    
    @pytest.mark.asyncio
    async def test_start_and_stop(self, mock_config_file, mock_paper_trader, mock_stream_manager):
        """Test starting and stopping the strategy runner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('btc_research.live.strategy_runner.Engine') as mock_engine_class:
                # Setup mock engine
                mock_engine = MagicMock()
                mock_engine.run.return_value = pd.DataFrame({
                    'close': [50000.0],
                    'volume': [1000.0]
                })
                mock_engine.indicator_objects = []
                mock_engine_class.return_value = mock_engine
                
                runner = StrategyRunner(
                    config_path=mock_config_file,
                    paper_trader=mock_paper_trader,
                    stream_manager=mock_stream_manager,
                    data_dir=Path(tmpdir)
                )
                
                # Test start
                await runner.start()
                
                assert runner.is_running
                assert runner.config is not None
                assert runner.engine is not None
                assert runner.strategy_logic is not None
                assert runner.strategy_state is not None
                assert runner.statistics is not None
                
                # Test stop
                await runner.stop()
                
                assert not runner.is_running
    
    @pytest.mark.asyncio
    async def test_start_invalid_symbol(self, mock_config_file, mock_paper_trader, mock_stream_manager):
        """Test error handling for invalid symbol in StreamManager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure stream manager with different symbol
            mock_stream_manager.symbols = ["ETH/USDT"]
            
            with patch('btc_research.live.strategy_runner.Engine') as mock_engine_class:
                mock_engine = MagicMock()
                mock_engine.run.return_value = pd.DataFrame({'close': [50000.0]})
                mock_engine_class.return_value = mock_engine
                
                runner = StrategyRunner(
                    config_path=mock_config_file,
                    paper_trader=mock_paper_trader,
                    stream_manager=mock_stream_manager,
                    data_dir=Path(tmpdir)
                )
                
                with pytest.raises(StrategyRunnerError, match="Required symbol BTC/USDT not available"):
                    await runner.start()
    
    def test_get_statistics(self, mock_config_file, mock_paper_trader, mock_stream_manager):
        """Test statistics retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = StrategyRunner(
                config_path=mock_config_file,
                paper_trader=mock_paper_trader,
                stream_manager=mock_stream_manager,
                data_dir=Path(tmpdir)
            )
            runner.config = {"symbol": "BTC/USDT"}
            
            stats = runner.get_statistics()
            
            assert 'strategy_id' in stats
            assert 'strategy_name' in stats
            assert 'is_running' in stats
            assert stats['symbol'] == "BTC/USDT"
    
    def test_get_current_position_no_position(self, mock_config_file, mock_paper_trader, mock_stream_manager):
        """Test getting current position when none exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = StrategyRunner(
                config_path=mock_config_file,
                paper_trader=mock_paper_trader,
                stream_manager=mock_stream_manager,
                data_dir=Path(tmpdir)
            )
            runner.config = {"symbol": "BTC/USDT"}
            
            position = runner.get_current_position()
            
            assert position is None
    
    def test_get_current_position_with_position(self, mock_config_file, mock_paper_trader, mock_stream_manager):
        """Test getting current position when one exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mock position
            mock_position = Position(symbol="BTC/USDT", size=0.1, average_price=50000.0)
            mock_paper_trader.get_position.return_value = mock_position
            
            runner = StrategyRunner(
                config_path=mock_config_file,
                paper_trader=mock_paper_trader,
                stream_manager=mock_stream_manager,
                data_dir=Path(tmpdir)
            )
            runner.config = {"symbol": "BTC/USDT"}
            
            position = runner.get_current_position()
            
            assert position is not None
            assert position['symbol'] == "BTC/USDT"
            assert position['size'] == 0.1


class TestStrategyRunnerIntegration:
    """Integration tests for StrategyRunner with real components."""
    
    @pytest.fixture
    def mock_config_file(self):
        """Create a temporary configuration file for integration tests."""
        config = {
            "name": "Integration Test Strategy",
            "symbol": "BTC/USDT",
            "timeframes": {"entry": "15m"},
            "indicators": [
                {"id": "EMA_20", "type": "EMA", "timeframe": "15m", "length": 20}
            ],
            "logic": {
                "entry_long": ["close > 49000"],
                "exit_long": ["close < 49500"]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def mock_paper_trader(self):
        """Create a mock PaperTrader for integration tests."""
        trader = MagicMock(spec=PaperTrader)
        trader.get_position.return_value = None
        trader.get_current_price.return_value = 50000.0
        trader.submit_order = AsyncMock()
        trader.get_balance_info.return_value = {'available_balance': 10000.0}
        trader.get_performance_stats.return_value = {'total_trades': 0}
        trader.set_stream_manager = MagicMock()
        return trader
    
    @pytest.fixture
    def mock_stream_manager(self):
        """Create a mock StreamManager for integration tests."""
        from datetime import UTC, datetime, timedelta
        
        manager = MagicMock(spec=StreamManager)
        manager.symbols = ["BTC/USDT"]
        
        # Create sample data
        timestamps = pd.date_range(
            start=datetime.now(UTC) - timedelta(hours=24),
            end=datetime.now(UTC),
            freq='15min'
        )
        
        sample_data = pd.DataFrame({
            'open': [50000.0] * len(timestamps),
            'high': [50100.0] * len(timestamps),
            'low': [49900.0] * len(timestamps),
            'close': [50000.0] * len(timestamps),
            'volume': [1000.0] * len(timestamps)
        }, index=timestamps)
        
        manager.get_data.return_value = sample_data
        return manager
    
    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self):
        """Test complete workflow with simulated components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test configuration
            config = {
                "name": "Integration Test Strategy",
                "symbol": "BTC/USDT",
                "timeframes": {"entry": "15m"},
                "indicators": [
                    {"id": "EMA_20", "type": "EMA", "timeframe": "15m", "length": 20}
                ],
                "logic": {
                    "entry_long": ["close > 49000"],
                    "exit_long": ["close < 49500"]
                }
            }
            
            config_file = Path(tmpdir) / "test_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            # Create mock components
            paper_trader = MagicMock(spec=PaperTrader)
            paper_trader.get_position.return_value = None
            paper_trader.get_current_price.return_value = 50000.0
            paper_trader.submit_order = AsyncMock()
            paper_trader.get_balance_info.return_value = {'available_balance': 10000.0}
            paper_trader.get_performance_stats.return_value = {'total_trades': 0}
            paper_trader.set_stream_manager = MagicMock()
            
            stream_manager = MagicMock(spec=StreamManager)
            stream_manager.symbols = ["BTC/USDT"]
            
            # Create sample data that should trigger entry signal
            timestamps = pd.date_range(
                start=datetime.now(UTC) - timedelta(hours=1),
                end=datetime.now(UTC),
                freq='15min'
            )
            
            sample_data = pd.DataFrame({
                'open': [50000.0] * len(timestamps),
                'high': [50100.0] * len(timestamps),
                'low': [49900.0] * len(timestamps),
                'close': [50000.0] * len(timestamps),  # close > 49000 should trigger long entry
                'volume': [1000.0] * len(timestamps)
            }, index=timestamps)
            
            stream_manager.get_data.return_value = sample_data
            
            with patch('btc_research.live.strategy_runner.Engine') as mock_engine_class:
                # Setup mock engine that returns data with EMA indicator
                mock_engine = MagicMock()
                engine_data = sample_data.copy()
                engine_data['EMA_20'] = [49500.0] * len(timestamps)  # EMA below close
                mock_engine.run.return_value = engine_data
                mock_engine.indicator_objects = []
                mock_engine_class.return_value = mock_engine
                
                # Create and start strategy runner
                runner = StrategyRunner(
                    config_path=str(config_file),
                    paper_trader=paper_trader,
                    stream_manager=stream_manager,
                    data_dir=Path(tmpdir)
                )
                
                try:
                    await runner.start()
                    
                    # Give it a moment to process
                    await asyncio.sleep(0.1)
                    
                    # Verify initialization
                    assert runner.is_running
                    assert runner.config is not None
                    assert runner.strategy_state is not None
                    
                    stats = runner.get_statistics()
                    assert stats['is_running']
                    assert stats['symbol'] == "BTC/USDT"
                    
                finally:
                    await runner.stop()
                    
                    assert not runner.is_running
    
    def test_strategy_runner_repr(self, mock_config_file, mock_paper_trader, mock_stream_manager):
        """Test string representation of StrategyRunner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = StrategyRunner(
                config_path=mock_config_file,
                paper_trader=mock_paper_trader,
                stream_manager=mock_stream_manager,
                data_dir=Path(tmpdir)
            )
            
            repr_str = repr(runner)
            
            assert "StrategyRunner" in repr_str
            assert runner.strategy_name in repr_str
            assert runner.strategy_id[:8] in repr_str
            assert "stopped" in repr_str  # Should be stopped initially


if __name__ == "__main__":
    pytest.main([__file__, "-v"])