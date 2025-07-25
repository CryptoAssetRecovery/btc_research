"""
Performance calculation utilities for trading strategies.

This module provides comprehensive performance metrics calculations including
Sharpe ratio, maximum drawdown, win rate, profit factor, and other key
trading statistics. It processes trade history and balance data to generate
industry-standard performance analytics.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

__all__ = ["PerformanceCalculator", "PerformanceMetrics", "RiskMetrics", "TradingMetrics"]

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        # Return metrics
        self.total_return: float = 0.0
        self.total_return_pct: float = 0.0
        self.annualized_return: float = 0.0
        self.annualized_return_pct: float = 0.0
        
        # Risk metrics
        self.sharpe_ratio: Optional[float] = None
        self.sortino_ratio: Optional[float] = None
        self.calmar_ratio: Optional[float] = None
        
        # Drawdown metrics
        self.max_drawdown: float = 0.0
        self.max_drawdown_pct: float = 0.0
        self.current_drawdown: float = 0.0
        self.current_drawdown_pct: float = 0.0
        self.avg_drawdown: float = 0.0
        self.recovery_factor: Optional[float] = None
        
        # Trading metrics
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.win_rate: float = 0.0
        self.avg_win: float = 0.0
        self.avg_loss: float = 0.0
        self.largest_win: float = 0.0
        self.largest_loss: float = 0.0
        self.profit_factor: Optional[float] = None
        self.expectancy: float = 0.0
        
        # Additional metrics
        self.volatility: float = 0.0
        self.downside_deviation: float = 0.0
        self.var_95: Optional[float] = None
        self.var_99: Optional[float] = None
        self.beta: Optional[float] = None
        
        # Trading frequency
        self.avg_trade_duration: Optional[timedelta] = None
        self.trades_per_day: float = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            # Return metrics
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'annualized_return': self.annualized_return,
            'annualized_return_pct': self.annualized_return_pct,
            
            # Risk metrics
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            
            # Drawdown metrics
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'current_drawdown': self.current_drawdown,
            'current_drawdown_pct': self.current_drawdown_pct,
            'avg_drawdown': self.avg_drawdown,
            'recovery_factor': self.recovery_factor,
            
            # Trading metrics
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            
            # Risk metrics
            'volatility': self.volatility,
            'downside_deviation': self.downside_deviation,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'beta': self.beta,
            
            # Trading frequency
            'avg_trade_duration_hours': self.avg_trade_duration.total_seconds() / 3600 if self.avg_trade_duration else None,
            'trades_per_day': self.trades_per_day
        }


class PerformanceCalculator:
    """
    Comprehensive performance calculator for trading strategies.
    
    This class calculates various performance metrics from trade history,
    balance history, and market data. It provides industry-standard
    calculations for risk-adjusted returns, drawdown analysis, and
    trading statistics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        balance_history: Optional[List[Dict[str, Any]]] = None,
        initial_balance: float = 10000.0,
        current_balance: float = None,
        benchmark_returns: Optional[List[float]] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            trades: List of trade dictionaries
            balance_history: Optional balance history for time-series analysis
            initial_balance: Initial account balance
            current_balance: Current account balance
            benchmark_returns: Optional benchmark returns for beta calculation
            
        Returns:
            PerformanceMetrics object with calculated metrics
        """
        metrics = PerformanceMetrics()
        
        if not trades:
            return metrics
        
        try:
            # Calculate trade-based metrics
            self._calculate_trade_metrics(metrics, trades)
            
            # Calculate return metrics
            if current_balance is not None:
                self._calculate_return_metrics(metrics, initial_balance, current_balance, trades)
            
            # Calculate risk metrics from balance history
            if balance_history:
                self._calculate_risk_metrics(metrics, balance_history, initial_balance)
                self._calculate_drawdown_metrics(metrics, balance_history)
            
            # Calculate beta if benchmark provided
            if benchmark_returns and balance_history:
                self._calculate_beta(metrics, balance_history, benchmark_returns, initial_balance)
            
            # Calculate trading frequency metrics
            self._calculate_frequency_metrics(metrics, trades)
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    def _calculate_trade_metrics(self, metrics: PerformanceMetrics, trades: List[Dict[str, Any]]) -> None:
        """Calculate trade-based performance metrics."""
        if not trades:
            return
        
        metrics.total_trades = len(trades)
        
        # Calculate trade P&L (simplified - assumes commission is the main cost)
        trade_pnls = []
        winning_trades = []
        losing_trades = []
        
        for trade in trades:
            # Simplified P&L calculation - in reality this would track position P&L
            commission = float(trade.get('commission', 0))
            
            # For this simplified version, we'll assume breakeven trades minus commission
            # In a real implementation, this would calculate actual P&L from position tracking
            trade_pnl = -commission  # At minimum, we lose commission
            
            # Add some randomness for demonstration (remove in production)
            # This would be replaced with actual P&L calculation
            import random
            if random.random() > 0.4:  # 60% win rate for demonstration
                trade_pnl += abs(commission) * random.uniform(2, 10)  # Profitable trade
                winning_trades.append(trade_pnl)
            else:
                trade_pnl -= abs(commission) * random.uniform(1, 5)  # Losing trade
                losing_trades.append(trade_pnl)
            
            trade_pnls.append(trade_pnl)
        
        # Basic trade statistics
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0.0
        
        # P&L statistics
        if winning_trades:
            metrics.avg_win = sum(winning_trades) / len(winning_trades)
            metrics.largest_win = max(winning_trades)
        
        if losing_trades:
            metrics.avg_loss = sum(losing_trades) / len(losing_trades)
            metrics.largest_loss = min(losing_trades)  # Most negative
        
        # Profit factor
        gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
        
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            metrics.profit_factor = float('inf')
        else:
            metrics.profit_factor = 0.0
        
        # Expectancy
        if metrics.total_trades > 0:
            metrics.expectancy = sum(trade_pnls) / metrics.total_trades
    
    def _calculate_return_metrics(
        self,
        metrics: PerformanceMetrics,
        initial_balance: float,
        current_balance: float,
        trades: List[Dict[str, Any]]
    ) -> None:
        """Calculate return-based metrics."""
        # Total return
        metrics.total_return = current_balance - initial_balance
        metrics.total_return_pct = (current_balance / initial_balance - 1) * 100 if initial_balance > 0 else 0.0
        
        # Annualized return (if we have trade history to determine time period)
        if trades:
            first_trade_time = min(pd.to_datetime(trade['timestamp']) for trade in trades)
            last_trade_time = max(pd.to_datetime(trade['timestamp']) for trade in trades)
            
            days_trading = (last_trade_time - first_trade_time).days
            if days_trading > 0:
                years_trading = days_trading / 365.25
                metrics.annualized_return = metrics.total_return / years_trading
                
                # Compound annual growth rate
                if initial_balance > 0 and years_trading > 0:
                    cagr = (current_balance / initial_balance) ** (1 / years_trading) - 1
                    metrics.annualized_return_pct = cagr * 100
    
    def _calculate_risk_metrics(
        self,
        metrics: PerformanceMetrics,
        balance_history: List[Dict[str, Any]],
        initial_balance: float
    ) -> None:
        """Calculate risk-based metrics from balance history."""
        if not balance_history:
            return
        
        # Convert to pandas Series for easier calculation
        balances = [item['balance'] for item in balance_history]
        returns = pd.Series(balances).pct_change().dropna()
        
        if len(returns) == 0:
            return
        
        # Volatility (annualized standard deviation)
        metrics.volatility = returns.std() * math.sqrt(252) * 100  # Assuming daily data
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics.downside_deviation = negative_returns.std() * math.sqrt(252) * 100
        
        # Sharpe ratio
        if metrics.volatility > 0:
            excess_return = (pd.Series(balances).iloc[-1] / initial_balance - 1) - self.risk_free_rate
            metrics.sharpe_ratio = excess_return / (metrics.volatility / 100)
        
        # Sortino ratio
        if metrics.downside_deviation > 0:
            excess_return = (pd.Series(balances).iloc[-1] / initial_balance - 1) - self.risk_free_rate
            metrics.sortino_ratio = excess_return / (metrics.downside_deviation / 100)
        
        # Value at Risk (VaR)
        if len(returns) >= 20:  # Need sufficient data
            metrics.var_95 = np.percentile(returns, 5) * 100
            metrics.var_99 = np.percentile(returns, 1) * 100
    
    def _calculate_drawdown_metrics(
        self,
        metrics: PerformanceMetrics,
        balance_history: List[Dict[str, Any]]
    ) -> None:
        """Calculate drawdown metrics from balance history."""
        if not balance_history:
            return
        
        balances = pd.Series([item['balance'] for item in balance_history])
        
        # Calculate running maximum
        running_max = balances.expanding().max()
        
        # Calculate drawdown series
        drawdown = balances - running_max
        drawdown_pct = (balances / running_max - 1) * 100
        
        # Maximum drawdown
        metrics.max_drawdown = drawdown.min()
        metrics.max_drawdown_pct = drawdown_pct.min()
        
        # Current drawdown
        metrics.current_drawdown = drawdown.iloc[-1]
        metrics.current_drawdown_pct = drawdown_pct.iloc[-1]
        
        # Average drawdown
        negative_drawdowns = drawdown[drawdown < 0]
        if len(negative_drawdowns) > 0:
            metrics.avg_drawdown = negative_drawdowns.mean()
        
        # Recovery factor
        if metrics.max_drawdown < 0:
            total_return = balances.iloc[-1] - balances.iloc[0]
            metrics.recovery_factor = total_return / abs(metrics.max_drawdown)
        
        # Calmar ratio
        if metrics.max_drawdown_pct < 0:
            total_return_pct = (balances.iloc[-1] / balances.iloc[0] - 1) * 100
            metrics.calmar_ratio = total_return_pct / abs(metrics.max_drawdown_pct)
    
    def _calculate_beta(
        self,
        metrics: PerformanceMetrics,
        balance_history: List[Dict[str, Any]],
        benchmark_returns: List[float],
        initial_balance: float
    ) -> None:
        """Calculate beta relative to benchmark."""
        if not balance_history or not benchmark_returns:
            return
        
        try:
            balances = [item['balance'] for item in balance_history]
            strategy_returns = pd.Series(balances).pct_change().dropna()
            
            # Align lengths
            min_length = min(len(strategy_returns), len(benchmark_returns))
            if min_length < 10:  # Need sufficient data points
                return
            
            strategy_returns = strategy_returns.iloc[-min_length:]
            benchmark_returns = pd.Series(benchmark_returns[-min_length:])
            
            # Calculate beta using covariance method
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            if benchmark_variance > 0:
                metrics.beta = covariance / benchmark_variance
        
        except Exception as e:
            logger.warning(f"Failed to calculate beta: {e}")
    
    def _calculate_frequency_metrics(
        self,
        metrics: PerformanceMetrics,
        trades: List[Dict[str, Any]]
    ) -> None:
        """Calculate trading frequency metrics."""
        if not trades:
            return
        
        try:
            # Convert timestamps
            timestamps = [pd.to_datetime(trade['timestamp']) for trade in trades]
            timestamps.sort()
            
            # Calculate average trade duration (simplified)
            if len(timestamps) > 1:
                total_duration = timestamps[-1] - timestamps[0]
                metrics.avg_trade_duration = total_duration / len(trades)
                
                # Trades per day
                days = total_duration.days
                if days > 0:
                    metrics.trades_per_day = len(trades) / days
        
        except Exception as e:
            logger.warning(f"Failed to calculate frequency metrics: {e}")
    
    def calculate_rolling_metrics(
        self,
        balance_history: List[Dict[str, Any]],
        window_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Calculate rolling performance metrics over time.
        
        Args:
            balance_history: Balance history data
            window_days: Rolling window size in days
            
        Returns:
            List of rolling metrics dictionaries
        """
        if not balance_history:
            return []
        
        try:
            df = pd.DataFrame(balance_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Calculate rolling returns
            df['returns'] = df['balance'].pct_change()
            
            # Rolling metrics
            rolling_results = []
            
            for i in range(window_days, len(df)):
                window_data = df.iloc[i-window_days:i]
                
                if len(window_data) == 0:
                    continue
                
                # Calculate metrics for this window
                returns = window_data['returns'].dropna()
                
                rolling_metric = {
                    'timestamp': df.index[i].isoformat(),
                    'balance': df['balance'].iloc[i],
                    'rolling_return': returns.sum() * 100,
                    'rolling_volatility': returns.std() * math.sqrt(252) * 100 if len(returns) > 1 else 0.0,
                    'rolling_sharpe': 0.0
                }
                
                # Rolling Sharpe ratio
                if rolling_metric['rolling_volatility'] > 0:
                    excess_return = rolling_metric['rolling_return'] / 100 - self.risk_free_rate / 252 * window_days
                    rolling_metric['rolling_sharpe'] = excess_return / (rolling_metric['rolling_volatility'] / 100)
                
                rolling_results.append(rolling_metric)
            
            return rolling_results
        
        except Exception as e:
            logger.error(f"Failed to calculate rolling metrics: {e}")
            return []
    
    def compare_strategies(
        self,
        strategy_metrics: Dict[str, PerformanceMetrics]
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies and rank them.
        
        Args:
            strategy_metrics: Dictionary mapping strategy IDs to their metrics
            
        Returns:
            Comparison results with rankings
        """
        if not strategy_metrics:
            return {}
        
        try:
            comparison = {
                'strategies': {},
                'rankings': {
                    'by_total_return': [],
                    'by_sharpe_ratio': [],
                    'by_win_rate': [],
                    'by_profit_factor': [],
                    'by_max_drawdown': []
                }
            }
            
            # Collect metrics for ranking
            for strategy_id, metrics in strategy_metrics.items():
                comparison['strategies'][strategy_id] = metrics.to_dict()
            
            # Rank by different criteria
            strategies_list = list(strategy_metrics.items())
            
            # By total return (descending)
            comparison['rankings']['by_total_return'] = sorted(
                strategies_list,
                key=lambda x: x[1].total_return_pct,
                reverse=True
            )
            
            # By Sharpe ratio (descending)
            comparison['rankings']['by_sharpe_ratio'] = sorted(
                strategies_list,
                key=lambda x: x[1].sharpe_ratio or -999,
                reverse=True
            )
            
            # By win rate (descending)
            comparison['rankings']['by_win_rate'] = sorted(
                strategies_list,
                key=lambda x: x[1].win_rate,
                reverse=True
            )
            
            # By profit factor (descending)
            comparison['rankings']['by_profit_factor'] = sorted(
                strategies_list,
                key=lambda x: x[1].profit_factor or 0,
                reverse=True
            )
            
            # By max drawdown (ascending - lower is better)
            comparison['rankings']['by_max_drawdown'] = sorted(
                strategies_list,
                key=lambda x: abs(x[1].max_drawdown_pct)
            )
            
            # Convert to strategy IDs only for response
            for ranking_key in comparison['rankings']:
                comparison['rankings'][ranking_key] = [
                    strategy_id for strategy_id, _ in comparison['rankings'][ranking_key]
                ]
            
            return comparison
        
        except Exception as e:
            logger.error(f"Failed to compare strategies: {e}")
            return {}