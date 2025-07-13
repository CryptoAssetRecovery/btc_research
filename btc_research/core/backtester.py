"""
Backtester module for the BTC research framework.

This module implements the Backtester class that wraps Backtrader functionality
to execute backtests on the DataFrame produced by the Engine. It dynamically
builds Backtrader strategy classes from YAML configuration logic expressions
and provides comprehensive performance statistics.
"""

import warnings
from typing import Any

import backtrader as bt
import numpy as np
import pandas as pd

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False


__all__ = ["Backtester", "BacktesterError", "StrategyLogic"]


class BacktesterError(Exception):
    """Raised when there are issues with backtester operations."""

    pass


class PandasData(bt.feeds.PandasData):
    """
    Custom Backtrader data feed for pandas DataFrame.

    Extends the standard PandasData feed to handle additional columns
    from indicators that are computed by the Engine.
    """

    # Define the lines (columns) that this data feed will contain
    lines = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    # Map DataFrame columns to Backtrader lines
    params = (
        ("datetime", None),  # Use DataFrame index as datetime
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),  # Not used for crypto
    )


class StrategyLogic:
    """
    Helper class to isolate strategy logic for testing and reuse.

    This class contains the core logic evaluation methods that can be
    tested independently of Backtrader's strategy framework.
    """

    def __init__(
        self, config: dict[str, Any], dataframe: pd.DataFrame, debug: bool = False
    ):
        """Initialize strategy logic with configuration and data."""
        self.config = config
        self.df = dataframe
        self.debug = debug

        # Extract logic expressions from config
        self.logic = self.config.get("logic", {})
        self.entry_long_rules = self.logic.get("entry_long", [])
        self.exit_long_rules = self.logic.get("exit_long", [])
        self.entry_short_rules = self.logic.get("entry_short", [])
        self.exit_short_rules = self.logic.get("exit_short", [])

    def evaluate_rules(self, rules: list[str], context: dict[str, Any]) -> bool:
        """
        Evaluate a list of logic rules using safe expression evaluation.

        Args:
            rules: List of expression strings to evaluate
            context: Dictionary of variables for evaluation

        Returns:
            True if ALL rules evaluate to True, False otherwise
        """
        if not rules:
            return False

        try:
            for rule in rules:
                result = self._evaluate_single_rule(rule, context)

                # Handle various result types
                if pd.isna(result) or result is None:
                    return False
                elif isinstance(result, (bool, np.bool_)):
                    if not result:
                        return False
                elif isinstance(result, (int, float, np.number)):
                    if not bool(result):
                        return False
                else:
                    # For other types, check truthiness
                    if not result:
                        return False

            return True  # All rules passed

        except Exception as e:
            if self.debug:
                print(f"Error evaluating rules {rules}: {e}")
            return False

    def _evaluate_single_rule(self, rule: str, context: dict[str, Any]) -> Any:
        """
        Evaluate a single rule expression safely.
        
        This method handles the pandas.eval string comparison bug by implementing
        manual parsing for common expression patterns.
        """
        rule = rule.strip()
        
        # Handle string equality comparisons manually (pandas.eval bug workaround)
        if " == " in rule and ("'" in rule or '"' in rule):
            parts = rule.split(" == ")
            if len(parts) == 2:
                var_name = parts[0].strip()
                expected_value = parts[1].strip().strip("'\"")
                
                if var_name in context:
                    actual_value = context[var_name]
                    return actual_value == expected_value
        
        # Handle string inequality comparisons
        if " != " in rule and ("'" in rule or '"' in rule):
            parts = rule.split(" != ")
            if len(parts) == 2:
                var_name = parts[0].strip()
                expected_value = parts[1].strip().strip("'\"")
                
                if var_name in context:
                    actual_value = context[var_name]
                    return actual_value != expected_value
        
        # For numeric comparisons, use pandas.eval as it works fine
        try:
            return pd.eval(rule, local_dict=context)
        except Exception as e:
            if self.debug:
                print(f"pandas.eval failed for rule '{rule}': {e}")
            raise


class DynamicStrategy(bt.Strategy):
    """
    Dynamically generated Backtrader strategy class.

    This strategy class is built at runtime based on the logic expressions
    defined in the YAML configuration. It evaluates entry and exit conditions
    using pandas.eval() for safe expression evaluation.
    """

    params = (
        ("config", None),  # Configuration dictionary
        ("dataframe", None),  # Original DataFrame with indicator columns
        ("debug", False),  # Enable debug logging
    )

    def __init__(self):
        """Initialize the strategy with configuration and data."""
        self.config = self.params.config
        self.df = self.params.dataframe
        self.debug = self.params.debug

        if self.config is None:
            raise BacktesterError("Strategy requires config parameter")
        if self.df is None:
            raise BacktesterError("Strategy requires dataframe parameter")

        # Initialize strategy logic helper
        self.strategy_logic = StrategyLogic(self.config, self.df, self.debug)

        # Track current position state
        self.position_size = 0
        self.current_index = 0

        if self.debug:
            print(f"Strategy initialized with {len(self.df)} data points")
            print(f"Entry long rules: {self.strategy_logic.entry_long_rules}")
            print(f"Exit long rules: {self.strategy_logic.exit_long_rules}")

    def next(self):
        """
        Called on each bar to evaluate trading logic.

        This method gets the current bar data from the DataFrame and evaluates
        all trading rules using pandas.eval() for safe expression evaluation.
        """
        try:
            # Get current bar index
            current_time = self.data.datetime.datetime(0)

            # Ensure timezone consistency for comparison
            if hasattr(self.df.index, 'tz') and self.df.index.tz is not None:
                # DataFrame has timezone, convert current_time to match
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=self.df.index.tz)
                else:
                    current_time = current_time.astimezone(self.df.index.tz)

            # Find corresponding row in DataFrame
            try:
                # Try to find exact timestamp match
                if current_time in self.df.index:
                    current_row = self.df.loc[current_time]
                else:
                    # Find nearest timestamp
                    idx = self.df.index.get_indexer([current_time], method="nearest")[0]
                    if idx >= 0 and idx < len(self.df):
                        current_row = self.df.iloc[idx]
                    else:
                        return  # Skip if no valid data

            except (KeyError, IndexError):
                return  # Skip if timestamp not found

            # Convert row to dictionary for pandas.eval
            eval_context = current_row.to_dict()

            # Handle NaN values - replace with None for evaluation
            for key, value in eval_context.items():
                if pd.isna(value):
                    eval_context[key] = None

            # Evaluate entry conditions for long positions
            if not self.position and len(self.strategy_logic.entry_long_rules) > 0:
                if self.strategy_logic.evaluate_rules(
                    self.strategy_logic.entry_long_rules, eval_context
                ):
                    order = self.buy()
                    if self.debug:
                        print(f"Long entry at {current_time}: {self.data.close[0]} (position: {self.position.size}, order: {order})")

            # Evaluate entry conditions for short positions
            if not self.position and len(self.strategy_logic.entry_short_rules) > 0:
                if self.strategy_logic.evaluate_rules(
                    self.strategy_logic.entry_short_rules, eval_context
                ):
                    order = self.sell()
                    if self.debug:
                        print(f"Short entry at {current_time}: {self.data.close[0]} (position: {self.position.size}, order: {order})")

            # Evaluate exit conditions for long positions
            if self.position.size > 0 and len(self.strategy_logic.exit_long_rules) > 0:
                if self.strategy_logic.evaluate_rules(
                    self.strategy_logic.exit_long_rules, eval_context
                ):
                    order = self.close()
                    if self.debug:
                        print(f"Long exit at {current_time}: {self.data.close[0]} (position: {self.position.size}, order: {order})")

            # Evaluate exit conditions for short positions
            if self.position.size < 0 and len(self.strategy_logic.exit_short_rules) > 0:
                if self.strategy_logic.evaluate_rules(
                    self.strategy_logic.exit_short_rules, eval_context
                ):
                    order = self.close()
                    if self.debug:
                        print(f"Short exit at {current_time}: {self.data.close[0]} (position: {self.position.size}, order: {order})")

        except Exception as e:
            if self.debug:
                print(f"Error in next(): {e}")
            # Don't raise - just skip this bar
            pass

    def notify_order(self, order):
        """Called when an order is executed."""
        if self.debug:
            if order.status in [order.Submitted, order.Accepted]:
                print(f"Order submitted: {order}")
            elif order.status in [order.Completed]:
                print(f"Order completed: {order} - Price: {order.executed.price}, Size: {order.executed.size}")
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                print(f"Order failed: {order} - Status: {order.status}")

    def notify_trade(self, trade):
        """Called when a trade is completed."""
        if self.debug:
            if trade.isclosed:
                print(f"Trade closed: P&L: {trade.pnl:.2f}, Comm: {trade.commission:.2f}")


class Backtester:
    """
    Backtester wrapper around Backtrader functionality.

    This class provides a simple interface to execute backtests on DataFrames
    produced by the Engine. It converts DataFrames to Backtrader feeds,
    dynamically builds strategy classes from configuration logic, and returns
    comprehensive performance statistics.

    Key features:
    - DataFrame to Backtrader feed conversion
    - Dynamic strategy generation from YAML logic expressions
    - Safe expression evaluation using pandas.eval()
    - Comprehensive performance statistics
    - Optional equity curve plotting

    Example:
        >>> import yaml
        >>> with open("config/demo.yaml") as f:
        ...     config = yaml.safe_load(f)
        >>> backtester = Backtester(config)
        >>> stats = backtester.run(df, cash=10000, commission=0.0004)
        >>> print(f"Total return: {stats['total_return']:.2%}")
    """

    def __init__(self, config: dict[str, Any], debug: bool = False):
        """
        Initialize the Backtester with configuration.

        Args:
            config: Configuration dictionary containing logic expressions
            debug: Enable debug output

        Raises:
            BacktesterError: If configuration is invalid
        """
        self.config = config
        self.debug = debug
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration structure."""
        if not isinstance(self.config, dict):
            raise BacktesterError("Config must be a dictionary")

        # Check for logic section
        if "logic" not in self.config:
            raise BacktesterError("Config must contain 'logic' section")

        logic = self.config["logic"]
        if not isinstance(logic, dict):
            raise BacktesterError("Logic section must be a dictionary")

        # Validate that logic expressions are lists
        for key in ["entry_long", "exit_long", "entry_short", "exit_short"]:
            if key in logic and not isinstance(logic[key], list):
                raise BacktesterError(f"Logic.{key} must be a list of expressions")

    def _create_data_feed(self, df: pd.DataFrame) -> bt.feeds.PandasData:
        """
        Convert pandas DataFrame to Backtrader data feed.

        Args:
            df: DataFrame with OHLCV data and indicator columns

        Returns:
            Backtrader PandasData feed

        Raises:
            BacktesterError: If DataFrame is invalid or missing required columns
        """
        if df is None or len(df) == 0:
            raise BacktesterError("DataFrame cannot be None or empty")

        # Check for required OHLCV columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise BacktesterError(f"DataFrame missing required columns: {missing_cols}")

        # Ensure DataFrame has datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise BacktesterError("DataFrame must have DatetimeIndex")

        # Create a copy for Backtrader (it may modify the data)
        feed_df = df[required_cols].copy()

        # Ensure proper data types
        for col in required_cols:
            feed_df[col] = pd.to_numeric(feed_df[col], errors="coerce")

        # Remove any rows with NaN in OHLCV data
        initial_len = len(feed_df)
        feed_df = feed_df.dropna()
        if len(feed_df) < initial_len:
            warnings.warn(
                f"Removed {initial_len - len(feed_df)} rows with NaN OHLCV data"
            )

        if len(feed_df) == 0:
            raise BacktesterError("No valid OHLCV data after cleaning")

        # Create Backtrader data feed
        return PandasData(dataname=feed_df)

    def run(
        self,
        df: pd.DataFrame,
        cash: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
    ) -> dict[str, Any]:
        """
        Execute backtest and return comprehensive statistics.

        Args:
            df: DataFrame with OHLCV data and indicator columns
            cash: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (not implemented yet)

        Returns:
            Dictionary with comprehensive backtest statistics including:
            - total_return: Total return percentage
            - sharpe_ratio: Sharpe ratio
            - max_drawdown: Maximum drawdown percentage
            - num_trades: Number of trades executed
            - win_rate: Percentage of winning trades
            - trades: List of trade details
            - equity_curve: Portfolio value over time

        Raises:
            BacktesterError: If backtest execution fails
        """
        try:
            # Create Backtrader cerebro engine
            cerebro = bt.Cerebro()

            # Set broker parameters
            cerebro.broker.setcash(cash)
            cerebro.broker.setcommission(commission=commission)

            # Add position sizer - use percentage of available cash
            cerebro.addsizer(bt.sizers.PercentSizer, percents=95)  # Use 95% of available cash

            # Create and add data feed
            data_feed = self._create_data_feed(df)
            cerebro.adddata(data_feed)

            # Add dynamically created strategy
            cerebro.addstrategy(
                DynamicStrategy, config=self.config, dataframe=df, debug=self.debug
            )

            # Add analyzers for comprehensive statistics
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

            if self.debug:
                print(f"Starting backtest with cash: {cash}, commission: {commission}")

            # Run the backtest
            strategies = cerebro.run()
            strategy = strategies[0]

            # Extract results from analyzers
            trade_analyzer = strategy.analyzers.trades.get_analysis()
            sharpe_analyzer = strategy.analyzers.sharpe.get_analysis()
            drawdown_analyzer = strategy.analyzers.drawdown.get_analysis()
            returns_analyzer = strategy.analyzers.returns.get_analysis()
            timereturn_analyzer = strategy.analyzers.timereturn.get_analysis()

            # Calculate comprehensive statistics
            stats = self._calculate_stats(
                trade_analyzer,
                sharpe_analyzer,
                drawdown_analyzer,
                returns_analyzer,
                timereturn_analyzer,
                cerebro.broker.getvalue(),
                cash,
            )

            if self.debug:
                print(
                    f"Backtest completed. Final portfolio value: {cerebro.broker.getvalue():.2f}"
                )

            return stats

        except Exception as e:
            raise BacktesterError(f"Backtest execution failed: {e}") from e

    def _calculate_stats(
        self,
        trade_analyzer: dict,
        sharpe_analyzer: dict,
        drawdown_analyzer: dict,
        returns_analyzer: dict,
        timereturn_analyzer: dict,
        final_value: float,
        initial_cash: float,
    ) -> dict[str, Any]:
        """
        Calculate comprehensive performance statistics from analyzer results.

        Args:
            trade_analyzer: TradeAnalyzer results
            sharpe_analyzer: SharpeRatio results
            drawdown_analyzer: DrawDown results
            returns_analyzer: Returns results
            timereturn_analyzer: TimeReturn results
            final_value: Final portfolio value
            initial_cash: Initial cash amount

        Returns:
            Dictionary with comprehensive statistics
        """
        stats = {}

        # Basic performance metrics
        total_return = (final_value - initial_cash) / initial_cash
        stats["total_return"] = total_return
        stats["final_value"] = final_value
        stats["initial_cash"] = initial_cash

        # Sharpe ratio - handle None and NaN values
        sharpe_ratio = sharpe_analyzer.get("sharperatio", None)
        if sharpe_ratio is not None and not np.isnan(sharpe_ratio):
            stats["sharpe_ratio"] = sharpe_ratio
        else:
            # Calculate basic Sharpe ratio from returns if available
            if returns_analyzer and hasattr(returns_analyzer, 'get'):
                rnorm = returns_analyzer.get('rnorm', None)
                rnorm100 = returns_analyzer.get('rnorm100', None)
                if rnorm is not None and rnorm != 0:
                    stats["sharpe_ratio"] = rnorm
                elif rnorm100 is not None:
                    stats["sharpe_ratio"] = rnorm100 / 100.0
                else:
                    stats["sharpe_ratio"] = 0.0
            else:
                stats["sharpe_ratio"] = 0.0

        # Drawdown metrics
        max_drawdown = drawdown_analyzer.get("max", {}).get("drawdown", 0.0)
        stats["max_drawdown"] = max_drawdown / 100.0 if max_drawdown else 0.0

        # Trade statistics - handle AutoOrderedDict structure correctly
        total_section = trade_analyzer.get("total", {})
        stats["num_trades"] = total_section.get("total", 0) if total_section else 0

        # Win rate calculation - get data from correct sections
        won_section = trade_analyzer.get("won", {})
        lost_section = trade_analyzer.get("lost", {})
        
        won_trades = won_section.get("total", 0) if won_section else 0
        lost_trades = lost_section.get("total", 0) if lost_section else 0
        total_trade_count = stats["num_trades"]
        
        stats["win_rate"] = (
            won_trades / total_trade_count if total_trade_count > 0 else 0.0
        )

        # Profit factor - use correct structure for P&L data
        if won_section and lost_section:
            won_pnl_section = won_section.get("pnl", {})
            lost_pnl_section = lost_section.get("pnl", {})
            
            gross_profit = won_pnl_section.get("total", 0) if won_pnl_section else 0
            gross_loss = abs(lost_pnl_section.get("total", 0)) if lost_pnl_section else 0
            
            stats["profit_factor"] = (
                gross_profit / gross_loss
                if gross_loss > 0
                else float("inf")
                if gross_profit > 0
                else 0.0
            )
        else:
            stats["profit_factor"] = 0.0

        # Average trade metrics - use net P&L average
        if stats["num_trades"] > 0:
            pnl_section = trade_analyzer.get("pnl", {})
            if pnl_section:
                net_pnl_section = pnl_section.get("net", {})
                avg_trade = net_pnl_section.get("average", 0) if net_pnl_section else 0
                stats["avg_trade"] = avg_trade
            else:
                stats["avg_trade"] = 0.0
        else:
            stats["avg_trade"] = 0.0

        # Equity curve from timereturn analyzer
        equity_curve = []
        if timereturn_analyzer:
            cumulative_return = 1.0
            for date, ret in timereturn_analyzer.items():
                cumulative_return *= 1 + ret
                equity_curve.append(
                    {
                        "timestamp": date.strftime("%Y-%m-%d %H:%M:%S")
                        if hasattr(date, "strftime")
                        else str(date),
                        "equity": initial_cash * cumulative_return,
                        "return": cumulative_return - 1,
                    }
                )

        stats["equity_curve"] = equity_curve

        # Detailed trade list (simplified for now)
        trades = []
        stats["trades"] = trades

        return stats

    def plot(self, cerebro=None, **kwargs) -> None:
        """
        Plot equity curve and trade analysis.

        Args:
            cerebro: Backtrader cerebro instance (if available)
            **kwargs: Additional plotting arguments
        """
        if not HAS_MATPLOTLIB:
            warnings.warn(
                "Matplotlib not available for plotting. Install with: pip install matplotlib"
            )
            return

        if cerebro is not None:
            cerebro.plot(**kwargs)
        else:
            warnings.warn("No cerebro instance available for plotting")


def create_backtest_summary(stats: dict[str, Any]) -> str:
    """
    Create a formatted summary of backtest results.

    Args:
        stats: Statistics dictionary from Backtester.run()

    Returns:
        Formatted string summary
    """
    summary = f"""
Backtest Results Summary
========================
Total Return:     {stats['total_return']:.2%}
Sharpe Ratio:     {stats['sharpe_ratio']:.2f}
Max Drawdown:     {stats['max_drawdown']:.2%}
Number of Trades: {stats['num_trades']}
Win Rate:         {stats['win_rate']:.2%}
Profit Factor:    {stats['profit_factor']:.2f}
Average Trade:    ${stats['avg_trade']:.2f}
Final Value:      ${stats['final_value']:.2f}
Initial Cash:     ${stats['initial_cash']:.2f}
"""
    return summary.strip()
