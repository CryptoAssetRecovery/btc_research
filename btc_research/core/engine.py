"""
Confluence engine for the BTC research framework.

This module implements the core Engine class that combines multi-timeframe data
and indicators into a single DataFrame ready for vectorized research or backtesting.
The engine follows the exact specifications from ROADMAP.md for maximum compatibility.
"""

import warnings
from typing import Any

import pandas as pd

from btc_research.core.datafeed import DataFeed
from btc_research.core.registry import RegistrationError, get

__all__ = ["Engine", "EngineError"]


class EngineError(Exception):
    """Raised when there are issues with engine operations."""

    pass


class Engine:
    """
    Confluence engine that combines multi-timeframe data and indicators.

    The Engine class reads a YAML configuration, loads requested timeframes,
    instantiates indicators via registry, calls their compute() methods to obtain
    columns, and combines them into one multi-indexed DataFrame ready for research.

    Key features:
    - Multi-timeframe data loading and alignment
    - Dynamic indicator instantiation via registry
    - Forward-fill alignment for higher timeframes
    - Single tidy DataFrame output for research/backtesting
    - Support for both vectorized research and Backtrader handoff

    Example:
        >>> import yaml
        >>> with open("config/demo.yaml") as f:
        ...     config = yaml.safe_load(f)
        >>> engine = Engine(config)
        >>> df = engine.run()
        >>> print(df.head())
    """

    def __init__(self, cfg: dict[str, Any]):
        """
        Initialize the Engine with configuration.

        Args:
            cfg: Configuration dictionary containing:
                - symbol: Trading pair symbol
                - exchange: Exchange name (optional, defaults to "binanceus")
                - timeframes: Dict with entry timeframe and named timeframes
                - indicators: List of indicator specifications
                - backtest: Backtest configuration with date range

        Raises:
            EngineError: If configuration is invalid or missing required fields
        """
        self._validate_config(cfg)
        self.cfg = cfg
        self.indicator_objects = []
        self.datafeed = DataFeed()

    def _validate_config(self, cfg: dict[str, Any]) -> None:
        """Validate the configuration structure and required fields."""
        required_fields = ["symbol", "timeframes", "indicators", "backtest"]
        for field in required_fields:
            if field not in cfg:
                raise EngineError(f"Missing required configuration field: {field}")

        # Validate timeframes
        if "entry" not in cfg["timeframes"]:
            raise EngineError(
                "Configuration must specify 'entry' timeframe in timeframes section"
            )

        # Validate backtest dates
        backtest = cfg["backtest"]
        if "from" not in backtest or "to" not in backtest:
            raise EngineError(
                "Backtest configuration must specify 'from' and 'to' dates"
            )

        # Validate indicators
        if not isinstance(cfg["indicators"], list):
            raise EngineError("Indicators must be specified as a list")

        for i, indicator in enumerate(cfg["indicators"]):
            if not isinstance(indicator, dict):
                raise EngineError(f"Indicator {i} must be a dictionary")

            required_indicator_fields = ["id", "type", "timeframe"]
            for field in required_indicator_fields:
                if field not in indicator:
                    raise EngineError(f"Indicator {i} missing required field: {field}")

    def _load_timeframes(self) -> dict[str, pd.DataFrame]:
        """
        Load data for all unique timeframes used by indicators.

        This method extracts all unique timeframes from the indicator specifications,
        loads OHLCV data for each timeframe using DataFeed, and returns a dictionary
        mapping timeframes to their respective DataFrames.

        Returns:
            Dict[str, pd.DataFrame]: Mapping of timeframes to OHLCV DataFrames

        Raises:
            EngineError: If data loading fails for any timeframe
        """
        data = {}

        # Extract unique timeframes from indicators
        timeframes = set(ind["timeframe"] for ind in self.cfg["indicators"])

        # Add entry timeframe to ensure it's loaded
        entry_tf = self.cfg["timeframes"]["entry"]
        timeframes.add(entry_tf)

        for tf in timeframes:
            try:
                data[tf] = self.datafeed.get(
                    symbol=self.cfg["symbol"],
                    timeframe=tf,
                    start=self.cfg["backtest"]["from"],
                    end=self.cfg["backtest"]["to"],
                    source=self.cfg.get("exchange", "binanceus"),
                )

                if len(data[tf]) == 0:
                    warnings.warn(f"No data loaded for timeframe {tf}")

            except Exception as e:
                raise EngineError(f"Failed to load data for timeframe {tf}: {e}") from e

        return data

    def _instantiate_indicators(self, data_by_tf: dict[str, pd.DataFrame]) -> None:
        """
        Instantiate indicators and compute their outputs on respective timeframes.

        This method loads indicators dynamically via registry, passes parameters from
        config (excluding id, type, timeframe), computes indicators on their respective
        timeframes, renames output columns to include indicator ID prefix, and
        integrates computed columns into timeframe DataFrames.

        Args:
            data_by_tf: Dictionary mapping timeframes to DataFrames (modified in-place)

        Raises:
            EngineError: If indicator instantiation or computation fails
        """
        for spec in self.cfg["indicators"]:
            try:
                # Get indicator class from registry
                indicator_class = get(spec["type"])

                # Extract parameters (exclude framework fields)
                params = {
                    k: v
                    for k, v in spec.items()
                    if k not in ("id", "type", "timeframe")
                }

                # Instantiate indicator
                indicator_obj = indicator_class(**params)

                # Get timeframe data
                timeframe = spec["timeframe"]
                if timeframe not in data_by_tf:
                    raise EngineError(f"Timeframe {timeframe} not found in loaded data")

                tf_data = data_by_tf[timeframe]
                if len(tf_data) == 0:
                    warnings.warn(
                        f"No data available for {spec['id']} on timeframe {timeframe}"
                    )
                    continue

                # Compute indicator output
                indicator_output = indicator_obj.compute(tf_data)

                # Validate output
                if not isinstance(indicator_output, pd.DataFrame):
                    raise EngineError(f"Indicator {spec['id']} must return a DataFrame")

                if len(indicator_output) != len(tf_data):
                    raise EngineError(f"Indicator {spec['id']} output length mismatch")

                # Rename columns to include indicator ID prefix
                # The ROADMAP.md specifies: "out.columns = [f"{spec['id']}_{c}" if c != spec["id"] else spec["id"] for c in out.columns]"
                # However, if columns already start with the indicator ID, don't add prefix again
                renamed_columns = {}
                for col in indicator_output.columns:
                    if col == spec["id"]:
                        # Keep the main indicator column as-is
                        renamed_columns[col] = spec["id"]
                    elif col.startswith(spec["id"] + "_"):
                        # Column already has the correct prefix, keep as-is
                        renamed_columns[col] = col
                    else:
                        # Add the indicator ID prefix
                        renamed_columns[col] = f"{spec['id']}_{col}"

                indicator_output = indicator_output.rename(columns=renamed_columns)

                # Join indicator output with timeframe data
                data_by_tf[timeframe] = data_by_tf[timeframe].join(
                    indicator_output, how="left"
                )

                # Store indicator object for potential debugging
                self.indicator_objects.append(indicator_obj)

            except RegistrationError as e:
                raise EngineError(
                    f"Indicator {spec['type']} not found in registry: {e}"
                ) from e
            except Exception as e:
                raise EngineError(
                    f"Failed to instantiate indicator {spec['id']}: {e}"
                ) from e

    def run(self) -> pd.DataFrame:
        """
        Execute the complete engine workflow and return combined DataFrame.

        This method orchestrates the entire process:
        1. Load multi-timeframe data
        2. Instantiate and compute indicators
        3. Combine timeframes using forward-fill alignment
        4. Return single tidy DataFrame ready for research/backtesting

        The output DataFrame uses the entry timeframe as the base index and forward-fills
        higher timeframe data onto this index. Column naming handles overlaps with
        timeframe suffixes when necessary.

        Returns:
            pd.DataFrame: Combined DataFrame with all OHLCV data and indicator columns
                         indexed by entry timeframe timestamps. Ready for:
                         - Vectorized research (df.query() operations)
                         - Backtrader strategy integration
                         - Statistical analysis and backtesting

        Raises:
            EngineError: If any step of the workflow fails
        """
        try:
            # Step 1: Load data for all timeframes
            data_by_tf = self._load_timeframes()

            # Step 2: Instantiate indicators and compute outputs
            self._instantiate_indicators(data_by_tf)

            # Step 3: Combine multi-timeframe data using forward-fill alignment
            entry_tf = self.cfg["timeframes"]["entry"]

            if entry_tf not in data_by_tf:
                raise EngineError(
                    f"Entry timeframe {entry_tf} not found in loaded data"
                )

            # Start with entry timeframe as base
            base = data_by_tf[entry_tf].copy()

            if len(base) == 0:
                warnings.warn(f"No data in entry timeframe {entry_tf}")
                return base

            # Combine other timeframes by forward-filling onto entry timeframe index
            for tf, df in data_by_tf.items():
                if tf == entry_tf:
                    continue

                if len(df) == 0:
                    warnings.warn(f"Skipping empty timeframe {tf}")
                    continue

                # Reindex higher timeframe data to entry timeframe using forward-fill
                # This aligns higher TF data with entry TF timestamps
                aligned_df = df.reindex(base.index, method="ffill")

                # Handle column name conflicts by adding timeframe suffix
                # Check for overlapping columns (excluding OHLCV which should be the same)
                overlapping_cols = set(base.columns) & set(aligned_df.columns)
                ohlcv_cols = {"open", "high", "low", "close", "volume"}

                # Only add suffix for non-OHLCV overlapping columns
                conflict_cols = overlapping_cols - ohlcv_cols
                if conflict_cols:
                    suffix_dict = {col: f"{col}_{tf}" for col in conflict_cols}
                    aligned_df = aligned_df.rename(columns=suffix_dict)

                # Join the aligned data (exclude OHLCV to avoid duplication)
                cols_to_join = [
                    col for col in aligned_df.columns if col not in ohlcv_cols
                ]
                if cols_to_join:
                    base = base.join(aligned_df[cols_to_join], how="left")

            return base

        except Exception as e:
            if isinstance(e, EngineError):
                raise
            else:
                raise EngineError(f"Engine execution failed: {e}") from e

    def get_indicator_objects(self) -> list[Any]:
        """
        Get list of instantiated indicator objects for debugging.

        Returns:
            List of indicator objects that were created during run()
        """
        return self.indicator_objects.copy()

    def get_config(self) -> dict[str, Any]:
        """
        Get a copy of the engine configuration.

        Returns:
            Copy of the configuration dictionary
        """
        return self.cfg.copy()
