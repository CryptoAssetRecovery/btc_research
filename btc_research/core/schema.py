"""JSON schema validation and YAML configuration loader for BTC research engine.

This module provides:
- JSON schema definition for validating YAML configuration files
- Configuration loading with environment variable substitution
- Comprehensive validation error reporting with line numbers
- Schema versioning support
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from jsonschema import Draft7Validator

# JSON Schema for BTC Research Configuration Files
CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "BTC Research Strategy Configuration",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name",
        "symbol",
        "exchange",
        "timeframes",
        "indicators",
        "logic",
        "backtest",
    ],
    "properties": {
        "version": {
            "type": "string",
            "description": "Schema version for compatibility checking",
            "pattern": r"^\d+\.\d+(\.\d+)?$",
            "default": "1.0",
        },
        "name": {
            "type": "string",
            "description": "Human-readable strategy name",
            "minLength": 1,
            "maxLength": 100,
        },
        "symbol": {
            "type": "string",
            "description": "Trading pair symbol (e.g., BTC/USDC)",
            "pattern": r"^[A-Z0-9]+\/[A-Z0-9]+$",
        },
        "exchange": {
            "type": "string",
            "description": "Exchange name (must be supported by CCXT)",
            "minLength": 1,
        },
        "timeframes": {
            "type": "object",
            "description": "Named timeframe definitions",
            "additionalProperties": {
                "type": "string",
                "pattern": r"^(1|5|15|30)m|[1-9]h|1d$",
            },
            "minProperties": 1,
        },
        "indicators": {
            "type": "array",
            "description": "List of indicators to compute",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": True,
                "required": ["id", "type", "timeframe"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique identifier for this indicator instance",
                        "pattern": r"^[A-Za-z][A-Za-z0-9_]*$",
                    },
                    "type": {
                        "type": "string",
                        "description": "Indicator type (must be registered in registry)",
                        "pattern": r"^[A-Z][A-Z0-9_]*$",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe for this indicator",
                        "pattern": r"^(1|5|15|30)m|[1-9]h|1d$",
                    },
                },
            },
        },
        "logic": {
            "type": "object",
            "description": "Trading logic expressions",
            "additionalProperties": False,
            "properties": {
                "entry_long": {
                    "type": "array",
                    "description": "Conditions for long entry (ALL must be true)",
                    "items": {"type": "string", "minLength": 1},
                    "default": [],
                },
                "exit_long": {
                    "type": "array",
                    "description": "Conditions for long exit (ANY can trigger)",
                    "items": {"type": "string", "minLength": 1},
                    "default": [],
                },
                "entry_short": {
                    "type": "array",
                    "description": "Conditions for short entry (ALL must be true)",
                    "items": {"type": "string", "minLength": 1},
                    "default": [],
                },
                "exit_short": {
                    "type": "array",
                    "description": "Conditions for short exit (ANY can trigger)",
                    "items": {"type": "string", "minLength": 1},
                    "default": [],
                },
            },
        },
        "backtest": {
            "type": "object",
            "description": "Backtesting parameters",
            "additionalProperties": False,
            "required": ["cash", "commission", "slippage", "from", "to"],
            "properties": {
                "cash": {
                    "type": "number",
                    "description": "Starting cash amount",
                    "exclusiveMinimum": 0,
                },
                "commission": {
                    "type": "number",
                    "description": "Commission rate (e.g., 0.001 = 0.1%)",
                    "minimum": 0,
                    "maximum": 1,
                },
                "slippage": {
                    "type": "number",
                    "description": "Slippage rate (e.g., 0.001 = 0.1%)",
                    "minimum": 0,
                    "maximum": 1,
                },
                "from": {
                    "type": "string",
                    "description": "Start date for backtest (YYYY-MM-DD)",
                    "pattern": r"^\d{4}-\d{2}-\d{2}$",
                },
                "to": {
                    "type": "string",
                    "description": "End date for backtest (YYYY-MM-DD)",
                    "pattern": r"^\d{4}-\d{2}-\d{2}$",
                },
            },
        },
    },
}


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(
        self,
        message: str,
        line_number: Optional[int] = None,
        yaml_path: Optional[str] = None,
    ):
        self.message = message
        self.line_number = line_number
        self.yaml_path = yaml_path
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with line number and path if available."""
        parts = []
        if self.yaml_path:
            parts.append(f"File: {self.yaml_path}")
        if self.line_number is not None:
            parts.append(f"Line {self.line_number}")
        if parts:
            return f"{' | '.join(parts)}: {self.message}"
        return self.message


def _substitute_env_vars(content: str) -> str:
    """Substitute environment variables in ${VAR} format.

    Args:
        content: YAML content as string

    Returns:
        Content with environment variables substituted

    Raises:
        ConfigValidationError: If required environment variable is missing
    """

    def replace_var(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ConfigValidationError(
                f"Environment variable '${{{var_name}}}' is not set"
            )
        return value

    # Pattern matches ${VAR_NAME} allowing letters, numbers, and underscores
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace_var, content)


def _convert_types(config: dict[str, Any]) -> dict[str, Any]:
    """Convert string types to appropriate Python types.

    Args:
        config: Raw configuration dictionary

    Returns:
        Configuration with converted types
    """
    # Convert date strings to datetime objects for easier handling
    if "backtest" in config:
        backtest = config["backtest"]
        if "from" in backtest:
            try:
                backtest["from"] = datetime.strptime(
                    backtest["from"], "%Y-%m-%d"
                ).date()
            except ValueError as e:
                raise ConfigValidationError(f"Invalid 'from' date format: {e}")

        if "to" in backtest:
            try:
                backtest["to"] = datetime.strptime(backtest["to"], "%Y-%m-%d").date()
            except ValueError as e:
                raise ConfigValidationError(f"Invalid 'to' date format: {e}")

    return config


def _validate_business_logic(config: dict[str, Any]) -> None:
    """Validate business logic constraints beyond JSON schema.

    Args:
        config: Configuration dictionary

    Raises:
        ConfigValidationError: If business logic validation fails
    """
    # Check that backtest end date is after start date
    if "backtest" in config:
        backtest = config["backtest"]
        from_date = backtest.get("from")
        to_date = backtest.get("to")

        # Check if dates are date objects (after conversion) or strings (before conversion)
        if from_date and to_date:
            if hasattr(from_date, "strftime") and hasattr(to_date, "strftime"):
                # Date objects - direct comparison
                if from_date >= to_date:
                    raise ConfigValidationError(
                        "Backtest 'to' date must be after 'from' date"
                    )
            elif isinstance(from_date, str) and isinstance(to_date, str):
                # String dates - parse and compare
                try:
                    from_parsed = datetime.strptime(from_date, "%Y-%m-%d").date()
                    to_parsed = datetime.strptime(to_date, "%Y-%m-%d").date()
                    if from_parsed >= to_parsed:
                        raise ConfigValidationError(
                            "Backtest 'to' date must be after 'from' date"
                        )
                except ValueError:
                    # Invalid date format - will be caught by schema validation
                    pass

    # Check that indicator timeframes are defined in timeframes section
    if "indicators" in config and "timeframes" in config:
        defined_timeframes = set(config["timeframes"].values())
        for i, indicator in enumerate(config["indicators"]):
            if indicator["timeframe"] not in defined_timeframes:
                raise ConfigValidationError(
                    f"Indicator {indicator['id']} uses timeframe '{indicator['timeframe']}' "
                    f"which is not defined in timeframes section"
                )

    # Check for duplicate indicator IDs
    if "indicators" in config:
        indicator_ids = [ind["id"] for ind in config["indicators"]]
        duplicates = [x for x in indicator_ids if indicator_ids.count(x) > 1]
        if duplicates:
            raise ConfigValidationError(
                f"Duplicate indicator IDs found: {', '.join(set(duplicates))}"
            )


def load_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Validated configuration dictionary with type conversions applied

    Raises:
        ConfigValidationError: If the file cannot be loaded or validation fails
        FileNotFoundError: If the configuration file doesn't exist
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        # Read and substitute environment variables
        with open(config_path, encoding="utf-8") as f:
            content = f.read()

        content = _substitute_env_vars(content)

        # Parse YAML
        try:
            config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            line_number = getattr(e, "problem_mark", None)
            line_num = line_number.line + 1 if line_number else None
            raise ConfigValidationError(
                f"YAML parsing error: {e}",
                line_number=line_num,
                yaml_path=str(config_path),
            )

        if config is None:
            raise ConfigValidationError(
                "Configuration file is empty", yaml_path=str(config_path)
            )

        if not isinstance(config, dict):
            raise ConfigValidationError(
                "Configuration must be a YAML object/dictionary",
                yaml_path=str(config_path),
            )

        # Validate against JSON schema
        validator = Draft7Validator(CONFIG_SCHEMA)
        errors = sorted(validator.iter_errors(config), key=lambda e: e.absolute_path)

        if errors:
            error_messages = []
            for error in errors:
                path = (
                    " -> ".join(str(p) for p in error.absolute_path)
                    if error.absolute_path
                    else "root"
                )
                error_messages.append(f"  {path}: {error.message}")

            raise ConfigValidationError(
                "Schema validation failed:\n" + "\n".join(error_messages),
                yaml_path=str(config_path),
            )

        # Apply type conversions
        config = _convert_types(config)

        # Validate business logic
        _validate_business_logic(config)

        return config

    except ConfigValidationError:
        raise
    except Exception as e:
        raise ConfigValidationError(
            f"Unexpected error loading configuration: {e}", yaml_path=str(config_path)
        )


def validate_config(config: dict[str, Any]) -> None:
    """Validate an already-loaded configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ConfigValidationError: If validation fails
    """
    validator = Draft7Validator(CONFIG_SCHEMA)
    errors = sorted(validator.iter_errors(config), key=lambda e: e.absolute_path)

    if errors:
        error_messages = []
        for error in errors:
            path = (
                " -> ".join(str(p) for p in error.absolute_path)
                if error.absolute_path
                else "root"
            )
            error_messages.append(f"  {path}: {error.message}")

        raise ConfigValidationError(
            "Schema validation failed:\n" + "\n".join(error_messages)
        )

    _validate_business_logic(config)


def get_schema_version() -> str:
    """Get the current schema version."""
    return CONFIG_SCHEMA["properties"]["version"]["default"]


def get_schema() -> dict[str, Any]:
    """Get a copy of the current JSON schema."""
    return CONFIG_SCHEMA.copy()
