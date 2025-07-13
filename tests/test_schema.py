"""Tests for schema validation and configuration loading."""

import os
import tempfile
from datetime import date
from pathlib import Path

import pytest
import yaml

from btc_research.core.schema import (
    ConfigValidationError,
    get_schema,
    get_schema_version,
    load_config,
    validate_config,
)


@pytest.fixture
def valid_config():
    """Valid configuration dictionary for testing."""
    return {
        "version": "1.0",
        "name": "Test Strategy",
        "symbol": "BTC/USDC",
        "exchange": "binanceus",
        "timeframes": {"main": "15m"},
        "indicators": [
            {"id": "RSI_14", "type": "RSI", "timeframe": "15m", "length": 14}
        ],
        "logic": {
            "entry_long": ["RSI_14 < 30"],
            "exit_long": ["RSI_14 > 70"],
            "entry_short": [],
            "exit_short": [],
        },
        "backtest": {
            "cash": 10000,
            "commission": 0.001,
            "slippage": 0.0,
            "from": "2024-01-01",
            "to": "2024-06-30",
        },
    }


@pytest.fixture
def temp_yaml_file():
    """Create a temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yield f.name
    # Cleanup
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


class TestSchemaValidation:
    """Test JSON schema validation functionality."""

    def test_valid_config_passes_validation(self, valid_config):
        """Test that a valid configuration passes validation."""
        validate_config(valid_config)  # Should not raise

    def test_missing_required_field_fails(self, valid_config):
        """Test that missing required fields cause validation to fail."""
        del valid_config["name"]

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

        assert "name" in str(exc_info.value)
        assert "required" in str(exc_info.value).lower()

    def test_invalid_symbol_format_fails(self, valid_config):
        """Test that invalid symbol format fails validation."""
        valid_config["symbol"] = "BTCUSDC"  # Missing slash

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

        assert "symbol" in str(exc_info.value)

    def test_invalid_timeframe_format_fails(self, valid_config):
        """Test that invalid timeframe format fails validation."""
        valid_config["timeframes"]["main"] = "2x"  # Invalid format

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

    def test_empty_indicators_fails(self, valid_config):
        """Test that empty indicators array fails validation."""
        valid_config["indicators"] = []

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

        assert "indicators" in str(exc_info.value)

    def test_invalid_indicator_id_fails(self, valid_config):
        """Test that invalid indicator ID fails validation."""
        valid_config["indicators"][0]["id"] = "123_invalid"  # Can't start with number

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

    def test_invalid_commission_range_fails(self, valid_config):
        """Test that commission outside valid range fails."""
        valid_config["backtest"]["commission"] = 1.5  # > 1.0

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

    def test_invalid_date_format_fails(self, valid_config):
        """Test that invalid date format fails validation."""
        valid_config["backtest"]["from"] = "2024/01/01"  # Wrong format

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

    def test_additional_properties_allowed_for_indicators(self, valid_config):
        """Test that additional properties are allowed for indicator parameters."""
        valid_config["indicators"][0]["custom_param"] = 42
        validate_config(valid_config)  # Should not raise

    def test_duplicate_indicator_ids_fails(self, valid_config):
        """Test that duplicate indicator IDs fail validation."""
        # Add another indicator with same ID
        valid_config["indicators"].append(
            {
                "id": "RSI_14",  # Duplicate ID
                "type": "EMA",
                "timeframe": "15m",
                "length": 50,
            }
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

        assert "duplicate" in str(exc_info.value).lower()
        assert "RSI_14" in str(exc_info.value)


class TestBusinessLogicValidation:
    """Test business logic validation beyond JSON schema."""

    def test_end_date_before_start_date_fails(self, valid_config):
        """Test that end date before start date fails validation."""
        valid_config["backtest"]["from"] = "2024-06-30"
        valid_config["backtest"]["to"] = "2024-01-01"

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

        assert "after" in str(exc_info.value).lower()

    def test_undefined_timeframe_in_indicator_fails(self, valid_config):
        """Test that using undefined timeframe in indicator fails."""
        valid_config["indicators"][0]["timeframe"] = "1h"  # Not defined in timeframes

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

        assert "timeframe" in str(exc_info.value)
        assert "not defined" in str(exc_info.value)


class TestConfigLoader:
    """Test YAML configuration file loading."""

    def test_load_valid_config_file(self, valid_config, temp_yaml_file):
        """Test loading a valid YAML configuration file."""
        with open(temp_yaml_file, "w") as f:
            yaml.dump(valid_config, f)

        loaded_config = load_config(temp_yaml_file)

        assert loaded_config["name"] == "Test Strategy"
        assert loaded_config["symbol"] == "BTC/USDC"
        # Check type conversion happened
        assert isinstance(loaded_config["backtest"]["from"], date)
        assert isinstance(loaded_config["backtest"]["to"], date)

    def test_load_nonexistent_file_raises_filenotfound(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_load_empty_file_fails(self, temp_yaml_file):
        """Test that loading empty file fails with clear error."""
        # Create empty file
        with open(temp_yaml_file, "w") as f:
            f.write("")

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(temp_yaml_file)

        assert "empty" in str(exc_info.value).lower()

    def test_load_invalid_yaml_fails_with_line_number(self, temp_yaml_file):
        """Test that invalid YAML fails with line number information."""
        with open(temp_yaml_file, "w") as f:
            f.write("name: Test\n")
            f.write("invalid: [\n")  # Unclosed bracket - line 2
            f.write("more: data\n")

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(temp_yaml_file)

        error_msg = str(exc_info.value)
        assert "Line 2" in error_msg or "line 2" in error_msg.lower()
        assert temp_yaml_file in error_msg

    def test_load_non_dict_yaml_fails(self, temp_yaml_file):
        """Test that non-dictionary YAML fails validation."""
        with open(temp_yaml_file, "w") as f:
            f.write("- item1\n- item2\n")  # YAML list instead of dict

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(temp_yaml_file)

        assert "dictionary" in str(exc_info.value).lower()


class TestEnvironmentVariableSubstitution:
    """Test environment variable substitution functionality."""

    def test_env_var_substitution(self, valid_config, temp_yaml_file):
        """Test that environment variables are properly substituted."""
        # Set environment variable
        os.environ["TEST_SYMBOL"] = "ETH/USDC"

        try:
            # Create config with env var reference
            valid_config["symbol"] = "${TEST_SYMBOL}"

            with open(temp_yaml_file, "w") as f:
                yaml.dump(valid_config, f)

            loaded_config = load_config(temp_yaml_file)
            assert loaded_config["symbol"] == "ETH/USDC"

        finally:
            # Cleanup
            del os.environ["TEST_SYMBOL"]

    def test_missing_env_var_fails(self, valid_config, temp_yaml_file):
        """Test that missing environment variable causes failure."""
        # Make sure variable is not set
        if "MISSING_VAR" in os.environ:
            del os.environ["MISSING_VAR"]

        valid_config["symbol"] = "${MISSING_VAR}"

        with open(temp_yaml_file, "w") as f:
            yaml.dump(valid_config, f)

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(temp_yaml_file)

        assert "MISSING_VAR" in str(exc_info.value)
        assert "not set" in str(exc_info.value)

    def test_multiple_env_vars_in_string(self, valid_config, temp_yaml_file):
        """Test multiple environment variables in a single string."""
        os.environ["BASE_SYMBOL"] = "BTC"
        os.environ["QUOTE_SYMBOL"] = "USDC"

        try:
            valid_config["symbol"] = "${BASE_SYMBOL}/${QUOTE_SYMBOL}"

            with open(temp_yaml_file, "w") as f:
                yaml.dump(valid_config, f)

            loaded_config = load_config(temp_yaml_file)
            assert loaded_config["symbol"] == "BTC/USDC"

        finally:
            del os.environ["BASE_SYMBOL"]
            del os.environ["QUOTE_SYMBOL"]


class TestSchemaUtilities:
    """Test schema utility functions."""

    def test_get_schema_version(self):
        """Test getting schema version."""
        version = get_schema_version()
        assert isinstance(version, str)
        assert "." in version  # Should be in format like "1.0"

    def test_get_schema(self):
        """Test getting schema copy."""
        schema = get_schema()
        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "properties" in schema

        # Verify it's a copy (modifying it doesn't affect original)
        original_title = schema.get("title")
        schema["title"] = "Modified"
        new_schema = get_schema()
        assert new_schema["title"] == original_title


class TestErrorMessages:
    """Test that error messages are helpful and informative."""

    def test_validation_error_includes_path(self, valid_config):
        """Test that validation errors include the path to the invalid field."""
        valid_config["backtest"]["cash"] = -1000  # Invalid negative value

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

        error_msg = str(exc_info.value)
        assert "backtest" in error_msg
        assert "cash" in error_msg

    def test_file_path_included_in_error(self, temp_yaml_file):
        """Test that file path is included in error messages."""
        with open(temp_yaml_file, "w") as f:
            f.write("invalid_yaml: [unclosed")

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(temp_yaml_file)

        assert temp_yaml_file in str(exc_info.value)

    def test_multiple_validation_errors_listed(self, valid_config):
        """Test that multiple validation errors are all listed."""
        # Create multiple validation errors
        del valid_config["name"]  # Missing required field
        valid_config["symbol"] = "INVALID"  # Wrong format
        valid_config["backtest"]["cash"] = -1  # Invalid value

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(valid_config)

        error_msg = str(exc_info.value)
        # Should mention multiple issues - count any separator used in paths
        separators = error_msg.count("->") + error_msg.count(": ")
        assert separators >= 3  # Multiple validation errors


class TestRealConfigFiles:
    """Test loading actual configuration files from the project."""

    def test_load_demo_config(self):
        """Test loading the demo configuration file."""
        config_path = (
            Path(__file__).parent.parent / "btc_research" / "config" / "demo.yaml"
        )

        if config_path.exists():
            config = load_config(config_path)
            assert config["name"] == "EMA bias + RSI entry"
            assert config["symbol"] == "BTC/USDC"
            assert len(config["indicators"]) == 2

    def test_load_simple_rsi_config(self):
        """Test loading the simple RSI configuration file."""
        config_path = (
            Path(__file__).parent.parent / "btc_research" / "config" / "simple-rsi.yaml"
        )

        if config_path.exists():
            config = load_config(config_path)
            assert config["name"] == "Simple RSI Strategy"
            assert len(config["indicators"]) == 1
            assert config["indicators"][0]["type"] == "RSI"
