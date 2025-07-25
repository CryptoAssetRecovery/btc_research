"""
Configuration management endpoints.
Handles strategy configuration listing and validation.
"""

import logging
import os
import yaml
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from slowapi import Limiter

from ..models import (
    ConfigListResponse, ConfigValidationRequest, ConfigValidationResponse,
    BaseResponse
)
from ..dependencies import RequiredAuth, get_limiter, Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/configs", tags=["configuration"])

# Cache for configuration data with timestamp-based invalidation
_config_cache: Optional[Dict] = None
_cache_timestamp: float = 0
_cache_duration: int = 5  # Cache for 5 seconds to reduce filesystem overhead


def _scan_config_directory(config_dir: Path) -> List[Dict]:
    """
    Scan configuration directory and return config information.
    
    This function is separated to enable caching and reduce filesystem overhead.
    """
    configs = []
    
    if not config_dir.exists():
        logger.warning(f"Configuration directory does not exist: {config_dir}")
        return configs
    
    # Scan for YAML configuration files
    for config_file in config_dir.glob("*.yaml"):
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Extract basic information from the config
            config_info = {
                "name": config_file.stem,
                "path": str(config_file),
                "description": config_data.get("description", "No description available"),
                "strategy_type": config_data.get("strategy", {}).get("name", "Unknown"),
                "symbols": config_data.get("data", {}).get("symbols", []),
                "timeframes": config_data.get("data", {}).get("timeframes", []),
                "parameters": config_data.get("strategy", {}).get("parameters", {}),
                "last_modified": config_file.stat().st_mtime
            }
            
            configs.append(config_info)
            
        except Exception as e:
            logger.warning(f"Failed to parse config file {config_file}: {e}")
            continue
    
    # Sort by last modified (newest first)
    configs.sort(key=lambda x: x["last_modified"], reverse=True)
    
    return configs


def _get_directory_last_modified(config_dir: Path) -> float:
    """Get the last modification time of the configuration directory."""
    try:
        if not config_dir.exists():
            return 0
        
        # Get directory modification time and all yaml files modification times
        dir_mtime = config_dir.stat().st_mtime
        file_mtimes = [config_file.stat().st_mtime for config_file in config_dir.glob("*.yaml")]
        
        # Return the most recent modification time
        return max([dir_mtime] + file_mtimes) if file_mtimes else dir_mtime
    except Exception:
        return 0


@router.get("/available", response_model=ConfigListResponse)
async def list_available_configs(
    request: Request,
    current_user: RequiredAuth,
    settings: Settings,
):
    """
    List all available strategy configuration files.
    
    Scans the configuration directory and returns information about
    available strategy configurations. Uses intelligent caching with
    filesystem-based cache invalidation for hot reloading.
    """
    # Apply rate limiting
    
    try:
        global _config_cache, _cache_timestamp
        
        config_dir = Path(settings.config_dir)
        current_time = time.time()
        dir_last_modified = _get_directory_last_modified(config_dir)
        
        # Check if cache is valid (not expired and directory hasn't changed)
        cache_valid = (
            _config_cache is not None and
            current_time - _cache_timestamp < _cache_duration and
            dir_last_modified <= _cache_timestamp
        )
        
        if not cache_valid:
            # Cache is invalid, rescan directory
            logger.debug(f"Rescanning configuration directory: {config_dir}")
            configs = _scan_config_directory(config_dir)
            
            # Update cache
            _config_cache = configs
            _cache_timestamp = current_time
            
            logger.info(f"Found {len(configs)} configuration files (cache updated)")
        else:
            # Use cached data
            configs = _config_cache
            logger.debug(f"Using cached configuration data ({len(configs)} files)")
        
        return ConfigListResponse(
            message="Configuration files retrieved successfully",
            configs=configs,
            total=len(configs)
        )
        
    except Exception as e:
        logger.error(f"Failed to list configuration files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list configuration files: {str(e)}"
        )


@router.post("/refresh", response_model=BaseResponse)
async def refresh_config_cache(
    request: Request,
    current_user: RequiredAuth,
    settings: Settings,
):
    """
    Manually refresh the configuration cache.
    
    Forces a rescan of the configuration directory and clears the cache.
    Useful for immediate updates when configurations are added, modified, or deleted.
    """
    try:
        global _config_cache, _cache_timestamp
        
        config_dir = Path(settings.config_dir)
        
        # Force rescan by clearing cache
        _config_cache = None
        _cache_timestamp = 0
        
        # Rescan directory
        configs = _scan_config_directory(config_dir)
        
        # Update cache
        _config_cache = configs
        _cache_timestamp = time.time()
        
        logger.info(f"Configuration cache refreshed: {len(configs)} files found")
        
        return BaseResponse(
            message=f"Configuration cache refreshed successfully. Found {len(configs)} configuration files.",
            data={"total_configs": len(configs), "refreshed_at": time.time()}
        )
        
    except Exception as e:
        logger.error(f"Failed to refresh configuration cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh configuration cache: {str(e)}"
        )


@router.post("/validate", response_model=ConfigValidationResponse)
async def validate_config(
    request: Request,
    validation_request: ConfigValidationRequest,
    current_user: RequiredAuth,
):
    """
    Validate a strategy configuration.
    
    Checks the provided configuration for required fields, valid parameters,
    and potential issues.
    """
    # Apply rate limiting
    
    try:
        config = validation_request.config
        errors = []
        warnings = []
        
        # Validate required top-level sections
        required_sections = ["data", "strategy", "backtester"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate data section
        if "data" in config:
            data_config = config["data"]
            
            # Check symbols
            if "symbols" not in data_config:
                errors.append("Missing 'symbols' in data section")
            elif not isinstance(data_config["symbols"], list) or not data_config["symbols"]:
                errors.append("'symbols' must be a non-empty list")
            
            # Check timeframes
            if "timeframes" not in data_config:
                errors.append("Missing 'timeframes' in data section")
            elif not isinstance(data_config["timeframes"], list) or not data_config["timeframes"]:
                errors.append("'timeframes' must be a non-empty list")
            else:
                valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                invalid_timeframes = [tf for tf in data_config["timeframes"] if tf not in valid_timeframes]
                if invalid_timeframes:
                    errors.append(f"Invalid timeframes: {invalid_timeframes}")
        
        # Validate strategy section
        if "strategy" in config:
            strategy_config = config["strategy"]
            
            # Check strategy name
            if "name" not in strategy_config:
                errors.append("Missing 'name' in strategy section")
            
            # Check parameters
            if "parameters" not in strategy_config:
                warnings.append("No parameters specified in strategy section")
            elif not isinstance(strategy_config["parameters"], dict):
                errors.append("'parameters' must be a dictionary")
        
        # Validate backtester section
        if "backtester" in config:
            backtester_config = config["backtester"]
            
            # Check initial capital
            if "initial_capital" in backtester_config:
                try:
                    initial_capital = float(backtester_config["initial_capital"])
                    if initial_capital <= 0:
                        errors.append("'initial_capital' must be positive")
                except (ValueError, TypeError):
                    errors.append("'initial_capital' must be a number")
            else:
                warnings.append("No 'initial_capital' specified, will use default")
            
            # Check commission
            if "commission" in backtester_config:
                try:
                    commission = float(backtester_config["commission"])
                    if commission < 0 or commission > 0.1:
                        warnings.append("Commission seems unusually high (>10%)")
                except (ValueError, TypeError):
                    errors.append("'commission' must be a number")
        
        # Additional validation checks
        
        # Check for conflicting parameters
        if "strategy" in config and "parameters" in config["strategy"]:
            params = config["strategy"]["parameters"]
            
            # Example: Check for reasonable parameter ranges
            if "volume_profile" in params:
                vp_params = params["volume_profile"]
                if "lookback_period" in vp_params:
                    try:
                        lookback = int(vp_params["lookback_period"])
                        if lookback < 10 or lookback > 1000:
                            warnings.append("Lookback period outside typical range (10-1000)")
                    except (ValueError, TypeError):
                        errors.append("'lookback_period' must be an integer")
        
        # Determine if configuration is valid
        is_valid = len(errors) == 0
        
        logger.info(f"Configuration validation completed: {len(errors)} errors, {len(warnings)} warnings")
        
        return ConfigValidationResponse(
            message="Configuration validation completed",
            valid=is_valid,
            errors=errors,
            warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate configuration: {str(e)}"
        )


@router.get("/{config_name}", response_model=BaseResponse)
async def get_config_details(
    request: Request,
    config_name: str,
    current_user: RequiredAuth,
    settings: Settings,
):
    """
    Get detailed information about a specific configuration file.
    
    Returns the full configuration content and metadata.
    """
    # Apply rate limiting
    
    try:
        config_dir = Path(settings.config_dir)
        config_file = config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Configuration file not found: {config_name}"
            )
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Get file metadata
        file_stat = config_file.stat()
        
        result = {
            "name": config_name,
            "path": str(config_file),
            "size": file_stat.st_size,
            "last_modified": file_stat.st_mtime,
            "content": config_data
        }
        
        logger.info(f"Retrieved configuration details for {config_name}")
        
        return BaseResponse(
            message="Configuration details retrieved successfully",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get configuration details for {config_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration details: {str(e)}"
        )