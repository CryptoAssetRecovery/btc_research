"""
Global plugin registry for BTC research engine.

This module provides a thread-safe registry system for indicator plugins using
the decorator pattern. It allows indicators to be registered once and retrieved
anywhere in the codebase.
"""

import threading
from collections.abc import Callable
from typing import TypeVar

__all__ = ["register", "get", "RegistrationError"]

# Type variable for generic class types
T = TypeVar("T")

# Global registry storage
_registry: dict[str, type] = {}
_registry_lock = threading.RLock()


class RegistrationError(Exception):
    """Raised when there are issues with plugin registration or retrieval."""

    pass


def register(name: str) -> Callable[[type[T]], type[T]]:
    """
    Decorator function that registers a class in the global plugin registry.

    This decorator allows indicators to be registered by name and retrieved later.
    The registration is thread-safe and maintains a global mapping of names to classes.

    Args:
        name: The unique name to register the class under. Must be a non-empty string.

    Returns:
        A decorator function that registers the class and returns it unchanged.

    Raises:
        RegistrationError: If the name is empty, not a string, or already registered.

    Example:
        >>> @register("RSI")
        ... class RSIIndicator:
        ...     pass
        >>>
        >>> # Later retrieve the class
        >>> rsi_class = get("RSI")
    """
    if not isinstance(name, str):
        raise RegistrationError(
            f"Registration name must be a string, got {type(name).__name__}"
        )

    if not name.strip():
        raise RegistrationError("Registration name cannot be empty")

    def _decorator(cls: type[T]) -> type[T]:
        with _registry_lock:
            if name in _registry:
                existing_cls = _registry[name]
                if existing_cls is not cls:
                    raise RegistrationError(
                        f"Name '{name}' is already registered to {existing_cls.__module__}.{existing_cls.__name__}"
                    )
                # Allow re-registration of the same class (useful for reloading modules)
                return cls

            _registry[name] = cls
            return cls

    return _decorator


def get(name: str) -> type:
    """
    Retrieve a registered class by name.

    Args:
        name: The name the class was registered under.

    Returns:
        The registered class.

    Raises:
        RegistrationError: If no class is registered under the given name.

    Example:
        >>> rsi_class = get("RSI")
        >>> indicator = rsi_class(length=14)
    """
    if not isinstance(name, str):
        raise RegistrationError(
            f"Registry lookup name must be a string, got {type(name).__name__}"
        )

    with _registry_lock:
        if name not in _registry:
            available = list(_registry.keys())
            raise RegistrationError(
                f"No class registered under name '{name}'. "
                f"Available registrations: {available}"
            )
        return _registry[name]


def _clear_registry() -> None:
    """
    Clear all registrations from the registry.

    This function is primarily intended for testing purposes to ensure
    a clean state between test runs.
    """
    with _registry_lock:
        _registry.clear()


def list_registered() -> dict[str, str]:
    """
    Get a mapping of all registered names to their class representations.

    Returns:
        A dictionary mapping registration names to string representations
        of their registered classes.
    """
    with _registry_lock:
        return {
            name: f"{cls.__module__}.{cls.__name__}" for name, cls in _registry.items()
        }
