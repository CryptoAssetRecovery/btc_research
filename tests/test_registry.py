"""
Comprehensive unit tests for the plugin registry system.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from btc_research.core.registry import (
    RegistrationError,
    _clear_registry,
    get,
    list_registered,
    register,
)


class TestRegistry:
    """Test suite for the plugin registry system."""

    def setup_method(self):
        """Clear registry before each test."""
        _clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        _clear_registry()

    def test_basic_registration_and_retrieval(self):
        """Test basic registration and retrieval of a class."""

        @register("TestIndicator")
        class TestIndicator:
            pass

        # Verify the class is registered
        retrieved_class = get("TestIndicator")
        assert retrieved_class is TestIndicator

        # Verify we can instantiate it
        instance = retrieved_class()
        assert isinstance(instance, TestIndicator)

    def test_decorator_pattern_usage(self):
        """Test that the decorator pattern works as shown in ROADMAP.md example."""

        @register("RSI")
        class RSI:
            def __init__(self, length=14):
                self.length = length

        # Test retrieval and instantiation
        rsi_class = get("RSI")
        rsi_instance = rsi_class(length=21)
        assert rsi_instance.length == 21

        # Test default parameter
        rsi_default = rsi_class()
        assert rsi_default.length == 14

    def test_multiple_registrations(self):
        """Test registering multiple different classes."""

        @register("EMA")
        class EMA:
            pass

        @register("RSI")
        class RSI:
            pass

        @register("MACD")
        class MACD:
            pass

        # Verify all are retrievable
        assert get("EMA") is EMA
        assert get("RSI") is RSI
        assert get("MACD") is MACD

    def test_missing_registration_error(self):
        """Test error handling for missing registrations."""
        with pytest.raises(RegistrationError) as exc_info:
            get("NonExistentIndicator")

        assert "No class registered under name 'NonExistentIndicator'" in str(
            exc_info.value
        )
        assert "Available registrations: []" in str(exc_info.value)

    def test_missing_registration_with_suggestions(self):
        """Test error message includes available registrations when lookup fails."""

        @register("RSI")
        class RSI:
            pass

        @register("EMA")
        class EMA:
            pass

        with pytest.raises(RegistrationError) as exc_info:
            get("MACD")

        error_msg = str(exc_info.value)
        assert "No class registered under name 'MACD'" in error_msg
        assert "RSI" in error_msg
        assert "EMA" in error_msg

    def test_duplicate_registration_error(self):
        """Test error handling for duplicate registrations of different classes."""

        @register("TestName")
        class FirstClass:
            pass

        # Attempting to register a different class with the same name should fail
        with pytest.raises(RegistrationError) as exc_info:

            @register("TestName")
            class SecondClass:
                pass

        assert "already registered" in str(exc_info.value)

    def test_same_class_reregistration_allowed(self):
        """Test that re-registering the same class is allowed."""

        @register("TestClass")
        class TestClass:
            pass

        # Store reference to the same class object
        original_class = TestClass

        # Re-registering the same class object should work (useful for module reloading)
        register("TestClass")(TestClass)

        # Should still work and return the same class
        retrieved = get("TestClass")
        assert retrieved is TestClass
        assert retrieved is original_class

    def test_invalid_registration_name_types(self):
        """Test error handling for invalid registration name types."""
        with pytest.raises(RegistrationError) as exc_info:

            @register(123)
            class TestClass:
                pass

        assert "Registration name must be a string" in str(exc_info.value)

        with pytest.raises(RegistrationError):

            @register(None)
            class TestClass2:
                pass

    def test_empty_registration_name(self):
        """Test error handling for empty registration names."""
        with pytest.raises(RegistrationError) as exc_info:

            @register("")
            class TestClass:
                pass

        assert "Registration name cannot be empty" in str(exc_info.value)

        with pytest.raises(RegistrationError):

            @register("   ")  # Only whitespace
            class TestClass2:
                pass

    def test_invalid_get_name_types(self):
        """Test error handling for invalid get() name types."""
        with pytest.raises(RegistrationError) as exc_info:
            get(123)

        assert "Registry lookup name must be a string" in str(exc_info.value)

    def test_list_registered(self):
        """Test the list_registered() function."""
        # Empty registry
        assert list_registered() == {}

        @register("TestA")
        class TestA:
            pass

        @register("TestB")
        class TestB:
            pass

        registered = list_registered()
        assert len(registered) == 2
        assert "TestA" in registered
        assert "TestB" in registered
        assert (
            "TestA" in registered["TestA"]
        )  # Class name should be in the string representation
        assert "TestB" in registered["TestB"]

    def test_thread_safety_basic(self):
        """Test basic thread safety with concurrent registrations."""
        results = []
        errors = []

        def register_class(name_suffix: int):
            try:

                @register(f"ThreadTest{name_suffix}")
                class ThreadTestClass:
                    id = name_suffix

                # Try to retrieve it immediately
                retrieved = get(f"ThreadTest{name_suffix}")
                results.append((name_suffix, retrieved.id))
            except Exception as e:
                errors.append(e)

        # Run multiple registrations concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_class, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()  # Will raise any exceptions

        # Check results
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert len(results) == 20

        # Verify all classes are still accessible
        for i in range(20):
            cls = get(f"ThreadTest{i}")
            assert cls.id == i

    def test_thread_safety_duplicate_prevention(self):
        """Test that duplicate registration protection works under concurrent access."""
        errors = []
        success_count = 0

        def try_register_same_name():
            nonlocal success_count
            try:

                @register("ConcurrentTest")
                class TestClass:
                    thread_id = threading.current_thread().ident

                success_count += 1
            except RegistrationError as e:
                errors.append(e)

        # Try to register the same name from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(try_register_same_name) for _ in range(10)]
            for future in as_completed(futures):
                future.result()

        # Only one should succeed, others should get RegistrationError
        assert success_count == 1
        assert len(errors) == 9

        # The registered class should be retrievable
        cls = get("ConcurrentTest")
        assert hasattr(cls, "thread_id")

    def test_integration_with_indicator_pattern(self):
        """Test integration with the indicator pattern from ROADMAP.md."""

        # Simulate the pattern shown in the roadmap
        @register("RSI")
        class RSI:
            @classmethod
            def params(cls):
                return {"length": 14}

            def __init__(self, length=14):
                self.length = length

            def compute(self, df):
                # Simplified compute method
                return f"RSI computed with length {self.length}"

        # Test that we can retrieve and use it as intended
        rsi_class = get("RSI")

        # Test default parameters
        default_params = rsi_class.params()
        assert default_params == {"length": 14}

        # Test instantiation with defaults
        rsi_default = rsi_class()
        assert rsi_default.length == 14
        assert rsi_default.compute(None) == "RSI computed with length 14"

        # Test instantiation with custom parameters
        rsi_custom = rsi_class(length=21)
        assert rsi_custom.length == 21
        assert rsi_custom.compute(None) == "RSI computed with length 21"

    def test_class_decorator_returns_original_class(self):
        """Test that the decorator returns the original class unchanged."""
        original_class_id = id(object)  # Get a unique ID to track the class

        @register("ReturnTest")
        class TestClass:
            class_id = original_class_id

        # The decorator should return the exact same class object
        retrieved = get("ReturnTest")
        assert retrieved is TestClass
        assert retrieved.class_id == original_class_id

    def test_registry_persistence_across_lookups(self):
        """Test that registrations persist across multiple lookups."""

        @register("PersistenceTest")
        class TestClass:
            value = 42

        # Multiple lookups should return the same class
        first_lookup = get("PersistenceTest")
        second_lookup = get("PersistenceTest")
        third_lookup = get("PersistenceTest")

        assert first_lookup is second_lookup is third_lookup
        assert all(
            cls.value == 42 for cls in [first_lookup, second_lookup, third_lookup]
        )
