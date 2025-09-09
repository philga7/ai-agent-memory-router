"""
Unit tests for logging utility
Adapted from test-automation-harness patterns
"""

import logging
import pytest
from unittest.mock import patch, MagicMock
from app.core.logging import get_logger, setup_logging


class TestLogging:
    """Test logging functionality."""
    
    def test_logger_initialization(self):
        """Test that logger is properly initialized."""
        logger = get_logger("test_module")
        
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'critical')
    
    def test_logger_methods(self):
        """Test that all required logging methods exist."""
        logger = get_logger("test_module")
        
        # Check all standard logging methods
        assert callable(logger.info)
        assert callable(logger.error)
        assert callable(logger.warning)
        assert callable(logger.debug)
        assert callable(logger.critical)
    
    def test_logger_calls_without_errors(self):
        """Test that logging methods can be called without throwing errors."""
        logger = get_logger("test_module")
        
        # Should not raise any exceptions
        logger.info("Test info message")
        logger.error("Test error message")
        logger.warning("Test warning message")
        logger.debug("Test debug message")
        logger.critical("Test critical message")
    
    def test_logger_with_additional_arguments(self):
        """Test logger handles additional arguments correctly."""
        logger = get_logger("test_module")
        
        # Test with extra arguments
        logger.info("Test message", extra={"key": "value"})
        logger.error("Test error", exc_info=True)
        logger.warning("Test warning", stack_info=True)
    
    def test_logger_with_exception(self):
        """Test logger handles exceptions correctly."""
        logger = get_logger("test_module")
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            # Should not raise any exceptions
            logger.exception("Test exception logging")
    
    def test_setup_logging_configuration(self):
        """Test logging setup configuration."""
        # Test that setup_logging can be called without errors
        # The actual implementation uses custom handlers, not basicConfig
        setup_logging()
        
        # Verify that logging is working
        logger = get_logger("test_setup")
        logger.info("Test setup logging")
    
    def test_logger_name_formatting(self):
        """Test that logger names are formatted correctly."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("another_module")
        
        # The actual implementation returns the name as provided
        assert logger1.name == "test_module"
        assert logger2.name == "another_module"
    
    def test_logger_level_handling(self):
        """Test logger handles different log levels gracefully."""
        logger = get_logger("test_module")
        
        # Test all log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
    
    def test_logger_with_structured_data(self):
        """Test logger with structured data."""
        logger = get_logger("test_module")
        
        structured_data = {
            "user_id": "123",
            "action": "test",
            "metadata": {"key": "value"}
        }
        
        # Should handle structured data without errors
        logger.info("Structured log message", extra=structured_data)
    
    def test_logger_performance(self):
        """Test logger performance with multiple calls."""
        logger = get_logger("test_module")
        
        # Test multiple rapid logging calls
        for i in range(100):
            logger.info(f"Performance test message {i}")
    
    def test_logger_thread_safety(self):
        """Test logger thread safety."""
        import threading
        import time
        
        logger = get_logger("test_module")
        results = []
        
        def log_worker(worker_id):
            for i in range(10):
                logger.info(f"Worker {worker_id} message {i}")
                results.append(f"worker_{worker_id}_{i}")
                time.sleep(0.001)  # Small delay to test concurrency
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all threads completed
        assert len(results) == 50  # 5 workers * 10 messages each
