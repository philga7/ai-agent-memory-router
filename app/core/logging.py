"""
Logging configuration for AI Agent Memory Router.

This module provides structured logging configuration with JSON formatting,
file rotation, and centralized log management.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

from app.core.config import get_settings

# Get settings
settings = get_settings()


def setup_logging() -> None:
    """Setup application logging configuration."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.logging.file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = create_console_handler()
    root_logger.addHandler(console_handler)
    
    # Add file handler
    file_handler = create_file_handler()
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    set_logger_levels()
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized", extra={
        "log_level": settings.log_level,
        "log_file": settings.logging.file_path,
        "environment": settings.environment
    })


def create_console_handler() -> logging.Handler:
    """Create console logging handler."""
    
    if settings.logging.format == "json":
        formatter = create_json_formatter()
    else:
        formatter = create_text_formatter()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))
    console_handler.setFormatter(formatter)
    
    return console_handler


def create_file_handler() -> logging.Handler:
    """Create file logging handler with rotation."""
    
    if settings.logging.format == "json":
        formatter = create_json_formatter()
    else:
        formatter = create_text_formatter()
    
    # Parse max size (e.g., "100MB" -> 100 * 1024 * 1024)
    max_size = parse_size_string(settings.logging.max_size)
    
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.logging.file_path,
        maxBytes=max_size,
        backupCount=settings.logging.backup_count,
        encoding='utf-8'
    )
    
    file_handler.setLevel(getattr(logging, settings.log_level))
    file_handler.setFormatter(formatter)
    
    return file_handler


def create_json_formatter() -> logging.Formatter:
    """Create JSON formatter for structured logging."""
    
    class JSONFormatter(logging.Formatter):
        """Custom JSON formatter for structured logging."""
        
        def format(self, record):
            """Format log record as JSON."""
            log_entry = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add extra fields if present
            if hasattr(record, 'extra') and record.extra:
                log_entry.update(record.extra)
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            
            # Add process and thread info
            log_entry["process"] = record.process
            log_entry["thread"] = record.thread
            
            return str(log_entry)
    
    return JSONFormatter()


def create_text_formatter() -> logging.Formatter:
    """Create text formatter for human-readable logging."""
    
    format_string = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(module)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    
    return logging.Formatter(format_string)


def parse_size_string(size_str: str) -> int:
    """Parse size string to bytes."""
    
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # Assume bytes if no suffix
        return int(size_str)


def set_logger_levels() -> None:
    """Set specific logger levels for different components."""
    
    # Set lower levels for verbose components
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    
    # Set higher levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


def log_function_call(func_name: str, **kwargs):
    """Decorator to log function calls with parameters."""
    
    def decorator(func):
        async def async_wrapper(*args, **func_kwargs):
            logger = logging.getLogger(func.__module__)
            logger.debug(f"Calling {func_name}", extra={
                "function": func_name,
                "args": args,
                "kwargs": func_kwargs,
                "call_type": "async"
            })
            
            try:
                result = await func(*args, **func_kwargs)
                logger.debug(f"Completed {func_name}", extra={
                    "function": func_name,
                    "success": True,
                    "call_type": "async"
                })
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}", extra={
                    "function": func_name,
                    "error": str(e),
                    "call_type": "async"
                })
                raise
        
        def sync_wrapper(*args, **func_kwargs):
            logger = logging.getLogger(func.__module__)
            logger.debug(f"Calling {func_name}", extra={
                "function": func_name,
                "args": args,
                "kwargs": func_kwargs,
                "call_type": "sync"
            })
            
            try:
                result = func(*args, **func_kwargs)
                logger.debug(f"Completed {func_name}", extra={
                    "function": func_name,
                    "success": True,
                    "call_type": "sync"
                })
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}", extra={
                    "function": func_name,
                    "error": str(e),
                    "call_type": "sync"
                })
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_performance(func_name: str, threshold_ms: float = 1000.0):
    """Decorator to log function performance metrics."""
    
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                logger = logging.getLogger(func.__module__)
                if execution_time > threshold_ms:
                    logger.warning(f"Slow execution: {func_name} took {execution_time:.2f}ms", extra={
                        "function": func_name,
                        "execution_time_ms": execution_time,
                        "threshold_ms": threshold_ms,
                        "call_type": "async"
                    })
                else:
                    logger.debug(f"Performance: {func_name} took {execution_time:.2f}ms", extra={
                        "function": func_name,
                        "execution_time_ms": execution_time,
                        "call_type": "async"
                    })
                
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(f"Error in {func_name} after {execution_time:.2f}ms: {e}", extra={
                    "function": func_name,
                    "execution_time_ms": execution_time,
                    "error": str(e),
                    "call_type": "async"
                })
                raise
        
        def sync_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                logger = logging.getLogger(func.__module__)
                if execution_time > threshold_ms:
                    logger.warning(f"Slow execution: {func_name} took {execution_time:.2f}ms", extra={
                        "function": func_name,
                        "execution_time_ms": execution_time,
                        "threshold_ms": threshold_ms,
                        "call_type": "sync"
                    })
                else:
                    logger.debug(f"Performance: {func_name} took {execution_time:.2f}ms", extra={
                        "function": func_name,
                        "execution_time_ms": execution_time,
                        "call_type": "sync"
                    })
                
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(f"Error in {func_name} after {execution_time:.2f}ms: {e}", extra={
                    "function": func_name,
                    "execution_time_ms": execution_time,
                    "error": str(e),
                    "call_type": "sync"
                })
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Import asyncio for the decorators
import asyncio
