# core/utils/logging.py
"""
Centralized logging utilities.
"""

import logging
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()

        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def suppress_logger(name: str, level: int = logging.WARNING) -> None:
    """
    Suppress a logger to only show warnings and above.

    Args:
        name: Logger name
        level: Minimum level to show
    """
    logging.getLogger(name).setLevel(level)
