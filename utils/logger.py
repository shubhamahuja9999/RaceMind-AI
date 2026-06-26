"""Project-wide logging setup for RaceMind AI.

Provides :func:`get_logger`, the single entry point every module should use to
obtain a logger. Logging is configured once, lazily, on first use: messages go
to both the console and ``logs/project.log`` (created automatically).

Example output::

    2026-06-25 18:41:22 INFO Environment initialized
"""

from __future__ import annotations

import logging
from typing import Optional

from config.logging import LoggingConfig
from utils.paths import ensure_directory

# Root name for the project's logger hierarchy. All module loggers are children
# of this logger, which carries the handlers.
PROJECT_LOGGER_NAME: str = "racemind"


def configure_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """Configure and return the project's root logger.

    Idempotent: if the project logger already has handlers it is returned
    unchanged, so repeated calls are safe and cheap.

    Args:
        config: Logging configuration; the default is used when omitted.

    Returns:
        The configured project logger.
    """
    logger = logging.getLogger(PROJECT_LOGGER_NAME)
    if logger.handlers:
        return logger

    config = config or LoggingConfig()
    logger.setLevel(config.level)
    logger.propagate = False

    formatter = logging.Formatter(config.log_format, datefmt=config.date_format)

    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    ensure_directory(config.log_dir)
    file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> logging.Logger:
    """Return a module logger, configuring project logging on first use.

    Args:
        name: Logical name for the logger, typically the module ``__name__``.
        config: Optional logging configuration applied on first configuration.

    Returns:
        A child logger of the project logger.
    """
    configure_logging(config)
    return logging.getLogger(f"{PROJECT_LOGGER_NAME}.{name}")
