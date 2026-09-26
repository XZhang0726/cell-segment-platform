"""
Logging configuration.

Provides consistent logging through loguru.
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_file: str = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_string: str = None
):
    """
    Configure logging.

    Args:
        log_file: Log file path; None enables console output only.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        rotation: Log rotation size.
        retention: Log retention period.
        format_string: Custom log format.
    """
    # Remove the default handler.
    logger.remove()

    # Default log format.
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console output.
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True
    )

    # Add file output when a log file is specified.
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )

    return logger


def get_logger(name: str = None):
    """
    Get a logger instance.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    if name:
        return logger.bind(name=name)
    return logger
