"""Logging configuration with console and rotating UTF-8 file handlers."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """Configure structured logging to console and rotating file."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / "bot.log"

    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Rotating file handler (5 MB x 5 backups, UTF-8 encoded)
    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any pre-existing handlers
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress verbose external loggers
    logging.getLogger("aiogram").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
