"""Central logging utilities for the LUAD Multi-Omics pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "luad",
    log_file: Optional[Union[str, Path]] = None,
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Parameters
    ----------
    name : str
        Logger name. Usually "luad" or a module-specific name.
    log_file : str | Path | None
        Optional path to log file. Parent directories are created if needed.
    level : str
        Logging level, e.g. "INFO", "DEBUG", "WARNING".
    console : bool
        Whether to add a console handler.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times.
    if getattr(logger, "_luad_configured", False):
        return logger

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    logger._luad_configured = True

    return logger