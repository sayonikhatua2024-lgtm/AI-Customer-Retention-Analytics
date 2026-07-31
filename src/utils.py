"""
utils.py
========
Small shared utilities: logging setup and a friendly file-existence
guard so every module fails with a clear, actionable message instead
of a raw stack trace (important for a first-run / Streamlit Cloud
deployment where a data file might genuinely be missing).
"""

from __future__ import annotations

import logging
from pathlib import Path

from src import config


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger. Safe to call repeatedly (e.g. once per
    module import) — handlers are only attached once per logger name.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(config.LOG_LEVEL)
        logger.propagate = False

    return logger


class MissingAssetError(FileNotFoundError):
    """Raised when a required data/model asset is not present on disk."""


def require_file(path: Path, friendly_name: str, how_to_fix: str) -> Path:
    """
    Verify a required file exists, raising a clear, user-actionable
    error if it does not. Used by both the training pipeline and the
    Streamlit dashboard so missing assets never surface as a bare
    ``FileNotFoundError`` or an unhandled Streamlit exception.

    Parameters
    ----------
    path : Path
        Expected location of the file.
    friendly_name : str
        Human-readable name of the asset, used in the error message.
    how_to_fix : str
        Short instruction shown to the user on how to resolve it.

    Returns
    -------
    Path
        The same path, returned for convenient chaining.

    Raises
    ------
    MissingAssetError
        If the file does not exist.
    """
    if not path.exists():
        raise MissingAssetError(
            f"Required asset '{friendly_name}' was not found at "
            f"'{path}'.\n➡ {how_to_fix}"
        )
    return path
