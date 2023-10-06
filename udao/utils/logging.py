"""Provides a logger object for the package"""
import logging


def _get_logger(name: str = "udao", level: int = logging.DEBUG) -> logging.Logger:
    """Get a logger object

    Args:
        name (str, optional): logger name. Defaults to "udao".
        level (int, optional): logging level (DEBUG, INFO...). Defaults to logging.DEBUG.
    Returns:
        logging.Logger: logger object
    """
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.WARNING)

    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    _logger.addHandler(handler)
    return _logger


logger = _get_logger()
