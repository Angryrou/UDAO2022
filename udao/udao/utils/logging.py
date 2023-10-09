"""Provides a logger object for the package"""
import logging


def _get_logger(name: str = "udao", level: int = logging.DEBUG) -> logging.Logger:
    """Generates a logger object for the UDAO library.

    Parameters
    ----------
    name : str, optional
        logger name, by default "udao".
    level : int, optional
        logging level (DEBUG, INFO...), by default logging.DEBUG

    Returns
    -------
    logging.Logger
        logger object to call for logging
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