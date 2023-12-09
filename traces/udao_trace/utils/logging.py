"""Provides a logger object for the udao library."""
import logging
import sys


def _get_logger(name: str = "udao_trace", level: int = logging.DEBUG, log_file_path: str = None) -> logging.Logger:
    """Generates a logger object for the UDAO library.

    Parameters
    ----------
    name : str, optional
        logger name, by default "udao_trace".
    level : int, optional
        logging level (DEBUG, INFO...), by default logging.DEBUG

    Returns
    -------
    logging.Logger
        logger object to call for logging
    """
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        log_format = (
            "%(asctime)s - [%(levelname)s] - %(name)s - "
            "(%(filename)s:%(lineno)d) - "
            "%(message)s"
        )
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

        _logger.addHandler(handler)

    # FileHandler to log to a file if log_file_path is provided
    if log_file_path and not any(isinstance(handler, logging.FileHandler) for handler in _logger.handlers):
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    return _logger


logger = _get_logger(log_file_path="spark_collect.log")
