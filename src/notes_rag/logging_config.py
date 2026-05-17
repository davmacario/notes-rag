import logging


def setup_logging(level: str) -> logging.Logger:
    """Configure logging for the notes_rag namespace.

    Sets the notes_rag logger to the specified level with a formatted
    StreamHandler. Third-party loggers are left untouched.

    Args:
        level: Logging level string (e.g., 'DEBUG', 'INFO', 'WARNING').

    Returns:
        The configured notes_rag root logger.

    Raises:
        ValueError: If level is not a valid logging level.
    """
    log_level = logging.getLevelNamesMapping().get(level.upper())
    if log_level is None:
        raise ValueError(f"Invalid log level: {level!r}. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")

    logger = logging.getLogger(__package__)
    logger.setLevel(log_level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
