import logging

import pytest

from notes_rag.logging_config import setup_logging


class TestSetupLogging:
    """Test the setup_logging function."""

    def test_sets_notes_rag_level(self):
        logger = setup_logging("DEBUG")
        assert logger.level == logging.DEBUG
        assert logger.name == "notes_rag"

    def test_verbose_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        logger = setup_logging("DEBUG")
        assert logger.level == logging.DEBUG

    def test_default_level(self):
        logger = setup_logging("INFO")
        assert logger.level == logging.INFO

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging("INVALID")

    def test_format_is_correct(self):
        logger = setup_logging("DEBUG")
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.formatter._fmt == "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

    def test_only_has_one_handler(self):
        setup_logging("DEBUG")
        setup_logging("DEBUG")
        logger = logging.getLogger("notes_rag")
        assert len(logger.handlers) == 1

    def test_returns_the_same_logger(self):
        logger1 = setup_logging("DEBUG")
        logger2 = logging.getLogger("notes_rag")
        assert logger1 is logger2
