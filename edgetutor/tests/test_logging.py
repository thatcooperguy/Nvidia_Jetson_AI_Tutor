"""Tests for edgetutor.core.logging_config — logging setup."""

import logging


class TestLoggingConfig:
    """Test logging configuration."""

    def test_get_logger_returns_named_logger(self):
        """get_logger() should return a logger with the given name."""
        from edgetutor.core.logging_config import get_logger

        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_different_names(self):
        """Different names should return different loggers."""
        from edgetutor.core.logging_config import get_logger

        logger_a = get_logger("module.a")
        logger_b = get_logger("module.b")
        assert logger_a is not logger_b
        assert logger_a.name != logger_b.name

    def test_setup_logging_is_idempotent(self):
        """Calling setup_logging() multiple times should not duplicate handlers."""
        from edgetutor.core.logging_config import setup_logging

        # setup_logging is guarded by the _CONFIGURED flag
        # Calling it twice should not add extra handlers to the root logger
        setup_logging()
        root = logging.getLogger()
        handler_count_before = len(root.handlers)

        setup_logging()
        handler_count_after = len(root.handlers)

        assert handler_count_after == handler_count_before

    def test_root_logger_has_handlers(self):
        """After setup, root logger should have console + file handlers."""
        from edgetutor.core.logging_config import setup_logging

        setup_logging()
        root = logging.getLogger()
        assert len(root.handlers) >= 1  # At minimum the console handler
