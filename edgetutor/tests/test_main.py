"""Tests for edgetutor.__main__ — application entry point."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestMain:
    """Test the main() entry point."""

    def test_main_calls_launch_ui(self):
        """main() should call launch_ui()."""
        mock_ui_module = MagicMock()
        mock_ui_module.launch_ui = MagicMock()

        with patch.dict(sys.modules, {"edgetutor.app.ui": mock_ui_module, "gradio": MagicMock()}):
            from edgetutor.__main__ import main

            main()
            mock_ui_module.launch_ui.assert_called_once()

    def test_main_handles_keyboard_interrupt(self):
        """main() should exit cleanly on KeyboardInterrupt."""
        mock_ui_module = MagicMock()
        mock_ui_module.launch_ui = MagicMock(side_effect=KeyboardInterrupt)

        with (
            patch.dict(sys.modules, {"edgetutor.app.ui": mock_ui_module, "gradio": MagicMock()}),
            pytest.raises(SystemExit) as exc_info,
        ):
            if "edgetutor.__main__" in sys.modules:
                importlib.reload(sys.modules["edgetutor.__main__"])
            from edgetutor.__main__ import main

            main()

        assert exc_info.value.code == 0

    def test_main_handles_runtime_error(self):
        """main() should exit with code 1 on unexpected error."""
        mock_ui_module = MagicMock()
        mock_ui_module.launch_ui = MagicMock(side_effect=RuntimeError("boom"))

        with (
            patch.dict(sys.modules, {"edgetutor.app.ui": mock_ui_module, "gradio": MagicMock()}),
            pytest.raises(SystemExit) as exc_info,
        ):
            if "edgetutor.__main__" in sys.modules:
                importlib.reload(sys.modules["edgetutor.__main__"])
            from edgetutor.__main__ import main

            main()

        assert exc_info.value.code == 1

    def test_version_is_available(self):
        """__version__ should be importable from edgetutor."""
        from edgetutor import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0
