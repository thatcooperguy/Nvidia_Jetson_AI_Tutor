"""
EdgeTutor AI — Main entry point.

Usage:
    python -m edgetutor
"""

import sys

from edgetutor.core.logging_config import get_logger

logger = get_logger(__name__)


def main():
    """Launch EdgeTutor AI."""
    logger.info("Starting EdgeTutor AI v0.1.0")
    logger.info("=" * 60)
    logger.info("  EdgeTutor AI — Offline AI Tutor for NVIDIA Jetson")
    logger.info("  Access the UI at: http://localhost:7860")
    logger.info("=" * 60)

    try:
        from edgetutor.app.ui import launch_ui

        launch_ui()
    except KeyboardInterrupt:
        logger.info("EdgeTutor shutting down.")
        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
