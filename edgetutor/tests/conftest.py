"""
Pytest configuration and shared fixtures for EdgeTutor tests.

These fixtures ensure tests can run in CI without hardware dependencies.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def disable_hardware_for_tests(monkeypatch):
    """Disable hardware-dependent features during testing."""
    monkeypatch.setenv("STT_ENABLED", "false")
    monkeypatch.setenv("TTS_ENABLED", "false")
    monkeypatch.setenv("CAMERA_ENABLED", "false")
    monkeypatch.setenv("LLM_MODEL_PATH", "models/nonexistent.gguf")


@pytest.fixture
def sample_ocr_text():
    """Sample OCR text for testing."""
    return """Math Homework - Chapter 5

    1. Solve: 3x + 7 = 22
    2. Calculate: 15/3 + 2
    3. What is the area of a rectangle with length 5 and width 3?
    4. Simplify: 2(x + 4) - 3
    """


@pytest.fixture
def sample_reading_text():
    """Sample reading text for testing."""
    return """The water cycle is a continuous process that describes how water
    moves through the Earth's systems. Water evaporates from oceans, lakes,
    and rivers, rising into the atmosphere as water vapor. As this vapor
    cools, it condenses to form clouds. Eventually, the water falls back
    to Earth as precipitation — rain, snow, sleet, or hail. This water
    then collects in bodies of water, and the cycle begins again."""
