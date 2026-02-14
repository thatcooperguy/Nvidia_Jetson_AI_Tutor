"""Tests for edgetutor.vision.ocr — OCR pipeline and math detection."""


class TestMathDetection:
    """Test the math detection heuristics (does not require Tesseract)."""

    def _get_engine(self):
        """Get OCR engine instance (mock availability)."""
        from edgetutor.vision.ocr import OCREngine

        engine = OCREngine.__new__(OCREngine)
        engine._available = False  # Don't require Tesseract for these tests
        return engine

    def test_detects_simple_equation(self):
        """Should detect simple arithmetic equations."""
        engine = self._get_engine()
        lines = ["2 + 3 = 5", "Hello world"]
        has_math, expressions = engine._detect_math("2 + 3 = 5\nHello world", lines)
        assert has_math
        assert "2 + 3 = 5" in expressions

    def test_detects_fractions(self):
        """Should detect fractions."""
        engine = self._get_engine()
        lines = ["1/2 + 1/4 = 3/4"]
        has_math, expressions = engine._detect_math("1/2 + 1/4 = 3/4", lines)
        assert has_math

    def test_detects_variables(self):
        """Should detect algebraic expressions."""
        engine = self._get_engine()
        lines = ["2x + 3y = 10"]
        has_math, expressions = engine._detect_math("2x + 3y = 10", lines)
        assert has_math

    def test_detects_math_keywords(self):
        """Should detect lines with math keywords."""
        engine = self._get_engine()
        lines = ["Solve the following equation:"]
        has_math, expressions = engine._detect_math("Solve the following equation:", lines)
        assert has_math

    def test_no_math_in_plain_text(self):
        """Plain text without math should not trigger detection."""
        engine = self._get_engine()
        lines = ["The cat sat on the mat.", "It was a sunny day."]
        text = "\n".join(lines)
        has_math, expressions = engine._detect_math(text, lines)
        assert not has_math
        assert len(expressions) == 0

    def test_detects_exponents(self):
        """Should detect exponent notation."""
        engine = self._get_engine()
        lines = ["x^2 + y^2 = z^2"]
        has_math, expressions = engine._detect_math("x^2 + y^2 = z^2", lines)
        assert has_math

    def test_detects_sqrt(self):
        """Should detect square root."""
        engine = self._get_engine()
        lines = ["sqrt(16) = 4"]
        has_math, expressions = engine._detect_math("sqrt(16) = 4", lines)
        assert has_math


class TestOCRErrorResult:
    """Test error result format."""

    def test_error_result_format(self):
        """Error result should have all required keys."""
        from edgetutor.vision.ocr import OCREngine

        result = OCREngine._error_result("test error")
        assert result["text"] == ""
        assert result["confidence"] == 0.0
        assert result["lines"] == []
        assert result["has_math"] is False
        assert result["math_expressions"] == []
        assert result["error"] == "test error"
