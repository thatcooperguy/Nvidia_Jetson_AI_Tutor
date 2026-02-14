"""
EdgeTutor AI — OCR pipeline (Tesseract).

Extracts text from images of worksheets, textbook pages, and handwritten notes.
Includes math detection heuristics to identify equations and expressions.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from edgetutor.core.logging_config import get_logger
from edgetutor.core.settings import get_settings

logger = get_logger(__name__)


class OCREngine:
    """Tesseract-based OCR with math detection."""

    def __init__(self):
        self._available = False
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if Tesseract is installed and accessible."""
        try:
            import pytesseract

            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR available — version %s", version)
            self._available = True
        except ImportError:
            logger.error("pytesseract not installed. Install with: pip install pytesseract")
        except Exception as e:
            logger.error(
                "Tesseract not found. Install with: sudo apt install tesseract-ocr — %s", e
            )

    @property
    def is_available(self) -> bool:
        return self._available

    def extract_text(
        self,
        image_input,
        language: Optional[str] = None,
    ) -> dict:
        """
        Extract text from an image.

        Args:
            image_input: PIL Image, numpy array, or file path.
            language: Tesseract language code (e.g., "eng", "eng+spa").

        Returns:
            dict: {
                "text": str,           # Full extracted text
                "confidence": float,   # Average confidence (0-100)
                "lines": list[str],    # Individual lines
                "has_math": bool,      # Whether math was detected
                "math_expressions": list[str],  # Detected math expressions
                "elapsed_s": float,
                "error": str | None,
            }
        """
        if not self._available:
            return self._error_result("Tesseract OCR is not available.")

        cfg = get_settings()
        lang = language or cfg.ocr_language

        try:
            from PIL import Image
            import pytesseract

            # Normalize input to PIL Image
            if isinstance(image_input, (str, Path)):
                image = Image.open(str(image_input))
            elif hasattr(image_input, "mode"):  # Already a PIL Image
                image = image_input
            else:
                # Assume numpy array
                image = Image.fromarray(image_input)

            # Preprocessing: convert to RGB, basic enhancement
            image = image.convert("RGB")
            image = self._preprocess(image)

            t0 = time.time()

            # Get detailed data for confidence
            data = pytesseract.image_to_data(
                image, lang=lang, output_type=pytesseract.Output.DICT
            )

            # Also get plain text (often cleaner)
            text = pytesseract.image_to_string(image, lang=lang).strip()

            elapsed = time.time() - t0

            # Calculate average confidence (ignore -1 entries)
            confidences = [c for c in data["conf"] if isinstance(c, (int, float)) and c >= 0]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            # Split into lines
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            # Detect math
            has_math, math_expressions = self._detect_math(text, lines)

            logger.info(
                "OCR: %d chars, %d lines, conf=%.0f%%, math=%s (%.1fs)",
                len(text),
                len(lines),
                avg_conf,
                has_math,
                elapsed,
            )

            return {
                "text": text,
                "confidence": avg_conf,
                "lines": lines,
                "has_math": has_math,
                "math_expressions": math_expressions,
                "elapsed_s": elapsed,
                "error": None,
            }

        except ImportError as e:
            return self._error_result(f"Missing dependency: {e}")
        except Exception as e:
            logger.error("OCR extraction failed: %s", e)
            return self._error_result(str(e))

    def _preprocess(self, image):
        """Basic image preprocessing to improve OCR accuracy."""
        from PIL import ImageEnhance, ImageFilter

        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)

        return image

    def _detect_math(self, text: str, lines: list[str]) -> tuple[bool, list[str]]:
        """
        Heuristic math detection in OCR text.

        Returns (has_math, list_of_math_expressions).
        """
        math_patterns = [
            # Basic equations: 2 + 3 = 5, x = 7
            r"[\d\w]+\s*[+\-*/÷×=]+\s*[\d\w]+",
            # Fractions: 1/2, 3/4
            r"\d+\s*/\s*\d+",
            # Exponents written as: 2^3, x^2
            r"\w+\s*\^\s*\d+",
            # Parenthesized expressions: (2 + 3)
            r"\([^)]*[+\-*/÷×=][^)]*\)",
            # Common math keywords
            r"\b(solve|calculate|simplify|factor|evaluate|equation|formula)\b",
            # Comparison: >, <, >=, <=
            r"\d+\s*[><≥≤]+\s*\d+",
            # Square root symbol or "sqrt"
            r"(√|sqrt)\s*[\d\w(]",
            # Variables with operations: 2x + 3y
            r"\d+\s*[a-zA-Z]\s*[+\-]",
        ]

        math_expressions = []
        for line in lines:
            for pattern in math_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                if matches:
                    # Add the whole line as context, not just the match
                    if line not in math_expressions:
                        math_expressions.append(line)
                    break

        has_math = len(math_expressions) > 0
        return has_math, math_expressions

    @staticmethod
    def _error_result(error_msg: str) -> dict:
        return {
            "text": "",
            "confidence": 0.0,
            "lines": [],
            "has_math": False,
            "math_expressions": [],
            "elapsed_s": 0.0,
            "error": error_msg,
        }


# Singleton
_ocr_instance = None


def get_ocr() -> OCREngine:
    """Return the singleton OCR engine."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = OCREngine()
    return _ocr_instance
