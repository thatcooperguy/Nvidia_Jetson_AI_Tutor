"""
EdgeTutor AI — Kid-safe guardrail layer.

Lightweight keyword + pattern filter that runs BEFORE the LLM sees user input
and AFTER the LLM produces output. This is a practical, low-overhead approach;
it is not meant to be bullet-proof but to catch common unsafe queries.

Includes leetspeak / variant spelling normalization so simple evasion
attempts like "b0mb" or "p0rn" are also caught.
"""

from __future__ import annotations

import re

from edgetutor.core.logging_config import get_logger

logger = get_logger(__name__)

# ── Leetspeak / variant normalization ─────────────────────────────────────────
_LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
    "+": "t",
}


def _normalize_leet(text: str) -> str:
    """Normalize common leetspeak substitutions back to plain English."""
    result = []
    for ch in text:
        result.append(_LEET_MAP.get(ch, ch))
    return "".join(result)


# ── Blocked input patterns (user queries) ─────────────────────────────────────
# Each tuple: (compiled regex, category label)
_BLOCKED_INPUT_PATTERNS: list[tuple[re.Pattern, str]] = []

_BLOCKED_KEYWORDS = [
    # Violence / weapons
    (
        r"\b(how\s+to\s+)?(make|build|create)\s+(a\s+)?(bomb|weapon|gun|explosive|poison)\b",
        "weapons",
    ),
    (
        r"\b(kill|murder|assassinate|shoot|stab)\s+(someone|a\s+person|people|him|her|them)\b",
        "violence",
    ),
    # Self-harm
    (r"\b(how\s+to\s+)?(commit\s+)?suicide\b", "self-harm"),
    (r"\b(cut|harm|hurt)\s+(myself|yourself|themselves)\b", "self-harm"),
    # Illegal activity
    (r"\b(how\s+to\s+)?(hack|break\s+into|steal|rob)\b", "illegal"),
    (r"\b(make|cook|produce)\s+(meth|drugs|cocaine|heroin)\b", "drugs"),
    # Explicit / sexual
    (r"\b(porn|pornography|hentai|xxx|nsfw)\b", "explicit"),
    (r"\b(sexual|sex\s+with|naked|nude)\b", "explicit"),
    # Hate speech
    (r"\b(hate|kill)\s+all\s+(jews|muslims|christians|blacks|whites|gays|women|men)\b", "hate"),
]

for pattern_str, category in _BLOCKED_KEYWORDS:
    _BLOCKED_INPUT_PATTERNS.append((re.compile(pattern_str, re.IGNORECASE), category))

# ── Calm redirect message ─────────────────────────────────────────────────────
REDIRECT_MESSAGE = (
    "I'm here to help you learn! That topic isn't something I can help with. "
    "Let's focus on something fun and educational instead. "
    "What subject would you like to work on? I can help with Math, Reading, "
    "Science, or really anything school-related! 📚"
)

# ── Blocked output patterns (LLM responses) ──────────────────────────────────
_BLOCKED_OUTPUT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(instructions\s+to\s+make\s+a\s+bomb)\b", re.IGNORECASE),
    re.compile(r"\b(step\s+\d+.*kill)\b", re.IGNORECASE),
    re.compile(r"\b(here\s+is\s+how\s+to\s+(harm|hurt|kill))\b", re.IGNORECASE),
]


def check_input_safety(text: str) -> tuple[bool, str]:
    """
    Check if user input is safe.

    Applies leetspeak normalization before pattern matching so that
    common evasion attempts like "b0mb" or "p0rn" are still caught.

    Returns:
        (is_safe, category_or_empty): True if safe, False + category if blocked.
    """
    if not text or not text.strip():
        return True, ""

    cleaned = text.strip()

    # Check both original text and leetspeak-normalized version
    variants = [cleaned, _normalize_leet(cleaned)]
    for variant in variants:
        for pattern, category in _BLOCKED_INPUT_PATTERNS:
            if pattern.search(variant):
                logger.warning("Blocked input — category=%s, length=%d", category, len(cleaned))
                return False, category

    return True, ""


def check_output_safety(text: str) -> str:
    """
    Scrub LLM output for unsafe content.

    Returns the original text if safe, or the redirect message if unsafe
    content is detected in the output.
    """
    if not text:
        return text

    for pattern in _BLOCKED_OUTPUT_PATTERNS:
        if pattern.search(text):
            logger.warning("Blocked output — matched pattern in LLM response")
            return REDIRECT_MESSAGE

    return text


def get_redirect_message() -> str:
    """Return the standard safe redirect message."""
    return REDIRECT_MESSAGE
