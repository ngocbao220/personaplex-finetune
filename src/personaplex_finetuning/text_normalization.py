"""Text normalization shared by target construction and evaluation."""

from __future__ import annotations

import unicodedata


def strip_vietnamese_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics while preserving case, spacing, and punctuation."""
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        character for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    without_marks = without_marks.translate(str.maketrans({"đ": "d", "Đ": "D"}))
    return unicodedata.normalize("NFC", without_marks)
