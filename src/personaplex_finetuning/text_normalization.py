"""Text normalization shared by target construction and evaluation."""

from __future__ import annotations

import unicodedata


_TONE_KEYS = {"\u0301": "s", "\u0300": "f", "\u0309": "r", "\u0303": "x", "\u0323": "j"}
_TONE_MARKS = {value: key for key, value in _TONE_KEYS.items()}
_SHAPE_KEYS = {
    ("a", "\u0306"): "w", ("a", "\u0302"): "a",
    ("e", "\u0302"): "e", ("o", "\u0302"): "o",
    ("o", "\u031b"): "w", ("u", "\u031b"): "w",
}
_VOWELS = set("aeiouy")


def strip_vietnamese_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics while preserving case, spacing, and punctuation."""
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        character for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    without_marks = without_marks.translate(str.maketrans({"đ": "d", "Đ": "D"}))
    return unicodedata.normalize("NFC", without_marks)


def normalize_vietnamese_text(text: str, mode: str = "diacritics") -> str:
    """Apply the configured representation to Vietnamese training/evaluation text."""
    if mode == "diacritics":
        return text
    if mode == "no_diacritics":
        return strip_vietnamese_diacritics(text)
    if mode == "telex":
        return encode_vietnamese_telex(text)
    raise ValueError(
        f"unsupported data.vietnamese_text_mode {mode!r}; "
        "expected diacritics, no_diacritics, or telex"
    )


def _encode_telex_syllable(syllable: str) -> str:
    decomposed = unicodedata.normalize("NFD", syllable)
    letters: list[tuple[str, set[str], bool]] = []
    for character in decomposed:
        if unicodedata.category(character) == "Mn" and letters:
            letters[-1][1].add(character)
        elif character.lower() == "đ":
            letters.append((character, set(), True))
        else:
            letters.append((character, set(), False))

    # Telex writes ươ as the base sequence "uo" followed by one w key.
    output: list[str] = []
    tone_key = ""
    index = 0
    while index < len(letters):
        character, marks, is_stroked_d = letters[index]
        lower = character.lower()
        if is_stroked_d:
            output.append(("D" if character.isupper() else "d") + "d")
            index += 1
            continue
        if lower in _VOWELS:
            tone = next((mark for mark in marks if mark in _TONE_KEYS), None)
            if tone:
                tone_key = _TONE_KEYS[tone].upper() if character.isupper() else _TONE_KEYS[tone]
            if index + 1 < len(letters):
                next_character, next_marks, next_stroked = letters[index + 1]
                follows_with_vowel = (
                    index + 2 < len(letters)
                    and letters[index + 2][0].lower() in _VOWELS
                )
                if (lower, next_character.lower()) == ("u", "o") and not next_stroked \
                        and "\u031b" in marks and "\u031b" in next_marks and not follows_with_vowel:
                    modifier = "W" if character.isupper() and next_character.isupper() else "w"
                    output.append(character + next_character + modifier)
                    index += 2
                    continue
            key = _SHAPE_KEYS.get((lower, next((m for m in marks if m in "\u0306\u0302\u031b"), "")), "")
            output.append(character + (key.upper() if character.isupper() else key))
        else:
            output.append(character)
        index += 1
    return "".join(output) + tone_key


def encode_vietnamese_telex(text: str) -> str:
    """Encode Unicode Vietnamese syllables as canonical Telex keystrokes."""
    # Vietnamese syllables are whitespace-separated; preserve all separators verbatim.
    return "".join(
        _encode_telex_syllable(part) if part and any(ch.isalpha() for ch in part) else part
        for part in _split_word_parts(text)
    )


def _split_word_parts(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    current_is_word = None
    for index, character in enumerate(text):
        is_word = character.isalpha() or unicodedata.category(character) == "Mn"
        if current_is_word is None:
            current_is_word = is_word
        elif is_word != current_is_word:
            parts.append(text[start:index])
            start = index
            current_is_word = is_word
    if text:
        parts.append(text[start:])
    return parts


def _apply_tone(syllable: str, tone_key: str) -> str:
    vowels = [
        i for i, char in enumerate(syllable)
        if next((c.lower() for c in unicodedata.normalize("NFD", char)
                 if unicodedata.category(c) != "Mn"), char.lower()) in _VOWELS
    ]
    # In qu- and gi- onsets, u/i is a glide rather than the vowel nucleus when a
    # following vowel is present (e.g. quá, giá, quyền).
    if len(syllable) > 2 and syllable[:2].lower() in {"qu", "gi"} and len(vowels) > 1:
        vowels = [index for index in vowels if index != 1]
    if not vowels:
        return syllable
    # A final consonant places the tone on the final vowel of a diphthong; an open
    # diphthong places it on the first. Triphthongs carry it on the middle vowel.
    if len(vowels) == 2:
        has_coda = any(not char.isalpha() or char.lower() not in _VOWELS for char in syllable[vowels[-1] + 1:])
        target = vowels[1] if has_coda else vowels[0]
    else:
        target = vowels[1] if len(vowels) >= 3 else vowels[0]
    character = syllable[target]
    decomposed = unicodedata.normalize("NFD", character)
    base = next((part for part in decomposed if unicodedata.category(part) != "Mn"), character)
    marks = [part for part in decomposed if unicodedata.category(part) == "Mn"]
    marks = [mark for mark in marks if mark not in _TONE_KEYS] + [_TONE_MARKS[tone_key]]
    replacement = unicodedata.normalize("NFC", base + "".join(marks))
    return syllable[:target] + replacement + syllable[target + 1:]


def _decode_telex_syllable(syllable: str) -> str:
    tone_key = ""
    if syllable and syllable[-1].lower() in _TONE_MARKS and any(c.lower() in _VOWELS for c in syllable[:-1]):
        tone_key = syllable[-1].lower()
        syllable = syllable[:-1]

    output: list[str] = []
    index = 0
    while index < len(syllable):
        char = syllable[index]
        lower = char.lower()
        if lower == "d" and index + 1 < len(syllable) and syllable[index + 1].lower() == "d":
            output.append("Đ" if char.isupper() else "đ")
            index += 2
            continue
        if lower in _VOWELS and index + 1 < len(syllable):
            following = syllable[index + 1].lower()
            pair = lower + following
            if pair == "uo" and index + 2 < len(syllable) and syllable[index + 2].lower() == "w":
                output.extend(("Ư" if char.isupper() else "ư", "Ơ" if syllable[index + 1].isupper() else "ơ"))
                index += 3
                continue
            doubled = {"aa": "â", "aw": "ă", "ee": "ê", "oo": "ô", "ow": "ơ", "uw": "ư"}
            if pair in doubled:
                mapped = doubled[pair]
                output.append(mapped.upper() if char.isupper() else mapped)
                index += 2
                continue
        output.append(char)
        index += 1
    decoded = "".join(output)
    return _apply_tone(decoded, tone_key) if tone_key else decoded


def decode_vietnamese_telex(text: str) -> str:
    """Decode canonical Telex keystrokes into Unicode Vietnamese for inspection."""
    return "".join(
        _decode_telex_syllable(part) if part and any(ch.isalpha() for ch in part) else part
        for part in _split_word_parts(text)
    )
