from __future__ import annotations

import re
from typing import Optional


_CANONICAL_TO_VARIANTS = {
    "A": ("ALFA", "ALPHA"),
    "B": ("BRAVO",),
    "C": ("CHARLIE",),
    "D": ("DELTA",),
    "E": ("ECHO",),
    "F": ("FOXTROT",),
    "G": ("GOLF",),
    "H": ("HOTEL",),
    "I": ("INDIA",),
    "J": ("JULIET", "JULIETT"),
    "K": ("KILO",),
    "L": ("LIMA",),
    "M": ("MIKE",),
    "N": ("NOVEMBER",),
    "O": ("OSCAR",),
    "P": ("PAPA",),
    "Q": ("QUEBEC",),
    "R": ("ROMEO",),
    "S": ("SIERRA",),
    "T": ("TANGO",),
    "U": ("UNIFORM",),
    "V": ("VICTOR",),
    "W": ("WHISKEY",),
    "X": ("XRAY", "X-RAY"),
    "Y": ("YANKEE",),
    "Z": ("ZULU",),
}

_VARIANT_TO_CANONICAL = {
    variant: letter
    for letter, variants in _CANONICAL_TO_VARIANTS.items()
    for variant in variants
}

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
_SINGLE_LETTER_RE = re.compile(r"^[A-Za-z]$")


def extract_first_nato_letter(text: str) -> Optional[str]:
    """Extract the first NATO code from text and return A-Z letter."""
    if not text:
        return None

    for token in _TOKEN_RE.findall(text):
        normalized = token.upper()
        if normalized in _VARIANT_TO_CANONICAL:
            return _VARIANT_TO_CANONICAL[normalized]

    for token in _TOKEN_RE.findall(text):
        if _SINGLE_LETTER_RE.match(token):
            return token.upper()
    return None


def extract_first_nato_word(text: str) -> Optional[str]:
    """Extract the first NATO code from text and return canonical word form."""
    letter = extract_first_nato_letter(text)
    if not letter:
        return None
    return _CANONICAL_TO_VARIANTS[letter][0]
