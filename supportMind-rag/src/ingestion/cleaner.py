"""
cleaner.py — Light text normalization before chunking.

Removes zero-width characters, collapses runs of whitespace, and normalizes line
endings. Deliberately conservative: KB articles often use bullet markers or
inline code that we don't want to discard.
"""
import re

_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
