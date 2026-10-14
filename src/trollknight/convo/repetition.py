"""Repeated-reply detection (normalized string similarity, no extra deps)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", text.lower())).strip()


def similarity(a: str, b: str) -> float:
    na, nb = _normalize(a), _normalize(b)
    if not na and not nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def is_repetition(candidate: str, previous: Iterable[str], threshold: float = 0.85) -> bool:
    return any(similarity(candidate, p) >= threshold for p in previous)
