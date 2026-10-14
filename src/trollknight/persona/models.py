"""Persona = tone and style only. Safety rules live in policy.py."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    style: str  # how it talks: e.g. "dry, sarcastic, bored"
    max_sentences: int = 3
    language: str = "ko"
    examples: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.style.strip():
            raise ValueError("persona requires id and style")
        if self.max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")


def load_persona(path: str | Path) -> Persona:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return Persona(
        id=data["id"],
        name=data.get("name", data["id"]),
        style=data["style"],
        max_sentences=int(data.get("max_sentences", 3)),
        language=data.get("language", "ko"),
        examples=tuple(data.get("examples", ())),
    )
