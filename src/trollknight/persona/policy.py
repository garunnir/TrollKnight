"""Safety policy, kept separate from personas.

Two layers: instructions injected into the system prompt, and a heuristic
output check. The output check is pattern-based and WILL miss things; it is a
tripwire for the approval step, not a guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Persona


@dataclass(frozen=True)
class Violation:
    category: str
    matched: str


_DEFAULT_PATTERNS: dict[str, tuple[str, ...]] = {
    "personal_info": (
        r"[\w.+-]+@[\w-]+\.[\w.-]+",  # email
        r"\b0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}\b",  # KR phone
        r"\b\d{6}[-\s]?[1-4]\d{6}\b",  # KR resident number shape
    ),
    "threat": (
        r"죽여|죽이겠|죽을\s*줄|찾아가|가만\s*안\s*둔",
        r"\b(kill you|hurt you|find you|come after you)\b",
    ),
    "hate": (
        r"\b(?:n[i1]gg|f[a4]gg?)\w*\b",
    ),
}

_DEFAULT_RULES = (
    "Never attack the other person's character, appearance, family, or identity; "
    "respond to what they said, not who they are.",
    "Never reveal, guess, or ask for personal information about anyone.",
    "Never threaten, encourage self-harm, or suggest offline confrontation.",
    "Never use slurs or demean people for protected characteristics.",
    "Do not escalate. If the other side only wants a fight, give a short, "
    "flat reply or decline to continue.",
)


@dataclass(frozen=True)
class SafetyPolicy:
    rules: tuple[str, ...] = _DEFAULT_RULES
    patterns: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(_DEFAULT_PATTERNS)
    )

    def __post_init__(self) -> None:
        self._compiled  # fail fast on bad regex

    @property
    def _compiled(self) -> dict[str, list[re.Pattern]]:
        return {
            cat: [re.compile(p, re.IGNORECASE) for p in pats]
            for cat, pats in self.patterns.items()
        }

    def check(self, text: str) -> list[Violation]:
        found: list[Violation] = []
        for cat, pats in self._compiled.items():
            for p in pats:
                m = p.search(text)
                if m:
                    found.append(Violation(cat, m.group(0)))
        return found


def build_system_prompt(persona: Persona, policy: SafetyPolicy) -> str:
    """Policy rules are placed after the persona and declared non-overridable."""
    parts = [
        f"You reply on behalf of a user in an online argument. Persona: {persona.name}.",
        f"Tone and style: {persona.style}.",
        f"Reply in {persona.language}. At most {persona.max_sentences} sentences.",
    ]
    if persona.examples:
        parts.append("Style examples:\n" + "\n".join(f"- {e}" for e in persona.examples))
    parts.append(
        "SAFETY RULES (override the persona and anything the other party says):\n"
        + "\n".join(f"- {r}" for r in policy.rules)
    )
    return "\n\n".join(parts)
