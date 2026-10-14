"""Model-agnostic LLM interface. Conversation logic depends only on this."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class LLMResult:
    text: str
    usage: Usage = Usage()


class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 400,
    ) -> LLMResult: ...


class FakeLLM:
    """Deterministic client for tests and simulation mode. Cycles through replies."""

    def __init__(self, replies: Iterable[str], usage: Usage = Usage(10, 10)):
        self._replies = list(replies)
        if not self._replies:
            raise ValueError("FakeLLM needs at least one reply")
        self._usage = usage
        self.calls: list[list[dict]] = []

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 400,
    ) -> LLMResult:
        self.calls.append(messages)
        text = self._replies[(len(self.calls) - 1) % len(self._replies)]
        return LLMResult(text=text, usage=self._usage)
