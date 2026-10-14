"""Conversation state, limits, and the reply loop. No UI, no posting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from ..llm.base import LLMClient, Usage
from .repetition import is_repetition


class StopReason(str, Enum):
    MAX_REPLIES = "max_replies"
    MAX_SECONDS = "max_seconds"
    MAX_COST = "max_cost"
    REPETITION = "repetition"
    USER_STOP = "user_stop"


@dataclass(frozen=True)
class Limits:
    max_replies: int = 5
    max_seconds: float = 300.0
    max_cost_usd: float = 0.50
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    repetition_threshold: float = 0.85
    max_repeats_in_a_row: int = 2


@dataclass
class Turn:
    role: str  # "other" (inbound) or "agent" (our reply)
    text: str
    generation_seconds: float = 0.0
    repeated: bool = False
    violations: list = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.violations)


@dataclass
class Conversation:
    llm: LLMClient
    system_prompt: str
    limits: Limits = field(default_factory=Limits)
    clock: Callable[[], float] = time.monotonic
    guard: Optional[Callable[[str], list]] = None
    turns: list[Turn] = field(default_factory=list)
    usage: Usage = Usage()
    stop_reason: StopReason | None = None
    _started_at: float | None = None
    _repeats_in_a_row: int = 0

    @property
    def stopped(self) -> bool:
        return self.stop_reason is not None

    @property
    def reply_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "agent")

    @property
    def elapsed_seconds(self) -> float:
        return 0.0 if self._started_at is None else self.clock() - self._started_at

    @property
    def estimated_cost_usd(self) -> float:
        l = self.limits
        return (
            self.usage.prompt_tokens * l.price_in_per_mtok
            + self.usage.completion_tokens * l.price_out_per_mtok
        ) / 1_000_000

    def stop(self, reason: StopReason = StopReason.USER_STOP) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason

    def _check_limits(self) -> None:
        l = self.limits
        if self.reply_count >= l.max_replies:
            self.stop(StopReason.MAX_REPLIES)
        elif self.elapsed_seconds >= l.max_seconds:
            self.stop(StopReason.MAX_SECONDS)
        elif self.estimated_cost_usd >= l.max_cost_usd:
            self.stop(StopReason.MAX_COST)

    def _messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        for t in self.turns:
            msgs.append(
                {"role": "user" if t.role == "other" else "assistant", "content": t.text}
            )
        return msgs

    async def draft_reply(self, inbound: str) -> Turn | None:
        """Record an inbound message and generate a reply draft.

        Returns None if the conversation is (or becomes) stopped. The draft is
        not final: callers decide whether to approve and send it.
        """
        if self._started_at is None:
            self._started_at = self.clock()
        self._check_limits()
        if self.stopped:
            return None

        self.turns.append(Turn(role="other", text=inbound))
        t0 = self.clock()
        result = await self.llm.complete(self._messages())
        gen = self.clock() - t0

        self.usage = Usage(
            self.usage.prompt_tokens + result.usage.prompt_tokens,
            self.usage.completion_tokens + result.usage.completion_tokens,
        )

        previous = [t.text for t in self.turns if t.role == "agent"]
        repeated = is_repetition(
            result.text, previous, self.limits.repetition_threshold
        )
        turn = Turn(
            role="agent",
            text=result.text,
            generation_seconds=gen,
            repeated=repeated,
            violations=self.guard(result.text) if self.guard else [],
        )
        self.turns.append(turn)

        self._repeats_in_a_row = self._repeats_in_a_row + 1 if repeated else 0
        if self._repeats_in_a_row >= self.limits.max_repeats_in_a_row:
            self.stop(StopReason.REPETITION)
        else:
            self._check_limits()
        return turn

    def metrics(self) -> dict:
        agent_turns = [t for t in self.turns if t.role == "agent"]
        n = len(agent_turns)
        return {
            "reply_count": n,
            "elapsed_seconds": self.elapsed_seconds,
            "avg_generation_seconds": (
                sum(t.generation_seconds for t in agent_turns) / n if n else 0.0
            ),
            "repetition_ratio": (
                sum(1 for t in agent_turns if t.repeated) / n if n else 0.0
            ),
            "blocked_count": sum(1 for t in agent_turns if t.blocked),
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
        }
