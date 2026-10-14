"""OpenAI-compatible client (OpenAI, DeepSeek, Ollama, LM Studio, vLLM, ...).

Adapted from MultiAgentDebate backend/deepseek_client.py (MIT);
see THIRD_PARTY_NOTICES.md. Requires the optional `openai` extra.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .base import LLMResult, Usage


@dataclass(frozen=True)
class OpenAICompatConfig:
    api_key: str
    model: str
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "OpenAICompatConfig":
        api_key = os.environ.get("TROLLKNIGHT_API_KEY")
        model = os.environ.get("TROLLKNIGHT_MODEL")
        if not api_key:
            raise RuntimeError("TROLLKNIGHT_API_KEY is not set")
        if not model:
            raise RuntimeError("TROLLKNIGHT_MODEL is not set")
        return cls(
            api_key=api_key,
            model=model,
            base_url=os.environ.get("TROLLKNIGHT_BASE_URL") or None,
        )


class OpenAICompatClient:
    def __init__(self, config: OpenAICompatConfig):
        from openai import AsyncOpenAI  # imported lazily: optional dependency

        self._model = config.model
        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 400,
    ) -> LLMResult:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        u = getattr(resp, "usage", None)
        usage = Usage(
            int(getattr(u, "prompt_tokens", 0) or 0),
            int(getattr(u, "completion_tokens", 0) or 0),
        )
        return LLMResult(text=text, usage=usage)
