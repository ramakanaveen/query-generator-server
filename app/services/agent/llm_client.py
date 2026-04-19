"""
LLM client abstraction for the v2 agent.
Supports two backends selected by LLM_BACKEND env var:
  "vertex"  → AnthropicVertex  (org-routed via Google Cloud)
  "direct"  → AsyncAnthropic   (direct Anthropic API)

Model resolution for Vertex:
  Direct API format:  claude-3-7-sonnet-20250219
  Vertex format:      claude-3-7-sonnet@20250219   (hyphen before date → @)
  Short aliases:      "claude-sonnet" → resolved via VERTEX_MODEL_ALIASES map

Set CLAUDE_DEFAULT_MODEL in .env to whichever model your Vertex project has access to.
"""
from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# Short alias → direct API model ID
_DIRECT_MODELS: dict[str, str] = {
    "claude-haiku":           "claude-haiku-4-5-20251001",
    "claude-haiku-4-5":       "claude-haiku-4-5-20251001",
    "claude-sonnet":          "claude-sonnet-4-20250514",
    "claude-sonnet-4":        "claude-sonnet-4-20250514",
    "claude-sonnet-4-6":      "claude-sonnet-4-6",
    "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
    "claude-opus":            "claude-opus-4-7",
    "claude-opus-4-7":        "claude-opus-4-7",
}

# Short alias → Vertex AI model ID
# Vertex uses @ instead of the last hyphen before the date: claude-3-7-sonnet@20250219
_VERTEX_MODELS: dict[str, str] = {
    "claude-haiku":          "claude-haiku-4-5@20251001",
    "claude-haiku-4-5":      "claude-haiku-4-5@20251001",
    "claude-sonnet":         "claude-3-7-sonnet@20250219",   # most likely available on Vertex
    "claude-sonnet-4-6":     "claude-sonnet-4-6@20250514",
    "claude-opus":           "claude-opus-4-7@20250514",
    "claude-opus-4-7":       "claude-opus-4-7@20250514",
    # Common direct-API style names users may pass
    "claude-3-7-sonnet-20250219":     "claude-3-7-sonnet@20250219",
    "claude-3-5-sonnet-20241022":     "claude-3-5-sonnet-v2@20241022",
    "claude-3-5-sonnet-v2-20241022":  "claude-3-5-sonnet-v2@20241022",
    "claude-3-5-haiku-20241022":      "claude-3-5-haiku@20241022",
}


def _direct_to_vertex(model_id: str) -> str:
    """
    Auto-convert direct API model IDs to Vertex format when not in the alias map.
    claude-3-7-sonnet-20250219  →  claude-3-7-sonnet@20250219
    """
    # Match pattern: <name>-<8-digit-date>
    m = re.match(r"^(.*)-(\d{8})$", model_id)
    if m:
        return f"{m.group(1)}@{m.group(2)}"
    return model_id


class LLMClient(ABC):
    """Uniform interface regardless of backend."""

    @abstractmethod
    def resolve_model(self, alias: str) -> str: ...

    @abstractmethod
    async def create(
        self,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        thinking: dict | None = None,
    ) -> Any:
        """Blocking call — returns full anthropic Message."""
        ...

    @abstractmethod
    def stream(
        self,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        thinking: dict | None = None,
    ):
        """Returns an anthropic AsyncMessageStream context manager."""
        ...


class AnthropicDirectClient(LLMClient):
    def __init__(self) -> None:
        import anthropic
        from app.core.config import settings
        self._client = anthropic.AsyncAnthropic(api_key=settings.API_KEY_ANTHROPIC)
        logger.info("LLMClient: Anthropic Direct API")

    def resolve_model(self, alias: str) -> str:
        resolved = _DIRECT_MODELS.get(alias, alias)
        logger.info(f"LLMClient model: {alias!r} → {resolved!r}")
        return resolved

    async def create(self, model, system, messages, tools, max_tokens, thinking=None):
        kwargs: dict[str, Any] = dict(
            model=self.resolve_model(model), system=system,
            messages=messages, tools=tools, max_tokens=max_tokens,
        )
        if thinking:
            kwargs["thinking"] = thinking
        return await self._client.messages.create(**kwargs)

    def stream(self, model, system, messages, tools, max_tokens, thinking=None):
        kwargs: dict[str, Any] = dict(
            model=self.resolve_model(model), system=system,
            messages=messages, tools=tools, max_tokens=max_tokens,
        )
        if thinking:
            kwargs["thinking"] = thinking
        return self._client.messages.stream(**kwargs)


class AnthropicVertexClient(LLMClient):
    def __init__(self) -> None:
        import anthropic
        from app.core.config import settings
        self._client = anthropic.AsyncAnthropicVertex(
            project_id=settings.GOOGLE_PROJECT_ID,
            region=settings.GOOGLE_LOCATION,
        )
        logger.info(
            f"LLMClient: Anthropic via Vertex AI "
            f"(project={settings.GOOGLE_PROJECT_ID}, region={settings.GOOGLE_LOCATION})"
        )

    def resolve_model(self, alias: str) -> str:
        # Check explicit alias map first
        if alias in _VERTEX_MODELS:
            resolved = _VERTEX_MODELS[alias]
        else:
            # Auto-convert direct API format to Vertex format
            resolved = _direct_to_vertex(alias)
        logger.info(f"LLMClient model: {alias!r} → {resolved!r} (Vertex)")
        return resolved

    async def create(self, model, system, messages, tools, max_tokens, thinking=None):
        kwargs: dict[str, Any] = dict(
            model=self.resolve_model(model), system=system,
            messages=messages, tools=tools, max_tokens=max_tokens,
        )
        if thinking:
            kwargs["thinking"] = thinking
        return await self._client.messages.create(**kwargs)

    def stream(self, model, system, messages, tools, max_tokens, thinking=None):
        kwargs: dict[str, Any] = dict(
            model=self.resolve_model(model), system=system,
            messages=messages, tools=tools, max_tokens=max_tokens,
        )
        if thinking:
            kwargs["thinking"] = thinking
        return self._client.messages.stream(**kwargs)


def get_llm_client() -> LLMClient:
    from app.core.config import settings
    if settings.LLM_BACKEND == "direct":
        return AnthropicDirectClient()
    return AnthropicVertexClient()
