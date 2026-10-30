"""Model backend abstraction.

Business services depend on :class:`ModelBackend` only. Provider selection
happens in the factory (see ``factory.py``) driven by configuration, so
switching between mock, Ollama, and vLLM never changes route or service code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class HealthState(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class BackendHealth:
    """Health probe result; never raises, never triggers a generation request."""

    state: HealthState
    backend: str
    detail: str | None = None


@dataclass(frozen=True)
class ChatMessage:
    """One conversation turn sent to the model."""

    role: str  # "system" | "user" | "assistant"
    text: str


@dataclass(frozen=True)
class BackendCapabilities:
    text: bool
    vision: bool
    structured_output: bool
    max_images: int = 0
    accepted_media_types: tuple[str, ...] = field(default_factory=tuple)


class ModelBackend(ABC):
    """Replaceable inference provider.

    Adapters translate the unified request into the provider protocol and map
    provider failures onto the unified errors in ``services.inference.errors``.
    Raw provider responses must never leak into domain contracts.
    """

    name: str = "abstract"

    @abstractmethod
    async def generate_text(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a structured (JSON object) response for a text-only task.

        Returns the parsed JSON object. Raises ``InvalidModelOutputError`` when
        the provider output cannot be strictly parsed as a JSON object.
        """
        raise NotImplementedError

    @abstractmethod
    async def analyze_image(
        self,
        image_bytes: bytes,
        media_type: str,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze a single sanitized image passed as bytes.

        Image bytes only; file paths and URLs are forbidden by ADR 0001.
        """
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> BackendHealth:
        """Probe provider readiness without triggering a generation request."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release shared resources such as HTTP clients."""
        return None
