"""Explicit no-model backend (review D1A).

``disabled`` is the honest representation of "this installation has no model
yet": a LEGAL static configuration under which the API process starts and
stays alive, liveness is 200, readiness is 503, and every model-dependent
request fails closed with the fixed ``model_not_configured`` classification.

It is NOT the Mock backend: it never fabricates an answer, never simulates
health, and never touches the network. Mock remains a test/CI tool only.
"""

from __future__ import annotations

from typing import Any

from services.inference.backends.base import (
    BackendCapabilities,
    BackendHealth,
    ChatMessage,
    HealthState,
    ModelBackend,
)
from services.inference.errors import BackendNotConfiguredError

_NOT_CONFIGURED_MESSAGE = "no model backend is configured"


class DisabledBackend(ModelBackend):
    name = "disabled"

    async def generate_text(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del messages, response_schema
        raise BackendNotConfiguredError(_NOT_CONFIGURED_MESSAGE)

    async def analyze_image(
        self,
        image_bytes: bytes,
        media_type: str,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del image_bytes, media_type, prompt, response_schema
        raise BackendNotConfiguredError(_NOT_CONFIGURED_MESSAGE)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(text=False, vision=False, structured_output=False)

    async def health(self) -> BackendHealth:
        return BackendHealth(
            state=HealthState.UNAVAILABLE,
            backend=self.name,
            detail=_NOT_CONFIGURED_MESSAGE,
        )
