"""vLLM backend stub.

This is an explicit, interface-compatible stub. The real adapter (planned for
the optional HPC track, see HPC_PLAN.md and ADR 0001 section 13.5) will talk
to a controlled vLLM server over direct HTTP. It must never be reachable from
the browser or the SOLIDWORKS add-in, and it is never a silent fallback for a
failing local backend.
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
from services.inference.errors import BackendUnavailableError


class VllmBackend(ModelBackend):
    name = "vllm"

    def __init__(
        self,
        model_id: str,
        base_url: str,
        connect_timeout_seconds: float = 5.0,
        deadline_seconds: float = 60.0,
    ) -> None:
        self._model_id = model_id
        self._base_url = base_url

    async def generate_text(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise BackendUnavailableError(
            "VllmBackend is a stub in the bootstrap; use 'mock' or 'ollama' instead"
        )

    async def analyze_image(
        self,
        image_bytes: bytes,
        media_type: str,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise BackendUnavailableError(
            "VllmBackend is a stub in the bootstrap; use 'mock' or 'ollama' instead"
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(text=False, vision=False, structured_output=False)

    async def health(self) -> BackendHealth:
        return BackendHealth(
            state=HealthState.UNAVAILABLE,
            backend=self.name,
            detail="vLLM adapter is not implemented yet (bootstrap stub)",
        )
