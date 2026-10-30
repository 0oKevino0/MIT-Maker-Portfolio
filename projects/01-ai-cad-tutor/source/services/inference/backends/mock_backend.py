"""Deterministic mock backend for tests, CI, and early UI integration.

The mock never touches the network. Its output flows through the same shared
draft validation as every real backend (``services.inference.validation``),
and its failure modes exercise that shared trust boundary:

- ``unavailable``: raises like an unreachable provider.
- ``invalid_json``: raises like an adapter that received non-JSON content.
- ``missing_field`` / ``extra_field``: RETURNS a structurally bad draft, so
  the shared validator (not the backend) must reject it.
- ``fabricated_citation``: returns a draft citing an evidence ID that is not
  in the allowed whitelist, so the domain assembler must reject it.

Production configuration must never enable this backend (enforced in
settings validation).
"""

from __future__ import annotations

from typing import Any, Literal

from services.inference.backends.base import (
    BackendCapabilities,
    BackendHealth,
    ChatMessage,
    HealthState,
    ModelBackend,
)
from services.inference.errors import BackendUnavailableError, InvalidModelOutputError

FailMode = Literal[
    "none",
    "unavailable",
    "invalid_json",
    "missing_field",
    "extra_field",
    "fabricated_citation",
]


class MockBackend(ModelBackend):
    name = "mock"

    def __init__(self, fail_mode: FailMode = "none") -> None:
        self._fail_mode = fail_mode

    async def generate_text(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._fail_mode == "unavailable":
            raise BackendUnavailableError("injected: mock backend is unavailable")
        if self._fail_mode == "invalid_json":
            # Simulates a provider whose content could not be parsed as JSON;
            # adapters raise this themselves because they cannot return non-JSON.
            raise InvalidModelOutputError("injected: provider content was not valid JSON")

        last_user_text = next(
            (m.text for m in reversed(messages) if m.role == "user"),
            "",
        )
        draft: dict[str, Any] = {
            "answer_text": (
                "This is a deterministic mock answer produced without course retrieval. "
                f"The question received was: {last_user_text[:200]}"
            ),
            "citation_ids": [],
        }
        if self._fail_mode == "missing_field":
            del draft["citation_ids"]
        elif self._fail_mode == "extra_field":
            # A model trying to invent server-owned data (e.g. a source path)
            # must be rejected by the shared draft validator.
            draft["source_path"] = "C:/fake/lecture.pdf"
        elif self._fail_mode == "fabricated_citation":
            draft["citation_ids"] = ["chunk-fabricated-999"]
            draft["answer_text"] += " [chunk-fabricated-999]"
        return draft

    async def analyze_image(
        self,
        image_bytes: bytes,
        media_type: str,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._fail_mode == "unavailable":
            raise BackendUnavailableError("injected: mock backend is unavailable")
        return {
            "disposition_hint": "insufficient_image_quality",
            "quality_reasons": ["Mock backend does not perform real image analysis."],
        }

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            text=True,
            vision=True,
            structured_output=True,
            max_images=1,
            accepted_media_types=("image/png", "image/jpeg"),
        )

    async def health(self) -> BackendHealth:
        if self._fail_mode == "unavailable":
            return BackendHealth(
                state=HealthState.UNAVAILABLE,
                backend=self.name,
                detail="injected unavailable state",
            )
        return BackendHealth(state=HealthState.READY, backend=self.name)
