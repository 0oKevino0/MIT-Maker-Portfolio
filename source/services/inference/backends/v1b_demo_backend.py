"""Adapter-aware backend for the INTERNAL DEMO profile.

Talks to the V1B adapter sidecar (``services/inference/v1b_demo_sidecar.py``) over plain HTTP on a
loopback address. The sidecar is the process that actually holds the frozen base checkpoint with
the owner-selected LoRA adapter attached; this class is the repository's unified ``ModelBackend``
view of it.

Why this exists rather than reusing a backend that already serves the base model: the vLLM backend
is an explicit stub, and the Ollama backend speaks to a GGUF runtime, which this PEFT safetensors
adapter is not. Converting the adapter is forbidden, so the demo needs a path that loads it as it
is. Only this connection was added.

Two properties matter for honesty and both are enforced here rather than configured:

- **identity, not naming.** The expected adapter digest is REQUIRED to construct this backend: with
  nothing to compare against, "verified" would mean nothing, so the backend refuses to exist. Every
  generation is preceded by a verified identity — if none has been established yet this backend
  probes the sidecar itself and refuses to answer when that probe fails, and a failed probe clears
  any identity verified earlier rather than leaving a stale one in place. Every answer must also
  carry the identity digest that was verified, so a sidecar restarted with another adapter cannot
  keep answering as this one.
- **structured output is the server's, not the model's.** The sidecar returns the tutor model's
  plain prose, and this backend returns it in the registered single-field internal draft shape.
  The model is never asked to emit JSON and never selects a citation. That difference from the
  default profile is recorded in the demo configuration record.

A generation that stopped at the token cap is reported as such (``finish_reason``) instead of being
passed off as a finished answer; the Course demo assembly turns that into a visible warning.

Error mapping follows the unified classes, so the API's existing error matrix keeps working: an
unreachable sidecar is ``BackendUnavailableError`` (503), an exceeded deadline is
``BackendTimeoutError`` (504), a malformed or identity-mismatched response is
``InvalidModelOutputError`` (502), and a misconfiguration is ``BackendConfigurationError`` (503).
No message carries a prompt, a course passage, a path or a URL.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from services.inference.backends.base import (
    BackendCapabilities,
    BackendHealth,
    ChatMessage,
    HealthState,
    ModelBackend,
)
from services.inference.errors import (
    BackendConfigurationError,
    BackendTimeoutError,
    BackendUnavailableError,
    InvalidModelOutputError,
)

#: The sidecar's two paths. Nothing else is ever requested.
_HEALTH_PATH = "/health"
_GENERATE_PATH = "/generate"
_UNREACHABLE = "cannot reach the local V1B adapter sidecar; is it running and tunnelled?"
_NOT_SERVING = "the V1B adapter sidecar is not serving"
_MALFORMED_HEALTH = "the sidecar health payload is malformed"
_DEADLINE = "model request exceeded the configured total deadline"
_IDENTITY_DRIFT = "the sidecar identity changed or was not reported during the request"
_NO_EXPECTED_DIGEST = (
    "the V1B demo backend requires the expected adapter sha256 as 64 hexadecimal characters; "
    "without it a loaded adapter cannot be verified against the selected one"
)
#: The two finish reasons the sidecar reports. Anything else is not passed on as fact.
_FINISH_REASONS = frozenset({"eos", "length"})
_SHA256_CHARS = frozenset("0123456789abcdef")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= _SHA256_CHARS


class V1bDemoBackend(ModelBackend):
    """INTERNAL DEMO text backend: one sidecar, one task, no retry of a generation."""

    name = "v1b_demo"

    def __init__(
        self,
        model_id: str,
        base_url: str,
        *,
        expected_adapter_sha256: str | None = None,
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 120.0,
        deadline_seconds: float = 180.0,
        max_output_tokens: int = 256,
        max_concurrency: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        expected = (expected_adapter_sha256 or "").strip().lower()
        if not _is_sha256(expected):
            # Fail at construction, which for the API means at startup: an unpinned demo would
            # report whatever adapter happened to load as "verified", which is the one claim this
            # profile exists to make honestly.
            raise BackendConfigurationError(_NO_EXPECTED_DIGEST)
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._expected_adapter_sha256 = expected
        self._deadline_seconds = deadline_seconds
        self._max_output_tokens = max_output_tokens
        # One generation at a time: the sidecar serializes anyway, and a demo has one reviewer.
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=read_timeout_seconds,
            pool=connect_timeout_seconds,
        )
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, trust_env=False, follow_redirects=False
        )
        self._identity: dict[str, Any] | None = None
        self._identity_sha256: str | None = None
        self._identity_refusal: str | None = None

    # ------------------------------------------------------------------ helpers
    def _invalidate_identity(self, reason: str) -> None:
        """Forget a previously verified identity and remember why it was dropped.

        A verified identity is a measurement, not a standing fact: the sidecar can be restarted
        with a different adapter, or stop answering. Any failed probe therefore clears it, so the
        health payload and the UI stop reporting an identity nothing currently confirms, and no
        generation can ride on a stale verification.
        """
        self._identity = None
        self._identity_sha256 = None
        self._identity_refusal = reason

    def _accept_identity(self, payload: Any) -> dict[str, Any]:
        """Validate a sidecar health payload and remember the identity it reports."""
        try:
            if not isinstance(payload, dict) or payload.get("status") != "ready":
                raise BackendUnavailableError("the V1B adapter sidecar is not ready")
            identity = payload.get("identity")
            digest = payload.get("identity_sha256")
            if not isinstance(identity, dict) or not isinstance(digest, str) or not digest:
                raise InvalidModelOutputError(_MALFORMED_HEALTH)
            measured = str(identity.get("adapter_sha256_measured") or "").strip().lower()
            if not measured:
                raise InvalidModelOutputError("the sidecar reported no measured adapter digest")
            if measured != self._expected_adapter_sha256:
                raise BackendConfigurationError(
                    "the loaded adapter digest does not match the configured expected digest"
                )
            if identity.get("adapter_merged_into_base_weights") is not False:
                raise BackendConfigurationError("the sidecar reports a merged adapter")
            if identity.get("inference_only") is not True:
                raise BackendConfigurationError(
                    "the sidecar does not report inference-only loading"
                )
        except (BackendConfigurationError, InvalidModelOutputError, BackendUnavailableError) as exc:
            self._invalidate_identity(str(exc))
            raise
        self._identity = identity
        self._identity_sha256 = digest
        self._identity_refusal = None
        return identity

    async def _probe_identity(self) -> dict[str, Any]:
        """Read the sidecar's identity record and verify it, or raise a classified error."""
        try:
            response = await self._client.get(_HEALTH_PATH)
        except httpx.HTTPError as exc:
            self._invalidate_identity(_UNREACHABLE)
            raise BackendUnavailableError(_UNREACHABLE) from exc
        if response.status_code != 200:
            self._invalidate_identity(_NOT_SERVING)
            raise BackendUnavailableError(_NOT_SERVING)
        try:
            payload = response.json()
        except ValueError as exc:
            self._invalidate_identity(_MALFORMED_HEALTH)
            raise InvalidModelOutputError(_MALFORMED_HEALTH) from exc
        return self._accept_identity(payload)

    @property
    def identity(self) -> dict[str, Any] | None:
        """The currently verified identity, or None before the first probe and after a failure."""
        return self._identity

    @property
    def identity_sha256(self) -> str | None:
        return self._identity_sha256

    @property
    def identity_verified(self) -> bool:
        """True only while a measured identity has passed the expected-digest check."""
        return self._identity is not None and self._identity_sha256 is not None

    @property
    def identity_refusal(self) -> str | None:
        """Why the last probe refused, when it did. Content-free, safe to display."""
        return self._identity_refusal

    # ------------------------------------------------------------------ contract
    async def generate_text(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the registered demo draft from one greedy sidecar generation.

        ``response_schema`` is accepted for interface compatibility and deliberately not sent: the
        single-field draft shape is produced by this backend, not by constrained decoding inside
        the model.
        """
        del response_schema
        deadline = time.monotonic() + self._deadline_seconds
        # Identity before generation, always. Answering first and trusting that some readiness
        # probe already ran would let an unverified process publish text under the V1B label, so
        # this verifies now. A refusal leaves the identity unset, so it is re-checked (and refused
        # again) on the next request rather than being remembered as a success.
        if self._identity_sha256 is None:
            await self._probe_identity()
        expected_identity = self._identity_sha256
        if expected_identity is None:  # pragma: no cover - _probe_identity raises instead
            raise BackendConfigurationError("the V1B adapter identity is not verified")

        payload = {
            "messages": [{"role": m.role, "content": m.text} for m in messages],
            "max_new_tokens": self._max_output_tokens,
        }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BackendTimeoutError(_DEADLINE)
        try:
            async with asyncio.timeout(remaining), self._semaphore:
                response = await self._client.post(_GENERATE_PATH, json=payload)
        except TimeoutError as exc:
            raise BackendTimeoutError(_DEADLINE) from exc
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise BackendUnavailableError(_UNREACHABLE) from exc
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
            raise BackendTimeoutError("model request timed out waiting for the sidecar") from exc
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(_UNREACHABLE) from exc

        if response.status_code in (502, 503, 504):
            raise BackendUnavailableError("the V1B adapter sidecar reported it is unavailable")
        if response.status_code == 400:
            raise BackendConfigurationError("the sidecar rejected the request shape")
        if response.status_code != 200:
            raise BackendUnavailableError("the V1B adapter sidecar returned an unexpected status")
        try:
            body = response.json()
        except ValueError as exc:
            raise InvalidModelOutputError("sidecar response is not JSON") from exc
        if not isinstance(body, dict):
            raise InvalidModelOutputError("sidecar response is not a JSON object")
        answer_text = body.get("answer_text")
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise InvalidModelOutputError("sidecar response carries no answer text")
        digest = body.get("identity_sha256")
        if not isinstance(digest, str) or digest != expected_identity:
            # The sidecar reloaded with a different model or adapter between the verification and
            # this answer, or did not say which model answered; either way that answer must not be
            # published as this model's, and the verification is dropped.
            self._invalidate_identity(_IDENTITY_DRIFT)
            raise InvalidModelOutputError(_IDENTITY_DRIFT)
        draft: dict[str, Any] = {"answer_text": answer_text}
        finish_reason = body.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason in _FINISH_REASONS:
            # Carried so the server can say that an answer stopped at the token cap instead of
            # presenting a cut-off sentence as a complete one.
            draft["finish_reason"] = finish_reason
        return draft

    async def analyze_image(
        self,
        image_bytes: bytes,
        media_type: str,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del image_bytes, media_type, prompt, response_schema
        raise BackendConfigurationError("the V1B demo backend serves text only")

    def capabilities(self) -> BackendCapabilities:
        # structured_output is declared because THIS BACKEND returns the registered internal draft
        # object; the model it fronts produces prose. The distinction is recorded in the demo
        # configuration record rather than hidden here.
        return BackendCapabilities(text=True, vision=False, structured_output=True)

    async def health(self) -> BackendHealth:
        """Probe the sidecar and verify the adapter identity. Never raises."""
        try:
            identity = await self._probe_identity()
        except (BackendConfigurationError, InvalidModelOutputError, BackendUnavailableError) as exc:
            return BackendHealth(state=HealthState.UNAVAILABLE, backend=self.name, detail=str(exc))
        measured = str(identity.get("adapter_sha256_measured"))
        return BackendHealth(
            state=HealthState.READY,
            backend=self.name,
            detail=f"V1B adapter verified by digest {measured[:12]}…",
        )

    async def close(self) -> None:
        await self._client.aclose()
