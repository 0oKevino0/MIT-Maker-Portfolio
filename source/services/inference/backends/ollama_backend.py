"""Ollama adapter with typed errors, one deadline budget, and bounded retry.

Talks to a local Ollama server over plain HTTP (no OpenAI/Anthropic SDKs).
The text path is implemented; the vision path is an explicit stub until a
local vision-language model is wired in (Day 9 of the development plan).

Behavior contract (review issue I-06, ADR 0001 section 13.6):

- connect and read timeouts are configured separately;
- all attempts share one total wall-clock deadline; exceeding it raises
  ``BackendTimeoutError`` (mapped to HTTP 504 by the API layer);
- at most ``max_attempts`` (<= 2) attempts, retrying only transient classes:
  connection failures, interrupted connections, and provider 429/502/503/504,
  with a short jittered backoff that never exceeds the remaining budget;
- provider 4xx (except 404/429) is a configuration error and is not retried;
  invalid JSON content is never retried;
- error and health messages never contain prompts, course content, tokens,
  or the configured base URL.
"""

from __future__ import annotations

import asyncio
import json
import random
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
    UnsupportedCapabilityError,
)

_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


class OllamaBackend(ModelBackend):
    name = "ollama"

    def __init__(
        self,
        model_id: str,
        base_url: str,
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 30.0,
        deadline_seconds: float = 60.0,
        max_attempts: int = 2,
        max_output_tokens: int = 2048,
        max_concurrency: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._deadline_seconds = deadline_seconds
        self._max_attempts = max(1, min(2, max_attempts))
        self._max_output_tokens = max_output_tokens
        self._semaphore = asyncio.Semaphore(max_concurrency)
        timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=read_timeout_seconds,
            pool=connect_timeout_seconds,
        )
        self._client = client or httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def generate_text(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [{"role": m.role, "content": m.text} for m in messages],
            "stream": False,
            # Ollama accepts "json" or a JSON Schema object in the "format" field.
            "format": response_schema if response_schema is not None else "json",
            "options": {"num_predict": self._max_output_tokens},
        }

        deadline = time.monotonic() + self._deadline_seconds
        last_transient: InferenceTransient | None = None

        for attempt in range(1, self._max_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BackendTimeoutError("model request exceeded the configured total deadline")
            try:
                async with asyncio.timeout(remaining):
                    async with self._semaphore:
                        response = await self._client.post("/api/chat", json=payload)
            except TimeoutError as exc:
                # Total budget exhausted; never retried.
                raise BackendTimeoutError(
                    "model request exceeded the configured total deadline"
                ) from exc
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_transient = InferenceTransient(
                    BackendUnavailableError(
                        "cannot reach the local Ollama server; is 'ollama serve' running?"
                    ),
                    exc,
                )
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
                raise BackendTimeoutError(
                    "model request timed out waiting for the Ollama server"
                ) from exc
            except httpx.RemoteProtocolError as exc:
                last_transient = InferenceTransient(
                    BackendUnavailableError("the connection to Ollama was interrupted"),
                    exc,
                )
            except httpx.HTTPError as exc:
                raise BackendUnavailableError(
                    "a transport error occurred while talking to Ollama"
                ) from exc
            else:
                status = response.status_code
                if status == 200:
                    return self._parse_structured_content(response)
                if status == 404:
                    raise BackendUnavailableError(
                        f"Ollama model '{self._model_id}' was not found; pull it first"
                    )
                if status in _RETRYABLE_STATUS_CODES:
                    last_transient = InferenceTransient(
                        BackendUnavailableError(
                            f"Ollama returned a transient error (HTTP {status})"
                        ),
                        None,
                    )
                elif status in (401, 403):
                    raise BackendConfigurationError(
                        "the Ollama server rejected the request credentials"
                    )
                elif 400 <= status < 500:
                    raise BackendConfigurationError(
                        f"the Ollama server rejected the request (HTTP {status})"
                    )
                else:
                    raise BackendUnavailableError(
                        f"the Ollama server returned an error (HTTP {status})"
                    )

            if attempt < self._max_attempts:
                backoff = random.uniform(0.05, 0.25) * attempt
                backoff = min(backoff, max(0.0, deadline - time.monotonic()))
                if backoff > 0:
                    await asyncio.sleep(backoff)

        assert last_transient is not None
        if last_transient.cause is not None:
            raise last_transient.error from last_transient.cause
        raise last_transient.error

    async def analyze_image(
        self,
        image_bytes: bytes,
        media_type: str,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise UnsupportedCapabilityError(
            "OllamaBackend vision path is not implemented in the bootstrap; "
            "a local vision-language model adapter is planned for Day 9"
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            text=True,
            vision=False,
            structured_output=True,
            max_images=0,
        )

    async def health(self) -> BackendHealth:
        """Probe readiness; never raises and never reveals the base URL."""
        try:
            response = await self._client.get("/api/tags")
        except httpx.HTTPError:
            return BackendHealth(
                state=HealthState.UNAVAILABLE,
                backend=self.name,
                detail="cannot reach the configured Ollama server",
            )
        try:
            if response.status_code != 200:
                return BackendHealth(
                    state=HealthState.UNAVAILABLE,
                    backend=self.name,
                    detail=f"Ollama model listing returned HTTP {response.status_code}",
                )
            body = response.json()
            models = body.get("models", []) if isinstance(body, dict) else None
            if not isinstance(models, list):
                return BackendHealth(
                    state=HealthState.DEGRADED,
                    backend=self.name,
                    detail="Ollama model listing had an unexpected shape",
                )
            names = [
                entry.get("name", "")
                for entry in models
                if isinstance(entry, dict) and isinstance(entry.get("name", ""), str)
            ]
            if not any(name.startswith(self._model_id) for name in names):
                return BackendHealth(
                    state=HealthState.DEGRADED,
                    backend=self.name,
                    detail=f"model '{self._model_id}' is not pulled",
                )
            return BackendHealth(state=HealthState.READY, backend=self.name)
        except Exception:
            return BackendHealth(
                state=HealthState.DEGRADED,
                backend=self.name,
                detail="Ollama health response could not be interpreted",
            )

    async def close(self) -> None:
        await self._client.aclose()

    def _parse_structured_content(self, response: httpx.Response) -> dict[str, Any]:
        """Strictly parse model content as a JSON object; never repair silently.

        Error messages never include the model content, prompts, or course text.
        """
        try:
            body = response.json()
            content = body["message"]["content"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise InvalidModelOutputError(
                "Ollama response body did not match the expected chat structure"
            ) from exc
        if not isinstance(content, str):
            raise InvalidModelOutputError("Ollama message content is not a string")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise InvalidModelOutputError(
                "model output is not valid JSON; refusing to accept it"
            ) from exc
        if not isinstance(parsed, dict):
            raise InvalidModelOutputError("model output is valid JSON but not a JSON object")
        return parsed


class InferenceTransient:
    """Internal holder for the last transient failure across retry attempts."""

    __slots__ = ("cause", "error")

    def __init__(self, error: BackendUnavailableError, cause: Exception | None) -> None:
        self.error = error
        self.cause = cause
