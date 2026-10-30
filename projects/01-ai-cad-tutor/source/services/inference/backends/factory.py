"""Backend factory.

Provider branching lives here and only here. Business services receive a
``ModelBackend`` through dependency injection and must never test which
provider they are talking to.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from services.inference.backends.base import ModelBackend
from services.inference.backends.disabled_backend import DisabledBackend
from services.inference.backends.mock_backend import MockBackend
from services.inference.backends.ollama_backend import OllamaBackend
from services.inference.backends.v1b_demo_backend import V1bDemoBackend
from services.inference.backends.vllm_backend import VllmBackend
from services.inference.errors import BackendConfigurationError

KNOWN_BACKENDS = ("disabled", "mock", "ollama", "v1b_demo", "vllm")
_REMOTE_RUNTIME_ERROR = "remote model execution is not supported by this runtime"


def _base_url_is_loopback(base_url: str) -> bool:
    try:
        hostname = urlsplit(base_url).hostname or ""
    except ValueError:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_runtime_capabilities(*base_urls: str) -> None:
    """Fail before provider construction while remote auth/CA transport is absent."""
    if any(not _base_url_is_loopback(base_url) for base_url in base_urls):
        raise BackendConfigurationError(_REMOTE_RUNTIME_ERROR)


def create_backend(
    backend_name: str,
    *,
    model_id: str,
    base_url: str,
    connect_timeout_seconds: float = 5.0,
    read_timeout_seconds: float = 30.0,
    deadline_seconds: float = 60.0,
    max_attempts: int = 2,
    max_output_tokens: int = 2048,
    max_concurrency: int = 2,
    expected_adapter_sha256: str | None = None,
) -> ModelBackend:
    """Create a backend from configuration; unknown names fail loudly at startup."""
    if backend_name not in KNOWN_BACKENDS:
        raise BackendConfigurationError(
            f"unknown model backend '{backend_name}'; expected one of {KNOWN_BACKENDS}"
        )
    if backend_name == "disabled":
        # The explicit no-model state (review D1A-02): a legal configuration
        # that never constructs a provider, never validates a model URL, and
        # never falls back to Mock.
        return DisabledBackend()
    validate_runtime_capabilities(base_url)
    if backend_name == "mock":
        return MockBackend()
    if backend_name == "ollama":
        return OllamaBackend(
            model_id=model_id,
            base_url=base_url,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
            deadline_seconds=deadline_seconds,
            max_attempts=max_attempts,
            max_output_tokens=max_output_tokens,
            max_concurrency=max_concurrency,
        )
    if backend_name == "v1b_demo":
        # The INTERNAL DEMO adapter sidecar. The loopback check above already ran, so a remote
        # sidecar must be reached through the project's existing private tunnel, not directly.
        return V1bDemoBackend(
            model_id=model_id,
            base_url=base_url,
            expected_adapter_sha256=expected_adapter_sha256,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
            deadline_seconds=deadline_seconds,
            max_output_tokens=max_output_tokens,
            max_concurrency=1,
        )
    if backend_name == "vllm":
        return VllmBackend(
            model_id=model_id,
            base_url=base_url,
            connect_timeout_seconds=connect_timeout_seconds,
            deadline_seconds=deadline_seconds,
        )
    raise AssertionError("known backend dispatch is incomplete")
