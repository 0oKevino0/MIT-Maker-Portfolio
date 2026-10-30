"""Strict backend capability contract (review D1A-04).

Capabilities are DECLARED by each backend implementation through
``ModelBackend.capabilities()`` — routes and services never guess them from a
backend or model name. This module is the single verification point:

- :func:`validate_capability_object` enforces the strict shape: the value must
  be exactly the shared :class:`BackendCapabilities` dataclass (no subclass,
  no extra fields, no string/int stand-ins for the boolean flags).
- :func:`require_course_capabilities` gates the Course request path BEFORE any
  evidence handling or generation call.

Classification is fail-closed and keeps distinct meanings (D1A-06):

- malformed capability declaration  -> ``BackendConfigurationError``  (503)
- no capabilities at all            -> ``BackendNotConfiguredError``  (503)
- required capability missing       -> ``UnsupportedCapabilityError`` (503)

None of the raised messages contains a model name, URL, or provider detail.
"""

from __future__ import annotations

from services.inference.backends.base import BackendCapabilities, ModelBackend
from services.inference.errors import (
    BackendConfigurationError,
    BackendNotConfiguredError,
    UnsupportedCapabilityError,
)

# The Course text path requires exactly these declared capabilities today:
# text generation with constrained/structured JSON output. Vision is declared
# and detected by the contract but not required (and not implemented) here.
COURSE_REQUIRED_CAPABILITIES: tuple[str, ...] = ("text", "structured_output")

_CAPABILITY_FLAGS: tuple[str, ...] = ("text", "vision", "structured_output")


def validate_capability_object(capabilities: object) -> BackendCapabilities:
    """Return a strictly valid ``BackendCapabilities`` or fail closed.

    Exactly the shared dataclass type is accepted — a subclass could carry
    extra fields, and a duck-typed object could carry coerced flag values.
    """
    if type(capabilities) is not BackendCapabilities:
        raise BackendConfigurationError("backend capability declaration has an invalid type")
    for flag in _CAPABILITY_FLAGS:
        if not isinstance(getattr(capabilities, flag), bool):
            raise BackendConfigurationError("backend capability flags must be strict booleans")
    if not isinstance(capabilities.max_images, int) or isinstance(capabilities.max_images, bool):
        raise BackendConfigurationError("backend capability max_images must be an integer")
    media_types = capabilities.accepted_media_types
    if not isinstance(media_types, tuple) or not all(isinstance(item, str) for item in media_types):
        raise BackendConfigurationError(
            "backend capability accepted_media_types must be a tuple of strings"
        )
    return capabilities


def declared_capabilities(backend: ModelBackend) -> BackendCapabilities:
    """Fetch and strictly validate a backend's declared capabilities."""
    try:
        declared = backend.capabilities()
    except Exception:
        # A capability probe must never take the service down or leak the
        # provider's exception; an undeclarable backend is a config defect.
        raise BackendConfigurationError("backend capability declaration failed") from None
    return validate_capability_object(declared)


def require_course_capabilities(backend: ModelBackend) -> None:
    """Gate for the Course request path; called before any generation."""
    capabilities = declared_capabilities(backend)
    if not any(getattr(capabilities, flag) for flag in _CAPABILITY_FLAGS):
        raise BackendNotConfiguredError("no model backend capability is configured")
    if not all(getattr(capabilities, flag) for flag in COURSE_REQUIRED_CAPABILITIES):
        raise UnsupportedCapabilityError(
            "the configured backend does not declare the capabilities required for course answering"
        )
