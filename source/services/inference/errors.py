"""Unified inference-layer errors.

Business services and API routes must depend on these errors only, never on
provider-specific exceptions such as httpx errors.
"""


class InferenceError(Exception):
    """Base class for all model-backend errors."""

    code: str = "inference_error"
    retryable: bool = False

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BackendConfigurationError(InferenceError):
    """The backend configuration is invalid; raised at startup, not per request."""

    code = "backend_configuration_error"
    retryable = False


class BackendNotConfiguredError(InferenceError):
    """No model backend is configured (the explicit 'disabled' state).

    Raised when a model-dependent operation reaches a deployment that has
    deliberately not configured any model. Distinct from
    ``BackendUnavailableError`` (a configured provider that cannot be
    reached) and from ``UnsupportedCapabilityError`` (a configured provider
    that lacks a required capability). Maps to HTTP 503 (review D1A).
    """

    code = "model_not_configured"
    retryable = False


class BackendUnavailableError(InferenceError):
    """The configured provider cannot be reached or is not ready."""

    code = "model_unavailable"
    retryable = True


class BackendTimeoutError(InferenceError):
    """The model request exceeded the configured total deadline."""

    code = "model_timeout"
    retryable = True


class UnsupportedCapabilityError(InferenceError):
    """The backend does not support the requested capability (e.g. vision)."""

    code = "unsupported_capability"
    retryable = False


class InvalidModelOutputError(InferenceError):
    """The provider returned output that failed strict structured validation."""

    code = "invalid_model_output"
    retryable = False
