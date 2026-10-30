"""FastAPI application factory with one ordered request-safety boundary."""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from services.api.app.config.settings import Settings, get_settings
from services.api.app.errors import (
    ApplicationError,
    http_error_spec,
    render_request_error,
    safe_http_exception_headers,
)
from services.api.app.middleware import RequestBoundaryMiddleware, RequestFramingPolicy
from services.api.app.models.errors import ErrorDetail
from services.api.app.routes import course, health
from services.api.app.security import SecurityPolicy, load_local_token
from services.inference.backends.factory import create_backend, validate_runtime_capabilities
from services.inference.errors import (
    BackendConfigurationError,
    BackendNotConfiguredError,
    BackendTimeoutError,
    BackendUnavailableError,
    InvalidModelOutputError,
    UnsupportedCapabilityError,
)

_SAFE_FIELD_SEGMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z", re.ASCII)
_VALIDATION_REASONS = {
    "missing": "field is required",
    "extra_forbidden": "unexpected field is not allowed",
    "literal_error": "value is not an allowed option",
    "enum": "value is not an allowed option",
    "string_too_short": "value has an invalid length",
    "string_too_long": "value has an invalid length",
    "too_short": "value has an invalid length",
    "too_long": "value has an invalid length",
}


def _safe_validation_details(errors: list[dict[str, Any]]) -> list[ErrorDetail]:
    details: list[ErrorDetail] = []
    for error in errors:
        error_type = str(error.get("type", ""))
        raw_location = list(error.get("loc", ()))
        location: list[str] = []
        for index, part in enumerate(raw_location):
            if isinstance(part, int):
                location.append(str(part))
            elif error_type == "extra_forbidden" and index == len(raw_location) - 1:
                location.append("<unexpected-field>")
            elif isinstance(part, str) and _SAFE_FIELD_SEGMENT.fullmatch(part):
                location.append(part)
            else:
                location.append("<field>")
        details.append(
            ErrorDetail(
                field_path=".".join(location) or "request",
                reason=_VALIDATION_REASONS.get(error_type, "value is invalid"),
            )
        )
    return details


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    validate_runtime_capabilities(settings.text_base_url, settings.vision_base_url)
    local_api_token = load_local_token(settings)
    security_policy = SecurityPolicy.from_settings(settings, local_api_token)
    framing_policy = RequestFramingPolicy.from_settings(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.text_backend.close()
        await app.state.vision_backend.close()

    app = FastAPI(title="CSWA AI Tutor API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.security_policy = security_policy
    app.state.framing_policy = framing_policy
    app.state.text_backend = create_backend(
        settings.text_backend,
        model_id=settings.text_model_id,
        base_url=settings.text_base_url,
        connect_timeout_seconds=settings.model_connect_timeout_seconds,
        read_timeout_seconds=settings.model_read_timeout_seconds,
        deadline_seconds=settings.model_text_deadline_seconds,
        max_attempts=settings.model_max_attempts,
        max_output_tokens=settings.model_max_output_tokens,
        max_concurrency=settings.model_max_concurrency,
        # Only the INTERNAL DEMO backend reads this; it refuses to be constructed without it, and
        # refuses to report ready or to answer unless the adapter it loaded measures to it.
        expected_adapter_sha256=settings.demo_expected_adapter_sha256,
    )
    app.state.vision_backend = create_backend(
        settings.vision_backend,
        model_id=settings.vision_model_id,
        base_url=settings.vision_base_url,
        connect_timeout_seconds=settings.model_connect_timeout_seconds,
        read_timeout_seconds=settings.model_read_timeout_seconds,
        deadline_seconds=settings.model_vision_deadline_seconds,
        max_attempts=settings.model_max_attempts,
        max_output_tokens=settings.model_max_output_tokens,
        max_concurrency=settings.model_max_concurrency,
    )

    app.add_middleware(
        RequestBoundaryMiddleware,
        security_policy=security_policy,
        framing_policy=framing_policy,
    )
    app.include_router(health.router)
    app.include_router(course.router)

    @app.exception_handler(ApplicationError)
    async def application_error_handler(request: Request, exc: ApplicationError) -> JSONResponse:
        return render_request_error(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=type(exc).default_message,
            retryable=exc.retryable,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        if any(error.get("type") == "json_invalid" for error in errors):
            return render_request_error(
                request,
                status_code=400,
                code="malformed_json",
                message="the request body is not valid JSON",
                retryable=False,
            )
        return render_request_error(
            request,
            status_code=422,
            code="contract_validation_error",
            message="request does not match the expected contract",
            retryable=False,
            details=_safe_validation_details(errors),
        )

    @app.exception_handler(InvalidModelOutputError)
    async def invalid_model_output_handler(
        request: Request, exc: InvalidModelOutputError
    ) -> JSONResponse:
        return render_request_error(
            request,
            status_code=502,
            code=exc.code,
            message="the model produced output that failed validation",
            retryable=False,
        )

    @app.exception_handler(BackendTimeoutError)
    async def backend_timeout_handler(request: Request, exc: BackendTimeoutError) -> JSONResponse:
        return render_request_error(
            request,
            status_code=504,
            code=exc.code,
            message="the model request exceeded the configured deadline",
            retryable=True,
        )

    @app.exception_handler(BackendUnavailableError)
    async def backend_unavailable_handler(
        request: Request, exc: BackendUnavailableError
    ) -> JSONResponse:
        return render_request_error(
            request,
            status_code=503,
            code=exc.code,
            message="the configured model backend is unavailable",
            retryable=True,
        )

    @app.exception_handler(BackendNotConfiguredError)
    async def backend_not_configured_handler(
        request: Request, exc: BackendNotConfiguredError
    ) -> JSONResponse:
        # The explicit no-model state (review D1A-02): alive but not serving
        # model-dependent requests; distinct from unavailable and mismatch.
        return render_request_error(
            request,
            status_code=503,
            code=exc.code,
            message="no model backend is configured on this installation",
            retryable=False,
        )

    @app.exception_handler(BackendConfigurationError)
    async def backend_configuration_handler(
        request: Request, exc: BackendConfigurationError
    ) -> JSONResponse:
        return render_request_error(
            request,
            status_code=503,
            code=exc.code,
            message="the model backend is misconfigured",
            retryable=False,
        )

    @app.exception_handler(UnsupportedCapabilityError)
    async def unsupported_capability_handler(
        request: Request, exc: UnsupportedCapabilityError
    ) -> JSONResponse:
        return render_request_error(
            request,
            status_code=503,
            code=exc.code,
            message="the configured model backend does not support this capability",
            retryable=False,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        spec = http_error_spec(exc.status_code)
        headers = safe_http_exception_headers(exc.status_code, exc.headers)
        if exc.status_code == 413:
            headers["Connection"] = "close"
        return render_request_error(
            request,
            status_code=exc.status_code,
            code=spec.code,
            message=spec.message,
            retryable=spec.retryable,
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        del exc
        return render_request_error(
            request,
            status_code=500,
            code="internal_error",
            message="an internal error occurred",
            retryable=False,
        )

    return app
