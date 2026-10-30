"""Typed errors and the single public ``ApiError`` renderer.

The renderer is shared by FastAPI exception handlers and the outer ASGI
request boundary.  Request-controlled exception text and unsafe request IDs
never cross this module's public-response boundary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from services.api.app.models.errors import ApiError, ErrorDetail

_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}\Z", re.ASCII)
_REQUEST_ID_HEADER = b"x-request-id"
_STANDARD_HTTP_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"}
)


class ApplicationError(Exception):
    """Base class for typed internal failures with a stable public mapping."""

    status_code: int = 500
    code: str = "internal_error"
    retryable: bool = False
    default_message: str = "an internal error occurred"

    def __init__(self, message: str | None = None) -> None:
        # The instance text remains useful to internal callers, but response
        # handlers always publish ``default_message`` rather than this value.
        self.message = message or self.default_message
        super().__init__(self.message)


class AuthenticationError(ApplicationError):
    status_code = 401
    code = "invalid_local_token"
    default_message = "missing or invalid local API token"


class ForbiddenError(ApplicationError):
    status_code = 403
    code = "forbidden"
    default_message = "this operation is not enabled on this installation"


class InstructorModeDisabledError(ForbiddenError):
    code = "instructor_mode_disabled"
    default_message = "instructor mode is not enabled in this local configuration"


class NotFoundError(ApplicationError):
    status_code = 404
    code = "not_found"
    default_message = "the requested resource does not exist"


class RequestTooLargeError(ApplicationError):
    status_code = 413
    code = "request_too_large"
    default_message = "the request body exceeds the configured size limit"


class MalformedRequestError(ApplicationError):
    status_code = 400
    code = "malformed_request"
    default_message = "the request could not be parsed"


class InternalContractError(ApplicationError):
    """A valid internal draft was assembled into an invalid public contract."""

    status_code = 500
    code = "internal_contract_error"
    default_message = "the server assembled an invalid response contract"


@dataclass(frozen=True, slots=True)
class ErrorSpec:
    code: str
    message: str
    retryable: bool = False


_HTTP_ERROR_SPECS: dict[int, ErrorSpec] = {
    400: ErrorSpec("malformed_request", "the request could not be parsed"),
    401: ErrorSpec("invalid_local_token", "missing or invalid local API token"),
    403: ErrorSpec("forbidden", "the request is not allowed"),
    404: ErrorSpec("not_found", "the requested resource does not exist"),
    405: ErrorSpec("method_not_allowed", "the requested method is not allowed"),
    413: ErrorSpec("request_too_large", "the request body exceeds the configured size limit"),
    415: ErrorSpec("unsupported_media_type", "the request media type is not supported"),
    422: ErrorSpec("contract_validation_error", "request does not match the expected contract"),
    500: ErrorSpec("internal_error", "an internal error occurred"),
    502: ErrorSpec("bad_gateway", "an upstream service returned an invalid response"),
    503: ErrorSpec("http_503", "the service is unavailable"),
    504: ErrorSpec("gateway_timeout", "an upstream service exceeded its deadline", True),
}


def raw_header_values(scope: Scope, name: bytes) -> list[bytes]:
    """Return every raw value for a case-insensitive ASGI header name."""
    return [value for header_name, value in scope.get("headers", []) if header_name.lower() == name]


def _new_request_id() -> str:
    return str(uuid4())


def initialize_request_id(scope: Scope) -> str:
    """Select one safe request ID and persist it in ASGI scope state."""
    values = raw_header_values(scope, _REQUEST_ID_HEADER)
    request_id: str | None = None
    if len(values) == 1:
        try:
            candidate = values[0].decode("ascii")
        except UnicodeDecodeError:
            candidate = ""
        if _REQUEST_ID_PATTERN.fullmatch(candidate):
            request_id = candidate
    if request_id is None:
        request_id = _new_request_id()
    state = scope.setdefault("state", {})
    state["request_id"] = request_id
    return request_id


def request_id_from_scope(scope: Scope) -> str:
    state = scope.get("state")
    if isinstance(state, dict):
        request_id = state.get("request_id")
        if isinstance(request_id, str) and _REQUEST_ID_PATTERN.fullmatch(request_id):
            return request_id
    return initialize_request_id(scope)


def request_id_from_request(request: Request) -> str:
    return request_id_from_scope(request.scope)


def render_api_error(
    *,
    request_id: str,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
    details: Sequence[ErrorDetail] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build the only public API error representation."""
    body = ApiError(
        request_id=request_id,
        code=code,
        message=message,
        retryable=retryable,
        details=list(details or ()),
    )
    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=response_headers,
    )


def render_request_error(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
    details: Sequence[ErrorDetail] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return render_api_error(
        request_id=request_id_from_request(request),
        status_code=status_code,
        code=code,
        message=message,
        retryable=retryable,
        details=details,
        headers=headers,
    )


async def send_api_error(
    scope: Scope,
    receive: Receive,
    send: Send,
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool = False,
    details: Sequence[ErrorDetail] | None = None,
    headers: Mapping[str, str] | None = None,
) -> None:
    response = render_api_error(
        request_id=request_id_from_scope(scope),
        status_code=status_code,
        code=code,
        message=message,
        retryable=retryable,
        details=details,
        headers=headers,
    )
    await response(scope, receive, send)


def http_error_spec(status_code: int) -> ErrorSpec:
    return _HTTP_ERROR_SPECS.get(
        status_code,
        ErrorSpec(f"http_{status_code}", "the request could not be completed"),
    )


def safe_http_exception_headers(
    status_code: int,
    headers: Mapping[str, str] | None,
) -> dict[str, str]:
    """Pass only protocol-required, syntax-checked exception headers."""
    if status_code != 405 or not headers:
        return {}
    allow = next((value for name, value in headers.items() if name.lower() == "allow"), None)
    if allow is not None:
        methods = [method.strip() for method in allow.split(",")]
        if methods and all(method in _STANDARD_HTTP_METHODS for method in methods):
            return {"Allow": ", ".join(methods)}
    return {}
