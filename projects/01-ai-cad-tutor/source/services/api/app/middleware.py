"""One ASGI request boundary for security, framing, CORS, and body limits."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from starlette._utils import get_route_path
from starlette.exceptions import HTTPException
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from services.api.app.errors import initialize_request_id, raw_header_values, send_api_error
from services.api.app.security import SecurityPolicy, SecurityPolicyError

if TYPE_CHECKING:
    from services.api.app.config.settings import Settings

_COURSE_ASK_PATH = "/api/v1/course/ask"
_MAX_CONTENT_LENGTH_DIGITS = 19
_MAX_CONTENT_LENGTH_VALUE = b"9223372036854775807"
_ALLOWED_CORS_METHODS = frozenset({"GET", "POST"})
_ALLOWED_CORS_HEADERS = frozenset({"content-type", "x-request-id", "x-cswa-local-token"})
_MEDIA_TYPE_TOKEN = re.compile(r"[!#$%&'*+.0-9A-Z^_`a-z|~-]+\Z", re.ASCII)


class RequestBoundaryError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ParsedFraming:
    content_length: int | None
    chunked: bool


@dataclass(frozen=True, slots=True)
class BodyPolicy:
    max_bytes: int
    media_kind: str

    def accepts(self, raw_content_type: bytes) -> bool:
        if b"," in raw_content_type:
            return False
        try:
            value = raw_content_type.decode("ascii")
        except UnicodeDecodeError:
            return False
        parts = value.split(";")
        raw_media_type = parts[0].strip()
        if raw_media_type.count("/") != 1:
            return False
        raw_type, raw_subtype = raw_media_type.split("/", 1)
        if not _MEDIA_TYPE_TOKEN.fullmatch(raw_type) or not _MEDIA_TYPE_TOKEN.fullmatch(
            raw_subtype
        ):
            return False
        type_name = raw_type.casefold()
        subtype_name = raw_subtype.casefold()
        media_type = f"{type_name}/{subtype_name}"
        if self.media_kind == "json":
            is_json = type_name == "application" and (
                subtype_name == "json"
                or (subtype_name.endswith("+json") and len(subtype_name) > len("+json"))
            )
            if not is_json:
                return False
            parameters: dict[str, str] = {}
            for raw_parameter in parts[1:]:
                if "=" not in raw_parameter:
                    return False
                name, raw_value = raw_parameter.split("=", 1)
                name = name.strip()
                if not _MEDIA_TYPE_TOKEN.fullmatch(name):
                    return False
                name = name.casefold()
                parameter_value = raw_value.strip()
                if '"' in parameter_value:
                    if (
                        len(parameter_value) < 2
                        or not parameter_value.startswith('"')
                        or not parameter_value.endswith('"')
                        or parameter_value.count('"') != 2
                    ):
                        return False
                    parameter_value = parameter_value[1:-1]
                parameter_value = parameter_value.casefold()
                if name in parameters:
                    return False
                parameters[name] = parameter_value
            return not parameters or parameters == {"charset": "utf-8"}
        if self.media_kind == "multipart":
            return media_type == "multipart/form-data"
        return False


class RequestFramingPolicy:
    """Strict raw framing parser plus method/path-specific body policies."""

    def __init__(self, *, max_json_body_bytes: int, max_multipart_body_bytes: int) -> None:
        self.json_body = BodyPolicy(max_json_body_bytes, "json")
        self.multipart_body = BodyPolicy(max_multipart_body_bytes, "multipart")
        self._route_policies = {("POST", _COURSE_ASK_PATH): self.json_body}

    @classmethod
    def from_settings(cls, settings: Settings) -> RequestFramingPolicy:
        return cls(
            max_json_body_bytes=settings.api_max_json_body_bytes,
            max_multipart_body_bytes=settings.api_max_multipart_body_bytes,
        )

    def for_request(self, method: str, path: str) -> BodyPolicy | None:
        return self._route_policies.get((method.upper(), path))

    def parse(self, scope: Scope) -> ParsedFraming:
        content_lengths = raw_header_values(scope, b"content-length")
        transfer_encodings = raw_header_values(scope, b"transfer-encoding")
        if len(content_lengths) > 1 or len(transfer_encodings) > 1:
            raise RequestBoundaryError(
                400, "malformed_request", "the request has ambiguous framing headers"
            )
        if content_lengths and transfer_encodings:
            raise RequestBoundaryError(
                400, "malformed_request", "the request has ambiguous framing headers"
            )

        chunked = False
        if transfer_encodings:
            if transfer_encodings[0] != b"chunked":
                raise RequestBoundaryError(
                    400, "malformed_request", "the request has an unsupported transfer encoding"
                )
            chunked = True

        declared: int | None = None
        if content_lengths:
            raw_length = content_lengths[0]
            if not _is_canonical_content_length(raw_length):
                raise RequestBoundaryError(
                    400, "malformed_request", "the request has an invalid Content-Length header"
                )
            declared = int(raw_length)
        return ParsedFraming(content_length=declared, chunked=chunked)


def _is_canonical_content_length(value: bytes) -> bool:
    if not value or len(value) > _MAX_CONTENT_LENGTH_DIGITS:
        return False
    if value == b"0":
        return True
    if value[:1] < b"1" or value[:1] > b"9":
        return False
    if any(byte < ord("0") or byte > ord("9") for byte in value[1:]):
        return False
    return not (len(value) == _MAX_CONTENT_LENGTH_DIGITS and value > _MAX_CONTENT_LENGTH_VALUE)


def _single_content_type(scope: Scope) -> bytes:
    values = raw_header_values(scope, b"content-type")
    if len(values) > 1:
        raise RequestBoundaryError(
            400, "malformed_request", "the request has duplicate Content-Type headers"
        )
    if not values:
        raise RequestBoundaryError(
            415, "unsupported_media_type", "the request media type is not supported"
        )
    return values[0]


def _cors_header_names(scope: Scope) -> set[str]:
    values = raw_header_values(scope, b"access-control-request-headers")
    if len(values) > 1:
        raise RequestBoundaryError(
            400, "malformed_request", "the request has duplicate preflight headers"
        )
    if not values:
        return set()
    try:
        raw_names = values[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise RequestBoundaryError(
            403, "cors_preflight_rejected", "the CORS preflight request is not allowed"
        ) from exc
    names = {name.strip().casefold() for name in raw_names.split(",")}
    if "" in names:
        raise RequestBoundaryError(
            403, "cors_preflight_rejected", "the CORS preflight request is not allowed"
        )
    return names


def _preflight_method(scope: Scope) -> str | None:
    values = raw_header_values(scope, b"access-control-request-method")
    if len(values) > 1:
        raise RequestBoundaryError(
            400, "malformed_request", "the request has duplicate preflight headers"
        )
    if not values:
        return None
    try:
        return values[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise RequestBoundaryError(
            403, "cors_preflight_rejected", "the CORS preflight request is not allowed"
        ) from exc


def _append_vary_origin(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    vary_tokens: list[str] = []
    retained: list[tuple[bytes, bytes]] = []
    for name, value in headers:
        if name.lower() == b"vary":
            vary_tokens.extend(token.strip() for token in value.decode("latin-1").split(","))
        else:
            retained.append((name, value))
    if not any(token.casefold() == "origin" for token in vary_tokens):
        vary_tokens.append("Origin")
    retained.append((b"vary", ", ".join(token for token in vary_tokens if token).encode("ascii")))
    return retained


def _replace_header(
    headers: Iterable[tuple[bytes, bytes]], name: bytes, value: bytes
) -> list[tuple[bytes, bytes]]:
    retained = [(key, item) for key, item in headers if key.lower() != name]
    retained.append((name, value))
    return retained


class RequestBoundaryMiddleware:
    """Enforce the fixed request order before endpoint or backend work."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        security_policy: SecurityPolicy,
        framing_policy: RequestFramingPolicy,
    ) -> None:
        self.app = app
        self._security = security_policy
        self._framing = framing_policy

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = initialize_request_id(scope)
        allowed_origin: str | None = None
        response_started = False

        async def boundary_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = _replace_header(
                    message.get("headers", []), b"x-request-id", request_id.encode("ascii")
                )
                if allowed_origin is not None:
                    headers = _replace_header(
                        headers,
                        b"access-control-allow-origin",
                        allowed_origin.encode("ascii"),
                    )
                    headers = _append_vary_origin(headers)
                message = dict(message)
                message["headers"] = headers
            await send(message)

        try:
            # Frozen order: request ID -> Host/framing -> Origin/CORS -> token
            # -> content type/body limit -> endpoint/backend.
            self._security.validate_host(scope)
            framing = self._framing.parse(scope)
            allowed_origin = self._security.validate_origin(scope)
            route_path = get_route_path(scope)

            preflight_method = (
                _preflight_method(scope) if scope.get("method") == "OPTIONS" else None
            )
            if allowed_origin is not None and preflight_method is not None:
                requested_headers = _cors_header_names(scope)
                if preflight_method not in _ALLOWED_CORS_METHODS or not requested_headers.issubset(
                    _ALLOWED_CORS_HEADERS
                ):
                    raise RequestBoundaryError(
                        403,
                        "cors_preflight_rejected",
                        "the CORS preflight request is not allowed",
                    )
                response = PlainTextResponse(
                    "OK",
                    status_code=200,
                    headers={
                        "Access-Control-Allow-Methods": "GET, POST",
                        "Access-Control-Allow-Headers": (
                            "Content-Type, X-Request-ID, X-CSWA-Local-Token"
                        ),
                        "Access-Control-Max-Age": "600",
                    },
                )
                await response(scope, receive, boundary_send)
                return

            if self._security.requires_auth(scope.get("method", ""), route_path):
                self._security.authenticate(scope)

            body_policy = self._framing.for_request(scope.get("method", ""), route_path)
            counted_receive = receive
            if body_policy is not None:
                content_type = _single_content_type(scope)
                if not body_policy.accepts(content_type):
                    raise RequestBoundaryError(
                        415,
                        "unsupported_media_type",
                        "the request media type is not supported",
                    )
                if (
                    framing.content_length is not None
                    and framing.content_length > body_policy.max_bytes
                ):
                    await send_api_error(
                        scope,
                        receive,
                        boundary_send,
                        status_code=413,
                        code="request_too_large",
                        message="the request body exceeds the configured size limit",
                        headers={"Connection": "close"},
                    )
                    return
                counted_receive = self._counting_receive(receive, body_policy.max_bytes)

            await self.app(scope, counted_receive, boundary_send)
        except (RequestBoundaryError, SecurityPolicyError) as exc:
            if response_started:
                raise
            extra_headers = {"Connection": "close"} if exc.status_code == 413 else None
            await send_api_error(
                scope,
                receive,
                boundary_send,
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
                headers=extra_headers,
            )
        except Exception:
            if response_started:
                raise
            await send_api_error(
                scope,
                receive,
                boundary_send,
                status_code=500,
                code="internal_error",
                message="an internal error occurred",
            )

    @staticmethod
    def _counting_receive(receive: Receive, max_bytes: int) -> Receive:
        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > max_bytes:
                    # FastAPI preserves HTTPException raised while parsing a
                    # request body; a custom exception is otherwise converted
                    # into an unrelated generic 400 by its parser boundary.
                    raise HTTPException(status_code=413)
            return message

        return counting_receive
