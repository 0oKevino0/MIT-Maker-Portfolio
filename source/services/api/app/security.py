"""Single request-security policy for token, Host, Origin, and CORS inputs."""

from __future__ import annotations

import ipaddress
import re
import secrets
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from starlette.types import Scope

from services.api.app.errors import raw_header_values

if TYPE_CHECKING:
    from services.api.app.config.settings import Settings

LOCAL_TOKEN_HEADER = b"x-cswa-local-token"
MIN_TOKEN_LENGTH = 32
MAX_TOKEN_LENGTH = 512
RUNTIME_TOKEN_CONFIGURATION_ERROR = "runtime API token configuration is invalid"
MODEL_BASE_URL_CONFIGURATION_ERROR = "model base URL configuration is invalid"

_DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z", re.ASCII)
_PORT = re.compile(r"[0-9]+\Z", re.ASCII)
_PLACEHOLDER_MARKERS = ("YOUR_TOKEN_HERE", "CHANGE_ME", "PLACEHOLDER", "EXAMPLE")
_BUSINESS_POST_PATHS = frozenset({"/api/v1/course/ask"})


class SecurityPolicyError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ParsedHost:
    hostname: str
    port: int | None


def _parse_port(raw_port: str | None) -> int | None:
    if raw_port is None:
        return None
    if not raw_port or not _PORT.fullmatch(raw_port):
        raise ValueError("host port must be decimal")
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise ValueError("host port is outside the valid range")
    return port


def _normalize_dns_or_ipv4(raw_hostname: str) -> str:
    if not raw_hostname or len(raw_hostname) > 253:
        raise ValueError("host name is empty or too long")
    try:
        return str(ipaddress.IPv4Address(raw_hostname))
    except ipaddress.AddressValueError:
        if "." in raw_hostname and all(char.isdigit() or char == "." for char in raw_hostname):
            raise ValueError("invalid IPv4 address") from None
    labels = raw_hostname.split(".")
    if any(not label or not _DNS_LABEL.fullmatch(label) for label in labels):
        raise ValueError("invalid DNS host name")
    return raw_hostname.casefold()


def parse_host_authority(value: str) -> ParsedHost:
    """Parse one strict Host/authority value without accepting suffix text."""
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("host must be ASCII") from exc
    if (
        not value
        or any(char.isspace() or ord(char) < 0x21 or ord(char) > 0x7E for char in value)
        or any(char in value for char in "@/?#%\\")
    ):
        raise ValueError("host contains forbidden syntax")

    if value.startswith("["):
        closing = value.find("]")
        if closing <= 1:
            raise ValueError("invalid bracketed IPv6 host")
        raw_address = value[1:closing]
        suffix = value[closing + 1 :]
        if suffix:
            if not suffix.startswith(":"):
                raise ValueError("unexpected text after IPv6 host")
            raw_port: str | None = suffix[1:]
        else:
            raw_port = None
        try:
            hostname = ipaddress.IPv6Address(raw_address).compressed.casefold()
        except ipaddress.AddressValueError as exc:
            raise ValueError("invalid IPv6 address") from exc
        return ParsedHost(hostname, _parse_port(raw_port))

    if "[" in value or "]" in value or value.count(":") > 1:
        raise ValueError("IPv6 Host values must be bracketed")
    if ":" in value:
        raw_hostname, raw_port = value.rsplit(":", 1)
    else:
        raw_hostname, raw_port = value, None
    return ParsedHost(_normalize_dns_or_ipv4(raw_hostname), _parse_port(raw_port))


def normalize_allowed_host(value: str) -> str:
    candidate = value.strip()
    if not candidate or "*" in candidate:
        raise ValueError("allowed hosts must be non-empty exact values")
    if candidate.startswith("[") or candidate.endswith("]"):
        if not (candidate.startswith("[") and candidate.endswith("]")):
            raise ValueError("allowed IPv6 hosts must use one complete bracket pair")
        if candidate.count("[") != 1 or candidate.count("]") != 1:
            raise ValueError("allowed IPv6 hosts must use one complete bracket pair")
        address_text = candidate[1:-1]
    else:
        address_text = candidate
    try:
        address = ipaddress.ip_address(address_text)
    except ValueError:
        return parse_host_authority(candidate).hostname
    return address.compressed.casefold()


def normalize_origin(value: str) -> str:
    """Return the canonical scheme/host/optional-port browser origin."""
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("origin must be ASCII") from exc
    if (
        not value
        or value.casefold() == "null"
        or "*" in value
        or "%" in value
        or any(char.isspace() or ord(char) < 0x21 for char in value)
    ):
        raise ValueError("origin is not an exact network origin")
    try:
        parts = urlsplit(value)
    except ValueError as exc:
        raise ValueError("origin is malformed") from exc
    scheme = parts.scheme.casefold()
    if scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("origin must use http or https")
    if parts.path or parts.query or parts.fragment or parts.username or parts.password:
        raise ValueError("origin must not include path, query, fragment, or userinfo")
    parsed_host = parse_host_authority(parts.netloc)
    port = parsed_host.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    rendered_host = (
        f"[{parsed_host.hostname}]" if ":" in parsed_host.hostname else parsed_host.hostname
    )
    port_suffix = f":{port}" if port is not None else ""
    return f"{scheme}://{rendered_host}{port_suffix}"


def normalize_model_base_url(value: str) -> str:
    """Return one strict, canonical model service base URL."""
    try:
        value.encode("ascii")
        if (
            not value
            or any(ord(char) < 0x21 or ord(char) > 0x7E for char in value)
            or "\\" in value
            or "?" in value
            or "#" in value
        ):
            raise ValueError("model base URL contains forbidden syntax")
        parts = urlsplit(value)
        scheme = parts.scheme.casefold()
        if scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("model base URL has an invalid scheme or authority")
        if "@" in parts.netloc or "%" in parts.netloc:
            raise ValueError("model base URL authority contains forbidden syntax")
        parsed_host = parse_host_authority(parts.netloc)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError(MODEL_BASE_URL_CONFIGURATION_ERROR) from exc

    rendered_host = (
        f"[{parsed_host.hostname}]" if ":" in parsed_host.hostname else parsed_host.hostname
    )
    port_suffix = f":{parsed_host.port}" if parsed_host.port is not None else ""
    return f"{scheme}://{rendered_host}{port_suffix}{parts.path}"


def is_loopback_hostname(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _is_sequential_token(value: str) -> bool:
    """Detect a complete ascending or descending decimal cycle."""
    if not value.isascii() or not value.isdigit():
        return False
    digits = [ord(char) - ord("0") for char in value]
    if all((right - left) % 10 == 1 for left, right in pairwise(digits)):
        return True
    return all((left - right) % 10 == 1 for left, right in pairwise(digits))


def _is_periodic_or_near_periodic(value: str) -> bool:
    """Detect values within a very small Hamming distance of a repeated pattern."""
    mismatch_budget = min(8, max(3, len(value) // 64))
    for period in range(1, len(value) // 2 + 1):
        mismatches = 0
        for offset in range(period):
            column = value[offset::period]
            common_count = Counter(column).most_common(1)[0][1]
            mismatches += len(column) - common_count
            if mismatches > mismatch_budget:
                break
        if mismatches <= mismatch_budget:
            return True
    return False


def _has_extreme_character_dominance(value: str) -> bool:
    """Reject values overwhelmingly composed of one to three characters."""
    counts = [count for _, count in Counter(value).most_common(3)]
    total = len(value)
    if counts[0] * 4 >= total * 3:
        return True
    if len(counts) >= 2 and sum(counts[:2]) * 8 >= total * 7:
        return True
    return len(counts) >= 3 and sum(counts[:3]) * 16 >= total * 15


def validate_runtime_token(value: str) -> str:
    """Validate a configured token without normalizing or exposing it."""
    if not MIN_TOKEN_LENGTH <= len(value) <= MAX_TOKEN_LENGTH:
        raise ValueError(RUNTIME_TOKEN_CONFIGURATION_ERROR)
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(RUNTIME_TOKEN_CONFIGURATION_ERROR) from exc
    if value != value.strip() or any(char.isspace() for char in value) or "," in value:
        raise ValueError(RUNTIME_TOKEN_CONFIGURATION_ERROR)
    upper = value.upper()
    if any(marker in upper for marker in _PLACEHOLDER_MARKERS):
        raise ValueError(RUNTIME_TOKEN_CONFIGURATION_ERROR)
    if re.search(r"(.)\1{7,}", value):
        raise ValueError(RUNTIME_TOKEN_CONFIGURATION_ERROR)
    if (
        _is_sequential_token(value)
        or _is_periodic_or_near_periodic(value)
        or _has_extreme_character_dominance(value)
    ):
        raise ValueError(RUNTIME_TOKEN_CONFIGURATION_ERROR)
    return value


def load_local_token(settings: Settings) -> str | None:
    """Resolve one configured runtime token without exposing configuration details."""
    direct = settings.cswa_local_api_token
    token_file: Path | None = settings.local_api_token_file
    if direct is not None and token_file is not None:
        raise RuntimeError(RUNTIME_TOKEN_CONFIGURATION_ERROR)
    if direct is not None:
        return validate_runtime_token(direct.get_secret_value())
    if token_file is None:
        return None
    try:
        value = token_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise RuntimeError(RUNTIME_TOKEN_CONFIGURATION_ERROR) from None
    try:
        return validate_runtime_token(value)
    except ValueError:
        raise RuntimeError(RUNTIME_TOKEN_CONFIGURATION_ERROR) from None


@dataclass(frozen=True, slots=True)
class SecurityPolicy:
    allowed_hosts: frozenset[str]
    allowed_origins: frozenset[str]
    expected_token: str | None = field(repr=False)

    @classmethod
    def from_settings(cls, settings: Settings, expected_token: str | None) -> SecurityPolicy:
        return cls(
            allowed_hosts=frozenset(settings.allowed_hosts_list),
            allowed_origins=frozenset(settings.allowed_origins_list),
            expected_token=expected_token,
        )

    def validate_host(self, scope: Scope) -> str:
        values = raw_header_values(scope, b"host")
        if len(values) != 1:
            raise SecurityPolicyError(400, "invalid_host", "invalid host header")
        try:
            raw_host = values[0].decode("ascii")
            hostname = parse_host_authority(raw_host).hostname
        except (UnicodeDecodeError, ValueError) as exc:
            raise SecurityPolicyError(400, "invalid_host", "invalid host header") from exc
        if hostname not in self.allowed_hosts:
            raise SecurityPolicyError(400, "invalid_host", "invalid host header")
        return hostname

    def validate_origin(self, scope: Scope) -> str | None:
        values = raw_header_values(scope, b"origin")
        if len(values) > 1:
            raise SecurityPolicyError(
                400, "malformed_request", "the request has duplicate Origin headers"
            )
        if not values:
            return None
        try:
            origin = normalize_origin(values[0].decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise SecurityPolicyError(
                403, "origin_not_allowed", "origin is not in the local allowlist"
            ) from exc
        if origin not in self.allowed_origins:
            raise SecurityPolicyError(
                403, "origin_not_allowed", "origin is not in the local allowlist"
            )
        return origin

    def requires_auth(self, method: str, path: str) -> bool:
        return (
            self.expected_token is not None
            and method.upper() == "POST"
            and path in _BUSINESS_POST_PATHS
        )

    def authenticate(self, scope: Scope) -> None:
        values = raw_header_values(scope, LOCAL_TOKEN_HEADER)
        if len(values) > 1:
            raise SecurityPolicyError(
                400, "malformed_request", "the request has duplicate authentication headers"
            )
        if len(values) == 1 and b"," in values[0]:
            raise SecurityPolicyError(
                400, "malformed_request", "the request has an ambiguous authentication header"
            )
        if not values or self.expected_token is None:
            raise SecurityPolicyError(
                401, "invalid_local_token", "missing or invalid local API token"
            )
        try:
            provided = values[0].decode("ascii")
        except UnicodeDecodeError:
            provided = ""
        if (
            not provided
            or len(provided) > MAX_TOKEN_LENGTH
            or not secrets.compare_digest(provided, self.expected_token)
        ):
            raise SecurityPolicyError(
                401, "invalid_local_token", "missing or invalid local API token"
            )
