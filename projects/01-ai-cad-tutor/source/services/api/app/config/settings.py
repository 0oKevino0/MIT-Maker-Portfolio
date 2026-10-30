"""Centralized environment-driven configuration.

All environment keys follow ADR 0001 section 13.7. Values come from process
environment variables or an optional local ``.env`` file (never committed).

Runtime startup guards (fail closed instead of running unsafely):

- neither model role may be ``mock``;
- development and production require exactly one strong local token source;
- allowed origins/hosts must be non-empty exact values;
- the bind host must be loopback unless explicitly allowed;
- remote model settings require HTTPS and existing auth/CA files; a separate
  runtime capability gate keeps transport execution disabled.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from services.api.app.security import (
    MODEL_BASE_URL_CONFIGURATION_ERROR,
    RUNTIME_TOKEN_CONFIGURATION_ERROR,
    is_loopback_hostname,
    normalize_allowed_host,
    normalize_model_base_url,
    normalize_origin,
    validate_runtime_token,
)

# "disabled" is the explicit no-model state (review D1A-02): the app starts
# and stays alive, but readiness is 503 and model requests fail closed with
# the fixed model_not_configured classification. It is the DEFAULT so that
# missing environment configuration can never silently select Mock (D1A-03).
BackendName = Literal["disabled", "mock", "ollama", "v1b_demo", "vllm"]

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}

_HEX_CHARACTERS = "0123456789abcdef"
#: One message for every way the demo adapter digest can be wrong: it never echoes the value.
#: One message for an incomplete or malformed tutorial-corpus configuration. Never echoes a path.
DEMO_TUTORIAL_CONFIGURATION_ERROR = (
    "the demo tutorial corpus needs all four of DEMO_TUTORIAL_CORPUS_PATH, "
    "DEMO_TUTORIAL_CORPUS_SHA256 (64 hex characters), DEMO_PERMISSION_DECISIONS_DIR and "
    "DEMO_TUTORIAL_SOURCE_ROOT, or none of them: an unverifiable corpus is not loaded"
)
DEMO_ADAPTER_DIGEST_CONFIGURATION_ERROR = (
    "DEMO_EXPECTED_ADAPTER_SHA256 must be the 64-hexadecimal-character digest of the selected "
    "adapter, and is required whenever TEXT_BACKEND=v1b_demo: a configured model name is not "
    "proof that the selected adapter is loaded"
)

# The default env-file source. Set CSWA_DISABLE_ENV_FILE=1 (the test session
# does this in the root conftest) so that importing the global application —
# including its module-level ``app = create_app()`` — never reads a `.env`
# from the current working directory. This keeps tests hermetic: a developer
# or CI `.env` in cwd can never leak into test app state (review 0004, I01-C).
_DEFAULT_ENV_FILE: str | None = None if os.getenv("CSWA_DISABLE_ENV_FILE") == "1" else ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    app_env: Literal["test", "development", "production"] = "development"

    api_bind_host: str = "127.0.0.1"
    api_port: int = 8000
    # Comma-separated exact browser origins (never wildcard or null).
    api_allowed_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
    # Comma-separated exact Host header values (port ignored during comparison).
    api_allowed_hosts: str = "127.0.0.1,localhost"
    # Explicit opt-in for non-loopback binding (e.g. a future lab deployment).
    api_allow_non_loopback: bool = False
    # Maximum accepted JSON request body in bytes, enforced before parsing.
    api_max_json_body_bytes: int = Field(default=64 * 1024, ge=1024)
    # Reserved independently for a future multipart route.
    api_max_multipart_body_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)

    # Local authentication for business POST routes (GET /health stays public,
    # per ADR 0001 section 3.3). Development/production configure exactly one
    # source; test fixtures may explicitly omit it.
    cswa_local_api_token: SecretStr | None = None
    local_api_token_file: Path | None = None

    # Instructor answer mode is a response-depth gate, disabled by default.
    instructor_mode_enabled: bool = False

    # ---- INTERNAL DEMO profile (docs/internal-demo/README.md) ----
    # An explicit opt-in profile for the supervised internal demonstration of the owner-selected
    # V1B adapter. It is OFF by default, changes nothing in any other profile, and is never a
    # production configuration: it wires the frozen experimental retrieval corpus and the adapter
    # sidecar into the existing Course path. Teaching reliability is not established by it.
    internal_demo_enabled: bool = False
    # How many retrieved records may become evidence for one question.
    demo_retrieval_top_k: int = Field(default=3, ge=1, le=10)
    # The tutorial corpus for the demo, with everything needed to verify it. All four are required
    # together or none is used: an unverifiable corpus is not loaded, and the demo answers from the
    # frozen concept corpus alone. None of these paths is inside the repository.
    demo_tutorial_corpus_path: str | None = None
    demo_tutorial_corpus_sha256: str | None = None
    demo_permission_decisions_dir: str | None = None
    demo_tutorial_source_root: str | None = None
    # Where to append one internal provenance line per demo request. Unset means no provenance log.
    demo_provenance_log: str | None = None

    # The adapter digest the loaded model MUST measure to. Required whenever TEXT_BACKEND=v1b_demo:
    # the backend refuses to be constructed without it, and refuses to answer or to report a
    # verified identity unless the adapter it actually loaded measures to exactly this value.
    demo_expected_adapter_sha256: str | None = None

    text_backend: BackendName = "disabled"
    text_model_id: str = "qwen2.5:7b-instruct"
    text_base_url: str = "http://127.0.0.1:11434"
    text_auth_token_file: Path | None = None
    text_tls_ca_file: Path | None = None

    vision_backend: BackendName = "disabled"
    vision_model_id: str = "qwen2.5vl:7b"
    vision_base_url: str = "http://127.0.0.1:11434"
    vision_auth_token_file: Path | None = None
    vision_tls_ca_file: Path | None = None

    model_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    model_read_timeout_seconds: float = Field(default=30.0, gt=0)
    model_text_deadline_seconds: float = Field(default=60.0, gt=0)
    model_vision_deadline_seconds: float = Field(default=120.0, gt=0)
    model_max_attempts: int = Field(default=2, ge=1, le=2)
    model_max_output_tokens: int = Field(default=2048, ge=1)
    model_max_concurrency: int = Field(default=2, ge=1)

    # ---- External data storage (docs/development/EXTERNAL_STORAGE.md) ----
    # Large/private content lives on an external drive, never in the repo.
    # Paths are plain values here: no directory is created or touched at
    # import time, and no Windows<->WSL2 conversion is guessed — each
    # environment sets its own paths in its own .env. Subdirectories default
    # to well-known locations under CSWA_DATA_ROOT and can be overridden
    # individually (resolution lives in services.storage.paths).
    cswa_data_root: Path | None = None
    cswa_raw_data_dir: Path | None = None
    cswa_processed_data_dir: Path | None = None
    cswa_approved_data_dir: Path | None = None
    cswa_benchmark_dir: Path | None = None
    cswa_synthetic_data_dir: Path | None = None
    cswa_cad_data_dir: Path | None = None
    cswa_model_dir: Path | None = None
    cswa_vector_db_dir: Path | None = None
    cswa_embedding_dir: Path | None = None
    cswa_checkpoint_dir: Path | None = None
    cswa_cache_dir: Path | None = None
    cswa_export_dir: Path | None = None
    cswa_log_dir: Path | None = None
    cswa_backup_dir: Path | None = None

    @property
    def allowed_origins_list(self) -> list[str]:
        values = [
            origin.strip() for origin in self.api_allowed_origins.split(",") if origin.strip()
        ]
        return [normalize_origin(origin) for origin in values]

    @property
    def allowed_hosts_list(self) -> list[str]:
        values = [host.strip() for host in self.api_allowed_hosts.split(",") if host.strip()]
        hosts = [normalize_allowed_host(host) for host in values]
        if self.app_env == "test" and "testserver" not in hosts:
            # Starlette's in-process TestClient uses this pseudo-host; it is
            # only reachable inside the test process, never over a network.
            hosts.append("testserver")
        return hosts

    @model_validator(mode="after")
    def enforce_runtime_guards(self) -> Self:
        try:
            normalized_text_base_url = normalize_model_base_url(self.text_base_url)
            normalized_vision_base_url = normalize_model_base_url(self.vision_base_url)
        except ValueError:
            raise ValueError(MODEL_BASE_URL_CONFIGURATION_ERROR) from None
        object.__setattr__(self, "text_base_url", normalized_text_base_url)
        object.__setattr__(self, "vision_base_url", normalized_vision_base_url)

        configured_hosts = [
            value.strip() for value in self.api_allowed_hosts.split(",") if value.strip()
        ]
        configured_origins = [
            value.strip() for value in self.api_allowed_origins.split(",") if value.strip()
        ]
        if not configured_hosts:
            raise ValueError("API_ALLOWED_HOSTS must contain at least one exact host")
        if not configured_origins:
            raise ValueError("API_ALLOWED_ORIGINS must contain at least one exact origin")
        # Evaluate both normalized lists here so malformed entries fail during
        # configuration construction rather than on the first request.
        _ = self.allowed_hosts_list
        _ = self.allowed_origins_list

        direct_token = self.cswa_local_api_token
        token_file = self.local_api_token_file
        if direct_token is not None and token_file is not None:
            raise ValueError(RUNTIME_TOKEN_CONFIGURATION_ERROR)
        if direct_token is not None:
            validate_runtime_token(direct_token.get_secret_value())
        if self.app_env != "test" and direct_token is None and token_file is None:
            raise ValueError(RUNTIME_TOKEN_CONFIGURATION_ERROR)

        if self.app_env == "production":
            if "mock" in (self.text_backend, self.vision_backend):
                raise ValueError(
                    "the mock backend must not be enabled in production configuration "
                    "(ADR 0001 section 13.7)"
                )
            if self.api_bind_host not in _LOOPBACK_HOSTS and not self.api_allow_non_loopback:
                raise ValueError(
                    "production refuses a non-loopback bind host unless "
                    "API_ALLOW_NON_LOOPBACK=true is set explicitly"
                )

        # ---- INTERNAL DEMO guards ----
        # The owner-selected adapter digest is the whole basis of the demo's "verified model"
        # claim, so the backend that loads it may not be selected without one, and a malformed
        # digest is rejected at configuration time rather than at the first probe.
        if self.demo_expected_adapter_sha256 is not None:
            digest = self.demo_expected_adapter_sha256.strip().lower()
            if len(digest) != 64 or set(digest) - set(_HEX_CHARACTERS):
                raise ValueError(DEMO_ADAPTER_DIGEST_CONFIGURATION_ERROR)
            object.__setattr__(self, "demo_expected_adapter_sha256", digest)
        if self.text_backend == "v1b_demo" and self.demo_expected_adapter_sha256 is None:
            raise ValueError(DEMO_ADAPTER_DIGEST_CONFIGURATION_ERROR)
        tutorial_fields = (
            self.demo_tutorial_corpus_path,
            self.demo_tutorial_corpus_sha256,
            self.demo_permission_decisions_dir,
            self.demo_tutorial_source_root,
        )
        if any(tutorial_fields) and not all(tutorial_fields):
            raise ValueError(DEMO_TUTORIAL_CONFIGURATION_ERROR)
        if self.demo_tutorial_corpus_sha256 is not None:
            digest = self.demo_tutorial_corpus_sha256.strip().lower()
            if len(digest) != 64 or set(digest) - set(_HEX_CHARACTERS):
                raise ValueError(DEMO_TUTORIAL_CONFIGURATION_ERROR)
            object.__setattr__(self, "demo_tutorial_corpus_sha256", digest)

        if self.app_env == "production" and self.internal_demo_enabled:
            # The demo is an internal, supervised, experimental profile. Production deployment of
            # it is not authorized, so the configuration refuses rather than relying on a document.
            raise ValueError(
                "the INTERNAL DEMO profile is not a production configuration "
                "(docs/internal-demo/README.md)"
            )

        remote_roles = (
            (
                "text",
                self.text_base_url,
                self.text_auth_token_file,
                self.text_tls_ca_file,
            ),
            (
                "vision",
                self.vision_base_url,
                self.vision_auth_token_file,
                self.vision_tls_ca_file,
            ),
        )
        for role, url, auth_file, ca_file in remote_roles:
            parts = urlsplit(url)
            hostname = parts.hostname or ""
            if is_loopback_hostname(hostname):
                continue
            if parts.scheme != "https":
                raise ValueError(f"remote {role} model base URL must use https")
            if auth_file is None or ca_file is None:
                raise ValueError(
                    f"remote {role} model configuration requires auth token and TLS CA files"
                )
            if not auth_file.is_file() or not ca_file.is_file():
                raise ValueError(f"remote {role} model auth token and TLS CA files must exist")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
