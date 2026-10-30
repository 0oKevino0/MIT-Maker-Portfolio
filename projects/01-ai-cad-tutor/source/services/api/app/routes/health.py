"""Liveness and readiness endpoint.

Liveness and readiness are separate (review issue I-07): the endpoint always
returns HTTP 200 with ``status: alive`` while the process runs, and a
``readiness`` value aggregated from storage, both model backends, and
retrieval. A Mock backend outside the test environment is reported as
degraded so a development setup can never look production-ready.

Nothing here triggers a generation request, and no detail field ever
contains an absolute path, base URL, token, or other topology information —
only stable logical labels and redacted messages.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from services.api.app.config.settings import Settings
from services.inference.backends.base import BackendHealth, HealthState, ModelBackend
from services.inference.capability import COURSE_REQUIRED_CAPABILITIES, declared_capabilities
from services.inference.errors import BackendConfigurationError
from services.storage.paths import resolve_storage_paths
from services.storage.service import StorageState, validate_storage

router = APIRouter()

API_VERSION = "0.1.0"

# Upper bound for one readiness probe (review D1A-05). ``asyncio.wait_for``
# cancels the underlying probe task on expiry, so a hanging provider can
# neither block the endpoint nor accumulate pending tasks.
READINESS_PROBE_TIMEOUT_SECONDS = 5.0

_READINESS_ORDER = {"ready": 0, "degraded": 1, "unavailable": 2}
#: What the demo profile reports when its retrieval corpus will not load.
_RETRIEVAL_NOT_IMPLEMENTED = "course retrieval index is not implemented yet"


def _demo_retrieval_block(settings: Settings) -> dict[str, Any]:
    """The honest retrieval state: the demo's frozen corpus, or nothing implemented.

    Importing the demo retrieval module lazily keeps the default profile free of it, and keeps a
    broken corpus from taking the health endpoint down: it is reported, not raised.
    """
    if not settings.internal_demo_enabled:
        return {"state": "unavailable", "detail": _RETRIEVAL_NOT_IMPLEMENTED}
    try:
        from services.rag.retrieval.demo_lexical import corpus_is_intact, demo_corpus_identity

        if not corpus_is_intact():
            return {
                "state": "unavailable",
                "detail": "the experimental demo corpus did not load",
            }
        identity = demo_corpus_identity().as_dict()
    except Exception:
        return {"state": "unavailable", "detail": "the experimental demo corpus did not load"}
    block = {
        "state": "ready",
        "detail": "INTERNAL DEMO profile: frozen experimental concept corpus, lexical retrieval",
        "corpus": identity,
        "top_k": settings.demo_retrieval_top_k,
    }
    block["tutorial_corpus"] = _tutorial_corpus_block(settings)
    return block


def _tutorial_corpus_block(settings: Settings) -> dict[str, Any]:
    """The second corpus, by digest and permission — or why it is not loaded.

    Reported rather than silently absent: a reviewer must be able to see whether procedure evidence
    is available, and if not, which check refused it.
    """
    if not settings.demo_tutorial_corpus_path:
        return {"state": "not_configured"}
    try:
        from services.rag.retrieval.demo_tutorials import (
            TutorialCorpusError,
            tutorial_corpus_identity,
        )

        identity = tutorial_corpus_identity(
            settings.demo_tutorial_corpus_path,
            settings.demo_tutorial_corpus_sha256 or "",
            settings.demo_permission_decisions_dir or "",
            settings.demo_tutorial_source_root or "",
        )
    except TutorialCorpusError as error:
        return {"state": "unavailable", "reason": error.code}
    except Exception:
        return {"state": "unavailable", "reason": "tutorial_corpus_unavailable"}
    return {"state": "ready", **identity}


def _model_identity_block(backend: ModelBackend, probe: BackendHealth) -> dict[str, Any] | None:
    """The identity the backend verified, when it currently has one.

    Three things must hold and the block is omitted otherwise: the backend declares the identity
    verified (its own flag, never a value manufactured here), it carries both the record and its
    digest, and the probe just taken reports READY. A configured model name is not identity, and an
    identity verified before the sidecar stopped answering is not a current fact.
    """
    if probe.state is not HealthState.READY or not getattr(backend, "identity_verified", False):
        return None
    identity = getattr(backend, "identity", None)
    digest = getattr(backend, "identity_sha256", None)
    if not isinstance(identity, dict) or not identity:
        return None
    return {
        "verified": True,
        "identity_sha256": digest,
        "base_model_id": identity.get("base_model_id"),
        "base_model_revision": identity.get("base_model_revision"),
        "adapter_sha256_measured": identity.get("adapter_sha256_measured"),
        "adapter_merged_into_base_weights": identity.get("adapter_merged_into_base_weights"),
        "inference_only": identity.get("inference_only"),
        "device": identity.get("device"),
        "dtype": identity.get("dtype"),
        "decoding": identity.get("decoding"),
    }


def _worst(*states: str) -> str:
    return max(states, key=lambda state: _READINESS_ORDER[state])


def _storage_readiness(state: StorageState) -> str:
    if state == StorageState.AVAILABLE:
        return "ready"
    if state == StorageState.DEGRADED:
        return "degraded"
    # misconfigured, not_configured, root_missing, root_not_directory, and
    # unavailable all mean data-dependent features must refuse to run — an
    # unsafe layout is never merely "degraded" (review 0002, I-01).
    return "unavailable"


def _backend_readiness(backend: ModelBackend, probe: BackendHealth, settings: Settings) -> str:
    if probe.state == HealthState.UNAVAILABLE:
        return "unavailable"
    if backend.name == "mock" and settings.app_env != "test":
        # A mock can answer, but it is not a real model; a development setup
        # must never be mistaken for a ready deployment.
        return "degraded"
    return "degraded" if probe.state == HealthState.DEGRADED else "ready"


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Process liveness only (review D1A-01).

    Answers "can this API process and its event loop handle requests?" and
    nothing else: no backend construction or call, no model-file check, no
    storage validation, no network access, and no credential read. The body
    is small, fixed, and free of any path/URL/token detail.
    """
    return {"status": "alive", "service": "cswa-ai-tutor-api", "version": API_VERSION}


def _not_ready(reason: str) -> JSONResponse:
    # Small fixed body: a stable machine label only — never a URL, model
    # path, token, exception text, or internal schema (review D1A-05/06).
    return JSONResponse(status_code=503, content={"status": "not_ready", "reason": reason})


@router.get("/health/ready")
async def readiness(request: Request) -> JSONResponse:
    """Course-serving readiness with real HTTP semantics (review D1A-01).

    Ready (200) only when the text backend declares the capabilities the
    Course path requires AND its bounded health probe reports READY. The
    probe is time-boxed and cancellable, has no side effects, and never
    constructs a fallback. Mock outside the test environment is never
    reported ready (D1A-03).
    """
    settings: Settings = request.app.state.settings
    backend: ModelBackend = request.app.state.text_backend

    # 1. Strict capability contract — no I/O, no name/model inference.
    try:
        capabilities = declared_capabilities(backend)
    except BackendConfigurationError:
        return _not_ready("capability_contract_invalid")
    if not (capabilities.text or capabilities.vision or capabilities.structured_output):
        return _not_ready("model_not_configured")
    if not all(getattr(capabilities, flag) for flag in COURSE_REQUIRED_CAPABILITIES):
        return _not_ready("capability_mismatch")

    # 2. Bounded, cancellable health probe (review D1A-05).
    try:
        probe = await asyncio.wait_for(backend.health(), READINESS_PROBE_TIMEOUT_SECONDS)
    except TimeoutError:
        return _not_ready("probe_timeout")
    except Exception:
        return _not_ready("probe_failed")
    if not isinstance(probe, BackendHealth):
        return _not_ready("probe_failed")
    if probe.state != HealthState.READY:
        return _not_ready("backend_unavailable")

    # 3. A Mock can answer, but it is not a real model (existing /health
    #    policy, D1A-03): outside the test environment it is never ready.
    if backend.name == "mock" and settings.app_env != "test":
        return _not_ready("mock_backend_not_ready")

    # 4. The INTERNAL DEMO profile answers from retrieved evidence, so a demo whose corpus will
    #    not load is not ready: the UI must not invite a question it can only refuse.
    if settings.internal_demo_enabled and _demo_retrieval_block(settings)["state"] != "ready":
        return _not_ready("retrieval_unavailable")

    return JSONResponse(status_code=200, content={"status": "ready"})


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    text_backend: ModelBackend = request.app.state.text_backend
    vision_backend: ModelBackend = request.app.state.vision_backend

    text_health = await text_backend.health()
    vision_health = await vision_backend.health()
    storage_status = validate_storage(resolve_storage_paths(settings))

    retrieval_block = _demo_retrieval_block(settings)
    retrieval_readiness = retrieval_block["state"]

    readiness_detail = {
        "storage": _storage_readiness(storage_status.state),
        "text_backend": _backend_readiness(text_backend, text_health, settings),
        "vision_backend": _backend_readiness(vision_backend, vision_health, settings),
        "retrieval": retrieval_readiness,
    }
    readiness = _worst(*readiness_detail.values())

    def backend_block(backend: ModelBackend, probe: BackendHealth, role: str) -> dict[str, Any]:
        block: dict[str, Any] = {
            "backend": probe.backend,
            "state": probe.state.value,
            "detail": probe.detail,
        }
        if backend.name == "mock" and settings.app_env != "test":
            block["detail"] = "mock backend is enabled; not a real model"
        return block

    payload: dict[str, Any] = {
        "status": "alive",
        "readiness": readiness,
        "readiness_detail": readiness_detail,
        "service": "cswa-ai-tutor-api",
        "version": API_VERSION,
        "backends": {
            "text": backend_block(text_backend, text_health, "text"),
            "vision": backend_block(vision_backend, vision_health, "vision"),
        },
        "storage": {
            "status": storage_status.state.value,
            "required_directories": storage_status.required_directories,
            "missing": list(storage_status.missing),
            "conflicts": list(storage_status.conflicts),
            "violations": list(storage_status.violations),
            "message": storage_status.message,
        },
        "retrieval": retrieval_block,
    }
    identity = _model_identity_block(text_backend, text_health)
    if identity is not None:
        payload["model_identity"] = identity
    if settings.internal_demo_enabled:
        payload["profile"] = {
            "internal_demo_enabled": True,
            "experimental": True,
            "teaching_reliability_established": False,
            "public_release_authorized": False,
            "production_deployment_authorized": False,
        }
    return payload
