"""Storage validation, safe directory creation, and feature gating.

All functions here are strictly non-destructive:

- ``validate_storage`` performs read-only checks and never modifies anything;
  it never raises for ordinary filesystem problems — it returns a stable
  status instead, so the health endpoint can never turn a storage fault
  into an HTTP 500.
- ``ensure_runtime_directories`` runs a complete preflight (canonicalize,
  policy, and every file conflict — including conflicts on ancestors of
  later directories) before creating anything; only when every directory
  can be created safely does it start ``mkdir``. It never deletes,
  overwrites, moves, or renames anything. (A time-of-check/time-of-use
  window between preflight and creation is unavoidable and documented.)
- ``require_storage_for`` is a read-only gate with an explicit per-feature
  directory/permission matrix that data-dependent features must pass before
  doing any work.

Nothing runs at module import time, and nothing ever falls back to the git
repository ``data/`` directory, the current working directory, a temporary
directory, a user directory, or an HPC path: when storage is not
configured, unsafe, or unreachable, the caller gets exactly that answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from services.storage.errors import (
    StorageError,
    StorageNotAvailableError,
    StorageNotConfiguredError,
    StoragePathConflictError,
    StoragePolicyError,
)
from services.storage.paths import StoragePaths
from services.storage.policy import (
    check_storage_policy,
    directory_is_readable,
    directory_is_writable,
)

__all__ = [
    "FEATURE_REQUIREMENTS",
    "DirectoryRequirement",
    "StorageError",
    "StorageNotAvailableError",
    "StorageNotConfiguredError",
    "StoragePathConflictError",
    "StoragePolicyError",
    "StorageState",
    "StorageStatus",
    "ensure_runtime_directories",
    "require_storage_for",
    "validate_storage",
]


@dataclass(frozen=True)
class DirectoryRequirement:
    """One directory a feature depends on, with its access requirements."""

    label: str
    readable: bool = True
    writable: bool = False
    must_exist: bool = True


# Explicit per-feature directory/permission matrix (review I-01). A feature
# gates on exactly the directories it needs, with exactly the access it needs.
FEATURE_REQUIREMENTS: dict[str, tuple[DirectoryRequirement, ...]] = {
    "course_ingestion": (
        DirectoryRequirement("datasets/raw", readable=True, writable=False),
        DirectoryRequirement("datasets/processed", readable=True, writable=True),
        DirectoryRequirement("datasets/approved", readable=True, writable=False),
    ),
    "course_retrieval": (
        DirectoryRequirement("datasets/approved", readable=True, writable=False),
        DirectoryRequirement("vector_db", readable=True, writable=False),
        DirectoryRequirement("embeddings", readable=True, writable=False),
    ),
    "benchmark_read": (DirectoryRequirement("datasets/benchmarks", readable=True, writable=False),),
    "vector_index_write": (DirectoryRequirement("vector_db", readable=True, writable=True),),
    "embedding_write": (DirectoryRequirement("embeddings", readable=True, writable=True),),
    "model_load": (
        DirectoryRequirement("models", readable=True, writable=False),
        DirectoryRequirement("cache", readable=True, writable=True),
    ),
    "cad_read": (DirectoryRequirement("cad", readable=True, writable=False),),
    "export_write": (DirectoryRequirement("exports", readable=True, writable=True),),
}


class StorageState(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"  # root fine, some required subdirectories missing/conflicting
    ROOT_MISSING = "root_missing"  # drive/mount reachable, root folder absent
    ROOT_NOT_DIRECTORY = "root_not_directory"  # root exists but is a file
    UNAVAILABLE = "unavailable"  # drive/mount unreachable or state undeterminable
    NOT_CONFIGURED = "not_configured"
    MISCONFIGURED = "misconfigured"  # layout violates the safety policy


@dataclass(frozen=True)
class StorageStatus:
    """Read-only storage probe result.

    ``missing``, ``conflicts``, and ``violations`` contain logical directory
    labels and rule text only — never absolute paths — so the status object
    is safe to expose through the health endpoint in any environment.
    """

    state: StorageState
    root_exists: bool
    root_is_directory: bool
    required_directories: bool
    message: str
    missing: tuple[str, ...] = field(default_factory=tuple)
    conflicts: tuple[str, ...] = field(default_factory=tuple)
    violations: tuple[str, ...] = field(default_factory=tuple)


def validate_storage(paths: StoragePaths | None) -> StorageStatus:
    """Check storage state without modifying anything on disk.

    Never raises for ordinary filesystem faults; every failure mode maps to
    a stable state so callers (including /health) stay alive.
    """
    if paths is None:
        return StorageStatus(
            state=StorageState.NOT_CONFIGURED,
            root_exists=False,
            root_is_directory=False,
            required_directories=False,
            message="External data storage is not configured (CSWA_DATA_ROOT is unset).",
        )

    try:
        return _validate_configured_storage(paths)
    except (OSError, RuntimeError, ValueError):
        # A permission fault, reparse loop, or malformed path must produce a
        # stable status, never a 500 — and never an absolute path.
        return StorageStatus(
            state=StorageState.UNAVAILABLE,
            root_exists=False,
            root_is_directory=False,
            required_directories=False,
            message="The storage state could not be determined safely.",
        )


def _validate_configured_storage(paths: StoragePaths) -> StorageStatus:
    violations = check_storage_policy(paths)
    if violations:
        return StorageStatus(
            state=StorageState.MISCONFIGURED,
            root_exists=paths.data_root.exists(),
            root_is_directory=paths.data_root.is_dir(),
            required_directories=False,
            message="Storage configuration violates the safety policy; refusing to use it.",
            violations=tuple(violations),
        )

    root = paths.data_root
    root_exists = root.exists()
    root_is_directory = root.is_dir()
    if not root_exists:
        # ROOT_MISSING must mean the root can legitimately be created: the
        # immediate parent has to be an existing directory. A parent that is
        # a regular file, is missing, is inaccessible, or is a dangling
        # reparse entry is not a "just run --create" situation; it is
        # unavailable (review 0004, I01-C).
        if root.parent.is_dir():
            return StorageStatus(
                state=StorageState.ROOT_MISSING,
                root_exists=False,
                root_is_directory=False,
                required_directories=False,
                message=(
                    "The storage drive is reachable but the data root does not exist; "
                    "run 'python -m services.storage.setup --create'."
                ),
            )
        return StorageStatus(
            state=StorageState.UNAVAILABLE,
            root_exists=False,
            root_is_directory=False,
            required_directories=False,
            message="Configured external data storage is not accessible.",
        )
    if not root_is_directory:
        return StorageStatus(
            state=StorageState.ROOT_NOT_DIRECTORY,
            root_exists=True,
            root_is_directory=False,
            required_directories=False,
            message="Configured storage root exists but is not a directory.",
        )

    missing: list[str] = []
    conflicts: list[str] = []
    for label, path in sorted(paths.required_directories().items()):
        if path.is_dir():
            continue
        if path.exists():
            conflicts.append(label)
        else:
            missing.append(label)

    if conflicts:
        return StorageStatus(
            state=StorageState.DEGRADED,
            root_exists=True,
            root_is_directory=True,
            required_directories=False,
            message=(
                "Some required storage locations exist but are not directories; "
                "resolve the conflicts manually."
            ),
            missing=tuple(missing),
            conflicts=tuple(conflicts),
        )
    if missing:
        return StorageStatus(
            state=StorageState.DEGRADED,
            root_exists=True,
            root_is_directory=True,
            required_directories=False,
            message=(
                "Storage root is accessible but required directories are missing; "
                "run 'python -m services.storage.setup --create'."
            ),
            missing=tuple(missing),
        )
    return StorageStatus(
        state=StorageState.AVAILABLE,
        root_exists=True,
        root_is_directory=True,
        required_directories=True,
        message="External data storage is available.",
    )


def _ancestor_chain(root: Path, path: Path) -> list[Path]:
    """Return ``path`` and each ancestor up to and including ``root``."""
    node = path
    chain: list[Path] = []
    while True:
        chain.append(node)
        if node == root or node.parent == node:
            break
        node = node.parent
    return chain


def _blocking_conflicts(root: Path, targets: dict[str, Path]) -> list[str]:
    """Find every location that would block safe creation, before any mkdir.

    Checks each target directory AND every ancestor between it and the root.
    A location blocks creation when it exists in the namespace but is not a
    usable directory — i.e. a regular file, or a dangling/broken symlink or
    Windows junction/reparse entry (``lexists`` true but ``isdir`` false).
    Reparse-aware so a dangling alias can never be discovered mid-creation.
    """
    conflicts: list[str] = []
    for label, path in sorted(targets.items()):
        for candidate in _ancestor_chain(root, path):
            try:
                if not os.path.lexists(candidate):
                    continue  # absent: will be created
                if candidate.is_dir():
                    continue  # a real directory (or a valid junction to one)
                conflicts.append(label)  # regular file or dangling reparse
                break
            except OSError:
                conflicts.append(label)  # unreadable/undeterminable: fail closed
                break
    return conflicts


def _rollback(created: list[Path], root: Path) -> None:
    """Remove only the empty directories this call created; never anything else.

    Removes in reverse creation order so children go before parents. Uses
    ``rmdir`` exclusively, which refuses non-empty directories, so a
    pre-existing file or populated directory is never touched. Rollback
    failures are swallowed so the original error is preserved.
    """
    for path in reversed(created):
        try:
            if path != root and not (root == path or root in path.parents):
                continue  # never step outside the root
            if os.path.islink(path):
                continue  # never remove a link we did not create as a dir
            path.rmdir()
        except OSError:
            pass  # leave what cannot be safely removed; keep the original error


def ensure_runtime_directories(paths: StoragePaths | None) -> list[Path]:
    """Create missing runtime directories; never delete or overwrite anything.

    Order of operations (the complete preflight finishes before the first
    ``mkdir``):

    1. configuration check;
    2. full policy validation on canonicalized paths — this already rejects
       ``..``, non-absolute paths, illegal/Windows-invalid path components,
       reserved device names, control characters, out-of-root/repository
       locations, layer overlap, and nested-owner escapes;
    3. root reachability (parent must be an existing directory);
    4. reparse-aware conflict preflight over every target directory and each
       of its ancestors (regular files and dangling reparse entries).

    Only when everything is preflighted safe does creation begin. If an
    unforeseeable runtime failure still occurs during ``mkdir`` (e.g. an
    injected permission error or a race), a bounded rollback removes only the
    empty directories this call created, leaving the tree exactly as it was.
    Returns the directories actually created.
    """
    if paths is None:
        raise StorageNotConfiguredError(
            "External data storage is not configured; set CSWA_DATA_ROOT (and optional "
            "CSWA_*_DIR overrides) in the environment or .env file. Refusing to fall "
            "back to the repository data/ directory."
        )

    violations = check_storage_policy(paths)
    if violations:
        raise StoragePolicyError(
            "storage configuration violates the safety policy; nothing was created: "
            + "; ".join(violations)
        )

    root = paths.data_root
    targets = paths.required_directories()

    # --- preflight (no filesystem mutation happens in this block) ---
    try:
        root_exists = root.exists()
        if root_exists and not root.is_dir():
            raise StoragePathConflictError(
                "storage root exists but is not a directory; refusing to touch it"
            )
        if not root_exists and not root.parent.is_dir():
            raise StorageNotAvailableError(
                "cannot create the storage root: its parent location is not an "
                "existing directory. Is the external SSD connected and mounted?"
            )
        conflicts = _blocking_conflicts(root, targets)
    except StorageError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise StorageNotAvailableError(
            "the storage layout could not be preflighted safely; nothing was created."
        ) from exc

    if conflicts:
        raise StoragePathConflictError(
            "refusing to create anything: the following locations are blocked by "
            "existing files or dangling reparse entries: " + ", ".join(sorted(set(conflicts)))
        )

    # --- creation (per-ancestor, tracked for bounded rollback) ---
    created: list[Path] = []
    try:
        if not root_exists:
            root.mkdir()
            created.append(root)
        for _, path in sorted(targets.items()):
            for candidate in reversed(_ancestor_chain(root, path)):
                if candidate == root:
                    continue
                if not os.path.lexists(candidate):
                    candidate.mkdir()
                    created.append(candidate)
    except (OSError, RuntimeError, ValueError) as exc:
        _rollback(created, root)
        raise StorageNotAvailableError(
            "a filesystem error occurred while creating directories; every directory "
            "created by this call was rolled back and pre-existing content was left intact."
        ) from exc
    return created


def require_storage_for(feature: str, paths: StoragePaths | None) -> dict[str, Path]:
    """Read-only gate a feature must pass before touching storage.

    Applies the safety policy and the feature's explicit directory/permission
    matrix (existence, readability, writability). Returns the label -> path
    mapping for the feature on success.

    Raises:
        KeyError: unknown feature name (a programming error, not a runtime state).
        StorageNotConfiguredError / StoragePolicyError / StorageNotAvailableError
    """
    requirements = FEATURE_REQUIREMENTS[feature]

    if paths is None:
        raise StorageNotConfiguredError(
            f"feature '{feature}' requires external data storage, which is not "
            "configured; refusing to fall back to the repository or a temporary directory."
        )

    violations = check_storage_policy(paths)
    if violations:
        raise StoragePolicyError(
            f"feature '{feature}' refused: storage configuration violates the safety "
            "policy: " + "; ".join(violations)
        )

    directories = paths.required_directories()
    result: dict[str, Path] = {}
    try:
        for requirement in requirements:
            path = directories[requirement.label]
            if requirement.must_exist and not path.is_dir():
                raise StorageNotAvailableError(
                    f"feature '{feature}' requires storage directory "
                    f"'{requirement.label}', which is not accessible. "
                    "Is the external SSD connected?"
                )
            if requirement.readable and not directory_is_readable(path):
                raise StorageNotAvailableError(
                    f"feature '{feature}' requires read access to storage directory "
                    f"'{requirement.label}', which is not readable."
                )
            if requirement.writable and not directory_is_writable(path):
                raise StorageNotAvailableError(
                    f"feature '{feature}' requires write access to storage directory "
                    f"'{requirement.label}', which is not writable."
                )
            result[requirement.label] = path
    except StorageError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise StorageNotAvailableError(
            f"feature '{feature}' refused: the storage state could not be determined safely."
        ) from exc
    return result
