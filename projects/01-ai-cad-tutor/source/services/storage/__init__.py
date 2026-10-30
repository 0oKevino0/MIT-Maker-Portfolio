"""External data storage layout, validation, and safe directory creation.

Large or private content (course material, CAD files, model weights, vector
databases, checkpoints, caches, exports, backups) lives on an external drive
configured through environment variables — never inside the git repository.
See docs/development/EXTERNAL_STORAGE.md.
"""

from services.storage.errors import (
    StorageError,
    StorageNotAvailableError,
    StorageNotConfiguredError,
    StoragePathConflictError,
    StoragePolicyError,
)
from services.storage.paths import (
    CAD_SUBDIRS,
    DERIVED_SUBDIRS,
    MODEL_SUBDIRS,
    StoragePaths,
    resolve_storage_paths,
)
from services.storage.policy import ISOLATED_LAYER_LABELS, check_storage_policy
from services.storage.service import (
    FEATURE_REQUIREMENTS,
    DirectoryRequirement,
    StorageState,
    StorageStatus,
    ensure_runtime_directories,
    require_storage_for,
    validate_storage,
)

__all__ = [
    "CAD_SUBDIRS",
    "DERIVED_SUBDIRS",
    "FEATURE_REQUIREMENTS",
    "ISOLATED_LAYER_LABELS",
    "MODEL_SUBDIRS",
    "DirectoryRequirement",
    "StorageError",
    "StorageNotAvailableError",
    "StorageNotConfiguredError",
    "StoragePathConflictError",
    "StoragePaths",
    "StoragePolicyError",
    "StorageState",
    "StorageStatus",
    "check_storage_policy",
    "ensure_runtime_directories",
    "require_storage_for",
    "resolve_storage_paths",
    "validate_storage",
]
