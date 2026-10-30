"""External storage path layout and resolution.

The storage root comes from ``CSWA_DATA_ROOT``; every subdirectory defaults
to a well-known location under the root and can be overridden individually
by its own environment variable (see ``Settings`` in
``services.api.app.config.settings``).

This module never touches the filesystem: it only resolves paths. Validation
and directory creation live in ``services.storage.service``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Dataclass field name -> default location relative to the storage root.
# The settings field for each entry is the name prefixed with "cswa_"
# (e.g. raw_data_dir -> Settings.cswa_raw_data_dir -> env CSWA_RAW_DATA_DIR).
DERIVED_SUBDIRS: dict[str, str] = {
    "raw_data_dir": "datasets/raw",
    "processed_data_dir": "datasets/processed",
    "approved_data_dir": "datasets/approved",
    "benchmark_dir": "datasets/benchmarks",
    "synthetic_data_dir": "datasets/synthetic",
    "cad_data_dir": "cad",
    "model_dir": "models",
    "vector_db_dir": "vector_db",
    "embedding_dir": "embeddings",
    "checkpoint_dir": "checkpoints",
    "cache_dir": "cache",
    "export_dir": "exports",
    "log_dir": "logs",
    "backup_dir": "backups",
}

# Standard nested layout created inside the CAD and model directories.
CAD_SUBDIRS: tuple[str, ...] = (
    "parts",
    "assemblies",
    "drawings",
    "screenshots",
    "reference_models",
)
MODEL_SUBDIRS: tuple[str, ...] = ("text", "vision", "embedding", "reranker")


def nested_directory_owners() -> dict[str, str]:
    """Map each standard nested directory label to its owning logical layer.

    ``cad/parts`` is owned by ``cad``; ``models/text`` by ``models``; and so
    on. The safety policy uses this to require every nested directory to
    canonicalize strictly beneath its owner (review 0004, I01-A), so a
    reparse alias cannot make a model/CAD subdirectory cross into another
    protected layer.
    """
    owners: dict[str, str] = {f"cad/{sub}": "cad" for sub in CAD_SUBDIRS}
    owners.update({f"models/{sub}": "models" for sub in MODEL_SUBDIRS})
    return owners


class StorageSettingsLike(Protocol):
    """The storage-related fields of the application Settings object."""

    cswa_data_root: Path | None
    cswa_raw_data_dir: Path | None
    cswa_processed_data_dir: Path | None
    cswa_approved_data_dir: Path | None
    cswa_benchmark_dir: Path | None
    cswa_synthetic_data_dir: Path | None
    cswa_cad_data_dir: Path | None
    cswa_model_dir: Path | None
    cswa_vector_db_dir: Path | None
    cswa_embedding_dir: Path | None
    cswa_checkpoint_dir: Path | None
    cswa_cache_dir: Path | None
    cswa_export_dir: Path | None
    cswa_log_dir: Path | None
    cswa_backup_dir: Path | None


@dataclass(frozen=True)
class StoragePaths:
    """Fully resolved external storage layout."""

    data_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    approved_data_dir: Path
    benchmark_dir: Path
    synthetic_data_dir: Path
    cad_data_dir: Path
    model_dir: Path
    vector_db_dir: Path
    embedding_dir: Path
    checkpoint_dir: Path
    cache_dir: Path
    export_dir: Path
    log_dir: Path
    backup_dir: Path

    def required_directories(self) -> dict[str, Path]:
        """All directories the runtime layout expects, keyed by logical name.

        Keys are stable logical labels (the default relative locations), never
        absolute paths, so they are safe to expose in health responses even
        when an operator overrides a directory to a custom location.
        """
        directories: dict[str, Path] = {
            label: getattr(self, field_name) for field_name, label in DERIVED_SUBDIRS.items()
        }
        for sub in CAD_SUBDIRS:
            directories[f"cad/{sub}"] = self.cad_data_dir / sub
        for sub in MODEL_SUBDIRS:
            directories[f"models/{sub}"] = self.model_dir / sub
        return directories


def resolve_storage_paths(settings: StorageSettingsLike) -> StoragePaths | None:
    """Resolve the storage layout from settings.

    Returns None when ``CSWA_DATA_ROOT`` is not configured — storage is then
    "not configured" and callers must refuse data-dependent work instead of
    falling back to the git repository or a temporary directory.

    No drive-letter guessing and no Windows/WSL2 path conversion happens
    here: each environment supplies its own paths via its own ``.env``.
    """
    root = settings.cswa_data_root
    if root is None:
        return None
    root = Path(root)

    resolved: dict[str, Path] = {"data_root": root}
    for field_name, relative in DERIVED_SUBDIRS.items():
        override = getattr(settings, f"cswa_{field_name}")
        resolved[field_name] = Path(override) if override is not None else root / relative
    return StoragePaths(**resolved)
