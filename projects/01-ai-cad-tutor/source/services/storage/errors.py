"""Storage error types.

Kept in their own module so both the service layer and the policy layer can
import them without circular imports.
"""


class StorageError(Exception):
    """Base class for storage configuration and availability errors."""


class StorageNotConfiguredError(StorageError):
    """CSWA_DATA_ROOT is not set; data-dependent features must refuse to run."""


class StorageNotAvailableError(StorageError):
    """The configured storage location is not reachable (SSD disconnected?)."""


class StoragePathConflictError(StorageError):
    """A required directory path exists but is a file; refusing to touch it."""


class StoragePolicyError(StorageError):
    """The configured storage layout violates the safety policy (I-01).

    Raised before any directory is created. The message contains logical
    directory labels and policy rule names only — never absolute paths.
    """
