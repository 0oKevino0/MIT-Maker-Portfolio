"""Storage layout safety policy (review issues I-01; hardened per review 0004).

Side-effect-free checks that run before any directory is created and before
any feature uses a storage location. Policy for this MVP:

- one canonical data root: every configured directory (including every
  explicit override) must canonicalize to a location strictly inside
  ``CSWA_DATA_ROOT``;
- every path must be absolute for the current OS, free of ``..`` segments,
  and composed only of legal path components (Windows-invalid characters,
  trailing dots/spaces, reserved device names, and control characters are
  rejected up front — see ``services.storage.pathcheck``);
- no path may be the repository, live inside it, or contain it;
- all fourteen logical data layers must be pairwise distinct and non-nested;
- **every standard nested directory belongs to exactly one canonical owner**
  (``cad/*`` inside the cad layer, ``models/*`` inside the models layer,
  review 0004 I01-A): after canonicalization it must sit strictly beneath
  its owner, must not equal it, must not overlap any other logical layer,
  and must not equal or nest with any sibling nested directory — so a
  junction/symlink at ``models/text`` can never alias the approved course
  layer, another layer, the repository, or anywhere outside the root;
- checks run on canonicalized paths (``Path.resolve()``), so symlinks,
  Windows junctions, and other reparse-point aliases — including dangling
  ones, whose lexical targets still resolve — cannot defeat them for the
  parts of a path that exist at validation time (a residual
  time-of-check/time-of-use window remains and is documented).

Canonicalization failures (permission errors, reparse loops, malformed
paths) are converted into stable policy violations — they must never
surface as unhandled exceptions or HTTP 500s. All reported violations use
logical directory labels and rule text only, never absolute path values.
"""

from __future__ import annotations

import os
from pathlib import Path

from services.storage.pathcheck import component_violations
from services.storage.paths import DERIVED_SUBDIRS, StoragePaths, nested_directory_owners

# services/storage/policy.py -> services/storage -> services -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The fourteen logical data layers that must stay pairwise isolated.
ISOLATED_LAYER_LABELS: tuple[str, ...] = tuple(DERIVED_SUBDIRS.values())

# Standard nested directories and the logical layer that owns each of them.
NESTED_DIRECTORY_OWNERS: dict[str, str] = nested_directory_owners()


def _safe_resolve(path: Path) -> Path | None:
    """Canonicalize a path, returning None on any filesystem/parsing failure.

    ``Path.resolve()`` can raise ``OSError`` (permissions, reparse loops,
    invalid names) or, on some platforms, ``RuntimeError``/``ValueError``.
    Callers convert None into a stable, path-free policy violation.
    """
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _is_within(path: Path, ancestor: Path) -> bool:
    return path == ancestor or ancestor in path.parents


def check_storage_policy(paths: StoragePaths, repo_root: Path | None = None) -> list[str]:
    """Return policy violations as label-based messages; empty when safe.

    Read-only: never creates, deletes, or modifies anything.
    """
    # _REPO_ROOT is canonicalized once at import; a caller-supplied anchor is
    # canonicalized defensively so a resolution fault cannot escape.
    repo = _REPO_ROOT if repo_root is None else (_safe_resolve(repo_root) or repo_root)
    violations: list[str] = []

    # --- canonicalize the root first; everything else is judged against it.
    root = paths.data_root
    root_resolved: Path | None = None
    if any(part == ".." for part in root.parts):
        violations.append("data_root: path must not contain '..' segments")
    elif not root.is_absolute():
        violations.append("data_root: path must be absolute")
    else:
        for issue in component_violations(root):
            violations.append(f"data_root: {issue}")
        root_resolved = _safe_resolve(root)
        if root_resolved is None:
            violations.append("data_root: path could not be canonicalized safely")
        elif _is_within(root_resolved, repo):
            violations.append("data_root: path must not be inside the code repository")
        elif _is_within(repo, root_resolved):
            violations.append("data_root: path must not contain the code repository")

    # --- every configured directory (defaults and overrides alike).
    resolved: dict[str, Path] = {}
    for label, path in paths.required_directories().items():
        if any(part == ".." for part in path.parts):
            violations.append(f"{label}: path must not contain '..' segments")
            continue
        if not path.is_absolute():
            violations.append(f"{label}: path must be absolute")
            continue
        for issue in component_violations(path):
            violations.append(f"{label}: {issue}")
        # Lexical containment (review 0006 F-01): the configured path must be a
        # strict lexical descendant of the configured root, so a junction/alias
        # whose *lexical* location is outside the root cannot be accepted merely
        # because its *canonical* target resolves inside the root. Policy,
        # preflight, and rollback then share one path identity.
        if path == root or root not in path.parents:
            violations.append(f"{label}: configured path must stay lexically inside the data root")
        real = _safe_resolve(path)
        if real is None:
            violations.append(f"{label}: path could not be canonicalized safely")
            continue
        resolved[label] = real
        if _is_within(real, repo):
            violations.append(f"{label}: path must not be inside the code repository")
        elif _is_within(repo, real):
            violations.append(f"{label}: path must not contain the code repository")
        if root_resolved is not None:
            if real == root_resolved:
                violations.append(f"{label}: directory must not equal the data root")
            elif root_resolved not in real.parents:
                violations.append(f"{label}: directory must stay inside the data root")

    # --- pairwise isolation of all fourteen logical layers.
    layers = [(label, resolved[label]) for label in ISOLATED_LAYER_LABELS if label in resolved]
    for index, (label_a, path_a) in enumerate(layers):
        for label_b, path_b in layers[index + 1 :]:
            if path_a == path_b:
                violations.append(f"{label_a} and {label_b}: directories must not be the same")
            elif _is_within(path_a, path_b) or _is_within(path_b, path_a):
                violations.append(f"{label_a} and {label_b}: directories must not be nested")

    # --- canonical-owner containment for every standard nested directory
    #     (review 0004 I01-A: a models/* or cad/* alias must never cross
    #     into another layer, a sibling, the repository, or out of the root).
    nested = [
        (label, resolved[label], owner)
        for label, owner in NESTED_DIRECTORY_OWNERS.items()
        if label in resolved
    ]
    for label, real, owner_label in nested:
        owner_resolved = resolved.get(owner_label)
        if owner_resolved is None:
            continue  # the owner itself already reported a violation
        if real == owner_resolved:
            violations.append(f"{label}: directory must not equal its owning layer")
        elif owner_resolved not in real.parents:
            violations.append(
                f"{label}: directory must stay inside its owning layer '{owner_label}'"
            )
        for layer_label, layer_resolved in layers:
            if layer_label == owner_label:
                continue
            if real == layer_resolved:
                violations.append(f"{label} and {layer_label}: directories must not be the same")
            elif _is_within(real, layer_resolved) or _is_within(layer_resolved, real):
                violations.append(f"{label} and {layer_label}: directories must not be nested")
    for index, (label_a, real_a, _) in enumerate(nested):
        for label_b, real_b, _ in nested[index + 1 :]:
            if real_a == real_b:
                violations.append(f"{label_a} and {label_b}: directories must not be the same")
            elif _is_within(real_a, real_b) or _is_within(real_b, real_a):
                violations.append(f"{label_a} and {label_b}: directories must not be nested")

    return violations


def directory_is_readable(path: Path) -> bool:
    """Best-effort, read-only readability probe (no file is ever touched)."""
    return os.access(path, os.R_OK)


def directory_is_writable(path: Path) -> bool:
    """Best-effort, read-only writability probe (no file is ever created)."""
    return os.access(path, os.W_OK)
