"""Path-component legality checks (review 0004, I01-B).

Validates every component of a configured storage path *before* any
filesystem operation, so an uncreatable name can never fail halfway through
directory creation.

Rules:

- on every platform: control characters (U+0000..U+001F, U+007F) are
  rejected;
- under Windows semantics: the characters ``< > : " | ? *``, trailing
  spaces, trailing periods, and the reserved device names (``CON``, ``PRN``,
  ``AUX``, ``NUL``, ``COM1``-``COM9``, ``LPT1``-``LPT9``, with or without an
  extension) are rejected.

Windows rules apply only when Windows semantics are in effect (the current
platform by default); normal POSIX names are never falsely rejected on
POSIX hosts. Violation strings contain rule descriptions only — never the
offending component or any absolute path.
"""

from __future__ import annotations

import os
from pathlib import Path

WINDOWS_FORBIDDEN_CHARS = frozenset('<>:"|?*')

WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _component_parts(path: Path) -> list[str]:
    """Path components excluding the drive/root anchor."""
    parts = list(path.parts)
    if parts and (path.drive or path.root):
        return parts[1:]
    return parts


def component_violations(path: Path, *, windows_semantics: bool | None = None) -> list[str]:
    """Return rule-name violations for illegal path components.

    Messages never include the component value or the path itself.
    """
    windows = (os.name == "nt") if windows_semantics is None else windows_semantics
    issues: set[str] = set()

    for component in _component_parts(path):
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in component):
            issues.add("path component contains control characters")
        if not windows:
            continue
        if any(ch in WINDOWS_FORBIDDEN_CHARS for ch in component):
            issues.add("path component contains characters invalid on Windows")
        if component.endswith(" "):
            issues.add("path component ends with a space, which is invalid on Windows")
        if component.endswith("."):
            issues.add("path component ends with a period, which is invalid on Windows")
        base = component.split(".", 1)[0].rstrip(" ").upper()
        if base in WINDOWS_RESERVED_NAMES:
            issues.add("path component is a reserved Windows device name")

    return sorted(issues)
