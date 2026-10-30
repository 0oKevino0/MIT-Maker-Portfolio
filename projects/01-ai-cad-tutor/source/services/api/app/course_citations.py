"""Canonical Course citation-marker parsing and relationship validation.

The public marker syntax is exactly ``[citation_id]`` where ``citation_id``
is a 1--64 character ASCII identifier.  A marker whose opening bracket is
preceded by an odd number of backslashes is escaped.  Ordinary bracketed
prose or expressions containing spaces (for example ``[i + 1]``) are not
markers.  An outer ordinary bracket never swallows a later unescaped opening
bracket, so a nested canonical marker is always discovered or explicitly
rejected (review 0026 IF-01).

This module deliberately knows nothing about model providers or server
evidence metadata.  Callers decide whether a validation failure belongs to
the upstream-model 502 boundary or to the server-assembly 500 boundary.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

_CITATION_ID_RE = re.compile(
    r"(?:c[0-9]+|[A-Za-z0-9][A-Za-z0-9._:-]*[._:-][A-Za-z0-9][A-Za-z0-9._:-]*)\Z",
    re.ASCII,
)


class CitationMarkerError(ValueError):
    """Citation ID, marker syntax, or marker relationship is invalid."""


def is_canonical_citation_id(value: object) -> bool:
    """Return whether *value* is a canonical, case-sensitive citation ID."""
    return (
        isinstance(value, str) and len(value) <= 64 and _CITATION_ID_RE.fullmatch(value) is not None
    )


def require_canonical_citation_id(value: object) -> str:
    """Return a canonical citation ID or raise a content-free error."""
    if not is_canonical_citation_id(value):
        raise CitationMarkerError("citation id is not canonical")
    return value


def _opening_bracket_is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _next_unescaped_opening(text: str, start: int, end: int) -> int:
    """Return the index of the first unescaped ``[`` in ``text[start:end]``."""
    cursor = start
    while cursor < end:
        candidate = text.find("[", cursor, end)
        if candidate < 0:
            return -1
        if not _opening_bracket_is_escaped(text, candidate):
            return candidate
        cursor = candidate + 1
    return -1


def _is_malformed_marker_candidate(content: str) -> bool:
    """Identify malformed marker-like tokens without claiming normal prose."""
    if not content:
        return True
    if any(ord(character) < 32 or ord(character) == 127 for character in content):
        return True
    if " " in content:
        return False
    if any(ord(character) > 127 for character in content):
        return True
    if len(content) > 64:
        return True
    # Citation IDs are distinguishable from ordinary variables/words: they
    # are either compact labels such as c1 or namespaced IDs containing one
    # of the established separators. Other printable tokens are ordinary
    # bracketed text rather than citation candidates.
    return any(separator in content for separator in "._:-")


def parse_citation_markers(text: str) -> list[str]:
    """Return unique canonical IDs in first unescaped marker occurrence order.

    Empty, overlong, non-ASCII, control-bearing, or otherwise marker-like
    single tokens fail closed.  An incomplete single-token marker candidate
    also fails closed; incomplete ordinary expressions containing spaces are
    ignored as non-markers.
    """
    if not isinstance(text, str):
        raise CitationMarkerError("citation marker source is not text")

    markers: list[str] = []
    seen: set[str] = set()
    cursor = 0
    while cursor < len(text):
        opening = text.find("[", cursor)
        if opening < 0:
            break
        cursor = opening + 1
        if _opening_bracket_is_escaped(text, opening):
            continue

        closing = text.find("]", cursor)
        if closing < 0:
            trailing_content = text[cursor:]
            if is_canonical_citation_id(trailing_content) or _is_malformed_marker_candidate(
                trailing_content
            ):
                raise CitationMarkerError("answer contains an incomplete citation marker")
            break

        # An outer bracket must never swallow a later unescaped opening
        # bracket (review 0026 IF-01): a nested canonical marker has to be
        # discovered.  If the text before the inner opening is itself
        # marker-like, this is an incomplete marker candidate and fails
        # closed; otherwise scanning continues FROM the inner opening, so
        # ordinary nested prose stays non-matching while every unescaped
        # inner candidate is still examined.
        inner_opening = _next_unescaped_opening(text, cursor, closing)
        if inner_opening >= 0:
            prefix = text[cursor:inner_opening]
            if prefix and (
                is_canonical_citation_id(prefix) or _is_malformed_marker_candidate(prefix)
            ):
                raise CitationMarkerError("answer contains an incomplete citation marker")
            cursor = inner_opening
            continue

        content = text[cursor:closing]
        cursor = closing + 1

        if is_canonical_citation_id(content):
            if content not in seen:
                seen.add(content)
                markers.append(content)
            continue
        if _is_malformed_marker_candidate(content):
            raise CitationMarkerError("answer contains a malformed citation marker")

    return markers


def validate_citation_relationship(
    answer_text: str,
    selected_ids: Sequence[str],
    allowed_ids: Collection[str],
    *,
    require_selected_order: bool = False,
) -> list[str]:
    """Validate whitelist membership and bidirectional ID/marker consistency.

    The returned list is the stable, first-occurrence marker order used for
    server-side Citation assembly.
    """
    if isinstance(selected_ids, (str, bytes)):
        raise CitationMarkerError("selected citation ids are not a sequence of ids")

    selected: list[str] = []
    selected_seen: set[str] = set()
    for value in selected_ids:
        citation_id = require_canonical_citation_id(value)
        if citation_id in selected_seen:
            raise CitationMarkerError("selected citation ids are not unique")
        selected_seen.add(citation_id)
        selected.append(citation_id)

    allowed: set[str] = set()
    for value in allowed_ids:
        allowed.add(require_canonical_citation_id(value))
    if not selected_seen.issubset(allowed):
        raise CitationMarkerError("selected citation id is outside the evidence whitelist")

    markers = parse_citation_markers(answer_text)
    marker_set = set(markers)
    if not marker_set.issubset(allowed):
        raise CitationMarkerError("citation marker is outside the evidence whitelist")
    if marker_set != selected_seen:
        raise CitationMarkerError("citation markers and selected ids are inconsistent")
    if require_selected_order and markers != selected:
        raise CitationMarkerError("citation order is inconsistent with marker order")
    return markers
