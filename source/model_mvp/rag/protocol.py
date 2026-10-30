"""Backend-independent RAG retrieval protocol (MODEL-MVP-M0, design-only).

Defines the retrieval interface the RAG Alpha phase will implement. It does NOT
build an index, load an embedding model, or read any formal course corpus. A
retrieved chunk only ever references EXPERIMENTAL, review-cleared material and
never claims formal admission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class RagProtocolError(ValueError):
    """A RAG record violated the protocol. ``code`` is a stable token."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One retrieved evidence chunk. Carries a source class, never a raw path."""

    chunk_id: str
    text: str
    score: float
    source_class: str
    source_ref: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.chunk_id) is not str or not self.chunk_id.strip():
            raise RagProtocolError("chunk_id_invalid")
        if type(self.text) is not str or not self.text.strip():
            raise RagProtocolError("chunk_text_invalid")
        if type(self.score) not in (int, float):
            raise RagProtocolError("chunk_score_invalid")
        # Experimental-only provenance; a formal/real-exam source is rejected.
        allowed = {
            "approved_experimental_material",
            "instructor_authored",
            "synthetic_reviewed",
            "original",
        }
        if self.source_class not in allowed:
            raise RagProtocolError("chunk_source_class_invalid")


@dataclass(frozen=True, slots=True)
class RagQuery:
    query_text: str
    top_k: int = 5

    def __post_init__(self) -> None:
        if type(self.query_text) is not str or not self.query_text.strip():
            raise RagProtocolError("query_text_invalid")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise RagProtocolError("query_top_k_invalid")


@runtime_checkable
class RagRetriever(Protocol):
    """Retrieve evidence chunks for a query. Implemented in the RAG Alpha phase."""

    def retrieve(self, query: RagQuery) -> tuple[RetrievedChunk, ...]: ...
