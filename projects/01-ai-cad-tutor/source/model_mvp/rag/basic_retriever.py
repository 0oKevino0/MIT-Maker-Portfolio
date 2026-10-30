"""Deterministic lexical retriever (MODEL-MVP-M3A1, scorer ``m3a_lexical_v1``).

Implements the frozen 0151 retrieval algorithm exactly: NFKC + casefold
normalization, maximal ASCII ``[0-9a-z]+`` tokens, unique-token presence, the
frozen 0152 generic-English stopword set, integer document-frequency band
scoring, the minimum-evidence gate, and the three-level deterministic
tie-break.

Dependency-free (standard library only): no embeddings, no learned weights, no
term-frequency weighting, no fuzzy matching, no phrase rules, and no
benchmark-keyed behavior of any kind. The retriever receives ONLY a
:class:`~model_mvp.rag.protocol.RagQuery`; it imports no benchmark or
evaluation code and holds no evaluation-item lookup table.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from model_mvp.rag.corpus import CORPUS_PATH, ConceptRecord, load_corpus
from model_mvp.rag.protocol import RagQuery, RetrievedChunk

#: Frozen semantic scorer version (0151 §31). A behavioral change requires a new
#: version, a new reviewed spec hash, and a new retrieval profile.
SCORER_SEMANTIC_VERSION = "m3a_lexical_v1"

#: The exact 45 generic-English stopwords frozen by review 0152. Not tunable.
STOPWORDS: frozenset[str] = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "is",
        "it",
        "its",
        "may",
        "must",
        "no",
        "not",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "they",
        "this",
        "to",
        "was",
        "when",
        "which",
        "while",
        "will",
        "with",
        "you",
        "your",
    )
)

_TOKEN_RE = re.compile(r"[0-9a-z]+")


class RetrievalQueryError(ValueError):
    """The retrieval query was not a valid ``RagQuery``. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class RetrievalNoResultError(ValueError):
    """No corpus record passed the frozen eligibility gate. ``code`` is stable.

    Fails closed by design (0151 §23): no zero-score context injection, no
    Stage-B fallback, no alternate corpus, no cloud fallback, and no inference
    call may follow.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def serialize_stopwords() -> bytes:
    """Serialize the frozen stopword set per the 0152 contract.

    One ASCII lowercase word per line, lexicographically ascending, UTF-8, LF
    newlines, no BOM, terminal LF. The SHA256 of these bytes is the frozen
    stopword-list identity.
    """
    return ("\n".join(sorted(STOPWORDS)) + "\n").encode("utf-8")


def tokenize(text: str) -> tuple[str, ...]:
    """Frozen tokenizer (0151 §17): NFKC, casefold, ``[0-9a-z]+`` runs, unique, destop.

    Returns unique scoring-eligible tokens in first-occurrence order. Underscore,
    hyphen, slash, apostrophe, punctuation, and any residual non-ASCII character
    act as separators; digits are retained; empty tokens are never emitted.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    seen: dict[str, None] = {}
    for token in _TOKEN_RE.findall(normalized):
        if token not in STOPWORDS:
            seen.setdefault(token, None)
    return tuple(seen)


def document_frequencies(records: tuple[ConceptRecord, ...]) -> dict[str, int]:
    """Document frequency of each token over the records' retrieval texts."""
    frequencies: dict[str, int] = {}
    for record in records:
        for token in set(tokenize(record.retrieval_text)):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def token_points(df: int, corpus_size: int) -> int:
    """Frozen integer band scoring (0151 §18): 2 iff ``8*df <= N``; else 1 iff ``2*df <= N``."""
    if 8 * df <= corpus_size:
        return 2
    if 2 * df <= corpus_size:
        return 1
    return 0


@dataclass(frozen=True, slots=True)
class _Scored:
    """Internal scoring result for one record (never model-visible)."""

    record: ConceptRecord
    score: int
    min_eligible_df: int


@dataclass(frozen=True, slots=True)
class BasicLexicalRetriever:
    """Deterministic lexical ``RagRetriever`` over the frozen experimental corpus.

    Construction performs local file I/O only (corpus load). ``retrieve`` accepts
    a ``RagQuery`` and nothing else — no benchmark item id, target term,
    reference answer, expected class, model answer, or evaluation metadata.
    """

    records: tuple[ConceptRecord, ...]
    _record_tokens: tuple[frozenset[str], ...] = field(repr=False)
    _df: dict[str, int] = field(repr=False)

    @classmethod
    def from_records(cls, records: tuple[ConceptRecord, ...]) -> BasicLexicalRetriever:
        return cls(
            records=records,
            _record_tokens=tuple(frozenset(tokenize(r.retrieval_text)) for r in records),
            _df=document_frequencies(records),
        )

    @classmethod
    def from_corpus(cls, path: str | None = None) -> BasicLexicalRetriever:
        return cls.from_records(load_corpus(path if path is not None else CORPUS_PATH))

    @property
    def corpus_size(self) -> int:
        return len(self.records)

    def _score(self, query_tokens: frozenset[str]) -> list[_Scored]:
        corpus_size = self.corpus_size
        scored: list[_Scored] = []
        for record, record_tokens in zip(self.records, self._record_tokens, strict=True):
            score = 0
            eligible_dfs: list[int] = []
            for token in query_tokens & record_tokens:
                df = self._df[token]
                points = token_points(df, corpus_size)
                if points >= 1:  # scoring-eligible token
                    score += points
                    eligible_dfs.append(df)
            # Frozen minimum-evidence gate (0151 §19): >= 2 distinct scoring-eligible
            # matched tokens AND a positive integer score. No other threshold exists.
            if len(eligible_dfs) >= 2 and score > 0:
                scored.append(
                    _Scored(record=record, score=score, min_eligible_df=min(eligible_dfs))
                )
        return scored

    def _ranked(self, query: RagQuery) -> list[_Scored]:
        """The single frozen scoring computation + frozen ordering (shared path).

        Exactly the historical ``retrieve`` internals, extracted so the public
        result and the M3B3 evidence ranking come from ONE computation — no
        second scoring route exists that could drift.
        """
        if not isinstance(query, RagQuery):
            raise RetrievalQueryError("query_invalid")
        scored = self._score(frozenset(tokenize(query.query_text)))
        if not scored:
            raise RetrievalNoResultError("no_eligible_records")
        # Frozen tie-break (0151 §21): score desc, then minimum eligible df asc,
        # then concept_id ascending by code point. No randomness, no locale.
        scored.sort(key=lambda s: (-s.score, s.min_eligible_df, s.record.concept_id))
        return scored

    @staticmethod
    def _to_chunk(s: _Scored) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=s.record.concept_id,
            text=s.record.model_visible_text,
            score=s.score,
            source_class=s.record.source_class,
            source_ref=s.record.source_ref,
            metadata={},
        )

    def retrieve(self, query: RagQuery) -> tuple[RetrievedChunk, ...]:
        """Return up to ``query.top_k`` chunks in frozen deterministic rank order."""
        return tuple(self._to_chunk(s) for s in self._ranked(query)[: query.top_k])

    def retrieve_with_full_ranking(
        self, query: RagQuery
    ) -> tuple[tuple[RetrievedChunk, ...], tuple[tuple[str, int], ...]]:
        """M3B3 evidence accessor (0155 §27): the frozen ``retrieve`` result PLUS
        the complete ELIGIBLE ranked list, from ONE scoring computation.

        ``query.top_k`` semantics are unchanged (the caller still requests the
        frozen experimental top_k). The first element is exactly what
        ``retrieve`` returns for the same query; the second is the full
        authoritative eligible ranking as ordered ``(concept_id, score)``
        pairs — OFFLINE EVIDENCE ONLY, never rendered into any model-visible
        context, user turn, or prompt. Eligibility, scoring, ordering, and
        failure semantics (including ``no_eligible_records``) are byte-for-byte
        the production ``retrieve`` semantics.
        """
        ranked = self._ranked(query)
        chunks = tuple(self._to_chunk(s) for s in ranked[: query.top_k])
        full = tuple((s.record.concept_id, s.score) for s in ranked)
        return chunks, full
