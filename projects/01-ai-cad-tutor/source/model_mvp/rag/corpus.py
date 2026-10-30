"""Experimental retrieval concept corpus loader (MODEL-MVP-M3A1).

Loads the committed, frozen experimental corpus
(``model_mvp/rag/solidworks_concepts_v0.jsonl``) whose exact field values were
frozen by review 0152. The loader is FAIL-CLOSED: a missing file, malformed
line, unexpected field, duplicate/out-of-order ``concept_id``, closed-vocabulary
violation, or benchmark-linking leakage marker rejects the whole corpus.

This module performs no network I/O, imports no benchmark or evaluation code,
and never exposes benchmark identifiers. The corpus is **Experimental MVP
Retrieval Corpus — Not Formally Admitted Production Data**.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

#: Frozen corpus version name (review 0152).
CORPUS_VERSION = "solidworks_concepts_v0"
#: The committed corpus file materialized verbatim from the 0152 specification.
CORPUS_PATH = Path(__file__).resolve().parent / "solidworks_concepts_v0.jsonl"
#: The frozen record count of ``solidworks_concepts_v0`` (12 targets + 12 distractors).
EXPECTED_RECORD_COUNT = 24

#: Required record fields (every record carries all six).
REQUIRED_FIELDS: tuple[str, ...] = (
    "concept_id",
    "canonical_term",
    "concept_type",
    "definition",
    "source_class",
    "source_ref",
)
#: Optional record fields, in the frozen retrieval/render order (0151 §16/§26).
OPTIONAL_FIELDS: tuple[str, ...] = ("applies_to", "design_intent", "distinguishing_notes")

#: Closed ``concept_type`` vocabulary (0152 specification).
CONCEPT_TYPES: frozenset[str] = frozenset(
    {"sketch_relation", "feature", "pattern_feature", "sketch_pattern", "reference_geometry"}
)
#: Mirrors the frozen allow-set in ``model_mvp/rag/protocol.py`` (never widened here).
ALLOWED_SOURCE_CLASSES: frozenset[str] = frozenset(
    {"approved_experimental_material", "instructor_authored", "synthetic_reviewed", "original"}
)

#: Field names that must never appear in a retrieval record (benchmark linkage).
PROHIBITED_FIELDS: frozenset[str] = frozenset(
    {
        "benchmark_id",
        "item_id",
        "reference_answer",
        "expected_class",
        "known_failure",
        "correction_for",
    }
)
#: Benchmark/evaluation leakage markers that must never appear in record text (0151 §34).
PROHIBITED_MARKERS: tuple[str, ...] = (
    "bench_v0_",
    "m2b_v0_",
    "m1c:",
    "run-",
    "WRONG_NEIGHBOR",
    "INVENTED_NAME",
    "CALIBRATED_UNCERTAINTY",
    "reference_answer",
    "expected_class",
)

_CONCEPT_ID_RE = re.compile(r"\A[a-z0-9_]+\Z")


class RetrievalCorpusError(ValueError):
    """The retrieval corpus failed validation. ``code`` is a stable closed token.

    Closed vocabulary: ``corpus_missing``, ``corpus_hash_mismatch``,
    ``corpus_malformed``, ``duplicate_concept_id``, ``invalid_record``.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ConceptRecord:
    """One validated, immutable experimental concept record.

    ``concept_id``, ``source_class``, and ``source_ref`` are provenance-side and
    are NEVER rendered into model-visible text (0151 §26).
    """

    concept_id: str
    canonical_term: str
    concept_type: str
    definition: str
    source_class: str
    source_ref: str
    applies_to: str | None = None
    design_intent: str | None = None
    distinguishing_notes: str | None = None

    @property
    def retrieval_text(self) -> str:
        """Similarity text in the frozen field order (0151 §16), newline-separated.

        Excludes ``concept_id``, ``source_ref``, and ``source_class``; absent
        optional fields are skipped deterministically.
        """
        parts = [self.canonical_term, self.concept_type, self.definition]
        parts.extend(
            value
            for value in (self.applies_to, self.design_intent, self.distinguishing_notes)
            if value is not None
        )
        return "\n".join(parts)

    @property
    def model_visible_text(self) -> str:
        """Per-record model-visible rendering WITHOUT the ``[rank]`` line (0151 §26).

        The context builder prepends the rank line; no trailing whitespace and no
        terminal newline are produced here.
        """
        lines = [
            f"Term: {self.canonical_term}",
            f"Type: {self.concept_type}",
            f"Definition: {self.definition}",
        ]
        for label, value in (
            ("Applies to", self.applies_to),
            ("Design intent", self.design_intent),
            ("Distinguishing notes", self.distinguishing_notes),
        ):
            if value is not None:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)


def corpus_sha256(path: str | Path = CORPUS_PATH) -> str:
    """SHA256 (lowercase hex) over the EXACT raw bytes of the corpus file.

    Identity helper only — M3A1 pins nothing; the runtime profile pin is an
    M3A2 concern.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except FileNotFoundError as exc:
        raise RetrievalCorpusError("corpus_missing") from exc


def _require_text(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if type(value) is not str or not value.strip():
        raise RetrievalCorpusError("invalid_record")
    return value


def _validate_record(payload: object) -> ConceptRecord:
    if not isinstance(payload, dict):
        raise RetrievalCorpusError("invalid_record")
    keys = set(payload)
    if keys & PROHIBITED_FIELDS:
        raise RetrievalCorpusError("invalid_record")
    if not set(REQUIRED_FIELDS) <= keys:
        raise RetrievalCorpusError("invalid_record")
    if keys - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS):
        raise RetrievalCorpusError("invalid_record")

    concept_id = _require_text(payload, "concept_id")
    if not _CONCEPT_ID_RE.match(concept_id):
        raise RetrievalCorpusError("invalid_record")
    canonical_term = _require_text(payload, "canonical_term")
    concept_type = _require_text(payload, "concept_type")
    if concept_type not in CONCEPT_TYPES:
        raise RetrievalCorpusError("invalid_record")
    definition = _require_text(payload, "definition")
    source_class = _require_text(payload, "source_class")
    if source_class not in ALLOWED_SOURCE_CLASSES:
        raise RetrievalCorpusError("invalid_record")
    source_ref = _require_text(payload, "source_ref")

    optionals: dict[str, str | None] = {}
    for field in OPTIONAL_FIELDS:
        if field in payload:
            optionals[field] = _require_text(payload, field)
        else:
            optionals[field] = None

    for value in (canonical_term, concept_type, definition, source_ref, *optionals.values()):
        if value is not None and any(marker in value for marker in PROHIBITED_MARKERS):
            raise RetrievalCorpusError("invalid_record")

    return ConceptRecord(
        concept_id=concept_id,
        canonical_term=canonical_term,
        concept_type=concept_type,
        definition=definition,
        source_class=source_class,
        source_ref=source_ref,
        **optionals,
    )


def load_corpus(
    path: str | Path = CORPUS_PATH,
    *,
    expected_record_count: int | None = EXPECTED_RECORD_COUNT,
    expected_sha256: str | None = None,
) -> tuple[ConceptRecord, ...]:
    """Load and fully validate the corpus, failing closed on any violation.

    ``expected_sha256`` is optional and unused by M3A1 itself (no pin exists
    yet); when supplied it is enforced against the raw file bytes.
    """
    corpus_path = Path(path)
    try:
        raw = corpus_path.read_bytes()
    except FileNotFoundError as exc:
        raise RetrievalCorpusError("corpus_missing") from exc
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RetrievalCorpusError("corpus_hash_mismatch")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RetrievalCorpusError("corpus_malformed") from exc

    lines = text.split("\n")
    if lines and lines[-1] == "":  # exactly one terminal LF is the frozen convention
        lines.pop()
    if not lines:
        raise RetrievalCorpusError("corpus_malformed")

    records: list[ConceptRecord] = []
    seen: set[str] = set()
    for line in lines:
        if not line.strip():  # blank lines are forbidden by the serialization contract
            raise RetrievalCorpusError("corpus_malformed")
        try:
            payload = json.loads(line)
        except ValueError as exc:
            raise RetrievalCorpusError("corpus_malformed") from exc
        record = _validate_record(payload)
        if record.concept_id in seen:
            raise RetrievalCorpusError("duplicate_concept_id")
        seen.add(record.concept_id)
        records.append(record)

    ids = [record.concept_id for record in records]
    if ids != sorted(ids):  # frozen deterministic materialization order
        raise RetrievalCorpusError("invalid_record")
    if expected_record_count is not None and len(records) != expected_record_count:
        raise RetrievalCorpusError("corpus_malformed")
    return tuple(records)
