"""INTERNAL DEMO tutorial corpus: procedure evidence, loaded only against its own permission.

The frozen concept corpus (``solidworks_concepts_v0``) states definitions and carries no procedures,
so a question like "how do I use Shell?" could only ever be answered at concept level. This module
loads a SECOND, separate corpus of reviewed procedure records built from one specifically authorized
source document, and it refuses to load unless four things check out:

1. **the corpus digest** equals the digest pinned in configuration;
2. **the permission decision** named by each record exists, digests to the value the record records,
   grants the ``INTERNAL_DEMO_RAG`` scope, and binds the record's source;
3. **the source document's own SHA256** still matches, hashed from the file at runtime — if the
   document changed, nothing loads;
4. **the record schema**: every required field present, every step a non-empty string.

Any failure raises :class:`TutorialCorpusError` and the demo answers from concepts alone. Neither
this module nor the corpus it loads is committed to the repository: the corpus lives in the external
demo workspace, because it contains course text.

The two corpora are never merged. Each retrieved item reports which corpus and which record it came
from, which source document and locator stand behind it, and which matcher found it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from model_mvp.rag.basic_retriever import tokenize

#: This corpus's own identity, reported alongside every item it supplies.
TUTORIAL_CORPUS_ID = "solidworks_demo_tutorials_v0"
#: The record schema version this loader accepts. A different value fails closed.
TUTORIAL_RECORD_SCHEMA = "cswa_demo_tutorial_v1"
#: The scope the permission decision must grant. Training or evaluation approval is not this.
REQUIRED_SCOPE = "INTERNAL_DEMO_RAG"
#: The matcher that finds tutorial records: the record's canonical term, present literally.
TUTORIAL_MATCHER_VERSION = "demo_tutorial_term_v1"
#: The label this corpus carries into the UI and the health payload.
TUTORIAL_CORPUS_LABEL = (
    "Internal Demo Tutorial Corpus — one owner-authorized source, INTERNAL_DEMO_RAG scope only"
)
_REQUIRED_FIELDS = (
    "schema_version",
    "tutorial_id",
    "canonical_term",
    "topic",
    "procedure_steps",
    "source_id",
    "source_sha256",
    "source_relative_path",
    "source_locator",
    "permission_decision_id",
    "permission_decision_sha256",
)


@dataclass(frozen=True)
class TutorialCorpusConfig:
    """Where the tutorial corpus, its permission decisions and its source documents live.

    All four are required together: a corpus without its pinned digest, its decision record or the
    source document to re-hash cannot be verified, and an unverifiable corpus is not loaded.
    """

    corpus_path: str
    corpus_sha256: str
    decisions_dir: str
    source_root: str


class TutorialCorpusError(RuntimeError):
    """The tutorial corpus could not be loaded under its permission. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class TutorialRecord:
    """One reviewed procedure, with the provenance that authorizes showing it."""

    tutorial_id: str
    canonical_term: str
    topic: str
    lesson_title: str
    objective: str
    step_group: str
    procedure_steps: tuple[str, ...]
    source_id: str
    source_sha256: str
    source_relative_path: str
    source_locator: dict[str, Any]
    permission_decision_id: str

    @property
    def model_visible_text(self) -> str:
        """The record as the model sees it: verbatim source lines in a fixed order.

        A ``Procedure:`` label is emitted so the coverage check can see that this record carries
        operating steps. Nothing is paraphrased and nothing is added — no thickness, no menu path,
        no dialog location, no face, no default. The steps are the authorized document's own
        sentences; only the numbering is this renderer's, as layout.

        Longer than the 300-character display limit, deliberately: that limit applies to what is
        SHOWN to a reviewer (the citation snippet enforces it), while the record sent to the model
        is the reviewed excerpt itself.
        """
        lines = [
            f"Term: {self.canonical_term}",
            f"Type: {self.topic}",
            f"Lesson: {self.lesson_title}",
            f"Objective: {self.objective}",
            f"Procedure: {self.step_group}",
        ]
        lines.extend(f"{index}. {step}" for index, step in enumerate(self.procedure_steps, 1))
        return "\n".join(lines)


def _as_text(value: Any, code: str, *, limit: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise TutorialCorpusError(code)
    return value


def _verify_permission(record: dict[str, Any], decisions_dir: Path) -> None:
    """The decision must exist, digest as recorded, grant the scope and bind this source."""
    decision_path = decisions_dir / f"{record['permission_decision_id']}.json"
    if not decision_path.is_file():
        raise TutorialCorpusError("permission_decision_missing")
    measured = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    if measured != record["permission_decision_sha256"]:
        raise TutorialCorpusError("permission_decision_digest_mismatch")
    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise TutorialCorpusError("permission_decision_malformed") from error
    if decision.get("scope") != REQUIRED_SCOPE:
        raise TutorialCorpusError("permission_scope_not_internal_demo_rag")
    if (decision.get("granted") or {}).get(REQUIRED_SCOPE) is not True:
        raise TutorialCorpusError("permission_scope_not_granted")
    bound = decision.get("bound_source") or {}
    if bound.get("source_sha256") != record["source_sha256"]:
        raise TutorialCorpusError("permission_source_sha_not_bound")
    if bound.get("source_id") != record["source_id"]:
        raise TutorialCorpusError("permission_source_id_not_bound")
    if record["source_relative_path"] not in (bound.get("source_relative_paths") or []):
        raise TutorialCorpusError("permission_source_path_not_bound")
    # the grant's own display limit governs; a record may not ask for more than it was given
    granted_limit = (decision.get("granted") or {}).get("excerpt_display_max_characters")
    if not isinstance(granted_limit, int) or granted_limit < 1:
        raise TutorialCorpusError("permission_excerpt_limit_missing")
    requested = record.get("excerpt_display_max_characters", granted_limit)
    if not isinstance(requested, int) or requested > granted_limit:
        raise TutorialCorpusError("permission_excerpt_limit_exceeded")


def _verify_source_sha(record: dict[str, Any], source_root: Path) -> None:
    """Hash the authorized document itself. A changed source loads nothing."""
    source_path = source_root / record["source_relative_path"]
    if not source_path.is_file():
        raise TutorialCorpusError("source_document_missing")
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != record["source_sha256"]:
        raise TutorialCorpusError("source_document_sha_mismatch")


def _record_from(payload: dict[str, Any]) -> TutorialRecord:
    for field in _REQUIRED_FIELDS:
        if field not in payload:
            raise TutorialCorpusError("record_field_missing")
    if payload["schema_version"] != TUTORIAL_RECORD_SCHEMA:
        raise TutorialCorpusError("record_schema_version_unsupported")
    steps = payload["procedure_steps"]
    if not isinstance(steps, list) or not steps or len(steps) > 20:
        raise TutorialCorpusError("record_steps_invalid")
    if not isinstance(payload["source_locator"], dict) or not payload["source_locator"]:
        raise TutorialCorpusError("record_locator_invalid")
    sha = _as_text(payload["source_sha256"], "record_source_sha_invalid", limit=64)
    if len(sha) != 64:
        raise TutorialCorpusError("record_source_sha_invalid")
    return TutorialRecord(
        tutorial_id=_as_text(payload["tutorial_id"], "record_id_invalid", limit=120),
        canonical_term=_as_text(payload["canonical_term"], "record_term_invalid", limit=120),
        topic=_as_text(payload["topic"], "record_topic_invalid", limit=120),
        lesson_title=_as_text(payload.get("lesson_title", ""), "record_lesson_invalid"),
        objective=_as_text(payload.get("objective", ""), "record_objective_invalid"),
        step_group=_as_text(payload.get("step_group", ""), "record_step_group_invalid"),
        procedure_steps=tuple(_as_text(step, "record_steps_invalid") for step in steps),
        source_id=_as_text(payload["source_id"], "record_source_id_invalid", limit=200),
        source_sha256=sha,
        source_relative_path=_as_text(payload["source_relative_path"], "record_path_invalid"),
        source_locator=payload["source_locator"],
        permission_decision_id=_as_text(
            payload["permission_decision_id"], "record_decision_invalid", limit=200
        ),
    )


@lru_cache(maxsize=4)
def load_tutorial_corpus(
    corpus_path: str, expected_sha256: str, decisions_dir: str, source_root: str
) -> tuple[TutorialRecord, ...]:
    """Load the tutorial corpus, or fail closed. Cached per configuration tuple."""
    path = Path(corpus_path)
    if not path.is_file():
        raise TutorialCorpusError("corpus_missing")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != str(expected_sha256).strip().lower():
        raise TutorialCorpusError("corpus_digest_mismatch")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TutorialCorpusError("corpus_malformed") from error
    records: list[TutorialRecord] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError as error:
            raise TutorialCorpusError("corpus_malformed") from error
        if not isinstance(payload, dict):
            raise TutorialCorpusError("corpus_malformed")
        _verify_permission(payload, Path(decisions_dir))
        _verify_source_sha(payload, Path(source_root))
        record = _record_from(payload)
        if record.tutorial_id in seen:
            raise TutorialCorpusError("record_id_duplicate")
        seen.add(record.tutorial_id)
        records.append(record)
    if not records:
        raise TutorialCorpusError("corpus_empty")
    return tuple(records)


def tutorial_corpus_identity(
    corpus_path: str, expected_sha256: str, decisions_dir: str, source_root: str
) -> dict[str, Any]:
    """What was loaded, by name, digest and permission — for the health payload and the UI."""
    records = load_tutorial_corpus(corpus_path, expected_sha256, decisions_dir, source_root)
    return {
        "corpus_id": TUTORIAL_CORPUS_ID,
        "corpus_sha256": str(expected_sha256).strip().lower(),
        "records": len(records),
        "matcher_version": TUTORIAL_MATCHER_VERSION,
        "label": TUTORIAL_CORPUS_LABEL,
        "permission_decision_ids": sorted({r.permission_decision_id for r in records}),
        "source_ids": sorted({r.source_id for r in records}),
        "terms": sorted({r.canonical_term for r in records}),
    }


def match_tutorial_records(
    question: str, records: tuple[TutorialRecord, ...], top_k: int
) -> tuple[TutorialRecord, ...]:
    """Records whose canonical term literally appears in the question.

    The same test the concept layer uses: the frozen tokenizer, whole-term containment, no synonym,
    no stem, no fuzzy match. Deterministic order: more specific term first, then tutorial id.
    """
    query_tokens = set(tokenize(question))
    if not query_tokens:
        return ()
    matched: list[tuple[int, str, TutorialRecord]] = []
    for record in records:
        term_tokens = tokenize(record.canonical_term)
        if term_tokens and set(term_tokens) <= query_tokens:
            matched.append((len(term_tokens), record.tutorial_id, record))
    matched.sort(key=lambda entry: (-entry[0], entry[1]))
    return tuple(record for _, _, record in matched[:top_k])
