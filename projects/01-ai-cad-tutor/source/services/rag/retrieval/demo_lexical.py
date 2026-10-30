"""INTERNAL DEMO retrieval: the frozen experimental concept corpus, as Course evidence.

This module adds no retrieval algorithm and no corpus. It reuses, unchanged:

- ``model_mvp/rag/solidworks_concepts_v0.jsonl`` — the committed, content-frozen experimental
  retrieval corpus (24 records), whose field values review 0152 froze and whose file digest review
  0154 pinned. Its own label is **Experimental MVP Retrieval Corpus — Not Formally Admitted
  Production Data**, and its loader fails closed on a benchmark identifier, a leakage marker, a
  duplicate id or an unexpected field.
- ``model_mvp/rag/corpus.py`` and ``model_mvp/rag/basic_retriever.py`` — the reviewed loader and
  the frozen deterministic lexical scorer ``m3a_lexical_v1`` (0151): standard library only, no
  embeddings, no learned weights, no benchmark-keyed behaviour, with a minimum-evidence gate that
  raises rather than returning a zero-score match.

This module adds one matcher of its own, in the demo layer only: ``demo_term_name_v1``, an exact
canonical-term lookup used ONLY when the frozen scorer returns nothing. The frozen gate needs two
distinct scoring-eligible tokens from one record, so a short natural question whose only corpus word
is the concept's own name ("how do I do a shell", "What is Shell?") retrieves nothing at all. The
fallback is stricter than the scorer rather than looser — the question must literally contain a
record's canonical term under the same frozen normalization — and the result reports which matcher
produced it.

It also reports what the retrieved records can support: ``none``, ``concept_only`` (a definition but
no operating steps, which is every record in this corpus) or ``procedure``. That state is derived
from the records, never from the wording of the question.

The rest of this module maps a retrieved chunk onto the server-owned public ``Citation`` the Course
contract requires, without inventing anything the corpus does not state:

- ``citation_id``, ``chunk_id`` — the record's own ``concept_id``;
- ``document_id`` — the frozen corpus version; ``document_version`` — the corpus file digest;
- ``title`` — the record's canonical term;
- ``document_type`` — ``instructor_note``, because these are project-authored concept notes rather
  than a lecture slide, a PDF page or a transcript;
- ``locator`` — a ``section`` locator naming the concept id. **No page, slide or timestamp number
  is ever fabricated**: the corpus has none, so none is claimed;
- ``snippet`` — the retrieved record's own text, truncated to the contract's 300-character limit
  with a visible ellipsis.

Coverage is small and is disclosed, not implied: 24 SOLIDWORKS concept records across sketch
relations, features, pattern features, sketch patterns and reference geometry. Nothing else is
indexed. Evaluation questions, references, model outputs, reviewer notes, benchmark items,
holdouts, M3B3 artifacts, reserved evaluation-only sources, Bank Vault material, Previous Students
content and the 303 SFT training pairs are all outside this corpus and are not read here.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from model_mvp.rag.basic_retriever import (
    BasicLexicalRetriever,
    RetrievalNoResultError,
    tokenize,
)
from model_mvp.rag.corpus import (
    CORPUS_PATH,
    CORPUS_VERSION,
    EXPECTED_RECORD_COUNT,
    load_corpus,
)
from model_mvp.rag.protocol import RagQuery, RetrievedChunk
from services.api.app.models.course import Citation
from services.api.app.services.course import RetrievedEvidence
from services.rag.retrieval.demo_tutorials import (
    TUTORIAL_CORPUS_ID,
    TUTORIAL_MATCHER_VERSION,
    TutorialCorpusConfig,
    TutorialCorpusError,
    TutorialRecord,
    load_tutorial_corpus,
    match_tutorial_records,
)

#: The contract's snippet ceiling; longer retrieved text is cut with a visible ellipsis.
SNIPPET_LIMIT = 300
#: Every demo citation is an instructor-authored concept note, which is what these records are.
DEMO_DOCUMENT_TYPE = "instructor_note"
#: The retrieval scorer's frozen semantic version, carried into the demo record for provenance.
SCORER_VERSION = "m3a_lexical_v1"
#: The demo layer's own concept-name matcher. Named separately and reported separately so a result
#: it produced can never be mistaken for one the frozen scorer produced.
TERM_MATCHER_VERSION = "demo_term_name_v1"
#: What the retrieved records can actually support. Derived from the records, never from the wording
#: of the question, and reported so a reader is told which of the three situations they are in.
COVERAGE_NONE = "none"
COVERAGE_CONCEPT_ONLY = "concept_only"
COVERAGE_PROCEDURE = "procedure"
#: Field labels that would carry operating steps. ``ConceptRecord`` has no such field — its fields
#: are the term, its type, a definition and three optional notes — so this is a structural check
#: against the corpus schema rather than a guess about wording, and COVERAGE_PROCEDURE is
#: unreachable until the corpus itself gains a step-bearing field.
_PROCEDURE_FIELD_LABELS = ("steps:", "procedure:", "workflow:", "how to:")
#: The corpus label, verbatim from its own reviews. It travels with the demo, never softened.
CORPUS_LABEL = "Experimental MVP Retrieval Corpus — Not Formally Admitted Production Data"
#: The frozen corpus file digest, pinned by review 0154 over the committed bytes. The demo loads the
#: corpus THROUGH this pin, so an edited definition — a changed sentence that keeps the record count
#: at 24 — fails the load and the readiness probe instead of being retrieved and cited as the
#: reviewed text. Re-pinning it is a corpus decision, not a demo decision.
EXPECTED_CORPUS_SHA256 = "6dd79f9f096b2f201aa2ef3ef48e7a2d6dcb28ef967e2b066dddba6343db00c2"


@dataclass(frozen=True)
class DemoCorpusIdentity:
    """What was indexed, by name and digest, so a reader can check it."""

    corpus_version: str
    corpus_sha256: str
    records: int
    scorer_version: str
    label: str

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus_version": self.corpus_version,
            "corpus_sha256": self.corpus_sha256,
            "records": self.records,
            "scorer_version": self.scorer_version,
            "label": self.label,
            "document_type": DEMO_DOCUMENT_TYPE,
        }


def corpus_sha256(path: Path | str | None = None) -> str:
    target = Path(path) if path is not None else CORPUS_PATH
    return hashlib.sha256(target.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def demo_retriever() -> BasicLexicalRetriever:
    """The frozen corpus loaded once per process through its own fail-closed loader.

    The loader is given the pinned digest, so content drift raises ``corpus_hash_mismatch`` here
    rather than being served: what the demo retrieves is the reviewed bytes or nothing.
    """
    return BasicLexicalRetriever.from_records(
        load_corpus(CORPUS_PATH, expected_sha256=EXPECTED_CORPUS_SHA256)
    )


@lru_cache(maxsize=1)
def demo_corpus_identity() -> DemoCorpusIdentity:
    retriever = demo_retriever()
    return DemoCorpusIdentity(
        corpus_version=CORPUS_VERSION,
        corpus_sha256=corpus_sha256(),
        records=retriever.corpus_size,
        scorer_version=SCORER_VERSION,
        label=CORPUS_LABEL,
    )


#: The retrieved record's first line is its term, rendered by the frozen 0151 record block as
#: "Term: <canonical term>". The citation title carries the term itself.
_TERM_PREFIX = "Term:"


def _title(text: str) -> str:
    first = text.splitlines()[0].strip() if text.strip() else ""
    if first.startswith(_TERM_PREFIX):
        return first[len(_TERM_PREFIX) :].strip() or first
    return first


def _snippet(text: str) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= SNIPPET_LIMIT:
        return collapsed
    return collapsed[: SNIPPET_LIMIT - 1].rstrip() + "…"


def evidence_from_chunk(chunk: RetrievedChunk) -> RetrievedEvidence:
    """One retrieved chunk as server-owned Course evidence."""
    identity = demo_corpus_identity()
    citation = Citation(
        citation_id=chunk.chunk_id,
        document_id=identity.corpus_version,
        document_version=identity.corpus_sha256,
        chunk_id=chunk.chunk_id,
        title=_title(chunk.text) or chunk.chunk_id,
        document_type=DEMO_DOCUMENT_TYPE,
        # a section locator naming the concept: never an invented page or slide number
        locator={"kind": "section", "section": chunk.chunk_id},
        snippet=_snippet(chunk.text),
    )
    return RetrievedEvidence(evidence_id=chunk.chunk_id, citation=citation)


@dataclass(frozen=True)
class DemoEvidenceItem:
    """One retrieved item and everything needed to say where it came from.

    The two corpora are never merged into one opaque pool: each item names its corpus, its record,
    the source document behind it, the locator inside that document and the matcher that found it.
    """

    corpus_id: str
    record_id: str
    canonical_term: str
    source_id: str
    source_locator: object
    matcher: str
    evidence: RetrievedEvidence
    record_text: str
    #: The source record itself (a ConceptRecord or a TutorialRecord), so a server-composed reply
    #: can quote a field — a definition, the lesson's objective, one step — without parsing text.
    record: object | None = None


@dataclass(frozen=True)
class DemoRetrieval:
    """One retrieval result for the demo, with what produced it and what it can support.

    ``items`` carries the provenance; ``evidence`` and ``record_texts`` are the parallel views the
    answer assembly uses (one retrieved record each, in rank order). ``coverage`` is derived from
    the
    records themselves. ``supported_topics`` is filled only when nothing was retrieved, and holds
    the
    corpora's own canonical terms — real topics, never a suggestion invented for the occasion.
    """

    items: tuple[DemoEvidenceItem, ...]
    coverage: str
    supported_topics: tuple[str, ...] = ()
    #: Why the tutorial corpus is absent, when it is. Reported, never silently swallowed.
    tutorial_corpus_error: str | None = None
    #: The terms the authorized tutorial corpus carries steps for, filled only when nothing was
    #: retrieved, so a not-covered answer can point at a topic that really has a procedure.
    procedure_topics: tuple[str, ...] = ()

    @property
    def evidence(self) -> tuple[RetrievedEvidence, ...]:
        return tuple(item.evidence for item in self.items)

    @property
    def record_texts(self) -> tuple[str, ...]:
        return tuple(item.record_text for item in self.items)

    @property
    def matchers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.matcher for item in self.items))

    @property
    def corpora_used(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.corpus_id for item in self.items))

    @property
    def concept_terms(self) -> tuple[str, ...]:
        """The canonical term of every retrieved record, in rank order, each once."""
        return tuple(dict.fromkeys(item.canonical_term for item in self.items))

    @property
    def procedure_terms(self) -> tuple[str, ...]:
        """The canonical terms of the retrieved records that carry operating steps."""
        return tuple(
            dict.fromkeys(
                item.canonical_term for item in self.items if _text_has_procedure(item.record_text)
            )
        )

    def tutorial_item_for(self, term: str) -> DemoEvidenceItem | None:
        """The retrieved step-bearing record about ``term``, if one was retrieved."""
        wanted = str(term).casefold()
        for item in self.items:
            if item.canonical_term.casefold() == wanted and _text_has_procedure(item.record_text):
                return item
        return None

    def concept_item_for(self, term: str) -> DemoEvidenceItem | None:
        """The retrieved concept record about ``term``, if one was retrieved."""
        wanted = str(term).casefold()
        for item in self.items:
            if item.canonical_term.casefold() == wanted and not _text_has_procedure(
                item.record_text
            ):
                return item
        return None

    def supports_procedure_for(self, terms: Sequence[str]) -> bool:
        """Whether a retrieved step-bearing record is about one of ``terms``.

        A concept definition is not procedural evidence: only a record with a step-bearing field
        counts, and only when it is about a term the request is about. With no terms given (a
        question that named none), any step-bearing record counts.
        """
        procedures = set(self.procedure_terms)
        if not terms:
            return bool(procedures)
        return any(str(term) in procedures for term in terms)


def supported_topics() -> tuple[str, ...]:
    """Every canonical term the indexed corpus actually carries, in frozen record order."""
    return tuple(record.canonical_term for record in demo_retriever().records)


def _text_has_procedure(text: str) -> bool:
    """True when a retrieved record carries a step-bearing field (see _PROCEDURE_FIELD_LABELS)."""
    return any(
        line.strip().casefold().startswith(_PROCEDURE_FIELD_LABELS)
        for line in str(text).splitlines()
    )


def term_name_matches(question: str, top_k: int) -> tuple[RetrievedChunk, ...]:
    """Records whose canonical term literally appears in the question.

    A narrower test than the frozen scorer's, not a looser one.

    The frozen minimum-evidence gate (0151 section 19) requires two distinct scoring-eligible
    tokens from one record. A short natural question such as "how do I do a shell" contributes
    exactly one — the concept's own name — because the remaining words ("how", "i", "do", "a") are
    either stopwords or absent from all 24 records, so the gate drops the very record that names
    the concept asked about. The same is true of "What is Shell?".

    This matcher asks a different and stricter question: does the question literally contain a
    record's canonical term, tokenized by the frozen tokenizer so case, punctuation and Unicode
    form are handled exactly as the scorer handles them? No threshold is lowered, no synonym, stem
    or fuzzy match is invented, and nothing can match that is not a real term in this corpus. It
    runs only when the frozen scorer returned nothing, so the frozen scorer stays the authority
    whenever it returns anything at all.
    """
    query_tokens = set(tokenize(question))
    if not query_tokens:
        return ()
    matched: list[tuple[int, str, RetrievedChunk]] = []
    for record in demo_retriever().records:
        term_tokens = tokenize(record.canonical_term)
        if not term_tokens or not set(term_tokens) <= query_tokens:
            continue
        matched.append(
            (
                len(term_tokens),
                record.concept_id,
                RetrievedChunk(
                    chunk_id=record.concept_id,
                    text=record.model_visible_text,
                    # the number of term tokens matched: a demo-layer count, never the frozen score
                    score=len(term_tokens),
                    source_class=record.source_class,
                    source_ref=record.source_ref,
                    metadata={"matcher": TERM_MATCHER_VERSION},
                ),
            )
        )
    # the most specific term first (more tokens matched), then the frozen id order: deterministic
    matched.sort(key=lambda entry: (-entry[0], entry[1]))
    return tuple(chunk for _, _, chunk in matched[:top_k])


def _concept_items(question: str, top_k: int) -> tuple[DemoEvidenceItem, ...]:
    """Concept evidence: every canonical term the question names, then the frozen scorer's rest.

    Two matchers contribute and neither overrides the other:

    - ``demo_term_name_v1`` supplies the record for EVERY canonical term the question names
      literally. "How is a Fillet different from a Chamfer?" names two, and the frozen gate alone
      returns only the chamfer record (the fillet record matches one scoring-eligible token, below
      the 0151 gate), so a comparison used to arrive with half its evidence;
    - the frozen ``m3a_lexical_v1`` result is then appended in its own rank order. Nothing the
    frozen
      scorer returned is ever dropped; named terms simply come first, which is why a question that
      names one term and scores three records keeps exactly the three it had.

    The combined list is capped at ``top_k``. When a question names more terms than that, the
    deterministic order decides — more specific term first, then concept id.
    """
    named = term_name_matches(question, top_k)
    frozen_evidence, frozen_texts = retrieve_course_records(question, top_k)
    ordered: list[tuple[str, str, object]] = []
    for chunk in named:
        ordered.append((chunk.chunk_id, TERM_MATCHER_VERSION, chunk))
    for evidence, text_block in zip(frozen_evidence, frozen_texts, strict=True):
        ordered.append((evidence.evidence_id, SCORER_VERSION, (evidence, text_block)))

    identity = demo_corpus_identity()
    by_id = {record.concept_id: record for record in demo_retriever().records}
    items: list[DemoEvidenceItem] = []
    seen: set[str] = set()
    for record_id, matcher, payload in ordered:
        if record_id in seen or len(items) >= top_k:
            continue
        seen.add(record_id)
        if matcher == TERM_MATCHER_VERSION:
            chunk = payload
            evidence_record, record_text = evidence_from_chunk(chunk), chunk.text
            source_ref = chunk.source_ref
        else:
            evidence_record, record_text = payload
            source_ref = identity.corpus_version
        items.append(
            DemoEvidenceItem(
                corpus_id=identity.corpus_version,
                record_id=record_id,
                canonical_term=evidence_record.citation.title,
                source_id=source_ref,
                source_locator={"kind": "section", "section": record_id},
                matcher=matcher,
                evidence=evidence_record,
                record_text=record_text,
                record=by_id.get(record_id),
            )
        )
    return tuple(items)


def names_course_term(question: str, tutorials: TutorialCorpusConfig | None = None) -> bool:
    """Whether this message literally names a canonical term of either corpus.

    The follow-up rule asks this question, and it must be answered against the corpora's own terms
    rather than a guess about what a technical word looks like.
    """
    if term_name_matches(question, 1):
        return True
    if tutorials is None:
        return False
    try:
        records = load_tutorial_corpus(
            tutorials.corpus_path,
            tutorials.corpus_sha256,
            tutorials.decisions_dir,
            tutorials.source_root,
        )
    except TutorialCorpusError:
        return False
    return bool(match_tutorial_records(question, records, 1))


def _loaded_tutorials(tutorials: TutorialCorpusConfig | None) -> tuple[TutorialRecord, ...]:
    """The tutorial records under their permission, or none when the corpus is absent or refused."""
    if tutorials is None:
        return ()
    try:
        return load_tutorial_corpus(
            tutorials.corpus_path,
            tutorials.corpus_sha256,
            tutorials.decisions_dir,
            tutorials.source_root,
        )
    except TutorialCorpusError:
        return ()


def named_course_terms(text: str, tutorials: TutorialCorpusConfig | None = None) -> tuple[str, ...]:
    """Every canonical term of either corpus that this text literally names, deterministically.

    The same whole-term test the retrieval matchers use (frozen tokenizer, no synonym, no stem),
    so the conversation layer can ask which TOPIC an earlier user message was about without
    guessing what a technical word looks like. Concept terms first in their matcher order, then
    tutorial terms; each term once.
    """
    concept_terms = {
        record.concept_id: record.canonical_term for record in demo_retriever().records
    }
    terms: list[str] = [
        concept_terms[chunk.chunk_id] for chunk in term_name_matches(text, len(concept_terms))
    ]
    records = _loaded_tutorials(tutorials)
    terms.extend(
        record.canonical_term for record in match_tutorial_records(text, records, len(records))
    )
    return _without_subsumed_terms(text, tuple(dict.fromkeys(terms)))


#: The frozen tokenizer's normalization (NFKC, casefold, ``[0-9a-z]+`` runs) WITHOUT its
#: de-duplication and stopword removal, because the test below needs the words in order.
_ORDERED_TOKEN_RE = re.compile(r"[0-9a-z]+")


def _ordered_tokens(text: str) -> list[str]:
    return _ORDERED_TOKEN_RE.findall(unicodedata.normalize("NFKC", str(text)).casefold())


def _without_subsumed_terms(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    """Drop a term matched only through the words of a longer term the text also names.

    The whole-term test matches a term when all of its words occur anywhere in the text, so "What is
    a Linear Sketch Pattern?" also matches "Linear Pattern". That term is dropped unless its own
    words also occur together, in order, somewhere in the text ("a Linear Pattern and a Linear
    Sketch Pattern" keeps both). Retrieval is not affected; this only decides which topics a
    message named.
    """
    if len(terms) < 2:
        return terms
    words = _ordered_tokens(text)
    term_words = {term: _ordered_tokens(term) for term in terms}

    def written_together(parts: list[str]) -> bool:
        return any(words[i : i + len(parts)] == parts for i in range(len(words) - len(parts) + 1))

    return tuple(
        term
        for term in terms
        if written_together(term_words[term])
        or not any(
            set(term_words[term]) < set(term_words[other]) for other in terms if other != term
        )
    )


def topic_vocabulary(
    terms: Sequence[str], tutorials: TutorialCorpusConfig | None = None
) -> frozenset[str]:
    """The frozen-tokenizer vocabulary of the authorized records that define these terms.

    This is what lets "can you explain the thickness setting?" count as a Shell follow-up: the
    word comes from the authorized Shell records themselves, never from the assistant's prose and
    never from a synonym list. The terms' own tokens are included.
    """
    wanted = {str(term).casefold() for term in terms}
    vocabulary: set[str] = set()
    for term in terms:
        vocabulary.update(tokenize(str(term)))
    for record in demo_retriever().records:
        if record.canonical_term.casefold() in wanted:
            vocabulary.update(tokenize(record.model_visible_text))
    for tutorial in _loaded_tutorials(tutorials):
        if tutorial.canonical_term.casefold() in wanted:
            vocabulary.update(tokenize(tutorial.model_visible_text))
    return frozenset(vocabulary)


def procedure_topics(tutorials: TutorialCorpusConfig | None = None) -> tuple[str, ...]:
    """The terms the authorized tutorial corpus carries steps for, sorted; empty if unloaded."""
    return tuple(sorted({record.canonical_term for record in _loaded_tutorials(tutorials)}))


def retrieve_for_demo(
    question: str, top_k: int, tutorials: TutorialCorpusConfig | None = None
) -> DemoRetrieval:
    """The demo's single retrieval entry point, over both corpora.

    Concept definitions first, then any authorized tutorial procedure for the same named terms. The
    result says which corpus and which matcher produced each item, and what the retrieved records
    can
    support, so "nothing covers this", "this is covered as a definition" and "this is covered with a
    procedure" are three different answers instead of one refusal.
    """
    text = str(question).strip()
    items = _concept_items(text, top_k)
    tutorial_error: str | None = None
    if tutorials is not None:
        tutorial_items, tutorial_error = _tutorial_items(
            text, top_k, tutorials, also_for=tuple(item.canonical_term for item in items)
        )
        items = items + tutorial_items
    if not items:
        return DemoRetrieval(
            (),
            COVERAGE_NONE,
            supported_topics(),
            tutorial_corpus_error=tutorial_error,
            procedure_topics=procedure_topics(tutorials),
        )
    coverage = (
        COVERAGE_PROCEDURE
        if any(_text_has_procedure(item.record_text) for item in items)
        else COVERAGE_CONCEPT_ONLY
    )
    return DemoRetrieval(items, coverage, tutorial_corpus_error=tutorial_error)


def evidence_from_tutorial(record: TutorialRecord, corpus_sha256: str) -> RetrievedEvidence:
    """One tutorial record as server-owned Course evidence.

    The snippet is capped at the same 300 characters the concept corpus uses, which is also the
    display limit the permission decision grants. The locator names the tutorial record; the source
    document and its own locator travel in the evidence item's provenance, not in an invented page
    or slide number.
    """
    citation = Citation(
        citation_id=record.tutorial_id,
        document_id=TUTORIAL_CORPUS_ID,
        document_version=str(corpus_sha256).strip().lower(),
        chunk_id=record.tutorial_id,
        title=record.canonical_term,
        document_type=DEMO_DOCUMENT_TYPE,
        locator={"kind": "section", "section": record.tutorial_id},
        snippet=_snippet(record.model_visible_text),
    )
    return RetrievedEvidence(evidence_id=record.tutorial_id, citation=citation)


def _tutorial_items(
    question: str,
    top_k: int,
    tutorials: TutorialCorpusConfig,
    also_for: Sequence[str] = (),
) -> tuple[tuple[DemoEvidenceItem, ...], str | None]:
    """Tutorial items for the named terms, or nothing plus the reason the corpus is unavailable.

    ``also_for`` are the canonical terms whose concept records were retrieved: a step-bearing
    record about one of them is evidence for the same question. "How do I hollow out a solid
    part?" names no term, but the frozen scorer retrieves the Shell definition, and the Shell
    steps belong beside it rather than being withheld because the student did not say "Shell".
    """
    try:
        records = load_tutorial_corpus(
            tutorials.corpus_path,
            tutorials.corpus_sha256,
            tutorials.decisions_dir,
            tutorials.source_root,
        )
    except TutorialCorpusError as error:
        # Fail closed for this corpus only: the concept corpus still answers, and the reason is
        # reported so a reviewer sees that procedures are missing rather than guessing.
        return (), error.code
    matched = list(match_tutorial_records(question, records, top_k))
    wanted = {str(term).casefold() for term in also_for}
    seen = {record.tutorial_id for record in matched}
    for record in sorted(records, key=lambda record: record.tutorial_id):
        if record.canonical_term.casefold() in wanted and record.tutorial_id not in seen:
            matched.append(record)
            seen.add(record.tutorial_id)
    items = tuple(
        DemoEvidenceItem(
            corpus_id=TUTORIAL_CORPUS_ID,
            record_id=record.tutorial_id,
            canonical_term=record.canonical_term,
            source_id=record.source_id,
            source_locator=record.source_locator,
            matcher=TUTORIAL_MATCHER_VERSION,
            evidence=evidence_from_tutorial(record, tutorials.corpus_sha256),
            record_text=record.model_visible_text,
            record=record,
        )
        for record in matched[:top_k]
    )
    return items, None


def retrieve_course_records(
    question: str, top_k: int
) -> tuple[tuple[RetrievedEvidence, ...], tuple[str, ...]]:
    """Retrieve up to ``top_k`` records as (evidence, model-visible record text) in rank order.

    The two tuples are parallel by construction: the citations the answer will carry and the text
    the model is shown come from the same retrieval result, so they can never disagree.

    This is the frozen path and only the frozen path: an empty result here means no record passed
    the frozen minimum-evidence gate, which is not the same as "the corpus does not cover this".
    ``retrieve_for_demo`` is what the API calls, and it distinguishes the two.
    """
    text = str(question).strip()
    if not text:
        return (), ()
    try:
        chunks = demo_retriever().retrieve(RagQuery(query_text=text, top_k=top_k))
    except RetrievalNoResultError:
        return (), ()
    return tuple(evidence_from_chunk(chunk) for chunk in chunks), tuple(
        chunk.text for chunk in chunks
    )


def corpus_is_intact() -> bool:
    """True when the committed corpus loads under its pinned digest with its frozen record count.

    Both facts are checked: the digest (enforced inside the loader) and the record count. A corpus
    whose bytes changed cannot load at all, so readiness reports it rather than serving text the
    reviews never saw.
    """
    try:
        return (
            demo_retriever().corpus_size == EXPECTED_RECORD_COUNT
            and corpus_sha256() == EXPECTED_CORPUS_SHA256
        )
    except Exception:
        # a readiness probe never raises: an unloadable corpus is reported, not propagated
        return False
