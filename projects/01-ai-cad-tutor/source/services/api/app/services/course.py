"""Course answer assembly across the model/evidence trust boundary.

The model owns only the typed internal draft's answer text and selected
evidence IDs.  The server validates the selection against the current
request's evidence, supplies every public Citation field, and re-validates
the finished public contract before returning it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from services.api.app.contracts import require_finite_numbers, validate_public_contract
from services.api.app.course_citations import (
    CitationMarkerError,
    require_canonical_citation_id,
    validate_citation_relationship,
)
from services.api.app.errors import InternalContractError
from services.api.app.models.course import (
    Citation,
    CourseAnswer,
    CourseQuestion,
    GroundingStatus,
    GuidedProgress,
)
from services.inference.backends.base import ChatMessage, ModelBackend
from services.inference.capability import require_course_capabilities
from services.inference.errors import InvalidModelOutputError
from services.inference.validation import generate_validated

INSUFFICIENT_EVIDENCE_TEXT = (
    "Approved course evidence is insufficient to answer this question. "
    "Please ask a teaching assistant to review it."
)
_INSUFFICIENT_EVIDENCE_WARNING = (
    "No approved course evidence supported a grounded answer; teaching-assistant "
    "review is recommended."
)
_SYSTEM_PROMPT = (
    "You are an offline SOLIDWORKS/CSWA tutor. Use only evidence selected from "
    "the server-provided allowed evidence IDs. Respond with exactly a JSON object "
    'containing {"answer_text": "...", "citation_ids": ["..."]}. Every selected '
    "ID must appear in answer_text as a case-sensitive [citation_id] marker, and "
    "every citation marker must be selected. If no approved evidence supports an "
    "answer, return an empty citation_ids list."
)
_CONTRACT_PROBE_UUID = "00000000-0000-0000-0000-000000000000"


@dataclass(frozen=True)
class RetrievedEvidence:
    """One server-owned Citation behind an allowed model-selectable ID."""

    evidence_id: str
    citation: Citation


def _citation_contract_probe(citation: Citation) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "request_id": _CONTRACT_PROBE_UUID,
        "answer_id": _CONTRACT_PROBE_UUID,
        "answer_mode": "hint_1",
        "grounding_status": "grounded",
        "answer_text": f"Approved evidence [{citation.citation_id}].",
        "citations": [citation.model_dump(mode="json", exclude_none=True)],
        "warnings": [],
    }


def _validated_evidence_map(
    evidence: Sequence[RetrievedEvidence],
) -> dict[str, RetrievedEvidence]:
    """Validate all server evidence before any model call and preserve metadata."""
    try:
        if isinstance(evidence, (str, bytes)):
            raise TypeError("evidence is not a sequence of records")

        allowed: dict[str, RetrievedEvidence] = {}
        for record in evidence:
            if not isinstance(record, RetrievedEvidence):
                raise TypeError("evidence contains an invalid record")
            evidence_id = require_canonical_citation_id(record.evidence_id)
            if evidence_id in allowed:
                raise ValueError("evidence ids must be unique")
            if not isinstance(record.citation, Citation):
                raise TypeError("evidence citation is not a Citation")
            if record.citation.citation_id != evidence_id:
                raise ValueError("evidence id and citation id are inconsistent")
            if "snippet" in record.citation.model_fields_set and record.citation.snippet is None:
                raise ValueError("an explicit null snippet is invalid")

            citation_payload = record.citation.model_dump(
                mode="python",
                exclude_none=True,
            )
            require_finite_numbers(citation_payload)
            validated_citation = Citation.model_validate(citation_payload)
            probe = _citation_contract_probe(validated_citation)
            require_finite_numbers(probe)
            validate_public_contract("course_answer", probe)
            allowed[evidence_id] = RetrievedEvidence(
                evidence_id=evidence_id,
                citation=validated_citation,
            )
        return allowed
    except Exception:
        # This boundary receives server-owned objects only.  A malformed or
        # inconsistent record is a server defect, never model output.
        raise InternalContractError() from None


def _finalize_answer(
    question: CourseQuestion,
    *,
    answer_text: str,
    citations: list[Citation],
    warnings: list[str],
    grounding_status: GroundingStatus | None = None,
    guided_progress: GuidedProgress | None = None,
) -> CourseAnswer:
    """Assemble and perform every final public-response validation layer.

    ``grounding_status`` is derived from the citations unless a caller states one, which only the
    assistant-interaction path does: a greeting is neither grounded nor short of evidence. Every
    validation layer below still runs, and the model still rejects a status that disagrees with the
    citations it was given.
    """
    try:
        answer = CourseAnswer(
            request_id=question.request_id,
            answer_id=uuid4(),
            answer_mode=question.answer_mode,
            grounding_status=grounding_status
            or ("grounded" if citations else "insufficient_evidence"),
            answer_text=answer_text,
            citations=citations,
            warnings=warnings,
            guided_progress=guided_progress,
        )
        require_finite_numbers(answer)
        payload = answer.model_dump(mode="json", exclude_none=True)
        require_finite_numbers(payload)
        validate_public_contract("course_answer", payload)

        # Re-enter the public Pydantic boundary from its serialized JSON form;
        # this prevents a model_construct/mutation or serialization-direction
        # mismatch from escaping as a successful response.
        answer = CourseAnswer.model_validate(payload)
        selected_ids = [citation.citation_id for citation in answer.citations]
        validate_citation_relationship(
            answer.answer_text,
            selected_ids,
            selected_ids,
            require_selected_order=True,
        )
        return answer
    except Exception:
        raise InternalContractError() from None


def _insufficient_evidence_answer(question: CourseQuestion) -> CourseAnswer:
    return _finalize_answer(
        question,
        answer_text=INSUFFICIENT_EVIDENCE_TEXT,
        citations=[],
        warnings=[_INSUFFICIENT_EVIDENCE_WARNING],
    )


# The INTERNAL DEMO profile assembles its answer from two of these steps, so they are reachable by
# name. The bodies are unchanged and the private names stay for existing callers. The demo composes
# its own not-covered answer rather than reusing _insufficient_evidence_answer, whose text speaks of
# approved course evidence — true of the default profile, false of an experimental corpus.
validated_evidence_map = _validated_evidence_map
finalize_course_answer = _finalize_answer


async def answer_course_question(
    question: CourseQuestion,
    backend: ModelBackend,
    evidence: Sequence[RetrievedEvidence] = (),
) -> CourseAnswer:
    """Answer from the current approved evidence or return a safe refusal."""
    # Fail-closed capability gate (review D1A-04): the Course path runs only
    # against a backend that DECLARES the required capabilities. A no-model
    # or capability-mismatched deployment answers 503 before any evidence
    # handling — it never emits an insufficient-evidence answer that would
    # disguise "model not ready" as a completed retrieval.
    require_course_capabilities(backend)

    allowed_evidence = _validated_evidence_map(evidence)
    if not allowed_evidence:
        return _insufficient_evidence_answer(question)

    allowed_ids = list(allowed_evidence)
    prompt = f"{_SYSTEM_PROMPT} Allowed evidence IDs: {json.dumps(allowed_ids)}"
    draft = await generate_validated(
        backend,
        task="course_answer_draft",
        messages=[
            ChatMessage(role="system", text=prompt),
            ChatMessage(role="user", text=question.question),
        ],
    )

    try:
        marker_order = validate_citation_relationship(
            draft.answer_text,
            draft.citation_ids,
            allowed_evidence.keys(),
        )
    except CitationMarkerError:
        raise InvalidModelOutputError(
            "course answer citation selection failed validation"
        ) from None

    if not draft.citation_ids:
        # A well-formed zero-selection draft is not grounded.  Its model text
        # is intentionally discarded rather than published as course fact.
        return _insufficient_evidence_answer(question)

    citations = [allowed_evidence[citation_id].citation for citation_id in marker_order]
    return _finalize_answer(
        question,
        answer_text=draft.answer_text,
        citations=citations,
        warnings=[],
    )
