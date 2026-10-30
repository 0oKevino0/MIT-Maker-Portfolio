"""Shared strict validation for model-generated drafts (review issue I-05).

Every backend — Mock, Ollama, and the future vLLM — returns its parsed JSON
object through :func:`generate_validated`, which applies the registered
internal draft schema. Routes and services must never validate model output
themselves, and public contracts are never used as model output targets
(ADR 0001 section 13.3): the model produces a small internal draft, and the
domain service assembles and re-validates the public contract.

Validation failure messages contain schema paths and rule names only — never
draft content, course passages, or prompts.
"""

from __future__ import annotations

from functools import cache
from typing import Annotated, Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from services.inference.backends.base import ChatMessage, ModelBackend
from services.inference.errors import InvalidModelOutputError

# One shared citation-ID body drives both validation dialects below.
_CITATION_ID_BODY = r"(?:c[0-9]+|[A-Za-z0-9][A-Za-z0-9._:-]*[._:-][A-Za-z0-9][A-Za-z0-9._:-]*)"
# Pydantic-core (Rust regex): ``$`` is already an absolute end-of-string
# anchor, so a trailing LF/CR can never sneak past it.
_CITATION_ID_PATTERN = rf"^{_CITATION_ID_BODY}$"
# JSON Schema patterns are evaluated with Python ``re.search``, where ``$``
# also matches just before a final newline.  ``(?![\s\S])`` is the portable
# (ECMA-262 + Python) absolute end-of-string anchor, so the Schema rejects
# exactly what Pydantic rejects (review 0026 IF-02).  ``\Z`` is deliberately
# not used because it is Python-specific.
_CITATION_ID_SCHEMA_PATTERN = rf"^{_CITATION_ID_BODY}(?![\s\S])"
CitationId = Annotated[
    str,
    StringConstraints(max_length=64, pattern=_CITATION_ID_PATTERN),
]


class CourseAnswerDraft(BaseModel):
    """Strict internal Course output containing only model-owned fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    answer_text: Annotated[str, Field(min_length=1, max_length=12000)]
    citation_ids: Annotated[list[CitationId], Field(max_length=20)]

    @field_validator("citation_ids")
    @classmethod
    def citation_ids_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("citation IDs must be unique")
        return values


class CourseAnswerDemoDraft(BaseModel):
    """The INTERNAL DEMO profile's draft: prose only, no model-selected citations.

    The demo's adapter sidecar returns the tutor model's plain answer text; the server, not the
    model, selects the citations from the evidence it actually retrieved and appends their markers.
    That is a narrower trust boundary than :class:`CourseAnswerDraft`, not a wider one: a demo
    answer can never cite a source the retriever did not return, and the model is never asked to
    emit JSON.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    answer_text: Annotated[str, Field(min_length=1, max_length=12000)]
    #: Why generation stopped, when the runtime reports it. Server-observed, not model-written: it
    #: exists so an answer cut off at the token cap can be shown as incomplete instead of being
    #: published as a finished one. Absent when the runtime does not say.
    finish_reason: Literal["eos", "length"] | None = None


# Internal draft schemas. The model may only produce these shapes; server-owned
# fields (IDs, hashes, citation metadata, statuses) are never model-writable.
DRAFT_SCHEMAS: dict[str, dict[str, Any]] = {
    "course_answer_draft": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["answer_text", "citation_ids"],
        "properties": {
            "answer_text": {"type": "string", "minLength": 1, "maxLength": 12000},
            "citation_ids": {
                "description": (
                    "IDs from the allowed-evidence whitelist provided by the current "
                    "retrieval result. The server maps them to full citations and "
                    "rejects any ID outside the whitelist."
                ),
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 64,
                    "pattern": _CITATION_ID_SCHEMA_PATTERN,
                },
                "maxItems": 20,
                "uniqueItems": True,
            },
        },
    },
    "course_answer_demo_draft": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["answer_text"],
        "properties": {
            "answer_text": {"type": "string", "minLength": 1, "maxLength": 12000},
            "finish_reason": {
                "description": (
                    "Observed by the backend, not written by the model: 'length' means the "
                    "generation stopped at the token cap and the answer may be incomplete."
                ),
                "type": "string",
                "enum": ["eos", "length"],
            },
        },
    },
}

_DRAFT_MODELS: dict[str, type[BaseModel]] = {
    "course_answer_draft": CourseAnswerDraft,
    "course_answer_demo_draft": CourseAnswerDemoDraft,
}


@cache
def _draft_validator(task: str) -> Draft202012Validator:
    schema = DRAFT_SCHEMAS[task]
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_draft(task: str, draft: Any) -> CourseAnswerDraft | CourseAnswerDemoDraft:
    """Strictly validate a model draft against the registered internal schema.

    Never repairs, never fills defaults, never accepts silently. Raises
    ``InvalidModelOutputError`` with a content-free message on any failure.
    """
    if task not in DRAFT_SCHEMAS:
        raise KeyError(f"unknown internal draft task '{task}'")
    if not isinstance(draft, dict):
        raise InvalidModelOutputError(f"{task}: model output is not a JSON object")
    try:
        _draft_validator(task).validate(draft)
    except JsonSchemaValidationError as exc:
        location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        raise InvalidModelOutputError(
            f"{task}: draft failed schema validation at '{location}' (rule: {exc.validator})"
        ) from None
    try:
        return _DRAFT_MODELS[task].model_validate(draft, strict=True)
    except ValidationError:
        raise InvalidModelOutputError(f"{task}: draft failed typed validation") from None


async def generate_validated(
    backend: ModelBackend,
    task: str,
    messages: list[ChatMessage],
) -> CourseAnswerDraft | CourseAnswerDemoDraft:
    """One shared trust boundary for every text-generation backend."""
    schema = DRAFT_SCHEMAS[task]
    draft = await backend.generate_text(messages, response_schema=schema)
    return validate_draft(task, draft)
