"""Pydantic models mirroring course_question and course_answer contracts.

The JSON Schemas in ``packages/contracts`` are the single source of truth;
these models must stay field-compatible with them, and every assembled
public response is re-validated against the schema files at runtime
(``services.api.app.contracts``).

Strictness policy (review issue I-08):

- integer fields are strict (booleans and numeric strings are rejected);
- float fields reject booleans and non-finite values;
- UUIDs must arrive in canonical lowercase form, matching the schema regex;
- cross-field rules that JSON Schema cannot express (timestamp ordering,
  citation-ID uniqueness, grounding/citation coupling, marker references)
  are enforced by model validators.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from services.api.app.course_citations import (
    CitationMarkerError,
    require_canonical_citation_id,
    validate_citation_relationship,
)

AnswerMode = Literal["hint_1", "hint_2", "guided_steps", "full_explanation", "instructor"]
#: The hard total bound on conversation history, across all turns of one request.
HISTORY_TOTAL_CHARACTER_LIMIT = 6000
#: The only roles a browser may send. A system or developer turn is rejected by the contract.
ConversationRole = Literal["user", "assistant"]


class ConversationTurn(BaseModel):
    """One bounded turn of the active conversation, as the browser kept it."""

    model_config = ConfigDict(extra="forbid", strict=True)

    role: ConversationRole
    content: str = Field(min_length=1, max_length=2000)


GroundingStatus = Literal[
    "grounded",
    "partially_grounded",
    "insufficient_evidence",
    # A built-in assistant reply to a greeting or a control message: no retrieval ran, no model was
    # called, and no course evidence is claimed. Carries no citations, like insufficient_evidence.
    "assistant_interaction",
]
#: The statuses that assert no course evidence, and therefore must carry no citations.
_STATUSES_WITHOUT_CITATIONS = frozenset({"insufficient_evidence", "assistant_interaction"})
DocumentType = Literal["lecture_slide", "lecture_pdf", "transcript", "homework", "instructor_note"]

_CANONICAL_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _require_canonical_uuid(value: object) -> object:
    """Match the contract regex exactly: lowercase, hyphenated, no URN form."""
    if isinstance(value, UUID):
        return value
    if not isinstance(value, str) or not _CANONICAL_UUID.fullmatch(value):
        raise ValueError("must be a canonical lowercase UUID string")
    return value


def _require_real_number(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("booleans are not valid numbers")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("value must be finite")
    return value


class _ContractModel(BaseModel):
    """Boundary models reject unknown fields instead of silently dropping them."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
    )


class PageLocator(_ContractModel):
    kind: Literal["page"]
    page_number: int = Field(ge=1, strict=True)


class SlideLocator(_ContractModel):
    kind: Literal["slide"]
    slide_number: int = Field(ge=1, strict=True)
    lecture_number: int | None = Field(default=None, ge=1, strict=True)


class TimestampLocator(_ContractModel):
    kind: Literal["timestamp"]
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(ge=0, allow_inf_nan=False)

    _numbers = field_validator("start_seconds", "end_seconds", mode="before")(_require_real_number)

    @model_validator(mode="after")
    def end_not_before_start(self) -> TimestampLocator:
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must not be earlier than start_seconds")
        return self


class HomeworkLocator(_ContractModel):
    kind: Literal["homework"]
    item: str = Field(min_length=1)


class SectionLocator(_ContractModel):
    kind: Literal["section"]
    section: str = Field(min_length=1)


Locator = Annotated[
    PageLocator | SlideLocator | TimestampLocator | HomeworkLocator | SectionLocator,
    Field(discriminator="kind"),
]


class Citation(_ContractModel):
    citation_id: str = Field(min_length=1, max_length=64)
    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    document_type: DocumentType
    locator: Locator
    snippet: str | None = Field(default=None, max_length=300)

    @model_validator(mode="before")
    @classmethod
    def explicit_null_snippet_is_forbidden(cls, value: object) -> object:
        if isinstance(value, Mapping) and "snippet" in value and value["snippet"] is None:
            raise ValueError("snippet may be omitted but must not be null")
        return value

    @field_validator("citation_id", mode="before")
    @classmethod
    def citation_id_is_canonical(cls, value: object) -> str:
        try:
            return require_canonical_citation_id(value)
        except CitationMarkerError:
            raise ValueError("citation_id is not canonical") from None

    @model_serializer(mode="wrap")
    def omit_absent_snippet(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        data = handler(self)
        if self.snippet is None:
            data.pop("snippet", None)
        return data


class GuidedProgress(_ContractModel):
    """The explicit Guided Steps position: which lesson, which record, how far, whether it ended.

    The browser keeps it between requests and sends it back; the API treats what arrives as
    untrusted and keeps it only when it matches the authorized record it names and its step
    count. The position is the student's own indication of progress, never a verified CAD state.
    """

    topic: str = Field(min_length=1, max_length=100)
    record_id: str = Field(min_length=1, max_length=200)
    step: int = Field(ge=0, le=50, strict=True)
    total: int = Field(ge=1, le=50, strict=True)
    ended: bool

    @model_validator(mode="after")
    def step_is_within_the_lesson(self) -> GuidedProgress:
        if self.step > self.total:
            raise ValueError("guided step is beyond the lesson's step count")
        return self


class CourseQuestion(_ContractModel):
    schema_version: Literal["1.0.0"]
    request_id: UUID = Field(strict=False)
    course_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)
    answer_mode: AnswerMode
    locale: str | None = Field(default=None, min_length=2, max_length=35)
    #: Bounded conversation context from the browser. Hard bounds, fail-closed: at most 6 messages,
    #: each at most 2000 characters, at most 6000 characters in total, and only user/assistant
    #: roles — the browser may not inject a system or developer turn.
    history: list[ConversationTurn] | None = Field(default=None, max_length=6)
    #: The explicit Guided Steps position the browser carries (see GuidedProgress); untrusted.
    guided_progress: GuidedProgress | None = None

    @model_validator(mode="after")
    def history_is_within_hard_bounds(self) -> CourseQuestion:
        if self.history:
            total = sum(len(turn.content) for turn in self.history)
            if total > HISTORY_TOTAL_CHARACTER_LIMIT:
                raise ValueError("conversation history exceeds the total character bound")
        return self

    _uuid_policy = field_validator("request_id", mode="before")(_require_canonical_uuid)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class CourseAnswer(_ContractModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    request_id: UUID = Field(strict=False)
    answer_id: UUID = Field(strict=False)
    answer_mode: AnswerMode
    grounding_status: GroundingStatus
    answer_text: str = Field(min_length=1, max_length=12000)
    citations: list[Citation]
    warnings: list[str]
    #: The Guided Steps position after this answer, for the browser to send back; absent when no
    #: valid lesson position exists.
    guided_progress: GuidedProgress | None = None

    _uuid_policy = field_validator("request_id", "answer_id", mode="before")(
        _require_canonical_uuid
    )

    @model_validator(mode="after")
    def grounding_and_citations_are_consistent(self) -> CourseAnswer:
        """Cross-field rules from ADR 0001 section 7."""
        if self.grounding_status in _STATUSES_WITHOUT_CITATIONS:
            if self.citations:
                raise ValueError(f"{self.grounding_status} answers must not carry citations")
        else:
            if not self.citations:
                raise ValueError(f"{self.grounding_status} answers require at least one citation")
        citation_ids = [citation.citation_id for citation in self.citations]
        try:
            validate_citation_relationship(
                self.answer_text,
                citation_ids,
                citation_ids,
                require_selected_order=True,
            )
        except CitationMarkerError:
            raise ValueError(
                "every citation must be referenced from answer_text; marker and citation "
                "IDs must be unique and consistent"
            ) from None
        return self
