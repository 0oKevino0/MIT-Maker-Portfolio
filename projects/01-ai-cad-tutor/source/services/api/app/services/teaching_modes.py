"""Teaching modes for the INTERNAL DEMO tutor: what each answer mode reveals, bounded by evidence.

The answer-mode control is part of the contract (``hint_1``, ``hint_2``, ``guided_steps``,
``full_explanation``) and, in this profile, it changes what the student gets for a request about
a topic whose authorized lesson carries steps:

- **Hint 1** gives one direction drawn from the lesson's goal and the concept record, and
  withholds the lesson's steps.
- **Hint 2** names the feature, quotes the lesson's first step and says how many follow.
- **Guided Steps** quotes one step of the lesson at a time. The next follows when the student asks
  ("next"), and the position is the explicit ``guided_progress`` the browser carries between
  requests — the topic, the record it is bound to, the last step shown, the step count and
  whether the lesson ended — validated against the authorized record on every request. Nothing is
  read from the conversation's text. Without a valid position the tutor asks which step was
  completed rather than guessing, and progress is the student's own indication, never a verified
  CAD state.
- **Full Explanation** sends every retrieved record to the model and asks for the complete
  explanation they support.

Every mode is bounded by the evidence: the procedure-evidence gate runs before any of this, so a
request for operations on a topic with no step-bearing record never reaches a mode that could
produce steps, whatever mode is selected. Hint, guided and partial-coverage replies are composed by
the server from record fields — the lesson's objective, step group and steps, the concept's
definition — inside fixed tutoring prose; no model runs for them and nothing is paraphrased.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from services.api.app.errors import InternalContractError
from services.api.app.models.course import CourseAnswer, CourseQuestion, GuidedProgress
from services.api.app.services.course import RetrievedEvidence, finalize_course_answer
from services.api.app.services.course_demo import (
    DEMO_PROCEDURE_SCOPE_WARNING,
    SOURCES_CHECKED_PREFIX,
    validated_evidence_map,
)

MODE_HINT_1 = "hint_1"
MODE_HINT_2 = "hint_2"
MODE_GUIDED_STEPS = "guided_steps"
MODE_FULL_EXPLANATION = "full_explanation"
HINT_MODES = (MODE_HINT_1, MODE_HINT_2)

#: Appended to the student's question in the model-facing turn, under the retrieved records. The
#: frozen Stage-C wrapper is unchanged; the question text it wraps carries the mode. Guided Steps
#: has no instruction: when it reaches the model at all, the request was conceptual.
MODE_INSTRUCTIONS: dict[str, str] = {
    MODE_HINT_1: (
        "Reply with one short hint only, drawn from the records above. Do not give the full "
        "explanation, and give no steps."
    ),
    MODE_HINT_2: (
        "Reply with a short, more specific hint drawn from the records above: name the relevant "
        "feature or concept and one thing to consider next. Do not give the full explanation."
    ),
    MODE_FULL_EXPLANATION: (
        "Give the complete explanation the records above support, in their order, and nothing "
        "they do not state."
    ),
}

#: Stable markers inside the demo-prefixed warnings; the page tells these replies apart by them.
LESSON_HINT_MARKER = "lesson hint"
GUIDED_STEP_MARKER = "guided step"
LAST_STEP_MARKER = "the last step"
GUIDED_COMPLETE_MARKER = "guided steps complete"
PARTIAL_STEPS_MARKER = "steps for some topics only"


class EvidenceItem(Protocol):
    """The shape of one retrieved item this module reads: its term, its citation, its record.

    The retrieval module that builds these lives in the experimental tree, which the default API
    must never import at load time, so this module names the shape rather than the class.
    """

    canonical_term: str
    evidence: RetrievedEvidence
    record: object | None


def question_for_mode(question_text: str, mode: str) -> str:
    """The model-facing question for this mode: the student's text, then the mode's instruction."""
    instruction = MODE_INSTRUCTIONS.get(str(mode))
    return f"{question_text}\n\n{instruction}" if instruction else question_text


def step_explanation_question(question_text: str, term: str, step_number: int, step: str) -> str:
    """The model-facing question for explaining one authorized step, and nothing beyond it."""
    return (
        f"{question_text}\n\nExplain what step {step_number} of the {term} lesson \u2014 "
        f"\u201c{step}\u201d \u2014 means, using only the records above. Do not add operations, "
        "values or interface details the records do not state, and do not list the other steps."
    )


def validate_progress(
    progress: GuidedProgress | None, tutorial_item: EvidenceItem | None
) -> GuidedProgress | None:
    """The browser's lesson position, kept only when it is about THIS record and within its steps.

    What arrives is untrusted: a position for another topic, another record, a different step
    count or a step beyond the lesson is discarded, and the caller then asks the student where
    to continue instead of guessing. Nothing is ever read from the conversation's text.
    """
    if progress is None or tutorial_item is None:
        return None
    _, _, steps = _tutorial_fields(tutorial_item)
    if progress.topic.casefold() != str(tutorial_item.canonical_term).casefold():
        return None
    if progress.record_id != tutorial_item.evidence.evidence_id:
        return None
    if progress.total != len(steps) or progress.step > len(steps):
        return None
    return progress


def progress_at(tutorial_item: EvidenceItem, step: int) -> GuidedProgress:
    """The explicit position after showing ``step`` of this record's lesson."""
    _, _, steps = _tutorial_fields(tutorial_item)
    shown = min(max(int(step), 0), len(steps))
    return GuidedProgress(
        topic=str(tutorial_item.canonical_term),
        record_id=str(tutorial_item.evidence.evidence_id),
        step=shown,
        total=len(steps),
        ended=shown >= len(steps),
    )


def _tutorial_fields(item: EvidenceItem) -> tuple[str, str, tuple[str, ...]]:
    record = item.record
    steps = tuple(str(step) for step in getattr(record, "procedure_steps", ()) or ())
    if not steps:
        raise InternalContractError()
    return str(getattr(record, "objective", "")), str(getattr(record, "step_group", "")), steps


def _definition(item: EvidenceItem | None) -> str:
    return "" if item is None else str(getattr(item.record, "definition", "") or "")


def _names(terms: Sequence[str]) -> str:
    names = [str(term) for term in terms]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _composed_answer(
    question: CourseQuestion,
    items: Sequence[EvidenceItem | None],
    body: str,
    warnings: Sequence[str],
    guided_progress: GuidedProgress | None = None,
) -> CourseAnswer:
    """A reply the server composed itself: the records it drew on are cited and marked checked."""
    allowed = validated_evidence_map([item.evidence for item in items if item is not None])
    ids = list(allowed)
    if not ids:
        raise InternalContractError()
    markers = " ".join(f"[{citation_id}]" for citation_id in ids)
    return finalize_course_answer(
        question,
        answer_text=f"{body}\n\n{SOURCES_CHECKED_PREFIX} {markers}",
        citations=[allowed[citation_id].citation for citation_id in ids],
        warnings=list(warnings),
        guided_progress=guided_progress,
    )


def hint_warning(mode: str) -> str:
    label = "Hint 1" if mode == MODE_HINT_1 else "Hint 2"
    return (
        f"Internal experimental prototype: lesson hint ({label}) — this reply is a direction drawn "
        "from the authorized lesson and the concept record; the lesson's steps are withheld in "
        "this mode and shown in Guided Steps and Full Explanation."
    )


def guided_warning(step_number: int, total: int) -> str:
    tail = LAST_STEP_MARKER if step_number == total else "the next follows when you ask"
    return (
        f"Internal experimental prototype: guided step {step_number} of {total} — one step of the "
        f"authorized lesson, quoted; {tail}. Progress is what you report: the tutor cannot see "
        "your SOLIDWORKS model and does not verify that a step was done."
    )


GUIDED_COMPLETE_WARNING = (
    "Internal experimental prototype: guided steps complete — every step of the authorized lesson "
    "has been shown; nothing beyond them is in the demo's sources."
)


def partial_warning(with_steps: Sequence[str], without_steps: Sequence[str]) -> str:
    return (
        f"Internal experimental prototype: steps for some topics only — the authorized lesson "
        f"covers {_names(with_steps)}; the records for {_names(without_steps)} define the concept "
        "and contain no steps, so no procedure was produced for them."
    )


def hint_answer(
    question: CourseQuestion,
    mode: str,
    tutorial_item: EvidenceItem,
    concept_item: EvidenceItem | None = None,
    guided_progress: GuidedProgress | None = None,
) -> CourseAnswer:
    """Hint 1 or Hint 2 for a topic whose lesson carries steps. No model runs.

    Hint 1 quotes the lesson's objective and the concept definition and withholds every step.
    Hint 2 quotes the first step and says how many follow, under the lesson's own step heading.
    """
    term = str(tutorial_item.canonical_term)
    objective, step_group, steps = _tutorial_fields(tutorial_item)
    definition = _definition(concept_item)
    feature = f"{term}: {definition}" if definition else f"{term}."
    items = [concept_item, tutorial_item] if concept_item is not None else [tutorial_item]
    if mode == MODE_HINT_1:
        body = (
            f"Hint 1 — {term}. The lesson's goal: {objective} The feature that does this is "
            f"{feature} Find that feature in your model first. Hint 2 names the first thing to do, "
            "and Guided Steps walks you through the lesson one step at a time."
        )
        return _composed_answer(question, items, body, [hint_warning(mode)], guided_progress)
    remaining = len(steps) - 1
    continues = (
        f"It continues with {remaining} more step{'s' if remaining != 1 else ''} under "
        f"“{step_group}”. "
        if remaining
        else "That is its only step. "
    )
    body = (
        f"Hint 2 — {term}. The feature you need is {feature} The lesson begins with: "
        f"“{steps[0]}” {continues}Switch to Guided Steps to take them one at a time, or "
        "Full Explanation to see them all."
    )
    return _composed_answer(
        question, items, body, [hint_warning(mode), DEMO_PROCEDURE_SCOPE_WARNING], guided_progress
    )


def guided_step_answer(
    question: CourseQuestion, tutorial_item: EvidenceItem, step_number: int
) -> CourseAnswer:
    """One quoted step of the lesson, or the end of the lesson once every step has been shown.

    The reply carries the explicit position it leaves the lesson in, for the browser to send back.
    """
    term = str(tutorial_item.canonical_term)
    _, _, steps = _tutorial_fields(tutorial_item)
    total = len(steps)
    if step_number > total:
        body = (
            f"All {total} steps of the {term} lesson have been shown. Say \u201cstart\u201d to go "
            "through them again, or switch to Full Explanation to see them together."
        )
        return _composed_answer(
            question,
            [tutorial_item],
            body,
            [GUIDED_COMPLETE_WARNING],
            progress_at(tutorial_item, total),
        )
    step_number = max(1, step_number)
    tail = (
        "That is the last step in the lesson."
        if step_number == total
        else "When you have done this, say \u201cnext\u201d for the next step."
    )
    body = f"Step {step_number} of {total} \u2014 {term}: {steps[step_number - 1]}\n\n{tail}"
    return _composed_answer(
        question,
        [tutorial_item],
        body,
        [guided_warning(step_number, total), DEMO_PROCEDURE_SCOPE_WARNING],
        progress_at(tutorial_item, step_number),
    )


def partial_coverage_answer(
    question: CourseQuestion,
    items: Sequence[EvidenceItem],
    with_steps: Sequence[str],
    without_steps: Sequence[str],
    guided_progress: GuidedProgress | None = None,
) -> CourseAnswer:
    """A request for operations on several topics when only some have steps. No model runs.

    A single step-bearing source never authorizes steps for another topic, so the reply says which
    topics the lesson covers and which it does not, and offers to continue with the covered one.
    """
    if not with_steps or not without_steps:
        raise InternalContractError()
    first = str(with_steps[0])
    body = (
        f"The available lesson contains steps for {_names(with_steps)}, but not verified creation "
        f"steps for {_names(without_steps)}. I can continue with {first} — say "
        f"“continue with {first}” — or ask what “{without_steps[0]}” means."
    )
    return _composed_answer(
        question, items, body, [partial_warning(with_steps, without_steps)], guided_progress
    )
