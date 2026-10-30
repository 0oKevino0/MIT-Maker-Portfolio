"""Course question-answering route.

The route only wires the validated contract to the course service and the
injected backend. It must never call a provider (Ollama, vLLM) directly.

Security: the ordered ASGI request boundary authenticates this business POST
before body parsing or model work. Instructor mode remains a local response-
depth gate disabled by default (ADR 0001 section 6).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from services.api.app.dependencies.inference import get_text_backend
from services.api.app.errors import InstructorModeDisabledError, InternalContractError
from services.api.app.models.course import CourseAnswer, CourseQuestion
from services.api.app.services.course import answer_course_question
from services.api.app.services.demo_provenance import record_demo_provenance
from services.inference.backends.base import ModelBackend
from services.inference.capability import require_course_capabilities

router = APIRouter()


# exclude_none keeps optional contract fields (e.g. citation snippet) absent
# instead of null, matching the JSON Schemas in packages/contracts.
@router.post(
    "/api/v1/course/ask",
    response_model=CourseAnswer,
    response_model_exclude_none=True,
)
async def ask_course_question(
    question: CourseQuestion,
    request: Request,
    backend: Annotated[ModelBackend, Depends(get_text_backend)],
) -> CourseAnswer:
    settings = request.app.state.settings
    if question.answer_mode == "instructor" and not settings.instructor_mode_enabled:
        raise InstructorModeDisabledError()
    if settings.internal_demo_enabled:
        # The INTERNAL DEMO profile: retrieve from the frozen experimental corpus, then let the
        # demo assembly attach the citations for exactly those records. Every other profile keeps
        # the default path below, unchanged — including its imports, which is why the demo's are
        # taken here: they reach the experimental retrieval tree, which is not part of the
        # installed distribution and must never be a load-time requirement of the default API.
        from services.api.app.services.assistant_interaction import (
            assistant_interaction_answer,
            detect_assistant_interaction,
        )

        # A greeting or a control message is not a question about course material. It is answered
        # here, before retrieval and before the capability gate: no evidence is needed, no model is
        # called, and the demo stays usable for "hi" even when the adapter sidecar is down. Anything
        # this does not recognise by whole-message equality falls through to the course path.
        interaction = detect_assistant_interaction(question.question)
        if interaction is not None:
            # The student sees only the reply; these facts go to the internal provenance log.
            record_demo_provenance(
                settings.demo_provenance_log,
                {
                    "route": "assistant_interaction",
                    "interaction_type": interaction.intent,
                    "retrieval_used": interaction.retrieval_used,
                    "model_called": interaction.model_called,
                    "original_user_message": question.question,
                    "retrieval_query": None,
                },
            )
            return assistant_interaction_answer(question, interaction, question.guided_progress)

        from model_mvp.rag.corpus import RetrievalCorpusError
        from services.api.app.services.assistant_interaction import (
            clarification_interaction,
            held_step_interaction,
            progress_interaction,
            step_choice_interaction,
            step_range_interaction,
        )
        from services.api.app.services.course_demo import (
            answer_course_question_demo,
            demo_procedure_unavailable_answer,
        )
        from services.api.app.services.demo_conversation import (
            CLARIFY_AMBIGUOUS_REQUEST,
            RESOLUTION_EXPLICIT,
            build_conversation_context,
            is_acknowledgement,
            is_next_request,
            is_restart_request,
            progress_step,
        )
        from services.api.app.services.teaching_modes import (
            HINT_MODES,
            MODE_GUIDED_STEPS,
            guided_step_answer,
            hint_answer,
            partial_coverage_answer,
            question_for_mode,
            step_explanation_question,
            validate_progress,
        )
        from services.rag.retrieval.demo_lexical import (
            named_course_terms,
            procedure_topics,
            retrieve_for_demo,
            supported_topics,
            topic_vocabulary,
        )
        from services.rag.retrieval.demo_tutorials import TutorialCorpusConfig

        # The fail-closed capability gate runs before any evidence handling (review D1A-04), and
        # retrieval is evidence handling: a no-model deployment answers 503 without reading the
        # corpus or scoring the question.
        require_course_capabilities(backend)
        tutorials = (
            TutorialCorpusConfig(
                corpus_path=settings.demo_tutorial_corpus_path,
                corpus_sha256=settings.demo_tutorial_corpus_sha256,
                decisions_dir=settings.demo_permission_decisions_dir,
                source_root=settings.demo_tutorial_source_root,
            )
            if settings.demo_tutorial_corpus_path
            else None
        )
        # A follow-up such as "how do I actually do it?" names no concept, so retrieval over named
        # concepts can only miss it. The conversation layer resolves it to the most recent topic
        # the student named themselves — never from the assistant's prose, and never shown to the
        # student as though they had typed it. Both questions it asks (which terms does this text
        # name, and which words do the authorized records for those terms use) are answered here
        # against the corpora, so the conversation layer never guesses what a course term is.
        steps_topics = procedure_topics(tutorials)
        # The explicit Guided Steps position the browser carries is untrusted until it is checked
        # against the authorized record it names: same topic, same record, same step count, a
        # step within the lesson. Anything else is dropped, and the student is asked where to
        # continue. The conversation's text is never read for a position.
        progress = None
        progress_rejected = question.guided_progress is not None
        if question.guided_progress is not None and any(
            question.guided_progress.topic.casefold() == topic.casefold() for topic in steps_topics
        ):
            try:
                check = retrieve_for_demo(
                    question.guided_progress.topic, settings.demo_retrieval_top_k, tutorials
                )
            except RetrievalCorpusError:
                raise InternalContractError() from None
            progress = validate_progress(
                question.guided_progress, check.tutorial_item_for(question.guided_progress.topic)
            )
            progress_rejected = progress is None
        context = build_conversation_context(
            question.question,
            question.history,
            terms_in=lambda text: named_course_terms(text, tutorials),
            vocabulary=lambda terms: topic_vocabulary(terms, tutorials),
            terms_with_steps=frozenset(steps_topics),
            fallback_topic=progress.topic if progress is not None else None,
        )
        if context.needs_clarification:
            # It refers back to no topic this conversation has named, or to one of several, or it
            # could ask what a thing is or how to operate it: ask rather than guess. Answered
            # locally — no retrieval, no model call.
            clarification = clarification_interaction(
                candidates=context.topic_terms,
                procedure_topics=steps_topics,
                concept_topics=supported_topics(),
                ambiguous=context.clarification_reason == CLARIFY_AMBIGUOUS_REQUEST,
            )
            record_demo_provenance(
                settings.demo_provenance_log,
                {
                    "route": "assistant_interaction",
                    "interaction_type": clarification.intent,
                    "retrieval_used": False,
                    "model_called": False,
                    "guided_progress_rejected": progress_rejected,
                    **context.provenance,
                },
            )
            return assistant_interaction_answer(question, clarification, progress)
        try:
            retrieval = retrieve_for_demo(
                context.retrieval_query, settings.demo_retrieval_top_k, tutorials
            )
        except RetrievalCorpusError:
            # The committed corpus is server-owned data. A corpus that will not load is a server
            # defect with its own classified code — never an answer that implies retrieval ran and
            # simply found nothing.
            raise InternalContractError() from None
        # The procedure-evidence gate, before any generation and whatever the answer mode: a
        # request for operations is answered from a step-bearing record about EACH topic it asks
        # about, or not by a model at all. A concept definition is not procedural evidence — a
        # model given only a definition and asked for steps invents them (observed live: "click
        # the checkmark" for a Fillet) — and a lesson about one topic never authorizes steps for
        # another. The reply then states what the sources cover, cites them, and leaves the
        # conversation open.
        mode = str(question.answer_mode)
        requested = tuple(context.topic_terms)
        with_steps = (
            tuple(t for t in requested if t in retrieval.procedure_terms)
            if requested
            else tuple(retrieval.procedure_terms)
        )
        without_steps = tuple(t for t in requested if t not in retrieval.procedure_terms)
        wants_procedure = context.asks_for_procedure and bool(retrieval.items)
        procedure_blocked = wants_procedure and not with_steps
        partial_coverage = wants_procedure and bool(with_steps) and bool(without_steps)
        mode_path = "model"
        guided_step: int | None = None

        def interaction(reply, kind: str) -> CourseAnswer:
            record_demo_provenance(
                settings.demo_provenance_log,
                {
                    "route": "assistant_interaction",
                    "interaction_type": reply.intent,
                    "retrieval_used": True,
                    "model_called": False,
                    "teaching_mode": mode,
                    "mode_path": kind,
                    "guided_progress_rejected": progress_rejected,
                    **context.provenance,
                },
            )
            return assistant_interaction_answer(question, reply, progress)

        if procedure_blocked:
            mode_path = "blocked"
            definitions = (
                [
                    (item.canonical_term, getattr(item.record, "definition", ""))
                    for term in (requested or retrieval.concept_terms)
                    for item in [retrieval.concept_item_for(term)]
                    if item is not None
                ]
                if mode in HINT_MODES
                else []
            )
            answer = demo_procedure_unavailable_answer(
                question,
                evidence=retrieval.evidence,
                terms=tuple(t for t in retrieval.concept_terms if t in requested)
                or retrieval.concept_terms,
                procedure_topics=steps_topics,
                concept_definitions=definitions,
                guided_progress=progress,
            )
        elif partial_coverage:
            mode_path = "partial"
            answer = partial_coverage_answer(
                question, retrieval.items, with_steps, without_steps, progress
            )
        elif context.step_explanation and retrieval.items:
            # "What does step 2 mean?": explain one authorized step, and advance nothing. The step
            # is the one named, or the lesson's current one from the explicit position; without
            # either, the tutor asks which step. A topic with no steps has none to explain.
            mode_path = "step_explanation"
            lesson_terms = tuple(t for t in requested if t in retrieval.procedure_terms) or tuple(
                retrieval.procedure_terms
            )
            tutorial = retrieval.tutorial_item_for(lesson_terms[0]) if lesson_terms else None
            if tutorial is None:
                answer = demo_procedure_unavailable_answer(
                    question,
                    evidence=retrieval.evidence,
                    terms=tuple(t for t in retrieval.concept_terms if t in requested)
                    or retrieval.concept_terms,
                    procedure_topics=steps_topics,
                    concept_definitions=[
                        (item.canonical_term, getattr(item.record, "definition", ""))
                        for term in (requested or retrieval.concept_terms)
                        for item in [retrieval.concept_item_for(term)]
                        if item is not None
                    ],
                    guided_progress=progress,
                )
            else:
                steps = tuple(getattr(tutorial.record, "procedure_steps", ()))
                term = tutorial.canonical_term
                state = validate_progress(progress, tutorial)
                number = context.step_number
                if number is None and context.current_step and state is not None:
                    number = state.step
                if number is None or number < 1:
                    return interaction(step_choice_interaction(term, len(steps)), mode_path)
                if number > len(steps):
                    return interaction(step_range_interaction(term, len(steps), number), mode_path)
                answer = await answer_course_question_demo(
                    question,
                    backend,
                    evidence=retrieval.evidence,
                    record_texts=retrieval.record_texts,
                    coverage=retrieval.coverage,
                    supported_topics=retrieval.supported_topics,
                    history=context.turns,
                    procedure_topics=retrieval.procedure_topics,
                    question_text=step_explanation_question(
                        question.question, term, number, steps[number - 1]
                    ),
                    guided_progress=progress,
                )
                mode_path = "step_explanation_model"
        elif wants_procedure and mode == MODE_GUIDED_STEPS:
            # One quoted step at a time, from the explicit position only. A fresh request that
            # names the topic starts at step 1; "next" continues after the step the position
            # records; a report ("2") continues after the step reported; a direct request ("show
            # me step 2") shows that step. A step the lesson does not have changes nothing. With
            # no valid position and no report, the tutor asks rather than guessing.
            mode_path = "guided"
            term = with_steps[0]
            tutorial = retrieval.tutorial_item_for(term)
            total = len(getattr(tutorial.record, "procedure_steps", ()))
            state = validate_progress(progress, tutorial)
            reported = progress_step(question.question)
            asked = context.step_number if reported is None else None
            fresh = context.resolution == RESOLUTION_EXPLICIT and not is_next_request(
                question.question
            )
            if reported is not None or asked is not None:
                number = reported if reported is not None else asked
                if number < 1 or number > total:
                    return interaction(step_range_interaction(term, total, number), mode_path)
                guided_step = number + 1 if reported is not None else number
            elif is_restart_request(question.question) or (fresh and asked is None):
                guided_step = 1
            elif state is None:
                if is_next_request(question.question):
                    return interaction(progress_interaction(term, total), mode_path)
                guided_step = 1
            else:
                guided_step = total + 1 if state.ended else state.step + 1
            answer = guided_step_answer(question, tutorial, guided_step)
        elif wants_procedure and mode in HINT_MODES:
            mode_path = "hint"
            term = with_steps[0]
            answer = hint_answer(
                question,
                mode,
                retrieval.tutorial_item_for(term),
                retrieval.concept_item_for(term),
                progress,
            )
        elif (
            mode == MODE_GUIDED_STEPS
            and progress is not None
            and is_acknowledgement(question.question)
            and any(t.casefold() == progress.topic.casefold() for t in requested)
        ):
            # "ok" after an explanation inside the lesson asks for nothing. The reply says where
            # the lesson stands; the model is not asked, since it would answer anyway and was
            # seen to pose as the next step.
            return interaction(
                held_step_interaction(
                    progress.topic, progress.step, progress.total, progress.ended
                ),
                "held",
            )
        else:
            answer = await answer_course_question_demo(
                question,
                backend,
                evidence=retrieval.evidence,
                record_texts=retrieval.record_texts,
                coverage=retrieval.coverage,
                supported_topics=retrieval.supported_topics,
                history=context.turns,
                procedure_topics=retrieval.procedure_topics,
                question_text=question_for_mode(question.question, mode),
                guided_progress=progress,
            )
        record_demo_provenance(
            settings.demo_provenance_log,
            {
                "route": "course",
                "interaction_type": None,
                "retrieval_used": True,
                "model_called": bool(retrieval.items)
                and mode_path in ("model", "step_explanation_model"),
                "procedure_blocked": procedure_blocked,
                "partial_coverage": partial_coverage,
                "teaching_mode": mode,
                "mode_path": mode_path,
                "guided_step": guided_step,
                "guided_progress_rejected": progress_rejected,
                "guided_progress_out": (
                    answer.guided_progress.model_dump() if answer.guided_progress else None
                ),
                **context.provenance,
                "coverage": retrieval.coverage,
                "corpora_used": list(retrieval.corpora_used),
                "matchers": list(retrieval.matchers),
                "record_ids": [item.record_id for item in retrieval.items],
                "source_ids": sorted({str(item.source_id) for item in retrieval.items}),
                "tutorial_corpus_error": retrieval.tutorial_corpus_error,
                "grounding_status": answer.grounding_status,
            },
        )
        return answer
    return await answer_course_question(question, backend)
