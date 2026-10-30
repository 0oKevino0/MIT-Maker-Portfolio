"""Assistant-interaction layer for the INTERNAL DEMO profile: greetings before retrieval.

A learning assistant that answers "HI" with "the knowledge base does not cover that" is behaving
consistently and still behaving wrongly: a greeting is not a question about course material, and
asking the retriever to find evidence for it can only fail. This module recognises a small, closed
set of **interaction** messages and answers them locally.

What it is not:

- it is not general chat. Five intents are supported — greeting, thanks, help, capabilities,
  goodbye — and everything else goes to the course path untouched;
- it never retrieves, never calls the model, and never calls anything remote;
- it never claims course evidence. Its answers carry the ``assistant_interaction`` contract status,
  which the shared schema binds to carrying no citations, so an interaction reply cannot be mistaken
  for a grounded course answer.

**The matching law.** Substring matching is forbidden here: "hi" occurs inside "thickness", "hey"
inside "they", "ty" inside "geometry". A message is classified only when its NORMALIZED WHOLE TEXT
equals one of the phrases listed below. Normalization is Unicode NFKC, casefold, surrounding
whitespace removed, internal whitespace collapsed, and trailing punctuation runs removed — so "HI!",
" hi ", "Hi?!" and "hi" are one phrase, while "What is Shell?" matches nothing and goes to
retrieval. Ambiguity therefore resolves toward the course path by construction: anything this module
does not recognise exactly, it declines to handle.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final

from services.api.app.models.course import CourseAnswer, CourseQuestion, GuidedProgress
from services.api.app.services.course import finalize_course_answer

#: Trailing characters stripped before matching, so "hi!!" and "hi" are the same phrase. Only
#: TRAILING runs are removed; internal punctuation is part of the phrase. ASCII is enough because
#: NFKC normalization already folds the fullwidth forms onto these.
_TERMINAL_PUNCTUATION: Final = ".!?,;:~"

GREETING: Final = "GREETING"
THANKS: Final = "THANKS"
HELP: Final = "HELP"
CAPABILITIES: Final = "CAPABILITIES"
GOODBYE: Final = "GOODBYE"
#: Not a phrase intent: produced by the conversation layer when a follow-up's topic cannot be
#: resolved unambiguously. It is answered locally for the same reasons a greeting is.
CLARIFICATION: Final = "CLARIFICATION"

#: Exact normalized phrases per intent. Deliberately small and closed: adding a phrase is a product
#: decision, and nothing here is a prefix test, a stem or a regular expression.
_PHRASES: Final[dict[str, frozenset[str]]] = {
    GREETING: frozenset(
        {
            "hi",
            "hii",
            "hiii",
            "hi there",
            "hiya",
            "hello",
            "hello there",
            "helo",
            "hey",
            "hey there",
            "heya",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
            "good day",
            "morning",
        }
    ),
    THANKS: frozenset(
        {
            "thanks",
            "thank you",
            "thank u",
            "thanks a lot",
            "thanks so much",
            "thank you very much",
            "thank you so much",
            "many thanks",
            "thx",
            "ty",
            "cheers",
            "appreciate it",
        }
    ),
    HELP: frozenset(
        {
            "help",
            "help me",
            "/help",
            "i need help",
            "can you help",
            "can you help me",
            "how does this work",
            "how do i use this",
        }
    ),
    CAPABILITIES: frozenset(
        {
            "what can you do",
            "what can you help with",
            "what can you help me with",
            "what do you do",
            "what are you",
            "who are you",
            "what are your capabilities",
            "capabilities",
            "what can i ask",
            "what can i ask you",
            "what topics do you cover",
        }
    ),
    GOODBYE: frozenset(
        {
            "bye",
            "bye bye",
            "goodbye",
            "good bye",
            "see you",
            "see ya",
            "see you later",
            "cya",
            "later",
            "good night",
            "goodnight",
        }
    ),
}

#: The deterministic reply per intent. Short, tutor-oriented, and never a claim about course
#: evidence: each one says what this assistant does, not what a source says.
_REPLIES: Final[dict[str, str]] = {
    GREETING: (
        "Hi! I'm your CSWA AI Tutor. I can help with SOLIDWORKS concepts and CSWA questions. "
        "What are you working on?"
    ),
    THANKS: "You're welcome! What would you like to work on next?",
    HELP: (
        "I can explain SOLIDWORKS concepts, modeling features and sketch relations covered by this "
        'demo\'s course sources. Ask a question such as "What is Shell?" or "How is a fillet '
        "different from a chamfer?\" and I'll answer from the sources I can show you."
    ),
    CAPABILITIES: (
        "I answer SOLIDWORKS and CSWA questions from the course sources available in this demo, "
        "and I show you the source behind each answer. I'll also tell you plainly when the current "
        "material does not contain enough information, rather than guessing."
    ),
    GOODBYE: "See you! Good luck with your CSWA practice.",
}

#: An interaction answer carries NO student-facing warning. "No model was called" is an
#: implementation detail, and a greeting is not a claim about course material that needs qualifying.
#: The same facts are recorded internally instead — see services/api/app/services/demo_provenance.py
#: and the ``assistant_interaction`` contract status, which the schema binds to carrying no
#: citations. The experimental notice on technical answers is untouched.
INTERACTION_WARNINGS: Final[list[str]] = []


def normalize_message(message: str) -> str:
    """NFKC, casefold, collapse whitespace, drop trailing punctuation. Never a partial match."""
    folded = unicodedata.normalize("NFKC", str(message)).casefold()
    collapsed = " ".join(folded.split())
    return collapsed.rstrip(_TERMINAL_PUNCTUATION + " ")


@dataclass(frozen=True)
class AssistantInteraction:
    """One recognised interaction, and the provenance that matters about it."""

    intent: str
    answer_text: str
    #: Stated explicitly so a caller, a log line or a test can assert it rather than infer it.
    retrieval_used: bool = False
    model_called: bool = False


def detect_assistant_interaction(message: str) -> AssistantInteraction | None:
    """Return the interaction this message IS, or None to send it down the course path.

    Whole-message equality only. A technical question, a greeting with a question attached, or
    anything else this module does not recognise exactly returns None.
    """
    normalized = normalize_message(message)
    if not normalized:
        return None
    for intent, phrases in _PHRASES.items():
        if normalized in phrases:
            return AssistantInteraction(intent=intent, answer_text=_REPLIES[intent])
    return None


def supported_intents() -> tuple[str, ...]:
    """The closed intent set, in the order they are matched."""
    return tuple(_PHRASES)


def _quoted_choice(terms: tuple[str, ...]) -> str:
    """'A or B', 'A, B or C': the candidates exactly as the corpora name them."""
    if len(terms) == 1:
        return terms[0]
    return f"{', '.join(terms[:-1])} or {terms[-1]}"


#: The intent of a guided-steps progress question: the tutor cannot see the lesson's earlier
#: steps in the bounded conversation, so it asks which one was completed rather than guessing.
PROGRESS: Final = "PROGRESS"


def clarification_interaction(
    candidates: tuple[str, ...] = (),
    procedure_topics: tuple[str, ...] = (),
    concept_topics: tuple[str, ...] = (),
    *,
    ambiguous: bool = False,
) -> AssistantInteraction:
    """Ask which topic a follow-up means, or what an ambiguous request wants. No model call.

    ``candidates`` are the terms an ambiguous follow-up could refer to (the student's own earlier
    question named all of them). With none — the conversation has not named a topic yet — the reply
    offers two example questions built from terms the corpora really carry: a how-to for a topic
    that has authorized steps, and a definition for a concept. With ``ambiguous`` the request
    itself could ask what a term is or how to operate it ("show me a fillet"), and the reply asks
    which, in words the conversation layer resolves ("what it is" / "the steps").
    """
    if ambiguous and candidates:
        single = len(candidates) == 1
        steps = (
            "the steps to use it"
            if single and candidates[0] in procedure_topics
            else "how to create it"
        )
        subject = candidates[0] if single else "it"
        which = "" if single else f", and which of {_quoted_choice(candidates)} you mean"
        text = (
            f"Do you want to know what {subject} is, or {steps}? Say \u201cwhat it is\u201d or "
            f"\u201cthe steps\u201d{which}."
        )
    elif candidates:
        text = (
            f"Do you mean {_quoted_choice(candidates)}? "
            "Tell me which one and I'll pick up from there."
        )
    else:
        examples: list[str] = []
        if procedure_topics:
            examples.append(f"“How do I use {procedure_topics[0]}?”")
        concept = next((term for term in concept_topics if term not in procedure_topics), None)
        if concept is not None:
            examples.append(f"“What is a {concept}?”")
        text = "Which SOLIDWORKS feature or topic do you mean?"
        if examples:
            text += f" For example, you can ask {' or '.join(examples)}"
    return AssistantInteraction(intent=CLARIFICATION, answer_text=text)


def progress_interaction(term: str, total: int) -> AssistantInteraction:
    """Ask which step of a lesson the student has completed. No retrieval, no model call."""
    numbers = ", ".join(str(n) for n in range(1, total)) + f" or {total}" if total > 1 else "1"
    text = (
        f"Which step of the {term} lesson have you completed? Say {numbers} \u2014 or "
        "\u201cstart\u201d to begin at step 1."
    )
    return AssistantInteraction(intent=PROGRESS, answer_text=text)


def step_choice_interaction(term: str, total: int) -> AssistantInteraction:
    """Ask which step of a lesson an explanation request means. No retrieval, no model call."""
    numbers = ", ".join(str(n) for n in range(1, total)) + f" or {total}" if total > 1 else "1"
    return AssistantInteraction(
        intent=PROGRESS,
        answer_text=f"Which step of the {term} lesson do you mean? Say {numbers}.",
    )


def step_range_interaction(term: str, total: int, requested: int) -> AssistantInteraction:
    """A step the lesson does not have ("step 999"): say how many there are, change nothing."""
    numbers = ", ".join(str(n) for n in range(1, total)) + f" or {total}" if total > 1 else "1"
    return AssistantInteraction(
        intent=PROGRESS,
        answer_text=(
            f"The {term} lesson has {total} step{'s' if total != 1 else ''}; there is no step "
            f"{requested}. Say {numbers} \u2014 or \u201cstart\u201d to begin at step 1."
        ),
    )


def held_step_interaction(term: str, step: int, total: int, ended: bool) -> AssistantInteraction:
    """Answer a bare "ok" inside a lesson: where it stands and how to go on. Nothing moves."""
    if ended:
        text = (
            f"All {total} steps of the {term} lesson have been shown. Say \u201cstart\u201d to "
            "go through them again, or ask about another feature."
        )
    elif step < 1:
        text = f"The {term} lesson has not started. Say \u201cstart\u201d to begin at step 1."
    else:
        text = (
            f"Step {step} of {total} in the {term} lesson is still the current step. Say "
            "\u201cnext\u201d when you have done it, or ask what it means."
        )
    return AssistantInteraction(intent=PROGRESS, answer_text=text)


def assistant_interaction_answer(
    question: CourseQuestion,
    interaction: AssistantInteraction,
    guided_progress: GuidedProgress | None = None,
) -> CourseAnswer:
    """The contract answer for one interaction, through the same final validation layers.

    ``assistant_interaction`` is stated rather than derived: with no citations the derivation would
    say insufficient_evidence, which would describe a missing source that was never needed. The
    shared schema binds this status to carrying no citations, so nothing is loosened by using it.
    """
    return finalize_course_answer(
        question,
        answer_text=interaction.answer_text,
        citations=[],
        warnings=list(INTERACTION_WARNINGS),
        grounding_status="assistant_interaction",
        guided_progress=guided_progress,
    )
