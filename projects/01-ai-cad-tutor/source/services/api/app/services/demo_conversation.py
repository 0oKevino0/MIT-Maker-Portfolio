"""Bounded conversation handling for the INTERNAL DEMO tutor: follow-ups, and what to log.

Two small, deterministic jobs:

**The retrieval query.** "How do I actually do it?" names no concept, so retrieval over a corpus of
named concepts can only miss. The conversation layer therefore resolves a follow-up to the ACTIVE
TOPIC: the most recent USER message in the bounded history that names a canonical course term,
provided no user message since then changed the subject. The retrieval query becomes that user
message followed by the current one. No LLM rewriter, no synonym table and no paraphrase: two pieces
of the user's own text, concatenated. Assistant prose is never used — the model's own words must not
steer what gets retrieved next.

A message counts as a follow-up only when it refers back AND introduces no new subject:

- it refers back through an anaphor ("it", "that", "those"), a continuation cue ("what's the next
  step?", "explain the difference more simply", or "Why?" / "How?" as the whole message), a word
  taken from the AUTHORIZED records that define the active topic ("can you explain the thickness
  setting?" after Shell — "thickness" is in the Shell records themselves), or, when the topic named
  several terms, a comparison word ("which should I use?", "both");
- it introduces a new subject when any of its content words is neither conversational nor found in
  those records. "What is the capital of France?" right after a Fillet/Chamfer question contains
  "capital" and "France", so it is a new question and never borrows that question's evidence —
  the failure this guard exists for was observed live. Such a question also closes the topic: a
  "Why?" after it asks about the refusal, not about Fillet and Chamfer.

"How" and "why" inside a longer message are not cues: "How are you?", "How does SOLIDWORKS work?"
and "How do I delete a feature?" are new questions, not follow-ups to whatever came before.

A message that names a term is retrieved as typed, with one exception: when it points back with
"it" or "that" and compares ("How is it different from a Chamfer?" after a Fillet question), the
other side of the comparison is the active topic, so the topic question is prepended and both
records are retrieved. After a question about several terms, "the first one", "the second one",
"the former" and "the latter" pick one of them in the order the student wrote them; the retrieval
query is then that term (one the student wrote) followed by the message.

When a message refers back but the topic cannot be resolved unambiguously — there is no open topic
in the window, or "it" could mean either of two terms — the result is ``clarify``: the tutor asks
which topic is meant instead of guessing.

**The model-facing turns.** The bounded history is passed to the model as ordinary prior turns, and
only the final user turn carries the retrieved records through the frozen Stage-C wrapper. The
browser's history is already bounded by the contract (6 messages, 2000 characters each, 6000 in
total); this module bounds it again before use, because a bound that is only checked at the edge is
a bound that moves when the edge moves.

The original message, the derived retrieval query, the resolution and the topic are all reported
for provenance, and the derived query is never shown to the student as though they had typed it.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from services.api.app.models.course import ConversationTurn

#: Hard ceilings applied here as well as in the contract. Lowest wins.
MAX_HISTORY_TURNS = 6
MAX_TURN_CHARACTERS = 2000
MAX_TOTAL_HISTORY_CHARACTERS = 6000
#: The retrieval query is two user messages at most; keep it bounded for the same reason.
MAX_RETRIEVAL_QUERY_CHARACTERS = 4000

#: How a message was resolved. Stable strings: they are written to the provenance log.
RESOLUTION_EXPLICIT = "explicit_term"
RESOLUTION_FOLLOW_UP = "follow_up"
RESOLUTION_NEW_QUESTION = "new_question"
RESOLUTION_CLARIFY = "clarify"

#: Whole-word anaphors and demonstratives: a message using one points back at what was just said.
_ANAPHORS: frozenset[str] = frozenset(
    {
        "it",
        "its",
        "it's",
        "that",
        "this",
        "these",
        "those",
        "them",
        "they",
        "their",
        "one",
        "ones",
        "same",
        "above",
    }
)
#: Continuation cues: words that ask for more about the current topic rather than naming one. A cue
#: makes a message a follow-up only when it brings no new subject word, and since the conversational
#: list below is broad, a cue must be a word a tutoring follow-up uses and a fresh question rarely
#: does. "How" and "why" are not listed: they open almost every question ("How are you?", "How
#: does SOLIDWORKS work?"), so they count only as a whole message (_CONTINUATION_MESSAGES). Nor is
#: "else", which as often changes the subject ("help me with something else") as it continues one.
_CONTINUATION_WORDS: frozenset[str] = frozenset(
    {
        "next",
        "step",
        "steps",
        "simpler",
        "simply",
        "again",
        "more",
        "example",
        "examples",
        "detail",
        "details",
        "detailed",
        "difference",
        "differences",
        "elaborate",
        "continue",
        "explain",
        "clarify",
        "rephrase",
    }
)
#: Whole messages that accept the tutor's offer to explain ("I can explain the concept …"):
#: continuations of the open topic, and conceptual requests of their own (see request_kind).
_ACCEPTANCES: frozenset[str] = frozenset(
    {
        "yes",
        "yes please",
        "yes pls",
        "please",
        "sure",
        "sure thing",
        "ok",
        "okay",
        "ok please",
        "okay please",
        "go ahead",
        "please do",
        "fine",
        "alright",
        "yeah",
        "yep",
        "the concept",
        "concept",
        "yes the concept",
    }
)
#: Whole messages that are continuations on their own ("Why?", "How?", "And then?").
_CONTINUATION_MESSAGES: frozenset[str] = frozenset(
    {
        "why",
        "why not",
        "but why",
        "how",
        "how so",
        "how come",
        "what",
        "what else",
        "anything else",
        "and",
        "and then",
        "then what",
        "go on",
        "keep going",
        "tell me more",
        "really",
    }
)
#: Words that never name a course subject: question words, pronouns, contraction pieces, generic
#: verbs and discourse words that the frozen stopword list (built for benchmark questions) keeps.
#: A follow-up may contain any of these; a word outside this set and outside the active topic's
#: authorized vocabulary is a NEW subject.
_CONVERSATIONAL: frozenset[str] = frozenset(
    {
        # question words, pronouns and demonstratives the frozen stopword list keeps
        "what",
        "why",
        "how",
        "where",
        "who",
        "whom",
        "whose",
        "i",
        "me",
        "my",
        "mine",
        "we",
        "us",
        "our",
        "he",
        "she",
        "him",
        "her",
        "his",
        "those",
        "these",
        "above",
        "below",
        # pieces the tokenizer leaves from contractions: what's, don't, I'm, I've, you're, I'll, I'd
        "s",
        "t",
        "m",
        "ve",
        "re",
        "ll",
        "d",
        "don",
        "doesn",
        "didn",
        "isn",
        "aren",
        "wasn",
        "won",
        "couldn",
        "wouldn",
        "shouldn",
        # modals and auxiliaries
        "could",
        "would",
        "should",
        "might",
        "shall",
        "am",
        "been",
        "being",
        "were",
        "had",
        "did",
        "doing",
        "done",
        # greetings and acknowledgements, so "thanks" between two follow-ups changes no subject
        "hi",
        "hello",
        "hey",
        "bye",
        "goodbye",
        "cool",
        "great",
        "nice",
        "awesome",
        "perfect",
        "alright",
        "cheers",
        # the tutor's own vocabulary for what it offers: "explain the concept" names no subject
        "concept",
        "concepts",
        "definition",
        "definitions",
        "define",
        "defines",
        "defined",
        "simplify",
        "summarize",
        "summarise",
        "summary",
        "shorten",
        "idea",
        "basics",
        "theory",
        "overview",
        "ahead",
        # discourse words
        "actually",
        "really",
        "just",
        "again",
        "more",
        "less",
        "much",
        "very",
        "please",
        "pls",
        "thanks",
        "thank",
        "ok",
        "okay",
        "sure",
        "yes",
        "yeah",
        "well",
        "now",
        "also",
        "too",
        "still",
        "even",
        "maybe",
        "perhaps",
        "about",
        "here",
        "there",
        "though",
        "anyway",
        "so",
        "basically",
        "exactly",
        "simply",
        "simple",
        "simpler",
        "simplest",
        "easier",
        "easy",
        "hard",
        "harder",
        "difficult",
        "tricky",
        "complicated",
        "complex",
        "fast",
        "slow",
        "long",
        "common",
        "important",
        "useful",
        "worth",
        "good",
        "bad",
        "better",
        "best",
        "plain",
        "plainly",
        "quick",
        "quickly",
        "briefly",
        "short",
        "shorter",
        "little",
        "bit",
        "lot",
        "kind",
        "sort",
        "way",
        "ways",
        "thing",
        "things",
        "something",
        "anything",
        "everything",
        "other",
        "another",
        "else",
        "instead",
        "rather",
        "same",
        "similar",
        # asking and explaining
        "explain",
        "explained",
        "explaining",
        "explanation",
        "describe",
        "tell",
        "show",
        "say",
        "said",
        "mean",
        "means",
        "meant",
        "meaning",
        "clarify",
        "elaborate",
        "expand",
        "rephrase",
        "repeat",
        "understand",
        "understood",
        "know",
        "get",
        "got",
        "see",
        "help",
        "walk",
        "through",
        "go",
        "going",
        "goes",
        "continue",
        "keep",
        "give",
        "guide",
        "teach",
        "learn",
        "ask",
        "answer",
        "question",
        "talk",
        # generic actions
        "make",
        "makes",
        "making",
        "made",
        "use",
        "uses",
        "used",
        "using",
        "work",
        "works",
        "working",
        "worked",
        "happen",
        "happens",
        "need",
        "needs",
        "want",
        "wants",
        "try",
        "set",
        "sets",
        "setting",
        "settings",
        "choose",
        "pick",
        "change",
        "changes",
        "apply",
        "start",
        "begin",
        "finish",
        "finished",
        "completed",
        "complete",
        "end",
        "put",
        "add",
        "remove",
        "removes",
        "select",
        "selects",
        "create",
        "creates",
        "delete",
        "edit",
        "specify",
        # sequence and comparison
        "step",
        "steps",
        "next",
        "first",
        "second",
        "third",
        "former",
        "latter",
        "last",
        "final",
        "previous",
        "current",
        "needed",
        "required",
        "necessary",
        "before",
        "after",
        "afterwards",
        "later",
        "earlier",
        "once",
        "finally",
        "order",
        "sequence",
        "difference",
        "differences",
        "different",
        "differ",
        "differs",
        "compare",
        "compared",
        "comparison",
        "versus",
        "vs",
        "between",
        "both",
        "either",
        "neither",
        "each",
        "all",
        # amounts and examples
        "example",
        "examples",
        "instance",
        "case",
        "cases",
        "detail",
        "details",
        "detailed",
        "value",
        "values",
        "number",
        "amount",
        "one",
        "ones",
        "new",
        # generic interface words: asking WHERE a control is is still asking about the topic
        "feature",
        "features",
        "tool",
        "tools",
        "command",
        "commands",
        "button",
        "buttons",
        "menu",
        "menus",
        "option",
        "options",
        "dialog",
        "box",
        "click",
        "solidworks",
    }
)
#: References that demand ONE referent: "how do I do it?" after a two-term question is ambiguous.
_SINGULAR_REFERENCES: frozenset[str] = frozenset({"it", "its", "it's", "one"})
#: Words that make a multi-term topic the intended referent ("the difference", "both", "them",
#: "which should I use?"). After a question about several terms they also count as referring back.
_COMPARISON_WORDS: frozenset[str] = frozenset(
    {
        "difference",
        "differences",
        "different",
        "differ",
        "compare",
        "comparison",
        "versus",
        "vs",
        "both",
        "between",
        "them",
        "they",
        "these",
        "those",
        "each",
        "either",
        "which",
    }
)
#: What lets a message that names a term point back at another one: "How is IT different from a
#: Chamfer?" needs a singular pointer and a comparison, both, before the topic joins the query.
_POINTERS: frozenset[str] = frozenset({"it", "its", "it's", "this", "that"})
_COMPARING: frozenset[str] = frozenset(
    {
        "difference",
        "differences",
        "different",
        "differ",
        "differs",
        "compare",
        "compared",
        "compares",
        "comparison",
        "versus",
        "vs",
        "than",
        "similar",
        "same",
        "between",
    }
)
#: After a question about several terms, a position picks one, in the order the student wrote them
#: — the same order the clarifying question lists them in.
_POSITIONS: dict[str, int] = {"first": 0, "former": 0, "second": 1, "latter": -1}
_WORD_RE = re.compile(r"[a-z0-9']+")

# ---------------------------------------------------------------- what the message asks for
#: What a message asks for. Four kinds, stable strings written to the provenance log:
#: an explanation of a concept; operations to perform in the software; a request that could be
#: either, which the tutor asks about; or a greeting / control message the assistant layer
#: answers on its own.
REQUEST_CONCEPT_EXPLANATION = "concept_explanation"
REQUEST_PROCEDURAL_INSTRUCTION = "procedural_instruction"
REQUEST_AMBIGUOUS = "ambiguous_request"
REQUEST_CONVERSATION_CONTROL = "conversation_control"
#: The short names this module uses for the two kinds that drive retrieval and the gate.
REQUEST_CONCEPTUAL = REQUEST_CONCEPT_EXPLANATION
REQUEST_PROCEDURAL = REQUEST_PROCEDURAL_INSTRUCTION

#: Why a turn was answered with a clarifying question instead of retrieval.
CLARIFY_NO_TOPIC = "no_topic"
CLARIFY_WHICH_TERM = "which_term"
CLARIFY_AMBIGUOUS_REQUEST = "ambiguous_request"

#: Question words and request verbs make a message a request of its own. A question word counts
#: at the start of the message, or anywhere when the message asks with a "?"; "Fillet is what I
#: meant" asks nothing. A message with neither ("Fillet", "the first one", "both", "Let's do
#: Fillet") answers the tutor's own question and carries on the request the student made before.
_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "what",
        "what's",
        "whats",
        "how",
        "how's",
        "why",
        "when",
        "where",
        "where's",
        "which",
        "who",
        "whom",
        "whose",
    }
)
_REQUEST_VERBS: frozenset[str] = frozenset(
    {
        "explain",
        "describe",
        "tell",
        "show",
        "give",
        "help",
        "compare",
        "define",
        "walk",
        "guide",
        "teach",
        "list",
        "name",
        "summarize",
        "summarise",
        "clarify",
        "elaborate",
        "rephrase",
        "repeat",
    }
)
#: A message that OPENS with one of these is a yes/no question of its own ("Is it hard?", "Can we
#: do Fillet?"); the same words elsewhere ("Fillet is what I meant") make no request.
_YES_NO_STARTERS: frozenset[str] = frozenset(
    {
        "is",
        "are",
        "am",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "has",
        "have",
        "had",
    }
)
#: Whole messages that ask for the next operation ("next", "done") …
_NEXT_MESSAGES: frozenset[str] = frozenset(
    {
        "next",
        "what's next",
        "whats next",
        "what next",
        "what comes next",
        "what now",
        "and then",
        "then what",
        "what then",
        "and next",
        "next step",
        "next steps",
        "the next step",
        "next please",
        "ok next",
        "okay next",
        "done",
        "finished",
        "i'm done",
        "im done",
        "did it",
        "i did it",
        "i did that",
        "ok done",
        "done next",
        "and after that",
        "what after that",
    }
)
#: … or to begin the lesson's steps from the first one …
_RESTART_MESSAGES: frozenset[str] = frozenset(
    {"start", "from the start", "start over", "start again", "restart", "from the beginning"}
)
#: … or for the operations as such. All of them ask for a procedure.
_PROCEDURAL_MESSAGES: frozenset[str] = (
    _NEXT_MESSAGES | _RESTART_MESSAGES | frozenset({"how", "steps", "the steps", "the procedure"})
)
#: Words a progress report may carry around its number: "I finished step 2", "done with 2".
_PROGRESS_FILLER: frozenset[str] = frozenset(
    {
        "step",
        "steps",
        "i",
        "i'm",
        "im",
        "i've",
        "ive",
        "did",
        "done",
        "finished",
        "completed",
        "complete",
        "with",
        "at",
        "on",
        "the",
        "number",
        "ok",
        "okay",
        "have",
        "finish",
        "just",
        "now",
        "am",
        "yes",
        "so",
        "far",
        "up",
        "to",
    }
)
_ORDINAL_NUMBERS: dict[str, int] = {
    "first": 1,
    "one": 1,
    "second": 2,
    "two": 2,
    "third": 3,
    "three": 3,
    "fourth": 4,
    "four": 4,
    "fifth": 5,
    "five": 5,
    "sixth": 6,
    "six": 6,
}
#: Interface and procedure nouns: a message using one asks how to operate, not what something is.
_PROCEDURE_NOUNS: frozenset[str] = frozenset(
    {
        "step",
        "steps",
        "instruction",
        "instructions",
        "procedure",
        "procedures",
        "process",
        "operation",
        "operations",
        "workflow",
        "tutorial",
        "walkthrough",
        "click",
        "clicks",
        "clicking",
        "button",
        "buttons",
        "menu",
        "menus",
        "toolbar",
        "toolbars",
        "icon",
        "icons",
        "dialog",
        "dialogs",
        "tab",
        "tabs",
        "shortcut",
        "shortcuts",
        "hotkey",
        "hotkeys",
        "keyboard",
        "command",
        "commands",
        "propertymanager",
        "checkmark",
        "tool",
        "tools",
        "mouse",
        "cursor",
    }
)
#: Phrases that ask to be shown how, wherever they sit in the message.
_PROCEDURE_PHRASES: tuple[str, ...] = (
    "walk me through",
    "walk through",
    "guide me through",
    "take me through",
    "talk me through",
    "show me how",
    "show how",
    "step by step",
    "how to",
    "what do i do",
    "what should i do",
    "what must i do",
    "get started",
    "getting started",
)
#: Comparison cues: a message asking for a difference is conceptual even when it says "walk me
#: through" — unless it also names a procedure noun such as "steps".
_COMPARISON_CUES: frozenset[str] = frozenset(
    {
        "difference",
        "differences",
        "different",
        "differ",
        "differs",
        "compare",
        "compared",
        "comparison",
        "versus",
        "vs",
    }
)
#: After "how do I", a verb that asks to know rather than to do: "how do I know which to use?".
#: "choose" and "pick" are here too: "how do I choose between them?" is a decision, not a click.
_KNOWING_VERBS: frozenset[str] = frozenset(
    {
        "know",
        "tell",
        "understand",
        "recognize",
        "recognise",
        "decide",
        "identify",
        "distinguish",
        "differentiate",
        "compare",
        "read",
        "interpret",
        "think",
        "say",
        "explain",
        "describe",
        "learn",
        "remember",
        "judge",
        "check",
        "verify",
        "determine",
        "figure",
        "choose",
        "pick",
        "see",
        "visualize",
        "visualise",
        "imagine",
        "work",
        "works",
        "differ",
        "relate",
        "look",
        "mean",
        "feel",
        "guess",
        "estimate",
        "avoid",
        "need",
        "want",
        "expect",
        "ask",
    }
)
#: Who acts in "how do I …": the question is about the actor's operations, not about the thing.
_ACTORS: frozenset[str] = frozenset(
    {"i", "we", "you", "u", "one", "someone", "somebody", "anyone", "people", "students", "users"}
)
_AUXILIARIES: frozenset[str] = frozenset(
    {"do", "does", "did", "can", "could", "would", "should", "shall", "will", "must", "might", "to"}
)
#: Adverbs skipped on the way to a verb ("how do I actually do it", "I want to just add one").
_ADVERBS: frozenset[str] = frozenset(
    {
        "actually",
        "really",
        "just",
        "then",
        "even",
        "exactly",
        "properly",
        "correctly",
        "quickly",
        "normally",
        "usually",
        "best",
        "also",
        "now",
        "first",
        "next",
        "generally",
        "typically",
        "manually",
        "please",
        "simply",
        "easily",
        "basically",
        "still",
        "again",
        "ever",
        "possibly",
    }
)
#: Words skipped between "how do I" and its verb: adverbs, articles and pronouns.
_SKIPPABLE: frozenset[str] = _ADVERBS | frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "it",
        "them",
        "these",
        "those",
        "my",
        "your",
        "our",
        "go",
        "about",
    }
)
#: "How is it done?", "how would a fillet be created?": the passive asks for the operations.
_PASSIVE_MARKERS: frozenset[str] = frozenset(
    {"is", "are", "was", "were", "be", "been", "being", "get", "gets", "got"}
)
#: Participles after a passive marker: "how is it done?" asks for operations; "how is it used?"
#: does not.
_DONE_PARTICIPLES: frozenset[str] = frozenset(
    {
        "done",
        "made",
        "created",
        "built",
        "applied",
        "added",
        "drawn",
        "performed",
        "achieved",
        "accomplished",
        "inserted",
        "generated",
        "removed",
        "selected",
        "modelled",
        "modeled",
    }
)
#: Interface actions: "which faces do I <verb>?", "can you <verb> it?", "I want to <verb>", or an
#: imperative "<verb> a fillet". A closed list, unlike the open "how do I <verb>" form, so that
#: "which one should I use?" stays a choice.
_ACTIONS: frozenset[str] = frozenset(
    {
        "click",
        "select",
        "press",
        "drag",
        "enter",
        "type",
        "draw",
        "sketch",
        "extrude",
        "revolve",
        "cut",
        "insert",
        "remove",
        "delete",
        "create",
        "build",
        "do",
        "perform",
        "generate",
        "launch",
        "activate",
        "enable",
        "toggle",
        "place",
        "move",
        "rotate",
        "mirror",
        "pattern",
        "hollow",
        "round",
        "bevel",
        "attach",
        "assign",
    }
)
#: Verbs that are interface actions only with a course term as their object: "can you add a
#: Fillet?" asks for operations, "could you add more detail?" asks for words.
_AMBIGUOUS_ACTIONS: frozenset[str] = frozenset(
    {
        "make",
        "add",
        "put",
        "apply",
        "change",
        "set",
        "modify",
        "edit",
        "adjust",
        "specify",
        "finish",
        "complete",
        "save",
        "undo",
        "redo",
        "copy",
        "paste",
        "trim",
        "extend",
        "offset",
        "open",
        "start",
        "run",
    }
)
#: After "where do I", only an interface verb makes a how-to: "where do I click?" asks for the
#: interface, "where do I use a fillet?" asks where the feature belongs.
_WHERE_ACTIONS: frozenset[str] = frozenset(
    {
        "click",
        "find",
        "select",
        "press",
        "enter",
        "type",
        "go",
        "look",
        "start",
        "set",
        "specify",
        "pick",
        "choose",
        "add",
        "insert",
        "put",
        "apply",
        "turn",
        "activate",
        "enable",
        "open",
        "access",
        "locate",
        "get",
        "define",
    }
)
#: "I want to <verb>", "I'm trying to <verb>", "what is the best way to <verb>".
_INTENT_LEADS: frozenset[str] = frozenset(
    {"want", "need", "like", "trying", "going", "wish", "hoping", "have", "got", "supposed", "way"}
)


def _is_action(verb: str, named: bool) -> bool:
    return verb in _ACTIONS or (named and verb in _AMBIGUOUS_ACTIONS)


def _asks_how_to_act(ordered: Sequence[str]) -> bool:
    """ "How do I <verb>?", "how to <verb>", "how is it done?", "where do I click?".

    The verb after the actor decides, and the form is open: any verb counts unless it asks to know
    ("how do I know which to use?"). Without an actor, "how to" reaches a verb and the passive
    reaches a participle ("how is a fillet created?"); "how does it work?" and "how is a fillet
    different?" are questions about the thing. After "where", only an interface verb counts.
    """
    for index, word in enumerate(ordered):
        if word not in ("how", "where"):
            continue
        marker: str | None = None
        actor_seen = to_seen = be_seen = False
        for later in ordered[index + 1 : index + 10]:
            if actor_seen:
                if later in _SKIPPABLE:
                    continue
                if word == "where":
                    return later in _WHERE_ACTIONS
                return later not in _KNOWING_VERBS
            if later in _AUXILIARIES:
                marker = marker or "aux"
                to_seen = to_seen or later == "to"
                continue
            if later in _PASSIVE_MARKERS:
                marker = marker or "passive"
                be_seen = True
                continue
            if marker == "aux" and later in _ACTORS:
                actor_seen = True
                continue
            if marker is None or later in _SKIPPABLE:
                # "how exactly …", "how thick …", "tell me how a fillet is created": keep looking
                continue
            if to_seen and not be_seen:
                return later not in _KNOWING_VERBS
            if be_seen and later in _DONE_PARTICIPLES:
                return True
        if actor_seen:
            return True  # "How do I go about it?": only skippable words followed the actor
    return False


def _asks_which_action(ordered: Sequence[str], named: bool) -> bool:
    """ "Which faces do I select?", "can you create it?", "can we do Fillet?", "help me do it"."""
    for index, word in enumerate(ordered):
        window = ordered[index + 1 : index + 8]
        if word in ("what", "which"):
            aux_seen = actor_seen = False
            for later in window:
                if later in _AUXILIARIES or later in _YES_NO_STARTERS:
                    aux_seen = True
                elif later in _ACTORS:
                    actor_seen = aux_seen
                elif actor_seen and later not in _SKIPPABLE:
                    if later in _ACTIONS:
                        return True
                    break
            continue
        addressed = (
            word in ("can", "could", "would", "will", "shall")
            and window[:1]
            and window[0] in ("you", "u", "we", "i")
        ) or (word == "help" and window[:1] == ["me"])
        if addressed:
            following = [w for w in window[1:4] if w not in _SKIPPABLE]
            if following and _is_action(following[0], named):
                return True
    return False


def _states_intent_to_act(ordered: Sequence[str], named: bool) -> bool:
    """ "I want to create a fillet", "the best way to add one": an intention to operate."""
    for index, word in enumerate(ordered):
        if word in _INTENT_LEADS and ordered[index + 1 : index + 2] == ["to"]:
            following = [w for w in ordered[index + 2 : index + 5] if w not in _ADVERBS]
            if following and _is_action(following[0], named):
                return True
    return False


def _is_imperative(ordered: Sequence[str], names_term: Callable[[str], bool] | None) -> bool:
    """ "Create a fillet", "Add a fillet to the edge", "Do Fillet", "Let's do Chamfer", "Fillet the
    edge": an instruction to operate, including a course term used as a verb."""
    named = bool(names_term(" ".join(ordered))) if names_term is not None else False
    first = ordered[0]
    if first in ("let's", "lets") and len(ordered) > 1:
        return _is_action(ordered[1], named) or ordered[1] in ("try", "begin")
    if first == "do":
        return named or ordered[1:2] in (["it"], ["that"], ["this"], ["one"])
    if first in _ACTIONS or (named and first in _AMBIGUOUS_ACTIONS):
        return True
    # a term used as a verb: "fillet the edge", "shell this part", "chamfer it"
    return (
        len(ordered) > 1
        and names_term is not None
        and bool(names_term(first))
        and ordered[1] in _OBJECT_LEADS
    )


#: What follows a term used as a verb: "fillet THE edge", "shell IT".
_OBJECT_LEADS: frozenset[str] = frozenset(
    {"the", "a", "an", "this", "that", "it", "my", "these", "those", "all", "every", "each", "both"}
)


def _is_ambiguous(ordered: Sequence[str], named: bool) -> bool:
    """ "Show me a fillet", "how does it work?": could ask what it is or how to operate it."""
    words = set(ordered)
    if ordered[:2] == ["show", "me"] and (named or words & {"it", "one", "this", "that"}):
        return True
    return (
        len(ordered) >= 3
        and ordered[0] == "how"
        and ordered[1] in ("does", "do", "did")
        and bool(words & {"work", "works"})
    )


def progress_step(message: str) -> int | None:
    """The step a progress report names ("2", "step 2", "I finished step two"), or None."""
    ordered = _words(message)
    content = [word for word in ordered if word not in _PROGRESS_FILLER]
    if not ordered or len(content) != 1:
        return None
    word = content[0]
    if word.isdigit():
        return int(word)
    return _ORDINAL_NUMBERS.get(word)


#: Words that ask what something means rather than how to do it: with "step" (or "this",
#: "that") they make a request to explain an authorized step, which never advances a lesson.
_EXPLANATION_CUES: frozenset[str] = frozenset(
    {
        "explain",
        "explained",
        "mean",
        "means",
        "meaning",
        "meant",
        "describe",
        "why",
        "clarify",
        "elaborate",
        "simpler",
        "simply",
        "purpose",
        "understand",
        "define",
        "what",
        "what's",
    }
)
_CURRENT_STEP_WORDS: frozenset[str] = frozenset({"this", "that", "current", "it", "last"})


def step_mention(message: str) -> int | None:
    """The step a message names ("step 2", "the second step", "step two"), or None."""
    ordered = _words(message)
    for index, word in enumerate(ordered):
        if word not in ("step", "steps"):
            continue
        for neighbour in (ordered[index + 1 : index + 2], ordered[index - 1 : index]):
            if not neighbour:
                continue
            candidate = neighbour[0]
            if candidate.isdigit():
                return int(candidate)
            if candidate in _ORDINAL_NUMBERS:
                return _ORDINAL_NUMBERS[candidate]
    return None


def is_step_explanation(message: str) -> bool:
    """ "What does step 2 mean?", "Explain this step more simply", "Why do I need to do that?".

    An explanation of a step is a concept request about that step: it never asks for the next
    operation and never advances a lesson. "What's the next step?" asks for the next operation.
    """
    ordered = _words(message)
    words = set(ordered)
    if not words & _EXPLANATION_CUES or words & {"next", "following"}:
        return False
    if "step" in words or "steps" in words:
        return True
    return ordered[0] == "why" and bool(words & _CURRENT_STEP_WORDS)


def refers_to_current_step(message: str) -> bool:
    """ "This step", "that", "the current step": the step the lesson is at, not a numbered one."""
    return step_mention(message) is None and bool(set(_words(message)) & _CURRENT_STEP_WORDS)


def is_next_request(message: str) -> bool:
    """ "next", "done", "what's next": the student asks for the next step of a lesson."""
    return " ".join(_words(message)) in _NEXT_MESSAGES


def is_restart_request(message: str) -> bool:
    """ "start", "from the beginning": the student asks for the lesson's first step."""
    return " ".join(_words(message)) in _RESTART_MESSAGES


def request_kind(message: str, names_term: Callable[[str], bool] | None = None) -> str:
    """What a message asks for: one of the four REQUEST_* kinds.

    Deterministic and lexical. "How" alone never decides: "How is a Fillet different from a
    Chamfer?" is conceptual, "How do I use Shell?", "What should I click?", "Walk me through the
    steps." and "What is the next operation?" are procedural, and "How does it work?" or "Show me
    a fillet" could be either, so they are ambiguous and the tutor asks. A "why" question asks for
    a reason, and a request to compare asks for an explanation, whatever else they contain.
    ``names_term`` says whether a text names a course term, which is what makes "can you add a
    Fillet?" an operation, "could you add more detail?" a request for words, and "Fillet the edge"
    an instruction.
    """
    from services.api.app.services.assistant_interaction import detect_assistant_interaction

    if detect_assistant_interaction(message) is not None:
        return REQUEST_CONVERSATION_CONTROL
    ordered = _words(message)
    if not ordered:
        return REQUEST_CONCEPTUAL
    joined = " ".join(ordered)
    if joined in _ACCEPTANCES:
        return REQUEST_CONCEPTUAL
    if joined in _PROCEDURAL_MESSAGES or progress_step(message) is not None:
        return REQUEST_PROCEDURAL
    if is_step_explanation(message):
        return REQUEST_CONCEPTUAL
    words = set(ordered)
    if words & _PROCEDURE_NOUNS:
        return REQUEST_PROCEDURAL
    if ordered[0] == "why" or words & _COMPARISON_CUES:
        return REQUEST_CONCEPTUAL
    named = bool(names_term(message)) if names_term is not None else False
    padded = f" {joined} "
    if any(f" {phrase} " in padded for phrase in _PROCEDURE_PHRASES):
        return REQUEST_PROCEDURAL
    if _is_ambiguous(ordered, named):
        return REQUEST_AMBIGUOUS
    if (
        _asks_how_to_act(ordered)
        or _asks_which_action(ordered, named)
        or _states_intent_to_act(ordered, named)
        or _is_imperative(ordered, names_term)
    ):
        return REQUEST_PROCEDURAL
    return REQUEST_CONCEPTUAL


def has_request_shape(message: str, names_term: Callable[[str], bool] | None = None) -> bool:
    """Whether a message is a request of its own, rather than an answer to the tutor's question.

    A request verb, a question word opening the message (or anywhere in a message with a "?"), a
    yes/no question opening ("Is it hard?"), an acceptance of the tutor's offer, or any form that
    asks for operations ("Create a fillet"). "Fillet", "the first one", "both", "Let's do Fillet",
    "Fillet would be good" and "Fillet is what I meant" have none.
    """
    ordered = _words(message)
    if not ordered:
        return False
    words = set(ordered)
    if words & _REQUEST_VERBS or ordered[0] in _QUESTION_WORDS or ordered[0] in _YES_NO_STARTERS:
        return True
    if words & _QUESTION_WORDS and "?" in str(message):
        return True
    joined = " ".join(ordered)
    if joined in _ACCEPTANCES or joined in _PROCEDURAL_MESSAGES:
        return True
    return request_kind(message, names_term) == REQUEST_PROCEDURAL


def previous_request(
    turns: Sequence[ConversationTurn], names_term: Callable[[str], bool] | None = None
) -> tuple[str, str] | None:
    """The most recent earlier USER message that made a definite request, with its kind.

    A greeting or control message is skipped — the tutor answered it as such — and so is an
    ambiguous request, which asked nothing definite. Assistant text is never consulted.
    """
    for turn in reversed(list(turns)):
        if turn.role != "user":
            continue
        content = turn.content.strip()
        kind = request_kind(content, names_term)
        if kind in (REQUEST_CONVERSATION_CONTROL, REQUEST_AMBIGUOUS):
            continue
        if has_request_shape(content, names_term):
            return kind, content
    return None


def resolve_request_kind(
    current: str,
    turns: Sequence[ConversationTurn],
    names_term: Callable[[str], bool] | None = None,
) -> tuple[str, str | None]:
    """The kind of request this turn carries, and the earlier user message it was read from.

    "The first one" or "Fillet", sent in reply to "Do you mean Fillet or Chamfer?", carries no
    request of its own: it answers the tutor and continues the request before it ("How do I do
    it?"). So a message without a request shape inherits the kind of the most recent earlier USER
    message that made a definite request (see previous_request).
    """
    if has_request_shape(current, names_term):
        return request_kind(current, names_term), None
    previous = previous_request(turns, names_term)
    if previous is None:
        return REQUEST_CONCEPTUAL, None
    return previous[0], previous[1]


def _words(message: str) -> list[str]:
    """Casefolded words, apostrophes kept ("it's"), for the anaphor and cue tests."""
    return _WORD_RE.findall(unicodedata.normalize("NFKC", str(message)).casefold())


def refers_back(message: str) -> bool:
    """Whether this message points at something already said through an anaphor."""
    return bool(set(_words(message)) & _ANAPHORS)


def is_acknowledgement(message: str) -> bool:
    """Whether this message only accepts or acknowledges ("ok", "sure") and asks for nothing."""
    return " ".join(_words(message)) in _ACCEPTANCES


def is_continuation(message: str) -> bool:
    """Whether this message asks to continue rather than naming a subject ("why?", "next step")."""
    words = _words(message)
    joined = " ".join(words)
    if joined in _CONTINUATION_MESSAGES or joined in _ACCEPTANCES:
        return True
    if joined in _NEXT_MESSAGES or joined in _RESTART_MESSAGES:
        return True  # "done", "start": the open lesson continues
    if progress_step(message) is not None:
        return True
    return bool(set(words) & _CONTINUATION_WORDS)


def subject_words(message: str) -> frozenset[str]:
    """The message's content words that could name a subject: frozen tokens, minus conversation.

    Digits are dropped as well ("what's step 2?" names no new subject). The frozen tokenizer is
    imported here, not at module scope: ``model_mvp`` is not part of the installed distribution,
    and this module must stay importable wherever the default API is.
    """
    from model_mvp.rag.basic_retriever import tokenize

    return frozenset(
        token
        for token in tokenize(str(message))
        if token not in _CONVERSATIONAL and not token.isdigit()
    )


@dataclass(frozen=True)
class ActiveTopic:
    """The most recent topic the student named themselves."""

    #: The user's own earlier message that named the topic, verbatim.
    question: str
    #: The canonical terms that message names, in the corpora's deterministic order.
    terms: tuple[str, ...]


@dataclass(frozen=True)
class ConversationContext:
    """What this request will actually use, after bounding, plus the provenance to record."""

    turns: tuple[ConversationTurn, ...]
    original_user_message: str
    retrieval_query: str
    #: True when the retrieval query was extended with the active topic's user question.
    follow_up_augmented: bool
    #: One of the RESOLUTION_* strings.
    resolution: str = RESOLUTION_NEW_QUESTION
    #: The terms this turn is about: those it names, those a follow-up was resolved to, or the
    #: candidates a clarification asks between.
    topic_terms: tuple[str, ...] = ()
    #: One of the REQUEST_* kinds: what the turn asks for (see resolve_request_kind).
    request_kind: str = REQUEST_CONCEPTUAL
    #: The earlier user message the request kind was read from, when this turn only answered the
    #: tutor's question; None when the turn is a request of its own.
    request_inherited_from: str | None = None
    #: One of the CLARIFY_* reasons when the resolution is a clarification, else None.
    clarification_reason: str | None = None
    #: True when the turn asks what an authorized step means (never for the next operation).
    step_explanation: bool = False
    #: The step the turn names ("step 2"), if any; a step explanation or a direct request for it.
    step_number: int | None = None
    #: True when the turn means the lesson's current step ("this step", "that") rather than a
    #: numbered one.
    current_step: bool = False

    @property
    def needs_clarification(self) -> bool:
        return self.resolution == RESOLUTION_CLARIFY

    @property
    def asks_for_procedure(self) -> bool:
        return self.request_kind == REQUEST_PROCEDURAL

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "original_user_message": self.original_user_message,
            "retrieval_query": self.retrieval_query,
            "follow_up_augmented": self.follow_up_augmented,
            "resolution": self.resolution,
            "topic_terms": list(self.topic_terms),
            "request_kind": self.request_kind,
            "request_inherited_from": self.request_inherited_from,
            "clarification_reason": self.clarification_reason,
            "step_explanation": self.step_explanation,
            "step_number": self.step_number,
            "history_turns_used": len(self.turns),
            "history_characters_used": sum(len(turn.content) for turn in self.turns),
        }


def bound_history(history: Sequence[ConversationTurn] | None) -> tuple[ConversationTurn, ...]:
    """The most recent turns that fit the bounds, oldest first, newest kept.

    Trimming drops the OLDEST turns, because a follow-up depends on what was just said.
    """
    if not history:
        return ()
    recent = [
        turn for turn in history if turn.role in ("user", "assistant") and turn.content.strip()
    ][-MAX_HISTORY_TURNS:]
    kept: list[ConversationTurn] = []
    total = 0
    for turn in reversed(recent):
        content = turn.content[:MAX_TURN_CHARACTERS]
        if total + len(content) > MAX_TOTAL_HISTORY_CHARACTERS:
            break
        total += len(content)
        kept.append(
            turn if content == turn.content else ConversationTurn(role=turn.role, content=content)
        )
    kept.reverse()
    return tuple(kept)


def previous_user_question(turns: Sequence[ConversationTurn]) -> str | None:
    """The most recent user message in the bounded history, or None. Assistant text is ignored."""
    for turn in reversed(list(turns)):
        if turn.role == "user" and turn.content.strip():
            return turn.content.strip()
    return None


def active_topic(
    turns: Sequence[ConversationTurn],
    terms_in: Callable[[str], Sequence[str]],
    vocabulary: Callable[[Sequence[str]], AbstractSet[str]] | None = None,
) -> ActiveTopic | None:
    """The topic still open in the bounded history: the newest USER message naming a course term.

    A follow-up's own message names nothing, so walking back past it is what lets a third or
    fourth turn ("can you explain the thickness setting?") still reach the turn that said "Shell".
    But a user message since then that brought a new subject ("What is the capital of France?")
    closed that topic: a "Why?" after it is about the refusal, so there is no open topic and the
    caller asks instead of reaching back. Given ``vocabulary``, that is checked with the same
    new-subject test the current message gets. A later reply that picked one of several terms
    ("the first one" after a Fillet/Chamfer question) narrowed the topic to that term, so "What is
    it used for?" after it is about Fillet alone. Assistant text is never consulted.
    """
    users = [turn.content.strip() for turn in turns if turn.role == "user" and turn.content.strip()]
    for index in range(len(users) - 1, -1, -1):
        terms = tuple(terms_in(users[index]))
        if not terms:
            continue
        topic = ActiveTopic(question=users[index], terms=_in_mention_order(users[index], terms))
        later_messages = users[index + 1 :]
        if vocabulary is not None:
            topic_vocabulary = vocabulary(topic.terms)
            if any(subject_words(later) - topic_vocabulary for later in later_messages):
                return None
        for later in later_messages:
            chosen = (
                _chosen_by_position(_words(later), topic.terms) if len(topic.terms) > 1 else None
            )
            if chosen is not None:
                topic = ActiveTopic(question=chosen, terms=(chosen,))
        return topic
    return None


def _in_mention_order(message: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    """The terms in the order the student wrote them ("Fillet ... Chamfer" stays Fillet, Chamfer).

    The corpora's matcher order decides only a tie, so the result is still deterministic.
    """
    folded = unicodedata.normalize("NFKC", message).casefold()

    def position(term: str) -> int:
        index = folded.find(unicodedata.normalize("NFKC", term).casefold())
        return index if index >= 0 else len(folded)

    return tuple(sorted(terms, key=lambda term: (position(term), terms.index(term))))


def build_conversation_context(
    question_text: str,
    history: Sequence[ConversationTurn] | None,
    *,
    terms_in: Callable[[str], Sequence[str]],
    vocabulary: Callable[[Sequence[str]], AbstractSet[str]],
    terms_with_steps: AbstractSet[str] = frozenset(),
    fallback_topic: str | None = None,
) -> ConversationContext:
    """Bound the history and resolve the retrieval query deterministically.

    ``terms_in`` returns the canonical course terms a text names and ``vocabulary`` the
    frozen-tokenizer vocabulary of the authorized records that define given terms. Both are
    answered by the retrieval layer against the corpora themselves, so this module never guesses
    what a course term is. ``terms_with_steps`` are the terms the authorized tutorial corpus
    carries steps for: after a request for operations on several terms of which only some have
    steps, an acceptance ("yes") continues with the ones that do. ``fallback_topic`` is the topic
    of a validated lesson position the browser carries: when the bounded history no longer holds
    the message that named it, a continuation ("next", "2", "explain this step") still belongs to
    that lesson.
    """
    turns = bound_history(history)
    current = str(question_text).strip()
    ordered = _words(current)
    words = set(ordered)
    joined = " ".join(ordered)

    def names(text: str) -> bool:
        return bool(terms_in(text))

    kind, inherited_from = resolve_request_kind(current, turns, names_term=names)
    explains_step = kind == REQUEST_CONCEPTUAL and is_step_explanation(current)
    step_number = step_mention(current)
    current_step = explains_step and refers_to_current_step(current)

    def context(
        query: str,
        augmented: bool,
        resolution: str,
        terms: tuple[str, ...] = (),
        *,
        request: str = kind,
        reason: str | None = None,
    ) -> ConversationContext:
        return ConversationContext(
            turns,
            current,
            query[:MAX_RETRIEVAL_QUERY_CHARACTERS],
            augmented,
            resolution,
            topic_terms=terms,
            request_kind=request,
            request_inherited_from=inherited_from,
            clarification_reason=reason,
            step_explanation=explains_step,
            step_number=step_number,
            current_step=current_step,
        )

    named = tuple(terms_in(current))
    mentioned = _in_mention_order(current, named) if named else ()
    topic = active_topic(turns, terms_in, vocabulary)
    if topic is None and fallback_topic:
        topic = ActiveTopic(question=str(fallback_topic), terms=(str(fallback_topic),))

    if kind == REQUEST_AMBIGUOUS:
        # "Show me a fillet", "how does it work?": ask what is wanted before retrieving for it.
        terms = mentioned or (topic.terms if topic is not None else ())
        reason = CLARIFY_AMBIGUOUS_REQUEST if terms else CLARIFY_NO_TOPIC
        return context(current, False, RESOLUTION_CLARIFY, terms, reason=reason)

    if named:
        # "How is it different from a Chamfer?" names one side of a comparison and points back
        # at the other, which only the open topic can supply.
        points_back = bool(words & _POINTERS and words & _COMPARING)
        if points_back and topic is not None and not set(topic.terms) <= set(named):
            added = tuple(t for t in mentioned if t not in topic.terms)
            return context(
                f"{topic.question} {current}", True, RESOLUTION_FOLLOW_UP, topic.terms + added
            )
        return context(current, False, RESOLUTION_EXPLICIT, mentioned)

    topic_vocabulary = vocabulary(topic.terms) if topic is not None else frozenset()
    subjects = subject_words(current)
    if subjects - topic_vocabulary:
        # A word that is neither conversation nor part of the active topic's authorized records:
        # this is a different subject, and it is retrieved exactly as typed.
        return context(current, False, RESOLUTION_NEW_QUESTION)
    several = topic is not None and len(topic.terms) > 1
    chosen = _chosen_by_position(ordered, topic.terms) if topic is not None and several else None
    refers = (
        refers_back(current)
        or is_continuation(current)
        or bool(subjects & topic_vocabulary)
        or (several and bool(words & _COMPARISON_WORDS))
        or chosen is not None
    )
    if not refers:
        return context(current, False, RESOLUTION_NEW_QUESTION)
    if topic is None:
        # It points back at something, but no topic is open in this conversation.
        return context(current, False, RESOLUTION_CLARIFY, reason=CLARIFY_NO_TOPIC)
    if chosen is not None:
        # "The first one": the term is the student's own, from the message that named it.
        return context(f"{chosen} {current}", True, RESOLUTION_FOLLOW_UP, (chosen,))
    if joined in _ACCEPTANCES and several:
        # "Yes" after "steps for Shell, but not for Fillet — I can continue with Shell": continue
        # with the terms that have steps, as the procedure the student asked for.
        previous = previous_request(turns, names)
        supported = tuple(t for t in topic.terms if t in terms_with_steps)
        procedural = previous is not None and previous[0] == REQUEST_PROCEDURAL
        if procedural and supported and len(supported) < len(topic.terms):
            return context(
                f"{supported[0]} {current}",
                True,
                RESOLUTION_FOLLOW_UP,
                supported,
                request=REQUEST_PROCEDURAL,
            )
    if several and words & _SINGULAR_REFERENCES and not words & _COMPARISON_WORDS:
        # "How do I do it?" after a question about two terms: either could be meant.
        return context(current, False, RESOLUTION_CLARIFY, topic.terms, reason=CLARIFY_WHICH_TERM)
    return context(f"{topic.question} {current}", True, RESOLUTION_FOLLOW_UP, topic.terms)


def _chosen_by_position(ordered: Sequence[str], terms: tuple[str, ...]) -> str | None:
    """The one term a positional reply picks ("the first one", "the latter"), or None.

    "First" and "second" pick only with "one" or as the whole reply, so "what's the second step?"
    stays a question about the steps rather than a choice between terms.
    """
    picks = [word for word in ordered if word in _POSITIONS]
    if len(set(picks)) != 1:
        return None
    pick = picks[0]
    standalone = pick in ("former", "latter") or "one" in ordered
    if not standalone and set(ordered) - {"the", pick}:
        return None
    index = _POSITIONS[pick]
    return terms[index] if index < len(terms) else None
