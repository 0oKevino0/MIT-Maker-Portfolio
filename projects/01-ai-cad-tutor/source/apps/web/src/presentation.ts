// Presentation helpers: pure functions that turn validated API values and the frontend-owned
// status machine into display decisions. Nothing here reads the network or invents a fact: every
// label is derived from a validated field, a server warning, or the fixed status table.

import type { CourseAnswer, DocumentType, Locator } from "./api";
import {
  DEMO_ANSWER_WARNING_PREFIX,
  DEMO_CONCEPT_ONLY_PREFIX,
  DEMO_GUIDED_COMPLETE_MARKER,
  DEMO_GUIDED_STEP_MARKER,
  DEMO_LAST_STEP_MARKER,
  DEMO_LESSON_HINT_MARKER,
  DEMO_NO_PROCEDURE_MARKER,
  DEMO_PARTIAL_STEPS_MARKER,
  DEMO_WITHHELD_MARKER,
} from "./demoStatus";
import type { ModelStatus } from "./modelStatus";

/** Semantic tone for a status chip. Always paired with text; colour alone never carries meaning. */
export type Tone = "neutral" | "ok" | "info" | "warn" | "err";

// ---- Header status pill --------------------------------------------------------------------------

export interface StatusPillView {
  readonly label: string;
  readonly tone: Tone;
  /** A check is in flight: the dot pulses (unless the viewer prefers reduced motion). */
  readonly busy: boolean;
}

/** Compact, frontend-owned label for the header. The full three-line status text stays in the
 * System details panel; this is the one-glance version of the same measured state. */
export function statusPill(status: ModelStatus): StatusPillView {
  switch (status) {
    case "checking":
      return { label: "Checking status…", tone: "neutral", busy: true };
    case "ready":
      return { label: "Model ready", tone: "ok", busy: false };
    case "api_unreachable":
      return { label: "API offline", tone: "err", busy: false };
    case "api_alive_model_not_configured":
      return { label: "Model not configured", tone: "warn", busy: false };
    case "api_alive_model_unavailable":
      return { label: "Model unavailable", tone: "err", busy: false };
    case "api_alive_model_timeout":
      return { label: "Status check timed out", tone: "warn", busy: false };
    case "api_alive_capability_mismatch":
      return { label: "Model not capable", tone: "warn", busy: false };
    case "status_contract_error":
      return { label: "Status unverified", tone: "warn", busy: false };
  }
}

// ---- Answer presentation -------------------------------------------------------------------------

export type AnswerKind =
  | "interaction"
  | "insufficient"
  | "withheld"
  | "no_procedure"
  | "partial"
  | "hint"
  | "guided"
  | "concept_only"
  | "partial"
  | "grounded";

export interface AnswerView {
  readonly kind: AnswerKind;
  /** The verdict line. Empty for an assistant interaction, which carries no verdict. */
  readonly label: string;
  readonly tone: Tone;
  readonly showSources: boolean;
  /** Server warnings are shown with every course answer. A built-in interaction reply hides them:
   * its warning only says that no retrieval and no model call happened, which is developer
   * provenance, not something a student needs under a greeting. */
  readonly showLimitations: boolean;
}

export function isDemoAnswer(answer: CourseAnswer): boolean {
  return answer.warnings.some((warning) => warning.startsWith(DEMO_ANSWER_WARNING_PREFIX));
}

/** Decide how a validated answer is presented. `demoProfile` is the page-level demo signal (the
 * backend's own profile flag, or the demo warning on the answer itself). */
export function answerView(answer: CourseAnswer, demoProfile: boolean): AnswerView {
  if (answer.grounding_status === "assistant_interaction") {
    return { kind: "interaction", label: "", tone: "neutral", showSources: false, showLimitations: false };
  }
  if (answer.grounding_status === "insufficient_evidence") {
    return {
      kind: "insufficient",
      label: demoProfile
        ? "Not covered by this demo's knowledge base."
        : "Not grounded in approved course material — the approved course evidence was insufficient for this question.",
      tone: "warn",
      showSources: false,
      showLimitations: true,
    };
  }
  if (answer.grounding_status === "partially_grounded") {
    return {
      kind: "partial",
      label: demoProfile
        ? "Partially grounded in the retrieved experimental reference records shown below — not verified course material."
        : "Partially grounded in approved course material.",
      tone: "warn",
      showSources: true,
      showLimitations: true,
    };
  }
  // grounded
  if (demoProfile) {
    const withheld = answer.warnings.some((warning) =>
      warning.toLowerCase().includes(DEMO_WITHHELD_MARKER),
    );
    if (withheld) {
      return {
        kind: "withheld",
        label: "Answer withheld — the model's text carried a figure the retrieved records do not support.",
        tone: "err",
        showSources: true,
        showLimitations: true,
      };
    }
    const has = (marker: string) =>
      answer.warnings.some((warning) => warning.toLowerCase().includes(marker));
    if (has(DEMO_PARTIAL_STEPS_MARKER)) {
      // The server's own reply to a how-to about several topics when the lesson covers only some.
      return {
        kind: "partial",
        label: "Steps for some topics only — the lesson covers one topic asked about, not the others.",
        tone: "warn",
        showSources: true,
        showLimitations: true,
      };
    }
    if (has(DEMO_LESSON_HINT_MARKER)) {
      return {
        kind: "hint",
        label: "Hint from the lesson — its steps are withheld in this mode.",
        tone: "info",
        showSources: true,
        showLimitations: true,
      };
    }
    if (has(DEMO_GUIDED_STEP_MARKER)) {
      return {
        kind: "guided",
        label: has(DEMO_GUIDED_COMPLETE_MARKER)
          ? "Guided steps complete — every step of the lesson has been shown."
          : "Guided step — one step of the authorized lesson, quoted.",
        tone: "ok",
        showSources: true,
        showLimitations: true,
      };
    }
    const noProcedure = answer.warnings.some((warning) =>
      warning.toLowerCase().includes(DEMO_NO_PROCEDURE_MARKER),
    );
    if (noProcedure) {
      // The server answered a how-to itself because its records carry no steps: the reply is a
      // statement about the sources, not a procedure and not model text, and it says so.
      return {
        kind: "no_procedure",
        label:
          "No verified steps in the demo's sources — the records define this concept only, so no procedure was produced.",
        tone: "warn",
        showSources: true,
        showLimitations: true,
      };
    }
    const conceptOnly = answer.warnings.some((warning) =>
      warning.startsWith(DEMO_CONCEPT_ONLY_PREFIX),
    );
    if (conceptOnly) {
      return {
        kind: "concept_only",
        label: "Concept explanation only — the retrieved records define this concept and carry no operating steps.",
        tone: "info",
        showSources: true,
        showLimitations: true,
      };
    }
    return {
      kind: "grounded",
      label: "Grounded in the retrieved experimental reference records shown below — not verified course material.",
      tone: "ok",
      showSources: true,
      showLimitations: true,
    };
  }
  return {
    kind: "grounded",
    label: "Grounded in approved course material.",
    tone: "ok",
    showSources: true,
    showLimitations: true,
  };
}

/** Whether a guided-steps reply has a next step to ask for: it is a guided step, and neither the
 * lesson's last step nor the end of the lesson. The page then offers "Next step", which sends the
 * word the tutor asked for. */
export function nextStepAvailable(answer: CourseAnswer, demoProfile: boolean): boolean {
  if (answerView(answer, demoProfile).kind !== "guided") return false;
  // The explicit position decides when the API sent one; the warning markers only stand in for an
  // answer that carries none.
  if (answer.guided_progress) return !answer.guided_progress.ended;
  return !answer.warnings.some((warning) => {
    const lower = warning.toLowerCase();
    return lower.includes(DEMO_LAST_STEP_MARKER) || lower.includes(DEMO_GUIDED_COMPLETE_MARKER);
  });
}

// ---- Answer text ---------------------------------------------------------------------------------

/** The server appends this line to every demo answer, naming the records the model was shown. It
 * is kept verbatim and only styled as a footnote. */
export const SOURCE_LINE_PREFIX = "Course sources shown to the model:";
/** The same line on a reply the server composed itself without a model (a how-to whose records
 * carry no steps): the records were checked, not shown to a model, and the line says which. */
export const SOURCE_CHECK_LINE_PREFIX = "Course sources checked:";

export interface AnswerBlock {
  readonly text: string;
  readonly sourceLine: boolean;
}

/** Split answer text into paragraphs on blank lines. Line breaks inside a paragraph are preserved
 * by the stylesheet (pre-wrap); no markup is parsed and no structure is invented. */
export function splitAnswerText(text: string): AnswerBlock[] {
  return text
    .split(/\n[ \t]*\n/)
    .map((block) => block.trim())
    .filter((block) => block.length > 0)
    .map((block) => ({
      text: block,
      sourceLine:
        block.startsWith(SOURCE_LINE_PREFIX) || block.startsWith(SOURCE_CHECK_LINE_PREFIX),
    }));
}

// ---- Citations -----------------------------------------------------------------------------------

export const DOCUMENT_TYPE_LABEL: Record<DocumentType, string> = {
  lecture_slide: "Lecture slide",
  lecture_pdf: "Lecture PDF",
  transcript: "Transcript",
  homework: "Homework",
  instructor_note: "Instructor note",
};

export function describeLocator(locator: Locator): string {
  switch (locator.kind) {
    case "page":
      return `page ${locator.page_number}`;
    case "slide":
      return locator.lecture_number != null
        ? `lecture ${locator.lecture_number}, slide ${locator.slide_number}`
        : `slide ${locator.slide_number}`;
    case "timestamp":
      return `${locator.start_seconds}s – ${locator.end_seconds}s`;
    case "homework":
      return `homework item ${locator.item}`;
    case "section":
      return `section ${locator.section}`;
  }
}
