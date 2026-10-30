// Single frontend Course contract boundary (Batch C2).
//
// The JSON Schemas in packages/contracts remain the ONLY source of truth:
// both validators below are compiled with Ajv (Draft 2020-12 dialect)
// directly from the shared schema files, so the frontend cannot drift from
// the backend contract. External input is `unknown` until it passes one of
// these validators; no Course value may enter React state any other way.
//
// Validation failures carry an internal `detail` (schema path and rule only —
// never payload content) for tests and debugging, while `message` is always
// a fixed, user-safe string.

import Ajv2020 from "ajv/dist/2020";
import type { ErrorObject, ValidateFunction } from "ajv";
import courseQuestionSchema from "../../../packages/contracts/course_question.schema.json";
import courseAnswerSchema from "../../../packages/contracts/course_answer.schema.json";

// ---- Canonical frontend Course types (one definition, no per-file copies) ---

/** Modes the student UI may offer (the selector never shows instructor). */
export type StudentAnswerMode = "hint_1" | "hint_2" | "guided_steps" | "full_explanation";

/** The COMPLETE shared-schema enum (review 0030 C2-IF-05): request and
 * response contracts both accept every value here, including instructor. */
export type CourseAnswerMode = StudentAnswerMode | "instructor";

/** Backwards-compatible alias for the student subset. */
export type AnswerMode = StudentAnswerMode;

export type GroundingStatus =
  | "grounded"
  | "partially_grounded"
  | "insufficient_evidence"
  /** A built-in assistant reply to a greeting or a control message: no retrieval, no model call,
   * no course evidence claimed. The shared schema binds it to carrying no citations. */
  | "assistant_interaction";

export type DocumentType =
  | "lecture_slide"
  | "lecture_pdf"
  | "transcript"
  | "homework"
  | "instructor_note";

export type Locator =
  | { kind: "page"; page_number: number }
  | { kind: "slide"; slide_number: number; lecture_number?: number | null }
  | { kind: "timestamp"; start_seconds: number; end_seconds: number }
  | { kind: "homework"; item: string }
  | { kind: "section"; section: string };

export interface Citation {
  citation_id: string;
  document_id: string;
  document_version: string;
  chunk_id: string;
  title: string;
  document_type: DocumentType;
  locator: Locator;
  snippet?: string;
}

/** The explicit Guided Steps position: which lesson, which record it is bound to, the last step
 * shown, the step count and whether the lesson ended. The page keeps the latest one the API
 * returned and sends it back with the next question; the API validates it against the authorized
 * record and ignores anything that does not match. It is the student's own indication of
 * progress, never a verified SOLIDWORKS state. */
export interface GuidedProgress {
  topic: string;
  record_id: string;
  step: number;
  total: number;
  ended: boolean;
}

export interface CourseAnswer {
  schema_version: "1.0.0";
  request_id: string;
  answer_id: string;
  answer_mode: CourseAnswerMode;
  grounding_status: GroundingStatus;
  answer_text: string;
  citations: Citation[];
  warnings: string[];
  guided_progress?: GuidedProgress | null;
}

/** One bounded turn of the active conversation. The browser may only send these two roles. */
export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}

/** The wire bounds, mirrored from the shared schema so the client trims before it sends. */
export const MAX_HISTORY_TURNS = 6;
export const MAX_HISTORY_TURN_CHARACTERS = 2000;
export const MAX_HISTORY_TOTAL_CHARACTERS = 6000;

export interface CourseQuestion {
  schema_version: "1.0.0";
  request_id: string;
  course_id: string;
  question: string;
  history?: ConversationTurn[];
  // The FULL schema union — the Ajv ValidateFunction generic must match what
  // the shared schema actually accepts (review 0030 C2-IF-05).
  answer_mode: CourseAnswerMode;
  locale?: string | null;
  guided_progress?: GuidedProgress | null;
}

export interface ApiErrorBody {
  request_id: string;
  code: string;
  message: string;
  retryable: boolean;
  details: { field_path: string; reason: string }[];
}

// ---- Typed error contract (C2-04) -------------------------------------------
// One base class keeps the existing `ApiRequestError` import surface; the
// subclasses distinguish request-contract, response-contract, HTTP, network,
// and timeout failures. Every `message` is fixed and user-safe.

export class ApiRequestError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly code?: string,
  ) {
    super(message);
    this.name = "ApiRequestError";
  }
}

/** The outgoing payload failed the shared CourseQuestion schema; not sent. */
export class RequestContractError extends ApiRequestError {
  constructor(public readonly detail: string) {
    super("The question could not be submitted because it failed the local contract check.");
    this.name = "RequestContractError";
  }
}

/** A 200 response failed the shared CourseAnswer schema; nothing is shown.
 * `message` must be one of the fixed safe strings (default: contract). */
export class ResponseContractError extends ApiRequestError {
  constructor(
    public readonly detail: string,
    message = "The API response did not match the expected contract.",
  ) {
    super(message);
    this.name = "ResponseContractError";
  }
}

/** The API answered with a non-2xx status; message is fixed per status. */
export class ApiHttpError extends ApiRequestError {
  constructor(message: string, status: number, code?: string) {
    super(message, status, code);
    this.name = "ApiHttpError";
  }
}

/** The request never reached the API (connection refused, DNS, CORS...). */
export class NetworkError extends ApiRequestError {
  constructor(message: string) {
    super(message);
    this.name = "NetworkError";
  }
}

/** The request was aborted by the local timeout. */
export class TimeoutError extends ApiRequestError {
  constructor(message: string) {
    super(message);
    this.name = "TimeoutError";
  }
}

// ---- Shared-schema validators (C2-01 / C2-02) --------------------------------

// strict:false — ajv's "strict mode" is a lint over schema STYLE (e.g. it
// wants a redundant `type: array` next to `maxItems` inside if/then). The
// shared schemas are valid Draft 2020-12 and are already structurally checked
// by the backend's Draft202012Validator; the frontend must consume them
// verbatim rather than fork them to satisfy a linter.
const ajv = new Ajv2020({ strict: false, allErrors: false });

const validateQuestionSchema: ValidateFunction<CourseQuestion> =
  ajv.compile<CourseQuestion>(courseQuestionSchema);
const validateAnswerSchema: ValidateFunction<CourseAnswer> =
  ajv.compile<CourseAnswer>(courseAnswerSchema);

function schemaDetail(errors: ErrorObject[] | null | undefined): string {
  // Path and rule only — never instance data.
  const first = errors?.[0];
  if (!first) return "schema validation failed";
  return `${first.instancePath || "/"} (${first.keyword})`;
}

// ---- Bounded iterative JSON-shape guard (review 0030 C2-IF-01) ---------------
// JSON.parse cannot produce NaN/Infinity, but the JSON Schema `number` type
// also cannot REJECT them when a value arrives from another source (tests,
// future non-JSON transports). This mirrors the backend's finite guard while
// staying safe against adversarial input: the walk is ITERATIVE (explicit
// stack, no recursion), bounded in depth and total value count, and rejects
// any repeated object reference (a cycle, or sharing — impossible in parsed
// JSON, so failing closed loses nothing). Limit violations are reported as a
// short rule code only; no value, path, or size detail is retained.

/** Maximum container nesting depth accepted from any external JSON value.
 * Legitimate Course payloads nest fewer than 8 levels; 64 leaves an order of
 * magnitude of headroom while staying far below engine stack limits. */
const MAX_JSON_DEPTH = 64;

/** Maximum total number of traversed values (containers and primitives).
 * The largest legitimate CourseAnswer stays under ~1,000 values; 10,000
 * bounds traversal work with generous headroom. */
const MAX_JSON_NODES = 10_000;

export type JsonShapeViolation =
  | "finite-number"
  | "max-depth"
  | "max-nodes"
  | "duplicate-reference";

/** Exported for the health boundary (D1B): same bounded walk, same limits. */
export function findJsonShapeViolation(root: unknown): JsonShapeViolation | null {
  const stack: { value: unknown; depth: number }[] = [{ value: root, depth: 0 }];
  const seenContainers = new WeakSet<object>();
  let visited = 0;

  while (stack.length > 0) {
    const entry = stack.pop();
    if (entry === undefined) break;
    const { value, depth } = entry;

    visited += 1;
    if (visited > MAX_JSON_NODES) return "max-nodes";

    if (typeof value === "number") {
      if (!Number.isFinite(value)) return "finite-number";
      continue;
    }
    if (typeof value !== "object" || value === null) continue;

    if (seenContainers.has(value)) return "duplicate-reference";
    seenContainers.add(value);
    if (depth >= MAX_JSON_DEPTH) return "max-depth";

    const children = Array.isArray(value) ? value : Object.values(value);
    for (const child of children) {
      stack.push({ value: child, depth: depth + 1 });
    }
  }
  return null;
}

// ---- Display invariants (C2-03) ----------------------------------------------
// Server-side C1 validation guarantees marker/citation consistency; the
// frontend re-checks the display-critical part before anything renders:
// every validated citation must appear as its canonical `[citation_id]`
// marker in answer_text, in first-occurrence order matching the citation
// list. Metadata is never inferred from the text — this check only decides
// whether the response is displayable at all.

function markerOrderMatches(answerText: string, citations: Citation[]): boolean {
  let previousIndex = -1;
  const seen = new Set<string>();
  for (const citation of citations) {
    if (seen.has(citation.citation_id)) return false;
    seen.add(citation.citation_id);
    const index = answerText.indexOf(`[${citation.citation_id}]`);
    if (index < 0 || index <= previousIndex) return false;
    previousIndex = index;
  }
  return true;
}

// Cross-field rule the JSON Schema cannot express but the backend enforces
// (TimestampLocator.end_not_before_start): a valid backend response can never
// carry a reversed timestamp range, so the frontend rejects it too.
function timestampOrdersValid(citations: Citation[]): boolean {
  return citations.every(
    (citation) =>
      citation.locator.kind !== "timestamp" ||
      citation.locator.end_seconds >= citation.locator.start_seconds,
  );
}

// ---- Public validation entries ------------------------------------------------

/** Validate an outgoing payload against the shared CourseQuestion schema.
 * Throws RequestContractError; only a passing payload may be fetched. */
export function validateCourseQuestion(payload: unknown): CourseQuestion {
  const shapeViolation = findJsonShapeViolation(payload);
  if (shapeViolation !== null) {
    throw new RequestContractError(`/ (${shapeViolation})`);
  }
  if (!validateQuestionSchema(payload)) {
    throw new RequestContractError(schemaDetail(validateQuestionSchema.errors));
  }
  return payload;
}

/** Validate a decoded 200 body against the shared CourseAnswer schema plus
 * the frontend display invariants. Throws ResponseContractError; only the
 * returned value may enter React state. */
export function validateCourseAnswer(data: unknown): CourseAnswer {
  const shapeViolation = findJsonShapeViolation(data);
  if (shapeViolation !== null) {
    throw new ResponseContractError(`/ (${shapeViolation})`);
  }
  if (!validateAnswerSchema(data)) {
    throw new ResponseContractError(schemaDetail(validateAnswerSchema.errors));
  }
  if (!markerOrderMatches(data.answer_text, data.citations)) {
    throw new ResponseContractError("/citations (marker-consistency)");
  }
  if (!timestampOrdersValid(data.citations)) {
    throw new ResponseContractError("/citations (timestamp-order)");
  }
  return data;
}
