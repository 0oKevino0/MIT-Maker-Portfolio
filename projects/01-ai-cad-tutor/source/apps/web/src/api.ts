// Typed client for the course/ask endpoint (Batch C2).
//
// packages/contracts remains the single source of truth: the outgoing
// CourseQuestion payload is validated against the shared schema BEFORE any
// fetch, and every 200 body is validated against the shared CourseAnswer
// schema (plus display invariants) before it may enter React state — both
// through the one boundary in ./contracts. Every failure maps to a typed
// error with a fixed, user-safe message.

import {
  ApiHttpError,
  ApiRequestError,
  MAX_HISTORY_TOTAL_CHARACTERS,
  MAX_HISTORY_TURNS,
  MAX_HISTORY_TURN_CHARACTERS,
  NetworkError,
  ResponseContractError,
  TimeoutError,
  validateCourseAnswer,
  validateCourseQuestion,
} from "./contracts";
import type {
  ConversationTurn,
  CourseAnswer,
  CourseAnswerMode,
  GuidedProgress,
  StudentAnswerMode,
} from "./contracts";

export {
  ApiHttpError,
  ApiRequestError,
  MAX_HISTORY_TOTAL_CHARACTERS,
  MAX_HISTORY_TURNS,
  MAX_HISTORY_TURN_CHARACTERS,
  NetworkError,
  RequestContractError,
  ResponseContractError,
  TimeoutError,
  validateCourseAnswer,
  validateCourseQuestion,
} from "./contracts";
export type {
  AnswerMode,
  ApiErrorBody,
  Citation,
  ConversationTurn,
  CourseAnswer,
  CourseAnswerMode,
  CourseQuestion,
  DocumentType,
  GroundingStatus,
  GuidedProgress,
  Locator,
  StudentAnswerMode,
} from "./contracts";

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

// Fixed for the MVP; the instructor-approved course collection is selected server-side.
export const COURSE_ID = "cswa-preparation";

export const DEFAULT_TIMEOUT_MS = 60_000;

const LOCAL_TOKEN_STORAGE_KEY = "cswa-local-token";

export const ANSWER_MODES: { value: StudentAnswerMode; label: string }[] = [
  { value: "hint_1", label: "Hint 1 (gentle nudge)" },
  { value: "hint_2", label: "Hint 2 (stronger hint)" },
  { value: "guided_steps", label: "Guided steps" },
  { value: "full_explanation", label: "Full explanation" },
];

// ---- Local token -----------------------------------------------------------
// The token is never hard-coded. Sources, in order:
//   1. VITE_CSWA_LOCAL_TOKEN from an ignored local .env file — development
//      convenience only; a production bundle must be built without it so the
//      artifact contains no real token.
//   2. localStorage "cswa-local-token", set once per browser by the operator:
//        localStorage.setItem("cswa-local-token", "<token>")
export function getLocalToken(): string | null {
  const fromEnv = import.meta.env.VITE_CSWA_LOCAL_TOKEN;
  if (typeof fromEnv === "string" && fromEnv.trim()) {
    return fromEnv.trim();
  }
  try {
    const stored = window.localStorage.getItem(LOCAL_TOKEN_STORAGE_KEY);
    return stored && stored.trim() ? stored.trim() : null;
  } catch {
    return null;
  }
}

// ---- Base URL safety --------------------------------------------------------
// The tutor is a local application: only loopback API addresses are accepted
// unless a remote address is explicitly allowed at build time.
const LOOPBACK_HOSTNAMES = new Set(["127.0.0.1", "localhost", "[::1]", "::1"]);

export function assertSafeApiBaseUrl(baseUrl: string): void {
  let parsed: URL;
  try {
    parsed = new URL(baseUrl);
  } catch {
    throw new ApiRequestError(`The configured API base URL is not a valid URL.`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new ApiRequestError("The API base URL must use http or https.");
  }
  // Review 0030 C2-IF-03: credentials or a fragment in the configured URL are
  // rejected outright (fixed messages — the URL itself is never echoed).
  if (parsed.username !== "" || parsed.password !== "") {
    throw new ApiRequestError("The API base URL must not contain credentials.");
  }
  if (parsed.hash !== "") {
    throw new ApiRequestError("The API base URL must not contain a fragment.");
  }
  if (
    !LOOPBACK_HOSTNAMES.has(parsed.hostname) &&
    import.meta.env.VITE_ALLOW_REMOTE_API !== "true"
  ) {
    throw new ApiRequestError(
      "Refusing a non-local API base URL; set VITE_ALLOW_REMOTE_API=true explicitly " +
        "if a remote deployment is intended.",
    );
  }
}

// ---- Error mapping -----------------------------------------------------------
// Every user-facing message is FIXED per status. The response body's message
// is never reflected (a broken or hostile upstream could put anything there);
// only its machine `code` is preserved for programmatic handling.

function messageForStatus(status: number): string {
  switch (status) {
    case 400:
      return "The request could not be parsed by the API.";
    case 401:
      return (
        "Authentication failed: the local API token is missing or invalid. " +
        "Configure it once via localStorage key 'cswa-local-token'."
      );
    case 403:
      return "The request was rejected by the local security policy.";
    case 413:
      return "The question is too large for the configured request limit.";
    case 422:
      return "The request did not match the expected contract. Check the question input.";
    case 500:
      return "The server reported an internal error. Try again.";
    case 502:
      return "The local model produced invalid output. Try again.";
    case 503:
      return "The local model backend is unavailable. Is it running?";
    case 504:
      return "The local model took too long to answer. Try again.";
    default:
      return `The API returned an unexpected error (status ${status}).`;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Extract only the machine-readable error code; never free-form text. */
function errorCodeFrom(payload: unknown): string | undefined {
  if (!isRecord(payload)) return undefined;
  return typeof payload.code === "string" ? payload.code : undefined;
}

function httpError(status: number, code: string | undefined): ApiHttpError {
  if (status === 403 && code === "instructor_mode_disabled") {
    return new ApiHttpError("Instructor mode is not enabled on this installation.", status, code);
  }
  return new ApiHttpError(messageForStatus(status), status, code);
}

// ---- Request ------------------------------------------------------------------

export interface AskOptions {
  /** The active conversation, oldest first. Trimmed to the wire bounds before it is sent. */
  history?: ConversationTurn[];
  /** The explicit Guided Steps position the API last returned, sent back as it was. */
  guidedProgress?: GuidedProgress | null;
  baseUrl?: string;
  timeoutMs?: number;
  token?: string | null;
  /** Deterministic request UUID for tests; when STRICTLY undefined the
   * production default crypto.randomUUID() is used. Typed `unknown` so tests
   * can probe the runtime contract (null, numbers, control characters)
   * without `any`; every value is validated by the shared schema. */
  requestId?: unknown;
}

/** Shared AbortError detection (also used by the health clients, D1B). */
export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

/** Trim a transcript to exactly what the shared contract accepts. */
export function boundHistoryForWire(history?: ConversationTurn[]): ConversationTurn[] {
  if (!history || history.length === 0) {
    return [];
  }
  const recent = history
    .filter((turn) => (turn.role === "user" || turn.role === "assistant") && turn.content.trim())
    .slice(-MAX_HISTORY_TURNS)
    .map((turn) => ({ role: turn.role, content: turn.content.slice(0, MAX_HISTORY_TURN_CHARACTERS) }));
  const kept: ConversationTurn[] = [];
  let total = 0;
  for (let index = recent.length - 1; index >= 0; index -= 1) {
    const turn = recent[index];
    if (total + turn.content.length > MAX_HISTORY_TOTAL_CHARACTERS) {
      break;
    }
    total += turn.content.length;
    kept.unshift(turn);
  }
  return kept;
}

export async function askCourseQuestion(
  question: string,
  answerMode: CourseAnswerMode,
  options: AskOptions = {},
): Promise<CourseAnswer> {
  const baseUrl = options.baseUrl ?? API_BASE_URL;
  assertSafeApiBaseUrl(baseUrl);

  // Review 0030 C2-IF-02: only STRICT undefined selects the generated UUID.
  // An explicit null (or any other injected value) must reach the contract
  // and be rejected there — never silently repaired.
  const requestId: unknown =
    options.requestId !== undefined ? options.requestId : crypto.randomUUID();

  // Preflight (C2-02): only a payload that passes the shared CourseQuestion
  // schema may be serialized and sent. RequestContractError means 0 fetches.
  // The conversation is trimmed HERE, before validation, so the request can never be rejected for
  // a bound the client could have respected: newest turns kept, oldest dropped, each turn capped,
  // and the total capped. The shared schema enforces the same bounds on the server.
  const history = boundHistoryForWire(options.history);
  const payload = validateCourseQuestion({
    schema_version: "1.0.0",
    request_id: requestId,
    course_id: COURSE_ID,
    question,
    answer_mode: answerMode,
    ...(history.length > 0 ? { history } : {}),
    ...(options.guidedProgress ? { guided_progress: options.guidedProgress } : {}),
  });

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const token = options.token !== undefined ? options.token : getLocalToken();
  if (token) {
    headers["X-CSWA-Local-Token"] = token;
  }

  // Review 0030 C2-IF-04: ONE abort lifecycle covers the fetch wait, the
  // header response, and BOTH body reads. The timer is cleared only in the
  // outer finally, after the whole response has been consumed (or failed).
  const controller = new AbortController();
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const fixedTimeoutMessage = `The request timed out after ${Math.round(timeoutMs / 1000)}s.`;

  try {
    let response: Response;
    try {
      response = await fetch(`${baseUrl}/api/v1/course/ask`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
    } catch (error) {
      if (isAbortError(error)) {
        throw new TimeoutError(fixedTimeoutMessage);
      }
      // Review 0030 C2-IF-03: fixed, URL-free message — the configured URL
      // and the underlying fetch error text are never reflected.
      throw new NetworkError("Cannot reach the local tutor API. Is the backend running?");
    }

    if (!response.ok) {
      let code: string | undefined;
      try {
        code = errorCodeFrom(await response.json());
      } catch (error) {
        if (isAbortError(error)) {
          // A stalled ERROR body is a timeout, not an HTTP classification.
          throw new TimeoutError(fixedTimeoutMessage);
        }
        // Non-JSON error body (e.g. an HTML gateway page): status-only mapping.
      }
      throw httpError(response.status, code);
    }

    let parsed: unknown;
    try {
      parsed = await response.json();
    } catch (error) {
      if (isAbortError(error)) {
        // A stalled SUCCESS body is a timeout, never a NetworkError.
        throw new TimeoutError(fixedTimeoutMessage);
      }
      throw new ResponseContractError(
        "/ (non-json-body)",
        "The API returned a non-JSON response.",
      );
    }
    // C2-01: the shared-schema boundary; only its return value reaches callers.
    return validateCourseAnswer(parsed);
  } finally {
    clearTimeout(timer);
  }
}
