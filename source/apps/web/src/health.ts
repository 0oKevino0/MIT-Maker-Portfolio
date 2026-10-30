// Frontend health runtime contract (Batch D1B-01).
//
// The single validated entry for the backend's /health/live and /health/ready
// responses. Every body is `unknown` until it passes: HTTP status check, safe
// JSON parse, the shared bounded-JSON shape guard, exact-shape validation,
// and HTTP<->body consistency. Only validated values reach the status state
// machine; failures map to the existing fixed C2 error types. No response
// field beyond the strict shape below exists on the backend, and none is
// invented here (fixture parity is asserted in health.test.ts).

import { API_BASE_URL, assertSafeApiBaseUrl, isAbortError } from "./api";
import {
  ApiHttpError,
  findJsonShapeViolation,
  NetworkError,
  ResponseContractError,
  TimeoutError,
} from "./contracts";

export const HEALTH_CHECK_TIMEOUT_MS = 10_000;

/** Maximum accepted health body size. Real health bodies are under 200
 * bytes; 64 KiB bounds a hostile/streaming body long before memory matters
 * (review 0040 D1B-IF-01A). */
export const MAX_HEALTH_BODY_BYTES = 64 * 1024;

export interface HealthCheckOptions {
  baseUrl?: string;
  timeoutMs?: number;
  /** External cancellation (recheck supersession / unmount). An abort from
   * here is rethrown as the original AbortError, never as a TimeoutError. */
  signal?: AbortSignal;
}

export interface LivenessResponse {
  status: "alive";
  service: string;
  version: string;
}

export interface ReadyResponse {
  status: "ready";
}

export interface NotReadyResponse {
  status: "not_ready";
  /** Stable machine label (validated shape; semantics mapped by the state
   * machine — unknown labels degrade to a generic not-ready state). */
  reason: string;
}

export type ReadinessResponse = ReadyResponse | NotReadyResponse;

const NETWORK_MESSAGE = "Cannot reach the local tutor API. Is the backend running?";
const CONTRACT_MESSAGE = "The service status response did not match the expected contract.";

// A machine label: short lowercase snake_case ASCII, nothing renderable as
// markup and nothing resembling free text, a URL, or a path.
const REASON_LABEL = /^[a-z][a-z0-9_]{0,63}$/;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function guardShape(data: unknown): void {
  const violation = findJsonShapeViolation(data);
  if (violation !== null) {
    throw new ResponseContractError(`/ (${violation})`, CONTRACT_MESSAGE);
  }
}

export function validateLivenessBody(data: unknown): LivenessResponse {
  guardShape(data);
  if (!isRecord(data)) throw new ResponseContractError("/ (type)", CONTRACT_MESSAGE);
  const keys = Object.keys(data).sort();
  if (keys.join(",") !== "service,status,version") {
    throw new ResponseContractError("/ (keys)", CONTRACT_MESSAGE);
  }
  if (data.status !== "alive") throw new ResponseContractError("/status (const)", CONTRACT_MESSAGE);
  if (typeof data.service !== "string" || data.service.length === 0) {
    throw new ResponseContractError("/service (type)", CONTRACT_MESSAGE);
  }
  if (typeof data.version !== "string" || data.version.length === 0) {
    throw new ResponseContractError("/version (type)", CONTRACT_MESSAGE);
  }
  return { status: "alive", service: data.service, version: data.version };
}

export function validateReadinessBody(data: unknown, httpStatus: number): ReadinessResponse {
  guardShape(data);
  if (!isRecord(data)) throw new ResponseContractError("/ (type)", CONTRACT_MESSAGE);
  if (httpStatus === 200) {
    if (Object.keys(data).join(",") !== "status" || data.status !== "ready") {
      throw new ResponseContractError("/ (ready-shape)", CONTRACT_MESSAGE);
    }
    return { status: "ready" };
  }
  if (httpStatus === 503) {
    const keys = Object.keys(data).sort();
    if (keys.join(",") !== "reason,status" || data.status !== "not_ready") {
      throw new ResponseContractError("/ (not-ready-shape)", CONTRACT_MESSAGE);
    }
    if (typeof data.reason !== "string" || !REASON_LABEL.test(data.reason)) {
      throw new ResponseContractError("/reason (label)", CONTRACT_MESSAGE);
    }
    return { status: "not_ready", reason: data.reason };
  }
  throw new ResponseContractError("/ (http-status)", CONTRACT_MESSAGE);
}

interface RawHealthResult {
  httpStatus: number;
  body: unknown;
}

/** Race ANY promise against the abort signal (review 0040 D1B-IF-01B).
 *
 * Even a hostile promise that never settles and ignores AbortSignal cannot
 * hang the application-level operation: an abort settles this wrapper. A
 * LATE settlement of the abandoned promise is absorbed silently — it can
 * never resolve this wrapper, trigger a next phase, or surface as an
 * unhandled rejection. The abort listener is removed whenever the race
 * settles through the promise side. */
function awaitBounded<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) {
    void promise.then(
      () => undefined,
      () => undefined,
    );
    return Promise.reject(new DOMException("aborted", "AbortError"));
  }
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const onAbort = () => {
      if (settled) return;
      settled = true;
      // Abandon the underlying promise; absorb its eventual settlement.
      void promise.then(
        () => undefined,
        () => undefined,
      );
      reject(new DOMException("aborted", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error: unknown) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener("abort", onAbort);
        reject(error as Error);
      },
    );
  });
}

/** Read and parse a health response body through an OWNED reader (review
 * 0040 D1B-IF-01A). The client never calls response.json()/text(): it takes
 * the body reader itself, bounds every chunk read, caps total bytes, and on
 * EVERY exit path (success, parse failure, contract failure, timeout,
 * external abort) cancels the actual read and releases the lock, so no
 * pending native body task survives the bounded operation. */
async function readBodyBounded(response: Response, signal: AbortSignal): Promise<unknown> {
  const body: ReadableStream<Uint8Array> | null | undefined = response.body;
  if (body === null || body === undefined || typeof body.getReader !== "function") {
    // No readable stream: fail closed — never fall back to response.json().
    throw new ResponseContractError("/ (body-unreadable)", CONTRACT_MESSAGE);
  }
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  try {
    for (;;) {
      const result = await awaitBounded(reader.read(), signal);
      if (result.done) break;
      const chunk = result.value;
      if (chunk !== undefined) {
        totalBytes += chunk.byteLength;
        if (totalBytes > MAX_HEALTH_BODY_BYTES) {
          // The oversized payload is not retained; cleanup cancels the rest.
          throw new ResponseContractError("/ (body-too-large)", CONTRACT_MESSAGE);
        }
        chunks.push(chunk);
      }
    }
  } finally {
    // Unified cleanup: terminate the ACTUAL read. cancel() is itself bounded
    // (a hostile cancel cannot hang the finally) and a cancel rejection is
    // absorbed without replacing the primary error; the lock is always
    // released so response.body.locked returns to false.
    try {
      await awaitBounded(reader.cancel(), signal);
    } catch {
      // aborted or rejected cancel: absorbed by design
    }
    try {
      reader.releaseLock();
    } catch {
      // already released
    }
  }
  const merged = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const text = new TextDecoder().decode(merged);
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new ResponseContractError("/ (non-json-body)", CONTRACT_MESSAGE);
  }
}

async function fetchHealthPath(
  path: string,
  options: HealthCheckOptions,
  acceptedStatuses: readonly number[],
): Promise<RawHealthResult> {
  const baseUrl = options.baseUrl ?? API_BASE_URL;
  assertSafeApiBaseUrl(baseUrl);

  const controller = new AbortController();
  let timedOut = false;
  const timeoutMs = options.timeoutMs ?? HEALTH_CHECK_TIMEOUT_MS;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const externalAbort = () => controller.abort();
  options.signal?.addEventListener("abort", externalAbort, { once: true });
  if (options.signal?.aborted) controller.abort();

  const fixedTimeoutMessage = `The status check timed out after ${Math.round(timeoutMs / 1000)}s.`;

  try {
    let response: Response;
    try {
      // Promise-level bound (review 0040 D1B-IF-01B): the operation settles
      // at the abort even if this fetch implementation ignores its signal
      // and stays pending forever; a late settlement is absorbed.
      response = await awaitBounded(
        fetch(`${baseUrl}${path}`, {
          method: "GET",
          signal: controller.signal,
        }),
        controller.signal,
      );
    } catch (error) {
      if (isAbortError(error)) {
        if (timedOut) throw new TimeoutError(fixedTimeoutMessage);
        throw error; // external cancellation: rethrown, caller ignores it
      }
      throw new NetworkError(NETWORK_MESSAGE);
    }
    // From this point an HTTP Response has been RECEIVED: the API answered
    // (review 0038 D1B-IF-01). Any internal timeout while consuming the body
    // is therefore a status-contract failure of a reachable API — it is
    // converted to the fixed ResponseContractError and must never surface as
    // a TimeoutError that the liveness phase would label "API Offline".
    if (!acceptedStatuses.includes(response.status)) {
      throw new ApiHttpError(
        `The status endpoint returned an unexpected response (status ${response.status}).`,
        response.status,
      );
    }
    let body: unknown;
    try {
      body = await readBodyBounded(response, controller.signal);
    } catch (error) {
      if (isAbortError(error)) {
        if (timedOut) {
          throw new ResponseContractError("/ (body-timeout)", CONTRACT_MESSAGE);
        }
        throw error; // external cancellation: rethrown, caller ignores it
      }
      if (error instanceof ResponseContractError) {
        throw error; // fixed unreadable/too-large/non-json classifications
      }
      // A native stream read failure: fixed error, no stream detail retained.
      throw new ResponseContractError("/ (non-json-body)", CONTRACT_MESSAGE);
    }
    return { httpStatus: response.status, body };
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", externalAbort);
  }
}

export async function fetchLiveness(options: HealthCheckOptions = {}): Promise<LivenessResponse> {
  const { body } = await fetchHealthPath("/health/live", options, [200]);
  return validateLivenessBody(body);
}

export async function fetchReadiness(
  options: HealthCheckOptions = {},
): Promise<ReadinessResponse> {
  const { httpStatus, body } = await fetchHealthPath("/health/ready", options, [200, 503]);
  return validateReadinessBody(body, httpStatus);
}
