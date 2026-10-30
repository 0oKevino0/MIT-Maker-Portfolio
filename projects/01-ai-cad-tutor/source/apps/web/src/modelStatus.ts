// Model-status state machine (Batch D1B-02/05/06).
//
// One strict union distinguishes API reachability from model readiness; the
// display layer renders ONLY the fixed frontend text below — never a backend
// reason label, URL, model name, or exception. Classification uses machine
// codes exclusively (never message text), and the readiness UI is a courtesy
// layer: the backend capability gate remains the authority.

import { isAbortError } from "./api";
import { ApiHttpError, NetworkError, ResponseContractError, TimeoutError } from "./contracts";
import type { CourseAnswer } from "./contracts";
import { DEMO_MODEL_RAN_MARKERS } from "./demoStatus";
import { fetchLiveness, fetchReadiness } from "./health";
import type { HealthCheckOptions } from "./health";

export type ModelStatus =
  | "checking"
  | "api_unreachable"
  | "api_alive_model_not_configured"
  | "api_alive_model_unavailable"
  | "api_alive_model_timeout"
  | "api_alive_capability_mismatch"
  | "ready"
  | "status_contract_error";

/** Fixed, frontend-owned display text; nothing backend-provided is shown. */
export const MODEL_STATUS_TEXT: Record<
  ModelStatus,
  { api: string; model: string; course: string }
> = {
  checking: {
    api: "Checking…",
    model: "Checking…",
    course: "Course tutor status is being checked.",
  },
  api_unreachable: {
    api: "Offline — cannot reach the local API",
    model: "Unknown",
    course: "Course tutor unavailable — the local API cannot be reached.",
  },
  api_alive_model_not_configured: {
    api: "Online",
    model: "Not configured",
    course: "Course tutor unavailable — no local model is configured.",
  },
  api_alive_model_unavailable: {
    api: "Online",
    model: "Unavailable",
    course: "Course tutor unavailable — the model backend is not available.",
  },
  api_alive_model_timeout: {
    api: "Online",
    model: "Status check timed out",
    course: "Course tutor unavailable — the model status check timed out.",
  },
  api_alive_capability_mismatch: {
    api: "Online",
    model: "Not capable of course answering",
    course:
      "Course tutor unavailable — the configured model does not support course answering.",
  },
  ready: {
    api: "Online",
    model: "Ready",
    course: "Course tutor available.",
  },
  status_contract_error: {
    api: "Unknown",
    model: "Unknown",
    course: "Course tutor unavailable — the service status could not be verified.",
  },
};

/** Map a validated readiness reason label to a status. Unknown but
 * well-formed labels degrade to the generic not-ready state (D1B-06). */
export function statusFromNotReadyReason(reason: string): ModelStatus {
  switch (reason) {
    case "model_not_configured":
      return "api_alive_model_not_configured";
    case "capability_mismatch":
      return "api_alive_capability_mismatch";
    case "probe_timeout":
      return "api_alive_model_timeout";
    default:
      return "api_alive_model_unavailable";
  }
}

/** Map a Course runtime failure to a status update, or null when the error
 * says nothing about readiness (502/500/422 and client-side timeouts). */
export function statusFromCourseError(error: unknown): ModelStatus | null {
  if (error instanceof ApiHttpError) {
    switch (error.code) {
      case "model_not_configured":
        return "api_alive_model_not_configured";
      case "model_unavailable":
      case "backend_configuration_error":
        return "api_alive_model_unavailable";
      case "unsupported_capability":
        return "api_alive_capability_mismatch";
      case "model_timeout":
        return "api_alive_model_timeout";
      default:
        return null;
    }
  }
  if (error instanceof NetworkError) {
    return "api_unreachable";
  }
  return null;
}

/** The status update a SUCCESSFUL Course answer implies, or null to leave the status as it is.
 * A grounded answer the model generated proves the model just ran (the backend verifies its
 * identity before every answer), and only a model outage or timeout is the kind of status such an
 * answer disproves. Any other answer proves nothing — an insufficient-evidence or
 * assistant-interaction answer needs no model, and neither does a grounded reply the server
 * composed itself (a lesson hint, a guided step, a sources statement), which carries none of the
 * model-ran markers — and every other state is left to a readiness check. Never "checking": that
 * is the coordinator's. */
export function statusAfterCourseAnswer(
  current: ModelStatus,
  answer: CourseAnswer,
): ModelStatus | null {
  if (answer.grounding_status !== "grounded") return null;
  const modelRan = answer.warnings.some((warning) =>
    DEMO_MODEL_RAN_MARKERS.some((marker) => warning.toLowerCase().includes(marker)),
  );
  if (!modelRan) return null;
  return current === "api_alive_model_unavailable" || current === "api_alive_model_timeout"
    ? "ready"
    : null;
}

/** One full status check: liveness first, then readiness. An external abort
 * (supersession/unmount) is rethrown so the caller can ignore it. */
export async function performStatusCheck(
  options: HealthCheckOptions = {},
): Promise<ModelStatus> {
  try {
    await fetchLiveness(options);
  } catch (error) {
    if (error instanceof ResponseContractError) return "status_contract_error";
    // An HTTP RESPONSE (any status) proves the API answered — an unexpected
    // liveness status is a contract failure, never "API Offline"
    // (review 0036 D1B-IF-01). Only a request that could not complete at all
    // (network failure / no response before the bound) is unreachable.
    if (error instanceof ApiHttpError) return "status_contract_error";
    if (error instanceof NetworkError || error instanceof TimeoutError) {
      return "api_unreachable";
    }
    throw error; // external AbortError or an unexpected defect
  }

  try {
    const readiness = await fetchReadiness(options);
    if (readiness.status === "ready") return "ready";
    return statusFromNotReadyReason(readiness.reason);
  } catch (error) {
    if (error instanceof ResponseContractError) return "status_contract_error";
    if (error instanceof TimeoutError) return "api_alive_model_timeout";
    // Same phase rule (review 0036 D1B-IF-01): liveness just succeeded, so an
    // unexpected readiness HTTP status is a status-contract failure of a
    // reachable API — it must never present as API Offline.
    if (error instanceof ApiHttpError) return "status_contract_error";
    if (error instanceof NetworkError) return "api_unreachable";
    throw error;
  }
}

// ---------------------------------------------------------------------------
// App-lifetime status-check coordination (review 0036 D1B-IF-03)
// ---------------------------------------------------------------------------

type StatusListener = (status: ModelStatus) => void;

/** One coordinator per APP LIFETIME, created outside the StrictMode boundary
 * (see main.tsx) and injected into <App/>.
 *
 * StrictMode replays every effect as setup -> cleanup -> setup within one
 * synchronous flush. The coordinator survives that replay: `attach` re-joins
 * the still-running initial check instead of starting a second one, and
 * `detach` only schedules a DEFERRED cancellation that an immediate re-attach
 * clears — so the initial check runs exactly once (live x1, ready x1) while a
 * real unmount still aborts the in-flight request. Manual `recheck()` is
 * never deduplicated. There is no module-level flag: a new coordinator (a new
 * app start, or an injected test instance) always checks again.
 */
export class StatusCheckCoordinator {
  private listener: StatusListener | null = null;
  private controller: AbortController | null = null;
  private sequence = 0;
  private active = false;
  private initialStarted = false;
  private detachCancelTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly checkOptions: HealthCheckOptions = {}) {}

  /** Subscribe the UI. Starts the one automatic initial check on the first
   * attach of this coordinator's lifetime; later attaches (StrictMode replay)
   * re-join the in-flight or completed lifecycle without a new request. */
  attach(listener: StatusListener): void {
    if (this.detachCancelTimer !== null) {
      clearTimeout(this.detachCancelTimer);
      this.detachCancelTimer = null;
    }
    this.listener = listener;
    if (this.active) {
      listener("checking");
      return;
    }
    if (!this.initialStarted) {
      this.initialStarted = true;
      this.startCheck();
    }
  }

  /** Unsubscribe. Cancellation is deferred one macrotask so the synchronous
   * StrictMode cleanup->setup replay can re-attach without losing the single
   * initial check; a real unmount lets the deferred cancellation run. */
  detach(listener: StatusListener): void {
    if (this.listener !== listener) return;
    this.listener = null;
    this.detachCancelTimer = setTimeout(() => {
      this.detachCancelTimer = null;
      this.cancelActive();
    }, 0);
  }

  /** Manual recheck: ALWAYS starts a fresh check, superseding any active one. */
  recheck(): void {
    this.startCheck();
  }

  private cancelActive(): void {
    if (this.controller !== null) {
      this.controller.abort();
      this.controller = null;
    }
    this.active = false;
    this.sequence += 1; // invalidate any late result of the cancelled check
  }

  private startCheck(): void {
    this.cancelActive();
    const controller = new AbortController();
    this.controller = controller;
    this.active = true;
    const sequence = ++this.sequence;
    this.deliver("checking");
    void (async () => {
      let status: ModelStatus;
      try {
        status = await performStatusCheck({ ...this.checkOptions, signal: controller.signal });
      } catch (error) {
        if (this.sequence !== sequence) return; // superseded
        this.active = false;
        this.controller = null;
        if (isAbortError(error)) return; // cancelled (unmount)
        this.deliver("status_contract_error");
        return;
      }
      if (this.sequence !== sequence) return; // stale result: never delivered
      this.active = false;
      this.controller = null;
      this.deliver(status);
    })();
  }

  private deliver(status: ModelStatus): void {
    this.listener?.(status);
  }
}
