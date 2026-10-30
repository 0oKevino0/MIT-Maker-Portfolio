import { useCallback, useEffect, useRef, useState } from "react";
import {
  AnswerMode,
  ApiRequestError,
  askCourseQuestion,
  ConversationTurn,
  CourseAnswer,
  GuidedProgress,
} from "./api";
import {
  MODEL_STATUS_TEXT,
  ModelStatus,
  StatusCheckCoordinator,
  statusAfterCourseAnswer,
  statusFromCourseError,
} from "./modelStatus";
import {
  DemoStatus,
  INTERNAL_DEMO_NOTICE,
  fetchDemoStatus,
  internalDemoUiEnabled,
} from "./demoStatus";
import { isDemoAnswer } from "./presentation";
import { AppHeader } from "./components/AppHeader";
import { Composer } from "./components/Composer";
import { Conversation } from "./components/Conversation";
import { EmptyState } from "./components/EmptyState";
import { SystemDetails } from "./components/SystemDetails";

export interface AppProps {
  /** App-lifetime status coordinator, created OUTSIDE the StrictMode boundary
   * (see main.tsx) so the automatic initial check survives StrictMode's
   * effect replay exactly once. Tests may inject a fresh instance. */
  statusCoordinator?: StatusCheckCoordinator;
}

export default function App({ statusCoordinator }: AppProps = {}) {
  const [question, setQuestion] = useState("");
  const [answerMode, setAnswerMode] = useState<AnswerMode>("hint_1");
  const [answer, setAnswer] = useState<CourseAnswer | null>(null);
  // The active tutoring conversation, in memory only: no account, no database, and a page refresh
  // starts a new one. It exists so a follow-up such as "how do I actually do it?" stays on topic.
  const [transcript, setTranscript] = useState<ConversationTurn[]>([]);
  // The explicit Guided Steps position the API last returned: sent back with every question so a
  // lesson resumes where it was, including after the bounded history has dropped its start. It
  // is cleared with the conversation (a readiness check, review 0036 D1B-IF-02) and mirrors the
  // API's answer exactly; the page never computes a position of its own.
  const [guidedProgress, setGuidedProgress] = useState<GuidedProgress | null>(null);
  // The question of the attempt in flight (or the one that just failed), shown in the flow before
  // it is committed to the transcript. A failed turn is never replayed as history.
  const [currentQuestion, setCurrentQuestion] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Whether the error was a readiness-class failure, which is when a re-check is a useful offer.
  const [errorRecheckable, setErrorRecheckable] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus>("checking");
  // Verified runtime facts for the internal demo. Null until /health is read, and null forever on
  // a backend that reports none; the UI then shows no identity rather than a guess.
  const [demo, setDemo] = useState<DemoStatus | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  // Bumped whenever a readiness check starts. A reply that comes back from a request begun under an
  // older generation is discarded: the check invalidated everything on the page, and a late answer
  // must not reappear with its citations after that (nor its error re-synchronise the status the
  // newer check is measuring).
  const generationRef = useRef(0);

  // The coordinator owns check lifecycle, dedup, abort, and stale-result
  // control (review 0036 D1B-IF-03). One per app lifetime: injected from
  // main.tsx; a per-instance fallback covers direct <App/> test renders.
  const coordinatorRef = useRef<StatusCheckCoordinator | null>(null);
  if (coordinatorRef.current === null) {
    coordinatorRef.current = statusCoordinator ?? new StatusCheckCoordinator();
  }
  const coordinator = coordinatorRef.current;

  // Stale-result control for the demo status read (see refreshDemoStatus).
  const demoSequenceRef = useRef(0);
  const demoAbortRef = useRef<AbortController | null>(null);

  const applyStatus = useCallback((status: ModelStatus) => {
    setModelStatus(status);
    if (status === "checking") {
      // Fail-closed display (review 0036 D1B-IF-02): entering a check
      // invalidates the previous result — an old grounded answer or Citation
      // must never be presented while (or after) readiness is re-evaluated.
      // The typed question is preserved; nothing is auto-resubmitted.
      generationRef.current += 1;
      setAnswer(null);
      setError(null);
      setErrorRecheckable(false);
      setTranscript([]);
      setGuidedProgress(null);
      setCurrentQuestion(null);
      // An in-flight request cannot be cancelled, but its reply will be discarded (see above), so
      // the page stops waiting for it now rather than showing a reply indicator for nothing.
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Exactly one automatic check per app lifetime (StrictMode-safe);
    // afterwards only the manual "Check again" control re-runs it.
    coordinator.attach(applyStatus);
    return () => coordinator.detach(applyStatus);
  }, [coordinator, applyStatus]);

  const refreshDemoStatus = useCallback((signal?: AbortSignal) => {
    // Opt-in only: a build that did not ask for the internal demo makes no extra request.
    if (!internalDemoUiEnabled()) {
      return;
    }
    // One read at a time, newest wins: a slow earlier /health answer must never overwrite a newer
    // one, which is the stale-result rule the readiness coordinator already follows (0036
    // D1B-IF-03). A displayed model identity one request out of date is a false claim.
    demoAbortRef.current?.abort();
    const controller = new AbortController();
    demoAbortRef.current = controller;
    if (signal) {
      if (signal.aborted) {
        return;
      }
      signal.addEventListener("abort", () => controller.abort());
    }
    const sequence = demoSequenceRef.current + 1;
    demoSequenceRef.current = sequence;
    void fetchDemoStatus({ signal: controller.signal }).then((status) => {
      if (sequence === demoSequenceRef.current && !controller.signal.aborted) {
        setDemo(status);
      }
    });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    refreshDemoStatus(controller.signal);
    return () => controller.abort();
  }, [refreshDemoStatus]);

  // Keep the newest message in view as the conversation grows.
  useEffect(() => {
    const end = endRef.current;
    if (end && typeof end.scrollIntoView === "function") {
      const reduced =
        typeof window.matchMedia === "function" &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      end.scrollIntoView({ block: "end", behavior: reduced ? "auto" : "smooth" });
    }
  }, [transcript.length, currentQuestion, loading, error]);

  const statusText = MODEL_STATUS_TEXT[modelStatus];
  const courseReady = modelStatus === "ready";

  // Is this an internal demo? Two independent signals, either sufficient: what the backend reports
  // about itself, and the warning the API attaches to every demo answer. The second arrives even in
  // a build served without the demo flag, which is what keeps this page from describing demo output
  // in the vocabulary of approved course material.
  const demoUiEnabled = internalDemoUiEnabled();
  const answerIsDemo = answer !== null && isDemoAnswer(answer);
  const demoProfile = demo?.internalDemo === true || answerIsDemo;
  // D1B-03 gates submission on readiness so the page sends no pointless requests. In the internal
  // demo a request is never pointless while the API itself is reachable: greetings and the other
  // assistant interactions, clarifying questions and not-covered answers need no model, and a
  // question that does need it fails closed with the explicit model-unavailable state. So in the
  // demo the composer stays usable whenever the API answers, and "hi" still gets its reply while
  // the GPU host is down. Every other profile keeps the readiness gate exactly as before; the
  // backend remains the authority either way.
  const apiReachable = modelStatus !== "checking" && modelStatus !== "api_unreachable";
  const canAsk = courseReady || (demoProfile && apiReachable);
  // The demo API is answering, but this build shows no verified identity for it: say so rather than
  // leaving the status panel silently incomplete.
  const demoUiMissing = !demoUiEnabled && answerIsDemo;
  const showNotice = demoUiEnabled || demoProfile;

  const recheck = useCallback(() => {
    coordinator.recheck();
    refreshDemoStatus();
  }, [coordinator, refreshDemoStatus]);

  async function sendQuestion(text: string) {
    const trimmed = text.trim();
    // D1B-03: the readiness UI gates submission (the backend gate remains
    // the authority; this only prevents pointless requests). See canAsk for the demo case.
    if (!trimmed || loading || !canAsk) {
      return;
    }
    const generation = generationRef.current;
    setCurrentQuestion(trimmed);
    setLoading(true);
    setError(null);
    setErrorRecheckable(false);
    setAnswer(null);
    try {
      const reply = await askCourseQuestion(trimmed, answerMode, {
        history: transcript,
        guidedProgress,
      });
      if (generation !== generationRef.current) {
        return; // a readiness check started meanwhile: this reply belongs to a page that is gone
      }
      setAnswer(reply);
      setGuidedProgress(reply.guided_progress ?? null);
      // The mirror of D1B-05 for a success: in the demo a question can be sent while the status
      // still shows a model outage, and a grounded reply proves the model has since answered with
      // a verified identity. Only that case updates the status; see statusAfterCourseAnswer. No
      // check can have started since this request was sent (the generation still matches), so
      // modelStatus is the status it was sent under. The identity shown beside the new status is
      // read again rather than left as it was while the model was down.
      const statusUpdate = statusAfterCourseAnswer(modelStatus, reply);
      if (statusUpdate !== null) {
        setModelStatus(statusUpdate);
        refreshDemoStatus();
      }
      // Keep the exchange for the next question. The API stays stateless; this is the only memory.
      setTranscript((previous) => [
        ...previous,
        { role: "user", content: trimmed },
        { role: "assistant", content: reply.answer_text },
      ]);
      setCurrentQuestion(null);
      // The sent question now lives in the conversation, so the composer is cleared for the next
      // one — unless the student already started typing something else while waiting. A typed
      // draft still survives a readiness change (review 0036 D1B-IF-02); only a successful send
      // consumes it, and a failed send leaves it in place to retry.
      setQuestion((current) => (current.trim() === trimmed ? "" : current));
    } catch (err) {
      if (generation !== generationRef.current) {
        return; // same rule for a late failure: the newer check owns the status now
      }
      setError(
        err instanceof ApiRequestError
          ? err.message
          : "An unexpected error occurred while contacting the tutor API.",
      );
      // A failed turn is not added to the transcript: the next question must not be answered as a
      // follow-up to something the assistant never said.
      // D1B-05: stable machine codes (503/504 model codes, network failure)
      // synchronize the status display; 502/500/422 say nothing about
      // readiness and leave it unchanged.
      const statusUpdate = statusFromCourseError(err);
      if (statusUpdate !== null) {
        setModelStatus(statusUpdate);
      }
      setErrorRecheckable(statusUpdate !== null);
    } finally {
      if (generation === generationRef.current) {
        setLoading(false);
      }
    }
  }

  function pickSuggestion(text: string) {
    setQuestion(text);
    textareaRef.current?.focus();
  }

  const conversationEmpty =
    transcript.length === 0 && currentQuestion === null && !loading && error === null;

  return (
    <div className="app">
      <AppHeader
        modelStatus={modelStatus}
        experimental={showNotice}
        detailsOpen={detailsOpen}
        onToggleDetails={() => setDetailsOpen((previous) => !previous)}
        onRecheck={recheck}
      />
      <SystemDetails
        open={detailsOpen}
        modelStatus={modelStatus}
        demo={demo}
        demoUiMissing={demoUiMissing}
      />
      <main className="app-main">
        <div className={`app-main__inner${conversationEmpty ? " app-main__inner--empty" : ""}`}>
          {conversationEmpty ? (
            <EmptyState onPick={pickSuggestion} />
          ) : (
            <Conversation
              transcript={transcript}
              answer={answer}
              currentQuestion={currentQuestion}
              loading={loading}
              error={error}
              errorRecheckable={errorRecheckable}
              demoProfile={demoProfile}
              onRecheck={recheck}
              onNextStep={() => void sendQuestion("next")}
              endRef={endRef}
            />
          )}
        </div>
      </main>
      <Composer
        question={question}
        onQuestionChange={setQuestion}
        answerMode={answerMode}
        onAnswerModeChange={setAnswerMode}
        onSubmit={() => void sendQuestion(question)}
        loading={loading}
        courseReady={courseReady}
        canAsk={canAsk}
        availabilityText={statusText.course}
        textareaRef={textareaRef}
        notice={
          // Persistent whenever this is a demo: shown before any status is known, while loading, on
          // every answer and on every error. It is not shown in a build that neither opted into the
          // demo UI nor is talking to a demo API, because there it would be a false statement.
          showNotice ? (
            <p className="composer__notice" data-testid="internal-demo-notice" role="note">
              {INTERNAL_DEMO_NOTICE}
            </p>
          ) : undefined
        }
      />
    </div>
  );
}
