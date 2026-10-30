import type { ReactNode, RefObject } from "react";
import type { ConversationTurn, CourseAnswer } from "../api";
import { answerView, nextStepAvailable } from "../presentation";
import { AnswerText } from "./AnswerText";
import { BrandMark } from "./Icons";
import { Sources } from "./Sources";

// ---- Message frames ------------------------------------------------------------------------------

function UserMessage({ text }: { readonly text: string }) {
  return (
    <div className="message message--user">
      <p className="bubble">{text}</p>
    </div>
  );
}

function AssistantFrame({ meta, children }: { readonly meta?: string; readonly children: ReactNode }) {
  return (
    <article className="message message--assistant">
      <span className="message__avatar" aria-hidden="true">
        <BrandMark />
      </span>
      <div className="message__body">
        {meta && <p className="message__meta">{meta}</p>}
        {children}
      </div>
    </article>
  );
}

// ---- The latest answer, in full ------------------------------------------------------------------

function Limitations({ warnings }: { readonly warnings: readonly string[] }) {
  if (warnings.length === 0) {
    return null;
  }
  return (
    <div className="limitations">
      <p className="limitations__title">Limitations</p>
      <ul>
        {warnings.map((warning, index) => (
          <li key={index}>{warning}</li>
        ))}
      </ul>
    </div>
  );
}

export function AssistantAnswer({
  answer,
  demoProfile,
  onNextStep,
}: {
  readonly answer: CourseAnswer;
  readonly demoProfile: boolean;
  /** Sends the tutor the word it asked for after a guided step. Offered only while a next step
   * really exists (see nextStepAvailable); nothing is sent on the student's behalf otherwise. */
  readonly onNextStep?: () => void;
}) {
  const view = answerView(answer, demoProfile);
  if (view.kind === "interaction") {
    // A greeting or control reply: not grounded and not short of evidence, so it carries neither
    // verdict nor sources. It reads as what it is — the assistant talking.
    return (
      <AssistantFrame>
        <div data-testid="assistant-interaction">
          <AnswerText text={answer.answer_text} />
        </div>
      </AssistantFrame>
    );
  }
  if (view.kind === "insufficient") {
    // An explicit, distinct state: never styled as a grounded answer, never with a Sources block.
    return (
      <AssistantFrame>
        <div className="answer answer--insufficient" data-testid="grounding-status" role="status">
          <p className={`verdict verdict--${view.tone}`}>
            <span className="verdict__dot" aria-hidden="true" />
            {view.label}
          </p>
          <AnswerText text={answer.answer_text} />
        </div>
        {view.showLimitations && <Limitations warnings={answer.warnings} />}
      </AssistantFrame>
    );
  }
  return (
    <AssistantFrame>
      <p className={`verdict verdict--${view.tone}`} data-testid="grounding-status">
        <span className="verdict__dot" aria-hidden="true" />
        {view.label}
      </p>
      <AnswerText text={answer.answer_text} />
      {onNextStep && nextStepAvailable(answer, demoProfile) && (
        <p className="notice__actions">
          <button type="button" className="button button--secondary" onClick={onNextStep}>
            Next step
          </button>
        </p>
      )}
      {view.showSources && <Sources citations={answer.citations} />}
      {view.showLimitations && <Limitations warnings={answer.warnings} />}
    </AssistantFrame>
  );
}

// ---- Transient replies ---------------------------------------------------------------------------

function LoadingReply() {
  return (
    <AssistantFrame>
      <p className="thinking" role="status">
        <span className="thinking__dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        Waiting for the local tutor…
      </p>
    </AssistantFrame>
  );
}

function ErrorReply({
  message,
  recheckable,
  onRecheck,
}: {
  readonly message: string;
  readonly recheckable: boolean;
  readonly onRecheck: () => void;
}) {
  return (
    <AssistantFrame>
      <div className="notice notice--error" role="alert">
        {message}
      </div>
      {recheckable && (
        <p className="notice__actions">
          <button type="button" className="button button--secondary" onClick={onRecheck}>
            Re-check connection
          </button>
          <span className="notice__hint">Re-checking starts a fresh conversation.</span>
        </p>
      )}
    </AssistantFrame>
  );
}

// ---- The conversation ----------------------------------------------------------------------------

export interface ConversationProps {
  /** Committed exchanges, oldest first; what the next question carries as history. */
  readonly transcript: readonly ConversationTurn[];
  /** The latest answer in full. Null while a request is in flight or after a failed one, which is
   * what keeps a stale verdict or citation off the page. */
  readonly answer: CourseAnswer | null;
  /** The question of the in-flight or failed attempt; null once it has been answered. */
  readonly currentQuestion: string | null;
  readonly loading: boolean;
  readonly error: string | null;
  readonly errorRecheckable: boolean;
  readonly demoProfile: boolean;
  readonly onRecheck: () => void;
  /** Asks the tutor for the next guided step; offered under a guided-step reply only. */
  readonly onNextStep?: () => void;
  readonly endRef: RefObject<HTMLDivElement>;
}

/** The conversation flow. Earlier replies keep their text; the verdict and the source cards are
 * shown for the latest reply only, which is the existing stale-answer rule: an answer's evidence
 * is never presented alongside a newer request. */
export function Conversation({
  transcript,
  answer,
  currentQuestion,
  loading,
  error,
  errorRecheckable,
  demoProfile,
  onRecheck,
  onNextStep,
  endRef,
}: ConversationProps) {
  const lastIndex = transcript.length - 1;
  return (
    <section className="conversation" aria-label="Conversation">
      {transcript.map((turn, index) => {
        if (turn.role === "user") {
          return <UserMessage key={index} text={turn.content} />;
        }
        if (index === lastIndex && answer !== null) {
          return (
            <AssistantAnswer
              key={index}
              answer={answer}
              demoProfile={demoProfile}
              onNextStep={loading ? undefined : onNextStep}
            />
          );
        }
        return (
          <AssistantFrame key={index} meta="Earlier reply">
            <AnswerText text={turn.content} />
          </AssistantFrame>
        );
      })}
      {currentQuestion !== null && <UserMessage text={currentQuestion} />}
      {loading && <LoadingReply />}
      {error && <ErrorReply message={error} recheckable={errorRecheckable} onRecheck={onRecheck} />}
      <div ref={endRef} className="conversation__end" />
    </section>
  );
}
