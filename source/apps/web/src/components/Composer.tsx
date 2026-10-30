import { FormEvent, KeyboardEvent, ReactNode, RefObject, useLayoutEffect } from "react";
import { ANSWER_MODES } from "../api";
import type { AnswerMode } from "../api";
import { ArrowUpIcon, SpinnerIcon } from "./Icons";

const MAX_QUESTION_CHARACTERS = 2000;

/** True on a touch-first device. Unknown environments (no matchMedia) count as a fine pointer. */
function coarsePointer(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches
  );
}

export interface ComposerProps {
  readonly question: string;
  readonly onQuestionChange: (value: string) => void;
  readonly answerMode: AnswerMode;
  readonly onAnswerModeChange: (mode: AnswerMode) => void;
  /** Submit the current question. Called by the form and by Enter in the textarea. */
  readonly onSubmit: () => void;
  readonly loading: boolean;
  /** Whether the course tutor is ready. Drives the hint under the composer. */
  readonly courseReady: boolean;
  /** The submission gate: the backend remains the authority, this only prevents pointless requests.
   * Equal to courseReady except in the internal demo, where the API answers greetings and the other
   * assistant interactions without the model and so stays askable whenever it is reachable. */
  readonly canAsk?: boolean;
  /** The fixed course-availability sentence, shown under the composer whenever the tutor is not
   * ready so the disabled send button never has to explain itself. */
  readonly availabilityText: string;
  readonly textareaRef: RefObject<HTMLTextAreaElement>;
  /** Rendered beneath the composer: the internal-use notice when this is a demo. */
  readonly notice?: ReactNode;
}

export function Composer({
  question,
  onQuestionChange,
  answerMode,
  onAnswerModeChange,
  onSubmit,
  loading,
  courseReady,
  canAsk = courseReady,
  availabilityText,
  textareaRef,
  notice,
}: ComposerProps) {
  const canSend = !loading && question.trim().length > 0 && canAsk;
  // When the tutor model is down but the demo can still answer locally, say what still works.
  const hintText =
    !courseReady && canAsk
      ? `${availabilityText} Greetings still work; course questions need the model.`
      : availabilityText;

  // Grow with the text up to the stylesheet's max-height; min-height keeps it one comfortable
  // line. scrollHeight is 0 in environments without layout, where min-height alone applies.
  useLayoutEffect(() => {
    const area = textareaRef.current;
    if (!area) return;
    area.style.height = "auto";
    if (area.scrollHeight > 0) {
      area.style.height = `${area.scrollHeight}px`;
    }
  }, [question, textareaRef]);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    onSubmit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter keeps a new line. A composing IME keystroke is left alone, and on a
    // touch keyboard (no Shift+Enter to reach for) Enter stays a new line and the button sends.
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.altKey &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.nativeEvent.isComposing &&
      !coarsePointer()
    ) {
      event.preventDefault();
      onSubmit();
    }
  }

  return (
    <div className="composer-dock">
      <form className="composer" onSubmit={handleSubmit}>
        <div className={`composer__card${loading ? " composer__card--busy" : ""}`}>
          <label htmlFor="question" className="sr-only">
            Question
          </label>
          <textarea
            id="question"
            ref={textareaRef}
            className="composer__input"
            value={question}
            onChange={(event) => onQuestionChange(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            maxLength={MAX_QUESTION_CHARACTERS}
            placeholder="Ask about a SOLIDWORKS concept or feature…"
            aria-describedby="composer-hint"
          />
          <div className="composer__row">
            <span
              id="composer-hint"
              className={`composer__hint ${courseReady ? "composer__hint--keys" : "composer__hint--status"}`}
            >
              {courseReady ? "Enter to send · Shift+Enter for a new line" : hintText}
            </span>
            <label htmlFor="answer-mode" className="composer__mode-label">
              Answer mode
            </label>
            <select
              id="answer-mode"
              className="composer__mode"
              value={answerMode}
              onChange={(event) => {
                // No type assertion: only a value from the fixed mode list is accepted.
                const mode = ANSWER_MODES.find((entry) => entry.value === event.target.value);
                if (mode) {
                  onAnswerModeChange(mode.value);
                }
              }}
            >
              {ANSWER_MODES.map((mode) => (
                <option key={mode.value} value={mode.value}>
                  {mode.label}
                </option>
              ))}
            </select>
            <button
              type="submit"
              className="send-button"
              aria-label="Submit question"
              title="Submit question"
              disabled={!canSend}
            >
              {loading ? <SpinnerIcon className="spin" /> : <ArrowUpIcon />}
            </button>
          </div>
        </div>
        {notice}
      </form>
    </div>
  );
}
