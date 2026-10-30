import { SUGGESTED_QUESTIONS } from "../suggestions";
import { BrandMark } from "./Icons";

export interface EmptyStateProps {
  /** Put a suggested question into the composer; the student still sends it. */
  readonly onPick: (question: string) => void;
}

/** The welcome shown before the first question. It disappears as soon as a question is sent. */
export function EmptyState({ onPick }: EmptyStateProps) {
  return (
    <div className="empty" data-testid="empty-state">
      <span className="empty__mark" aria-hidden="true">
        <BrandMark />
      </span>
      <h2 className="empty__title">What are you working on?</h2>
      <p className="empty__lead">Ask about a SOLIDWORKS concept or feature.</p>
      <ul className="chips" aria-label="Suggested questions">
        {SUGGESTED_QUESTIONS.map((question) => (
          <li key={question}>
            <button type="button" className="chip" onClick={() => onPick(question)}>
              {question}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
