import { splitAnswerText } from "../presentation";

export interface AnswerTextProps {
  readonly text: string;
}

/** Answer prose as paragraphs of plain text. React escapes everything; no markup is parsed, no
 * list or step structure is inferred, and the server's source line is only styled, never edited. */
export function AnswerText({ text }: AnswerTextProps) {
  return (
    <div className="answer-text">
      {splitAnswerText(text).map((block, index) => (
        <p key={index} className={block.sourceLine ? "answer-text__sourceline" : undefined}>
          {block.text}
        </p>
      ))}
    </div>
  );
}
