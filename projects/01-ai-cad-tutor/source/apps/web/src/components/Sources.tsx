import { useId, useState } from "react";
import type { Citation } from "../api";
import { DOCUMENT_TYPE_LABEL, describeLocator } from "../presentation";
import { ChevronIcon } from "./Icons";

export interface SourcesProps {
  readonly citations: readonly Citation[];
}

/** The compact, collapsible source list under a grounded answer. Every field shown comes from a
 * validated citation: title, citation id, document type, locator, corpus id and the retrieved
 * excerpt. Nothing is added, and a citation's presence is not a claim that the answer is right. */
export function Sources({ citations }: SourcesProps) {
  const [open, setOpen] = useState(true);
  const listId = useId();
  if (citations.length === 0) {
    return null;
  }
  return (
    <section className="sources">
      <button
        type="button"
        className="sources__toggle"
        aria-expanded={open}
        aria-controls={listId}
        onClick={() => setOpen((previous) => !previous)}
      >
        <ChevronIcon className={`chevron${open ? " chevron--open" : ""}`} />
        <span className="sources__title">Sources</span>
        <span className="sources__count">{citations.length}</span>
      </button>
      <ul id={listId} className="sources__list" hidden={!open}>
        {citations.map((citation) => (
          <li key={citation.citation_id} className="source">
            <div className="source__head">
              <span className="source__title">{citation.title}</span>
              <code className="source__id">[{citation.citation_id}]</code>
            </div>
            <p className="source__meta">
              {DOCUMENT_TYPE_LABEL[citation.document_type]} · {describeLocator(citation.locator)} ·{" "}
              <span className="mono" data-testid="citation-corpus">
                {citation.document_id}
              </span>
            </p>
            {citation.snippet !== undefined && (
              <blockquote className="source__excerpt" data-testid="citation-excerpt">
                {citation.snippet}
              </blockquote>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
