import type { DemoStatus } from "../demoStatus";
import { shortDigest } from "../demoStatus";
import type { ModelStatus } from "../modelStatus";
import { MODEL_STATUS_TEXT } from "../modelStatus";

export interface SystemDetailsProps {
  readonly open: boolean;
  readonly modelStatus: ModelStatus;
  /** Verified demo facts read from /health; null until read, and null on a backend without them. */
  readonly demo: DemoStatus | null;
  /** The API answered as the demo but this build was served without the demo status display. */
  readonly demoUiMissing: boolean;
}

/** The diagnostics panel. It is always rendered (collapsed with `hidden`) so the measured status
 * is always in the document; the header toggle controls visibility. Every value here is either the
 * fixed frontend status text or a field the backend measured and reported. */
export function SystemDetails({ open, modelStatus, demo, demoUiMissing }: SystemDetailsProps) {
  const statusText = MODEL_STATUS_TEXT[modelStatus];
  return (
    <div id="system-details" className="system-details" hidden={!open}>
      <section aria-label="Service status" className="system-details__inner">
        <div className="system-details__head">
          <h2 className="system-details__title">System details</h2>
          <p className="system-details__note">
            Readiness is the backend's own verdict, including its model identity check: its
            readiness check, or an answer it just generated with a verified identity. Nothing here
            is derived from configuration.
          </p>
        </div>
        <dl className="details-grid">
          <div className="details-row">
            <dt>API</dt>
            <dd data-testid="api-status">{statusText.api}</dd>
          </div>
          <div className="details-row">
            <dt>Model</dt>
            <dd data-testid="model-status">{statusText.model}</dd>
          </div>
          <div className="details-row">
            <dt>Course tutor</dt>
            <dd data-testid="course-availability">{statusText.course}</dd>
          </div>
          {demo?.identity && (
            <div className="details-row">
              <dt>Model identity</dt>
              <dd data-testid="model-identity">
                <span>{demo.identity.baseModelId}</span>
                {demo.identity.baseModelRevision && (
                  <span className="details-muted">
                    {" "}
                    @ <code className="mono">{shortDigest(demo.identity.baseModelRevision)}</code>
                  </span>
                )}
                <br />
                V1B adapter <code className="mono">{shortDigest(demo.identity.adapterSha256)}</code>
                {demo.identity.adapterMerged ? " — merged" : " — not merged"}
                {demo.identity.inferenceOnly ? ", inference only" : ""}
                {demo.identity.device ? `, on ${demo.identity.device}` : ""}
                <span className="details-muted"> (verified by the backend)</span>
              </dd>
            </div>
          )}
          {demo && (
            <div className="details-row">
              <dt>Retrieval</dt>
              <dd data-testid="retrieval-status">
                {demo.retrievalState}
                {demo.corpus
                  ? ` — ${demo.corpus.version}, ${demo.corpus.records} records${
                      demo.topK ? `, top ${demo.topK}` : ""
                    }`
                  : ""}
                {demo.corpus?.label ? ` (${demo.corpus.label})` : ""}
              </dd>
            </div>
          )}
        </dl>
        {demoUiMissing && (
          <p className="system-details__warning" data-testid="demo-ui-off">
            This API answered as the internal demo, but this build was served without the demo
            status display: no verified model identity or retrieval source is shown here. Set
            VITE_INTERNAL_DEMO=true to see them.
          </p>
        )}
      </section>
    </div>
  );
}
