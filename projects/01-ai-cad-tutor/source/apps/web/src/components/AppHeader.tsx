import type { ModelStatus } from "../modelStatus";
import { statusPill } from "../presentation";
import { BrandMark, InfoIcon, RefreshIcon } from "./Icons";

export interface AppHeaderProps {
  readonly modelStatus: ModelStatus;
  /** Show the experimental tag: true whenever the page is in a demo context. */
  readonly experimental: boolean;
  readonly detailsOpen: boolean;
  readonly onToggleDetails: () => void;
  readonly onRecheck: () => void;
}

/** The compact application header: identity on the left, measured status on the right. The full
 * diagnostics live in the System details panel; nothing here is derived from configuration. */
export function AppHeader({
  modelStatus,
  experimental,
  detailsOpen,
  onToggleDetails,
  onRecheck,
}: AppHeaderProps) {
  const pill = statusPill(modelStatus);
  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="brand">
          <span className="brand__mark">
            <BrandMark />
          </span>
          <h1 className="brand__name">CSWA AI Tutor</h1>
        </div>
        <div className="header-tools">
          {experimental && <span className="tag">Experimental</span>}
          <span
            className={`status-pill status-pill--${pill.tone}${pill.busy ? " status-pill--busy" : ""}`}
            role="status"
            data-testid="status-pill"
          >
            <span className="status-pill__dot" aria-hidden="true" />
            <span className="status-pill__label">{pill.label}</span>
          </span>
          {/* Always clickable: a new check ABORTS the previous one (never stacks), and the
              sequence guard drops any superseded result. A check also invalidates the page's
              answers (0036 D1B-IF-02), so the conversation starts over — said here, not hidden. */}
          <button
            type="button"
            className="icon-button icon-button--recheck"
            aria-label="Check again"
            aria-describedby="check-again-note"
            title="Check readiness again — starts a fresh conversation"
            onClick={onRecheck}
          >
            <RefreshIcon />
            <span className="icon-button__label">Check again</span>
          </button>
          <span id="check-again-note" className="sr-only">
            Starts a fresh conversation.
          </span>
          <button
            type="button"
            className="icon-button"
            aria-label="System details"
            title="System details"
            aria-expanded={detailsOpen}
            aria-controls="system-details"
            onClick={onToggleDetails}
          >
            <InfoIcon />
            <span className="icon-button__label">System details</span>
          </button>
        </div>
      </div>
    </header>
  );
}
