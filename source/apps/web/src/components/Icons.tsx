// Small inline icons. All decorative: hidden from assistive technology, named by their buttons.

interface IconProps {
  readonly className?: string;
}

function base(className: string | undefined) {
  return {
    className,
    width: 16,
    height: 16,
    viewBox: "0 0 16 16",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.75,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
    focusable: "false" as const,
  };
}

/** The product mark: a wireframe cube, the plainest possible nod to solid modelling. */
export function BrandMark({ className }: IconProps) {
  return (
    <svg
      className={className}
      width={22}
      height={22}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M12 2.75 20.25 7.5v9L12 21.25 3.75 16.5v-9z" />
      <path d="M12 12 20.25 7.5M12 12 3.75 7.5M12 12v9.25" />
    </svg>
  );
}

export function ArrowUpIcon({ className }: IconProps) {
  return (
    <svg {...base(className)}>
      <path d="M8 13V3M3.5 7.5 8 3l4.5 4.5" />
    </svg>
  );
}

export function RefreshIcon({ className }: IconProps) {
  return (
    <svg {...base(className)}>
      <path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9" />
      <path d="M13.5 2.5v3h-3" />
    </svg>
  );
}

export function ChevronIcon({ className }: IconProps) {
  return (
    <svg {...base(className)}>
      <path d="m6 3.5 4.5 4.5L6 12.5" />
    </svg>
  );
}

export function InfoIcon({ className }: IconProps) {
  return (
    <svg {...base(className)}>
      <circle cx="8" cy="8" r="6" />
      <path d="M8 7.25v4M8 5v.01" />
    </svg>
  );
}

export function SpinnerIcon({ className }: IconProps) {
  return (
    <svg {...base(className)}>
      <path d="M8 2.5a5.5 5.5 0 1 1-5.5 5.5" />
    </svg>
  );
}
