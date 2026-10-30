/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  /** Development-only local token; never set this for a production build. */
  readonly VITE_CSWA_LOCAL_TOKEN?: string;
  /** Explicit opt-in for a non-loopback API base URL. */
  readonly VITE_ALLOW_REMOTE_API?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
