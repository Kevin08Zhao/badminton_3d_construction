/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute origin for API (e.g. https://api.example.com). Omit to use same-origin `/api` (Vite dev/preview proxy). */
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
