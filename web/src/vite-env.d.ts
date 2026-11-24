/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** GitHub repository that notes are filed in, as owner/name. */
  readonly VITE_REPO?: string;
  /** Commit the viewer was built from. */
  readonly VITE_COMMIT?: string;
  /** Public address of the viewer, for links in tickets. */
  readonly VITE_SITE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
