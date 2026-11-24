import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset paths, so the built site works from any folder (GitHub Pages
  // serves it under /<repository>/).
  base: './',
  build: { target: 'es2022', chunkSizeWarningLimit: 1200 },
  server: { port: 5173, strictPort: true },
});
