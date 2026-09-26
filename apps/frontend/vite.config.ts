import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Vite only serves as a dev server + build tool here — there is no Node
// backend in this project. All real work (ingestion, retrieval, the
// LangGraph agent loop) happens in the separate FastAPI service; this app
// only talks to it over HTTP (see src/api/client.ts + .env's VITE_API_BASE_URL).
export default defineConfig({
  plugins: [react()],
  resolve: {
    // Mirrors the "@/*" path in tsconfig.json — Vite does not read tsconfig
    // paths itself, so the alias must also be declared here or every
    // `@/...` import fails at dev/build time.
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: {
    port: 5173,
    // Optional convenience: proxy /api/* to the FastAPI server during dev so
    // the browser never has to deal with CORS. Disable by setting
    // VITE_API_BASE_URL to an absolute URL instead (see src/api/client.ts).
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, "/v1"),
      },
    },
  },
});
