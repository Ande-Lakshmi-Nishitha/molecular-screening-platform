import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server forwards the API and the health probes to the backend, so the browser only talks to one origin.
// Default: Flask on http://localhost:5000. For the Docker stack (nginx with a self-signed certificate) start with
//   set VITE_PROXY_TARGET=https://localhost   and then   npm run dev
const target = process.env.VITE_PROXY_TARGET || "http://localhost:5000";
const backend = { target, secure: false };

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": backend, "/healthz": backend, "/readyz": backend },
  },
});
