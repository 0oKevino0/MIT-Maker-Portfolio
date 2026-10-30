/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    fs: {
      // The shared JSON contracts in packages/contracts are imported from
      // outside this app root; allow the repository root in dev mode only.
      allow: ["../.."],
    },
  },
  test: {
    environment: "jsdom",
  },
});
