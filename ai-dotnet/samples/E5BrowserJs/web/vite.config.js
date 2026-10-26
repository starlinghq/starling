import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Cross-origin isolation (SharedArrayBuffer) lets onnxruntime-web use several threads; production hosts send the same
// two headers (see ../README.md).
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [react()],
  server: { headers: isolation },
  preview: { headers: isolation },
});
