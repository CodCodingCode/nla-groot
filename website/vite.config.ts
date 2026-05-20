import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Use relative base so the bundle works under any GitHub Pages subpath.
  base: "./",
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
