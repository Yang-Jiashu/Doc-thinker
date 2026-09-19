import { defineConfig } from "vitest/config";
import { cpSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

export default defineConfig({
  base: "./",
  publicDir: resolve(__dirname, "frontend/public"),
  plugins: [{
    name: "copy-mediapipe-wasm",
    closeBundle() {
      const target = resolve(__dirname, "static/dist/mediapipe/wasm");
      mkdirSync(target, { recursive: true });
      cpSync(resolve(__dirname, "node_modules/@mediapipe/tasks-vision/wasm"), target, { recursive: true });
    },
  }],
  build: {
    outDir: resolve(__dirname, "static/dist"),
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
    lib: {
      entry: resolve(__dirname, "frontend/src/main.ts"),
      formats: ["es"],
      fileName: () => "promo-graph.js",
    },
    rollupOptions: {
      output: {
        assetFileNames: asset => asset.name?.endsWith(".css") ? "promo-graph.css" : "assets/[name]-[hash][extname]",
        chunkFileNames: "chunks/[name]-[hash].js",
      },
    },
  },
  test: {
    environment: "node",
    include: ["frontend/tests/**/*.test.ts"],
  },
});
