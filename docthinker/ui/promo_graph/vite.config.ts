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
      entry: {
        "promo-graph": resolve(__dirname, "frontend/src/main.ts"),
        "gesture-runtime": resolve(__dirname, "frontend/src/gesture-runtime.ts"),
      },
      formats: ["es"],
      fileName: (_format, entryName) => `${entryName}.js`,
    },
    rollupOptions: {
      output: {
        assetFileNames: asset => asset.name?.endsWith(".css") ? "promo-graph.css" : "assets/[name]-[hash][extname]",
        chunkFileNames: "[name]-[hash].js",
      },
    },
  },
  test: {
    environment: "node",
    include: ["frontend/tests/**/*.test.ts"],
  },
});
