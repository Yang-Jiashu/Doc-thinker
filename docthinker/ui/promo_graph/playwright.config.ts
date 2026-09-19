import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "frontend/e2e",
  timeout: 30_000,
  fullyParallel: false,
  use: {
    baseURL: process.env.PROMO_GRAPH_URL || "http://127.0.0.1:5000",
    permissions: ["camera"],
    launchOptions: {
      args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
    },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
});
