import { expect, test, type Page } from "@playwright/test";
import { PNG } from "pngjs";

function graphFixture(nodeCount = 120) {
  const nodes = Array.from({ length: nodeCount }, (_, index) => ({
    id: `entity-${index}`,
    label: index === 0 ? "青铜城" : `知识节点 ${index}`,
    type: ["person", "location", "object", "event", "concept"][index % 5],
    description: `节点 ${index} 的完整描述`,
    source_id: `chunk-${index}`,
    degree: index === 0 ? 30 : 2,
  }));
  const edges = Array.from({ length: nodeCount * 2 }, (_, index) => ({
    source: `entity-${index % nodeCount}`,
    target: `entity-${(index * 7 + 1) % nodeCount}`,
  })).filter(edge => edge.source !== edge.target);
  return { nodes, edges, metadata: { total_nodes: nodes.length, total_edges: edges.length, truncated: false } };
}

async function mockApi(page: Page) {
  await page.route("**/api/v1/sessions", route => route.fulfill({ json: { sessions: [{ id: "visual-test", title: "视觉测试" }] } }));
  await page.route("**/api/v1/knowledge-graph/data**", route => route.fulfill({ json: graphFixture() }));
  await page.route("**/api/v1/knowledge-graph/entity-chunks**", route => route.fulfill({
    json: { entity_id: "entity-0", source_ids: ["chunk-0"], chunks: [{ chunk_id: "chunk-0", content: "这是用于验证右侧证据面板的原文内容。", chunk_order_index: 0 }] },
  }));
  await page.route("**/api/v1/knowledge-graph/edge-chunks**", route => route.fulfill({
    json: { edge_id: "edge-test", source_ids: ["chunk-edge"], chunks: [{ chunk_id: "chunk-edge", content: "这是用于验证关系边证据面板的原文内容。", chunk_order_index: 1 }] },
  }));
}

for (const viewport of [
  { width: 1440, height: 900 },
  { width: 1280, height: 720 },
  { width: 390, height: 844 },
]) {
  test(`renders the semantic star map at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await mockApi(page);
    await page.goto("/promo-graph");
    const stage = page.locator("#star-map-stage");
    await expect(stage).toHaveAttribute("data-webgl-ready", "true", { timeout: 15_000 });
    await expect(stage).toHaveAttribute("data-node-count", "120");

    await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.zoom(0.6));
    await expect(stage).toHaveAttribute("data-label-level", "0");
    await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.zoom(5.8));
    await expect(stage).toHaveAttribute("data-label-level", "4");

    await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.select(0));
    await expect(page.locator("#node-panel")).toBeVisible();
    await expect(page.locator("[data-detail-title]")).toHaveText("青铜城");
    await expect(page.locator(".chunk-card")).toContainText("用于验证右侧证据面板");

    const image = PNG.sync.read(await stage.screenshot());
    let nonBackground = 0;
    for (let index = 0; index < image.data.length; index += 40) {
      const red = image.data[index];
      const green = image.data[index + 1];
      const blue = image.data[index + 2];
      if (red > 15 || green > 18 || blue > 26) nonBackground += 1;
    }
    expect(nonBackground).toBeGreaterThan(30);
    const stats = await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.stats());
    expect(stats?.calls).toBeLessThanOrEqual(12);
  });
}

test("uses the shared conversation session and shows ids for duplicate titles", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("current_session_id", "visual-test");
    localStorage.setItem("docthinker.promo.session", "stale-session");
  });
  await page.route("**/api/v1/sessions", route => route.fulfill({
    json: {
      sessions: [
        { id: "visual-test", title: "新对话" },
        { id: "stale-session", title: "新对话" },
      ],
    },
  }));
  await page.route("**/api/v1/knowledge-graph/data**", route => route.fulfill({ json: graphFixture() }));
  await page.route("**/api/v1/knowledge-graph/entity-chunks**", route => route.fulfill({ json: { chunks: [] } }));

  await page.goto("/promo-graph");
  const selector = page.locator("#session-select");
  await expect(page.locator("#star-map-stage")).toHaveAttribute("data-webgl-ready", "true", { timeout: 15_000 });
  await expect(selector).toHaveValue("visual-test");
  await expect(selector.locator('option[value="visual-test"]')).toHaveText("visual-test | 新对话");
  await expect(selector.locator('option[value="stale-session"]')).toHaveText("stale-session | 新对话");
  expect(await page.evaluate(() => localStorage.getItem("current_session_id"))).toBe("visual-test");
});

test("selects fact and ECLRR-v4 edges and opens their relation evidence", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.route("**/api/v1/sessions", route => route.fulfill({ json: { sessions: [{ id: "edge-test", title: "关系测试" }] } }));
  await page.route("**/api/v1/knowledge-graph/data**", route => route.fulfill({
    json: {
      nodes: ["A", "B", "C", "D"].map(id => ({ id, label: id, type: "person" })),
      edges: [
        { id: "fact-edge", source: "A", target: "B", relation: "协作", description: "A 与 B 共同执行任务。", source_id: "chunk-fact", edge_kind: "original" },
        {
          id: "rel-eclrr-edge",
          source: "C",
          target: "D",
          relation: "间接影响",
          description: "C 经证据链间接影响 D。",
          source_id: "chunk-1<SEP>chunk-2",
          edge_kind: "eclrr_v4",
          is_promoted: true,
          path_used: JSON.stringify(["C", "X", "Y", "D"]),
          evidence_chain: JSON.stringify([{ source: "C", target: "X", chunk_id: "chunk-1", quote: "C 影响 X" }]),
          evidence_chunk_ids: JSON.stringify(["chunk-1", "chunk-2"]),
          judge_scores: JSON.stringify({ total: 9 }),
        },
      ],
      metadata: { total_nodes: 4, total_edges: 2, truncated: false },
    },
  }));
  await page.route("**/api/v1/knowledge-graph/edge-chunks**", route => {
    const url = new URL(route.request().url());
    const edgeId = url.searchParams.get("edge_id") || "";
    route.fulfill({
      json: {
        edge_id: edgeId,
        source_ids: edgeId === "fact-edge" ? ["chunk-fact"] : ["chunk-1", "chunk-2"],
        chunks: [{ chunk_id: edgeId === "fact-edge" ? "chunk-fact" : "chunk-1", content: `关系 ${edgeId} 的原文证据。` }],
      },
    });
  });

  await page.goto("/promo-graph");
  const stage = page.locator("#star-map-stage");
  await expect(stage).toHaveAttribute("data-webgl-ready", "true", { timeout: 15_000 });
  await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.autoRotate(false));
  const stageBounds = await stage.boundingBox();
  const factPosition = await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.edgePosition(0));
  expect(stageBounds).not.toBeNull();
  expect(factPosition).not.toBeNull();
  await page.mouse.click((stageBounds?.x ?? 0) + (factPosition?.x ?? 0), (stageBounds?.y ?? 0) + (factPosition?.y ?? 0));
  await expect(stage).toHaveAttribute("data-selected-edge", "0");
  await expect(page.locator("[data-detail-kind]")).toHaveText("FACT EDGE");
  await expect(page.locator("[data-detail-title]")).toHaveText("协作");
  await expect(page.locator("[data-detail-meta]")).toContainText("入库事实关系 · 实线");
  await expect(page.locator(".chunk-card")).toContainText("fact-edge");

  const inferredPosition = await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.edgePosition(1));
  expect(inferredPosition).not.toBeNull();
  await page.mouse.click((stageBounds?.x ?? 0) + (inferredPosition?.x ?? 0), (stageBounds?.y ?? 0) + (inferredPosition?.y ?? 0));
  await expect(stage).toHaveAttribute("data-selected-edge", "1");
  await expect(page.locator("[data-detail-kind]")).toHaveText("ECLRR-V4 EDGE");
  await expect(page.locator("[data-detail-title]")).toHaveText("间接影响");
  await expect(page.locator("[data-detail-meta]")).toContainText("ECLRR-v4 推断关系 · 虚线");
  await expect(page.locator("[data-relation-evidence]")).toBeVisible();
  await expect(page.locator(".evidence-path")).toHaveText("C → X → Y → D");
  await expect(page.locator(".chunk-card")).toContainText("rel-eclrr-edge");
});

test("auto rotation pauses and resumes without resetting the current zoom", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await mockApi(page);
  await page.goto("/promo-graph");
  const stage = page.locator("#star-map-stage");
  await expect(stage).toHaveAttribute("data-webgl-ready", "true", { timeout: 15_000 });
  const bounds = await stage.boundingBox();
  expect(bounds).not.toBeNull();
  await page.mouse.move((bounds?.x ?? 0) + (bounds?.width ?? 1) / 2, (bounds?.y ?? 0) + (bounds?.height ?? 1) / 2);
  await expect(stage).toHaveAttribute("data-auto-rotating", "true", { timeout: 3_000 });

  const started = await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.rotation());
  await page.waitForTimeout(240);
  const advanced = await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.rotation());
  expect((advanced?.state.camera.azimuth ?? 0) - (started?.state.camera.azimuth ?? 0)).toBeGreaterThan(0);
  expect(advanced?.state.camera.polar ?? 0).toBeGreaterThan(0);

  await page.mouse.wheel(0, -420);
  await expect(stage).toHaveAttribute("data-auto-rotating", "false");
  const zoomed = await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.rotation());
  await expect(stage).toHaveAttribute("data-auto-rotating", "true", { timeout: 3_000 });
  const resumed = await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.rotation());
  expect(resumed?.state.camera.zoom).toBeCloseTo(zoomed?.state.camera.zoom ?? 0, 6);
  expect(resumed?.state.camera.target).toEqual(zoomed?.state.camera.target);

  await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.select(0));
  await expect(stage).toHaveAttribute("data-rotation-pause-reason", "selection");
  await expect(page.locator("#toggle-auto-rotation")).toHaveAttribute("aria-pressed", "true");
});

test("pulls direct neighbors while held and restores their frozen positions on release", async ({ page }) => {
  await mockApi(page);
  await page.goto("/promo-graph");
  const stage = page.locator("#star-map-stage");
  await expect(stage).toHaveAttribute("data-webgl-ready", "true", { timeout: 15_000 });

  const original = await page.evaluate(() => ({
    selected: window.__PROMO_GRAPH_DEBUG__?.position(0),
    neighbor: window.__PROMO_GRAPH_DEBUG__?.position(1),
  }));
  await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.attract(0));
  await page.waitForTimeout(460);
  const focused = await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.position(1));
  expect(Math.hypot(
    (focused?.x ?? 0) - (original.neighbor?.x ?? 0),
    (focused?.y ?? 0) - (original.neighbor?.y ?? 0),
  )).toBeGreaterThan(1);

  await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.releaseAttraction());
  await page.waitForTimeout(380);
  const restored = await page.evaluate(() => ({
    selected: window.__PROMO_GRAPH_DEBUG__?.position(0),
    neighbor: window.__PROMO_GRAPH_DEBUG__?.position(1),
  }));
  expect(restored.selected?.x).toBeCloseTo(original.selected?.x ?? 0, 4);
  expect(restored.selected?.y).toBeCloseTo(original.selected?.y ?? 0, 4);
  expect(restored.neighbor?.x).toBeCloseTo(original.neighbor?.x ?? 0, 4);
  expect(restored.neighbor?.y).toBeCloseTo(original.neighbor?.y ?? 0, 4);
});

test("starts the locally bundled gesture runtime and camera preview", async ({ page }) => {
  await mockApi(page);
  await page.goto("/promo-graph");
  const stage = page.locator("#star-map-stage");
  const layer = page.locator("#gesture-layer");
  const video = page.locator("#gesture-video");
  const toggle = page.locator("#toggle-gesture");
  await expect(stage).toHaveAttribute("data-webgl-ready", "true", { timeout: 15_000 });
  await expect(toggle).toHaveAttribute("aria-pressed", "false");
  await expect(layer).toHaveAttribute("data-state", "idle");
  await expect(layer).toBeHidden();
  expect((await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.gesture()))?.enabled).toBe(false);

  await toggle.click();
  await expect(layer).toHaveAttribute("data-state", "active", { timeout: 30_000 });
  await expect(toggle).toHaveAttribute("aria-pressed", "true");
  await expect(layer).toBeVisible();
  await expect(video).toBeVisible();
  const videoBox = await video.boundingBox();
  expect(videoBox?.width).toBeCloseTo(176, 0);
  expect(videoBox?.height).toBeCloseTo(132, 0);
  const gesture = await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.gesture());
  expect(gesture?.enabled).toBe(true);

  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-pressed", "false");
  await expect(layer).toHaveAttribute("data-state", "idle");
  await expect(layer).toBeHidden();
  expect((await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.gesture()))?.enabled).toBe(false);
});

test("gesture mode degrades without changing the graph when camera access fails", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: () => Promise.reject(new DOMException("Camera unavailable", "NotAllowedError")) },
    });
  });
  await mockApi(page);
  await page.goto("/promo-graph");
  const stage = page.locator("#star-map-stage");
  const toggle = page.locator("#toggle-gesture");
  await expect(stage).toHaveAttribute("data-webgl-ready", "true", { timeout: 15_000 });
  await expect(stage).toHaveAttribute("data-node-count", "120");
  await expect(toggle).toHaveAttribute("aria-pressed", "false");
  await toggle.click();
  await expect(page.locator("#gesture-layer")).toHaveAttribute("data-state", "unavailable");
  await expect(page.locator("#gesture-layer")).toBeHidden();
  await expect(toggle).toHaveAttribute("aria-pressed", "false");
});

test("gesture experience chooser opens the GPU graph with gesture control off", async ({ page }) => {
  await page.goto("/gesture-experience");
  const gestureLink = page.locator('a.chooser-option.is-promo[href="/promo-graph"]');
  await expect(gestureLink).toHaveCount(1);
});
