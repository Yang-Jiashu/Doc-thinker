import { expect, test } from "@playwright/test";

function performanceGraph(nodeCount: number, edgeCount: number) {
  const nodes = Array.from({ length: nodeCount }, (_, index) => ({
    id: `perf-${nodeCount}-${index}`,
    label: `性能节点 ${index}`,
    type: ["person", "location", "object", "event", "concept"][index % 5],
    degree: 6,
  }));
  const edges = Array.from({ length: edgeCount }, (_, index) => {
    const source = index % nodeCount;
    let target = (source * 17 + Math.floor(index / nodeCount) * 37 + 1) % nodeCount;
    if (target === source) target = (target + 1) % nodeCount;
    return { source: nodes[source].id, target: nodes[target].id };
  });
  return { nodes, edges, metadata: { total_nodes: nodeCount, total_edges: edgeCount, truncated: false } };
}

for (const scenario of [
  { nodes: 2_000, edges: 6_000, minimumFps: 55, maximumLoadMs: 15_000 },
  { nodes: 10_000, edges: 30_000, minimumFps: 45, maximumLoadMs: 35_000 },
  { nodes: 50_000, edges: 150_000, minimumFps: 30, maximumLoadMs: 90_000 },
]) {
  test(`benchmarks ${scenario.nodes} nodes and ${scenario.edges} edges`, async ({ page }) => {
    test.setTimeout(120_000);
    const graph = performanceGraph(scenario.nodes, scenario.edges);
    await page.route("**/api/v1/sessions", route => route.fulfill({ json: { sessions: [{ id: `perf-${scenario.nodes}`, title: "性能测试" }] } }));
    await page.route("**/api/v1/knowledge-graph/data**", route => route.fulfill({ json: graph }));
    const startedAt = Date.now();
    await page.goto("/promo-graph");
    const stage = page.locator("#star-map-stage");
    await expect(stage).toHaveAttribute("data-node-count", String(scenario.nodes), { timeout: scenario.maximumLoadMs });
    await expect(stage).toHaveAttribute("data-webgl-ready", "true", { timeout: scenario.maximumLoadMs });
    const loadMs = Date.now() - startedAt;
    await page.waitForTimeout(1_200);
    const stats = await page.evaluate(() => window.__PROMO_GRAPH_DEBUG__?.stats());
    console.log(JSON.stringify({ nodes: scenario.nodes, edges: scenario.edges, loadMs, fps: stats?.fps, calls: stats?.calls, renderer: stats?.renderer }));
    expect(loadMs).toBeLessThanOrEqual(scenario.maximumLoadMs);
    if (process.env.PROMO_GRAPH_ENFORCE_FPS === "1") {
      expect(stats?.fps ?? 0).toBeGreaterThanOrEqual(scenario.minimumFps);
    } else {
      expect(stats?.fps ?? 0).toBeGreaterThan(2);
    }
    expect(stats?.calls ?? 99).toBeLessThanOrEqual(12);
  });
}
