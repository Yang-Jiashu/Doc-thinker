import { describe, expect, it } from "vitest";
import { buildCsr, neighborsOf, normalizeGraph } from "../src/graph-data";

describe("full graph normalization", () => {
  it("keeps every valid node instead of applying a display cap", () => {
    const nodes = Array.from({ length: 750 }, (_, index) => ({
      id: `node-${index}`,
      label: `节点 ${index}`,
      type: index % 2 ? "person" : "location",
    }));
    const edges = Array.from({ length: 749 }, (_, index) => ({ source: `node-${index}`, target: `node-${index + 1}` }));
    const graph = normalizeGraph({ nodes, edges, metadata: { total_nodes: 750 } });
    expect(graph.nodes).toHaveLength(750);
    expect(graph.edges).toHaveLength(749);
    expect(graph.positions).toHaveLength(2250);
    expect(graph.fingerprint).toMatch(/^750-749-/);
  });

  it("builds deterministic CSR adjacency", () => {
    const pairs = Uint32Array.from([0, 1, 1, 2, 1, 3]);
    const csr = buildCsr(4, pairs);
    expect([...csr.offsets]).toEqual([0, 1, 4, 5, 6]);
    const graph = normalizeGraph({
      nodes: [0, 1, 2, 3].map(id => ({ id: String(id) })),
      edges: [{ source: "0", target: "1" }, { source: "1", target: "2" }, { source: "1", target: "3" }],
    });
    expect([...neighborsOf(graph, 1)].sort()).toEqual([0, 2, 3]);
  });
});
