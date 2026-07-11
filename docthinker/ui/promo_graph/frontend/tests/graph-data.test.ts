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

  it("keeps fact and ECLRR-v4 edge metadata distinct", () => {
    const graph = normalizeGraph({
      nodes: [{ id: "A" }, { id: "D" }, { id: "E" }],
      edges: [
        { source: "A", target: "D", label: "同场出现", source_id: "chunk-fact", edge_kind: "original" },
        {
          id: "rel-eclrr-1",
          source: "D",
          target: "E",
          relation: "间接影响",
          source_id: "chunk-1<SEP>chunk-2",
          edge_kind: "eclrr_v4",
          is_promoted: true,
          review_status: "promoted",
          provenance: "eclrr_v4",
          algorithm_version: "eclrr_v4",
          path_used: JSON.stringify(["D", "B", "C", "E"]),
          evidence_chain: JSON.stringify([{ source: "D", target: "B", chunk_id: "chunk-1", quote: "D 影响 B" }]),
          evidence_chunk_ids: JSON.stringify(["chunk-1", "chunk-2"]),
          judge_scores: JSON.stringify({ total: 9, evidence_coverage: 4 }),
        },
      ],
    });

    expect(graph.edges[0].kind).toBe("original");
    expect(graph.edges[0].isPromoted).toBe(false);
    expect(graph.edges[1]).toMatchObject({
      id: "rel-eclrr-1",
      kind: "eclrr_v4",
      isPromoted: true,
      label: "间接影响",
      pathUsed: ["D", "B", "C", "E"],
      evidenceChunkIds: ["chunk-1", "chunk-2"],
      judgeScores: { total: 9, evidence_coverage: 4 },
    });
  });
});
