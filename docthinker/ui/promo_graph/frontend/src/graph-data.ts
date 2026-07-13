import type {
  GraphEdge,
  GraphModel,
  GraphNode,
  RawGraphEdge,
  RawGraphResponse,
} from "./types";

const GROUP_COLORS: Array<[number, number, number]> = [
  [0x65 / 255, 0xe7 / 255, 0xf2 / 255],
  [0x7b / 255, 0xa7 / 255, 1],
  [0xf4 / 255, 0xc8 / 255, 0x6a / 255],
  [1, 0x7a / 255, 0x7a / 255],
  [0xb9 / 255, 0x94 / 255, 1],
  [0xda / 255, 0xe5 / 255, 0xf4 / 255],
];

const TYPE_GROUPS = new Map<string, number>([
  ["person", 0], ["people", 0], ["human", 0], ["人物", 0],
  ["location", 1], ["place", 1], ["space", 1], ["地点", 1], ["空间", 1],
  ["artifact", 2], ["object", 2], ["item", 2], ["物品", 2], ["器物", 2],
  ["event", 3], ["事件", 3],
  ["concept", 4], ["content", 4], ["概念", 4], ["内容", 4],
]);

function text(value: unknown): string {
  return String(value ?? "").trim();
}

function endpoint(value: unknown): string {
  if (value && typeof value === "object" && "id" in value) {
    return text((value as { id?: unknown }).id);
  }
  return text(value);
}

function truthy(value: unknown): boolean {
  return value === true || value === 1 || ["1", "true", "yes"].includes(text(value).toLowerCase());
}

function jsonValue(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return undefined;
  }
}

function jsonArray<T = unknown>(value: unknown): T[] {
  const parsed = jsonValue(value);
  return Array.isArray(parsed) ? parsed as T[] : [];
}

function numberRecord(value: unknown): Record<string, number> {
  const parsed = jsonValue(value);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
  return Object.fromEntries(
    Object.entries(parsed).flatMap(([key, item]) => Number.isFinite(Number(item)) ? [[key, Number(item)]] : []),
  );
}

export function stableHash(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function splitSourceIds(value: unknown): string[] {
  const parts = Array.isArray(value) ? value : text(value).split("<SEP>");
  const seen = new Set<string>();
  return parts.map(text).filter(item => item.length > 0 && !seen.has(item) && Boolean(seen.add(item)));
}

export function cleanDescription(value: unknown): string {
  const result = text(value).split("<SEP>").map(part => part.trim()).filter(Boolean).join("\n");
  return result || "暂无可展示的描述。";
}

export function groupForType(value: unknown): number {
  return TYPE_GROUPS.get(text(value).toLowerCase()) ?? 5;
}

function graphFingerprint(nodes: GraphNode[], edges: GraphEdge[]): string {
  let hash = 2166136261;
  const feed = (value: number) => {
    hash ^= value;
    hash = Math.imul(hash, 16777619);
  };
  nodes.forEach(node => feed(stableHash(node.id)));
  edges.forEach(edge => {
    feed(edge.source);
    feed(edge.target);
  });
  return `${nodes.length}-${edges.length}-${(hash >>> 0).toString(16)}`;
}

export function buildCsr(nodeCount: number, edgePairs: Uint32Array): { offsets: Uint32Array; neighbors: Uint32Array } {
  const counts = new Uint32Array(nodeCount);
  for (let index = 0; index < edgePairs.length; index += 2) {
    const source = edgePairs[index];
    const target = edgePairs[index + 1];
    if (source === target) continue;
    counts[source] += 1;
    counts[target] += 1;
  }
  const offsets = new Uint32Array(nodeCount + 1);
  for (let index = 0; index < nodeCount; index += 1) offsets[index + 1] = offsets[index] + counts[index];
  const neighbors = new Uint32Array(offsets[nodeCount]);
  const cursor = offsets.slice(0, nodeCount);
  for (let index = 0; index < edgePairs.length; index += 2) {
    const source = edgePairs[index];
    const target = edgePairs[index + 1];
    if (source === target) continue;
    neighbors[cursor[source]++] = target;
    neighbors[cursor[target]++] = source;
  }
  return { offsets, neighbors };
}

export function neighborsOf(model: GraphModel, nodeIndex: number): Uint32Array {
  return model.csrNeighbors.subarray(model.csrOffsets[nodeIndex], model.csrOffsets[nodeIndex + 1]);
}

export function normalizeGraph(payload: RawGraphResponse): GraphModel {
  const rawNodes = Array.isArray(payload.nodes) ? payload.nodes : [];
  const rawEdges = Array.isArray(payload.edges) ? payload.edges : Array.isArray(payload.links) ? payload.links : [];
  const nodes: GraphNode[] = [];
  const nodeIndex = new Map<string, number>();

  rawNodes.forEach(raw => {
    const id = text(raw.id ?? raw.entity_id ?? raw.label);
    if (!id || nodeIndex.has(id)) return;
    const type = text(raw.type ?? raw.entity_type) || "unknown";
    const group = groupForType(type);
    nodeIndex.set(id, nodes.length);
    nodes.push({
      id,
      label: text(raw.label) || id,
      type,
      description: cleanDescription(raw.description),
      sourceId: text(raw.source_id),
      filePath: text(raw.file_path),
      degree: Math.max(0, Number(raw.degree) || 0),
      group,
      color: GROUP_COLORS[group],
      size: 4,
      isExpanded: String(raw.is_expanded ?? "").toLowerCase() === "true" || String(raw.is_expanded) === "1",
      isImageNode: String(raw.is_image_node ?? "").toLowerCase() === "true" || String(raw.is_image_node) === "1",
    });
  });

  const computedDegree = new Uint32Array(nodes.length);
  const edges: GraphEdge[] = [];
  rawEdges.forEach((raw: RawGraphEdge, rawIndex) => {
    const sourceId = endpoint(raw.source ?? raw.src_id);
    const targetId = endpoint(raw.target ?? raw.tgt_id);
    const source = nodeIndex.get(sourceId);
    const target = nodeIndex.get(targetId);
    if (source === undefined || target === undefined || source === target) return;
    computedDegree[source] += 1;
    computedDegree[target] += 1;
    const reviewStatus = text(raw.review_status).toLowerCase();
    const provenance = text(raw.provenance).toLowerCase();
    const algorithmVersion = text(raw.algorithm_version).toLowerCase();
    const isPromoted = truthy(raw.is_promoted) || (
      reviewStatus === "promoted" && provenance === "eclrr_v4" && algorithmVersion === "eclrr_v4"
    );
    const sourceIds = splitSourceIds(raw.source_id);
    const evidenceChunkIds = jsonArray<unknown>(raw.evidence_chunk_ids).map(text).filter(Boolean);
    const decisionScore = Number(raw.decision_score);
    edges.push({
      id: text(raw.id) || `edge-${rawIndex}`,
      source,
      target,
      label: text(raw.relation ?? raw.label) || "related",
      description: cleanDescription(raw.description),
      sourceId: text(raw.source_id),
      weight: Number(raw.weight) || 1,
      isDiscovered: truthy(raw.is_discovered),
      isPromoted,
      kind: isPromoted || text(raw.edge_kind).toLowerCase() === "eclrr_v4" ? "eclrr_v4" : "original",
      relationFamily: text(raw.relation_family),
      direction: text(raw.direction),
      relationId: text(raw.relation_id),
      canonicalKey: text(raw.canonical_key),
      pathUsed: jsonArray<unknown>(raw.path_used).map(text).filter(Boolean),
      supportingPaths: jsonArray(raw.supporting_paths),
      evidenceChain: jsonArray(raw.evidence_chain),
      evidenceChunkIds: evidenceChunkIds.length ? evidenceChunkIds : sourceIds,
      judgeScores: numberRecord(raw.judge_scores),
      decisionScore: Number.isFinite(decisionScore) ? decisionScore : null,
    });
  });

  const positions = new Float32Array(nodes.length * 3);
  const colors = new Float32Array(nodes.length * 3);
  const sizes = new Float32Array(nodes.length);
  const degrees = new Float32Array(nodes.length);
  const groups = new Uint8Array(nodes.length);
  const hashes = new Uint32Array(nodes.length);
  nodes.forEach((node, index) => {
    node.degree = Math.max(node.degree, computedDegree[index]);
    node.size = Math.min(12, 3.8 + Math.sqrt(node.degree + 1) * 0.72);
    positions[index * 3] = ((stableHash(node.id) & 0xffff) / 0xffff - 0.5) * 300;
    positions[index * 3 + 1] = (((stableHash(node.id) >>> 16) & 0xffff) / 0xffff - 0.5) * 300;
    positions[index * 3 + 2] = 0;
    colors.set(node.color, index * 3);
    sizes[index] = node.size;
    degrees[index] = node.degree;
    groups[index] = node.group;
    hashes[index] = stableHash(node.id);
  });

  const edgePairs = new Uint32Array(edges.length * 2);
  edges.forEach((edge, index) => {
    edgePairs[index * 2] = edge.source;
    edgePairs[index * 2 + 1] = edge.target;
  });
  const csr = buildCsr(nodes.length, edgePairs);
  return {
    nodes,
    edges,
    nodeIndex,
    positions,
    colors,
    sizes,
    degrees,
    groups,
    hashes,
    edgePairs,
    csrOffsets: csr.offsets,
    csrNeighbors: csr.neighbors,
    fingerprint: graphFingerprint(nodes, edges),
    metadata: payload.metadata ?? {},
  };
}
