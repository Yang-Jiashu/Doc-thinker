/// <reference lib="webworker" />

import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import { UndirectedGraph } from "graphology";
import louvain from "graphology-communities-louvain";
import forceAtlas2 from "graphology-layout-forceatlas2";
import { computeLocalFocusLayout } from "./layout-algorithms";

interface LayoutRequest {
  type: "layout" | "hydrate";
  requestId: number;
  nodeCount: number;
  hashes: Uint32Array;
  degrees: Float32Array;
  groups: Uint8Array;
  edgePairs: Uint32Array;
  positions?: Float32Array;
}

interface FocusRequest {
  type: "focus";
  requestId: number;
  nodeIndex: number;
}

type WorkerRequest = LayoutRequest | FocusRequest;

interface ForceNode extends SimulationNodeDatum {
  id: number;
  radius: number;
}

interface ForceEdge extends SimulationLinkDatum<ForceNode> {
  source: number | ForceNode;
  target: number | ForceNode;
}

let basePositions: Float32Array<ArrayBufferLike> = new Float32Array();
let adjacency: Uint32Array[] = [];
let edgePairs = new Uint32Array();

function seededPosition(hash: number, group: number, count: number): [number, number] {
  const angle = (hash / 0xffffffff) * Math.PI * 2 + group * 0.73;
  const radius = 35 + Math.sqrt(Math.max(1, count)) * 10 * (((hash >>> 8) & 0xff) / 255 + 0.35);
  const groupAngle = group / 6 * Math.PI * 2;
  const groupRadius = Math.min(640, Math.sqrt(count) * 20);
  return [
    Math.cos(groupAngle) * groupRadius + Math.cos(angle) * radius,
    Math.sin(groupAngle) * groupRadius + Math.sin(angle) * radius,
  ];
}

function buildAdjacency(nodeCount: number, pairs: Uint32Array): Uint32Array[] {
  const buckets: number[][] = Array.from({ length: nodeCount }, () => []);
  for (let index = 0; index < pairs.length; index += 2) {
    const source = pairs[index];
    const target = pairs[index + 1];
    if (source === target) continue;
    buckets[source].push(target);
    buckets[target].push(source);
  }
  return buckets.map(values => Uint32Array.from(values));
}

function centerAndScale(positions: Float32Array): Float32Array {
  const count = positions.length / 3;
  if (!count) return positions;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < count; index += 1) {
    minX = Math.min(minX, positions[index * 3]);
    maxX = Math.max(maxX, positions[index * 3]);
    minY = Math.min(minY, positions[index * 3 + 1]);
    maxY = Math.max(maxY, positions[index * 3 + 1]);
  }
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const targetSpan = Math.max(520, Math.sqrt(count) * 82);
  const scale = targetSpan / Math.max(1, maxX - minX, maxY - minY);
  for (let index = 0; index < count; index += 1) {
    positions[index * 3] = (positions[index * 3] - centerX) * scale;
    positions[index * 3 + 1] = (positions[index * 3 + 1] - centerY) * scale;
    positions[index * 3 + 2] = 0;
  }
  return positions;
}

function runD3Layout(request: LayoutRequest): Float32Array {
  const nodes: ForceNode[] = Array.from({ length: request.nodeCount }, (_, index) => {
    const [x, y] = seededPosition(request.hashes[index], request.groups[index], request.nodeCount);
    return { id: index, x, y, radius: 4 + Math.sqrt(request.degrees[index] + 1) * 0.45 };
  });
  const links: ForceEdge[] = [];
  for (let index = 0; index < request.edgePairs.length; index += 2) {
    links.push({ source: request.edgePairs[index], target: request.edgePairs[index + 1] });
  }
  const simulation = forceSimulation(nodes)
    .alpha(1)
    .alphaDecay(0.028)
    .velocityDecay(0.38)
    .force("link", forceLink<ForceNode, ForceEdge>(links).id(node => node.id).distance(58).strength(0.22))
    .force("charge", forceManyBody().strength(node => -32 - (node as ForceNode).radius * 4).theta(0.92))
    .force("collision", forceCollide<ForceNode>().radius(node => node.radius + 5).strength(0.82))
    .force("center", forceCenter(0, 0))
    .stop();
  const iterations = Math.min(420, 180 + Math.ceil(Math.sqrt(request.nodeCount) * 4));
  for (let index = 0; index < iterations; index += 1) simulation.tick();
  const positions = new Float32Array(request.nodeCount * 3);
  nodes.forEach((node, index) => {
    positions[index * 3] = node.x ?? 0;
    positions[index * 3 + 1] = node.y ?? 0;
  });
  return centerAndScale(positions);
}

function runForceAtlasLayout(request: LayoutRequest): Float32Array {
  const graph = new UndirectedGraph();
  for (let index = 0; index < request.nodeCount; index += 1) {
    const [x, y] = seededPosition(request.hashes[index], request.groups[index], request.nodeCount);
    graph.addNode(String(index), {
      x,
      y,
      size: 1 + Math.sqrt(request.degrees[index] + 1) * 0.2,
      group: request.groups[index],
    });
  }
  for (let index = 0; index < request.edgePairs.length; index += 2) {
    const source = String(request.edgePairs[index]);
    const target = String(request.edgePairs[index + 1]);
    if (source !== target && !graph.hasEdge(source, target)) graph.addUndirectedEdge(source, target, { weight: 1 });
  }

  if (graph.size > 0) {
    louvain.assign(graph, { getEdgeWeight: "weight", resolution: 1 });
    const communities = new Map<number, number>();
    graph.forEachNode((node, attributes) => {
      const community = Number(attributes.community ?? attributes.group ?? 0);
      if (!communities.has(community)) communities.set(community, communities.size);
      const rank = communities.get(community) ?? 0;
      const angle = rank * 2.399963229728653;
      const radius = 90 * Math.sqrt(rank);
      const jitter = seededPosition(request.hashes[Number(node)], request.groups[Number(node)], Math.max(20, request.nodeCount / Math.max(1, communities.size)));
      graph.setNodeAttribute(node, "x", Math.cos(angle) * radius + jitter[0] * 0.28);
      graph.setNodeAttribute(node, "y", Math.sin(angle) * radius + jitter[1] * 0.28);
    });
  }

  const inferred = forceAtlas2.inferSettings(graph);
  const iterations = request.nodeCount > 10_000 ? 45 : request.nodeCount > 5_000 ? 75 : 120;
  for (let completed = 0; completed < iterations; completed += 10) {
    forceAtlas2.assign(graph, {
      iterations: Math.min(10, iterations - completed),
      settings: {
        ...inferred,
        barnesHutOptimize: true,
        barnesHutTheta: 0.72,
        gravity: 0.08,
        scalingRatio: request.nodeCount > 10_000 ? 8 : 5,
        slowDown: 2,
      },
    });
  }

  const positions = new Float32Array(request.nodeCount * 3);
  graph.forEachNode((node, attributes) => {
    const index = Number(node);
    positions[index * 3] = Number(attributes.x) || 0;
    positions[index * 3 + 1] = Number(attributes.y) || 0;
  });
  return centerAndScale(positions);
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === "hydrate") {
    basePositions = request.positions?.slice() ?? new Float32Array(request.nodeCount * 3);
    edgePairs = request.edgePairs.slice();
    adjacency = buildAdjacency(request.nodeCount, edgePairs);
    return;
  }
  if (request.type === "layout") {
    edgePairs = request.edgePairs.slice();
    adjacency = buildAdjacency(request.nodeCount, edgePairs);
    basePositions = request.nodeCount < 2_000 ? runD3Layout(request) : runForceAtlasLayout(request);
    const positions = basePositions.slice();
    self.postMessage({ type: "layout", requestId: request.requestId, positions }, [positions.buffer]);
    return;
  }
  if (request.type !== "focus") return;
  const result = computeLocalFocusLayout(basePositions, adjacency, edgePairs, request.nodeIndex);
  self.postMessage(
    { type: "focus", requestId: request.requestId, ...result },
    [result.indices.buffer, result.positions.buffer],
  );
};

export {};
