import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";

interface LocalNode extends SimulationNodeDatum {
  id: number;
  radius: number;
}

interface LocalEdge extends SimulationLinkDatum<LocalNode> {
  source: number | LocalNode;
  target: number | LocalNode;
}

export interface LocalFocusResult {
  indices: Uint32Array;
  positions: Float32Array;
}

export function computeLocalFocusLayout(
  basePositions: Float32Array<ArrayBufferLike>,
  adjacency: Uint32Array[],
  edgePairs: Uint32Array,
  nodeIndex: number,
): LocalFocusResult {
  const direct = adjacency[nodeIndex] ?? new Uint32Array();
  const unique = Array.from(new Set<number>([nodeIndex, ...direct]));
  const selectedX = basePositions[nodeIndex * 3];
  const selectedY = basePositions[nodeIndex * 3 + 1];
  if (unique.length > 600) {
    const indices = Uint32Array.from(unique);
    const positions = new Float32Array(unique.length * 2);
    unique.forEach((index, order) => {
      if (index === nodeIndex) {
        positions[order * 2] = selectedX;
        positions[order * 2 + 1] = selectedY;
        return;
      }
      const angle = order / Math.max(1, unique.length - 1) * Math.PI * 2;
      const radius = 85 + Math.sqrt(unique.length) * 8;
      positions[order * 2] = selectedX + Math.cos(angle) * radius;
      positions[order * 2 + 1] = selectedY + Math.sin(angle) * radius;
    });
    return { indices, positions };
  }

  const nodes: LocalNode[] = unique.map(index => ({
    id: index,
    x: basePositions[index * 3],
    y: basePositions[index * 3 + 1],
    radius: index === nodeIndex ? 15 : 8,
    fx: index === nodeIndex ? selectedX : undefined,
    fy: index === nodeIndex ? selectedY : undefined,
  }));
  const localSet = new Set(unique);
  const links: LocalEdge[] = [];
  for (let index = 0; index < edgePairs.length; index += 2) {
    const source = edgePairs[index];
    const target = edgePairs[index + 1];
    if (localSet.has(source) && localSet.has(target)) links.push({ source, target });
  }
  const simulation = forceSimulation(nodes)
    .alpha(0.9)
    .velocityDecay(0.38)
    .force("link", forceLink<LocalNode, LocalEdge>(links).id(node => node.id).distance(72).strength(0.72))
    .force("charge", forceManyBody().strength(-115))
    .force("collision", forceCollide<LocalNode>().radius(node => node.radius + 8).strength(1))
    .force("x", forceX(selectedX).strength(0.025))
    .force("y", forceY(selectedY).strength(0.025))
    .stop();
  for (let index = 0; index < 100; index += 1) simulation.tick();
  const indices = Uint32Array.from(unique);
  const positions = new Float32Array(unique.length * 2);
  nodes.forEach((node, index) => {
    positions[index * 2] = node.x ?? selectedX;
    positions[index * 2 + 1] = node.y ?? selectedY;
  });
  return { indices, positions };
}
