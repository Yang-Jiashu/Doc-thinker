import type { GraphModel, LabelPlacement } from "./types";

const LEVEL_THRESHOLDS = [0.85, 1.6, 3, 5];
const LEVEL_FRACTIONS = [0, 0.03, 0.15, 0.5, 1];
const HYSTERESIS = 0.12;

export interface ScreenNode {
  nodeIndex: number;
  x: number;
  y: number;
  visible: boolean;
}

export interface LabelBoxCandidate {
  nodeIndex: number;
  text: string;
  anchorX: number;
  anchorY: number;
  nodeRadius: number;
  priority: number;
  color: [number, number, number];
  forced: boolean;
}

export class SemanticZoomPolicy {
  private level = 1;

  update(zoomRatio: number): number {
    while (
      this.level < LEVEL_THRESHOLDS.length
      && zoomRatio >= LEVEL_THRESHOLDS[this.level] * (1 + HYSTERESIS)
    ) this.level += 1;
    while (
      this.level > 0
      && zoomRatio < LEVEL_THRESHOLDS[this.level - 1] * (1 - HYSTERESIS)
    ) this.level -= 1;
    return this.level;
  }

  reset(zoomRatio = 1): number {
    this.level = zoomLevelForRatio(zoomRatio);
    return this.level;
  }

  get currentLevel(): number {
    return this.level;
  }
}

export function zoomLevelForRatio(zoomRatio: number): number {
  if (zoomRatio < LEVEL_THRESHOLDS[0]) return 0;
  if (zoomRatio < LEVEL_THRESHOLDS[1]) return 1;
  if (zoomRatio < LEVEL_THRESHOLDS[2]) return 2;
  if (zoomRatio < LEVEL_THRESHOLDS[3]) return 3;
  return 4;
}

export function labelFractionForLevel(level: number): number {
  return LEVEL_FRACTIONS[Math.max(0, Math.min(LEVEL_FRACTIONS.length - 1, level))];
}

function typePriority(group: number): number {
  return [6, 4, 3, 5, 2, 1][group] ?? 0;
}

export function selectLabelCandidates(
  model: GraphModel,
  screenNodes: ScreenNode[],
  level: number,
  special: Set<number>,
): LabelBoxCandidate[] {
  const visible = screenNodes.filter(node => node.visible);
  const fraction = labelFractionForLevel(level);
  const ordinaryCount = level === 0 ? 0 : Math.max(1, Math.ceil(visible.length * fraction));
  return visible
    .map(screen => {
      const node = model.nodes[screen.nodeIndex];
      const forced = special.has(screen.nodeIndex);
      return {
        nodeIndex: screen.nodeIndex,
        text: node.label,
        anchorX: screen.x,
        anchorY: screen.y,
        nodeRadius: node.size + 3,
        priority: (forced ? 1_000_000 : 0) + node.degree * 100 + typePriority(node.group) * 10 - screen.nodeIndex * 1e-6,
        color: forced ? [0.78, 1, 0.98] as [number, number, number] : [0.88, 0.93, 0.98] as [number, number, number],
        forced,
      };
    })
    .sort((a, b) => b.priority - a.priority)
    .filter((candidate, index) => candidate.forced || index < ordinaryCount);
}

export function estimateLabelWidth(value: string, fontSize = 13): number {
  let units = 0;
  for (const character of value) units += character.charCodeAt(0) <= 0x7f ? 0.56 : 1;
  return Math.max(18, units * fontSize + 4);
}

function overlaps(a: LabelPlacement, b: LabelPlacement): boolean {
  const ax = a.anchorX + a.offsetX;
  const ay = a.anchorY + a.offsetY;
  const bx = b.anchorX + b.offsetX;
  const by = b.anchorY + b.offsetY;
  return ax < bx + b.width && ax + a.width > bx && ay < by + b.height && ay + a.height > by;
}

export function resolveLabelCollisions(
  candidates: LabelBoxCandidate[],
  width: number,
  height: number,
): LabelPlacement[] {
  const accepted: LabelPlacement[] = [];
  const margin = 8;
  const offsets = (radius: number, labelWidth: number, labelHeight: number): Array<[number, number]> => [
    [radius, -labelHeight / 2],
    [-labelWidth - radius, -labelHeight / 2],
    [-labelWidth / 2, -radius - labelHeight],
    [-labelWidth / 2, radius],
    [radius, -radius - labelHeight],
    [-labelWidth - radius, -radius - labelHeight],
    [radius, radius],
    [-labelWidth - radius, radius],
  ];

  candidates.forEach(candidate => {
    const labelWidth = estimateLabelWidth(candidate.text);
    const labelHeight = 18;
    let placement: LabelPlacement | null = null;
    for (const [offsetX, offsetY] of offsets(candidate.nodeRadius, labelWidth, labelHeight)) {
      const next: LabelPlacement = {
        ...candidate,
        offsetX,
        offsetY,
        width: labelWidth,
        height: labelHeight,
        opacity: 1,
      };
      const left = next.anchorX + next.offsetX;
      const top = next.anchorY + next.offsetY;
      if (left < margin || top < margin || left + labelWidth > width - margin || top + labelHeight > height - margin) continue;
      if (!accepted.some(item => overlaps(next, item))) {
        placement = next;
        break;
      }
    }
    if (!placement && candidate.forced) {
      placement = {
        ...candidate,
        offsetX: Math.max(margin, Math.min(width - labelWidth - margin, candidate.anchorX + candidate.nodeRadius)) - candidate.anchorX,
        offsetY: Math.max(margin, Math.min(height - labelHeight - margin, candidate.anchorY - labelHeight / 2)) - candidate.anchorY,
        width: labelWidth,
        height: labelHeight,
        opacity: 1,
      };
    }
    if (placement) accepted.push(placement);
  });
  return accepted;
}
