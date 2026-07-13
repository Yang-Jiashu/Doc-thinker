import { describe, expect, it } from "vitest";
import {
  labelFractionForLevel,
  resolveLabelCollisions,
  SemanticZoomPolicy,
  zoomLevelForRatio,
  type LabelBoxCandidate,
} from "../src/semantic-labels";

describe("semantic zoom labels", () => {
  it("uses the specified reveal levels", () => {
    expect(zoomLevelForRatio(0.5)).toBe(0);
    expect(zoomLevelForRatio(1)).toBe(1);
    expect(zoomLevelForRatio(2)).toBe(2);
    expect(zoomLevelForRatio(4)).toBe(3);
    expect(zoomLevelForRatio(5)).toBe(4);
    expect([0, 1, 2, 3, 4].map(labelFractionForLevel)).toEqual([0, 0.03, 0.15, 0.5, 1]);
  });

  it("holds its current level inside the hysteresis band", () => {
    const policy = new SemanticZoomPolicy();
    policy.reset(1);
    expect(policy.update(1.75)).toBe(1);
    expect(policy.update(1.82)).toBe(2);
    expect(policy.update(1.45)).toBe(2);
    expect(policy.update(1.39)).toBe(1);
  });

  it("keeps accepted labels within the viewport without overlap", () => {
    const candidates: LabelBoxCandidate[] = Array.from({ length: 24 }, (_, index) => ({
      nodeIndex: index,
      text: `节点 ${index}`,
      anchorX: 40 + index % 6 * 88,
      anchorY: 38 + Math.floor(index / 6) * 72,
      nodeRadius: 7,
      priority: 100 - index,
      color: [1, 1, 1],
      forced: index === 0,
    }));
    const labels = resolveLabelCollisions(candidates, 620, 400);
    labels.forEach(label => {
      const left = label.anchorX + label.offsetX;
      const top = label.anchorY + label.offsetY;
      expect(left).toBeGreaterThanOrEqual(8);
      expect(top).toBeGreaterThanOrEqual(8);
      expect(left + label.width).toBeLessThanOrEqual(612);
      expect(top + label.height).toBeLessThanOrEqual(392);
    });
    for (let a = 0; a < labels.length; a += 1) {
      for (let b = a + 1; b < labels.length; b += 1) {
        const first = labels[a];
        const second = labels[b];
        const firstLeft = first.anchorX + first.offsetX;
        const firstTop = first.anchorY + first.offsetY;
        const secondLeft = second.anchorX + second.offsetX;
        const secondTop = second.anchorY + second.offsetY;
        const overlapX = Math.min(firstLeft + first.width, secondLeft + second.width) - Math.max(firstLeft, secondLeft);
        const overlapY = Math.min(firstTop + first.height, secondTop + second.height) - Math.max(firstTop, secondTop);
        expect(overlapX <= 0 || overlapY <= 0).toBe(true);
      }
    }
  });
});
