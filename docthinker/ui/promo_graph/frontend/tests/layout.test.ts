import { describe, expect, it } from "vitest";
import { computeLocalFocusLayout } from "../src/layout-algorithms";
import { isLayoutCompatible } from "../src/layout-cache";

describe("layout behavior", () => {
  it("keeps the selected node anchored and pulls direct neighbors nearby", () => {
    const positions = Float32Array.from([
      0, 0, 0,
      500, 0, 0,
      -500, 0, 0,
      0, 700, 0,
    ]);
    const adjacency = [Uint32Array.from([1, 2]), Uint32Array.from([0]), Uint32Array.from([0]), Uint32Array.from([])];
    const result = computeLocalFocusLayout(positions, adjacency, Uint32Array.from([0, 1, 0, 2]), 0);
    expect([...result.indices]).toEqual([0, 1, 2]);
    expect(result.positions[0]).toBeCloseTo(0, 5);
    expect(result.positions[1]).toBeCloseTo(0, 5);
    expect(Math.hypot(result.positions[2], result.positions[3])).toBeLessThan(220);
    expect(Math.hypot(result.positions[4], result.positions[5])).toBeLessThan(220);
  });

  it("rejects stale cache buffers with a different node count", () => {
    expect(isLayoutCompatible(new Float32Array(30), 10)).toBe(true);
    expect(isLayoutCompatible(new Float32Array(27), 10)).toBe(false);
    expect(isLayoutCompatible(null, 10)).toBe(false);
  });
});
