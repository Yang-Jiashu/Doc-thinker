import { describe, expect, it } from "vitest";
import { computeVisualDepths } from "../src/visual-depth";

describe("render-only visual depth", () => {
  it("is deterministic without modifying base positions", () => {
    const positions = new Float32Array([
      -20, 0, 0,
      0, 0, 0,
      20, 0, 0,
      0, 12, 0,
    ]);
    const original = positions.slice();
    const hashes = new Uint32Array([2, 3, 4, 5]);
    const first = computeVisualDepths(positions, hashes);
    const second = computeVisualDepths(positions, hashes);

    expect([...positions]).toEqual([...original]);
    expect([...first.depths]).toEqual([...second.depths]);
    expect(first.maximumDepth).toBeGreaterThan(0);
    expect(first.depths.some(depth => Math.abs(depth) > 0.01)).toBe(true);
    expect(Math.max(...first.depths.map(Math.abs))).toBeLessThanOrEqual(first.maximumDepth);
  });
});
