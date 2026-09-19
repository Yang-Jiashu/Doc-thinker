import { describe, expect, it } from "vitest";
import {
  extractGestureMetrics,
  resolveGestureHandedness,
  smoothGesturePoint,
  toScreenMetrics,
  type LandmarkLike,
} from "../src/gesture/gesture-metrics";

function handLandmarks(): LandmarkLike[] {
  const points: LandmarkLike[] = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.7, z: 0 }));
  points[0] = { x: 0.5, y: 0.9 };
  points[4] = { x: 0.28, y: 0.48 };
  points[5] = { x: 0.4, y: 0.68 };
  points[6] = { x: 0.4, y: 0.55 };
  points[8] = { x: 0.4, y: 0.28 };
  points[9] = { x: 0.5, y: 0.66 };
  points[10] = { x: 0.5, y: 0.54 };
  points[12] = { x: 0.5, y: 0.25 };
  points[13] = { x: 0.6, y: 0.68 };
  points[14] = { x: 0.6, y: 0.56 };
  points[16] = { x: 0.6, y: 0.3 };
  points[17] = { x: 0.68, y: 0.72 };
  points[18] = { x: 0.68, y: 0.58 };
  points[20] = { x: 0.68, y: 0.34 };
  return points;
}

describe("gesture metrics", () => {
  it("detects an open palm and mirrors screen coordinates", () => {
    const metrics = extractGestureMetrics(handLandmarks());
    expect(metrics?.openPalm).toBe(true);
    expect(metrics?.extendedFingers).toBe(4);
    const screen = toScreenMetrics(metrics!, 1_000, 500);
    expect(screen.point.x).toBeCloseTo(600);
    expect(screen.point.y).toBeCloseTo(140);
  });

  it("uses hysteresis for a stable pinch", () => {
    const landmarks = handLandmarks();
    landmarks[4] = { x: 0.37, y: 0.29 };
    const entered = extractGestureMetrics(landmarks, false);
    expect(entered?.pinch).toBe(true);
    landmarks[4] = { x: 0.34, y: 0.29 };
    expect(extractGestureMetrics(landmarks, false)?.pinch).toBe(false);
    expect(extractGestureMetrics(landmarks, true)?.pinch).toBe(true);
  });

  it("smooths fingertip movement deterministically", () => {
    expect(smoothGesturePoint({ x: 0, y: 10 }, { x: 10, y: 20 }, 0.25)).toEqual({ x: 2.5, y: 12.5 });
  });

  it("uses model handedness and a mirrored-position fallback", () => {
    expect(resolveGestureHandedness("Left", 0.2)).toBe("left");
    expect(resolveGestureHandedness("Right", 0.8)).toBe("right");
    expect(resolveGestureHandedness(undefined, 0.75)).toBe("left");
    expect(resolveGestureHandedness(undefined, 0.25)).toBe("right");
  });
});
