import { describe, expect, it } from "vitest";
import { DEFAULT_GESTURE_PREFERENCES, normalizeGesturePreferences } from "../src/gesture/gesture-preferences";

describe("gesture preferences", () => {
  it("uses precise inspection by default without mutating shared defaults", () => {
    expect(normalizeGesturePreferences(null)).toEqual(DEFAULT_GESTURE_PREFERENCES);
    const preferences = normalizeGesturePreferences({});
    preferences.mode = "presentation";
    expect(DEFAULT_GESTURE_PREFERENCES.mode).toBe("inspect");
    expect(Object.isFrozen(DEFAULT_GESTURE_PREFERENCES)).toBe(true);
  });

  it("clamps controls and refuses nonfinite/string values from persisted input", () => {
    const preferences = normalizeGesturePreferences({mode: "bad", dominantHand: "bad", dwellMs: "10", panGain: Infinity,
      zoomGain: -20, smoothingMs: 1_000, resetHoldMs: 0, movementThresholdPx: NaN});
    expect(preferences.mode).toBe("inspect"); expect(preferences.dominantHand).toBe("right");
    expect(preferences.dwellMs).toBe(DEFAULT_GESTURE_PREFERENCES.dwellMs);
    expect(preferences.panGain).toBe(1); expect(preferences.zoomGain).toBe(0.25);
    expect(preferences.smoothingMs).toBe(250); expect(preferences.resetHoldMs).toBe(500);
    expect(preferences.movementThresholdPx).toBe(DEFAULT_GESTURE_PREFERENCES.movementThresholdPx);
  });

  it("always preserves a hysteresis gap between pinch entry and release", () => {
    const preferences = normalizeGesturePreferences({pinchEnterRatio: 0.4, pinchReleaseRatio: 0.15, dominantHand: "left"});
    expect(preferences.pinchReleaseRatio).toBeCloseTo(0.45);
    expect(preferences.dominantHand).toBe("left");
    expect(normalizeGesturePreferences({dominantHand: "auto"}).dominantHand).toBe("auto");
  });
});
