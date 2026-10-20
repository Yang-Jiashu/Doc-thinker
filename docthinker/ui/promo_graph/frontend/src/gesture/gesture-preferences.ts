export type GestureMode = "inspect" | "presentation";
export type GestureDominantHand = "left" | "right" | "auto";

export interface GesturePreferences {
  mode: GestureMode;
  dominantHand: GestureDominantHand;
  smoothingMs: number;
  dwellMs: number;
  resetHoldMs: number;
  activationMs: number;
  movementThresholdPx: number;
  pinchEnterRatio: number;
  pinchReleaseRatio: number;
  panGain: number;
  zoomGain: number;
  orbitGain: number;
}

export const DEFAULT_GESTURE_PREFERENCES: Readonly<GesturePreferences> = Object.freeze({
  mode: "inspect",
  dominantHand: "right",
  smoothingMs: 70,
  dwellMs: 650,
  resetHoldMs: 1_000,
  activationMs: 180,
  movementThresholdPx: 5,
  pinchEnterRatio: 0.24,
  pinchReleaseRatio: 0.34,
  panGain: 1,
  zoomGain: 1,
  orbitGain: 1,
});

/** Pure validation shared by both views. No storage or device access here. */
export function normalizeGesturePreferences(value: unknown): GesturePreferences {
  const data = value && typeof value === "object" ? value as Record<string, unknown> : {};
  const number = (key: keyof GesturePreferences, min: number, max: number): number => {
    const candidate = data[key];
    return typeof candidate === "number" && Number.isFinite(candidate)
      ? Math.min(max, Math.max(min, candidate))
      : DEFAULT_GESTURE_PREFERENCES[key] as number;
  };
  const pinchEnterRatio = number("pinchEnterRatio", 0.1, 0.4);
  return {
    mode: data.mode === "presentation" ? "presentation" : "inspect",
    dominantHand: data.dominantHand === "left" || data.dominantHand === "auto" ? data.dominantHand : "right",
    smoothingMs: number("smoothingMs", 0, 250),
    dwellMs: number("dwellMs", 250, 2_000),
    resetHoldMs: number("resetHoldMs", 500, 3_000),
    activationMs: number("activationMs", 80, 600),
    movementThresholdPx: number("movementThresholdPx", 2, 30),
    pinchEnterRatio,
    pinchReleaseRatio: Math.max(pinchEnterRatio + 0.05, number("pinchReleaseRatio", 0.15, 0.6)),
    panGain: number("panGain", 0.25, 3),
    zoomGain: number("zoomGain", 0.25, 3),
    orbitGain: number("orbitGain", 0.25, 3),
  };
}
