export interface LandmarkLike {
  x: number;
  y: number;
  z?: number;
}

export interface GesturePoint {
  x: number;
  y: number;
}

export type GestureHandedness = "left" | "right";

export interface NormalizedGestureMetrics {
  point: GesturePoint;
  thumbPoint: GesturePoint;
  palmPoint: GesturePoint;
  pinch: boolean;
  pinchDistance: number;
  openPalm: boolean;
  extendedFingers: number;
}

export interface ScreenGestureMetrics extends Omit<NormalizedGestureMetrics, "point" | "thumbPoint" | "palmPoint"> {
  handedness: GestureHandedness;
  point: GesturePoint;
  thumbPoint: GesturePoint;
  palmPoint: GesturePoint;
}

export function resolveGestureHandedness(
  categoryName: string | undefined,
  mirroredX: number,
): GestureHandedness {
  const normalized = String(categoryName ?? "").trim().toLowerCase();
  if (normalized === "left") return "left";
  if (normalized === "right") return "right";
  return mirroredX >= 0.5 ? "left" : "right";
}

export function landmarkDistance(a: LandmarkLike, b: LandmarkLike): number {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));
}

export function extractGestureMetrics(
  landmarks: LandmarkLike[],
  previousPinch = false,
): NormalizedGestureMetrics | null {
  if (landmarks.length < 21) return null;
  const wrist = landmarks[0];
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const palmSize = Math.max(landmarkDistance(wrist, landmarks[9]), 0.035);
  const pinchDistance = landmarkDistance(thumbTip, indexTip) / palmSize;
  const pinch = previousPinch ? pinchDistance < 0.34 : pinchDistance < 0.24;
  const fingerTipIds = [8, 12, 16, 20];
  const fingerPipIds = [6, 10, 14, 18];
  const extendedFingers = fingerTipIds.reduce((count, tipId, index) => {
    const extended = landmarkDistance(landmarks[tipId], wrist) > landmarkDistance(landmarks[fingerPipIds[index]], wrist) * 1.08;
    return count + Number(extended);
  }, 0);
  const palmPoint = [0, 5, 9, 13, 17].reduce((point, landmarkIndex) => ({
    x: point.x + landmarks[landmarkIndex].x / 5,
    y: point.y + landmarks[landmarkIndex].y / 5,
  }), { x: 0, y: 0 });

  return {
    point: { x: 1 - indexTip.x, y: indexTip.y },
    thumbPoint: { x: 1 - thumbTip.x, y: thumbTip.y },
    palmPoint: { x: 1 - palmPoint.x, y: palmPoint.y },
    pinch,
    pinchDistance,
    openPalm: !pinch && extendedFingers >= 4,
    extendedFingers,
  };
}

export function toScreenMetrics(
  metrics: NormalizedGestureMetrics,
  width: number,
  height: number,
  handedness: GestureHandedness = "right",
): ScreenGestureMetrics {
  const scale = (point: GesturePoint): GesturePoint => ({ x: point.x * width, y: point.y * height });
  return {
    ...metrics,
    handedness,
    point: scale(metrics.point),
    thumbPoint: scale(metrics.thumbPoint),
    palmPoint: scale(metrics.palmPoint),
  };
}

export function smoothGesturePoint(
  previous: GesturePoint | null,
  next: GesturePoint,
  alpha = 0.34,
): GesturePoint {
  if (!previous) return { ...next };
  return {
    x: previous.x * (1 - alpha) + next.x * alpha,
    y: previous.y * (1 - alpha) + next.y * alpha,
  };
}
