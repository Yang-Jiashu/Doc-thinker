import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

function staticAssetUrl(relativePath: string): string {
  const base = import.meta.url.slice(0, import.meta.url.lastIndexOf("/") + 1);
  return new URL(relativePath, base).href;
}

export async function createHandLandmarker(): Promise<HandLandmarker> {
  const wasmRoot = staticAssetUrl("mediapipe/wasm").replace(/\/$/, "");
  const modelAssetPath = staticAssetUrl("mediapipe/hand_landmarker.task");
  const vision = await FilesetResolver.forVisionTasks(wasmRoot);
  const options = {
    baseOptions: { modelAssetPath, delegate: "GPU" as const },
    runningMode: "VIDEO" as const,
    numHands: 2,
    minHandDetectionConfidence: 0.68,
    minHandPresenceConfidence: 0.68,
    minTrackingConfidence: 0.68,
  };
  try {
    return await HandLandmarker.createFromOptions(vision, options);
  } catch {
    return await HandLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath, delegate: "CPU" },
    });
  }
}
