import type { Category, HandLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { StarMapRenderer } from "../star-map-renderer";
import {
  extractGestureMetrics,
  resolveGestureHandedness,
  toScreenMetrics,
  type ScreenGestureMetrics,
} from "./gesture-metrics";
import { createHandLandmarker } from "./mediapipe-adapter";
import { GestureStateMachine, type GestureInputPort } from "./gesture-state-machine";
import type { GesturePreferences } from "./gesture-preferences";

/** Both the GPU view and the standard D3 view implement this small input surface. */
export type GestureRenderer = Pick<StarMapRenderer,
  "pick" | "setHovered" | "fitToGraph" | "orbitBy" | "panBy" | "zoomAt" |
  "getNodeWorldPosition" | "screenToWorld" | "setNodeTransientPosition">;

export interface GestureControllerState {
  enabled: boolean;
  suspended: boolean;
  starting: boolean;
  layerState: string;
  error: string;
}

/** Mouse/touch owns the next interaction; releasing it never re-enables capture. */
export function pauseGestureForPointer(
  controller: Pick<GestureController, "getState" | "setSuspended">,
  pointerDown: boolean,
): boolean {
  return pointerDown && pauseGestureForManualInput(controller);
}

/** Wheel, keyboard and toolbar actions take the same ownership as a pointer. */
export function pauseGestureForManualInput(
  controller: Pick<GestureController, "getState" | "setSuspended">,
): boolean {
  const state = controller.getState();
  if (state.suspended || (!state.enabled && !state.starting)) return false;
  controller.setSuspended(true);
  return true;
}

export interface GestureControllerCallbacks {
  onSelect: (nodeIndex: number) => void;
  onClearSelection: () => void;
  onNodeAttractionStart: (nodeIndex: number) => void;
  onNodeAttractionEnd: () => void;
  onHover: (nodeIndex: number) => void;
  onNodeMoved: (nodeIndex: number) => void;
  onGestureActiveChange: (active: boolean) => void;
  onNodeDraggingChange: (active: boolean) => void;
  onStateChange?: (state: GestureControllerState) => void;
}

export class GestureController {
  private stage: HTMLElement;
  private layer: HTMLElement;
  private video: HTMLVideoElement;
  private cursor: HTMLElement;
  private renderer: GestureRenderer;
  private callbacks: GestureControllerCallbacks;
  private stateMachine: GestureStateMachine;
  private landmarker: HandLandmarker | null = null;
  private stream: MediaStream | null = null;
  private enabled = false;
  private suspended = false;
  private starting = false;
  private startPromise: Promise<boolean> | null = null;
  private generation = 0;
  private disposed = false;
  private error = "";
  private lastInferenceAt = 0;
  private lastVideoTime = -1;
  private lastFrameAt: number | null = null;
  private previousPinch: Record<"left" | "right", boolean> = { left: false, right: false };
  private dragAnchors = new Map<number, { x: number; y: number; z: number }>();

  constructor(
    stage: HTMLElement,
    layer: HTMLElement,
    video: HTMLVideoElement,
    cursor: HTMLElement,
    renderer: GestureRenderer,
    callbacks: GestureControllerCallbacks,
    preferences: Partial<GesturePreferences> = {},
  ) {
    this.stage = stage;
    this.layer = layer;
    this.video = video;
    this.cursor = cursor;
    this.renderer = renderer;
    this.callbacks = callbacks;
    this.stateMachine = new GestureStateMachine(this.createInputPort(), preferences);
  }

  async start(): Promise<boolean> {
    if (this.disposed) return false;
    if (this.enabled) return true;
    if (this.startPromise) return this.startPromise;
    if (!navigator.mediaDevices?.getUserMedia) {
      this.layer.dataset.state = "unavailable";
      this.error = "当前浏览器无法使用摄像头，请使用 HTTPS 或本机访问；鼠标操作仍可用。";
      this.notifyState();
      return false;
    }
    const generation = ++this.generation;
    this.starting = true;
    this.error = "";
    this.layer.dataset.state = "starting";
    this.notifyState();
    const pending = this.startCamera(generation);
    this.startPromise = pending;
    try {
      return await pending;
    } finally {
      if (this.startPromise === pending) this.startPromise = null;
    }
  }

  private async startCamera(generation: number): Promise<boolean> {
    let stream: MediaStream | null = null;
    let landmarker: HandLandmarker | null = null;
    let landmarkerHandedOff = false;
    const current = () => generation === this.generation && !this.disposed;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, max: 30 },
        },
      });
      if (!current()) return false;
      this.stream = stream;
      landmarker = await createHandLandmarker();
      if (!current()) return false;
      this.landmarker = landmarker;
      landmarkerHandedOff = true;
      this.video.srcObject = stream;
      await this.video.play();
      if (!current()) return false;
      this.enabled = true;
      this.starting = false;
      this.suspended = false;
      this.layer.hidden = false;
      this.layer.dataset.state = "active";
      stream.getVideoTracks().forEach(track => track.addEventListener?.("ended", () => {
        if (!current()) return;
        this.stop();
        this.error = "摄像头连接已结束，可重新开启；鼠标操作仍可用。";
        this.notifyState();
      }, { once: true }));
      this.notifyState();
      return true;
    } catch (error) {
      if (current()) {
        this.stop();
        this.layer.dataset.state = "unavailable";
        const name = (error as Error)?.name;
        this.error = name === "NotAllowedError" || name === "SecurityError"
          ? "未获得摄像头权限。可在浏览器设置中允许后重试；鼠标操作仍可用。"
          : name === "NotFoundError"
            ? "没有找到摄像头；鼠标操作仍可用。"
            : name === "NotReadableError"
              ? "摄像头被占用或无法读取，请关闭其他占用程序后重试。"
              : "手势识别加载失败，请检查摄像头和本地模型资源后重试；鼠标操作仍可用。";
        this.notifyState();
      }
      return false;
    } finally {
      // A late permission/model/play result must never revive a stopped camera.
      if (!current()) {
        stream?.getTracks().forEach(track => track.stop());
        if (landmarker && !landmarkerHandedOff) landmarker.close();
      }
    }
  }

  tick(now: number): void {
    if (!this.enabled || this.suspended || !this.landmarker) return;
    // A frozen/undecodable video frame is lost tracking too: release any drag
    // even though no new landmark result arrives to tell the state machine.
    if (this.lastFrameAt !== null && now - this.lastFrameAt > 300) {
      this.stateMachine.reset();
      this.previousPinch = { left: false, right: false };
      this.lastFrameAt = null;
    }
    if (this.video.readyState < 2) return;
    if (now - this.lastInferenceAt < 1_000 / 24 || this.video.currentTime === this.lastVideoTime) return;
    this.lastInferenceAt = now;
    this.lastVideoTime = this.video.currentTime;
    try {
      const result = this.landmarker.detectForVideo(this.video, now);
      this.lastFrameAt = now;
      this.processLandmarks(result.landmarks, result.handedness);
    } catch {
      this.setSuspended(true);
      this.error = "手势识别已暂停：当前帧处理失败。可关闭后重试；鼠标操作仍可用。";
      this.notifyState();
    }
  }

  setSuspended(suspended: boolean): void {
    if (suspended && this.starting) {
      this.stop();
      return;
    }
    if (!this.enabled) return;
    this.suspended = suspended;
    this.stream?.getVideoTracks().forEach(track => { track.enabled = !suspended; });
    this.stateMachine.reset();
    this.previousPinch = { left: false, right: false };
    this.lastVideoTime = -1;
    this.lastFrameAt = null;
    this.layer.dataset.state = suspended ? "paused" : "active";
    this.error = "";
    this.notifyState();
  }

  setPreferences(preferences: Partial<GesturePreferences>): void {
    this.stateMachine.setPreferences(preferences);
    this.previousPinch = { left: false, right: false };
  }

  getPreferences(): GesturePreferences {
    return this.stateMachine.getPreferences();
  }

  stop(): void {
    this.generation += 1;
    this.startPromise = null;
    this.starting = false;
    this.enabled = false;
    this.suspended = false;
    this.error = "";
    this.stateMachine.reset();
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.video.pause();
    this.video.srcObject = null;
    this.landmarker?.close();
    this.landmarker = null;
    this.layer.hidden = true;
    this.layer.dataset.state = "idle";
    this.cursor.style.opacity = "0";
    this.dragAnchors.clear();
    this.previousPinch = { left: false, right: false };
    this.lastInferenceAt = 0;
    this.lastVideoTime = -1;
    this.lastFrameAt = null;
    this.notifyState();
  }

  dispose(): void {
    this.disposed = true;
    this.stop();
  }

  getState(): GestureControllerState {
    return {
      enabled: this.enabled, suspended: this.suspended, starting: this.starting,
      layerState: this.layer.dataset.state || "idle", error: this.error,
    };
  }

  private notifyState(): void {
    this.callbacks.onStateChange?.(this.getState());
  }

  private processLandmarks(hands: NormalizedLandmark[][], handedness: Category[][]): void {
    const width = this.stage.clientWidth;
    const height = this.stage.clientHeight;
    const mirroredXs = hands.map(landmarks => 1 - (landmarks[8]?.x ?? 0.5));
    let sides = hands.map((_, index) => resolveGestureHandedness(handedness[index]?.[0]?.categoryName, mirroredXs[index]));
    if (hands.length >= 2 && new Set(sides).size < 2) {
      sides = mirroredXs.map(x => resolveGestureHandedness(undefined, x));
    }
    const metrics = hands.flatMap((landmarks, index) => {
      const side = sides[index];
      const extracted = extractGestureMetrics(landmarks, this.previousPinch[side], this.getPreferences());
      if (!extracted) return [];
      const screen = toScreenMetrics(extracted, width, height, side);
      this.previousPinch[side] = screen.pinch;
      return [screen];
    });
    for (const side of ["left", "right"] as const) {
      if (!metrics.some(hand => hand.handedness === side)) {
        this.previousPinch[side] = false;
      }
    }
    this.stateMachine.process(metrics as ScreenGestureMetrics[], performance.now());
  }

  private createInputPort(): GestureInputPort {
    return {
      pick: point => this.renderer.pick(point.x, point.y),
      hover: nodeIndex => {
        this.renderer.setHovered(nodeIndex);
        this.callbacks.onHover(nodeIndex);
      },
      select: nodeIndex => this.callbacks.onSelect(nodeIndex),
      clearSelection: () => this.callbacks.onClearSelection(),
      fitToGraph: () => this.renderer.fitToGraph(),
      orbitBy: deltaAzimuth => this.renderer.orbitBy(deltaAzimuth),
      panBy: (dx, dy) => this.renderer.panBy(dx, dy),
      zoomAt: (point, factor) => this.renderer.zoomAt(point.x, point.y, factor),
      beginNodeDrag: nodeIndex => {
        this.dragAnchors.set(nodeIndex, this.renderer.getNodeWorldPosition(nodeIndex));
        this.callbacks.onNodeAttractionStart(nodeIndex);
      },
      moveNode: (nodeIndex, point) => {
        const anchor = this.dragAnchors.get(nodeIndex) ?? this.renderer.getNodeWorldPosition(nodeIndex);
        const world = this.renderer.screenToWorld(point.x, point.y, anchor, { x: 0, y: 0, z: 1 });
        this.renderer.setNodeTransientPosition(nodeIndex, world.x, world.y);
      },
      endNodeDrag: (nodeIndex, _moved) => {
        this.dragAnchors.delete(nodeIndex);
        this.callbacks.onNodeAttractionEnd();
      },
      setCursor: (point, mode) => {
        if (!point) {
          this.cursor.style.opacity = "0";
          return;
        }
        this.cursor.style.opacity = "1";
        this.cursor.style.left = `${point.x}px`;
        this.cursor.style.top = `${point.y}px`;
        this.cursor.dataset.mode = mode;
      },
      setGestureActive: active => this.callbacks.onGestureActiveChange(active),
      setNodeDragging: active => this.callbacks.onNodeDraggingChange(active),
    };
  }
}
