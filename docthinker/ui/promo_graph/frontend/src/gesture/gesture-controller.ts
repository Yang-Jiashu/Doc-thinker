import type { Category, HandLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { StarMapRenderer } from "../star-map-renderer";
import {
  extractGestureMetrics,
  resolveGestureHandedness,
  smoothGesturePoint,
  toScreenMetrics,
  type GesturePoint,
  type ScreenGestureMetrics,
} from "./gesture-metrics";
import { createHandLandmarker } from "./mediapipe-adapter";
import { GestureStateMachine, type GestureInputPort } from "./gesture-state-machine";

export interface GestureControllerCallbacks {
  onSelect: (nodeIndex: number) => void;
  onClearSelection: () => void;
  onNodeAttractionStart: (nodeIndex: number) => void;
  onNodeAttractionEnd: () => void;
  onHover: (nodeIndex: number) => void;
  onNodeMoved: (nodeIndex: number) => void;
  onGestureActiveChange: (active: boolean) => void;
  onNodeDraggingChange: (active: boolean) => void;
}

export class GestureController {
  private stage: HTMLElement;
  private layer: HTMLElement;
  private video: HTMLVideoElement;
  private cursor: HTMLElement;
  private renderer: StarMapRenderer;
  private callbacks: GestureControllerCallbacks;
  private stateMachine: GestureStateMachine;
  private landmarker: HandLandmarker | null = null;
  private stream: MediaStream | null = null;
  private enabled = false;
  private suspended = false;
  private lastInferenceAt = 0;
  private lastVideoTime = -1;
  private previousPinch: Record<"left" | "right", boolean> = { left: false, right: false };
  private smoothedPoint: Record<"left" | "right", GesturePoint | null> = { left: null, right: null };
  private dragAnchors = new Map<number, { x: number; y: number; z: number }>();

  constructor(
    stage: HTMLElement,
    layer: HTMLElement,
    video: HTMLVideoElement,
    cursor: HTMLElement,
    renderer: StarMapRenderer,
    callbacks: GestureControllerCallbacks,
  ) {
    this.stage = stage;
    this.layer = layer;
    this.video = video;
    this.cursor = cursor;
    this.renderer = renderer;
    this.callbacks = callbacks;
    this.stateMachine = new GestureStateMachine(this.createInputPort());
  }

  async start(): Promise<boolean> {
    if (this.enabled) return true;
    if (!navigator.mediaDevices?.getUserMedia) {
      this.layer.dataset.state = "unavailable";
      return false;
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, max: 30 },
        },
      });
      this.landmarker = await createHandLandmarker();
      this.video.srcObject = this.stream;
      await this.video.play();
      this.enabled = true;
      this.layer.hidden = false;
      this.layer.dataset.state = "active";
      return true;
    } catch (error) {
      console.warn("Gesture camera unavailable:", error);
      this.stop();
      this.layer.dataset.state = "unavailable";
      return false;
    }
  }

  tick(now: number): void {
    if (!this.enabled || this.suspended || !this.landmarker || this.video.readyState < 2) return;
    if (now - this.lastInferenceAt < 1_000 / 24 || this.video.currentTime === this.lastVideoTime) return;
    this.lastInferenceAt = now;
    this.lastVideoTime = this.video.currentTime;
    try {
      const result = this.landmarker.detectForVideo(this.video, now);
      this.processLandmarks(result.landmarks, result.handedness);
    } catch (error) {
      console.warn("HandLandmarker frame failed:", error);
    }
  }

  setSuspended(suspended: boolean): void {
    this.suspended = suspended;
    this.stream?.getVideoTracks().forEach(track => { track.enabled = !suspended; });
    if (suspended) this.stateMachine.reset();
  }

  stop(): void {
    this.enabled = false;
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
    this.smoothedPoint = { left: null, right: null };
  }

  dispose(): void {
    this.stop();
  }

  getState(): { enabled: boolean; suspended: boolean; layerState: string } {
    return { enabled: this.enabled, suspended: this.suspended, layerState: this.layer.dataset.state || "idle" };
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
      const extracted = extractGestureMetrics(landmarks, this.previousPinch[side]);
      if (!extracted) return [];
      const screen = toScreenMetrics(extracted, width, height, side);
      this.smoothedPoint[side] = smoothGesturePoint(this.smoothedPoint[side], screen.point);
      screen.point = this.smoothedPoint[side];
      this.previousPinch[side] = screen.pinch;
      return [screen];
    });
    for (const side of ["left", "right"] as const) {
      if (!metrics.some(hand => hand.handedness === side)) {
        this.previousPinch[side] = false;
        this.smoothedPoint[side] = null;
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
