import type { GesturePoint, ScreenGestureMetrics } from "./gesture-metrics";

type GestureAction = "idle" | "node-captured" | "canvas-pan" | "orbit-swing" | "zoom-in" | "zoom-out";
type ZoomMode = "in" | "out";

export interface GestureInputPort {
  pick(point: GesturePoint): number;
  hover(nodeIndex: number): void;
  select(nodeIndex: number): void;
  clearSelection(): void;
  fitToGraph(): void;
  orbitBy(deltaAzimuth: number): void;
  zoomAt(point: GesturePoint, factor: number): void;
  beginNodeDrag(nodeIndex: number): void;
  moveNode(nodeIndex: number, point: GesturePoint): void;
  endNodeDrag(nodeIndex: number, moved: boolean): void;
  setCursor(point: GesturePoint | null, mode: GestureAction): void;
  setGestureActive(active: boolean): void;
  setNodeDragging(active: boolean): void;
}

export class GestureStateMachine {
  private port: GestureInputPort;
  private hoveredNode = -1;
  private hoverSince = 0;
  private dwellSelectedNode = -1;
  private swingPoint: GesturePoint | null = null;
  private swingAccumulator = 0;
  private zoomMode: ZoomMode | null = null;
  private zoomLastAt = 0;
  private fistSince = 0;
  private fistResetDone = false;

  constructor(port: GestureInputPort) {
    this.port = port;
  }

  process(hands: ScreenGestureMetrics[], now: number): void {
    const left = hands.find(hand => hand.handedness === "left") ?? null;
    const right = hands.find(hand => hand.handedness === "right") ?? null;
    const primary = right ?? left;
    if (!primary) {
      this.port.setGestureActive(false);
      this.clearInteractionState();
      this.resetFistHold();
      return;
    }

    this.port.setGestureActive(true);
    if (this.processSingleHandReset(primary, now)) return;
    if (this.processHorizontalSwing(primary)) {
      this.resetZoom();
      return;
    }

    if (primary.openPalm && !primary.pinch) {
      this.processContinuousZoom(primary, "in", now);
      return;
    }
    if (primary.pinch) {
      this.processContinuousZoom(primary, "out", now);
      return;
    }

    this.resetZoom();
    this.processPointer(primary, now);
  }

  reset(): void {
    this.clearInteractionState();
    this.resetFistHold();
    this.port.setGestureActive(false);
  }

  private processSingleHandReset(hand: ScreenGestureMetrics, now: number): boolean {
    const closedFist = hand.extendedFingers === 0 && !hand.openPalm;
    if (!closedFist) {
      this.resetFistHold();
      return false;
    }
    this.clearHover();
    this.resetSwing();
    this.resetZoom();
    this.port.setCursor(hand.palmPoint, "idle");
    if (!this.fistSince) this.fistSince = now;
    if (!this.fistResetDone && now - this.fistSince >= 500) {
      this.port.clearSelection();
      this.port.fitToGraph();
      this.fistResetDone = true;
    }
    return true;
  }

  private resetFistHold(): void {
    this.fistSince = 0;
    this.fistResetDone = false;
  }

  private processContinuousZoom(hand: ScreenGestureMetrics, mode: ZoomMode, now: number): void {
    this.clearHover();
    this.port.setCursor(hand.palmPoint, mode === "in" ? "zoom-in" : "zoom-out");
    if (this.zoomMode !== mode || !this.zoomLastAt) {
      this.zoomMode = mode;
      this.zoomLastAt = now;
      return;
    }
    const elapsed = Math.max(0, Math.min(now - this.zoomLastAt, 100));
    this.zoomLastAt = now;
    if (elapsed <= 0) return;
    const direction = mode === "in" ? 1 : -1;
    this.port.zoomAt(hand.palmPoint, Math.exp(direction * elapsed / 1_600));
  }

  private resetZoom(): void {
    this.zoomMode = null;
    this.zoomLastAt = 0;
  }

  private processPointer(hand: ScreenGestureMetrics, now: number): void {
    const hovered = this.port.pick(hand.point);
    this.port.setCursor(hand.point, hovered >= 0 ? "idle" : "orbit-swing");
    if (hovered >= 0) {
      if (hovered !== this.hoveredNode) {
        this.hoveredNode = hovered;
        this.hoverSince = now;
        this.dwellSelectedNode = -1;
      }
      this.port.hover(hovered);
      if (this.dwellSelectedNode !== hovered && now - this.hoverSince >= 550) {
        this.port.select(hovered);
        this.dwellSelectedNode = hovered;
      }
      return;
    }

    this.clearHover();
  }

  private processHorizontalSwing(hand: ScreenGestureMetrics): boolean {
    if (!this.swingPoint) {
      this.swingPoint = { ...hand.palmPoint };
      this.swingAccumulator = 0;
      return false;
    }
    const rawDx = hand.palmPoint.x - this.swingPoint.x;
    const rawDy = hand.palmPoint.y - this.swingPoint.y;
    this.swingPoint = { ...hand.palmPoint };

    if (Math.abs(rawDx) < Math.abs(rawDy) * 0.55) {
      this.swingAccumulator *= 0.5;
      return false;
    }
    this.swingAccumulator = this.swingAccumulator * 0.62 + rawDx;
    if (Math.abs(this.swingAccumulator) < 1.25) return false;

    const deltaAzimuth = this.swingAccumulator * 0.006;
    this.swingAccumulator *= 0.18;
    this.clearHover();
    this.port.setCursor(hand.palmPoint, "orbit-swing");
    this.port.orbitBy(deltaAzimuth);
    return true;
  }

  private clearHover(): void {
    if (this.hoveredNode >= 0) this.port.hover(-1);
    this.hoveredNode = -1;
    this.hoverSince = 0;
    this.dwellSelectedNode = -1;
  }

  private resetSwing(): void {
    this.swingPoint = null;
    this.swingAccumulator = 0;
  }

  private clearInteractionState(): void {
    this.clearHover();
    this.resetSwing();
    this.resetZoom();
    this.port.setNodeDragging(false);
    this.port.setCursor(null, "idle");
  }
}
