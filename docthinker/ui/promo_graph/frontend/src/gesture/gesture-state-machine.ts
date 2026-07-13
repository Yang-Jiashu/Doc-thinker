import type { GesturePoint, ScreenGestureMetrics } from "./gesture-metrics";

type GestureAction = "idle" | "node-captured" | "canvas-pan";

export interface GestureInputPort {
  pick(point: GesturePoint): number;
  hover(nodeIndex: number): void;
  select(nodeIndex: number): void;
  clearSelection(): void;
  fitToGraph(): void;
  panBy(dx: number, dy: number): void;
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
  private rightAction: GestureAction = "idle";
  private hoveredNode = -1;
  private hoverSince = 0;
  private candidateNode = -1;
  private pinchFrames = 0;
  private releaseFrames = 0;
  private cooldownUntil = 0;
  private capturedNode = -1;
  private actionPoint: GesturePoint | null = null;
  private actionMoved = false;
  private leftPalmPoint: GesturePoint | null = null;
  private twoPalmSince = 0;
  private twoPalmResetDone = false;

  constructor(port: GestureInputPort) {
    this.port = port;
  }

  process(hands: ScreenGestureMetrics[], now: number): void {
    const left = hands.find(hand => hand.handedness === "left") ?? null;
    const right = hands.find(hand => hand.handedness === "right") ?? null;
    if (!left && !right) {
      this.port.setGestureActive(false);
      this.releaseRightAction(false, now);
      this.clearRightPointer();
      this.leftPalmPoint = null;
      this.twoPalmSince = 0;
      this.twoPalmResetDone = false;
      return;
    }

    this.port.setGestureActive(true);
    if (this.processTwoPalmReset(left, right, now)) return;
    this.processLeftPalmZoom(left);
    this.processRightPointer(right, now);
  }

  reset(now = performance.now()): void {
    this.releaseRightAction(false, now);
    this.clearRightPointer();
    this.leftPalmPoint = null;
    this.port.setGestureActive(false);
  }

  private processTwoPalmReset(
    left: ScreenGestureMetrics | null,
    right: ScreenGestureMetrics | null,
    now: number,
  ): boolean {
    if (!left?.openPalm || !right?.openPalm) {
      this.twoPalmSince = 0;
      this.twoPalmResetDone = false;
      return false;
    }
    this.releaseRightAction(false, now);
    this.leftPalmPoint = null;
    this.port.setCursor(right.point, "idle");
    if (!this.twoPalmSince) this.twoPalmSince = now;
    if (!this.twoPalmResetDone && now - this.twoPalmSince >= 420) {
      this.port.clearSelection();
      this.port.fitToGraph();
      this.twoPalmResetDone = true;
    }
    return true;
  }

  private processLeftPalmZoom(left: ScreenGestureMetrics | null): void {
    if (!left?.openPalm || left.pinch) {
      this.leftPalmPoint = null;
      return;
    }
    if (!this.leftPalmPoint) {
      this.leftPalmPoint = { ...left.palmPoint };
      return;
    }
    const dy = left.palmPoint.y - this.leftPalmPoint.y;
    if (Math.abs(dy) >= 2) {
      this.port.zoomAt(left.palmPoint, Math.exp(-dy / 230));
      this.leftPalmPoint = { ...left.palmPoint };
    }
  }

  private processRightPointer(right: ScreenGestureMetrics | null, now: number): void {
    if (!right) {
      this.releaseRightAction(false, now);
      this.clearRightPointer();
      return;
    }
    this.port.setCursor(right.point, this.rightAction);
    if (!right.pinch || this.rightAction !== "idle") this.updateHover(right.point, now);
    this.processRightPinch(right, now);
  }

  private updateHover(point: GesturePoint, now: number): void {
    if (this.rightAction !== "idle") return;
    const hovered = this.port.pick(point);
    if (hovered !== this.hoveredNode) {
      this.hoveredNode = hovered;
      this.hoverSince = now;
      this.candidateNode = -1;
    }
    if (hovered >= 0 && now - this.hoverSince >= 140) this.candidateNode = hovered;
    else if (hovered < 0) this.candidateNode = -1;
    this.port.hover(this.candidateNode >= 0 ? this.candidateNode : hovered);
  }

  private processRightPinch(metrics: ScreenGestureMetrics, now: number): void {
    if (now < this.cooldownUntil) return;
    if (metrics.pinch) {
      this.pinchFrames += 1;
      this.releaseFrames = 0;
    } else {
      this.releaseFrames += 1;
      this.pinchFrames = 0;
    }

    if (!metrics.pinch) {
      if (this.releaseFrames >= 2 && this.rightAction !== "idle") {
        this.releaseRightAction(this.rightAction === "node-captured", now);
      }
      return;
    }
    if (this.pinchFrames < 3) return;

    if (this.rightAction === "idle") {
      this.actionPoint = { ...metrics.point };
      this.actionMoved = false;
      if (this.candidateNode >= 0) {
        this.rightAction = "node-captured";
        this.capturedNode = this.candidateNode;
        this.port.select(this.capturedNode);
        this.port.beginNodeDrag(this.capturedNode);
      } else {
        this.rightAction = "canvas-pan";
      }
    }

    if (!this.actionPoint) return;
    const dx = metrics.point.x - this.actionPoint.x;
    const dy = metrics.point.y - this.actionPoint.y;
    if (this.rightAction === "node-captured" && this.capturedNode >= 0) {
      if (!this.actionMoved && Math.hypot(dx, dy) >= 5) {
        this.actionMoved = true;
        this.port.setNodeDragging(true);
      }
      if (this.actionMoved) this.port.moveNode(this.capturedNode, metrics.point);
    } else if (this.rightAction === "canvas-pan" && Math.hypot(dx, dy) >= 1) {
      this.actionMoved = true;
      this.port.panBy(dx, dy);
      this.actionPoint = { ...metrics.point };
    }
    this.port.setCursor(metrics.point, this.rightAction);
  }

  private releaseRightAction(clearSelection: boolean, now: number): void {
    if (this.rightAction === "node-captured" && this.capturedNode >= 0) {
      this.port.endNodeDrag(this.capturedNode, this.actionMoved);
      this.port.setNodeDragging(false);
      if (clearSelection) this.port.clearSelection();
    }
    this.rightAction = "idle";
    this.capturedNode = -1;
    this.actionPoint = null;
    this.actionMoved = false;
    this.pinchFrames = 0;
    this.releaseFrames = 0;
    this.cooldownUntil = now + 160;
  }

  private clearRightPointer(): void {
    this.hoveredNode = -1;
    this.hoverSince = 0;
    this.candidateNode = -1;
    this.port.hover(-1);
    this.port.setCursor(null, "idle");
  }
}
