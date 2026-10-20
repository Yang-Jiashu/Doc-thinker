import { smoothGesturePoint, type GestureHandedness, type GesturePoint, type ScreenGestureMetrics } from "./gesture-metrics";
import { normalizeGesturePreferences, type GesturePreferences } from "./gesture-preferences";

export type GestureAction = "idle" | "node-captured" | "canvas-pan" | "orbit-swing" | "zoom-in" | "zoom-out";

export interface GestureInputPort {
  pick(point: GesturePoint): number;
  hover(nodeIndex: number): void;
  select(nodeIndex: number): void;
  clearSelection(): void;
  fitToGraph(): void;
  orbitBy(deltaAzimuth: number): void;
  panBy(screenDx: number, screenDy: number): void;
  zoomAt(point: GesturePoint, factor: number): void;
  beginNodeDrag(nodeIndex: number): void;
  moveNode(nodeIndex: number, point: GesturePoint): void;
  endNodeDrag(nodeIndex: number, moved: boolean): void;
  setCursor(point: GesturePoint | null, mode: GestureAction): void;
  setGestureActive(active: boolean): void;
  setNodeDragging(active: boolean): void;
}

interface Grab { node: number; origin: GesturePoint; last: GesturePoint; moved: boolean; }
const distance = (a: GesturePoint, b: GesturePoint): number => Math.hypot(a.x - b.x, a.y - b.y);
const midpoint = (a: GesturePoint, b: GesturePoint): GesturePoint => ({x: (a.x + b.x) / 2, y: (a.y + b.y) / 2});
const validPoint = (point: GesturePoint): boolean => Number.isFinite(point.x) && Number.isFinite(point.y);
// A tracking gap is not evidence of a continuous hold.
const MAX_TRACKING_GAP_MS = 300;

/** One action per frame. Inspect never rotates or zooms from an unarmed open palm. */
export class GestureStateMachine {
  private preferences: GesturePreferences;
  private primaryHand: GestureHandedness | null = null;
  private lastAt: number | null = null;
  private smoothed = new Map<GestureHandedness, ScreenGestureMetrics>();
  private rawPalm = new Map<GestureHandedness, GesturePoint>();
  private hoveredNode = -1;
  private hoverSince = 0;
  private hoverAnchor: GesturePoint | null = null;
  private dwellSelectedNode = -1;
  private grab: Grab | null = null;
  private pinchNeedsRelease = false;
  private pairDistance: number | null = null;
  private fistSince: number | null = null;
  private fistResetDone = false;
  private presentationPose: "pending" | "orbit" | "zoom-in" | "zoom-out" | null = null;
  private presentationOrigin: GesturePoint | null = null;
  private presentationLast: GesturePoint | null = null;
  private presentationSince = 0;

  constructor(private port: GestureInputPort, preferences: Partial<GesturePreferences> = {}) {
    this.preferences = normalizeGesturePreferences(preferences);
  }
  getPreferences(): GesturePreferences { return {...this.preferences}; }
  setPreferences(preferences: Partial<GesturePreferences>): void {
    this.reset();
    this.preferences = normalizeGesturePreferences({...this.preferences, ...preferences});
  }

  process(hands: ScreenGestureMetrics[], now: number): void {
    const valid = hands.filter(hand => validPoint(hand.point) && validPoint(hand.thumbPoint) && validPoint(hand.palmPoint)
      && Number.isFinite(hand.pinchDistance) && Number.isInteger(hand.extendedFingers)
      && hand.extendedFingers >= 0 && hand.extendedFingers <= 4);
    if (!Number.isFinite(now) || !valid.length || new Set(valid.map(hand => hand.handedness)).size !== valid.length) {
      this.reset(); return;
    }
    if (this.lastAt !== null && (now < this.lastAt || now - this.lastAt > MAX_TRACKING_GAP_MS)) this.reset();
    if (this.lastAt === now) return;
    let elapsed = this.lastAt === null ? 0 : now - this.lastAt;
    // A tracker may swap identities while retaining the same left/right label.
    // Discontinuous palm motion releases rather than teleports an armed drag.
    const jumped = valid.some(hand => {
      const previous = this.rawPalm.get(hand.handedness);
      return previous && distance(previous, hand.palmPoint) > Math.max(120, elapsed * 2.5);
    });
    if (jumped) { this.reset(); elapsed = 0; }
    this.lastAt = now;
    const alpha = this.preferences.smoothingMs === 0 ? 1 : -Math.expm1(-elapsed / this.preferences.smoothingMs);
    const visible = new Set(valid.map(hand => hand.handedness));
    for (const side of this.smoothed.keys()) if (!visible.has(side)) this.smoothed.delete(side);
    for (const side of this.rawPalm.keys()) if (!visible.has(side)) this.rawPalm.delete(side);
    for (const hand of valid) this.rawPalm.set(hand.handedness, {...hand.palmPoint});
    const filtered = valid.map(hand => {
      const previous = this.smoothed.get(hand.handedness);
      const next = {
        ...hand,
        point: smoothGesturePoint(previous?.point ?? null, hand.point, alpha),
        thumbPoint: smoothGesturePoint(previous?.thumbPoint ?? null, hand.thumbPoint, alpha),
        palmPoint: smoothGesturePoint(previous?.palmPoint ?? null, hand.palmPoint, alpha),
      };
      this.smoothed.set(hand.handedness, next);
      return next;
    });
    const preferred = this.preferences.dominantHand === "auto" ? this.primaryHand : this.preferences.dominantHand;
    const primary = filtered.find(hand => hand.handedness === preferred)
      ?? filtered.find(hand => hand.handedness === "right") ?? filtered[0];
    if (this.primaryHand && this.primaryHand !== primary.handedness) {
      this.clearInteraction();
      this.pinchNeedsRelease = true;
    }
    this.primaryHand = primary.handedness;
    this.port.setGestureActive(true);
    const left = filtered.find(hand => hand.handedness === "left");
    const right = filtered.find(hand => hand.handedness === "right");
    if (this.preferences.mode === "inspect" && left?.pinch && right?.pinch) {
      this.resetFist();
      this.processPairZoom(left, right);
      return;
    }
    if (this.pairDistance !== null) {
      this.pairDistance = null;
      // Lifting one hand must not turn the remaining pinch into a grab.
      this.pinchNeedsRelease = true;
    }
    if (this.processFist(primary, now)) return;
    if (this.preferences.mode === "presentation") {
      this.processPresentation(primary, now, elapsed);
      return;
    }
    if (primary.pinch) {
      this.clearHover();
      if (!this.pinchNeedsRelease) this.processGrab(primary);
      else this.port.setCursor(primary.point, "idle");
      return;
    }
    this.pinchNeedsRelease = false;
    this.releaseGrab();
    if (primary.pointing) this.processPointer(primary, now);
    else { this.clearHover(); this.port.setCursor(primary.point, "idle"); }
  }

  reset(): void {
    this.clearInteraction();
    this.primaryHand = null;
    this.lastAt = null;
    this.smoothed.clear();
    this.rawPalm.clear();
    this.pinchNeedsRelease = true;
    this.port.setCursor(null, "idle");
    this.port.setGestureActive(false);
  }

  private processFist(hand: ScreenGestureMetrics, now: number): boolean {
    // Pinch often resembles a fist; it must never reset the graph.
    if (hand.extendedFingers !== 0 || hand.openPalm || hand.pinch) { this.resetFist(); return false; }
    this.releaseGrab(); this.clearHover(); this.resetPresentation();
    this.port.setCursor(hand.palmPoint, "idle");
    if (this.fistSince === null) this.fistSince = now;
    if (!this.fistResetDone && now - this.fistSince >= this.preferences.resetHoldMs) {
      this.port.clearSelection(); this.port.fitToGraph();
      this.fistResetDone = true;
    }
    return true;
  }

  private processGrab(hand: ScreenGestureMetrics): void {
    const point = midpoint(hand.point, hand.thumbPoint);
    if (!this.grab) {
      const node = this.port.pick(point);
      this.grab = {node, origin: {...point}, last: {...point}, moved: false};
      if (node >= 0) {
        this.port.select(node);
        this.port.beginNodeDrag(node);
        this.port.setNodeDragging(true);
      }
    }
    const grab = this.grab;
    this.port.setCursor(point, grab.node >= 0 ? "node-captured" : "canvas-pan");
    if (!grab.moved && distance(point, grab.origin) < this.preferences.movementThresholdPx) return;
    if (distance(point, grab.last) === 0) return;
    grab.moved = true;
    if (grab.node >= 0) this.port.moveNode(grab.node, point);
    else this.port.panBy((point.x - grab.last.x) * this.preferences.panGain, (point.y - grab.last.y) * this.preferences.panGain);
    grab.last = {...point};
  }

  private releaseGrab(): void {
    if (this.grab && this.grab.node >= 0) {
      this.port.endNodeDrag(this.grab.node, this.grab.moved);
      this.port.setNodeDragging(false);
    }
    this.grab = null;
  }

  private processPairZoom(left: ScreenGestureMetrics, right: ScreenGestureMetrics): void {
    this.releaseGrab(); this.clearHover(); this.resetPresentation();
    const a = midpoint(left.point, left.thumbPoint), b = midpoint(right.point, right.thumbPoint);
    const span = distance(a, b), center = midpoint(a, b);
    if (span < 32) { this.pairDistance = 0; this.port.setCursor(center, "idle"); return; }
    if (!this.pairDistance) { this.pairDistance = span; this.port.setCursor(center, "idle"); return; }
    const previous = this.pairDistance;
    if (Math.abs(span - previous) < this.preferences.movementThresholdPx) return;
    // A huge tracking jump is a new baseline, not a large zoom command.
    if (span / previous > 2 || span / previous < 0.5) { this.pairDistance = span; return; }
    this.port.zoomAt(center, Math.pow(span / previous, this.preferences.zoomGain));
    this.port.setCursor(center, span > previous ? "zoom-in" : "zoom-out");
    this.pairDistance = span;
  }

  private processPointer(hand: ScreenGestureMetrics, now: number): void {
    const hovered = this.port.pick(hand.point);
    this.port.setCursor(hand.point, "idle");
    if (hovered < 0) { this.clearHover(); return; }
    const changed = hovered !== this.hoveredNode;
    if (changed || !this.hoverAnchor || distance(hand.point, this.hoverAnchor) > this.preferences.movementThresholdPx) {
      this.hoverSince = now;
      this.hoverAnchor = {...hand.point};
      if (changed) this.dwellSelectedNode = -1;
    }
    this.hoveredNode = hovered;
    this.port.hover(hovered);
    if (this.dwellSelectedNode !== hovered && now - this.hoverSince >= this.preferences.dwellMs) {
      this.port.select(hovered);
      this.dwellSelectedNode = hovered;
    }
  }

  private processPresentation(hand: ScreenGestureMetrics, now: number, elapsed: number): void {
    this.releaseGrab();
    if (hand.pinch) {
      this.clearHover();
      if (this.pinchNeedsRelease) { this.port.setCursor(hand.point, "idle"); return; }
      if (this.presentationPose !== "zoom-out") this.startPresentation("zoom-out", hand.palmPoint, now);
      this.applyTimedZoom(hand.palmPoint, -1, now, elapsed);
      return;
    }
    this.pinchNeedsRelease = false;
    if (!hand.openPalm) {
      this.resetPresentation();
      if (hand.pointing) this.processPointer(hand, now);
      else { this.clearHover(); this.port.setCursor(hand.point, "idle"); }
      return;
    }
    this.clearHover();
    if (!this.presentationPose || this.presentationPose === "zoom-out") this.startPresentation("pending", hand.palmPoint, now);
    if (this.presentationPose === "pending") {
      const origin = this.presentationOrigin!;
      const dx = hand.palmPoint.x - origin.x, dy = hand.palmPoint.y - origin.y;
      if (Math.abs(dx) >= this.preferences.movementThresholdPx && Math.abs(dx) >= Math.abs(dy) * 0.8) {
        this.presentationPose = "orbit";
      } else if (distance(hand.palmPoint, origin) >= this.preferences.movementThresholdPx) {
        this.startPresentation("pending", hand.palmPoint, now);
      } else if (now - this.presentationSince >= this.preferences.activationMs) {
        this.presentationPose = "zoom-in";
      }
    }
    if (this.presentationPose === "orbit") {
      const dx = hand.palmPoint.x - this.presentationLast!.x;
      if (dx) this.port.orbitBy(dx * 0.006 * this.preferences.orbitGain);
      this.presentationLast = {...hand.palmPoint};
      this.port.setCursor(hand.palmPoint, "orbit-swing");
    } else if (this.presentationPose === "zoom-in") this.applyTimedZoom(hand.palmPoint, 1, now, elapsed);
    else this.port.setCursor(hand.palmPoint, "idle");
  }

  private applyTimedZoom(point: GesturePoint, direction: number, now: number, elapsed: number): void {
    this.port.setCursor(point, direction > 0 ? "zoom-in" : "zoom-out");
    const activeMs = Math.max(0, now - Math.max(now - elapsed, this.presentationSince + this.preferences.activationMs));
    if (activeMs) this.port.zoomAt(point, Math.exp(direction * activeMs / 1_600 * this.preferences.zoomGain));
  }
  private startPresentation(pose: "pending" | "zoom-out", point: GesturePoint, now: number): void {
    this.presentationPose = pose;
    this.presentationOrigin = {...point}; this.presentationLast = {...point}; this.presentationSince = now;
  }
  private clearHover(): void {
    if (this.hoveredNode >= 0) this.port.hover(-1);
    this.hoveredNode = -1; this.hoverAnchor = null; this.hoverSince = 0; this.dwellSelectedNode = -1;
  }
  private resetFist(): void { this.fistSince = null; this.fistResetDone = false; }
  private resetPresentation(): void { this.presentationPose = null; this.presentationOrigin = null; this.presentationLast = null; }
  private clearInteraction(): void {
    this.releaseGrab(); this.clearHover(); this.resetPresentation(); this.resetFist(); this.pairDistance = null;
  }
}
