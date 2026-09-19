import type { StarMapRenderer } from "./star-map-renderer";

interface PointerState {
  x: number;
  y: number;
}

export interface InputCallbacks {
  onSelect: (nodeIndex: number) => void;
  onSelectEdge: (edgeIndex: number) => void;
  onNodeAttractionStart: (nodeIndex: number) => void;
  onNodeAttractionEnd: () => void;
  onHover: (nodeIndex: number) => void;
  onEscape: () => void;
  onNodeMoved: (nodeIndex: number) => void;
  onPointerInsideChange: (inside: boolean) => void;
  onPointerDownChange: (down: boolean) => void;
  onOrbitInteractionChange: (active: boolean) => void;
  onNodeDraggingChange: (active: boolean) => void;
  onWheel: () => void;
  onKeyboardInteraction: () => void;
}

export class InputController {
  private renderer: StarMapRenderer;
  private canvas: HTMLCanvasElement;
  private callbacks: InputCallbacks;
  private pointers = new Map<number, PointerState>();
  private activePointer = -1;
  private downNode = -1;
  private downEdge = -1;
  private draggingNode = -1;
  private attractionNode = -1;
  private moved = false;
  private lastX = 0;
  private lastY = 0;
  private pinchDistance = 0;
  private pinchCenter: PointerState | null = null;
  private hoverFrame = 0;
  private hoverPoint: PointerState | null = null;
  private dragPlaneAnchor: { x: number; y: number; z: number } | null = null;

  constructor(renderer: StarMapRenderer, callbacks: InputCallbacks) {
    this.renderer = renderer;
    this.canvas = renderer.canvas;
    this.callbacks = callbacks;
    this.canvas.addEventListener("wheel", this.onWheel, { passive: false });
    this.canvas.addEventListener("pointerdown", this.onPointerDown);
    this.canvas.addEventListener("pointermove", this.onPointerMove);
    this.canvas.addEventListener("pointerup", this.onPointerUp);
    this.canvas.addEventListener("pointercancel", this.onPointerUp);
    this.canvas.addEventListener("pointerenter", this.onPointerEnter);
    this.canvas.addEventListener("pointerleave", this.onPointerLeave);
    this.canvas.addEventListener("contextmenu", event => event.preventDefault());
    window.addEventListener("keydown", this.onKeyDown);
  }

  private localPoint(event: PointerEvent | WheelEvent): PointerState {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  private onWheel = (event: WheelEvent) => {
    event.preventDefault();
    this.callbacks.onWheel();
    const point = this.localPoint(event);
    const factor = Math.exp(-event.deltaY * 0.0014);
    this.renderer.zoomAt(point.x, point.y, factor);
  };

  private onPointerDown = (event: PointerEvent) => {
    const point = this.localPoint(event);
    this.pointers.set(event.pointerId, point);
    this.callbacks.onPointerInsideChange(true);
    this.callbacks.onPointerDownChange(true);
    this.canvas.setPointerCapture(event.pointerId);
    if (this.pointers.size === 2) {
      if (this.attractionNode >= 0) this.callbacks.onNodeAttractionEnd();
      this.attractionNode = -1;
      this.activePointer = -1;
      this.draggingNode = -1;
      this.dragPlaneAnchor = null;
      this.downNode = -1;
      this.downEdge = -1;
      this.callbacks.onNodeDraggingChange(false);
      this.callbacks.onOrbitInteractionChange(true);
      this.updatePinchBaseline();
      return;
    }
    this.activePointer = event.pointerId;
    this.lastX = point.x;
    this.lastY = point.y;
    this.moved = false;
    this.downNode = this.renderer.pick(point.x, point.y);
    this.downEdge = this.downNode >= 0 ? -1 : this.renderer.pickEdge(point.x, point.y);
    if (this.downNode >= 0) {
      this.attractionNode = this.downNode;
      this.callbacks.onNodeAttractionStart(this.downNode);
    }
  };

  private updatePinchBaseline(): void {
    const [first, second] = [...this.pointers.values()];
    if (!first || !second) return;
    this.pinchDistance = Math.hypot(first.x - second.x, first.y - second.y);
    this.pinchCenter = { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 };
  }

  private onPointerMove = (event: PointerEvent) => {
    const point = this.localPoint(event);
    if (this.pointers.has(event.pointerId)) this.pointers.set(event.pointerId, point);
    if (this.pointers.size >= 2) {
      const [first, second] = [...this.pointers.values()];
      const distance = Math.max(1, Math.hypot(first.x - second.x, first.y - second.y));
      const center = { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 };
      if (this.pinchDistance > 0) this.renderer.zoomAt(center.x, center.y, distance / this.pinchDistance);
      if (this.pinchCenter) this.renderer.panBy(center.x - this.pinchCenter.x, center.y - this.pinchCenter.y);
      this.pinchDistance = distance;
      this.pinchCenter = center;
      return;
    }
    if (this.activePointer === event.pointerId) {
      const dx = point.x - this.lastX;
      const dy = point.y - this.lastY;
      if (!this.moved && Math.hypot(dx, dy) > 4) this.moved = true;
      if (this.moved && this.downNode >= 0 && this.draggingNode < 0) {
        this.draggingNode = this.downNode;
        this.dragPlaneAnchor = this.renderer.getNodeWorldPosition(this.draggingNode);
        this.callbacks.onNodeDraggingChange(true);
      }
      if (this.draggingNode >= 0) {
        const world = this.renderer.screenToWorld(
          point.x,
          point.y,
          this.dragPlaneAnchor ?? undefined,
          { x: 0, y: 0, z: 1 },
        );
        this.renderer.setNodeTransientPosition(this.draggingNode, world.x, world.y);
      } else if (this.moved) {
        this.callbacks.onOrbitInteractionChange(true);
        this.renderer.panBy(dx, dy);
      }
      this.lastX = point.x;
      this.lastY = point.y;
      return;
    }
    if (event.pointerType !== "touch") this.scheduleHover(point);
  };

  private scheduleHover(point: PointerState): void {
    this.hoverPoint = point;
    if (this.hoverFrame) return;
    this.hoverFrame = requestAnimationFrame(() => {
      this.hoverFrame = 0;
      if (!this.hoverPoint) return;
      const nodeIndex = this.renderer.pick(this.hoverPoint.x, this.hoverPoint.y);
      this.renderer.setHovered(nodeIndex);
      this.callbacks.onHover(nodeIndex);
      this.canvas.style.cursor = nodeIndex >= 0 ? "pointer" : "grab";
    });
  }

  private onPointerUp = (event: PointerEvent) => {
    const point = this.localPoint(event);
    const wasActive = this.activePointer === event.pointerId;
    this.pointers.delete(event.pointerId);
    this.callbacks.onPointerDownChange(this.pointers.size > 0);
    if (this.pointers.size < 2) {
      this.pinchDistance = 0;
      this.pinchCenter = null;
    }
    if (!wasActive) {
      if (this.pointers.size === 1) {
        const [remainingId, remaining] = [...this.pointers.entries()][0];
        this.activePointer = remainingId;
        this.lastX = remaining.x;
        this.lastY = remaining.y;
      }
      return;
    }
    if (this.draggingNode < 0 && !this.moved) {
      const picked = this.downNode >= 0 ? this.downNode : this.renderer.pick(point.x, point.y);
      if (picked >= 0) this.callbacks.onSelect(picked);
      else {
        const edge = this.downEdge >= 0 ? this.downEdge : this.renderer.pickEdge(point.x, point.y);
        if (edge >= 0) this.callbacks.onSelectEdge(edge);
        else this.callbacks.onSelect(-1);
      }
    }
    if (this.attractionNode >= 0) this.callbacks.onNodeAttractionEnd();
    this.activePointer = -1;
    this.downNode = -1;
    this.downEdge = -1;
    this.draggingNode = -1;
    this.attractionNode = -1;
    this.dragPlaneAnchor = null;
    this.moved = false;
    this.callbacks.onNodeDraggingChange(false);
    this.callbacks.onOrbitInteractionChange(false);
  };

  private onPointerEnter = () => this.callbacks.onPointerInsideChange(true);

  private onPointerLeave = () => this.callbacks.onPointerInsideChange(false);

  private onKeyDown = (event: KeyboardEvent) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement || event.target instanceof HTMLTextAreaElement) {
      if (event.key === "Escape") this.callbacks.onEscape();
      return;
    }
    const centerX = this.canvas.clientWidth / 2;
    const centerY = this.canvas.clientHeight / 2;
    this.callbacks.onKeyboardInteraction();
    if (event.key === "+" || event.key === "=") this.renderer.zoomAt(centerX, centerY, 1.25);
    else if (event.key === "-" || event.key === "_") this.renderer.zoomAt(centerX, centerY, 0.8);
    else if (event.key === "0") this.renderer.fitToGraph();
    else if (event.key === "Escape") this.callbacks.onEscape();
    else if (event.key === "ArrowLeft") this.renderer.panBy(48, 0);
    else if (event.key === "ArrowRight") this.renderer.panBy(-48, 0);
    else if (event.key === "ArrowUp") this.renderer.panBy(0, 48);
    else if (event.key === "ArrowDown") this.renderer.panBy(0, -48);
    else return;
    event.preventDefault();
  };

  dispose(): void {
    if (this.attractionNode >= 0) this.callbacks.onNodeAttractionEnd();
    this.canvas.removeEventListener("wheel", this.onWheel);
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerup", this.onPointerUp);
    this.canvas.removeEventListener("pointercancel", this.onPointerUp);
    this.canvas.removeEventListener("pointerenter", this.onPointerEnter);
    this.canvas.removeEventListener("pointerleave", this.onPointerLeave);
    window.removeEventListener("keydown", this.onKeyDown);
    cancelAnimationFrame(this.hoverFrame);
  }
}
