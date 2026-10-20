import { afterEach, describe, expect, it, vi } from "vitest";
import { InputController, type InputCallbacks } from "../src/input-controller";
import type { StarMapRenderer } from "../src/star-map-renderer";

function fixture() {
  const listeners = new Map<string, (event: PointerEvent) => void>();
  const windowListeners = new Map<string, (event: KeyboardEvent) => void>();
  const captures = new Set<number>();
  const canvas = {
    addEventListener: vi.fn((name: string, callback: (event: PointerEvent) => void) => listeners.set(name, callback)),
    removeEventListener: vi.fn(), focus: vi.fn(), getBoundingClientRect: () => ({ left: 0, top: 0 }),
    setPointerCapture: (id: number) => captures.add(id), hasPointerCapture: (id: number) => captures.has(id),
    releasePointerCapture: vi.fn((id: number) => captures.delete(id)), style: {},
  };
  vi.stubGlobal("window", {
    addEventListener: vi.fn((name: string, callback: (event: KeyboardEvent) => void) => windowListeners.set(name, callback)),
    removeEventListener: vi.fn(),
  });
  vi.stubGlobal("document", { querySelector: vi.fn(() => null) });
  vi.stubGlobal("requestAnimationFrame", vi.fn(() => 1));
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  const callbacks: InputCallbacks = {
    onSelect: vi.fn(), onSelectEdge: vi.fn(), onNodeAttractionStart: vi.fn(), onNodeAttractionEnd: vi.fn(), onHover: vi.fn(),
    onEscape: vi.fn(), onNodeMoved: vi.fn(), onPointerInsideChange: vi.fn(), onPointerDownChange: vi.fn(),
    onOrbitInteractionChange: vi.fn(), onNodeDraggingChange: vi.fn(), onWheel: vi.fn(), onKeyboardInteraction: vi.fn(),
  };
  const renderer = {
    canvas, pick: vi.fn(() => 3), pickEdge: vi.fn(() => -1), panBy: vi.fn(), zoomAt: vi.fn(),
    getNodeWorldPosition: vi.fn(() => ({ x: 0, y: 0, z: 0 })), screenToWorld: vi.fn(() => ({ x: 6, y: 0 })),
    setNodeTransientPosition: vi.fn(), setHovered: vi.fn(), fitToGraph: vi.fn(),
  };
  const controller = new InputController(renderer as unknown as StarMapRenderer, callbacks);
  const send = (type: string, x = 0, pointerId = 1) => listeners.get(type)!({ pointerId, clientX: x, clientY: 0, pointerType: "touch" } as PointerEvent);
  const key = (key: string, extra: Partial<KeyboardEvent> = {}) => {
    const event = { key, preventDefault: vi.fn(), ...extra } as unknown as KeyboardEvent;
    windowListeners.get("keydown")!(event);
    return event;
  };
  return { controller, callbacks, send, key, windowListeners, canvas, renderer, captures };
}

describe("pointer cancellation and slow drags", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("cancelling a pointer releases capture and never selects a node", () => {
    const f = fixture();
    f.send("pointerdown");
    expect(f.canvas.focus).toHaveBeenCalledExactlyOnceWith({ preventScroll: true });
    f.send("pointercancel");
    expect(f.callbacks.onSelect).not.toHaveBeenCalled();
    expect(f.callbacks.onSelectEdge).not.toHaveBeenCalled();
    expect(f.callbacks.onNodeAttractionEnd).toHaveBeenCalledTimes(1);
    expect(f.callbacks.onPointerDownChange).toHaveBeenLastCalledWith(false);
    expect(f.callbacks.onNodeDraggingChange).toHaveBeenLastCalledWith(false);
    expect(f.captures.size).toBe(0);
    f.controller.dispose();
  });

  it("recognizes cumulative drag distance instead of treating slow movement as a click", () => {
    const f = fixture();
    f.send("pointerdown");
    for (const x of [2, 4, 6, 8]) f.send("pointermove", x);
    f.send("pointerup", 8);
    expect(f.renderer.setNodeTransientPosition).toHaveBeenCalled();
    expect(f.callbacks.onSelect).not.toHaveBeenCalled();
    expect(f.captures.size).toBe(0);
    f.controller.dispose();
  });

  it("does not turn the remaining finger after a pinch into a click", () => {
    const f = fixture();
    f.send("pointerdown", 0, 1);
    f.send("pointerdown", 20, 2);
    f.send("pointerup", 20, 2);
    f.send("pointerup", 0, 1);
    expect(f.callbacks.onSelect).not.toHaveBeenCalled();
    expect(f.captures.size).toBe(0);
    f.controller.dispose();
  });

  it("resets in-flight pointer state before a session or interaction-mode change", () => {
    const f = fixture();
    f.send("pointerdown");
    f.controller.resetInteraction();
    f.send("pointerup");
    expect(f.callbacks.onSelect).not.toHaveBeenCalled();
    expect(f.callbacks.onPointerDownChange).toHaveBeenLastCalledWith(false);
    f.controller.dispose();
  });

  it("ignores a third finger without replacing the pinch or selecting its release", () => {
    const f = fixture();
    f.send("pointerdown", 0, 1); f.send("pointerdown", 20, 2);
    f.send("pointerdown", 200, 3); f.send("pointermove", 220, 3); f.send("pointerup", 220, 3);
    expect(f.captures.size).toBe(2);
    expect(f.renderer.zoomAt).not.toHaveBeenCalled();
    f.send("pointermove", 30, 2);
    expect(f.renderer.zoomAt).toHaveBeenCalledExactlyOnceWith(15, 0, 1.5);
    f.send("pointerup", 30, 2); f.send("pointerup", 0, 1);
    expect(f.callbacks.onSelect).not.toHaveBeenCalled();
    expect(f.callbacks.onOrbitInteractionChange).toHaveBeenLastCalledWith(false);
    f.controller.dispose();
  });

  it("releases all pointers on capture loss; a later pointerup is not a click", () => {
    const f = fixture();
    f.send("pointerdown"); f.send("pointermove", 10);
    f.send("lostpointercapture"); f.send("pointerup", 10);
    expect(f.captures.size).toBe(0);
    expect(f.callbacks.onNodeAttractionEnd).toHaveBeenCalledTimes(1);
    expect(f.callbacks.onNodeDraggingChange).toHaveBeenLastCalledWith(false);
    expect(f.callbacks.onSelect).not.toHaveBeenCalled();
    f.controller.dispose();
  });

  it("window blur releases touch ownership and hover without selecting", () => {
    const f = fixture();
    f.send("pointerdown");
    f.windowListeners.get("blur")!({} as KeyboardEvent);
    f.send("pointerup");
    expect(f.captures.size).toBe(0);
    expect(f.renderer.setHovered).toHaveBeenLastCalledWith(-1);
    expect(f.callbacks.onPointerInsideChange).toHaveBeenLastCalledWith(false);
    expect(f.callbacks.onSelect).not.toHaveBeenCalled();
    f.controller.dispose();
  });

  it("only recognized unmodified graph keys claim keyboard ownership", () => {
    const f = fixture();
    for (const [key, extra] of [
      ["a", {}], ["ArrowLeft", { ctrlKey: true }], ["+", { metaKey: true }],
      ["0", { altKey: true }], ["ArrowDown", { isComposing: true }], ["=", { keyCode: 229 }],
      ["Escape", { defaultPrevented: true }],
    ] as [string, Partial<KeyboardEvent>][]) {
      expect(f.key(key, extra).preventDefault).not.toHaveBeenCalled();
    }
    expect(f.callbacks.onKeyboardInteraction).not.toHaveBeenCalled();
    expect(f.renderer.panBy).not.toHaveBeenCalled();
    expect(f.renderer.zoomAt).not.toHaveBeenCalled();
    expect(f.key("ArrowLeft").preventDefault).toHaveBeenCalledOnce();
    expect(f.callbacks.onKeyboardInteraction).toHaveBeenCalledOnce();
    expect(f.renderer.panBy).toHaveBeenCalledExactlyOnceWith(48, 0);
    f.controller.dispose();
  });

  it("leaves focused controls and editable content to native keyboard interaction", () => {
    const f = fixture();
    const closest = vi.fn(() => ({}));
    const target = { closest } as unknown as EventTarget;
    for (const key of ["ArrowLeft", "ArrowDown", "+", "Escape"]) f.key(key, { target });
    expect(closest).toHaveBeenCalledWith(expect.stringContaining("contenteditable"));
    expect(f.callbacks.onKeyboardInteraction).not.toHaveBeenCalled();
    expect(f.callbacks.onEscape).not.toHaveBeenCalled();
    expect(f.renderer.panBy).not.toHaveBeenCalled();
    f.controller.dispose();
  });

  it("unregisters capture, blur and context menu handlers on disposal", () => {
    const f = fixture();
    f.controller.dispose();
    expect(f.canvas.removeEventListener).toHaveBeenCalledWith("lostpointercapture", expect.any(Function));
    expect(f.canvas.removeEventListener).toHaveBeenCalledWith("contextmenu", expect.any(Function));
    expect(window.removeEventListener).toHaveBeenCalledWith("blur", expect.any(Function));
  });
});
