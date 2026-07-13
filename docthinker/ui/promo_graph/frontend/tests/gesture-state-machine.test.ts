import { describe, expect, it } from "vitest";
import { GestureStateMachine, type GestureInputPort } from "../src/gesture/gesture-state-machine";
import type { GesturePoint, ScreenGestureMetrics } from "../src/gesture/gesture-metrics";

function metrics(overrides: Partial<ScreenGestureMetrics> = {}): ScreenGestureMetrics {
  return {
    point: { x: 100, y: 100 },
    thumbPoint: { x: 80, y: 100 },
    palmPoint: { x: 100, y: 130 },
    pinch: false,
    pinchDistance: 0.5,
    openPalm: false,
    extendedFingers: 1,
    handedness: "right",
    ...overrides,
  };
}

function fakePort(pickedNode = 7) {
  const events: string[] = [];
  const port: GestureInputPort = {
    pick: () => pickedNode,
    hover: index => events.push(`hover:${index}`),
    select: index => events.push(`select:${index}`),
    clearSelection: () => events.push("clear"),
    fitToGraph: () => events.push("fit"),
    orbitBy: deltaAzimuth => events.push(`orbit:${deltaAzimuth.toFixed(3)}`),
    zoomAt: (_point, factor) => events.push(`zoom:${factor.toFixed(3)}`),
    beginNodeDrag: index => events.push(`begin:${index}`),
    moveNode: (index, point) => events.push(`move:${index}:${point.x},${point.y}`),
    endNodeDrag: (index, moved) => events.push(`end:${index}:${moved}`),
    setCursor: (_point: GesturePoint | null, mode) => events.push(`cursor:${mode}`),
    setGestureActive: active => events.push(`active:${active}`),
    setNodeDragging: active => events.push(`dragging:${active}`),
  };
  return { port, events };
}

describe("GestureStateMachine", () => {
  it("selects a node after a steady one-finger hover", () => {
    const { port, events } = fakePort(7);
    const machine = new GestureStateMachine(port);
    machine.process([metrics()], 10);
    machine.process([metrics()], 570);
    machine.process([metrics()], 800);
    expect(events).toContain("hover:7");
    expect(events.filter(event => event === "select:7")).toHaveLength(1);
  });

  it("prioritizes graph orbit while one hand waves across nodes", () => {
    const { port, events } = fakePort(7);
    const machine = new GestureStateMachine(port);
    machine.process([metrics({ palmPoint: { x: 100, y: 130 } })], 10);
    machine.process([metrics({ palmPoint: { x: 116, y: 145 } })], 40);
    expect(events).toContain("orbit:0.096");
    expect(events.some(event => event.startsWith("select:"))).toBe(false);
  });

  it("accumulates slow horizontal movement instead of dropping it frame by frame", () => {
    const { port, events } = fakePort(-1);
    const machine = new GestureStateMachine(port);
    machine.process([metrics({ palmPoint: { x: 100, y: 130 } })], 10);
    machine.process([metrics({ palmPoint: { x: 100.7, y: 130 } })], 40);
    machine.process([metrics({ palmPoint: { x: 101.4, y: 130 } })], 70);
    machine.process([metrics({ palmPoint: { x: 102.1, y: 130 } })], 100);
    expect(events.some(event => event.startsWith("orbit:"))).toBe(true);
  });

  it("prioritizes an open-palm wave over the static zoom-in pose", () => {
    const { port, events } = fakePort(-1);
    const machine = new GestureStateMachine(port);
    const palm = metrics({ openPalm: true, extendedFingers: 4, palmPoint: { x: 100, y: 130 } });
    machine.process([palm], 10);
    machine.process([metrics({ ...palm, palmPoint: { x: 108, y: 130 } })], 40);
    expect(events).toContain("orbit:0.048");
    expect(events.some(event => event.startsWith("zoom:"))).toBe(false);
  });

  it.each(["left", "right"] as const)("zooms in while one %s palm is open", handedness => {
    const { port, events } = fakePort(-1);
    const machine = new GestureStateMachine(port);
    const palm = metrics({ handedness, openPalm: true, extendedFingers: 4 });
    machine.process([palm], 10);
    machine.process([palm], 110);
    const factors = events.filter(event => event.startsWith("zoom:")).map(event => Number(event.split(":")[1]));
    expect(factors.every(factor => factor > 1)).toBe(true);
    expect(factors.length).toBeGreaterThan(0);
  });

  it.each(["left", "right"] as const)("zooms out while one %s hand pinches", handedness => {
    const { port, events } = fakePort(7);
    const machine = new GestureStateMachine(port);
    const pinch = metrics({ handedness, pinch: true, pinchDistance: 0.2 });
    machine.process([pinch], 10);
    machine.process([pinch], 110);
    const factors = events.filter(event => event.startsWith("zoom:")).map(event => Number(event.split(":")[1]));
    expect(factors.every(factor => factor < 1)).toBe(true);
    expect(events.some(event => event.startsWith("begin:"))).toBe(false);
  });

  it.each(["left", "right"] as const)("resets after holding one %s fist", handedness => {
    const { port, events } = fakePort(-1);
    const machine = new GestureStateMachine(port);
    const fist = metrics({ handedness, pinch: false, openPalm: false, extendedFingers: 0 });
    machine.process([fist], 10);
    machine.process([fist], 520);
    machine.process([fist], 800);
    expect(events.filter(event => event === "clear")).toHaveLength(1);
    expect(events.filter(event => event === "fit")).toHaveLength(1);
  });

  it("releases hover and interaction when no hand is visible", () => {
    const { port, events } = fakePort(3);
    const machine = new GestureStateMachine(port);
    machine.process([metrics()], 0);
    machine.process([], 20);
    expect(events).toContain("active:false");
    expect(events).toContain("hover:-1");
  });
});
