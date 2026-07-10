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
    panBy: (dx, dy) => events.push(`pan:${dx},${dy}`),
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
  it("selects, drags and releases a stable pinched node", () => {
    const { port, events } = fakePort(7);
    const machine = new GestureStateMachine(port);
    machine.process([metrics()], 0);
    machine.process([metrics()], 150);
    const pinch = metrics({ pinch: true, pinchDistance: 0.2 });
    machine.process([pinch], 170);
    machine.process([pinch], 190);
    machine.process([pinch], 210);
    machine.process([metrics({ ...pinch, point: { x: 112, y: 106 } })], 230);
    machine.process([metrics()], 250);
    machine.process([metrics()], 270);

    expect(events).toContain("select:7");
    expect(events).toContain("begin:7");
    expect(events).toContain("move:7:112,106");
    expect(events).toContain("end:7:true");
    expect(events).toContain("clear");
  });

  it("pans when pinching empty space", () => {
    const { port, events } = fakePort(-1);
    const machine = new GestureStateMachine(port);
    const pinch = metrics({ pinch: true });
    machine.process([pinch], 10);
    machine.process([pinch], 30);
    machine.process([pinch], 50);
    machine.process([metrics({ pinch: true, point: { x: 108, y: 95 } })], 70);
    expect(events).toContain("pan:8,-5");
  });

  it("zooms with one open palm and resets with two", () => {
    const { port, events } = fakePort(-1);
    const machine = new GestureStateMachine(port);
    const leftPalm = metrics({ handedness: "left", openPalm: true, extendedFingers: 4 });
    const rightPalm = metrics({ handedness: "right", openPalm: true, extendedFingers: 4 });
    machine.process([leftPalm], 10);
    machine.process([metrics({ ...leftPalm, palmPoint: { x: 100, y: 105 } })], 40);
    machine.process([metrics({ ...leftPalm, palmPoint: { x: 100, y: 150 } })], 70);
    machine.process([leftPalm, rightPalm], 100);
    machine.process([leftPalm, rightPalm], 530);
    const zoomFactors = events.filter(event => event.startsWith("zoom:")).map(event => Number(event.split(":")[1]));
    expect(zoomFactors.some(factor => factor > 1)).toBe(true);
    expect(zoomFactors.some(factor => factor < 1)).toBe(true);
    expect(events).toContain("clear");
    expect(events).toContain("fit");
  });

  it("does not zoom when only the right palm moves", () => {
    const { port, events } = fakePort(-1);
    const machine = new GestureStateMachine(port);
    const rightPalm = metrics({ handedness: "right", openPalm: true, extendedFingers: 4 });
    machine.process([rightPalm], 10);
    machine.process([metrics({ ...rightPalm, palmPoint: { x: 100, y: 80 } })], 40);
    expect(events.some(event => event.startsWith("zoom:"))).toBe(false);
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
