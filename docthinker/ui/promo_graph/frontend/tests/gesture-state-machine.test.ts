import { describe, expect, it, vi } from "vitest";
import { GestureStateMachine, type GestureInputPort } from "../src/gesture/gesture-state-machine";
import type { ScreenGestureMetrics } from "../src/gesture/gesture-metrics";

function metrics(overrides: Partial<ScreenGestureMetrics> = {}): ScreenGestureMetrics {
  const point = overrides.point ?? {x: 100, y: 100};
  return {point, thumbPoint: point, palmPoint: point, pinch: false, pinchDistance: 0.5,
    openPalm: false, extendedFingers: 1, pointing: true, handedness: "right", ...overrides};
}
const pinch = (x = 100, handedness: "left" | "right" = "right") => metrics({
  point: {x, y: 100}, pinch: true, pinchDistance: 0.2, pointing: false, handedness,
});
const palm = (x = 100) => metrics({point: {x, y: 100}, openPalm: true, extendedFingers: 4, pointing: false});
const fist = () => metrics({extendedFingers: 0, pointing: false});
function fakePort(pickedNode = 7) {
  return {
    pick: vi.fn(() => pickedNode), hover: vi.fn(), select: vi.fn(), clearSelection: vi.fn(), fitToGraph: vi.fn(),
    orbitBy: vi.fn(), panBy: vi.fn(), zoomAt: vi.fn(), beginNodeDrag: vi.fn(), moveNode: vi.fn(),
    endNodeDrag: vi.fn(), setCursor: vi.fn(), setGestureActive: vi.fn(), setNodeDragging: vi.fn(),
  } satisfies GestureInputPort;
}
function hold(machine: GestureStateMachine, hand: ScreenGestureMetrics, start: number, end: number, step = 100) {
  for (let time = start; time <= end; time += step) machine.process([hand], time);
}
const product = (port: ReturnType<typeof fakePort>) => port.zoomAt.mock.calls.reduce((value, [, factor]) => value * factor, 1);

describe("precise inspection gestures", () => {
  it("requires a real pointing pose and selects only once per stable dwell", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    hold(machine, metrics({pointing: false}), 0, 800);
    expect(port.select).not.toHaveBeenCalled();
    hold(machine, metrics(), 900, 1_800);
    expect(port.select).toHaveBeenCalledExactlyOnceWith(7);
  });

  it("does not orbit from horizontal pointer movement or zoom from a palm", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    for (let time = 0; time <= 800; time += 100) machine.process([metrics({point: {x: 100 + time, y: 100}})], time);
    hold(machine, palm(), 900, 2_000);
    expect(port.orbitBy).not.toHaveBeenCalled(); expect(port.zoomAt).not.toHaveBeenCalled();
    expect(port.select).not.toHaveBeenCalled();
  });

  it("restarts dwell when movement exceeds the stability radius even on the same node", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0, dwellMs: 500});
    hold(machine, metrics(), 0, 400);
    hold(machine, metrics({point: {x: 110, y: 100}}), 500, 900);
    expect(port.select).not.toHaveBeenCalled();
    machine.process([metrics({point: {x: 110, y: 100}})], 1_000);
    expect(port.select).toHaveBeenCalledOnce();
  });

  it("captures a node once and cannot switch to another node or zoom mid-drag", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    machine.process([pinch()], 0);
    port.pick.mockReturnValue(99);
    machine.process([pinch(120)], 100); machine.process([pinch(150)], 200);
    expect(port.beginNodeDrag).toHaveBeenCalledExactlyOnceWith(7);
    expect(port.pick).toHaveBeenCalledOnce();
    expect(port.moveNode).toHaveBeenLastCalledWith(7, {x: 150, y: 100});
    expect(port.zoomAt).not.toHaveBeenCalled(); expect(port.panBy).not.toHaveBeenCalled();
    machine.process([metrics()], 300);
    expect(port.endNodeDrag).toHaveBeenCalledExactlyOnceWith(7, true);
    expect(port.setNodeDragging).toHaveBeenLastCalledWith(false);
  });

  it("does not classify subthreshold pinch jitter as a moved drag", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    machine.process([pinch()], 0); machine.process([pinch(103)], 100);
    machine.process([metrics()], 200);
    expect(port.moveNode).not.toHaveBeenCalled();
    expect(port.endNodeDrag).toHaveBeenCalledExactlyOnceWith(7, false);
  });

  it("pans only when pinching empty space and applies gain to total displacement", () => {
    const port = fakePort(-1); const machine = new GestureStateMachine(port, {smoothingMs: 0, panGain: 2});
    machine.process([pinch()], 0); machine.process([pinch(103)], 100); machine.process([pinch(110)], 200);
    machine.process([pinch(130)], 300);
    expect(port.panBy.mock.calls).toEqual([[20, 0], [40, 0]]);
    expect(port.beginNodeDrag).not.toHaveBeenCalled(); expect(port.zoomAt).not.toHaveBeenCalled();
  });

  it("gives two-pinches exclusive zoom ownership and releases an existing drag", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    machine.process([pinch(200)], 0);
    machine.process([pinch(100, "left"), pinch(200)], 100);
    expect(port.endNodeDrag).toHaveBeenCalledExactlyOnceWith(7, false);
    expect(port.zoomAt).not.toHaveBeenCalled();
    machine.process([pinch(75, "left"), pinch(225)], 200);
    expect(port.zoomAt).toHaveBeenCalledExactlyOnceWith({x: 150, y: 100}, 1.5);
    expect(port.moveNode).not.toHaveBeenCalled(); expect(port.panBy).not.toHaveBeenCalled();
  });

  it("requires releasing the remaining pinch after two-hand zoom", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    machine.process([pinch(100, "left"), pinch(200)], 0);
    machine.process([pinch(75, "left"), pinch(225)], 100);
    machine.process([pinch(250)], 200); machine.process([pinch(270)], 300);
    expect(port.beginNodeDrag).not.toHaveBeenCalled();
    machine.process([metrics()], 400); machine.process([pinch(270)], 500);
    expect(port.beginNodeDrag).toHaveBeenCalledOnce();
  });

  it("rebases degenerate or implausibly jumping pair distances instead of explosive zoom", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    machine.process([pinch(100, "left"), pinch(100)], 0);
    machine.process([pinch(100, "left"), pinch(200)], 100);
    machine.process([pinch(100, "left"), pinch(1_000)], 200);
    expect(port.zoomAt).not.toHaveBeenCalled();
    machine.process([pinch(100, "left"), pinch(1_100)], 300);
    expect(product(port)).toBeCloseTo(1_000 / 900);
  });

  it("releases a lost drag once and does not resume it until the returned hand opens", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    machine.process([pinch()], 0); machine.process([pinch(130)], 100);
    machine.process([], 200); machine.process([], 250); machine.process([pinch(700)], 300);
    expect(port.endNodeDrag).toHaveBeenCalledExactlyOnceWith(7, true);
    expect(port.beginNodeDrag).toHaveBeenCalledOnce();
    expect(port.moveNode).toHaveBeenCalledTimes(1);
    machine.process([metrics({point: {x: 700, y: 100}})], 400); machine.process([pinch(700)], 500);
    expect(port.beginNodeDrag).toHaveBeenCalledTimes(2);
  });

  it("releases on primary-hand handoff and prevents using the old hand's anchor", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    machine.process([pinch(100, "left")], 0); machine.process([pinch(120, "left")], 100);
    machine.process([pinch(500)], 200);
    expect(port.endNodeDrag).toHaveBeenCalledExactlyOnceWith(7, true);
    expect(port.beginNodeDrag).toHaveBeenCalledOnce();
    expect(port.moveNode).toHaveBeenCalledExactlyOnceWith(7, {x: 120, y: 100});
  });

  it("honors a left-hand preference and keeps auto selection sticky", () => {
    for (const dominantHand of ["left", "auto"] as const) {
      const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0, dominantHand});
      const left = metrics({handedness: "left", point: {x: 20, y: 100}});
      machine.process([left], 0); machine.process([metrics(), left], 100);
      expect(port.pick).toHaveBeenLastCalledWith({x: 20, y: 100});
    }
  });

  it("does not reuse a dwell or drag across a tracking gap", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0, dwellMs: 500});
    hold(machine, metrics(), 0, 400); machine.process([metrics()], 1_000);
    expect(port.select).not.toHaveBeenCalled();
    machine.process([pinch()], 1_100); machine.process([pinch(500)], 2_000);
    expect(port.endNodeDrag).toHaveBeenCalledExactlyOnceWith(7, false);
    expect(port.moveNode).not.toHaveBeenCalled();
  });

  it("treats a nonfinite frame or backward clock as a fresh interaction", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    machine.process([pinch()], 100);
    machine.process([pinch(Number.NaN)], 200);
    expect(port.endNodeDrag).toHaveBeenCalledOnce();
    machine.process([metrics()], 300); machine.process([pinch()], 400);
    machine.process([pinch(500)], 250);
    expect(port.endNodeDrag).toHaveBeenCalledTimes(2);
    expect(port.moveNode).not.toHaveBeenCalled();
  });

  it("changing preferences safely releases a captured node and returns a copy", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port);
    machine.process([pinch()], 0);
    machine.setPreferences({mode: "presentation", zoomGain: 2});
    expect(port.endNodeDrag).toHaveBeenCalledOnce();
    const snapshot = machine.getPreferences(); snapshot.mode = "inspect";
    expect(machine.getPreferences().mode).toBe("presentation");
    expect(machine.getPreferences().zoomGain).toBe(2);
  });

  it("releases a discontinuous same-label palm track without teleporting the node", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    machine.process([pinch(100)], 0); machine.process([pinch(130)], 100);
    machine.process([pinch(700)], 140); machine.process([pinch(710)], 180);
    expect(port.endNodeDrag).toHaveBeenCalledExactlyOnceWith(7, true);
    expect(port.beginNodeDrag).toHaveBeenCalledOnce();
    expect(port.moveNode).toHaveBeenCalledExactlyOnceWith(7, {x: 130, y: 100});
  });

  it("releases when two tracks have the same ambiguous hand identity", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {smoothingMs: 0});
    machine.process([pinch(100)], 0);
    machine.process([pinch(110), pinch(300)], 100);
    expect(port.endNodeDrag).toHaveBeenCalledExactlyOnceWith(7, false);
    expect(port.moveNode).not.toHaveBeenCalled(); expect(port.zoomAt).not.toHaveBeenCalled();
    expect(port.setGestureActive).toHaveBeenLastCalledWith(false);
  });

  it("holds fist from timestamp zero, resets once, and rearms only after leaving the pose", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {resetHoldMs: 500});
    hold(machine, fist(), 0, 1_000);
    expect(port.fitToGraph).toHaveBeenCalledOnce(); expect(port.clearSelection).toHaveBeenCalledOnce();
    machine.process([metrics()], 1_100); hold(machine, fist(), 1_200, 1_800);
    expect(port.fitToGraph).toHaveBeenCalledTimes(2);
  });

  it("never treats a zero-extended-finger pinch as the fist-reset pose", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {resetHoldMs: 500});
    hold(machine, {...pinch(), extendedFingers: 0}, 0, 1_000);
    expect(port.fitToGraph).not.toHaveBeenCalled();
  });
});

describe("presentation gesture mode", () => {
  it("arms horizontal orbit only with an open palm and locks it against zoom", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {mode: "presentation", smoothingMs: 0});
    machine.process([metrics()], 0); machine.process([metrics({point: {x: 200, y: 100}})], 100);
    expect(port.orbitBy).not.toHaveBeenCalled();
    machine.process([palm()], 200); machine.process([palm(110)], 300);
    hold(machine, palm(110), 400, 1_200);
    expect(port.orbitBy).toHaveBeenCalledExactlyOnceWith(0.06);
    expect(port.zoomAt).not.toHaveBeenCalled();
  });

  it("waits for a stable palm, then locks zoom until the pose is released", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {mode: "presentation", smoothingMs: 0, activationMs: 200});
    machine.process([palm()], 0); machine.process([palm()], 100);
    expect(port.zoomAt).not.toHaveBeenCalled();
    machine.process([palm()], 200); machine.process([palm()], 300); machine.process([palm(150)], 400);
    expect(port.orbitBy).not.toHaveBeenCalled(); expect(product(port)).toBeCloseTo(Math.exp(200 / 1_600));
  });

  it("uses pinch for zoom-out without capture in presentation mode", () => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {mode: "presentation", smoothingMs: 0, activationMs: 200});
    hold(machine, pinch(), 0, 1_000);
    expect(product(port)).toBeCloseTo(Math.exp(-800 / 1_600));
    expect(port.beginNodeDrag).not.toHaveBeenCalled();
  });

  it.each([20, 40, 100])("integrates equal held time identically at %s ms frame spacing", step => {
    const port = fakePort(); const machine = new GestureStateMachine(port, {mode: "presentation", activationMs: 200});
    hold(machine, palm(), 0, 1_000, step);
    expect(product(port)).toBeCloseTo(Math.exp(800 / 1_600), 10);
  });

  it("uses elapsed-time smoothing rather than a fixed per-frame alpha", () => {
    const finalPoint = (step: number) => {
      const port = fakePort(-1); const machine = new GestureStateMachine(port, {smoothingMs: 100});
      machine.process([metrics({point: {x: 0, y: 100}})], 0);
      hold(machine, metrics({point: {x: 100, y: 100}}), step, 200, step);
      return port.setCursor.mock.calls.at(-1)![0].x;
    };
    expect(finalPoint(20)).toBeCloseTo(finalPoint(100), 10);
    expect(finalPoint(20)).toBeCloseTo(100 * (1 - Math.exp(-2)), 10);
  });
});
