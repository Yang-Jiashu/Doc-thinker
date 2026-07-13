import { describe, expect, it } from "vitest";
import { AutoRotationController } from "../src/auto-rotation-controller";

function readyController() {
  const controller = new AutoRotationController({ resumeDelayMs: 100, wheelIdleMs: 40 }, true, 0);
  controller.setLayoutStable(true, 0);
  controller.setPointerInside(true, 0);
  return controller;
}

describe("AutoRotationController", () => {
  it("rotates only after the graph is idle and ready", () => {
    const controller = readyController();
    expect(controller.evaluate(99)).toEqual({ shouldRotate: false, pausedReason: "resume-delay" });
    expect(controller.evaluate(100)).toEqual({ shouldRotate: true, pausedReason: null });

    controller.setHasSelection(true, 110);
    expect(controller.evaluate(1_000).pausedReason).toBe("selection");
    controller.setHasSelection(false, 1_000);
    expect(controller.evaluate(1_099).pausedReason).toBe("resume-delay");
    expect(controller.evaluate(1_100).shouldRotate).toBe(true);
  });

  it("honors interaction, visibility and layout pause conditions", () => {
    const controller = new AutoRotationController(
      { resumeDelayMs: 100, wheelIdleMs: 40, pauseWhenPointerOutside: true },
      true,
      0,
    );
    controller.setLayoutStable(true, 0);
    controller.setPointerInside(true, 0);
    controller.setPointerDown(true, 120);
    expect(controller.evaluate(500).pausedReason).toBe("pointer-down");
    controller.setPointerDown(false, 500);
    controller.setOrbitInteracting(true, 510);
    expect(controller.evaluate(800).pausedReason).toBe("orbit-interaction");
    controller.setOrbitInteracting(false, 800);
    controller.setNodeDragging(true, 810);
    expect(controller.evaluate(1_000).pausedReason).toBe("node-drag");
    controller.setNodeDragging(false, 1_000);

    controller.setPointerInside(false, 1_010);
    expect(controller.evaluate(2_000).pausedReason).toBe("pointer-outside");
    controller.setPointerInside(true, 2_000);
    controller.setDocumentVisible(false, 2_010);
    expect(controller.evaluate(3_000).pausedReason).toBe("document-hidden");
    controller.setDocumentVisible(true, 3_000);
    controller.setLayoutStable(false, 3_010);
    expect(controller.evaluate(4_000).pausedReason).toBe("layout-running");
  });

  it("can rotate without requiring a prior pointer entry", () => {
    const controller = new AutoRotationController({ resumeDelayMs: 100 }, true, 0);
    controller.setLayoutStable(true, 0);
    expect(controller.snapshot.pointerInside).toBe(false);
    expect(controller.evaluate(100)).toEqual({ shouldRotate: true, pausedReason: null });
  });

  it("debounces wheel input and never re-enables a disabled preference", () => {
    const controller = readyController();
    controller.noteWheel(120);
    expect(controller.evaluate(150).pausedReason).toBe("wheel");
    expect(controller.evaluate(219).pausedReason).toBe("resume-delay");
    expect(controller.evaluate(220).shouldRotate).toBe(true);

    controller.setEnabled(false, 230);
    expect(controller.evaluate(10_000).pausedReason).toBe("disabled");
    expect(controller.snapshot.enabled).toBe(false);
  });
});
