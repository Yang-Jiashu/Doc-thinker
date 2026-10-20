import { afterEach, describe, expect, it, vi } from "vitest";
import { LayoutController } from "../src/layout-controller";
import { LayoutCache } from "../src/layout-cache";
import { normalizeGraph } from "../src/graph-data";

describe("layout request cancellation", () => {
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

  it.each([null, new Float32Array([1, 2, 0])])("does not post to a terminated worker after a late cache result: %s", async cached => {
    const workers: { terminate: ReturnType<typeof vi.fn>; postMessage: ReturnType<typeof vi.fn> }[] = [];
    class WorkerMock {
      onmessage: ((event: { data: unknown }) => void) | null = null;
      terminate = vi.fn();
      postMessage = vi.fn((message: { type: string; requestId: number }) => {
        if (message.type === "layout") this.onmessage?.({ data: { requestId: message.requestId, positions: new Float32Array([3, 4, 0]) } });
      });
      constructor() { workers.push(this); }
    }
    vi.stubGlobal("Worker", WorkerMock);
    let finishCache!: (result: Float32Array | null) => void;
    vi.spyOn(LayoutCache.prototype, "get").mockReturnValueOnce(new Promise(resolve => { finishCache = resolve; })).mockResolvedValue(null);
    vi.spyOn(LayoutCache.prototype, "set").mockResolvedValue();
    const controller = new LayoutController();
    const model = normalizeGraph({ nodes: [{ id: "entity" }] });
    const old = controller.layout(model);
    const rejection = expect(old).rejects.toThrow("Layout request cancelled");
    expect((await controller.layout(model)).positions[0]).toBe(3);
    finishCache(cached);
    await rejection;
    expect(workers[0].terminate).toHaveBeenCalledTimes(1);
    expect(workers[0].postMessage).not.toHaveBeenCalled();
    controller.dispose();
  });
});
