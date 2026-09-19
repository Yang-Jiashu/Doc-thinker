import { LayoutCache } from "./layout-cache";
import type { GraphModel } from "./types";

interface FocusResult {
  indices: Uint32Array;
  positions: Float32Array;
}

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
};

export class LayoutController {
  private worker: Worker | null = null;
  private requestId = 0;
  private pending = new Map<number, PendingRequest>();
  private cache = new LayoutCache();

  private createWorker(): Worker {
    this.disposeWorker();
    const worker = new Worker(new URL("./layout.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = event => {
      const pending = this.pending.get(event.data.requestId);
      if (!pending) return;
      this.pending.delete(event.data.requestId);
      if (event.data.type === "focus") {
        pending.resolve({ indices: event.data.indices, positions: event.data.positions });
      } else {
        pending.resolve(event.data.positions);
      }
    };
    worker.onerror = event => {
      const error = new Error(event.message || "Graph layout worker failed");
      this.pending.forEach(request => request.reject(error));
      this.pending.clear();
    };
    this.worker = worker;
    return worker;
  }

  private graphMessage(model: GraphModel) {
    return {
      nodeCount: model.nodes.length,
      hashes: model.hashes.slice(),
      degrees: model.degrees.slice(),
      groups: model.groups.slice(),
      edgePairs: model.edgePairs.slice(),
    };
  }

  async layout(model: GraphModel): Promise<{ positions: Float32Array; cached: boolean }> {
    const worker = this.createWorker();
    const cacheKey = `v2:${model.fingerprint}`;
    const cached = await this.cache.get(cacheKey, model.nodes.length);
    if (cached) {
      const graph = this.graphMessage(model);
      const positions = cached.slice();
      worker.postMessage({ type: "hydrate", requestId: 0, ...graph, positions }, [
        graph.hashes.buffer,
        graph.degrees.buffer,
        graph.groups.buffer,
        graph.edgePairs.buffer,
        positions.buffer,
      ]);
      return { positions: cached, cached: true };
    }

    const requestId = ++this.requestId;
    const graph = this.graphMessage(model);
    const result = new Promise<Float32Array>((resolve, reject) => this.pending.set(requestId, {
      resolve: value => resolve(value as Float32Array),
      reject,
    }));
    worker.postMessage({ type: "layout", requestId, ...graph }, [
      graph.hashes.buffer,
      graph.degrees.buffer,
      graph.groups.buffer,
      graph.edgePairs.buffer,
    ]);
    const positions = await result;
    void this.cache.set(cacheKey, positions);
    return { positions, cached: false };
  }

  async focus(nodeIndex: number): Promise<FocusResult> {
    if (!this.worker) return { indices: new Uint32Array(), positions: new Float32Array() };
    const requestId = ++this.requestId;
    const result = new Promise<FocusResult>((resolve, reject) => this.pending.set(requestId, {
      resolve: value => resolve(value as FocusResult),
      reject,
    }));
    this.worker.postMessage({ type: "focus", requestId, nodeIndex });
    return await result;
  }

  disposeWorker(): void {
    this.worker?.terminate();
    this.worker = null;
    const error = new Error("Layout request cancelled");
    this.pending.forEach(request => request.reject(error));
    this.pending.clear();
  }

  dispose(): void {
    this.disposeWorker();
  }
}
