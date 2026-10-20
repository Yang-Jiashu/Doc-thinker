import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HandLandmarker } from "@mediapipe/tasks-vision";
import { GestureController, pauseGestureForManualInput, pauseGestureForPointer, type GestureRenderer } from "../src/gesture/gesture-controller";
import { createHandLandmarker } from "../src/gesture/mediapipe-adapter";

vi.mock("../src/gesture/mediapipe-adapter", () => ({ createHandLandmarker: vi.fn() }));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function fixture() {
  const track = { enabled: true, stop: vi.fn(), addEventListener: vi.fn() };
  const stream = { getTracks: () => [track], getVideoTracks: () => [track] } as unknown as MediaStream;
  const getUserMedia = vi.fn().mockResolvedValue(stream);
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });
  const detector = { close: vi.fn(), detectForVideo: vi.fn(() => ({ landmarks: [], handedness: [] })) };
  vi.mocked(createHandLandmarker).mockResolvedValue(detector as unknown as HandLandmarker);
  const stage = { clientWidth: 1000, clientHeight: 600 } as HTMLElement;
  const layer = { dataset: { state: "idle" }, hidden: true } as unknown as HTMLElement;
  const video = { srcObject: null, readyState: 2, currentTime: 1, play: vi.fn().mockResolvedValue(undefined), pause: vi.fn() } as unknown as HTMLVideoElement;
  const cursor = { style: {}, dataset: {} } as unknown as HTMLElement;
  const renderer = {
    pick: vi.fn(() => -1), setHovered: vi.fn(), fitToGraph: vi.fn(), orbitBy: vi.fn(), panBy: vi.fn(), zoomAt: vi.fn(),
    getNodeWorldPosition: vi.fn(() => ({ x: 0, y: 0, z: 0 })),
    screenToWorld: vi.fn(() => ({ x: 1, y: 2, z: 0 })), setNodeTransientPosition: vi.fn(),
  } satisfies GestureRenderer;
  const callbacks = {
    onSelect: vi.fn(), onClearSelection: vi.fn(), onNodeAttractionStart: vi.fn(), onNodeAttractionEnd: vi.fn(),
    onHover: vi.fn(), onNodeMoved: vi.fn(), onGestureActiveChange: vi.fn(), onNodeDraggingChange: vi.fn(), onStateChange: vi.fn(),
  };
  const controller = new GestureController(stage, layer, video, cursor, renderer, callbacks);
  return { controller, track, stream, getUserMedia, detector, video, layer, callbacks, renderer };
}

describe("gesture controller camera lifecycle (no real devices)", () => {
  beforeEach(() => vi.clearAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it("settings and construction never request a camera, defaulting to precise mode", () => {
    const f = fixture();
    f.controller.setPreferences({ dominantHand: "left", dwellMs: 900 });
    expect(f.controller.getPreferences()).toMatchObject({ mode: "inspect", dominantHand: "left", dwellMs: 900 });
    expect(f.controller.getState()).toMatchObject({ enabled: false, starting: false });
    expect(f.getUserMedia).not.toHaveBeenCalled();
    expect(createHandLandmarker).not.toHaveBeenCalled();
  });

  it("deduplicates explicit starts, supports pause/resume, and closes all resources", async () => {
    const f = fixture();
    const [first, second] = await Promise.all([f.controller.start(), f.controller.start()]);
    expect([first, second]).toEqual([true, true]);
    expect(f.getUserMedia).toHaveBeenCalledTimes(1);
    f.controller.tick(100);
    expect(f.detector.detectForVideo).toHaveBeenCalledTimes(1);
    f.controller.setSuspended(true);
    expect(f.track.enabled).toBe(false);
    expect(f.controller.getState()).toMatchObject({ enabled: true, suspended: true, layerState: "paused" });
    f.controller.tick(200);
    expect(f.detector.detectForVideo).toHaveBeenCalledTimes(1);
    f.controller.setSuspended(false);
    expect(f.track.enabled).toBe(true);
    f.controller.tick(250);
    expect(f.detector.detectForVideo).toHaveBeenCalledTimes(2);
    expect(f.getUserMedia).toHaveBeenCalledTimes(1);
    f.controller.stop();
    expect(f.track.stop).toHaveBeenCalled();
    expect(f.detector.close).toHaveBeenCalledTimes(1);
    expect(f.video.srcObject).toBeNull();
    expect(f.controller.getState()).toMatchObject({ enabled: false, suspended: false, starting: false });
    expect(f.layer.hidden).toBe(true);
  });

  it("stops a permission result arriving after close without loading a detector", async () => {
    const f = fixture();
    const permission = deferred<MediaStream>();
    f.getUserMedia.mockReturnValue(permission.promise);
    const started = f.controller.start();
    expect(f.controller.getState().starting).toBe(true);
    f.controller.stop();
    permission.resolve(f.stream);
    expect(await started).toBe(false);
    expect(f.track.stop).toHaveBeenCalled();
    expect(createHandLandmarker).not.toHaveBeenCalled();
    expect(f.controller.getState().enabled).toBe(false);
  });

  it("closes a late detector after disposing during model loading", async () => {
    const f = fixture();
    const model = deferred<HandLandmarker>();
    vi.mocked(createHandLandmarker).mockReturnValue(model.promise);
    const started = f.controller.start();
    await Promise.resolve();
    f.controller.dispose();
    model.resolve(f.detector as unknown as HandLandmarker);
    expect(await started).toBe(false);
    expect(f.track.stop).toHaveBeenCalled();
    expect(f.detector.close).toHaveBeenCalledTimes(1);
    expect(f.video.play).not.toHaveBeenCalled();
    expect(await f.controller.start()).toBe(false);
  });

  it("does not revive capture if closed while video.play is pending", async () => {
    const f = fixture();
    const playback = deferred<void>();
    vi.mocked(f.video.play).mockReturnValue(playback.promise);
    const started = f.controller.start();
    await Promise.resolve();
    await Promise.resolve();
    expect(f.video.play).toHaveBeenCalled();
    f.controller.stop();
    playback.resolve();
    expect(await started).toBe(false);
    expect(f.detector.close).toHaveBeenCalledTimes(1);
    expect(f.video.srcObject).toBeNull();
  });

  it("keeps a newer successful start intact when an older permission request returns", async () => {
    const f = fixture();
    const permission = deferred<MediaStream>();
    f.getUserMedia.mockReturnValueOnce(permission.promise);
    const oldStart = f.controller.start();
    f.controller.stop();
    expect(await f.controller.start()).toBe(true);
    const oldTrack = { stop: vi.fn() };
    permission.resolve({ getTracks: () => [oldTrack] } as unknown as MediaStream);
    expect(await oldStart).toBe(false);
    expect(oldTrack.stop).toHaveBeenCalled();
    expect(f.controller.getState().enabled).toBe(true);
    expect(f.track.stop).not.toHaveBeenCalled();
    expect(f.detector.close).not.toHaveBeenCalled();
    f.controller.stop();
  });

  it("cancels pending activation on suspension and requires an explicit restart", async () => {
    const f = fixture();
    const permission = deferred<MediaStream>();
    f.getUserMedia.mockReturnValue(permission.promise);
    const started = f.controller.start();
    f.controller.setSuspended(true);
    permission.resolve(f.stream);
    expect(await started).toBe(false);
    f.controller.setSuspended(false);
    expect(f.getUserMedia).toHaveBeenCalledTimes(1);
    expect(f.controller.getState().enabled).toBe(false);
  });

  it("explains permission failures without leaking exception content", async () => {
    const f = fixture();
    f.getUserMedia.mockRejectedValue(Object.assign(new Error("<script>secret-device</script>"), { name: "NotAllowedError" }));
    expect(await f.controller.start()).toBe(false);
    expect(f.controller.getState().error).toContain("摄像头权限");
    expect(f.controller.getState().error).not.toContain("secret-device");
    expect(f.controller.getState().layerState).toBe("unavailable");
    expect(f.renderer.panBy).not.toHaveBeenCalled();
  });

  it("pauses on inference failure instead of retrying every frame", async () => {
    const f = fixture();
    await f.controller.start();
    f.detector.detectForVideo.mockImplementation(() => { throw new Error("frame failed"); });
    f.controller.tick(100);
    f.controller.tick(200);
    expect(f.detector.detectForVideo).toHaveBeenCalledTimes(1);
    expect(f.controller.getState()).toMatchObject({ enabled: true, suspended: true });
    expect(f.controller.getState().error).toContain("当前帧处理失败");
    f.controller.stop();
  });

  it("gives mouse/touch priority and never resumes on pointer release", async () => {
    const f = fixture();
    await f.controller.start();
    expect(pauseGestureForPointer(f.controller, true)).toBe(true);
    expect(f.track.enabled).toBe(false);
    expect(pauseGestureForPointer(f.controller, false)).toBe(false);
    f.controller.tick(100);
    expect(f.controller.getState().suspended).toBe(true);
    expect(f.detector.detectForVideo).not.toHaveBeenCalled();
    f.controller.setSuspended(false); // The explicit resume button.
    expect(f.track.enabled).toBe(true);
    f.controller.stop();
  });

  it("wheel, keyboard and toolbar takeover pauses once without auto-resuming", async () => {
    const f = fixture();
    expect(pauseGestureForManualInput(f.controller)).toBe(false);
    await f.controller.start();
    expect(pauseGestureForManualInput(f.controller)).toBe(true);
    expect(pauseGestureForManualInput(f.controller)).toBe(false);
    f.controller.tick(100);
    expect(f.track.enabled).toBe(false);
    expect(f.detector.detectForVideo).not.toHaveBeenCalled();
    f.controller.stop();
  });

  it.each(["frozen", "not-decodable"])("releases interaction once when video is %s without a landmark result", async mode => {
    const f = fixture();
    await f.controller.start(); f.controller.tick(100);
    f.callbacks.onGestureActiveChange.mockClear();
    if (mode === "not-decodable") Object.assign(f.video, { readyState: 1 });
    f.controller.tick(350);
    expect(f.callbacks.onGestureActiveChange).not.toHaveBeenCalled();
    f.controller.tick(401); f.controller.tick(500);
    expect(f.callbacks.onGestureActiveChange).toHaveBeenCalledExactlyOnceWith(false);
    expect(f.detector.detectForVideo).toHaveBeenCalledTimes(1);
    expect(f.callbacks.onClearSelection).not.toHaveBeenCalled();
    expect(f.renderer.fitToGraph).not.toHaveBeenCalled();
    expect(f.renderer.setNodeTransientPosition).not.toHaveBeenCalled();
    expect(f.controller.getState()).toMatchObject({ enabled: true, suspended: false });
    Object.assign(f.video, { readyState: 2, currentTime: 2 });
    f.controller.tick(600);
    expect(f.detector.detectForVideo).toHaveBeenCalledTimes(2);
    f.controller.stop();
  });
});
