import { readFileSync } from "node:fs";
import vm from "node:vm";
import { describe, expect, it, vi } from "vitest";

// Exercise the real classic adapter without importing a model or requesting devices.
const source = readFileSync(new URL("../../static/gesture_chooser.js", import.meta.url), "utf8");
const body = source.split("(function () {", 2)[1].split("})();")[0];
function fixture() {
  const events = new Map<string, (event: object) => void>();
  const windowEvents = new Map<string, () => void>();
  const inside = {}, outside = {};
  const elements = new Map<string, object>([
    ["graph-container", { contains: (target: object) => target === inside }],
    ["shared-gesture-state", {textContent: ""}], ["gesture-mode", {textContent: ""}],
    ["gesture-layer", {classList: {toggle: vi.fn()}}],
  ]);
  const state = {enabled: true, suspended: false, starting: false, error: ""};
  const setSuspended = vi.fn((value: boolean) => { state.suspended = value; });
  const document = {
    currentScript: {dataset: {}}, hidden: false, createElement: vi.fn(() => ({})), head: {append: vi.fn()},
    getElementById: (id: string) => elements.get(id),
    addEventListener: (name: string, callback: (event: object) => void) => events.set(name, callback),
  };
  vm.runInNewContext(`${body}\n controller = mockController;`, {
    document, mockController: {getState: () => state, setSuspended},
    window: {addEventListener: (name: string, callback: () => void) => windowEvents.set(name, callback)},
  });
  return {state, setSuspended, document, inside, outside, events, windowEvents};
}

describe("classic graph manual-input ownership", () => {
  it("only graph wheel takes over and no release/visibility event resumes", () => {
    const f = fixture();
    f.events.get("wheel")!({target: f.outside});
    expect(f.setSuspended).not.toHaveBeenCalled();
    f.events.get("wheel")!({target: f.inside});
    f.events.get("wheel")!({target: f.inside});
    f.events.get("visibilitychange")!({});
    expect(f.setSuspended).toHaveBeenCalledExactlyOnceWith(true);
    expect(f.state.suspended).toBe(true);
  });

  it("toolbar activation takes over, including clicks produced by keyboard", () => {
    const f = fixture();
    const closest = vi.fn(() => ({}));
    f.events.get("click")!({target: {closest}});
    expect(closest).toHaveBeenCalledWith(".kg-header-actions, .kg-toolbar, #detail-focus-btn");
    expect(f.setSuspended).toHaveBeenCalledExactlyOnceWith(true);
    f.events.get("click")!({target: {closest: () => null}});
    expect(f.setSuspended).toHaveBeenCalledTimes(1);
  });

  it("only unmodified Escape outside controls takes keyboard ownership", () => {
    const f = fixture();
    for (const extra of [{ctrlKey:true}, {metaKey:true}, {altKey:true}, {isComposing:true}, {keyCode:229},
      {defaultPrevented:true}, {target:{closest:()=>({})}}]) {
      f.events.get("keydown")!({key:"Escape", ...extra});
    }
    f.events.get("keydown")!({key:"a"});
    expect(f.setSuspended).not.toHaveBeenCalled();
    f.events.get("keydown")!({key:"Escape"});
    expect(f.setSuspended).toHaveBeenCalledExactlyOnceWith(true);
  });

  it("window blur pauses active capture without cancelling permission prompts", () => {
    const f = fixture();
    f.state.enabled = false; f.state.starting = true;
    f.windowEvents.get("blur")!();
    expect(f.setSuspended).not.toHaveBeenCalled();
    f.state.enabled = true; f.state.starting = false;
    f.windowEvents.get("blur")!(); f.windowEvents.get("blur")!();
    expect(f.setSuspended).toHaveBeenCalledExactlyOnceWith(true);
  });
});
