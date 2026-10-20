import { describe, expect, it } from "vitest";
import { graphViewUrl, legacyGestureHelpRequested, readGraphViewState } from "../src/view-state";

describe("graph view handoff", () => {
  it("round-trips session, actual entity ID and search with reserved characters", () => {
    const state = { sessionId: "#00003", node: '政策 & 能源/#?"', search: "A+B → C / 政策" };
    const url = graphViewUrl("/knowledge-graph", state);
    expect(url).toContain("session_id=%2300003");
    expect(readGraphViewState(new URL(url, "https://example.test").search)).toEqual(state);
    expect(new URL(url, "https://example.test").hash).toBe("");
  });

  it("never carries legacy camera activation into a view link", () => {
    const state = readGraphViewState("?session_id=%2300001&node=entity&gesture=1&camera=on");
    expect(graphViewUrl("/promo-graph", state)).toBe("/promo-graph?session_id=%2300001&node=entity&search=");
    expect(legacyGestureHelpRequested("?gesture=1")).toBe(true);
    expect(legacyGestureHelpRequested("?gesture=false")).toBe(false);
    expect(legacyGestureHelpRequested("")).toBe(false);
  });

  it("explicitly carries cleared node and search instead of allowing stale saved values", () => {
    const state = { sessionId: "#00003", node: "", search: "" };
    const url = new URL(graphViewUrl("/knowledge-graph", state), "https://example.test");
    expect(url.searchParams.has("node")).toBe(true);
    expect(url.searchParams.has("search")).toBe(true);
    expect(url.searchParams.get("node")).toBe("");
    expect(url.searchParams.get("search")).toBe("");
    expect(readGraphViewState(url.search)).toEqual(state);
    const previous = { node: "previous-entity", search: "previous query" };
    const restored = {
      node: url.searchParams.has("node") ? url.searchParams.get("node") : previous.node,
      search: url.searchParams.has("search") ? url.searchParams.get("search") : previous.search,
    };
    expect(restored).toEqual({ node: "", search: "" });
  });
});
