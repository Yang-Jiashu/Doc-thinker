export interface GraphViewState {
  sessionId: string;
  node: string;
  search: string;
}

export function readGraphViewState(search: string): GraphViewState {
  const params = new URLSearchParams(search);
  return { sessionId: params.get("session_id") || "", node: params.get("node") || "", search: params.get("search") || "" };
}

/** Only these view parameters cross page boundaries; camera activation never does. */
export function graphViewUrl(path: string, state: GraphViewState): string {
  const params = new URLSearchParams();
  if (state.sessionId) params.set("session_id", state.sessionId);
  // Empty values explicitly clear selection/search in the destination. Omitting
  // them would permit that view to resurrect its locally saved previous state.
  params.set("node", state.node);
  params.set("search", state.search);
  return `${path}${params.size ? `?${params}` : ""}`;
}

export function legacyGestureHelpRequested(search: string): boolean {
  const value = new URLSearchParams(search).get("gesture");
  return value !== null && value !== "0" && value !== "false";
}
